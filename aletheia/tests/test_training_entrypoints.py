import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.aletheia_train import (
    _ModelWrapper,
    OptimizerBundle,
    TrainingConfig,
    TrainingLoop,
    TrainingStep,
    compute_policy_outputs,
    compute_dreamer_actor_loss,
    compute_gae,
)


def _single_rl_opt_bundle(optimizer: torch.optim.Optimizer) -> OptimizerBundle:
    return OptimizerBundle(rl_optimizer=optimizer)


class _LossPacket:
    def __init__(self, total: torch.Tensor):
        self.total = total
        self.metrics = {"kl_mean": torch.tensor(0.0, device=total.device)}


class _DummyWorldModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def observe_sequence(self, vitals, actions, dones):
        return {}

    def build_trajectory(
        self,
        seq_result,
        vitals,
        actions,
        rewards,
        dones,
        remaining_steps=None,
    ):
        del remaining_steps
        return {}

    def compute_loss(self, trajectory):
        return _LossPacket(torch.tensor(0.0, device=self.device))


class _DummyModel(nn.Module):
    def __init__(self, vital_dim: int, action_dim: int, device: torch.device):
        super().__init__()
        self.fc = nn.Linear(vital_dim, action_dim)
        self.world_model = _DummyWorldModel(device)
        self.actor = SimpleNamespace(is_discrete=False)
        self.critic = _FlatCritic(vital_dim)

    def forward(self, vitals, temperature: float = 1.0, intent=None):
        mu = self.fc(vitals)
        dist = D.Normal(mu, torch.ones_like(mu))
        value = self.critic(vitals)
        return dist, value


class _FlatCritic(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.v = nn.Linear(feat_dim, 1)
        self.gammas = (0.99,)
        self.primary_gamma = 0.99

    def forward(self, x, use_target: bool = False):
        del use_target
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        return self.v(x).squeeze(-1)

    def compute_loss(self, feat, targets, intent=None, weights=None, detach_features: bool = False):
        del intent
        if detach_features:
            feat = feat.detach()
        if isinstance(targets, dict):
            target = targets.get(self.primary_gamma, next(iter(targets.values())))
        else:
            target = targets
        pred = self.forward(feat)
        if target.shape != pred.shape and target.numel() == pred.numel():
            target = target.reshape_as(pred)
        per_elem = F.huber_loss(pred, target, reduction="none", delta=1.0)
        if weights is not None:
            w = weights.reshape(-1)
            per_elem = per_elem * w[: per_elem.shape[0]]
        loss = per_elem.mean()
        return {
            "loss": loss,
            "per_gamma": {self.primary_gamma: loss.detach()},
        }


class _DummyModelPrecomputed(nn.Module):
    def __init__(self, vital_dim: int, action_dim: int, device: torch.device):
        super().__init__()
        self.fc = nn.Linear(vital_dim, action_dim)
        self.world_model = _DummyWorldModel(device)
        self.actor = SimpleNamespace(is_discrete=False)
        self.critic = _FlatCritic(vital_dim)

    def forward(self, vitals, temperature: float = 1.0, intent=None):
        mu = self.fc(vitals)
        dist = D.Normal(mu, torch.ones_like(mu))
        value = self.critic(vitals)
        return dist, value


class _TargetGapCritic(_FlatCritic):
    def __init__(self, feat_dim: int, target_shift: float = 5.0):
        super().__init__(feat_dim)
        self.target_shift = float(target_shift)

    def forward(self, x, use_target: bool = False):
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        out = self.v(x).squeeze(-1)
        if use_target:
            out = out + self.target_shift
        return out


class _RecordingCritic(nn.Module):
    def __init__(self, feat_dim: int, fixed_loss: float = 3.25):
        super().__init__()
        self.v = nn.Linear(feat_dim, 1)
        self.gammas = (0.99,)
        self.primary_gamma = 0.99
        self.fixed_loss = float(fixed_loss)
        self.compute_loss_calls = 0
        self.last_detach_features = None
        self.last_targets = None

    def forward(self, x, use_target: bool = False):
        del use_target
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        return self.v(x).squeeze(-1)

    def compute_loss(self, feat, targets, intent=None, weights=None, detach_features: bool = False):
        del intent, weights
        self.compute_loss_calls += 1
        self.last_detach_features = bool(detach_features)
        self.last_targets = targets
        anchor = self.v(feat.reshape(-1, feat.shape[-1])).mean() * 0.0
        loss = anchor + self.fixed_loss
        return {
            "loss": loss,
            "per_gamma": {self.primary_gamma: loss.detach()},
        }


class _RouterAwareRecordingCritic(_RecordingCritic):
    def __init__(self, feat_dim: int, fixed_loss: float = 3.25, router_loss: float = 0.75):
        super().__init__(feat_dim=feat_dim, fixed_loss=fixed_loss)
        self.router_loss = float(router_loss)
        self.last_include_router_loss = None

    def compute_loss(
        self,
        feat,
        targets,
        intent=None,
        weights=None,
        include_router_loss: bool = True,
        detach_features: bool = False,
    ):
        del intent, weights
        self.compute_loss_calls += 1
        self.last_detach_features = bool(detach_features)
        self.last_targets = targets
        self.last_include_router_loss = bool(include_router_loss)
        anchor = self.v(feat.reshape(-1, feat.shape[-1])).mean() * 0.0
        core_loss = anchor + self.fixed_loss
        result = {
            "loss": core_loss,
            "loss_critic": core_loss.detach(),
            "per_gamma": {self.primary_gamma: core_loss.detach()},
        }
        if include_router_loss:
            router_loss = anchor + self.router_loss
            result.update(
                {
                    "loss": core_loss + router_loss,
                    "loss_router": router_loss.detach(),
                    "L_route": torch.tensor(0.2, device=feat.device),
                    "L_lb": torch.tensor(0.1, device=feat.device),
                    "L_ent": torch.tensor(0.05, device=feat.device),
                    "H_gamma": torch.tensor(0.4, device=feat.device),
                    "w_gamma_mean": torch.tensor([0.3, 0.7], device=feat.device),
                }
            )
        return result


class _DummyModelPrecomputedDrift(nn.Module):
    def __init__(self, vital_dim: int, action_dim: int, device: torch.device, target_shift: float = 5.0):
        super().__init__()
        self.fc = nn.Linear(vital_dim, action_dim)
        self.world_model = _DummyWorldModel(device)
        self.actor = SimpleNamespace(is_discrete=False)
        self.critic = _TargetGapCritic(vital_dim, target_shift=target_shift)

    def forward(self, vitals, temperature: float = 1.0, intent=None):
        mu = self.fc(vitals)
        dist = D.Normal(mu, torch.ones_like(mu))
        value = self.critic(vitals)
        return dist, value


class _DummyModelRecordingCritic(nn.Module):
    def __init__(self, vital_dim: int, action_dim: int, device: torch.device):
        super().__init__()
        self.fc = nn.Linear(vital_dim, action_dim)
        self.world_model = _DummyWorldModel(device)
        self.actor = SimpleNamespace(is_discrete=False)
        self.critic = _RecordingCritic(vital_dim)

    def forward(self, vitals, temperature: float = 1.0, intent=None):
        del temperature, intent
        mu = self.fc(vitals)
        dist = D.Normal(mu, torch.ones_like(mu))
        value = self.critic(vitals)
        return dist, value


class _DummyModelRouterCritic(nn.Module):
    def __init__(self, vital_dim: int, action_dim: int, device: torch.device):
        super().__init__()
        self.fc = nn.Linear(vital_dim, action_dim)
        self.world_model = _DummyWorldModel(device)
        self.actor = SimpleNamespace(is_discrete=False)
        self.critic = _RouterAwareRecordingCritic(vital_dim)

    def forward(self, vitals, temperature: float = 1.0, intent=None):
        del temperature, intent
        mu = self.fc(vitals)
        dist = D.Normal(mu, torch.ones_like(mu))
        value = self.critic(vitals)
        return dist, value


class _FallbackGradTrackingCritic(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.v = nn.Linear(feat_dim, 1, bias=False)
        nn.init.ones_(self.v.weight)
        self.last_grad_enabled = None
        self.last_input_requires_grad = None

    def forward(self, x):
        self.last_grad_enabled = bool(torch.is_grad_enabled())
        self.last_input_requires_grad = bool(x.requires_grad)
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        return self.v(x).squeeze(-1)


class _SemanticWMState:
    def __init__(self, x_t, h_shared, z):
        self.x_t = x_t
        self.h_shared = h_shared
        self.z = z

    def isolate_gradients(self, context: str, wall_strength: float = 1.0):
        if context == "all":
            return self.detach_all()
        if context == "rl":
            return _SemanticWMState(
                self.x_t,
                self.h_shared.detach(),
                self.z.detach(),
            )
        if context == "policy":
            if wall_strength >= 1.0:
                x_t = self.x_t.detach()
            elif wall_strength <= 0.0:
                x_t = self.x_t
            else:
                x_t = wall_strength * self.x_t.detach() + (1.0 - wall_strength) * self.x_t
            return _SemanticWMState(
                x_t,
                self.h_shared.detach(),
                self.z.detach(),
            )
        raise ValueError(f"unknown isolation context: {context}")

    def detach_all(self):
        return _SemanticWMState(
            self.x_t.detach(),
            self.h_shared.detach(),
            self.z.detach(),
        )


class _WallStrengthTrackingWMState:
    def __init__(self, x_t, x_proj, s_ctrl, tracker):
        self.x_t = x_t
        self.x_proj = x_proj
        self.s_ctrl = s_ctrl
        self.z_task = None
        self._tracker = tracker

    def isolate_gradients(self, context: str, wall_strength: float = 1.0):
        self._tracker["context"] = str(context)
        self._tracker["wall_strength"] = float(wall_strength)
        return _WallStrengthTrackingWMState(
            x_t=self.x_t.detach(),
            x_proj=self.x_proj,
            s_ctrl=self.s_ctrl,
            tracker=self._tracker,
        )


class _WallStrengthTrackingWorldModel(nn.Module):
    def __init__(self, wall_strength: float, ctrl_dim: int = 3):
        super().__init__()
        self._wall_strength = float(wall_strength)
        self._wm_config = SimpleNamespace(ctrl_dim=ctrl_dim)

    def get_wall_strength(self) -> float:
        return self._wall_strength


class _WallStrengthTrackingRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_wall_strength = None

    def forward_components(self, x_rl, s_ctrl, x_t, x_proj, wall_strength):
        del s_ctrl, x_t, x_proj
        self.last_wall_strength = float(wall_strength)
        return SimpleNamespace(f_policy=x_rl)


class _SemanticRSSM:
    def get_features(self, h_shared, z):
        return torch.cat([h_shared, z], dim=-1)


class _SemanticWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_transition = _SemanticRSSM()
        self._wm_config = SimpleNamespace(is_discrete_action=False)
        self.action_to_h = nn.Linear(2, 2, bias=False)

    def init_wm_state(self, batch_size, device):
        zeros2 = torch.zeros(batch_size, 2, device=device)
        zeros4 = torch.zeros(batch_size, 4, device=device)
        return _SemanticWMState(zeros4, zeros2, zeros2)

    def forward_context(self, obs, action_prev, prev_state):
        h_shared = obs[:, :2] + self.action_to_h(action_prev)
        z = obs[:, 2:4]
        return _SemanticWMState(obs, h_shared, z), SimpleNamespace()

    def forward_imagination(self, action_prev, prev_state):
        h_shared = prev_state.h_shared + self.action_to_h(action_prev)
        z = prev_state.z + 0.25
        return _SemanticWMState(prev_state.x_t, h_shared, z), SimpleNamespace()

    def observe_sequence(self, vitals, actions, dones):
        del vitals, actions, dones
        return {}

    def build_trajectory(
        self,
        seq_result,
        vitals,
        actions,
        rewards,
        dones,
        remaining_steps=None,
    ):
        del seq_result, vitals, actions, rewards, dones, remaining_steps
        return {}

    def compute_loss(self, trajectory):
        del trajectory
        return _LossPacket(torch.tensor(0.0, device=self.action_to_h.weight.device))


class _ReplayAnchoredSemanticWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_transition = _SemanticRSSM()
        self._wm_config = SimpleNamespace(is_discrete_action=False)
        self.action_to_h = nn.Linear(2, 2, bias=False)

    def init_wm_state(self, batch_size, device):
        zeros2 = torch.zeros(batch_size, 2, device=device)
        zeros4 = torch.zeros(batch_size, 4, device=device)
        return _SemanticWMState(zeros4, zeros2, zeros2)

    def forward_context(self, obs, action_prev, prev_state):
        del obs
        h_shared = prev_state.h_shared + self.action_to_h(action_prev)
        z = prev_state.z + 0.1
        return _SemanticWMState(prev_state.x_t, h_shared, z), SimpleNamespace()

    def forward_imagination(self, action_prev, prev_state):
        h_shared = prev_state.h_shared + self.action_to_h(action_prev)
        z = prev_state.z + 0.1
        return _SemanticWMState(prev_state.x_t, h_shared, z), SimpleNamespace()

    def observe_sequence(self, vitals, actions, dones):
        del vitals, actions, dones
        return {}

    def build_trajectory(
        self,
        seq_result,
        vitals,
        actions,
        rewards,
        dones,
        remaining_steps=None,
    ):
        del seq_result, vitals, actions, rewards, dones, remaining_steps
        return {}

    def compute_loss(self, trajectory):
        del trajectory
        return _LossPacket(torch.tensor(0.0, device=self.action_to_h.weight.device))


class _FailingSemanticWorldModel(_SemanticWorldModel):
    def compute_loss(self, trajectory):
        del trajectory
        raise RuntimeError("boom from wm")


class _SemanticModel(nn.Module):
    def __init__(self, vital_dim: int):
        super().__init__()
        self.world_model = _SemanticWorldModel()
        self.critic = _TargetGapCritic(vital_dim, target_shift=1.5)
        self.actor = SimpleNamespace(is_discrete=False)

    def forward(self, vitals, temperature: float = 1.0, intent=None):
        del temperature, intent
        value = self.critic(vitals)
        dist = D.Normal(torch.zeros_like(vitals[..., :2]), torch.ones_like(vitals[..., :2]))
        return dist, value


class _FailingSemanticModel(_SemanticModel):
    def __init__(self, vital_dim: int):
        super().__init__(vital_dim)
        self.world_model = _FailingSemanticWorldModel()


class _ReplayAnchoredSemanticModel(nn.Module):
    def __init__(self, vital_dim: int):
        super().__init__()
        self.world_model = _ReplayAnchoredSemanticWorldModel()
        self.critic = _TargetGapCritic(vital_dim, target_shift=1.5)
        self.actor = SimpleNamespace(is_discrete=False)

    def forward(self, vitals, temperature: float = 1.0, intent=None):
        del temperature, intent
        value = self.critic(vitals)
        dist = D.Normal(torch.zeros_like(vitals[..., :2]), torch.ones_like(vitals[..., :2]))
        return dist, value


class TestTrainingEntryPoints(unittest.TestCase):
    def _make_batch(self, model: nn.Module, device: torch.device):
        torch.manual_seed(0)
        B, T, D, A = 2, 3, 4, 2
        vitals = torch.randn(B, T, D, device=device)
        vitals_flat = vitals.reshape(B * T, D)
        with torch.no_grad():
            dist, _ = model(vitals_flat)
            actions_flat = dist.sample()
        actions = actions_flat.view(B, T, A)
        log_probs, values, _, _ = compute_policy_outputs(model, vitals, actions)
        batch = {
            "vitals": vitals,
            "actions": actions,
            "rewards": torch.randn(B, T, device=device),
            "dones": torch.zeros(B, T, device=device),
            "log_probs": log_probs.detach(),
            "advantages": torch.randn(B, T, device=device),
            "returns": torch.randn(B, T, device=device),
            "values": values.detach(),
        }
        return batch

    def test_training_step_requires_keys(self):
        device = torch.device("cpu")
        model = _DummyModel(4, 2, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stepper = TrainingStep(model=model, opt_bundle=_single_rl_opt_bundle(optimizer), config=TrainingConfig(), device=device)
        batch = {"vitals": torch.zeros(1, 1, 4), "actions": torch.zeros(1, 1, 2)}
        with self.assertRaises(ValueError):
            stepper.run_step(batch)

    def test_build_wm_batch_preserves_remaining_steps(self):
        device = torch.device("cpu")
        model = _DummyModel(4, 2, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = TrainingConfig(batch_size=2, wm_batch_size=2, seq_len=4, wm_seq_len=4)
        loop = TrainingLoop(
            model=model,
            opt_bundle=_single_rl_opt_bundle(optimizer),
            config=config,
            buffer=None,
            device=device,
        )

        class _Buffer:
            def sample_sequences(self, *args, **kwargs):
                raise AssertionError("WM batch should request remaining_steps-aware sampling")

            def sample_sequences_with_remaining(
                self,
                batch_size,
                seq_len,
                include_obs=False,
                max_remaining=100,
            ):
                del include_obs, max_remaining
                return {
                    "vitals": np.zeros((batch_size, seq_len + 1, 4), dtype=np.float32),
                    "actions": np.zeros((batch_size, seq_len, 2), dtype=np.float32),
                    "rewards": np.zeros((batch_size, seq_len), dtype=np.float32),
                    "dones": np.zeros((batch_size, seq_len), dtype=np.float32),
                    "log_probs": np.zeros((batch_size, seq_len), dtype=np.float32),
                    "remaining_steps": np.full((batch_size, seq_len), 7, dtype=np.int32),
                }

        loop.buffer = _Buffer()
        batch = loop._build_wm_batch()

        self.assertIsNotNone(batch)
        self.assertIn("remaining_steps", batch)
        self.assertTrue(torch.equal(batch["remaining_steps"], torch.full((2, 4), 7, dtype=torch.int32)))

    def test_training_step_matches_cloned_training_step(self):
        device = torch.device("cpu")
        config = TrainingConfig(wm_pretrain_steps=0, warmup_steps=0)

        model1 = _DummyModel(4, 2, device).to(device)
        model2 = _DummyModel(4, 2, device).to(device)
        model2.load_state_dict(model1.state_dict())
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        batch = self._make_batch(model1, device)
        stepper1 = TrainingStep(model=model1, opt_bundle=_single_rl_opt_bundle(optimizer1), config=config, device=device)
        stepper2 = TrainingStep(model=model2, opt_bundle=_single_rl_opt_bundle(optimizer2), config=config, device=device)
        metrics1 = stepper1.run_step(batch)
        metrics2 = stepper2.run_step(batch)
        for key in ("loss_total", "loss_actor", "loss_critic"):
            self.assertAlmostEqual(metrics1[key], metrics2[key], places=6)

    def test_gae_matches_manual_recursion(self):
        torch.manual_seed(0)
        B, T = 3, 5
        rewards = torch.randn(B, T)
        values = torch.randn(B, T + 1)
        dones = torch.zeros(B, T)
        dones[0, 2] = 1.0
        dones[2, 4] = 1.0
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

        expected_adv = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, dtype=rewards.dtype)
        for t in reversed(range(T)):
            disc_t = 0.99 * (1.0 - dones[:, t])
            delta = rewards[:, t] + disc_t * values[:, t + 1] - values[:, t]
            last_gae = delta + disc_t * 0.95 * last_gae
            expected_adv[:, t] = last_gae
        expected_ret = expected_adv + values[:, :-1]

        self.assertTrue(torch.allclose(adv, expected_adv, atol=1e-6))
        self.assertTrue(torch.allclose(ret, expected_ret, atol=1e-6))

    def test_actor_loss_dynamics_mode_backprops_target(self):
        torch.manual_seed(0)
        x = torch.randn(6, requires_grad=True)
        target = x * 2.0
        base = torch.zeros_like(target)
        weights = torch.ones_like(target)
        log_probs = torch.zeros_like(target)
        entropy = torch.zeros_like(target)

        loss, _ = compute_dreamer_actor_loss(
            log_probs=log_probs,
            target=target,
            base=base,
            weights=weights,
            entropy=entropy,
            imag_gradient="dynamics",
            use_reward_ema=False,
        )
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum().item()), 0.0)

    def test_actor_loss_dynamics_ignores_baseline_in_analytic_term(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.ones_like(target)
        log_probs = torch.zeros_like(target)
        entropy = torch.zeros_like(target)

        loss_zero_base, _ = compute_dreamer_actor_loss(
            log_probs=log_probs,
            target=target,
            base=torch.zeros_like(target),
            weights=weights,
            entropy=entropy,
            imag_gradient="dynamics",
            use_reward_ema=False,
        )
        loss_large_base, _ = compute_dreamer_actor_loss(
            log_probs=log_probs,
            target=target,
            base=torch.full_like(target, 100.0),
            weights=weights,
            entropy=entropy,
            imag_gradient="dynamics",
            use_reward_ema=False,
        )

        self.assertAlmostEqual(float(loss_zero_base.item()), float(loss_large_base.item()), places=7)

    def test_actor_loss_dynamics_corridor_semantic_blend_makes_negative_advantage_reduce_main_target(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        base = torch.tensor([3.0, 3.0, 3.0])
        weights = torch.ones_like(target)
        log_probs = torch.zeros_like(target)
        entropy = torch.zeros_like(target)

        _, base_metrics = compute_dreamer_actor_loss(
            log_probs=log_probs,
            target=target,
            base=base,
            weights=weights,
            entropy=entropy,
            imag_gradient="dynamics",
            use_reward_ema=False,
        )
        _, blended_metrics = compute_dreamer_actor_loss(
            log_probs=log_probs,
            target=target,
            base=base,
            weights=weights,
            entropy=entropy,
            imag_gradient="dynamics",
            use_reward_ema=False,
            corridor_semantic_blend_beta=torch.ones_like(target),
        )

        self.assertGreater(float(base_metrics["weighted_actor_target_mean"]), 0.0)
        self.assertLess(float(blended_metrics["weighted_actor_target_mean"]), 0.0)
        self.assertAlmostEqual(
            float(blended_metrics["corridor_semantic_blend_mean"]),
            1.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(blended_metrics["corridor_semantic_blend_active_fraction"]),
            1.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(blended_metrics["corridor_semantic_adv_term_mean"]),
            -1.0,
            places=7,
        )

    def test_actor_loss_dynamics_corridor_semantic_blend_clamps_negative_advantage_term(self):
        target = torch.tensor([1.0, 1.0])
        base = torch.tensor([10.0, 10.0])
        weights = torch.ones_like(target)
        log_probs = torch.zeros_like(target)
        entropy = torch.zeros_like(target)

        _, metrics = compute_dreamer_actor_loss(
            log_probs=log_probs,
            target=target,
            base=base,
            weights=weights,
            entropy=entropy,
            imag_gradient="dynamics",
            use_reward_ema=False,
            corridor_semantic_blend_beta=torch.ones_like(target),
            corridor_semantic_advantage_term_clamp_min=0.5,
        )

        self.assertAlmostEqual(
            float(metrics["corridor_semantic_adv_term_clamp"]),
            0.5,
            places=7,
        )
        self.assertAlmostEqual(
            float(metrics["corridor_semantic_adv_term_mean"]),
            -0.5,
            places=7,
        )
        self.assertAlmostEqual(
            float(metrics["weighted_actor_target_mean"]),
            -0.5,
            places=7,
        )

    def test_actor_loss_reinforce_mode_backprops_logprob_only(self):
        torch.manual_seed(0)
        lp_src = torch.randn(6, requires_grad=True)
        tgt_src = torch.randn(6, requires_grad=True)
        log_probs = lp_src
        target = tgt_src
        base = torch.zeros_like(target)
        weights = torch.ones_like(target)
        entropy = torch.zeros_like(target)

        loss, _ = compute_dreamer_actor_loss(
            log_probs=log_probs,
            target=target,
            base=base,
            weights=weights,
            entropy=entropy,
            imag_gradient="reinforce",
            use_reward_ema=False,
        )
        loss.backward()
        self.assertIsNotNone(lp_src.grad)
        self.assertGreater(float(lp_src.grad.abs().sum().item()), 0.0)
        if tgt_src.grad is not None:
            self.assertAlmostEqual(float(tgt_src.grad.abs().sum().item()), 0.0, places=7)

    def test_actor_loss_discrete_dynamics_adds_reinforce_aux(self):
        torch.manual_seed(0)
        lp_src = torch.randn(6, requires_grad=True)
        tgt_src = torch.randn(6, requires_grad=True)
        base = torch.zeros_like(tgt_src)
        weights = torch.ones_like(tgt_src)
        entropy = torch.zeros_like(tgt_src)

        loss, metrics = compute_dreamer_actor_loss(
            log_probs=lp_src,
            target=tgt_src,
            base=base,
            weights=weights,
            entropy=entropy,
            imag_gradient="dynamics",
            actor_analytic_weight=1.0,
            actor_reinforce_aux_weight_discrete=0.25,
            is_discrete_action=True,
            use_reward_ema=False,
        )
        loss.backward()
        self.assertIsNotNone(lp_src.grad)
        self.assertGreater(float(lp_src.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(tgt_src.grad)
        self.assertGreater(float(tgt_src.grad.abs().sum().item()), 0.0)
        self.assertAlmostEqual(float(metrics["actor_analytic_coef"]), 1.0, places=7)
        self.assertAlmostEqual(float(metrics["actor_reinforce_aux_applied"]), 0.25, places=7)
        self.assertAlmostEqual(float(metrics["actor_reinforce_coef"]), 0.25, places=7)

    def test_value_real_anchor_contributes_on_imag_batch(self):
        device = torch.device("cpu")
        config = TrainingConfig()
        config.lambda_value_real_anchor = 0.5
        config.rl.use_value_real_anchor = True
        model = _DummyModel(4, 2, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stepper = TrainingStep(model=model, opt_bundle=_single_rl_opt_bundle(optimizer), config=config, device=device)

        batch = self._make_batch(model, device)
        imag_batch = dict(batch)
        imag_batch["target_actor"] = imag_batch["returns"] + 1.0
        real_anchor_batch = dict(batch)
        real_anchor_batch["value_real"] = real_anchor_batch["returns"] + 2.0

        metrics = stepper.run_step(
            rl_batch=imag_batch,
            wm_batch=batch,
            value_real_batch=real_anchor_batch,
            source_tag="imag",
        )
        self.assertIn("critic/value_real_anchor", metrics)
        self.assertGreater(float(metrics["critic/value_real_anchor"]), 0.0)

    def test_imag_semantic_anchor_contributes_on_imag_batch(self):
        device = torch.device("cpu")
        torch.manual_seed(0)
        config = TrainingConfig()
        config.adaptive_imag_critic_semantic_anchor_enabled = True
        config.adaptive_imag_critic_semantic_anchor_weight = 0.5
        model = _DummyModel(4, 2, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stepper = TrainingStep(
            model=model,
            opt_bundle=_single_rl_opt_bundle(optimizer),
            config=config,
            device=device,
        )

        batch = self._make_batch(model, device)
        imag_batch = dict(batch)
        imag_batch["target_actor"] = imag_batch["returns"] + 1.0
        imag_batch["weights_actor"] = torch.ones_like(imag_batch["returns"])
        with torch.no_grad():
            anchor_pred = model.critic(
                batch["vitals"].reshape(
                    -1, batch["vitals"].shape[-1]
                )
            ).reshape_as(batch["returns"])
        imag_batch["critic_semantic_anchor_target"] = anchor_pred.detach() + 3.0
        imag_batch["critic_semantic_anchor_weight"] = torch.ones_like(
            imag_batch["returns"]
        )
        imag_batch["critic_semantic_anchor_clean_weight"] = torch.full_like(
            imag_batch["returns"],
            0.25,
        )
        imag_batch["critic_semantic_anchor_external_weight"] = torch.full_like(
            imag_batch["returns"],
            0.75,
        )

        metrics = stepper.run_step(
            rl_batch=imag_batch,
            wm_batch=batch,
            source_tag="imag",
        )

        self.assertIn("critic/imag_semantic_anchor", metrics)
        self.assertGreater(float(metrics["critic/imag_semantic_anchor"]), 0.0)
        self.assertAlmostEqual(
            float(metrics["critic/imag_semantic_anchor_active_fraction"]),
            1.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(metrics["critic/imag_semantic_anchor_clean_weight_mean"]),
            0.25,
            places=7,
        )
        self.assertAlmostEqual(
            float(metrics["critic/imag_semantic_anchor_external_weight_mean"]),
            0.75,
            places=7,
        )

    def test_imag_semantic_anchor_ignores_zero_actor_weights(self):
        device = torch.device("cpu")
        torch.manual_seed(0)
        config = TrainingConfig()
        config.adaptive_imag_critic_semantic_anchor_enabled = True
        config.adaptive_imag_critic_semantic_anchor_weight = 0.5
        model = _DummyModel(4, 2, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stepper = TrainingStep(
            model=model,
            opt_bundle=_single_rl_opt_bundle(optimizer),
            config=config,
            device=device,
        )

        batch = self._make_batch(model, device)
        imag_batch = dict(batch)
        imag_batch["target_actor"] = imag_batch["returns"] + 1.0
        imag_batch["weights_actor"] = torch.zeros_like(imag_batch["returns"])
        with torch.no_grad():
            anchor_pred = model.critic(
                batch["vitals"].reshape(
                    -1, batch["vitals"].shape[-1]
                )
            ).reshape_as(batch["returns"])
        imag_batch["critic_semantic_anchor_target"] = anchor_pred.detach() + 4.0
        imag_batch["critic_semantic_anchor_weight"] = torch.ones_like(
            imag_batch["returns"]
        )
        imag_batch["critic_semantic_anchor_clean_weight"] = torch.full_like(
            imag_batch["returns"],
            1.0,
        )
        imag_batch["critic_semantic_anchor_external_weight"] = torch.zeros_like(
            imag_batch["returns"]
        )

        metrics = stepper.run_step(
            rl_batch=imag_batch,
            wm_batch=batch,
            source_tag="imag",
        )

        self.assertIn("critic/imag_semantic_anchor", metrics)
        self.assertAlmostEqual(float(metrics["actor/weights_mean"]), 0.0, places=7)
        self.assertGreater(float(metrics["critic/imag_semantic_anchor"]), 0.0)
        self.assertAlmostEqual(
            float(metrics["critic/imag_semantic_anchor_active_fraction"]),
            1.0,
            places=7,
        )

    def test_run_step_corridor_semantic_blend_reduces_imag_actor_target_and_reports_metrics(self):
        device = torch.device("cpu")
        base_config = TrainingConfig()
        base_config.rl.use_reward_ema = False
        base_config.rl.use_advantage_normalization = False
        base_config.rl.actor_entropy_scale = 0.0
        base_config.adaptive_imag_idle_corridor_adv_term_clamp_min = 0.25

        blend_config = TrainingConfig()
        blend_config.rl.use_reward_ema = False
        blend_config.rl.use_advantage_normalization = False
        blend_config.rl.actor_entropy_scale = 0.0
        blend_config.adaptive_imag_idle_corridor_adv_term_clamp_min = 0.25

        torch.manual_seed(0)
        base_model = _DummyModel(4, 2, device).to(device)
        blend_model = _DummyModel(4, 2, device).to(device)
        blend_model.load_state_dict(base_model.state_dict())

        base_stepper = TrainingStep(
            model=base_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(base_model.parameters(), lr=1e-3)),
            config=base_config,
            device=device,
        )
        blend_stepper = TrainingStep(
            model=blend_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(blend_model.parameters(), lr=1e-3)),
            config=blend_config,
            device=device,
        )

        wm_batch = self._make_batch(base_model, device)
        base_log_probs, _, base_entropy, _ = compute_policy_outputs(
            base_model,
            wm_batch["vitals"],
            wm_batch["actions"],
        )
        target_actor = torch.full_like(wm_batch["returns"], 3.0)
        base_actor = torch.full_like(wm_batch["returns"], 5.0)
        weights_actor = torch.ones_like(wm_batch["returns"])

        base_rl_batch = dict(wm_batch)
        base_rl_batch["target_actor"] = target_actor
        base_rl_batch["base_actor"] = base_actor
        base_rl_batch["weights_actor"] = weights_actor
        # Corridor semantic blend is defined for the differentiable imagined
        # branch, so exercise the same precomputed policy path used in training.
        base_rl_batch["use_precomputed_policy_outputs"] = True
        base_rl_batch["policy_features"] = wm_batch["vitals"]
        base_rl_batch["log_probs"] = base_log_probs.detach()
        base_rl_batch["entropy"] = base_entropy.detach()

        blend_rl_batch = dict(base_rl_batch)
        blend_rl_batch["corridor_semantic_blend_beta"] = torch.ones_like(target_actor)
        blend_rl_batch["corridor_semantic_high_value_fraction"] = 1.0
        blend_rl_batch["corridor_semantic_neg_adv_excess_mean"] = 2.0
        blend_rl_batch["corridor_semantic_score_threshold"] = 5.0
        blend_rl_batch["corridor_semantic_uses_reference_base"] = 1.0
        blend_rl_batch["corridor_semantic_blend_max"] = 0.5

        base_metrics = base_stepper.run_step(
            rl_batch=base_rl_batch,
            wm_batch=wm_batch,
            source_tag="imag",
        )
        blend_metrics = blend_stepper.run_step(
            rl_batch=blend_rl_batch,
            wm_batch=wm_batch,
            source_tag="imag",
        )

        self.assertGreater(
            float(base_metrics["actor/weighted_actor_target_mean"]),
            0.0,
        )
        self.assertLess(
            float(blend_metrics["actor/weighted_actor_target_mean"]),
            float(base_metrics["actor/weighted_actor_target_mean"]),
        )
        self.assertAlmostEqual(
            float(blend_metrics["actor/corridor_semantic_blend_mean"]),
            1.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(blend_metrics["actor/corridor_semantic_high_value_fraction"]),
            1.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(blend_metrics["actor/corridor_semantic_uses_reference_base"]),
            1.0,
            places=7,
        )

    def test_run_step_task_corridor_scale_reduces_imag_actor_target_and_reports_metrics(self):
        device = torch.device("cpu")
        base_config = TrainingConfig()
        base_config.rl.use_reward_ema = False
        base_config.rl.use_advantage_normalization = False
        base_config.rl.actor_entropy_scale = 0.0

        scaled_config = TrainingConfig()
        scaled_config.rl.use_reward_ema = False
        scaled_config.rl.use_advantage_normalization = False
        scaled_config.rl.actor_entropy_scale = 0.0

        torch.manual_seed(0)
        base_model = _DummyModel(4, 2, device).to(device)
        scaled_model = _DummyModel(4, 2, device).to(device)
        scaled_model.load_state_dict(base_model.state_dict())

        base_stepper = TrainingStep(
            model=base_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(base_model.parameters(), lr=1e-3)),
            config=base_config,
            device=device,
        )
        scaled_stepper = TrainingStep(
            model=scaled_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(scaled_model.parameters(), lr=1e-3)),
            config=scaled_config,
            device=device,
        )

        wm_batch = self._make_batch(base_model, device)
        base_log_probs, _, base_entropy, _ = compute_policy_outputs(
            base_model,
            wm_batch["vitals"],
            wm_batch["actions"],
        )
        target_actor = torch.full_like(wm_batch["returns"], 3.0)
        base_actor = torch.full_like(wm_batch["returns"], 1.0)
        weights_actor = torch.ones_like(wm_batch["returns"])

        base_rl_batch = dict(wm_batch)
        base_rl_batch["target_actor"] = target_actor
        base_rl_batch["base_actor"] = base_actor
        base_rl_batch["weights_actor"] = weights_actor
        base_rl_batch["use_precomputed_policy_outputs"] = True
        base_rl_batch["policy_features"] = wm_batch["vitals"]
        base_rl_batch["log_probs"] = base_log_probs.detach()
        base_rl_batch["entropy"] = base_entropy.detach()

        scaled_rl_batch = dict(base_rl_batch)
        scaled_rl_batch["task_corridor_analytic_scale"] = torch.full_like(
            target_actor,
            0.25,
        )
        scaled_rl_batch["task_corridor_score_mean"] = -0.5
        scaled_rl_batch["task_corridor_gate_mean"] = 0.1
        scaled_rl_batch["task_corridor_active_fraction"] = 1.0
        scaled_rl_batch["task_corridor_confidence_mean"] = 1.0
        scaled_rl_batch["task_corridor_high_return_similarity_mean"] = -0.5
        scaled_rl_batch["task_corridor_low_return_similarity_mean"] = 0.5
        scaled_rl_batch["task_corridor_real_separation"] = 1.0
        scaled_rl_batch["task_corridor_score_threshold"] = 0.0
        scaled_rl_batch["task_corridor_analytic_scale_mean"] = 0.25
        scaled_rl_batch["high_value_but_low_task_fraction"] = 1.0
        scaled_rl_batch["task_geom_corridor_disagreement"] = 0.9

        base_metrics = base_stepper.run_step(
            rl_batch=base_rl_batch,
            wm_batch=wm_batch,
            source_tag="imag",
        )
        scaled_metrics = scaled_stepper.run_step(
            rl_batch=scaled_rl_batch,
            wm_batch=wm_batch,
            source_tag="imag",
        )

        self.assertGreater(
            float(base_metrics["actor/weighted_actor_target_mean"]),
            0.0,
        )
        self.assertLess(
            float(scaled_metrics["actor/weighted_actor_target_mean"]),
            float(base_metrics["actor/weighted_actor_target_mean"]),
        )
        self.assertAlmostEqual(
            float(scaled_metrics["actor/analytic_weight_scale_task_corridor"]),
            0.25,
            places=7,
        )
        self.assertAlmostEqual(
            float(scaled_metrics["actor/task_corridor_analytic_scale_mean"]),
            0.25,
            places=7,
        )
        self.assertAlmostEqual(
            float(scaled_metrics["actor/high_value_but_low_task_fraction"]),
            1.0,
            places=7,
        )

    def test_value_real_anchor_corridor_mask_limits_penalty_to_high_value_states(self):
        device = torch.device("cpu")
        config = TrainingConfig()
        config.lambda_value_real_anchor = 0.5
        config.rl.use_value_real_anchor = True

        torch.manual_seed(0)
        masked_model = _DummyModel(4, 2, device).to(device)
        plain_model = _DummyModel(4, 2, device).to(device)
        plain_model.load_state_dict(masked_model.state_dict())

        masked_optimizer = torch.optim.Adam(masked_model.parameters(), lr=1e-3)
        plain_optimizer = torch.optim.Adam(plain_model.parameters(), lr=1e-3)

        masked_stepper = TrainingStep(
            model=masked_model,
            opt_bundle=_single_rl_opt_bundle(masked_optimizer),
            config=config,
            device=device,
        )
        plain_stepper = TrainingStep(
            model=plain_model,
            opt_bundle=_single_rl_opt_bundle(plain_optimizer),
            config=config,
            device=device,
        )

        masked_batch = self._make_batch(masked_model, device)
        plain_batch = self._make_batch(plain_model, device)

        masked_imag_batch = dict(masked_batch)
        masked_imag_batch["target_actor"] = masked_imag_batch["returns"] + 1.0
        plain_imag_batch = dict(plain_batch)
        plain_imag_batch["target_actor"] = plain_imag_batch["returns"] + 1.0

        with torch.no_grad():
            masked_anchor_pred = masked_model.critic(
                masked_batch["vitals"].reshape(-1, masked_batch["vitals"].shape[-1])
            ).reshape_as(masked_batch["returns"])
            plain_anchor_pred = plain_model.critic(
                plain_batch["vitals"].reshape(-1, plain_batch["vitals"].shape[-1])
            ).reshape_as(plain_batch["returns"])

        masked_real_anchor_batch = dict(masked_batch)
        masked_real_anchor_batch["value_real"] = masked_anchor_pred.detach().clone()
        masked_real_anchor_batch["value_real"][0, 1:] += 10.0
        masked_real_anchor_batch["value_real"][1, :] += 10.0
        masked_real_anchor_batch["value_real_corridor_mask"] = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=masked_batch["returns"].dtype,
            device=device,
        )

        plain_real_anchor_batch = dict(plain_batch)
        plain_real_anchor_batch["value_real"] = plain_anchor_pred.detach().clone()
        plain_real_anchor_batch["value_real"][0, 1:] += 10.0
        plain_real_anchor_batch["value_real"][1, :] += 10.0

        masked_metrics = masked_stepper.run_step(
            rl_batch=masked_imag_batch,
            wm_batch=masked_batch,
            value_real_batch=masked_real_anchor_batch,
            source_tag="imag",
        )
        plain_metrics = plain_stepper.run_step(
            rl_batch=plain_imag_batch,
            wm_batch=plain_batch,
            value_real_batch=plain_real_anchor_batch,
            source_tag="imag",
        )

        self.assertIn("critic/value_real_anchor_corridor_active", masked_metrics)
        self.assertAlmostEqual(
            float(masked_metrics["critic/value_real_anchor_corridor_active"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(masked_metrics["critic/value_real_anchor_corridor_fraction"]),
            1.0 / 6.0,
            places=6,
        )
        self.assertLess(
            float(masked_metrics["critic/value_real_anchor"]),
            float(plain_metrics["critic/value_real_anchor"]),
        )

    def test_actor_entropy_scale_is_exposed_in_rl_cfg(self):
        device = torch.device("cpu")
        config = TrainingConfig()
        config.rl.actor_entropy_scale = 9e-4
        config.rl.imag_gradient = "both"
        config.rl.imag_gradient_mix = 0.3
        config.rl.use_reward_ema = True
        config.rl.actor_analytic_weight = 0.8
        config.rl.actor_reinforce_aux_weight_discrete = 0.2
        model = _DummyModel(4, 2, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stepper = TrainingStep(model=model, opt_bundle=_single_rl_opt_bundle(optimizer), config=config, device=device)
        rl_cfg = stepper._rl_cfg()
        self.assertIn("actor_entropy_scale", rl_cfg)
        self.assertAlmostEqual(float(rl_cfg["actor_entropy_scale"]), 9e-4, places=10)
        self.assertAlmostEqual(float(rl_cfg["actor_analytic_weight"]), 0.8, places=10)
        self.assertAlmostEqual(float(rl_cfg["actor_reinforce_aux_weight_discrete"]), 0.2, places=10)
        self.assertEqual(str(rl_cfg["imag_gradient"]), "both")
        self.assertAlmostEqual(float(rl_cfg["imag_gradient_mix"]), 0.3, places=10)
        self.assertTrue(bool(rl_cfg["use_reward_ema"]))

    def test_precomputed_policy_outputs_handles_flat_value_shape(self):
        device = torch.device("cpu")
        config = TrainingConfig()
        model = _DummyModelPrecomputed(4, 2, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stepper = TrainingStep(model=model, opt_bundle=_single_rl_opt_bundle(optimizer), config=config, device=device)

        torch.manual_seed(1)
        B, T, D, A = 2, 4, 4, 2
        vitals = torch.randn(B, T, D, device=device)
        actions = torch.randn(B, T, A, device=device)
        log_probs = torch.randn(B, T, device=device)
        entropy = torch.rand(B, T, device=device)
        returns = torch.randn(B, T, device=device)
        advantages = torch.randn(B, T, device=device)
        policy_features = vitals.clone()

        batch = {
            "vitals": vitals,
            "actions": actions,
            "rewards": torch.randn(B, T, device=device),
            "dones": torch.zeros(B, T, device=device),
            "log_probs": log_probs,
            "entropy": entropy,
            "advantages": advantages,
            "returns": returns,
            "use_precomputed_policy_outputs": True,
            "policy_features": policy_features,
            "target_actor": returns,
            "base_actor": torch.zeros_like(returns),
            "weights_actor": torch.ones_like(returns),
        }
        metrics = stepper.run_step(batch=batch, rl_batch=batch, wm_batch=batch)
        self.assertIn("loss_actor", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss_actor"])))

    def test_imagined_critic_detach_blocks_critic_grad_into_policy_features(self):
        device = torch.device("cpu")
        B, T, D, A = 2, 3, 4, 2

        def make_batch(req_grad: bool = True):
            policy_features = torch.randn(B, T, D, device=device, requires_grad=req_grad)
            return {
                "vitals": policy_features,
                "policy_features": policy_features,
                "use_precomputed_policy_outputs": True,
                "actions": torch.randn(B, T, A, device=device),
                "rewards": torch.randn(B, T, device=device),
                "dones": torch.zeros(B, T, device=device),
                "log_probs": torch.zeros(B, T, device=device),
                "entropy": torch.zeros(B, T, device=device),
                "advantages": torch.zeros(B, T, device=device),
                "returns": torch.ones(B, T, device=device),
                "target_actor": torch.ones(B, T, device=device).detach(),
                "base_actor": torch.zeros(B, T, device=device),
                "weights_actor": torch.ones(B, T, device=device),
            }

        config = TrainingConfig()
        config.rl.use_value_normalization = False
        config.rl.detach_critic_features_on_imagination = True
        model = _DummyModelPrecomputedDrift(4, 2, device, target_shift=5.0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stepper = TrainingStep(model=model, opt_bundle=_single_rl_opt_bundle(optimizer), config=config, device=device)

        batch_detached = make_batch(req_grad=True)
        stepper.run_step(batch=batch_detached, rl_batch=batch_detached, wm_batch=batch_detached, source_tag="imag")
        grad_detached = batch_detached["policy_features"].grad
        grad_detached_sum = 0.0 if grad_detached is None else float(grad_detached.abs().sum().item())
        self.assertAlmostEqual(grad_detached_sum, 0.0, places=7)

        config_no_detach = TrainingConfig()
        config_no_detach.rl.use_value_normalization = False
        config_no_detach.rl.detach_critic_features_on_imagination = False
        model_no_detach = _DummyModelPrecomputedDrift(4, 2, device, target_shift=5.0).to(device)
        optimizer_no_detach = torch.optim.Adam(model_no_detach.parameters(), lr=1e-3)
        stepper_no_detach = TrainingStep(model=model_no_detach, opt_bundle=_single_rl_opt_bundle(optimizer_no_detach), config=config_no_detach, device=device)
        batch_live = make_batch(req_grad=True)
        stepper_no_detach.run_step(batch=batch_live, rl_batch=batch_live, wm_batch=batch_live, source_tag="imag")
        grad_live = batch_live["policy_features"].grad
        self.assertIsNotNone(grad_live)
        self.assertGreater(float(grad_live.abs().sum().item()), 0.0)

    def test_training_step_policy_value_helper_keeps_live_gradients_when_falling_back_from_target_kwarg(self):
        device = torch.device("cpu")
        model = _DummyModel(4, 2, device).to(device)
        model.critic = _FallbackGradTrackingCritic(4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stepper = TrainingStep(
            model=model,
            opt_bundle=_single_rl_opt_bundle(optimizer),
            config=TrainingConfig(),
            device=device,
        )

        policy_features = torch.randn(2, 3, 4, device=device, requires_grad=True)

        values = stepper._compute_policy_feature_values(
            policy_features,
            use_target=True,
        )

        self.assertIsNotNone(values)
        self.assertTrue(bool(model.critic.last_grad_enabled))
        self.assertTrue(bool(model.critic.last_input_requires_grad))
        values.sum().backward()
        self.assertIsNotNone(policy_features.grad)
        self.assertGreater(float(policy_features.grad.abs().sum().item()), 0.0)

    def test_training_loop_policy_value_helper_detaches_and_disables_grad_when_falling_back_from_target_kwarg(self):
        device = torch.device("cpu")
        model = _DummyModel(4, 2, device).to(device)
        model.critic = _FallbackGradTrackingCritic(4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loop = TrainingLoop(
            model=model,
            opt_bundle=_single_rl_opt_bundle(optimizer),
            config=TrainingConfig(),
            buffer=None,
            device=device,
        )

        policy_features = torch.randn(2, 3, 4, device=device, requires_grad=True)

        values = loop._compute_policy_feature_values(
            policy_features,
            use_target=True,
        )

        self.assertIsNotNone(values)
        self.assertFalse(bool(model.critic.last_grad_enabled))
        self.assertFalse(bool(model.critic.last_input_requires_grad))
        self.assertFalse(bool(values.requires_grad))
        self.assertIsNone(values.grad_fn)
        self.assertIsNone(policy_features.grad)

    def test_training_step_uses_critic_compute_loss_contract(self):
        device = torch.device("cpu")
        config = TrainingConfig()
        config.rl.use_value_normalization = False
        config.rl.detach_critic_features_on_imagination = True

        model = _DummyModelRecordingCritic(4, 2, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stepper = TrainingStep(model=model, opt_bundle=_single_rl_opt_bundle(optimizer), config=config, device=device)

        torch.manual_seed(7)
        B, T, D, A = 2, 3, 4, 2
        policy_features = torch.randn(B, T, D, device=device)
        batch = {
            "vitals": policy_features,
            "policy_features": policy_features,
            "use_precomputed_policy_outputs": True,
            "actions": torch.randn(B, T, A, device=device),
            "rewards": torch.randn(B, T, device=device),
            "dones": torch.zeros(B, T, device=device),
            "log_probs": torch.zeros(B, T, device=device),
            "entropy": torch.zeros(B, T, device=device),
            "advantages": torch.zeros(B, T, device=device),
            "returns": torch.ones(B, T, device=device),
            "target_actor": torch.ones(B, T, device=device),
            "base_actor": torch.zeros(B, T, device=device),
            "weights_actor": torch.ones(B, T, device=device),
        }

        metrics = stepper.run_step(batch=batch, rl_batch=batch, wm_batch=batch, source_tag="imag")

        self.assertEqual(model.critic.compute_loss_calls, 1)
        self.assertTrue(bool(model.critic.last_detach_features))
        self.assertIsInstance(model.critic.last_targets, dict)
        self.assertIn(model.critic.primary_gamma, model.critic.last_targets)
        self.assertAlmostEqual(float(metrics["loss_critic"]), model.critic.fixed_loss, places=6)

    def test_training_step_keeps_router_loss_enabled_when_critic_supports_it(self):
        device = torch.device("cpu")
        config = TrainingConfig()
        config.rl.use_value_normalization = False
        config.rl.detach_critic_features_on_imagination = True

        model = _DummyModelRouterCritic(4, 2, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stepper = TrainingStep(model=model, opt_bundle=_single_rl_opt_bundle(optimizer), config=config, device=device)

        torch.manual_seed(11)
        B, T, D, A = 2, 3, 4, 2
        policy_features = torch.randn(B, T, D, device=device)
        batch = {
            "vitals": policy_features,
            "policy_features": policy_features,
            "use_precomputed_policy_outputs": True,
            "actions": torch.randn(B, T, A, device=device),
            "rewards": torch.randn(B, T, device=device),
            "dones": torch.zeros(B, T, device=device),
            "log_probs": torch.zeros(B, T, device=device),
            "entropy": torch.zeros(B, T, device=device),
            "advantages": torch.zeros(B, T, device=device),
            "returns": torch.ones(B, T, device=device),
            "target_actor": torch.ones(B, T, device=device),
            "base_actor": torch.zeros(B, T, device=device),
            "weights_actor": torch.ones(B, T, device=device),
        }

        metrics = stepper.run_step(batch=batch, rl_batch=batch, wm_batch=batch, source_tag="imag")

        self.assertEqual(model.critic.compute_loss_calls, 1)
        self.assertTrue(bool(model.critic.last_include_router_loss))
        self.assertAlmostEqual(
            float(metrics["critic/loss_contract_router"]),
            model.critic.router_loss,
            places=6,
        )
        self.assertAlmostEqual(float(metrics["critic/router_route"]), 0.2, places=6)
        self.assertAlmostEqual(float(metrics["critic/router_load_balance"]), 0.1, places=6)
        self.assertAlmostEqual(float(metrics["critic/router_entropy"]), 0.05, places=6)

    def test_model_wrapper_policy_from_wm_state_uses_world_model_wall_strength(self):
        device = torch.device("cpu")
        tracker = {}
        wm_state = _WallStrengthTrackingWMState(
            x_t=torch.randn(2, 4, device=device, requires_grad=True),
            x_proj=torch.randn(2, 4, device=device),
            s_ctrl=torch.randn(2, 3, device=device),
            tracker=tracker,
        )
        world_model = _WallStrengthTrackingWorldModel(wall_strength=0.25, ctrl_dim=3)
        router = _WallStrengthTrackingRouter()
        wrapper = _ModelWrapper(
            actor=nn.Identity(),
            critic=nn.Identity(),
            world_model=world_model,
            router=router,
        )

        features = wrapper.policy_from_wm_state(wm_state)

        self.assertAlmostEqual(float(tracker["wall_strength"]), 0.25, places=6)
        self.assertEqual(str(tracker["context"]), "policy")
        self.assertAlmostEqual(float(router.last_wall_strength), 0.25, places=6)
        self.assertTrue(torch.allclose(features, wm_state.x_proj))

    def test_actor_drift_guard_and_slow_reg_scale_activate_on_large_gap(self):
        device = torch.device("cpu")
        config = TrainingConfig()
        config.rl.use_value_normalization = False
        config.rl.use_actor_drift_guard = True
        config.rl.actor_drift_guard_threshold = 1.0
        config.rl.actor_drift_guard_gain = 2.0
        config.rl.actor_drift_guard_floor = 0.2
        config.rl.slow_value_reg_drift_threshold = 1.0
        config.rl.slow_value_reg_drift_gain = 3.0
        model = _DummyModelPrecomputedDrift(4, 2, device, target_shift=4.0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stepper = TrainingStep(model=model, opt_bundle=_single_rl_opt_bundle(optimizer), config=config, device=device)

        torch.manual_seed(2)
        B, T, D, A = 2, 3, 4, 2
        policy_features = torch.randn(B, T, D, device=device)
        batch = {
            "vitals": policy_features,
            "policy_features": policy_features,
            "use_precomputed_policy_outputs": True,
            "actions": torch.randn(B, T, A, device=device),
            "rewards": torch.randn(B, T, device=device),
            "dones": torch.zeros(B, T, device=device),
            "log_probs": torch.zeros(B, T, device=device),
            "entropy": torch.zeros(B, T, device=device),
            "advantages": torch.zeros(B, T, device=device),
            "returns": torch.ones(B, T, device=device),
            "target_actor": torch.ones(B, T, device=device).detach(),
            "base_actor": torch.zeros(B, T, device=device),
            "weights_actor": torch.ones(B, T, device=device),
        }
        metrics = stepper.run_step(batch=batch, rl_batch=batch, wm_batch=batch, source_tag="imag")
        self.assertIn("actor/drift_guard_scale", metrics)
        self.assertIn("critic/slow_reg_scale", metrics)
        self.assertLess(float(metrics["actor/drift_guard_scale"]), 1.0)
        self.assertGreater(float(metrics["critic/slow_reg_scale"]), 1.0)
        self.assertGreater(float(metrics["critic/slow_value_gap_abs_mean"]), 1.0)
        self.assertGreater(float(metrics["actor/drift_guard_triggered"]), 0.0)

    def test_target_value_consistency_penalty_contributes_to_world_model_update(self):
        device = torch.device("cpu")
        config = TrainingConfig()
        config.adaptive_imag_target_value_consistency_weight = 0.5
        config.adaptive_imag_target_value_consistency_horizon = 2
        config.adaptive_imag_target_value_consistency_delta = 0.5
        model = _SemanticModel(4).to(device)
        opt_bundle = OptimizerBundle(
            wm_optimizer=torch.optim.Adam(model.world_model.parameters(), lr=1e-3),
            rl_optimizer=torch.optim.Adam(model.critic.parameters(), lr=1e-3),
        )
        stepper = TrainingStep(model=model, opt_bundle=opt_bundle, config=config, device=device)

        wm_batch = {
            "vitals": torch.tensor(
                [
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0, 1.0],
                        [2.0, 0.0, 1.0, 1.0],
                        [3.0, 0.0, 1.0, 1.0],
                        [4.0, 0.0, 1.0, 1.0],
                    ]
                ],
                device=device,
            ),
            "actions": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                ],
                device=device,
            ),
            "rewards": torch.zeros((1, 4), device=device),
            "dones": torch.zeros((1, 4), device=device),
        }

        metrics = stepper.run_step(wm_batch=wm_batch, skip_rl=True, source_tag="real")
        self.assertAlmostEqual(float(metrics["wm/semantic_consistency_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["wm/semantic_consistency_horizon"]), 2.0, places=6)
        self.assertAlmostEqual(float(metrics["wm/semantic_consistency_uses_target_critic"]), 1.0, places=6)
        self.assertGreater(float(metrics["wm/semantic_consistency_loss"]), 0.0)
        self.assertGreater(float(metrics["wm/semantic_consistency_penalty"]), 0.0)
        self.assertGreater(float(metrics["loss_wm"]), 0.0)
        self.assertGreater(float(metrics["wm_grad_norm"]), 0.0)

    def test_world_model_failure_aborts_step_before_rl_update(self):
        device = torch.device("cpu")
        model = _FailingSemanticModel(4).to(device)
        opt_bundle = OptimizerBundle(
            wm_optimizer=torch.optim.Adam(model.world_model.parameters(), lr=1e-3),
            rl_optimizer=torch.optim.Adam(model.critic.parameters(), lr=1e-3),
        )
        stepper = TrainingStep(model=model, opt_bundle=opt_bundle, config=TrainingConfig(), device=device)
        batch = self._make_batch(model, device)
        critic_before = {
            name: param.detach().clone()
            for name, param in model.critic.named_parameters()
        }

        with self.assertRaisesRegex(RuntimeError, "World-model update failed"):
            stepper.run_step(batch=batch, rl_batch=batch, wm_batch=batch, source_tag="real")

        for name, param in model.critic.named_parameters():
            self.assertTrue(
                torch.allclose(param.detach(), critic_before[name]),
                msg=f"critic parameter {name} changed despite world-model failure",
            )

    def test_target_value_consistency_anchors_to_replay_short_returns_when_teacher_imag_match(self):
        device = torch.device("cpu")
        config = TrainingConfig()
        config.adaptive_imag_target_value_consistency_weight = 0.5
        config.adaptive_imag_target_value_consistency_horizon = 2
        config.adaptive_imag_target_value_consistency_delta = 0.5
        model = _ReplayAnchoredSemanticModel(4).to(device)
        with torch.no_grad():
            model.world_model.action_to_h.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
            )
            model.critic.v.weight.fill_(0.25)
            model.critic.v.bias.zero_()
        opt_bundle = OptimizerBundle(
            wm_optimizer=torch.optim.Adam(model.world_model.parameters(), lr=1e-3),
            rl_optimizer=torch.optim.Adam(model.critic.parameters(), lr=1e-3),
        )
        stepper = TrainingStep(model=model, opt_bundle=opt_bundle, config=config, device=device)

        wm_batch = {
            "vitals": torch.zeros((1, 5, 4), device=device),
            "actions": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                ],
                device=device,
            ),
            "rewards": torch.tensor([[0.0, 0.0, 2.0, 1.0]], device=device),
            "dones": torch.zeros((1, 4), device=device),
        }

        metrics = stepper.run_step(wm_batch=wm_batch, skip_rl=True, source_tag="real")
        self.assertAlmostEqual(float(metrics["wm/semantic_consistency_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["wm/semantic_consistency_target_source_replay"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["wm/semantic_consistency_teacher_imag_gap_mean"]), 0.0, places=6)
        self.assertGreater(float(metrics["wm/semantic_consistency_teacher_to_real_gap_mean"]), 0.0)
        self.assertGreater(float(metrics["wm/semantic_consistency_imag_to_real_gap_mean"]), 0.0)
        self.assertGreater(float(metrics["wm/semantic_consistency_loss"]), 0.0)
        self.assertGreater(float(metrics["wm/semantic_consistency_penalty"]), 0.0)
        self.assertGreater(float(metrics["wm_grad_norm"]), 0.0)

    def test_target_value_consistency_can_boost_high_value_imagined_states(self):
        device = torch.device("cpu")
        base_config = TrainingConfig()
        base_config.adaptive_imag_target_value_consistency_weight = 0.5
        base_config.adaptive_imag_target_value_consistency_horizon = 2
        base_config.adaptive_imag_target_value_consistency_delta = 0.5

        boosted_config = TrainingConfig()
        boosted_config.adaptive_imag_target_value_consistency_weight = 0.5
        boosted_config.adaptive_imag_target_value_consistency_horizon = 2
        boosted_config.adaptive_imag_target_value_consistency_delta = 0.5
        boosted_config.adaptive_imag_target_value_consistency_high_value_boost = 2.0
        boosted_config.adaptive_imag_target_value_consistency_high_value_quantile = 0.5
        boosted_config.adaptive_imag_target_value_consistency_high_value_feature_scale = 0.25

        base_model = _SemanticModel(4).to(device)
        boosted_model = _SemanticModel(4).to(device)
        boosted_model.load_state_dict(base_model.state_dict())

        base_stepper = TrainingStep(
            model=base_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(base_model.parameters(), lr=1e-3)),
            config=base_config,
            device=device,
        )
        boosted_stepper = TrainingStep(
            model=boosted_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(boosted_model.parameters(), lr=1e-3)),
            config=boosted_config,
            device=device,
        )

        wm_batch = {
            "vitals": torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 2.0, 2.0],
                        [3.0, 0.0, 4.0, 4.0],
                        [4.0, 0.0, 6.0, 6.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.5, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 0.0],
                    ],
                ],
                device=device,
            ),
            "actions": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                ],
                device=device,
            ),
            "rewards": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 8.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                device=device,
            ),
            "dones": torch.zeros((2, 4), device=device),
        }

        base_loss, base_metrics = base_stepper._compute_target_value_consistency_loss(
            wm_batch,
            base_stepper._rl_cfg(),
        )
        boosted_loss, boosted_metrics = boosted_stepper._compute_target_value_consistency_loss(
            wm_batch,
            boosted_stepper._rl_cfg(),
        )

        self.assertGreater(
            abs(float(boosted_loss.detach().item()) - float(base_loss.detach().item())),
            1e-6,
        )
        self.assertGreater(float(boosted_metrics["wm/semantic_consistency_high_value_fraction_mean"]), 0.0)
        self.assertLess(float(boosted_metrics["wm/semantic_consistency_high_value_fraction_mean"]), 1.0)
        self.assertGreater(float(boosted_metrics["wm/semantic_consistency_high_value_weight_mean"]), 1.0)
        self.assertGreater(float(boosted_metrics["wm/semantic_consistency_high_value_feature_loss_mean"]), 0.0)
        self.assertAlmostEqual(
            float(boosted_metrics["wm/semantic_consistency_high_value_boost"]),
            2.0,
            places=6,
        )

    def test_policy_open_loop_consistency_penalty_contributes_to_world_model_update(self):
        device = torch.device("cpu")
        config = TrainingConfig()
        config.adaptive_imag_policy_open_loop_consistency_weight = 0.5
        config.adaptive_imag_policy_open_loop_consistency_horizon = 2
        config.adaptive_imag_policy_open_loop_consistency_delta = 0.5
        model = _SemanticModel(4).to(device)
        opt_bundle = OptimizerBundle(
            wm_optimizer=torch.optim.Adam(model.world_model.parameters(), lr=1e-3),
            rl_optimizer=torch.optim.Adam(model.critic.parameters(), lr=1e-3),
        )
        stepper = TrainingStep(model=model, opt_bundle=opt_bundle, config=config, device=device)

        wm_batch = {
            "vitals": torch.tensor(
                [
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0, 1.0],
                        [2.0, 0.0, 1.0, 1.0],
                        [3.0, 0.0, 1.0, 1.0],
                        [4.0, 0.0, 1.0, 1.0],
                    ]
                ],
                device=device,
            ),
            "actions": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                ],
                device=device,
            ),
            "rewards": torch.zeros((1, 4), device=device),
            "dones": torch.zeros((1, 4), device=device),
        }

        metrics = stepper.run_step(wm_batch=wm_batch, skip_rl=True, source_tag="real")
        self.assertAlmostEqual(
            float(metrics["wm/policy_open_loop_consistency_active"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["wm/policy_open_loop_consistency_horizon"]),
            2.0,
            places=6,
        )
        self.assertGreater(float(metrics["wm/policy_open_loop_consistency_loss"]), 0.0)
        self.assertGreater(float(metrics["wm/policy_open_loop_consistency_penalty"]), 0.0)
        self.assertGreater(float(metrics["loss_wm"]), 0.0)
        self.assertGreater(float(metrics["wm_grad_norm"]), 0.0)

    def test_policy_open_loop_consistency_is_zero_when_teacher_and_imagined_match(self):
        device = torch.device("cpu")
        config = TrainingConfig()
        config.adaptive_imag_policy_open_loop_consistency_weight = 0.5
        config.adaptive_imag_policy_open_loop_consistency_horizon = 2
        config.adaptive_imag_policy_open_loop_consistency_delta = 0.5
        model = _ReplayAnchoredSemanticModel(4).to(device)
        stepper = TrainingStep(
            model=model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(model.parameters(), lr=1e-3)),
            config=config,
            device=device,
        )

        wm_batch = {
            "vitals": torch.zeros((1, 5, 4), device=device),
            "actions": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                ],
                device=device,
            ),
            "rewards": torch.tensor([[0.0, 0.0, 2.0, 1.0]], device=device),
            "dones": torch.zeros((1, 4), device=device),
        }

        loss, metrics = stepper._compute_policy_open_loop_consistency_loss(
            wm_batch,
            stepper._rl_cfg(),
        )

        self.assertAlmostEqual(
            float(metrics["wm/policy_open_loop_consistency_active"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(float(loss.detach().item()), 0.0, places=6)
        self.assertAlmostEqual(
            float(metrics["wm/policy_open_loop_consistency_loss"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["wm/policy_open_loop_consistency_penalty"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["wm/policy_open_loop_consistency_feature_l1_mean"]),
            0.0,
            places=6,
        )

    def test_policy_open_loop_consistency_value_scale_penalizes_semantic_drift_even_when_features_match(self):
        device = torch.device("cpu")
        base_config = TrainingConfig()
        base_config.adaptive_imag_policy_open_loop_consistency_weight = 0.5
        base_config.adaptive_imag_policy_open_loop_consistency_horizon = 2
        base_config.adaptive_imag_policy_open_loop_consistency_delta = 0.5

        value_config = TrainingConfig()
        value_config.adaptive_imag_policy_open_loop_consistency_weight = 0.5
        value_config.adaptive_imag_policy_open_loop_consistency_horizon = 2
        value_config.adaptive_imag_policy_open_loop_consistency_delta = 0.5
        value_config.adaptive_imag_policy_open_loop_consistency_value_scale = 1.0

        base_model = _ReplayAnchoredSemanticModel(4).to(device)
        value_model = _ReplayAnchoredSemanticModel(4).to(device)
        value_model.load_state_dict(base_model.state_dict())

        base_stepper = TrainingStep(
            model=base_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(base_model.parameters(), lr=1e-3)),
            config=base_config,
            device=device,
        )
        value_stepper = TrainingStep(
            model=value_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(value_model.parameters(), lr=1e-3)),
            config=value_config,
            device=device,
        )

        wm_batch = {
            "vitals": torch.zeros((1, 5, 4), device=device),
            "actions": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                ],
                device=device,
            ),
            "rewards": torch.tensor([[0.0, 0.0, 2.0, 1.0]], device=device),
            "dones": torch.zeros((1, 4), device=device),
        }

        base_loss, _ = base_stepper._compute_policy_open_loop_consistency_loss(
            wm_batch,
            base_stepper._rl_cfg(),
        )
        value_loss, value_metrics = value_stepper._compute_policy_open_loop_consistency_loss(
            wm_batch,
            value_stepper._rl_cfg(),
        )

        self.assertAlmostEqual(float(base_loss.detach().item()), 0.0, places=6)
        self.assertGreater(float(value_loss.detach().item()), 0.0)
        self.assertAlmostEqual(
            float(value_metrics["wm/policy_open_loop_consistency_feature_l1_mean"]),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(value_metrics["wm/policy_open_loop_consistency_value_loss_mean"]),
            0.0,
        )
        self.assertGreater(
            float(value_metrics["wm/policy_open_loop_consistency_imag_target_gap_mean"]),
            0.0,
        )
        self.assertAlmostEqual(
            float(value_metrics["wm/policy_open_loop_consistency_value_scale"]),
            1.0,
            places=6,
        )

    def test_policy_open_loop_consistency_can_boost_high_value_states(self):
        device = torch.device("cpu")
        torch.manual_seed(0)
        base_config = TrainingConfig()
        base_config.adaptive_imag_policy_open_loop_consistency_weight = 0.5
        base_config.adaptive_imag_policy_open_loop_consistency_horizon = 2
        base_config.adaptive_imag_policy_open_loop_consistency_delta = 0.5

        boosted_config = TrainingConfig()
        boosted_config.adaptive_imag_policy_open_loop_consistency_weight = 0.5
        boosted_config.adaptive_imag_policy_open_loop_consistency_horizon = 2
        boosted_config.adaptive_imag_policy_open_loop_consistency_delta = 0.5
        boosted_config.adaptive_imag_policy_open_loop_consistency_high_value_boost = 2.0
        boosted_config.adaptive_imag_policy_open_loop_consistency_high_value_quantile = 0.5

        base_model = _SemanticModel(4).to(device)
        boosted_model = _SemanticModel(4).to(device)
        boosted_model.load_state_dict(base_model.state_dict())

        base_stepper = TrainingStep(
            model=base_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(base_model.parameters(), lr=1e-3)),
            config=base_config,
            device=device,
        )
        boosted_stepper = TrainingStep(
            model=boosted_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(boosted_model.parameters(), lr=1e-3)),
            config=boosted_config,
            device=device,
        )

        wm_batch = {
            "vitals": torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0, 0.0],
                        [4.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.5, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 0.0],
                    ],
                ],
                device=device,
            ),
            "actions": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                ],
                device=device,
            ),
            "rewards": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 8.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                device=device,
            ),
            "dones": torch.zeros((2, 4), device=device),
        }

        base_loss, base_metrics = base_stepper._compute_policy_open_loop_consistency_loss(
            wm_batch,
            base_stepper._rl_cfg(),
        )
        boosted_loss, boosted_metrics = boosted_stepper._compute_policy_open_loop_consistency_loss(
            wm_batch,
            boosted_stepper._rl_cfg(),
        )

        self.assertNotAlmostEqual(
            float(boosted_loss.detach().item()),
            float(base_loss.detach().item()),
            places=6,
        )
        self.assertGreater(
            float(boosted_metrics["wm/policy_open_loop_consistency_high_value_fraction_mean"]),
            0.0,
        )
        self.assertLess(
            float(boosted_metrics["wm/policy_open_loop_consistency_high_value_fraction_mean"]),
            1.0,
        )
        self.assertGreater(
            float(boosted_metrics["wm/policy_open_loop_consistency_high_value_weight_mean"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(boosted_metrics["wm/policy_open_loop_consistency_high_value_boost"]),
            2.0,
            places=6,
        )

    def test_policy_open_loop_consistency_late_step_boost_emphasizes_deeper_open_loop_steps(self):
        device = torch.device("cpu")
        torch.manual_seed(0)
        base_config = TrainingConfig()
        base_config.adaptive_imag_policy_open_loop_consistency_weight = 0.5
        base_config.adaptive_imag_policy_open_loop_consistency_horizon = 2
        base_config.adaptive_imag_policy_open_loop_consistency_delta = 0.5

        boosted_config = TrainingConfig()
        boosted_config.adaptive_imag_policy_open_loop_consistency_weight = 0.5
        boosted_config.adaptive_imag_policy_open_loop_consistency_horizon = 2
        boosted_config.adaptive_imag_policy_open_loop_consistency_delta = 0.5
        boosted_config.adaptive_imag_policy_open_loop_consistency_late_step_boost = 1.0

        base_model = _SemanticModel(4).to(device)
        boosted_model = _SemanticModel(4).to(device)
        boosted_model.load_state_dict(base_model.state_dict())

        base_stepper = TrainingStep(
            model=base_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(base_model.parameters(), lr=1e-3)),
            config=base_config,
            device=device,
        )
        boosted_stepper = TrainingStep(
            model=boosted_model,
            opt_bundle=_single_rl_opt_bundle(torch.optim.Adam(boosted_model.parameters(), lr=1e-3)),
            config=boosted_config,
            device=device,
        )

        wm_batch = {
            "vitals": torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0, 0.0],
                        [4.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.5, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 0.0],
                    ],
                ],
                device=device,
            ),
            "actions": torch.tensor(
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                ],
                device=device,
            ),
            "rewards": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 8.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                device=device,
            ),
            "dones": torch.zeros((2, 4), device=device),
        }

        base_loss, _ = base_stepper._compute_policy_open_loop_consistency_loss(
            wm_batch,
            base_stepper._rl_cfg(),
        )
        boosted_loss, boosted_metrics = boosted_stepper._compute_policy_open_loop_consistency_loss(
            wm_batch,
            boosted_stepper._rl_cfg(),
        )

        self.assertGreater(float(boosted_loss.detach().item()), float(base_loss.detach().item()))
        self.assertAlmostEqual(
            float(boosted_metrics["wm/policy_open_loop_consistency_late_step_boost"]),
            1.0,
            places=6,
        )
        self.assertGreater(
            float(boosted_metrics["wm/policy_open_loop_consistency_step_weight_mean"]),
            1.0,
        )
        self.assertGreater(
            float(boosted_metrics["wm/policy_open_loop_consistency_step_weight_step2"]),
            float(boosted_metrics["wm/policy_open_loop_consistency_step_weight_step1"]),
        )


if __name__ == "__main__":
    unittest.main()
