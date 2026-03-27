import copy
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
import numpy as np

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.aletheia_config import TrainingConfig
from aletheia.aletheia_train import (
    ReplayBuffer,
    RolloutCollector,
    TrainingLoop,
    _allocate_priority_budget_with_capacity,
    _apply_actor_unified_tail_relief,
    _blend_actor_unified_contract,
    _compute_bootstrap_authority_source_replacement_contract,
    _compute_bootstrap_authority_source_recertification_contract,
    _compute_bootstrap_external_authority_floor_contract,
    _compute_bootstrap_external_authority_takeover_floor_contract,
    _resolve_bootstrap_external_eval_feedback_snapshot,
    _compute_bootstrap_task_request_contract,
    _masked_mean_tensor,
    build_training_model,
)
from aletheia.contracts.authority import (
    compute_bootstrap_external_authority_floor_live_unwind_contract,
    compute_bootstrap_trigger_entry_contract,
    compute_post_transition_retention_capture,
)

from aletheia.contracts.consumers import (
    BootstrapTargetExecutionResult,
    compute_bootstrap_bonus_consumer_retention_contract,
    compute_bootstrap_external_seed_assembly_contract,
    compute_bootstrap_final_external_value_assembly_contract,
    compute_bootstrap_final_external_value_quality_contract,
    compute_bootstrap_internal_value_relief_contract,
    compute_bootstrap_midlate_external_value_ratio_cap_contract,
    compute_bootstrap_prehold_target_cap_contract,
    compute_bootstrap_target_execution_contract,
    compute_bootstrap_bonus_source_hold_persistence_contract,
    compute_bootstrap_bonus_source_seed_quality_contract,
    compute_bootstrap_bonus_terminal_truth_source_contract,
    compute_bootstrap_bonus_source_transition_bridge_contract,
    compute_bootstrap_bonus_source_value_replacement_contract,
    compute_post_transition_certified_retention_floor_contract,
)


class _NoDoneEnv:
    """Deterministic env that never emits terminated/truncated on its own."""

    def __init__(self):
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(n=2)
        self._step = 0

    def reset(self, seed=None):
        self._step = 0
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.full((4,), float(self._step), dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        return None


class _TinyPolicyValueModel(nn.Module):
    def __init__(self, obs_dim: int = 4, action_dim: int = 2):
        super().__init__()
        self.actor_head = nn.Linear(obs_dim, action_dim)
        self.critic = _FlatContractCritic(obs_dim)

    def forward(self, vitals, temperature: float = 1.0, intent=None):
        mu = self.actor_head(vitals)
        dist = D.Normal(mu, torch.ones_like(mu))
        value = self.critic(vitals)
        return dist, value


class _ContractCriticBase(nn.Module):
    gammas = (0.99,)
    primary_gamma = 0.99

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
        per_elem = F.huber_loss(pred, target, reduction="none", delta=1.0).reshape(-1)
        if weights is not None:
            per_elem = per_elem * weights.reshape(-1)[: per_elem.shape[0]]
        loss = per_elem.mean()
        return {
            "loss": loss,
            "per_gamma": {self.primary_gamma: loss.detach()},
        }


class _FlatContractCritic(_ContractCriticBase):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.v = nn.Linear(feat_dim, 1)

    @property
    def weight(self):
        return self.v.weight

    @property
    def bias(self):
        return self.v.bias

    def forward(self, feat, use_target: bool = False):
        del use_target
        flat_feat = feat.reshape(-1, feat.shape[-1])
        pred = self.v(flat_feat).squeeze(-1)
        if feat.dim() == 3:
            return pred.view(feat.shape[0], feat.shape[1])
        return pred


class _StatefulWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._wm_config = SimpleNamespace(is_discrete_action=True)

    def init_wm_state(self, batch_size, device):
        return SimpleNamespace(feature=torch.zeros(batch_size, 3, device=device))

    def forward_context(self, obs, prev_action, state):
        del state
        feat = torch.stack((obs[:, 0], obs[:, 1], prev_action[:, 0]), dim=-1)
        return SimpleNamespace(feature=feat), None


class _StatefulActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(3, 2)

    def forward(self, feat, temperature: float = 1.0, intent=None):
        del temperature, intent
        return D.OneHotCategorical(logits=self.net(feat))


class _StatefulSequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.world_model = _StatefulWorldModel()
        self.actor = _StatefulActor()
        self.critic = _FlatContractCritic(3)

    def policy_from_wm_state(self, wm_state):
        return wm_state.feature

    def forward(self, vitals, temperature: float = 1.0, intent=None):
        dist = self.actor(vitals, temperature=temperature, intent=intent)
        value = self.critic(vitals).squeeze(-1)
        return dist, value


class _TinyImagActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, feat, temperature: float = 1.0, intent=None):
        del temperature, intent
        logits = self.net(feat)
        return D.OneHotCategorical(logits=logits)


class _TinyImagModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.world_model = None
        self.actor = _TinyImagActor()
        self.critic = _FlatContractCritic(4)

    def forward(self, vitals, temperature: float = 1.0, intent=None):
        del temperature, intent
        dist = self.actor(vitals)
        value = self.critic(vitals).squeeze(-1)
        return dist, value


class _TinyImagModelWithOptimizerDrift(_TinyImagModel):
    def __init__(self):
        super().__init__()
        self.actor.extra_projection = nn.Linear(4, 4, bias=False)


class TestRealBatchMcAnchor(unittest.TestCase):
    def test_build_real_batch_uses_full_mc_returns_and_corridor_mask_when_enabled(self):
        device = torch.device("cpu")
        model = _StatefulSequenceModel().to(device)
        with torch.no_grad():
            model.critic.weight.zero_()
            model.critic.bias.zero_()

        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), -1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), -2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=0.5,
            done=True,
            log_prob=0.0,
        )
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=2.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 4.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=3.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=3,
            batch_size=1,
            seq_len=3,
            wm_seq_len=3,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            value_real_anchor_use_mc_returns=True,
            value_real_anchor_corridor_quantile=0.8,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)

        real_batch = loop._build_real_batch()
        self.assertIsNotNone(real_batch)
        self.assertTrue(bool(real_batch["use_precomputed_policy_outputs"]))

        expected = torch.tensor(
            [[
                1.0 + 0.99 * 2.0 + (0.99 ** 2) * 3.0,
                2.0 + 0.99 * 3.0,
                3.0,
            ]],
            dtype=real_batch["value_real"].dtype,
            device=device,
        )
        self.assertTrue(torch.allclose(real_batch["value_real"], expected, atol=1e-6))
        self.assertAlmostEqual(float(real_batch["value_real_is_mc"]), 1.0, places=6)
        self.assertIn("value_real_corridor_mask", real_batch)
        self.assertTrue(
            torch.equal(
                real_batch["value_real_corridor_mask"],
                torch.tensor([[1.0, 0.0, 0.0]], dtype=real_batch["value_real"].dtype, device=device),
            )
        )
        self.assertFalse(torch.allclose(real_batch["value_real"], real_batch["returns"], atol=1e-6))


class _LazyV45WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._wm_config = SimpleNamespace(
            ctrl_dim=3,
            s_pred_dim=4,
            use_projection=True,
            use_abstractor=False,
        )
        self._action_dim = 2
        self._encoder_out_dim = 4
        self._control_head = None
        self._projection = None
        self._abstractor = None

    def _ensure_v45_components(self):
        if self._control_head is None:
            self._control_head = nn.Linear(self._encoder_out_dim, self._wm_config.ctrl_dim)
        if self._projection is None and self._wm_config.use_projection:
            self._projection = nn.Linear(self._wm_config.s_pred_dim, self._encoder_out_dim)


class _FakeImaginationEngine:
    def __init__(self, device: torch.device):
        self.device = device
        self.max_horizon = 2
        self._gamma_buf = 0.99
        self.gae_lambda = 0.95

    @staticmethod
    def _compute_lambda_returns(rewards, values, continue_probs, gamma, lambda_):
        del lambda_
        if continue_probs is None:
            return rewards + gamma * values[:, 1:]
        return rewards + gamma * continue_probs * values[:, 1:]

    def imagine_rollout_differentiable(
        self,
        initial_state,
        initial_prev_action,
        horizon,
        actor,
        critic,
        initial_wm_state=None,
    ):
        del initial_prev_action, horizon, actor, critic, initial_wm_state
        batch = initial_state.shape[0]
        policy_features = torch.zeros(batch, 3, 4, device=self.device)
        actions = torch.zeros(batch, 2, 2, device=self.device)
        actions[..., 0] = 1.0
        rewards = torch.ones(batch, 2, device=self.device)
        values = torch.tensor([[2.0, 3.0, 4.0]], device=self.device).repeat(batch, 1)
        continue_probs = torch.full((batch, 2), 0.95, device=self.device)
        returns = self._compute_lambda_returns(rewards, values, continue_probs, gamma=0.99, lambda_=0.95)
        return {
            "policy_features": policy_features,
            "actions": actions,
            "log_probs": torch.zeros(batch, 2, device=self.device),
            "entropies": torch.zeros(batch, 2, device=self.device),
            "rewards": rewards,
            "values": values,
            "returns": returns,
            "continue_probs": continue_probs,
        }


class _AdaptiveCapImaginationEngine(_FakeImaginationEngine):
    def imagine_rollout_differentiable(
        self,
        initial_state,
        initial_prev_action,
        horizon,
        actor,
        critic,
        initial_wm_state=None,
    ):
        del initial_prev_action, horizon, actor, critic, initial_wm_state
        batch = initial_state.shape[0]
        policy_features = torch.zeros(batch, 3, 4, device=self.device)
        actions = torch.zeros(batch, 2, 2, device=self.device)
        actions[..., 0] = 1.0
        rewards = torch.tensor([[5.0, 5.0]], device=self.device).repeat(batch, 1)
        values = torch.tensor([[5.0, 40.0, 40.0]], device=self.device).repeat(batch, 1)
        continue_probs = torch.full((batch, 2), 0.99, device=self.device)
        returns = self._compute_lambda_returns(rewards, values, continue_probs, gamma=0.99, lambda_=0.95)
        return {
            "policy_features": policy_features,
            "actions": actions,
            "log_probs": torch.zeros(batch, 2, device=self.device),
            "entropies": torch.zeros(batch, 2, device=self.device),
            "rewards": rewards,
            "values": values,
            "returns": returns,
            "continue_probs": continue_probs,
        }


class _MidPressureImaginationEngine(_FakeImaginationEngine):
    def imagine_rollout_differentiable(
        self,
        initial_state,
        initial_prev_action,
        horizon,
        actor,
        critic,
        initial_wm_state=None,
    ):
        del initial_prev_action, horizon, actor, critic, initial_wm_state
        batch = initial_state.shape[0]
        policy_features = torch.zeros(batch, 3, 4, device=self.device)
        actions = torch.zeros(batch, 2, 2, device=self.device)
        actions[..., 0] = 1.0
        rewards = torch.ones(batch, 2, device=self.device)
        values = torch.tensor([[2.0, 3.0, 4.0]], device=self.device).repeat(batch, 1)
        continue_probs = torch.full((batch, 2), 0.985, device=self.device)
        returns = self._compute_lambda_returns(rewards, values, continue_probs, gamma=0.99, lambda_=0.95)
        return {
            "policy_features": policy_features,
            "actions": actions,
            "log_probs": torch.zeros(batch, 2, device=self.device),
            "entropies": torch.zeros(batch, 2, device=self.device),
            "rewards": rewards,
            "values": values,
            "returns": returns,
            "continue_probs": continue_probs,
        }


class _FixedImaginationEngine(_FakeImaginationEngine):
    def __init__(self, device: torch.device, *, rewards, values, continue_probs, entropies, log_probs, actions):
        super().__init__(device)
        self._rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        self._values = torch.tensor(values, device=device, dtype=torch.float32)
        self._continue_probs = torch.tensor(continue_probs, device=device, dtype=torch.float32)
        self._entropies = torch.tensor(entropies, device=device, dtype=torch.float32)
        self._log_probs = torch.tensor(log_probs, device=device, dtype=torch.float32)
        self._actions = torch.tensor(actions, device=device, dtype=torch.float32)

    def imagine_rollout_differentiable(
        self,
        initial_state,
        initial_prev_action,
        horizon,
        actor,
        critic,
        initial_wm_state=None,
    ):
        del initial_prev_action, horizon, actor, critic, initial_wm_state
        batch = initial_state.shape[0]
        policy_features = torch.zeros(batch, self._values.shape[1], 4, device=self.device)
        rewards = self._rewards.repeat(batch, 1)
        values = self._values.repeat(batch, 1)
        continue_probs = self._continue_probs.repeat(batch, 1)
        returns = self._compute_lambda_returns(rewards, values, continue_probs, gamma=0.99, lambda_=0.95)
        return {
            "policy_features": policy_features,
            "actions": self._actions.repeat(batch, 1, 1),
            "log_probs": self._log_probs.repeat(batch, 1),
            "entropies": self._entropies.repeat(batch, 1),
            "rewards": rewards,
            "values": values,
            "returns": returns,
            "continue_probs": continue_probs,
        }


class _ContractFeatureImaginationEngine(_FixedImaginationEngine):
    def __init__(
        self,
        device: torch.device,
        *,
        policy_features,
        rewards,
        values,
        continue_probs,
        entropies,
        log_probs,
        actions,
    ):
        super().__init__(
            device,
            rewards=rewards,
            values=values,
            continue_probs=continue_probs,
            entropies=entropies,
            log_probs=log_probs,
            actions=actions,
        )
        self._policy_features = torch.tensor(
            policy_features,
            device=device,
            dtype=torch.float32,
        )

    def imagine_rollout_differentiable(
        self,
        initial_state,
        initial_prev_action,
        horizon,
        actor,
        critic,
        initial_wm_state=None,
    ):
        out = super().imagine_rollout_differentiable(
            initial_state,
            initial_prev_action,
            horizon,
            actor,
            critic,
            initial_wm_state=initial_wm_state,
        )
        batch = initial_state.shape[0]
        out["policy_features"] = self._policy_features.repeat(batch, 1, 1)
        return out

class _SignedRecoveryImaginationEngine(_FakeImaginationEngine):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def imagine_rollout_differentiable(
        self,
        initial_state,
        initial_prev_action,
        horizon,
        actor,
        critic,
        initial_wm_state=None,
    ):
        del initial_prev_action, horizon, actor, critic, initial_wm_state
        batch = initial_state.shape[0]
        policy_features = torch.zeros(batch, 3, 4, device=self.device)
        actions = torch.zeros(batch, 2, 2, device=self.device)
        actions[..., 0] = 1.0
        rewards = torch.zeros(batch, 2, device=self.device)
        values = torch.tensor([[0.0, 6.0, 0.0]], device=self.device).repeat(batch, 1)
        returns = torch.tensor([[5.0, 3.0]], device=self.device).repeat(batch, 1)
        continue_probs = torch.full((batch, 2), 0.95, device=self.device)
        return {
            "policy_features": policy_features,
            "actions": actions,
            "log_probs": torch.zeros(batch, 2, device=self.device),
            "entropies": torch.zeros(batch, 2, device=self.device),
            "rewards": rewards,
            "values": values,
            "returns": returns,
            "continue_probs": continue_probs,
        }


class _LowContinueImaginationEngine(_FakeImaginationEngine):
    def imagine_rollout_differentiable(
        self,
        initial_state,
        initial_prev_action,
        horizon,
        actor,
        critic,
        initial_wm_state=None,
    ):
        del initial_prev_action, horizon, actor, critic, initial_wm_state
        batch = initial_state.shape[0]
        policy_features = torch.zeros(batch, 3, 4, device=self.device)
        actions = torch.zeros(batch, 2, 2, device=self.device)
        actions[..., 0] = 1.0
        rewards = torch.ones(batch, 2, device=self.device)
        values = torch.tensor([[4.0, 4.5, 5.0]], device=self.device).repeat(batch, 1)
        continue_probs = torch.full((batch, 2), 0.80, device=self.device)
        returns = self._compute_lambda_returns(rewards, values, continue_probs, gamma=0.99, lambda_=0.95)
        return {
            "policy_features": policy_features,
            "actions": actions,
            "log_probs": torch.zeros(batch, 2, device=self.device),
            "entropies": torch.zeros(batch, 2, device=self.device),
            "rewards": rewards,
            "values": values,
            "returns": returns,
            "continue_probs": continue_probs,
        }


def _make_actor_contract_trust_test_loop(
    device: torch.device,
    *,
    drift_actor: bool,
) -> TrainingLoop:
    model = _TinyImagModel().to(device)
    with torch.no_grad():
        model.critic.weight.zero_()
        model.critic.bias.fill_(6.0)
        model.actor.net.weight.zero_()
        model.actor.net.bias.zero_()

    buffer = ReplayBuffer(capacity=8)
    buffer.start_episode(initial_vitals=np.full((4,), 0.0, dtype=np.float32))
    buffer.add_step(
        vitals=np.full((4,), 1.0, dtype=np.float32),
        action=np.array([1.0, 0.0], dtype=np.float32),
        reward=2.0,
        done=False,
        log_prob=0.0,
    )
    buffer.add_step(
        vitals=np.full((4,), 2.0, dtype=np.float32),
        action=np.array([0.0, 1.0], dtype=np.float32),
        reward=1.0,
        done=False,
        log_prob=0.0,
    )
    buffer.add_step(
        vitals=np.full((4,), 3.0, dtype=np.float32),
        action=np.array([1.0, 0.0], dtype=np.float32),
        reward=1.0,
        done=False,
        log_prob=0.0,
    )
    buffer.add_step(
        vitals=np.full((4,), 4.0, dtype=np.float32),
        action=np.array([0.0, 1.0], dtype=np.float32),
        reward=0.0,
        done=True,
        terminated=True,
        truncated=False,
        log_prob=0.0,
    )
    buffer._select_uniform = lambda valid, batch_size, required: [(0, 0)] * batch_size

    config = TrainingConfig(
        config_mode="strict",
        validation_mode="off",
        total_steps=1,
        num_train_steps=1,
        total_env_steps=4,
        batch_size=1,
        seq_len=4,
        wm_seq_len=4,
        wm_batch_size=1,
        rl_batch_size=1,
        wm_pretrain_steps=0,
        warmup_steps=0,
        imagination_only=True,
        imagination_horizon=2,
        imagination_batch_size=1,
        adaptive_imag_idle_corridor_advantage_blend_max=0.0,
        adaptive_imag_idle_corridor_clean_target_blend_max=0.0,
        adaptive_imag_idle_corridor_quantile=0.0,
        adaptive_imag_task_corridor_enabled=True,
        adaptive_imag_task_corridor_high_quantile=0.75,
        adaptive_imag_task_corridor_low_quantile=0.25,
        adaptive_imag_task_corridor_gate_tau=0.05,
        adaptive_imag_task_corridor_analytic_floor=0.25,
        adaptive_imag_task_corridor_confidence_scale=0.25,
        adaptive_imag_actor_contract_trust_floor=0.0,
        log_interval=1000,
        eval_interval=1000,
        save_interval=1000,
    )
    config.rl.use_reward_ema = False
    config.rl.use_advantage_normalization = False
    config.rl.actor_entropy_scale = 0.0
    loop = TrainingLoop(
        model=model,
        buffer=buffer,
        config=config,
        device=device,
        env=None,
    )
    loop.imagination_engine = _ContractFeatureImaginationEngine(
        device,
        policy_features=[
            [[-1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]
        ],
        rewards=[[0.0, 0.0]],
        values=[[6.0, 6.0, 6.0]],
        continue_probs=[[1.0, 0.0]],
        entropies=[[0.0, 0.0]],
        log_probs=[[0.0, 0.0]],
        actions=[[[1.0, 0.0], [1.0, 0.0]]],
    )
    registry_actor = copy.deepcopy(model.actor).to(device)
    registry_actor.eval()
    for param in registry_actor.parameters():
        param.requires_grad_(False)
    loop._real_stability_certified_anchor_registry_actors = [registry_actor]
    loop._real_stability_certified_anchor_registry_steps = [100]
    loop._real_stability_certified_anchor_registry_evals = [500.0]
    loop._real_stability_certified_anchor_registry_telemetries = [{}]
    if drift_actor:
        with torch.no_grad():
            model.actor.net.bias.copy_(torch.tensor([3.0, -3.0], device=device))
    return loop


def _prime_critic_bootstrap_contract_loop(
    loop: TrainingLoop,
    *,
    enabled: bool,
    actor_unified_contract_enabled: bool = False,
    actor_unified_contract_blend: float = 0.5,
    semantic_debt_decay: float = 0.9,
    contact_persistence_decay: float = 0.95,
    contact_persistence_floor: float = 0.35,
    global_step: int = 2000,
    best_eval: float = 500.0,
    last_eval: float = 150.0,
    bootstrap_min_step: int = 0,
    bootstrap_step_ramp: int = 1,
    bootstrap_eval_threshold: float = 0.0,
    bootstrap_eval_ramp: float = 1.0,
) -> TrainingLoop:
    loop.config.adaptive_imag_actor_unified_contract_enabled = (
        actor_unified_contract_enabled
    )
    loop.config.adaptive_imag_actor_unified_contract_blend = float(
        actor_unified_contract_blend
    )
    loop.config.adaptive_imag_critic_bootstrap_contract_enabled = enabled
    loop.config.adaptive_imag_critic_bootstrap_clean_mix_max = 0.8
    loop.config.adaptive_imag_critic_bootstrap_semantic_debt_decay = (
        semantic_debt_decay
    )
    loop.config.adaptive_imag_critic_bootstrap_contact_persistence_decay = (
        contact_persistence_decay
    )
    loop.config.adaptive_imag_critic_bootstrap_contact_persistence_floor = (
        contact_persistence_floor
    )
    loop.config.adaptive_imag_critic_bootstrap_anchor_confidence_floor = 0.0
    loop.config.adaptive_imag_critic_bootstrap_anchor_contact_priority_bonus = 2.0
    loop.config.adaptive_imag_critic_bootstrap_dense_contact_priority_floor = 0.35
    loop.config.adaptive_imag_critic_bootstrap_anchor_contact_cap = 1.0
    loop.config.adaptive_imag_critic_bootstrap_min_step = int(bootstrap_min_step)
    loop.config.adaptive_imag_critic_bootstrap_step_ramp = int(bootstrap_step_ramp)
    loop.config.adaptive_imag_critic_bootstrap_eval_threshold = float(
        bootstrap_eval_threshold
    )
    loop.config.adaptive_imag_critic_bootstrap_eval_ramp = float(
        bootstrap_eval_ramp
    )
    loop.global_step = int(global_step)
    loop._external_eval_best_mean = float(best_eval)
    loop._external_eval_last_mean = float(last_eval)
    loop._real_stability_last_step = 1990
    loop._real_stability_last_telemetry = {
        "real_behavior_action_entropy_mean": 0.69,
        "real_behavior_action_switch_rate": 0.85,
        "real_behavior_action_oscillation_rate": 0.75,
        "real_corridor_occupancy_fraction": 1.0,
        "real_corridor_persistence_rate": 1.0,
        "real_corridor_entry_rate": 0.0,
        "real_corridor_exit_rate": 0.0,
        "real_policy_certified_anchor_available": 1.0,
        "real_policy_kl_to_certified_anchor_mean": 0.001,
        "real_policy_certified_registry_available": 1.0,
        "real_policy_certified_registry_size": 1.0,
        "real_policy_kl_to_certified_registry_mean": 0.001,
    }
    return loop

class _BadQualityLowPressureImaginationEngine(_FakeImaginationEngine):
    def imagine_rollout_differentiable(
        self,
        initial_state,
        initial_prev_action,
        horizon,
        actor,
        critic,
        initial_wm_state=None,
    ):
        del initial_prev_action, horizon, actor, critic, initial_wm_state
        batch = initial_state.shape[0]
        policy_features = torch.zeros(batch, 3, 4, device=self.device)
        actions = torch.zeros(batch, 2, 2, device=self.device)
        actions[..., 0] = 1.0
        rewards = torch.full((batch, 2), 4.0, device=self.device)
        values = torch.full((batch, 3), 2.0, device=self.device)
        continue_probs = torch.full((batch, 2), 0.985, device=self.device)
        returns = self._compute_lambda_returns(rewards, values, continue_probs, gamma=0.99, lambda_=0.95)
        return {
            "policy_features": policy_features,
            "actions": actions,
            "log_probs": torch.zeros(batch, 2, device=self.device),
            "entropies": torch.zeros(batch, 2, device=self.device),
            "rewards": rewards,
            "values": values,
            "returns": returns,
            "continue_probs": continue_probs,
        }


class TestTrainingLoopRolloutReplayIntegration(unittest.TestCase):
    def test_loop_run_finishes_episode_on_collector_max_steps(self):
        torch.manual_seed(0)
        np.random.seed(0)

        env = _NoDoneEnv()
        model = _TinyPolicyValueModel()
        buffer = ReplayBuffer(capacity=16, store_obs=False)
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=3,
            batch_size=1,
            seq_len=1,
            wm_seq_len=1,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=False,
            imagination_horizon=3,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )

        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=env,
            collect_steps_per_cycle=3,
            train_steps_per_cycle=1,
        )
        # Keep this integration test focused on real-data flow.
        loop.imagination_engine = None

        collector = RolloutCollector(
            model=loop.model,
            env=env,
            device=device,
            max_steps=2,
        )
        loop.run(num_steps=1, data_collector=collector)

        self.assertEqual(loop.global_step, 1)
        self.assertGreaterEqual(len(loop.buffer.episodes), 1)

        ep0 = loop.buffer.episodes[0]
        self.assertEqual(ep0["vitals"].shape[0], 3)
        self.assertEqual(ep0["actions"].shape[0], 2)
        self.assertTrue(np.allclose(ep0["terminated"], np.array([0.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(ep0["truncated"], np.array([0.0, 1.0], dtype=np.float32)))
        self.assertEqual(float(ep0["dones"][-1]), 1.0)

    def test_add_to_buffer_accepts_prestarted_empty_episode(self):
        torch.manual_seed(0)
        np.random.seed(0)

        env = _NoDoneEnv()
        model = _TinyPolicyValueModel()
        buffer = ReplayBuffer(capacity=16, store_obs=False)
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=1,
            wm_seq_len=1,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=False,
            imagination_horizon=2,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )

        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=env,
            collect_steps_per_cycle=2,
            train_steps_per_cycle=1,
        )
        loop.imagination_engine = None

        buffer.start_episode()
        self.assertEqual(len(buffer.current["vitals"]), 0)

        loop._add_to_buffer({
            "observations": np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
            "next_observations": np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32),
            "actions": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "rewards": np.array([0.0, 0.0], dtype=np.float32),
            "dones": np.array([0.0, 1.0], dtype=np.float32),
            "terminated": np.array([0.0, 0.0], dtype=np.float32),
            "truncated": np.array([0.0, 1.0], dtype=np.float32),
            "log_probs": np.array([0.0, 0.0], dtype=np.float32),
        })

        self.assertEqual(len(loop.buffer.episodes), 1)
        ep0 = loop.buffer.episodes[0]
        self.assertEqual(ep0["vitals"].shape[0], 3)
        self.assertTrue(np.allclose(ep0["vitals"][0], np.zeros((4,), dtype=np.float32)))
        self.assertEqual(float(ep0["truncated"][-1]), 1.0)

    def test_compute_gae_uses_stateful_policy_features_when_available(self):
        env = _NoDoneEnv()
        model = _StatefulSequenceModel()
        buffer = ReplayBuffer(capacity=8, store_obs=False)
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=False,
            imagination_horizon=2,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=env,
            collect_steps_per_cycle=2,
            train_steps_per_cycle=1,
        )
        batch = {
            "vitals": torch.tensor([[[0.0, 1.0, 0.0, 0.0], [2.0, 3.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0]]]),
            "actions": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            "rewards": torch.tensor([[1.0, 1.0]]),
            "dones": torch.tensor([[0.0, 1.0]]),
        }

        batch = loop._compute_gae_for_batch(batch)

        self.assertTrue(bool(batch["use_precomputed_policy_outputs"]))
        self.assertEqual(tuple(batch["policy_features"].shape), (1, 2, 3))
        self.assertTrue(torch.allclose(batch["policy_features"][0, 0], torch.tensor([0.0, 1.0, 1.0])))
        self.assertTrue(torch.allclose(batch["policy_features"][0, 1], torch.tensor([2.0, 3.0, 1.0])))


    def test_imagined_continue_cap_recomputes_returns_and_weights(self):
        device = torch.device("cpu")
        model = _TinyImagModel()
        buffer = ReplayBuffer(capacity=8, store_obs=False)
        buffer.start_episode(np.zeros((4,), dtype=np.float32))
        buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=0.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.8,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _FakeImaginationEngine(device)

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        expected_returns = torch.tensor([[1.0 + 0.99 * 0.8 * 3.0, 1.0 + 0.99 * 0.8 * 4.0]], device=device)
        self.assertTrue(torch.allclose(batch["returns"], expected_returns, atol=1e-6))
        expected_weights = torch.tensor([[1.0, 0.99 * 0.8]], device=device)
        self.assertTrue(torch.allclose(batch["weights_actor"], expected_weights, atol=1e-6))
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_mean_raw"]), 0.95, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_mean"]), 0.8, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_cap"]), 0.8, places=6)


    def test_persistence_tail_mismatch_repairs_actor_targets_weights_and_critic(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8, store_obs=False)
        buffer.start_episode(np.zeros((4,), dtype=np.float32))
        buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=0.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=4,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_requires_post_entry_commit=False,
            adaptive_imag_compensation_persistence_continue_threshold=0.85,
            adaptive_imag_compensation_persistence_cap=0.86,
            adaptive_imag_compensation_persistence_actor_scale=0.87,
            adaptive_imag_compensation_persistence_hold_steps=10,
            adaptive_imag_compensation_persistence_tail_window=2,
            adaptive_imag_compensation_persistence_tail_gap_threshold=5.0,
            adaptive_imag_compensation_persistence_tail_continue_threshold=0.65,
            adaptive_imag_compensation_persistence_tail_weight_threshold=0.15,
            adaptive_imag_compensation_persistence_tail_target_gap_cap=4.0,
            adaptive_imag_compensation_persistence_tail_weight_floor_ratio=0.25,
            adaptive_imag_compensation_persistence_tail_critic_boost=0.5,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FixedImaginationEngine(
            device,
            rewards=[[1.0, 1.0, 1.0, 1.0]],
            values=[[6.0, 9.0, 12.0, 12.0, 12.0]],
            continue_probs=[[0.95, 0.15, 0.05, 0.03]],
            entropies=[[0.0, 0.0, 0.0, 0.0]],
            log_probs=[[0.0, 0.0, 0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
        )
        loop.set_external_eval_feedback(180.0, step=100)
        loop.global_step = 130

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertEqual(loop.metrics["imag/compensation_phase"], "persistence")
        self.assertAlmostEqual(float(loop.metrics["imag/persistence_tail_window"]), 2.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/persistence_tail_continue_mean"]), 0.04, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/persistence_tail_mismatch_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/persistence_tail_target_repair_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/persistence_tail_weight_floor_active"]), 1.0, places=6)
        dynamic_cap = float(loop.metrics["imag/continue_prob_cap_dynamic"])
        expected_targets = torch.tensor(
            [[1.0 + 0.99 * dynamic_cap * 9.0, 2.7820, 8.0, 8.0]],
            device=device,
        )
        self.assertTrue(torch.allclose(batch["target_actor"], expected_targets, atol=1e-4))
        expected_weight_floor = (1.0 + 0.99 * dynamic_cap) / 2.0 * 0.25
        self.assertTrue(torch.allclose(batch["weights_actor"][0, -2:], torch.full((2,), expected_weight_floor, device=device), atol=1e-6))
        self.assertAlmostEqual(float(batch["persistence_tail_mismatch_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["persistence_tail_critic_multiplier"]), 1.5, places=6)

        metrics = loop.train_step(rl_batch=batch, source_tag="imag", imag_ratio=1.0)

        self.assertAlmostEqual(float(metrics["critic/persistence_tail_mismatch_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["critic/persistence_tail_mismatch_multiplier"]), 1.5, places=6)


    def test_persistence_tail_mismatch_stays_off_outside_persistence(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8, store_obs=False)
        buffer.start_episode(np.zeros((4,), dtype=np.float32))
        buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=0.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=4,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_continue_threshold=0.85,
            adaptive_imag_compensation_persistence_cap=0.86,
            adaptive_imag_compensation_persistence_actor_scale=0.87,
            adaptive_imag_compensation_persistence_hold_steps=10,
            adaptive_imag_compensation_persistence_tail_window=1,
            adaptive_imag_compensation_persistence_tail_gap_threshold=5.0,
            adaptive_imag_compensation_persistence_tail_continue_threshold=0.65,
            adaptive_imag_compensation_persistence_tail_weight_threshold=0.3,
            adaptive_imag_compensation_persistence_tail_target_gap_cap=4.0,
            adaptive_imag_compensation_persistence_tail_weight_floor_ratio=0.25,
            adaptive_imag_compensation_persistence_tail_critic_boost=0.5,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FixedImaginationEngine(
            device,
            rewards=[[1.0, 1.0, 1.0, 1.0]],
            values=[[6.0, 9.0, 12.0, 12.0, 12.0]],
            continue_probs=[[0.95, 0.15, 0.05, 0.03]],
            entropies=[[0.0, 0.0, 0.0, 0.0]],
            log_probs=[[0.0, 0.0, 0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
        )

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertNotIn(loop.metrics["imag/compensation_phase"], {"persistence", "persistence_release"})
        self.assertAlmostEqual(float(loop.metrics["imag/persistence_tail_mismatch_active"]), 0.0, places=6)
        dynamic_cap = float(loop.metrics["imag/continue_prob_cap_dynamic"])
        expected_returns = torch.tensor(
            [[1.0 + 0.99 * dynamic_cap * 9.0, 2.7820, 1.5940, 1.3564]],
            device=device,
        )
        self.assertTrue(torch.allclose(batch["target_actor"], expected_returns, atol=1e-4))
        expected_weights = torch.tensor(
            [[1.0, 0.99 * dynamic_cap, 0.99 * dynamic_cap * 0.99 * 0.15, 0.99 * dynamic_cap * 0.99 * 0.15 * 0.99 * 0.05]],
            device=device,
        )
        self.assertTrue(torch.allclose(batch["weights_actor"], expected_weights, atol=1e-6))
        self.assertAlmostEqual(float(batch["persistence_tail_mismatch_active"]), 0.0, places=6)
        self.assertAlmostEqual(float(batch["persistence_tail_critic_multiplier"]), 1.0, places=6)


    def test_real_batch_does_not_include_persistence_tail_repair_fields(self):
        device = torch.device("cpu")
        model = _StatefulSequenceModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)

        real_batch = loop._build_real_batch()

        self.assertIsNotNone(real_batch)
        self.assertNotIn("persistence_tail_mismatch_active", real_batch)
        self.assertNotIn("persistence_tail_critic_multiplier", real_batch)


    def test_adaptive_imagined_continue_cap_tightens_under_target_inflation(self):
        device = torch.device("cpu")
        model = _TinyImagModel()
        buffer = ReplayBuffer(capacity=8, store_obs=False)
        buffer.start_episode(np.zeros((4,), dtype=np.float32))
        buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=0.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _AdaptiveCapImaginationEngine(device)

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        dynamic_cap = float(loop.metrics["imag/continue_prob_cap_dynamic"])
        self.assertLess(dynamic_cap, 0.95)
        self.assertGreaterEqual(dynamic_cap, 0.80)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_mean_raw"]), 0.99, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_cap"]), dynamic_cap, places=6)
        self.assertGreater(float(loop.metrics["imag/continue_cap_pressure"]), 0.0)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_mean"]), dynamic_cap, places=6)

    def test_adaptive_imagined_continue_cap_warmup_defers_tightening(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_continue_cap_warmup_steps=200,
            adaptive_imag_continue_cap_ramp_steps=100,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _AdaptiveCapImaginationEngine(device)
        loop.global_step = 100

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_cap_dynamic"]), 0.95, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_cap"]), 0.95, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_cap_adaptive_scale"]), 0.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_mean"]), 0.95, places=6)

    def test_entry_window_lowers_continue_target_and_tightens_cap(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_continue_cap_warmup_steps=200,
            adaptive_imag_continue_cap_ramp_steps=100,
            adaptive_imag_entry_window_steps=300,
            adaptive_imag_entry_target_continue=0.90,
            adaptive_imag_entry_continue_gain_scale=2.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _MidPressureImaginationEngine(device)
        loop.global_step = 250
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(loop.metrics["imag/entry_window_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/entry_target_continue"]), 0.90, places=6)
        self.assertGreater(float(loop.metrics["imag/continue_cap_pressure"]), 0.0)
        self.assertLess(float(loop.metrics["imag/continue_prob_cap_dynamic"]), 0.95)

    def test_persistence_min_step_blocks_early_drawdown_arm(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_min_step=200,
            adaptive_imag_compensation_persistence_continue_threshold=0.85,
            adaptive_imag_compensation_persistence_eval_confirmation_count=1,
            adaptive_imag_compensation_persistence_eval_drop=20.0,
            adaptive_imag_compensation_persistence_cap=0.86,
            adaptive_imag_compensation_persistence_actor_scale=0.87,
            adaptive_imag_compensation_persistence_hold_steps=10,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _LowContinueImaginationEngine(device)

        loop.set_external_eval_feedback(170.0, step=100)
        loop.set_external_eval_feedback(140.0, step=180)
        loop.global_step = 190

        compensation_decision = loop._resolve_imag_compensation_phase(
            gap_abs=3.0,
            gap_mean=-1.0,
            commit_gap_abs=3.0,
            commit_gap_mean=-1.0,
            cont_mean=0.80,
            actor_target_raw=5.0,
            adaptive_pressure=0.0,
            adaptive_scale=0.0,
            trigger_confirmed=False,
        )

        self.assertEqual(loop._adaptive_imag_compensation_persistence_hold_until_step, -1)
        self.assertEqual(compensation_decision["stage"], "idle")
        self.assertFalse(compensation_decision["persistence_active"])

    def test_persistence_min_step_allows_drawdown_arm_after_gate(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_min_step=200,
            adaptive_imag_compensation_persistence_continue_threshold=0.85,
            adaptive_imag_compensation_persistence_eval_confirmation_count=1,
            adaptive_imag_compensation_persistence_eval_drop=20.0,
            adaptive_imag_compensation_persistence_cap=0.86,
            adaptive_imag_compensation_persistence_actor_scale=0.87,
            adaptive_imag_compensation_persistence_hold_steps=50,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _LowContinueImaginationEngine(device)

        loop.set_external_eval_feedback(170.0, step=100)
        loop.set_external_eval_feedback(140.0, step=220)
        loop.global_step = 230

        compensation_decision = loop._resolve_imag_compensation_phase(
            gap_abs=1.0,
            gap_mean=-1.0,
            commit_gap_abs=1.0,
            commit_gap_mean=-1.0,
            cont_mean=0.80,
            actor_target_raw=5.0,
            adaptive_pressure=0.0,
            adaptive_scale=0.0,
            trigger_confirmed=False,
        )

        self.assertGreater(loop._adaptive_imag_compensation_persistence_hold_until_step, 220)
        self.assertEqual(compensation_decision["stage"], "persistence")
        self.assertTrue(compensation_decision["persistence_active"])

    def test_trigger_persistence_landing_guard_is_disabled_by_default(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=1,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)

        self.assertFalse(
            loop._is_trigger_persistence_handoff_landing_guard_active(
                gap_abs=12.0,
                continue_mean=0.60,
            )
        )

    def test_persistence_stage_holds_guard_on_internal_continue_drop_after_high_eval(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_continue_threshold=0.85,
            adaptive_imag_compensation_persistence_cap=0.86,
            adaptive_imag_compensation_persistence_actor_scale=0.87,
            adaptive_imag_compensation_persistence_hold_steps=50,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _LowContinueImaginationEngine(device)
        loop.set_external_eval_feedback(180.0, step=100)
        loop.global_step = 130

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertEqual(loop.metrics["imag/compensation_phase"], "persistence")
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_cap_dynamic"]), 0.86, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/post_trigger_scale"]), 0.87, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/compensation_persistence_active"]), 1.0, places=6)

    def test_persistence_stage_holds_guard_after_eval_drawdown(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_eval_drop=20.0,
            adaptive_imag_compensation_persistence_cap=0.86,
            adaptive_imag_compensation_persistence_actor_scale=0.87,
            adaptive_imag_compensation_persistence_hold_steps=50,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FixedImaginationEngine(
            device,
            rewards=[[0.0, 0.0]],
            values=[[5.0, 5.0, 5.0]],
            continue_probs=[[0.98, 0.98]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        loop.set_external_eval_feedback(180.0, step=100)
        loop.set_external_eval_feedback(140.0, step=120)
        loop.global_step = 130

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertEqual(loop.metrics["imag/compensation_phase"], "persistence")
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_cap_dynamic"]), 0.86, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/post_trigger_scale"]), 0.87, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/compensation_persistence_active"]), 1.0, places=6)

    def test_actor_persistence_signal_reduces_scale_on_internal_drift(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_continue_threshold=0.85,
            adaptive_imag_compensation_persistence_adv_threshold=0.6,
            adaptive_imag_compensation_persistence_actor_scale=0.87,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _LowContinueImaginationEngine(device)
        loop.set_external_eval_feedback(180.0, step=100)
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(loop.metrics["actor/persistence_signal"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/post_trigger_scale"]), 0.87, places=6)

    def test_actor_persistence_relative_signal_uses_ema_drift(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_use_relative=True,
            adaptive_imag_compensation_persistence_gap_scale=1.2,
            adaptive_imag_compensation_persistence_continue_delta=0.05,
            adaptive_imag_compensation_persistence_adv_delta=0.5,
            adaptive_imag_compensation_persistence_actor_scale=0.87,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _LowContinueImaginationEngine(device)
        loop.set_external_eval_feedback(180.0, step=100)
        loop._adaptive_imag_compensation_persistence_ema_initialized = True
        loop._adaptive_imag_compensation_persistence_adv_ema = 1.2
        loop._adaptive_imag_compensation_persistence_continue_ema = 0.92
        loop._adaptive_imag_compensation_persistence_gap_ema = 0.4
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(loop.metrics["actor/persistence_signal"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/post_trigger_scale"]), 0.87, places=6)

    def test_post_solved_can_require_prior_persistence_stage(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_eval_drop=20.0,
            adaptive_imag_compensation_persistence_hold_steps=50,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_requires_persistence=True,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FakeImaginationEngine(device)
        loop.global_step = 120
        loop.set_external_eval_feedback(350.0, step=100)
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertNotEqual(loop.metrics["imag/compensation_phase"], "post_solved")

        loop.set_external_eval_feedback(180.0, step=130)
        loop.set_external_eval_feedback(140.0, step=140)
        loop.global_step = 141
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(loop.metrics["imag/compensation_phase"], "persistence")
        self.assertEqual(loop.metrics["imag/compensation_persistence_seen"], 1.0)

        loop.set_external_eval_feedback(320.0, step=145)
        loop.global_step = 146
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(loop.metrics["imag/compensation_phase"], "post_solved")

    def test_post_solved_does_not_skip_directly_from_eval_spike_when_persistence_is_required(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_eval_drop=0.0,
            adaptive_imag_compensation_persistence_hold_steps=50,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_requires_persistence=True,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FakeImaginationEngine(device)

        loop.set_external_eval_feedback(350.0, step=100)
        loop.global_step = 101
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(loop.metrics["imag/compensation_phase"], "persistence")
        self.assertEqual(loop.metrics["imag/compensation_persistence_seen"], 1.0)

    def test_post_solved_highwater_relaxes_cap_and_actor_scale(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_min_step=100,
            adaptive_imag_compensation_post_solved_requires_persistence=False,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            adaptive_imag_compensation_post_solved_highwater_eval_threshold=400.0,
            adaptive_imag_compensation_post_solved_highwater_cap=0.88,
            adaptive_imag_compensation_post_solved_highwater_actor_scale=0.92,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FakeImaginationEngine(device)

        loop.set_external_eval_feedback(350.0, step=100)
        loop.global_step = 101
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(loop.metrics["imag/compensation_phase"], "post_solved")
        self.assertEqual(loop.metrics["imag/compensation_post_solved_highwater_active"], 0.0)
        self.assertLessEqual(loop.metrics["imag/continue_prob_cap_dynamic"], 0.82 + 1e-6)
        self.assertLessEqual(loop.metrics["actor/post_trigger_scale"], 0.75 + 1e-6)

        loop.set_external_eval_feedback(450.0, step=120)
        loop.global_step = 121
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(loop.metrics["imag/compensation_phase"], "post_solved")
        self.assertEqual(loop.metrics["imag/compensation_post_solved_highwater_active"], 1.0)
        self.assertGreaterEqual(loop.metrics["imag/continue_prob_cap_dynamic"], 0.88 - 1e-6)
        self.assertGreaterEqual(loop.metrics["actor/post_trigger_scale"], 0.92 - 1e-6)
        self.assertEqual(loop.metrics["actor/post_solved_highwater_scale_active"], 1.0)

    def test_post_solved_min_step_blocks_early_switch_but_allows_later_switch(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_min_step=140,
            adaptive_imag_compensation_post_solved_requires_persistence=False,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FakeImaginationEngine(device)

        loop.set_external_eval_feedback(350.0, step=100)
        loop.global_step = 101
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertNotEqual(loop.metrics["imag/compensation_phase"], "post_solved")

        loop.set_external_eval_feedback(320.0, step=145)
        loop.global_step = 146
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(loop.metrics["imag/compensation_phase"], "post_solved")

    def test_post_solved_stage_holds_guard_after_external_eval_feedback(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _FakeImaginationEngine(device)
        loop.global_step = 120
        loop.set_external_eval_feedback(350.0, step=100)

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertEqual(loop.metrics["imag/compensation_phase"], "post_solved")
        self.assertAlmostEqual(float(loop.metrics["imag/continue_prob_cap_dynamic"]), 0.82, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/post_trigger_scale"]), 0.75, places=6)

    def test_negative_imagined_advantage_guard_downscales_actor_and_boosts_critic_in_post_solved(self):
        device = torch.device("cpu")
        base_model = _TinyImagModel().to(device)
        guarded_model = _TinyImagModel().to(device)
        guarded_model.load_state_dict(base_model.state_dict())

        def make_loop(model):
            buffer = ReplayBuffer(capacity=8)
            buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
            buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
            buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=2,
                batch_size=1,
                seq_len=4,
                wm_seq_len=4,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                imagination_batch_size=1,
                imag_continue_prob_cap=0.95,
                adaptive_imag_continue_cap=True,
                adaptive_imag_continue_cap_min=0.80,
                adaptive_imag_continue_cap_max=0.95,
                adaptive_imag_continue_cap_target_gap=4.0,
                adaptive_imag_continue_cap_target_continue=0.97,
                adaptive_imag_continue_cap_target_actor=20.0,
                adaptive_imag_continue_cap_gap_gain=0.02,
                adaptive_imag_continue_cap_continue_gain=0.5,
                adaptive_imag_continue_cap_actor_gain=0.002,
                adaptive_imag_continue_cap_ema=0.0,
                adaptive_imag_compensation_post_solved_eval_threshold=300.0,
                adaptive_imag_compensation_post_solved_cap=0.82,
                adaptive_imag_compensation_post_solved_actor_scale=0.75,
                adaptive_imag_compensation_post_solved_hold_steps=50,
                adaptive_imag_negative_adv_guard_enabled=True,
                adaptive_imag_negative_adv_guard_threshold=0.5,
                adaptive_imag_negative_adv_actor_scale=0.5,
                adaptive_imag_negative_adv_critic_boost=1.0,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
            loop.imagination_engine = _FixedImaginationEngine(
                device,
                rewards=[[1.0, 1.0]],
                values=[[5.0, 5.0, 5.0]],
                continue_probs=[[0.0, 0.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            loop.global_step = 120
            loop.set_external_eval_feedback(350.0, step=100)
            return loop

        plain_loop = make_loop(base_model)
        guarded_loop = make_loop(guarded_model)
        plain_loop.config.adaptive_imag_negative_adv_guard_enabled = False

        plain_batch = plain_loop._build_imagined_batch()
        guarded_batch = guarded_loop._build_imagined_batch()
        self.assertIsNotNone(plain_batch)
        self.assertIsNotNone(guarded_batch)
        self.assertEqual(guarded_batch["compensation_phase"], "post_solved")

        plain_metrics = plain_loop.train_step(rl_batch=plain_batch, source_tag="imag", imag_ratio=1.0)
        guarded_metrics = guarded_loop.train_step(rl_batch=guarded_batch, source_tag="imag", imag_ratio=1.0)

        self.assertAlmostEqual(float(plain_metrics["actor/negative_adv_guard_active"]), 0.0, places=6)
        self.assertAlmostEqual(float(guarded_metrics["actor/negative_adv_guard_active"]), 1.0, places=6)
        self.assertEqual(guarded_metrics["actor/negative_adv_guard_stage"], "post_solved")
        self.assertLess(abs(float(guarded_metrics["actor/objective_effective"])), abs(float(plain_metrics["actor/objective_effective"])))
        self.assertAlmostEqual(float(guarded_metrics["critic/negative_adv_guard_multiplier"]), 2.0, places=6)
        self.assertLess(float(guarded_metrics["actor/negative_adv_guard_raw_adv_mean"]), -0.5)


    def test_post_solved_stage_specific_negative_adv_guard_triggers_without_global_guard(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            adaptive_imag_negative_adv_guard_enabled=False,
            adaptive_imag_compensation_post_solved_negative_adv_threshold=0.5,
            adaptive_imag_compensation_post_solved_negative_adv_actor_scale=0.35,
            adaptive_imag_compensation_post_solved_negative_adv_critic_boost=1.5,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FixedImaginationEngine(
            device,
            rewards=[[0.0, 0.0]],
            values=[[5.0, 5.0, 5.0]],
            continue_probs=[[0.0, 0.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        loop.global_step = 120
        loop.set_external_eval_feedback(350.0, step=100)

        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(batch["compensation_phase"], "post_solved")

        metrics = loop.train_step(rl_batch=batch, source_tag="imag", imag_ratio=1.0)
        self.assertAlmostEqual(float(metrics["actor/negative_adv_guard_active"]), 1.0, places=6)
        self.assertEqual(metrics["actor/negative_adv_guard_stage"], "post_solved")
        self.assertAlmostEqual(float(metrics["actor/negative_adv_guard_scale"]), 0.35, places=6)
        self.assertAlmostEqual(float(metrics["critic/negative_adv_guard_multiplier"]), 2.5, places=6)

    def test_post_solved_actor_anchor_softens_negative_adv_guard(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            adaptive_imag_negative_adv_guard_enabled=False,
            adaptive_imag_compensation_post_solved_negative_adv_threshold=0.5,
            adaptive_imag_compensation_post_solved_negative_adv_actor_scale=0.35,
            adaptive_imag_compensation_post_solved_negative_adv_critic_boost=1.5,
            adaptive_imag_compensation_post_solved_actor_anchor_kl=0.05,
            adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_actor_scale=0.55,
            adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_critic_boost=0.5,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FixedImaginationEngine(
            device,
            rewards=[[0.0, 0.0]],
            values=[[5.0, 5.0, 5.0]],
            continue_probs=[[0.0, 0.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        loop.global_step = 120
        loop.set_external_eval_feedback(350.0, step=100)

        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        metrics = loop.train_step(rl_batch=batch, source_tag="imag", imag_ratio=1.0)
        self.assertAlmostEqual(float(metrics["actor/post_solved_anchor_guard_relax_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["actor/negative_adv_guard_scale"]), 0.55, places=6)
        self.assertAlmostEqual(float(metrics["critic/negative_adv_guard_multiplier"]), 1.5, places=6)

    def test_post_solved_negative_adv_guard_latch_holds_recovery_pressure_after_sign_flip(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            adaptive_imag_negative_adv_guard_enabled=False,
            adaptive_imag_compensation_post_solved_negative_adv_threshold=0.5,
            adaptive_imag_compensation_post_solved_negative_adv_actor_scale=0.35,
            adaptive_imag_compensation_post_solved_negative_adv_critic_boost=1.5,
            adaptive_imag_compensation_post_solved_negative_adv_latch_steps=30,
            adaptive_imag_compensation_post_solved_negative_adv_latch_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_negative_adv_latch_critic_boost=0.5,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FixedImaginationEngine(
            device,
            rewards=[[0.0, 0.0]],
            values=[[5.0, 5.0, 5.0]],
            continue_probs=[[0.0, 0.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        loop.global_step = 120
        loop.set_external_eval_feedback(350.0, step=100)

        trigger_batch = loop._build_imagined_batch()
        self.assertIsNotNone(trigger_batch)
        trigger_batch["advantages"] = torch.full_like(trigger_batch["advantages"], -1.0)
        trigger_batch["target_actor"] = torch.full_like(trigger_batch["target_actor"], 4.0)
        trigger_batch["base_actor"] = torch.full_like(trigger_batch["base_actor"], 5.0)
        trigger_metrics = loop.train_step(rl_batch=trigger_batch, source_tag="imag", imag_ratio=1.0)
        self.assertAlmostEqual(float(trigger_metrics["actor/negative_adv_guard_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(trigger_metrics["actor/negative_adv_guard_scale"]), 0.35, places=6)
        self.assertAlmostEqual(float(trigger_metrics["critic/negative_adv_guard_multiplier"]), 2.5, places=6)
        self.assertGreaterEqual(int(loop._adaptive_imag_compensation_post_solved_negative_adv_latched_until_step), 150)

        loop.global_step = 130
        latched_batch = loop._build_imagined_batch()
        self.assertIsNotNone(latched_batch)
        latched_batch["advantages"] = torch.full_like(latched_batch["advantages"], 0.25)
        latched_batch["target_actor"] = torch.full_like(latched_batch["target_actor"], 5.25)
        latched_batch["base_actor"] = torch.full_like(latched_batch["base_actor"], 5.0)
        latched_metrics = loop.train_step(rl_batch=latched_batch, source_tag="imag", imag_ratio=1.0)
        self.assertAlmostEqual(float(latched_metrics["actor/negative_adv_guard_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(latched_metrics["actor/negative_adv_guard_latched"]), 1.0, places=6)
        self.assertEqual(latched_metrics["actor/negative_adv_guard_stage"], "post_solved")
        self.assertAlmostEqual(float(latched_metrics["actor/negative_adv_guard_scale"]), 0.75, places=6)
        self.assertAlmostEqual(float(latched_metrics["critic/negative_adv_guard_multiplier"]), 1.5, places=6)
        self.assertAlmostEqual(float(latched_metrics["imag/compensation_post_solved_negative_adv_latched"]), 1.0, places=6)

        loop.global_step = 200
        released_batch = loop._build_imagined_batch()
        self.assertIsNotNone(released_batch)
        released_batch["advantages"] = torch.full_like(released_batch["advantages"], 0.25)
        released_batch["target_actor"] = torch.full_like(released_batch["target_actor"], 5.25)
        released_batch["base_actor"] = torch.full_like(released_batch["base_actor"], 5.0)
        released_metrics = loop.train_step(rl_batch=released_batch, source_tag="imag", imag_ratio=1.0)
        self.assertAlmostEqual(float(released_metrics["actor/negative_adv_guard_active"]), 0.0, places=6)
        self.assertAlmostEqual(float(released_metrics["imag/compensation_post_solved_negative_adv_latched"]), 0.0, places=6)

    def test_post_solved_actor_anchor_pull_reduces_distance_to_best_snapshot(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            adaptive_imag_compensation_post_solved_actor_anchor_pull=0.5,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.set_external_eval_feedback(350.0, step=100)
        self.assertGreaterEqual(int(loop._adaptive_imag_compensation_post_solved_actor_anchor_step), 100)

        actor = getattr(loop.model, "actor")
        with torch.no_grad():
            for param in actor.parameters():
                param.add_(1.0)

        def dist() -> float:
            total = 0.0
            for name, param in actor.named_parameters():
                anchor_param = loop._adaptive_imag_compensation_post_solved_actor_anchor_params[name]
                total += float((param.detach() - anchor_param).pow(2).sum())
            return total

        before = dist()
        applied = loop._apply_post_solved_actor_anchor_pull(0.5)
        after = dist()
        self.assertTrue(applied)
        self.assertLess(after, before)

    def test_post_solved_actor_anchor_uses_hard_pull_before_latched_pull(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            adaptive_imag_compensation_post_solved_actor_anchor_pull=0.02,
            adaptive_imag_compensation_post_solved_actor_anchor_hard_pull=0.05,
            adaptive_imag_compensation_post_solved_actor_anchor_latched_pull=0.03,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.set_external_eval_feedback(350.0, step=100)

        self.assertAlmostEqual(
            loop._resolve_post_solved_actor_anchor_pull(
                compensation_phase="post_solved",
                negative_adv_guard_active=0.0,
                negative_adv_guard_latched=0.0,
                external_eval_best_mean=0.0,
            ),
            0.02,
            places=6,
        )
        self.assertAlmostEqual(
            loop._resolve_post_solved_actor_anchor_pull(
                compensation_phase="post_solved",
                negative_adv_guard_active=1.0,
                negative_adv_guard_latched=0.0,
                external_eval_best_mean=0.0,
            ),
            0.05,
            places=6,
        )
        self.assertAlmostEqual(
            loop._resolve_post_solved_actor_anchor_pull(
                compensation_phase="post_solved",
                negative_adv_guard_active=1.0,
                negative_adv_guard_latched=1.0,
                external_eval_best_mean=0.0,
            ),
            0.03,
            places=6,
        )

    def test_post_solved_real_actor_anchor_uses_replay_policy_features(self):
        device = torch.device("cpu")
        model = _StatefulSequenceModel().to(device)
        buffer = ReplayBuffer(capacity=8, store_obs=False)
        buffer.start_episode(initial_vitals=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        buffer.add_step(
            vitals=np.array([2.0, 3.0, 0.0, 0.0], dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.array([4.0, 5.0, 0.0, 0.0], dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_real_actor_anchor_kl=0.05,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.global_step = 120
        loop.set_external_eval_feedback(350.0, step=100)
        real_batch = loop._build_real_batch()
        self.assertIsNotNone(real_batch)
        self.assertTrue(bool(real_batch["use_precomputed_policy_outputs"]))
        self.assertEqual(tuple(real_batch["policy_features"].shape), (1, 2, 3))

        imag_batch = {
            "vitals": real_batch["policy_features"].clone(),
            "policy_features": real_batch["policy_features"].clone(),
            "use_precomputed_policy_outputs": True,
            "actions": real_batch["actions"].clone(),
            "rewards": real_batch["rewards"].clone(),
            "dones": real_batch["dones"].clone(),
            "log_probs": torch.zeros((1, 2), device=device),
            "advantages": torch.ones((1, 2), device=device),
            "returns": torch.zeros((1, 2), device=device),
            "values": torch.zeros((1, 2), device=device),
            "weights_actor": torch.ones((1, 2), device=device),
            "target_actor": torch.ones((1, 2), device=device),
            "base_actor": torch.zeros((1, 2), device=device),
            "compensation_phase": "post_solved",
            "external_eval_best_mean": 350.0,
        }

        zero_metrics = loop.train_step(
            rl_batch=imag_batch,
            wm_batch=None,
            value_real_batch=real_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )
        self.assertAlmostEqual(float(zero_metrics["actor/post_solved_real_anchor_kl"]), 0.0, places=6)
        self.assertAlmostEqual(float(zero_metrics["actor/post_solved_real_anchor_kl_weight"]), 0.05, places=6)
        self.assertAlmostEqual(float(zero_metrics["actor/post_solved_real_anchor_active"]), 1.0, places=6)

        with torch.no_grad():
            torch.manual_seed(0)
            for param in loop.model.actor.parameters():
                param.add_(0.25 * torch.randn_like(param))

        drift_metrics = loop.train_step(
            rl_batch=imag_batch,
            wm_batch=None,
            value_real_batch=real_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )
        self.assertGreater(float(drift_metrics["actor/post_solved_real_anchor_kl"]), 0.0)

        real_source_metrics = loop.train_step(
            rl_batch=imag_batch,
            wm_batch=None,
            value_real_batch=real_batch,
            source_tag="real",
            imag_ratio=0.0,
        )
        self.assertAlmostEqual(float(real_source_metrics["actor/post_solved_real_anchor_kl_weight"]), 0.0, places=6)
        self.assertAlmostEqual(float(real_source_metrics["actor/post_solved_real_anchor_active"]), 0.0, places=6)

    def test_post_solved_real_advantage_correction_activates_on_imag_real_conflict(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        with torch.no_grad():
            model.critic.weight.zero_()
            model.critic.bias.zero_()
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_negative_adv_threshold=0.25,
            adaptive_imag_compensation_post_solved_real_advantage_threshold=0.0,
            adaptive_imag_compensation_post_solved_real_advantage_actor_scale=0.2,
            adaptive_imag_compensation_post_solved_real_advantage_critic_boost=1.0,
            adaptive_imag_compensation_post_solved_real_value_anchor_scale=3.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.global_step = 120
        loop.set_external_eval_feedback(350.0, step=100)
        real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.zeros((1, 2, 4), device=device),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], device=device),
            "rewards": torch.zeros((1, 2), device=device),
            "dones": torch.zeros((1, 2), device=device),
            "returns": torch.full((1, 2), 2.0, device=device),
            "value_real": torch.full((1, 2), 2.0, device=device),
            "target_actor": torch.full((1, 2), 2.0, device=device),
            "base_actor": torch.zeros((1, 2), device=device),
            "weights_actor": torch.ones((1, 2), device=device),
        }
        imag_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.zeros((1, 2, 4), device=device),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], device=device),
            "rewards": torch.zeros((1, 2), device=device),
            "dones": torch.zeros((1, 2), device=device),
            "log_probs": torch.zeros((1, 2), device=device),
            "advantages": torch.full((1, 2), -1.0, device=device),
            "returns": torch.zeros((1, 2), device=device),
            "values": torch.ones((1, 2), device=device),
            "weights_actor": torch.ones((1, 2), device=device),
            "target_actor": torch.zeros((1, 2), device=device),
            "base_actor": torch.ones((1, 2), device=device),
            "compensation_phase": "post_solved",
            "external_eval_best_mean": 350.0,
        }
        metrics = loop.train_step(
            rl_batch=imag_batch,
            wm_batch=None,
            value_real_batch=real_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )
        self.assertAlmostEqual(float(metrics["actor/post_solved_real_adv_mean"]), 2.0, places=6)
        self.assertAlmostEqual(float(metrics["actor/post_solved_real_adv_correction_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["actor/post_solved_real_adv_actor_scale"]), 0.2, places=6)
        self.assertAlmostEqual(float(metrics["critic/post_solved_real_adv_critic_multiplier"]), 2.0, places=6)
        self.assertAlmostEqual(float(metrics["critic/post_solved_real_value_anchor_scale"]), 3.0, places=6)
        self.assertAlmostEqual(float(metrics["critic/post_solved_real_adv_stats_available"]), 1.0, places=6)
        self.assertGreater(float(metrics["critic/value_real_anchor"]), 0.0)

    def test_persistence_base_return_cap_correction_scales_critic_and_real_anchor(self):
        device = torch.device("cpu")
        base_model = _TinyImagModel().to(device)
        with torch.no_grad():
            base_model.critic.weight.zero_()
            base_model.critic.bias.zero_()
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)

        def make_loop(critic_boost: float, anchor_scale: float) -> TrainingLoop:
            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=2,
                batch_size=1,
                seq_len=2,
                wm_seq_len=2,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                lambda_value_real_anchor=0.5,
                adaptive_imag_compensation_persistence_base_return_cap_critic_boost=critic_boost,
                adaptive_imag_compensation_persistence_base_return_cap_value_real_anchor_scale=anchor_scale,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            config.rl.use_value_real_anchor = True
            loop = TrainingLoop(model=_TinyImagModel().to(device), buffer=buffer, config=config, device=device, env=None)
            loop.model.load_state_dict(base_model.state_dict())
            return loop

        real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.zeros((1, 2, 4), device=device),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], device=device),
            "rewards": torch.zeros((1, 2), device=device),
            "dones": torch.zeros((1, 2), device=device),
            "returns": torch.full((1, 2), 2.0, device=device),
            "value_real": torch.full((1, 2), 2.0, device=device),
            "target_actor": torch.full((1, 2), 2.0, device=device),
            "base_actor": torch.zeros((1, 2), device=device),
            "weights_actor": torch.ones((1, 2), device=device),
        }
        imag_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.zeros((1, 2, 4), device=device),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], device=device),
            "rewards": torch.zeros((1, 2), device=device),
            "dones": torch.zeros((1, 2), device=device),
            "log_probs": torch.zeros((1, 2), device=device),
            "advantages": torch.zeros((1, 2), device=device),
            "returns": torch.zeros((1, 2), device=device),
            "values": torch.ones((1, 2), device=device),
            "weights_actor": torch.ones((1, 2), device=device),
            "target_actor": torch.zeros((1, 2), device=device),
            "base_actor": torch.zeros((1, 2), device=device),
            "compensation_phase": "persistence",
            "base_return_cap_active": 1.0,
            "base_return_cap_fraction": 0.5,
            "external_eval_best_mean": 180.0,
        }

        plain_loop = make_loop(0.0, 1.0)
        corrected_loop = make_loop(1.0, 3.0)
        plain_metrics = plain_loop.train_step(
            rl_batch=dict(imag_batch),
            wm_batch=None,
            value_real_batch=real_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )
        corrected_metrics = corrected_loop.train_step(
            rl_batch=dict(imag_batch),
            wm_batch=None,
            value_real_batch=real_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )

        self.assertAlmostEqual(float(plain_metrics["critic/base_return_cap_correction_active"]), 0.0, places=6)
        self.assertAlmostEqual(float(plain_metrics["critic/base_return_cap_critic_multiplier"]), 1.0, places=6)
        self.assertAlmostEqual(float(plain_metrics["critic/base_return_cap_value_anchor_scale"]), 1.0, places=6)
        self.assertAlmostEqual(float(corrected_metrics["critic/base_return_cap_correction_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(corrected_metrics["critic/base_return_cap_critic_multiplier"]), 1.5, places=6)
        self.assertAlmostEqual(float(corrected_metrics["critic/base_return_cap_value_anchor_scale"]), 2.0, places=6)
        self.assertGreater(float(corrected_metrics["critic/value_real_anchor"]), float(plain_metrics["critic/value_real_anchor"]))

    def test_persistence_real_advantage_correction_strengthens_critic_when_imag_support_weak(self):
        device = torch.device("cpu")
        base_model = _TinyImagModel().to(device)
        with torch.no_grad():
            base_model.critic.weight.zero_()
            base_model.critic.bias.zero_()
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)

        def make_loop(critic_boost: float, anchor_scale: float) -> TrainingLoop:
            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=2,
                batch_size=1,
                seq_len=2,
                wm_seq_len=2,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                lambda_value_real_anchor=0.5,
                adaptive_imag_compensation_persistence_real_advantage_threshold=0.0,
                adaptive_imag_compensation_persistence_real_advantage_imag_threshold=0.25,
                adaptive_imag_compensation_persistence_real_advantage_critic_boost=critic_boost,
                adaptive_imag_compensation_persistence_real_value_anchor_scale=anchor_scale,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            config.rl.use_value_real_anchor = True
            loop = TrainingLoop(model=_TinyImagModel().to(device), buffer=buffer, config=config, device=device, env=None)
            loop.model.load_state_dict(base_model.state_dict())
            return loop

        real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.zeros((1, 2, 4), device=device),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], device=device),
            "rewards": torch.zeros((1, 2), device=device),
            "dones": torch.zeros((1, 2), device=device),
            "returns": torch.full((1, 2), 2.0, device=device),
            "value_real": torch.full((1, 2), 2.0, device=device),
            "target_actor": torch.full((1, 2), 2.0, device=device),
            "base_actor": torch.zeros((1, 2), device=device),
            "weights_actor": torch.ones((1, 2), device=device),
        }
        imag_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.zeros((1, 2, 4), device=device),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], device=device),
            "rewards": torch.zeros((1, 2), device=device),
            "dones": torch.zeros((1, 2), device=device),
            "log_probs": torch.zeros((1, 2), device=device),
            "advantages": torch.full((1, 2), -1.0, device=device),
            "returns": torch.zeros((1, 2), device=device),
            "values": torch.ones((1, 2), device=device),
            "weights_actor": torch.ones((1, 2), device=device),
            "target_actor": torch.zeros((1, 2), device=device),
            "base_actor": torch.ones((1, 2), device=device),
            "compensation_phase": "persistence",
            "external_eval_best_mean": 220.0,
            "prebuild_real_adv_mean": 0.1,
            "prebuild_real_adv_available": 1.0,
        }

        plain_loop = make_loop(0.0, 1.0)
        corrected_loop = make_loop(1.0, 3.0)
        plain_metrics = plain_loop.train_step(
            rl_batch=dict(imag_batch),
            wm_batch=None,
            value_real_batch=real_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )
        corrected_metrics = corrected_loop.train_step(
            rl_batch=dict(imag_batch),
            wm_batch=None,
            value_real_batch=real_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )

        self.assertAlmostEqual(float(plain_metrics["actor/persistence_real_adv_correction_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(plain_metrics["critic/persistence_real_adv_critic_multiplier"]), 1.0, places=6)
        self.assertAlmostEqual(float(plain_metrics["critic/persistence_real_value_anchor_scale"]), 1.0, places=6)
        self.assertAlmostEqual(float(corrected_metrics["actor/persistence_real_adv_mean"]), 2.0, places=6)
        self.assertAlmostEqual(float(corrected_metrics["actor/persistence_real_adv_correction_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(corrected_metrics["critic/persistence_real_adv_critic_multiplier"]), 2.0, places=6)
        self.assertAlmostEqual(float(corrected_metrics["critic/persistence_real_value_anchor_scale"]), 3.0, places=6)
        self.assertGreater(float(corrected_metrics["critic/value_real_anchor"]), float(plain_metrics["critic/value_real_anchor"]))

    def test_post_solved_actor_base_return_cap_blocks_false_negative_imagined_advantage(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        with torch.no_grad():
            model.critic.weight.zero_()
            model.critic.bias.fill_(5.0)

        def make_loop(return_cap_margin: float) -> TrainingLoop:
            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=2,
                batch_size=1,
                seq_len=4,
                wm_seq_len=4,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                imagination_batch_size=1,
                imag_continue_prob_cap=0.95,
                adaptive_imag_continue_cap=True,
                adaptive_imag_continue_cap_min=0.80,
                adaptive_imag_continue_cap_max=0.95,
                adaptive_imag_continue_cap_target_gap=4.0,
                adaptive_imag_continue_cap_target_continue=0.97,
                adaptive_imag_continue_cap_target_actor=20.0,
                adaptive_imag_continue_cap_gap_gain=0.02,
                adaptive_imag_continue_cap_continue_gain=0.5,
                adaptive_imag_continue_cap_actor_gain=0.002,
                adaptive_imag_continue_cap_ema=0.0,
                adaptive_imag_compensation_post_solved_eval_threshold=300.0,
                adaptive_imag_compensation_post_solved_cap=0.82,
                adaptive_imag_compensation_post_solved_actor_scale=0.75,
                adaptive_imag_compensation_post_solved_hold_steps=50,
                adaptive_imag_compensation_post_solved_actor_base_return_cap_margin=return_cap_margin,
                adaptive_imag_negative_adv_guard_enabled=True,
                adaptive_imag_negative_adv_guard_threshold=0.5,
                adaptive_imag_negative_adv_actor_scale=0.5,
                adaptive_imag_negative_adv_critic_boost=1.0,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            loop = TrainingLoop(model=_TinyImagModel().to(device), buffer=buffer, config=config, device=device, env=None)
            loop.model.load_state_dict(model.state_dict())
            loop.imagination_engine = _FixedImaginationEngine(
                device,
                rewards=[[0.0, 0.0]],
                values=[[5.0, 5.0, 5.0]],
                continue_probs=[[0.0, 0.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            loop.global_step = 120
            loop.set_external_eval_feedback(350.0, step=100)
            return loop

        plain_loop = make_loop(-1.0)
        capped_loop = make_loop(0.0)

        plain_batch = plain_loop._build_imagined_batch()
        capped_batch = capped_loop._build_imagined_batch()
        self.assertIsNotNone(plain_batch)
        self.assertIsNotNone(capped_batch)
        self.assertEqual(capped_batch["compensation_phase"], "post_solved")

        self.assertAlmostEqual(float(plain_loop.metrics["actor/base_mean"]), 5.0, places=6)
        self.assertAlmostEqual(float(capped_loop.metrics["actor/base_mean"]), 0.0, places=6)
        self.assertAlmostEqual(float(capped_loop.metrics["actor/base_return_cap_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(capped_loop.metrics["actor/base_return_cap_fraction"]), 1.0, places=6)
        self.assertEqual(capped_loop.metrics["actor/base_source"], "return_cap")

        plain_metrics = plain_loop.train_step(rl_batch=plain_batch, source_tag="imag", imag_ratio=1.0)
        capped_metrics = capped_loop.train_step(rl_batch=capped_batch, source_tag="imag", imag_ratio=1.0)

        self.assertAlmostEqual(float(plain_metrics["actor/negative_adv_guard_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(capped_metrics["actor/negative_adv_guard_active"]), 0.0, places=6)
        self.assertLess(float(plain_metrics["actor/negative_adv_guard_raw_adv_mean"]), -0.5)
        self.assertAlmostEqual(float(capped_metrics["actor/negative_adv_guard_raw_adv_mean"]), 0.0, places=6)

    def test_post_solved_target_critic_actor_base_clears_false_negative_imagined_advantage(self):
        device = torch.device("cpu")

        class _TargetShiftCritic(_ContractCriticBase):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(5.0))

            def forward(self, feat, use_target: bool = False):
                if feat.dim() == 3:
                    out = torch.zeros(feat.shape[:2], device=feat.device) + self.bias
                else:
                    out = torch.zeros((feat.shape[0],), device=feat.device) + self.bias
                if use_target:
                    out = out - 6.0
                return out

        class _TargetBaseImagModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.world_model = None
                self.actor = _TinyImagActor()
                self.critic = _TargetShiftCritic()

            def forward(self, vitals, temperature: float = 1.0, intent=None):
                del temperature, intent
                dist = self.actor(vitals)
                value = self.critic(vitals)
                return dist, value

        def make_loop(target_base_blend: float) -> TrainingLoop:
            buffer = ReplayBuffer(capacity=8)
            buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
            buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
            buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=2,
                batch_size=1,
                seq_len=4,
                wm_seq_len=4,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                imagination_batch_size=1,
                imag_continue_prob_cap=0.95,
                adaptive_imag_continue_cap=True,
                adaptive_imag_continue_cap_min=0.80,
                adaptive_imag_continue_cap_max=0.95,
                adaptive_imag_continue_cap_target_gap=4.0,
                adaptive_imag_continue_cap_target_continue=0.97,
                adaptive_imag_continue_cap_target_actor=20.0,
                adaptive_imag_continue_cap_gap_gain=0.02,
                adaptive_imag_continue_cap_continue_gain=0.5,
                adaptive_imag_continue_cap_actor_gain=0.002,
                adaptive_imag_continue_cap_ema=0.0,
                adaptive_imag_compensation_post_solved_eval_threshold=300.0,
                adaptive_imag_compensation_post_solved_cap=0.82,
                adaptive_imag_compensation_post_solved_actor_scale=0.75,
                adaptive_imag_compensation_post_solved_hold_steps=50,
                adaptive_imag_compensation_post_solved_actor_target_base_blend=target_base_blend,
                adaptive_imag_negative_adv_guard_enabled=True,
                adaptive_imag_negative_adv_guard_threshold=0.5,
                adaptive_imag_negative_adv_actor_scale=0.5,
                adaptive_imag_negative_adv_critic_boost=1.0,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            loop = TrainingLoop(model=_TargetBaseImagModel().to(device), buffer=buffer, config=config, device=device, env=None)
            loop.imagination_engine = _FixedImaginationEngine(
                device,
                rewards=[[0.0, 0.0]],
                values=[[5.0, 5.0, 5.0]],
                continue_probs=[[0.0, 0.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            loop.global_step = 120
            loop.set_external_eval_feedback(350.0, step=100)
            return loop

        plain_loop = make_loop(0.0)
        target_loop = make_loop(1.0)

        plain_batch = plain_loop._build_imagined_batch()
        target_batch = target_loop._build_imagined_batch()
        self.assertIsNotNone(plain_batch)
        self.assertIsNotNone(target_batch)
        self.assertEqual(target_batch["compensation_phase"], "post_solved")

        self.assertAlmostEqual(float(plain_loop.metrics["actor/online_adv_mean"]), -5.0, places=6)
        self.assertAlmostEqual(float(target_loop.metrics["actor/online_adv_mean"]), -5.0, places=6)
        self.assertAlmostEqual(float(target_loop.metrics["actor/base_mean"]), -1.0, places=6)
        self.assertAlmostEqual(float(target_loop.metrics["imag/target_value_mean"]), -1.0, places=6)
        self.assertAlmostEqual(float(target_loop.metrics["imag/online_target_value_gap_mean"]), 6.0, places=6)
        self.assertAlmostEqual(float(target_loop.metrics["imag/target_value_target_gap_mean"]), -1.0, places=6)
        self.assertEqual(target_loop.metrics["actor/base_source"], "target")

        plain_metrics = plain_loop.train_step(rl_batch=plain_batch, source_tag="imag", imag_ratio=1.0)
        target_metrics = target_loop.train_step(rl_batch=target_batch, source_tag="imag", imag_ratio=1.0)

        self.assertAlmostEqual(float(plain_metrics["actor/negative_adv_guard_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(target_metrics["actor/negative_adv_guard_active"]), 0.0, places=6)
        self.assertLess(float(plain_metrics["actor/negative_adv_guard_raw_adv_mean"]), -0.5)
        self.assertGreater(float(target_metrics["actor/negative_adv_guard_raw_adv_mean"]), 0.5)

    def test_global_target_critic_actor_base_repairs_idle_false_negative_imagined_advantage(self):
        device = torch.device("cpu")

        class _TargetShiftCritic(_ContractCriticBase):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(5.0))

            def forward(self, feat, use_target: bool = False):
                if feat.dim() == 3:
                    out = torch.zeros(feat.shape[:2], device=feat.device) + self.bias
                else:
                    out = torch.zeros((feat.shape[0],), device=feat.device) + self.bias
                if use_target:
                    out = out - 6.0
                return out

        class _TargetBaseImagModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.world_model = None
                self.actor = _TinyImagActor()
                self.critic = _TargetShiftCritic()

            def forward(self, vitals, temperature: float = 1.0, intent=None):
                del temperature, intent
                dist = self.actor(vitals)
                value = self.critic(vitals)
                return dist, value

        def make_loop(target_base_blend: float) -> TrainingLoop:
            buffer = ReplayBuffer(capacity=8)
            buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
            buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
            buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=2,
                batch_size=1,
                seq_len=4,
                wm_seq_len=4,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                imagination_batch_size=1,
                imag_continue_prob_cap=0.95,
                adaptive_imag_continue_cap=True,
                adaptive_imag_continue_cap_min=0.80,
                adaptive_imag_continue_cap_max=0.95,
                adaptive_imag_continue_cap_target_gap=4.0,
                adaptive_imag_continue_cap_target_continue=0.97,
                adaptive_imag_continue_cap_target_actor=20.0,
                adaptive_imag_continue_cap_gap_gain=0.02,
                adaptive_imag_continue_cap_continue_gain=0.5,
                adaptive_imag_continue_cap_actor_gain=0.002,
                adaptive_imag_continue_cap_ema=0.0,
                adaptive_imag_global_target_base_blend=target_base_blend,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            loop = TrainingLoop(model=_TargetBaseImagModel().to(device), buffer=buffer, config=config, device=device, env=None)
            loop.imagination_engine = _FixedImaginationEngine(
                device,
                rewards=[[0.0, 0.0]],
                values=[[5.0, 5.0, 5.0]],
                continue_probs=[[0.0, 0.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            loop.global_step = 120
            original_resolve = loop._resolve_imag_continue_cap

            def _force_idle(continue_probs_im_raw, values_im, returns_raw):
                cap, info = original_resolve(continue_probs_im_raw, values_im, returns_raw)
                info = dict(info)
                info["stage"] = "idle"
                return cap, info

            loop._resolve_imag_continue_cap = _force_idle
            return loop

        plain_loop = make_loop(0.0)
        target_loop = make_loop(1.0)

        plain_batch = plain_loop._build_imagined_batch()
        target_batch = target_loop._build_imagined_batch()
        self.assertIsNotNone(plain_batch)
        self.assertIsNotNone(target_batch)
        self.assertEqual(target_batch["compensation_phase"], "idle")

        self.assertAlmostEqual(float(plain_loop.metrics["actor/base_mean"]), 5.0, places=6)
        self.assertAlmostEqual(float(target_loop.metrics["actor/base_mean"]), -1.0, places=6)
        self.assertAlmostEqual(float(target_loop.metrics["imag/target_value_mean"]), -1.0, places=6)
        self.assertEqual(target_loop.metrics["actor/base_source"], "target")
        self.assertLess(float(plain_batch["advantages"].mean()), -0.5)
        self.assertGreater(float(target_batch["advantages"].mean()), 0.5)

    def test_actor_use_target_value_ruler_unifies_bootstrap_and_base(self):
        device = torch.device("cpu")

        class _ConstImagActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(2))

            def forward(self, feat, temperature: float = 1.0, intent=None):
                del temperature, intent
                logits = self.logits.view(*((1,) * (feat.dim() - 1)), -1).expand(*feat.shape[:-1], -1)
                return D.OneHotCategorical(logits=logits)

        class _TargetShiftCritic(_ContractCriticBase):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(5.0))

            def forward(self, feat, use_target: bool = False):
                if feat.dim() == 3:
                    out = torch.zeros(feat.shape[:2], device=feat.device) + self.bias
                else:
                    out = torch.zeros((feat.shape[0],), device=feat.device) + self.bias
                if use_target:
                    out = out - 6.0
                return out

        class _TargetRulerImagModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.world_model = None
                self.actor = _ConstImagActor()
                self.critic = _TargetShiftCritic()

            def forward(self, vitals, temperature: float = 1.0, intent=None):
                del temperature, intent
                dist = self.actor(vitals)
                value = self.critic(vitals)
                return dist, value

        def make_loop(distill_weight: float = 0.0) -> TrainingLoop:
            buffer = ReplayBuffer(capacity=8)
            buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
            buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
            buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=2,
                batch_size=1,
                seq_len=4,
                wm_seq_len=4,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                imagination_batch_size=1,
                imag_continue_prob_cap=1.0,
                adaptive_imag_continue_cap=False,
                adaptive_imag_actor_use_target_value_ruler_enabled=True,
                adaptive_imag_actor_use_target_value_ruler_blend=1.0,
                adaptive_imag_actor_use_target_value_ruler_critic_distill_weight=distill_weight,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            loop = TrainingLoop(
                model=_TargetRulerImagModel().to(device),
                buffer=buffer,
                config=config,
                device=device,
                env=None,
            )
            loop.imagination_engine = _FixedImaginationEngine(
                device,
                rewards=[[0.0, 0.0]],
                values=[[5.0, 5.0, 5.0]],
                continue_probs=[[1.0, 1.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            loop.global_step = 120

            def _force_idle(continue_probs_im_raw, values_im, returns_raw):
                del values_im, returns_raw
                return 1.0, {"stage": "idle"}

            loop._resolve_imag_continue_cap = _force_idle
            return loop

        loop = make_loop()
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)

        self.assertAlmostEqual(float(batch["returns"].mean()), -0.99, places=6)
        self.assertAlmostEqual(float(batch["base_actor"].mean()), -1.0, places=6)
        self.assertGreater(float(batch["advantages"].mean()), 0.0)
        self.assertLess(float(batch["online_advantages"].mean()), -5.0)
        self.assertAlmostEqual(float(batch["reference_value_ruler_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["reference_value_ruler_blend"]), 1.0, places=6)
        self.assertEqual(loop.metrics["actor/base_source"], "target_ruler")
        self.assertAlmostEqual(float(loop.metrics["imag/reference_value_ruler_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/reference_value_ruler_blend"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/reference_value_mean"]), -1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/reference_return_mean"]), -0.99, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/reference_online_value_gap_abs_mean"]), 6.0, places=6)

    def test_actor_use_target_value_ruler_critic_distill_penalty_activates(self):
        device = torch.device("cpu")

        class _ConstImagActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(2))

            def forward(self, feat, temperature: float = 1.0, intent=None):
                del temperature, intent
                logits = self.logits.view(*((1,) * (feat.dim() - 1)), -1).expand(*feat.shape[:-1], -1)
                return D.OneHotCategorical(logits=logits)

        class _TargetShiftCritic(_ContractCriticBase):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(5.0))

            def forward(self, feat, use_target: bool = False):
                if feat.dim() == 3:
                    out = torch.zeros(feat.shape[:2], device=feat.device) + self.bias
                else:
                    out = torch.zeros((feat.shape[0],), device=feat.device) + self.bias
                if use_target:
                    out = out - 6.0
                return out

        class _TargetRulerImagModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.world_model = None
                self.actor = _ConstImagActor()
                self.critic = _TargetShiftCritic()

            def forward(self, vitals, temperature: float = 1.0, intent=None):
                del temperature, intent
                dist = self.actor(vitals)
                value = self.critic(vitals)
                return dist, value

        def make_loop(distill_weight: float) -> TrainingLoop:
            buffer = ReplayBuffer(capacity=8)
            buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
            buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
            buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=2,
                batch_size=1,
                seq_len=4,
                wm_seq_len=4,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                imagination_batch_size=1,
                imag_continue_prob_cap=1.0,
                adaptive_imag_continue_cap=False,
                adaptive_imag_actor_use_target_value_ruler_enabled=True,
                adaptive_imag_actor_use_target_value_ruler_blend=1.0,
                adaptive_imag_actor_use_target_value_ruler_critic_distill_weight=distill_weight,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            loop = TrainingLoop(
                model=_TargetRulerImagModel().to(device),
                buffer=buffer,
                config=config,
                device=device,
                env=None,
            )
            loop.imagination_engine = _FixedImaginationEngine(
                device,
                rewards=[[0.0, 0.0]],
                values=[[5.0, 5.0, 5.0]],
                continue_probs=[[1.0, 1.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            loop.global_step = 120

            def _force_idle(continue_probs_im_raw, values_im, returns_raw):
                del values_im, returns_raw
                return 1.0, {"stage": "idle"}

            loop._resolve_imag_continue_cap = _force_idle
            return loop

        zero_loop = make_loop(0.0)
        distill_loop = make_loop(1.5)

        zero_batch = zero_loop._build_imagined_batch()
        distill_batch = distill_loop._build_imagined_batch()
        self.assertIsNotNone(zero_batch)
        self.assertIsNotNone(distill_batch)

        zero_metrics = zero_loop.train_step(rl_batch=zero_batch, source_tag="imag", imag_ratio=1.0)
        distill_metrics = distill_loop.train_step(rl_batch=distill_batch, source_tag="imag", imag_ratio=1.0)

        self.assertAlmostEqual(float(zero_metrics["critic/target_ruler_distill_active"]), 0.0, places=6)
        self.assertAlmostEqual(float(zero_metrics["critic/target_ruler_distill"]), 0.0, places=6)
        self.assertAlmostEqual(float(distill_metrics["critic/target_ruler_distill_active"]), 1.0, places=6)
        self.assertGreater(float(distill_metrics["critic/target_ruler_distill"]), 0.0)
        self.assertGreater(float(distill_metrics["loss_total"]), float(zero_metrics["loss_total"]))

    def test_actor_use_target_value_ruler_soft_gate_preserves_low_gap_states(self):
        device = torch.device("cpu")

        class _ConstImagActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(2))

            def forward(self, feat, temperature: float = 1.0, intent=None):
                del temperature, intent
                logits = self.logits.view(*((1,) * (feat.dim() - 1)), -1).expand(*feat.shape[:-1], -1)
                return D.OneHotCategorical(logits=logits)

        class _FeatureSplitCritic(_ContractCriticBase):
            def forward(self, feat, use_target: bool = False):
                source = feat[..., 1] if use_target else feat[..., 0]
                if feat.dim() == 2:
                    return source
                return source

        class _FeatureRulerImagModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.world_model = None
                self.actor = _ConstImagActor()
                self.critic = _FeatureSplitCritic()

            def forward(self, vitals, temperature: float = 1.0, intent=None):
                del temperature, intent
                dist = self.actor(vitals)
                value = self.critic(vitals)
                return dist, value

        class _FeatureImaginationEngine(_FixedImaginationEngine):
            def __init__(self, device: torch.device, *, policy_features, rewards, values, continue_probs, entropies, log_probs, actions):
                super().__init__(
                    device,
                    rewards=rewards,
                    values=values,
                    continue_probs=continue_probs,
                    entropies=entropies,
                    log_probs=log_probs,
                    actions=actions,
                )
                self._policy_features = torch.tensor(policy_features, device=device, dtype=torch.float32)

            def imagine_rollout_differentiable(
                self,
                initial_state,
                initial_prev_action,
                horizon,
                actor,
                critic,
                initial_wm_state=None,
            ):
                out = super().imagine_rollout_differentiable(
                    initial_state,
                    initial_prev_action,
                    horizon,
                    actor,
                    critic,
                    initial_wm_state=initial_wm_state,
                )
                batch = initial_state.shape[0]
                out["policy_features"] = self._policy_features.repeat(batch, 1, 1)
                return out

        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=1.0,
            adaptive_imag_continue_cap=False,
            adaptive_imag_actor_use_target_value_ruler_enabled=True,
            adaptive_imag_actor_use_target_value_ruler_blend=1.0,
            adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled=True,
            adaptive_imag_actor_use_target_value_ruler_gap_margin=0.5,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=_FeatureRulerImagModel().to(device),
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _FeatureImaginationEngine(
            device,
            policy_features=[[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [5.0, -1.0, 0.0, 0.0]]],
            rewards=[[0.0, 0.0]],
            values=[[1.0, 1.0, 5.0]],
            continue_probs=[[1.0, 1.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        loop.global_step = 120

        def _force_idle(continue_probs_im_raw, values_im, returns_raw):
            del values_im, returns_raw
            return 1.0, {"stage": "idle"}

        loop._resolve_imag_continue_cap = _force_idle

        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)

        expected_alpha = (5.94 - 0.5) / (5.94 - 0.5 + 1.0)
        expected_return_step1 = (1.0 - expected_alpha) * 4.95 + expected_alpha * (-0.99)

        self.assertAlmostEqual(float(batch["returns"][0, 0]), 0.99, places=6)
        self.assertAlmostEqual(float(batch["returns"][0, 1]), expected_return_step1, places=5)
        self.assertAlmostEqual(float(batch["base_actor"][0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["base_actor"][0, 1]), 1.0, places=6)
        self.assertEqual(loop.metrics["actor/base_source"], "target_ruler_soft_gate")
        self.assertAlmostEqual(float(batch["reference_value_ruler_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/reference_value_ruler_alpha_active_fraction"]), 0.5, places=6)
        self.assertGreater(float(loop.metrics["imag/reference_value_ruler_alpha_mean"]), 0.4)
        self.assertLess(float(loop.metrics["imag/reference_value_ruler_alpha_mean"]), 0.5)

    def test_actor_use_target_value_ruler_soft_gate_reuses_local_alpha_for_distill(self):
        device = torch.device("cpu")

        class _ConstImagActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(2))

            def forward(self, feat, temperature: float = 1.0, intent=None):
                del temperature, intent
                logits = self.logits.view(*((1,) * (feat.dim() - 1)), -1).expand(*feat.shape[:-1], -1)
                return D.OneHotCategorical(logits=logits)

        class _FeatureSplitCritic(_ContractCriticBase):
            def forward(self, feat, use_target: bool = False):
                source = feat[..., 1] if use_target else feat[..., 0]
                return source

        class _FeatureRulerImagModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.world_model = None
                self.actor = _ConstImagActor()
                self.critic = _FeatureSplitCritic()

            def forward(self, vitals, temperature: float = 1.0, intent=None):
                del temperature, intent
                dist = self.actor(vitals)
                value = self.critic(vitals)
                return dist, value

        class _FeatureImaginationEngine(_FixedImaginationEngine):
            def __init__(self, device: torch.device, *, policy_features, rewards, values, continue_probs, entropies, log_probs, actions):
                super().__init__(
                    device,
                    rewards=rewards,
                    values=values,
                    continue_probs=continue_probs,
                    entropies=entropies,
                    log_probs=log_probs,
                    actions=actions,
                )
                self._policy_features = torch.tensor(policy_features, device=device, dtype=torch.float32)

            def imagine_rollout_differentiable(
                self,
                initial_state,
                initial_prev_action,
                horizon,
                actor,
                critic,
                initial_wm_state=None,
            ):
                out = super().imagine_rollout_differentiable(
                    initial_state,
                    initial_prev_action,
                    horizon,
                    actor,
                    critic,
                    initial_wm_state=initial_wm_state,
                )
                batch = initial_state.shape[0]
                out["policy_features"] = self._policy_features.repeat(batch, 1, 1)
                return out

        def make_loop(distill_weight: float) -> TrainingLoop:
            buffer = ReplayBuffer(capacity=8)
            buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
            buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
            buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=2,
                batch_size=1,
                seq_len=4,
                wm_seq_len=4,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                imagination_batch_size=1,
                imag_continue_prob_cap=1.0,
                adaptive_imag_continue_cap=False,
                adaptive_imag_actor_use_target_value_ruler_enabled=True,
                adaptive_imag_actor_use_target_value_ruler_blend=1.0,
                adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled=True,
                adaptive_imag_actor_use_target_value_ruler_gap_margin=0.5,
                adaptive_imag_actor_use_target_value_ruler_critic_distill_weight=distill_weight,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            loop = TrainingLoop(
                model=_FeatureRulerImagModel().to(device),
                buffer=buffer,
                config=config,
                device=device,
                env=None,
            )
            loop.imagination_engine = _FeatureImaginationEngine(
                device,
                policy_features=[[[5.0, -1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]],
                rewards=[[0.0, 0.0]],
                values=[[5.0, 1.0, 1.0]],
                continue_probs=[[1.0, 1.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            loop.global_step = 120

            def _force_idle(continue_probs_im_raw, values_im, returns_raw):
                del values_im, returns_raw
                return 1.0, {"stage": "idle"}

            loop._resolve_imag_continue_cap = _force_idle
            return loop

        zero_loop = make_loop(0.0)
        distill_loop = make_loop(1.0)

        zero_batch = zero_loop._build_imagined_batch()
        distill_batch = distill_loop._build_imagined_batch()
        self.assertIsNotNone(zero_batch)
        self.assertIsNotNone(distill_batch)

        zero_metrics = zero_loop.train_step(rl_batch=zero_batch, source_tag="imag", imag_ratio=1.0)
        distill_metrics = distill_loop.train_step(rl_batch=distill_batch, source_tag="imag", imag_ratio=1.0)

        self.assertAlmostEqual(float(distill_loop.metrics["imag/reference_value_ruler_alpha_active_fraction"]), 0.5, places=6)
        self.assertAlmostEqual(float(zero_metrics["critic/target_ruler_distill_active"]), 0.0, places=6)
        self.assertAlmostEqual(float(distill_metrics["critic/target_ruler_distill_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(distill_metrics["critic/target_ruler_distill_alpha_active_fraction"]), 0.5, places=6)
        self.assertGreater(float(distill_metrics["critic/target_ruler_distill_alpha_mean"]), 0.4)
        self.assertLess(float(distill_metrics["critic/target_ruler_distill_alpha_mean"]), 0.5)
        self.assertGreater(float(distill_metrics["critic/target_ruler_distill"]), 0.0)

    def test_build_imagined_batch_prebuild_real_advantage_veto_releases_latch(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.83,
            adaptive_imag_compensation_post_solved_actor_scale=0.8,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            adaptive_imag_compensation_post_solved_real_advantage_latch_veto=True,
            adaptive_imag_compensation_post_solved_real_advantage_latch_veto_threshold=0.5,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FixedImaginationEngine(
            device,
            rewards=[[0.0, 0.0]],
            values=[[5.0, 5.0, 5.0]],
            continue_probs=[[0.95, 0.95]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        loop.global_step = 1300
        loop._external_eval_best_mean = 450.6
        loop._external_eval_last_mean = 450.6
        loop._adaptive_imag_compensation_phase = "post_solved"
        loop._adaptive_imag_compensation_post_solved_hold_until_step = 1400
        loop._adaptive_imag_compensation_post_solved_negative_adv_latched_until_step = 1500
        loop.metrics = {}
        real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.zeros((1, 2, 4), device=device),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], device=device),
            "returns": torch.full((1, 2), 2.0, device=device),
            "value_real": torch.full((1, 2), 2.0, device=device),
        }
        with torch.no_grad():
            model.critic.weight.zero_()
            model.critic.bias.zero_()
        batch = loop._build_imagined_batch(reference_real_batch=real_batch)
        self.assertIsNotNone(batch)
        self.assertLess(int(getattr(loop, "_adaptive_imag_compensation_post_solved_negative_adv_latched_until_step", -1)), loop.global_step)
        self.assertAlmostEqual(float(loop.metrics["imag/prebuild_real_adv_latch_veto_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/prebuild_real_adv_mean"]), 2.0, places=6)

    def test_post_solved_real_advantage_can_veto_negative_adv_latch(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        with torch.no_grad():
            model.critic.weight.zero_()
            model.critic.bias.zero_()
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_negative_adv_threshold=0.25,
            adaptive_imag_compensation_post_solved_negative_adv_latch_steps=50,
            adaptive_imag_compensation_post_solved_real_advantage_latch_veto=True,
            adaptive_imag_compensation_post_solved_real_advantage_latch_veto_threshold=0.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.global_step = 120
        loop.set_external_eval_feedback(350.0, step=100)
        real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.zeros((1, 2, 4), device=device),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], device=device),
            "rewards": torch.zeros((1, 2), device=device),
            "dones": torch.zeros((1, 2), device=device),
            "returns": torch.full((1, 2), 2.0, device=device),
            "value_real": torch.full((1, 2), 2.0, device=device),
            "target_actor": torch.full((1, 2), 2.0, device=device),
            "base_actor": torch.zeros((1, 2), device=device),
            "weights_actor": torch.ones((1, 2), device=device),
        }
        imag_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.zeros((1, 2, 4), device=device),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], device=device),
            "rewards": torch.zeros((1, 2), device=device),
            "dones": torch.zeros((1, 2), device=device),
            "log_probs": torch.zeros((1, 2), device=device),
            "advantages": torch.full((1, 2), -1.0, device=device),
            "returns": torch.zeros((1, 2), device=device),
            "values": torch.ones((1, 2), device=device),
            "weights_actor": torch.ones((1, 2), device=device),
            "target_actor": torch.zeros((1, 2), device=device),
            "base_actor": torch.ones((1, 2), device=device),
            "compensation_phase": "post_solved",
            "external_eval_best_mean": 350.0,
        }
        metrics = loop.train_step(
            rl_batch=imag_batch,
            wm_batch=None,
            value_real_batch=real_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )
        self.assertAlmostEqual(float(metrics["actor/post_solved_real_adv_mean"]), 2.0, places=6)
        self.assertAlmostEqual(float(metrics["actor/post_solved_real_adv_latch_veto_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["actor/negative_adv_guard_active"]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["actor/negative_adv_guard_latched"]), 0.0, places=6)

    def test_post_solved_actor_anchor_kl_penalizes_policy_drift(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            adaptive_imag_compensation_post_solved_actor_anchor_kl=0.05,
            adaptive_imag_compensation_post_solved_actor_anchor_latched_kl=0.1,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.set_external_eval_feedback(350.0, step=100)
        feats = torch.randn(6, 4, device=device)
        current_dist = loop.model.actor(feats)
        zero_penalty = loop._compute_post_solved_actor_anchor_kl(
            policy_inputs=feats,
            current_dist=current_dist,
            weight=0.05,
        )
        self.assertAlmostEqual(float(zero_penalty.detach()), 0.0, places=6)

        with torch.no_grad():
            torch.manual_seed(0)
            for param in loop.model.actor.parameters():
                param.add_(0.25 * torch.randn_like(param))
        drift_dist = loop.model.actor(feats)
        penalty = loop._compute_post_solved_actor_anchor_kl(
            policy_inputs=feats,
            current_dist=drift_dist,
            weight=0.05,
        )
        self.assertGreater(float(penalty.detach()), 0.0)
        self.assertAlmostEqual(
            loop._resolve_post_solved_actor_anchor_kl(
                compensation_phase="post_solved",
                negative_adv_guard_latched=0.0,
                external_eval_best_mean=0.0,
            ),
            0.05,
            places=6,
        )
        self.assertAlmostEqual(
            loop._resolve_post_solved_actor_anchor_kl(
                compensation_phase="post_solved",
                negative_adv_guard_latched=1.0,
                external_eval_best_mean=0.0,
            ),
            0.1,
            places=6,
        )

    def test_post_solved_actor_anchor_highwater_strengthens_pull_and_kl(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            adaptive_imag_compensation_post_solved_actor_anchor_pull=0.02,
            adaptive_imag_compensation_post_solved_actor_anchor_kl=0.05,
            adaptive_imag_compensation_post_solved_actor_anchor_highwater_eval_threshold=450.0,
            adaptive_imag_compensation_post_solved_actor_anchor_highwater_pull=0.08,
            adaptive_imag_compensation_post_solved_actor_anchor_highwater_kl=0.09,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.set_external_eval_feedback(350.0, step=100)
        self.assertAlmostEqual(
            loop._resolve_post_solved_actor_anchor_pull(
                compensation_phase="post_solved",
                negative_adv_guard_active=0.0,
                negative_adv_guard_latched=0.0,
                external_eval_best_mean=350.0,
            ),
            0.02,
            places=6,
        )
        self.assertAlmostEqual(
            loop._resolve_post_solved_actor_anchor_kl(
                compensation_phase="post_solved",
                negative_adv_guard_latched=0.0,
                external_eval_best_mean=350.0,
            ),
            0.05,
            places=6,
        )
        self.assertAlmostEqual(
            loop._resolve_post_solved_actor_anchor_pull(
                compensation_phase="post_solved",
                negative_adv_guard_active=0.0,
                negative_adv_guard_latched=0.0,
                external_eval_best_mean=460.0,
            ),
            0.08,
            places=6,
        )
        self.assertAlmostEqual(
            loop._resolve_post_solved_actor_anchor_kl(
                compensation_phase="post_solved",
                negative_adv_guard_latched=0.0,
                external_eval_best_mean=460.0,
            ),
            0.09,
            places=6,
        )

    def test_post_solved_critic_anchor_penalizes_value_drift(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            adaptive_imag_compensation_post_solved_critic_anchor_weight=0.05,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.set_external_eval_feedback(350.0, step=100)
        feats = torch.randn(6, 4, device=device)
        current_values = loop.model.critic(feats).squeeze(-1)
        zero_penalty = loop._compute_post_solved_critic_anchor_penalty(
            critic_inputs=feats,
            current_values=current_values,
            weight=0.05,
        )
        self.assertAlmostEqual(float(zero_penalty.detach()), 0.0, places=6)
        with torch.no_grad():
            torch.manual_seed(0)
            for param in loop.model.critic.parameters():
                param.add_(0.25 * torch.randn_like(param))
        drift_values = loop.model.critic(feats).squeeze(-1)
        penalty = loop._compute_post_solved_critic_anchor_penalty(
            critic_inputs=feats,
            current_values=drift_values,
            weight=0.05,
        )
        self.assertGreater(float(penalty.detach()), 0.0)
        self.assertAlmostEqual(
            loop._resolve_post_solved_critic_anchor_weight(compensation_phase="post_solved"),
            0.05,
            places=6,
        )
        self.assertAlmostEqual(
            loop._resolve_post_solved_critic_anchor_weight(compensation_phase="post_entry"),
            0.0,
            places=6,
        )

    def test_post_solved_critic_anchor_is_disabled_by_default(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.set_external_eval_feedback(350.0, step=100)

        self.assertIsNotNone(
            getattr(loop, "_adaptive_imag_compensation_post_solved_critic_anchor_critic", None)
        )
        self.assertAlmostEqual(
            loop._resolve_post_solved_critic_anchor_weight(compensation_phase="post_solved"),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            loop._resolve_post_solved_critic_anchor_weight(compensation_phase="post_entry"),
            0.0,
            places=6,
        )

    def test_post_solved_drift_damping_scales_strengthen_after_highwater(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_post_solved_eval_threshold=300.0,
            adaptive_imag_compensation_post_solved_cap=0.82,
            adaptive_imag_compensation_post_solved_actor_scale=0.75,
            adaptive_imag_compensation_post_solved_hold_steps=50,
            adaptive_imag_compensation_post_solved_drift_damping_eval_threshold=450.0,
            adaptive_imag_compensation_post_solved_drift_damping_wm_scale=0.5,
            adaptive_imag_compensation_post_solved_drift_damping_critic_scale=0.6,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FixedImaginationEngine(
            device,
            rewards=[[0.0, 0.0]],
            values=[[5.0, 5.0, 5.0]],
            continue_probs=[[0.0, 0.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        loop.global_step = 120
        loop.set_external_eval_feedback(460.0, step=100)
        self.assertEqual(
            loop._resolve_post_solved_drift_damping_scales(
                compensation_phase="post_solved",
                external_eval_best_mean=460.0,
            ),
            (1.0, 0.5, 0.6),
        )
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(batch["compensation_phase"], "post_solved")
        metrics = loop.train_step(rl_batch=batch, source_tag="imag", imag_ratio=1.0)
        self.assertAlmostEqual(float(metrics["critic/post_solved_drift_damping_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["critic/post_solved_scale"]), 0.6, places=6)

    def test_persistence_stage_negative_adv_guard_boosts_critic_without_extra_actor_clamp(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_eval_drop=20.0,
            adaptive_imag_compensation_persistence_cap=0.86,
            adaptive_imag_compensation_persistence_actor_scale=0.87,
            adaptive_imag_compensation_persistence_hold_steps=50,
            adaptive_imag_compensation_persistence_negative_adv_threshold=0.5,
            adaptive_imag_compensation_persistence_negative_adv_actor_scale=1.0,
            adaptive_imag_compensation_persistence_negative_adv_critic_boost=0.5,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FixedImaginationEngine(
            device,
            rewards=[[0.0, 0.0]],
            values=[[5.0, 5.0, 5.0]],
            continue_probs=[[0.0, 0.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        loop.set_external_eval_feedback(180.0, step=100)
        loop.set_external_eval_feedback(140.0, step=120)
        loop.global_step = 130

        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(batch["compensation_phase"], "persistence")

        metrics = loop.train_step(rl_batch=batch, source_tag="imag", imag_ratio=1.0)
        self.assertAlmostEqual(float(metrics["actor/negative_adv_guard_active"]), 1.0, places=6)
        self.assertEqual(metrics["actor/negative_adv_guard_stage"], "persistence")
        self.assertAlmostEqual(float(metrics["actor/negative_adv_guard_scale"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["critic/negative_adv_guard_multiplier"]), 1.5, places=6)
        self.assertLess(float(metrics["actor/negative_adv_guard_raw_adv_mean"]), -0.5)

    def test_post_trigger_return_delta_clip_activates_only_under_adaptive_pressure(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_trigger_return_delta_clip=5.0,
            adaptive_imag_compensation_trigger_min_pressure=0.01,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _AdaptiveCapImaginationEngine(device)

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertGreater(float(loop.metrics["imag/continue_cap_pressure"]), 0.01)
        self.assertAlmostEqual(float(loop.metrics["imag/post_trigger_return_delta_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/post_trigger_return_delta_clip"]), 5.0, places=6)
        baseline = batch["values"]
        delta = (batch["returns"] - baseline).abs()
        self.assertLessEqual(float(delta.max()), 5.000001)

    def test_post_trigger_actor_scale_reduces_actor_objective_only(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_trigger_min_pressure=0.01,
            adaptive_imag_compensation_trigger_actor_scale_gain=4.0,
            adaptive_imag_compensation_trigger_actor_scale_floor=0.2,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _AdaptiveCapImaginationEngine(device)

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertGreater(float(loop.metrics["imag/continue_cap_pressure"]), 0.01)
        self.assertLess(float(loop.metrics["actor/post_trigger_scale"]), 1.0)
        self.assertGreaterEqual(float(loop.metrics["actor/post_trigger_scale"]), 0.2)
        self.assertAlmostEqual(
            float(batch["actor_post_trigger_scale"].mean()),
            float(loop.metrics["actor/post_trigger_scale"]),
            places=6,
        )

    def test_post_trigger_actor_hysteresis_holds_scale_between_engage_and_release(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_trigger_min_pressure=0.01,
            adaptive_imag_compensation_trigger_actor_scale_gain=3.0,
            adaptive_imag_compensation_trigger_actor_scale_floor=0.25,
            adaptive_imag_compensation_trigger_hysteresis=True,
            adaptive_imag_compensation_trigger_release_ratio=0.5,
            adaptive_imag_compensation_trigger_attack_ema=0.0,
            adaptive_imag_compensation_trigger_release_ema=0.9,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _AdaptiveCapImaginationEngine(device)

        first = loop._build_imagined_batch()
        self.assertIsNotNone(first)
        engaged_scale = float(loop.metrics["actor/post_trigger_scale"])
        self.assertLess(engaged_scale, 1.0)

        loop.imagination_engine = _MidPressureImaginationEngine(device)
        second = loop._build_imagined_batch()
        self.assertIsNotNone(second)
        held_scale = float(loop.metrics["actor/post_trigger_scale"])
        self.assertAlmostEqual(held_scale, engaged_scale, places=6)

        loop.imagination_engine = _FakeImaginationEngine(device)
        third = loop._build_imagined_batch()
        self.assertIsNotNone(third)
        self.assertGreater(float(loop.metrics["actor/post_trigger_scale"]), engaged_scale)

    def test_idle_corridor_clean_target_replacement_emits_replay_suffix_target_audit_without_rewriting_actor_target(self):
        device = torch.device("cpu")

        def make_loop(clean_target_blend_max: float) -> TrainingLoop:
            model = _TinyImagModel().to(device)
            with torch.no_grad():
                model.critic.weight.zero_()
                model.critic.bias.fill_(6.0)

            buffer = ReplayBuffer(capacity=8)
            buffer.start_episode(initial_vitals=np.full((4,), 0.0, dtype=np.float32))
            buffer.add_step(
                vitals=np.full((4,), 1.0, dtype=np.float32),
                action=np.array([1.0, 0.0], dtype=np.float32),
                reward=2.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 2.0, dtype=np.float32),
                action=np.array([0.0, 1.0], dtype=np.float32),
                reward=1.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 3.0, dtype=np.float32),
                action=np.array([1.0, 0.0], dtype=np.float32),
                reward=1.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 4.0, dtype=np.float32),
                action=np.array([0.0, 1.0], dtype=np.float32),
                reward=0.0,
                done=True,
                terminated=True,
                truncated=False,
                log_prob=0.0,
            )
            buffer._select_uniform = lambda valid, batch_size, required: [(0, 0)] * batch_size

            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=4,
                batch_size=1,
                seq_len=4,
                wm_seq_len=4,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                imagination_batch_size=1,
                adaptive_imag_idle_corridor_advantage_blend_max=1.0,
                adaptive_imag_idle_corridor_negative_adv_threshold=0.0,
                adaptive_imag_idle_corridor_negative_adv_tau=0.01,
                adaptive_imag_idle_corridor_quantile=0.0,
                adaptive_imag_idle_corridor_clean_target_blend_max=clean_target_blend_max,
                adaptive_imag_idle_corridor_inflation_threshold=1.0,
                adaptive_imag_idle_corridor_inflation_tau=0.01,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            config.rl.use_reward_ema = False
            config.rl.use_advantage_normalization = False
            config.rl.actor_entropy_scale = 0.0
            loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
            loop.imagination_engine = _FixedImaginationEngine(
                device,
                rewards=[[0.0, 0.0]],
                values=[[6.0, 6.0, 6.0]],
                continue_probs=[[1.0, 0.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            return loop

        plain_loop = make_loop(0.0)
        clean_loop = make_loop(0.8)
        clean_loop.model.load_state_dict(plain_loop.model.state_dict())

        plain_batch = plain_loop._build_imagined_batch()
        clean_batch = clean_loop._build_imagined_batch()

        self.assertIsNotNone(plain_batch)
        self.assertIsNotNone(clean_batch)
        self.assertAlmostEqual(
            float(clean_batch["corridor_semantic_clean_target_available_fraction"]),
            1.0,
            places=6,
        )
        self.assertGreater(
            float(clean_batch["corridor_semantic_inflation_gate_mean"]),
            0.0,
        )
        self.assertGreater(
            float(clean_batch["corridor_semantic_clean_target_blend_mean"]),
            0.0,
        )
        self.assertAlmostEqual(
            float(clean_batch["target_actor"].mean()),
            float(plain_batch["target_actor"].mean()),
            places=6,
        )
        self.assertLess(
            float(clean_batch["corridor_semantic_target_inflation_weighted_mean"]),
            float(clean_batch["corridor_semantic_target_raw_inflation_weighted_mean"]),
        )

    def test_idle_corridor_clean_target_replacement_extends_partial_replay_support_into_tail_as_audit_signal(self):
        device = torch.device("cpu")

        model = _TinyImagModel().to(device)
        with torch.no_grad():
            model.critic.weight.zero_()
            model.critic.bias.fill_(6.0)

        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 0.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 1.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=2.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 4.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        buffer._select_uniform = lambda valid, batch_size, required: [(0, 0)] * batch_size

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=4,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=4,
            imagination_batch_size=1,
            adaptive_imag_idle_corridor_advantage_blend_max=1.0,
            adaptive_imag_idle_corridor_negative_adv_threshold=0.0,
            adaptive_imag_idle_corridor_negative_adv_tau=0.01,
            adaptive_imag_idle_corridor_quantile=0.0,
            adaptive_imag_idle_corridor_clean_target_blend_max=0.8,
            adaptive_imag_idle_corridor_inflation_threshold=1.0,
            adaptive_imag_idle_corridor_inflation_tau=0.01,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        config.rl.use_reward_ema = False
        config.rl.use_advantage_normalization = False
        config.rl.actor_entropy_scale = 0.0
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _FixedImaginationEngine(
            device,
            rewards=[[0.0, 0.0, 0.0, 0.0]],
            values=[[6.0, 6.0, 6.0, 6.0, 6.0]],
            continue_probs=[[1.0, 1.0, 1.0, 0.0]],
            entropies=[[0.0, 0.0, 0.0, 0.0]],
            log_probs=[[0.0, 0.0, 0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
        )

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(
            float(batch["corridor_semantic_clean_target_available_fraction"]),
            0.5,
            places=6,
        )
        self.assertGreater(
            float(batch["corridor_semantic_clean_target_blend_active_fraction"]),
            0.5,
        )
        self.assertAlmostEqual(
            float(batch["target_actor"][0, 2]),
            float(batch["target_actor_raw"][0, 2]),
            places=6,
        )
        self.assertLess(
            float(batch["corridor_semantic_target_inflation_weighted_mean"]),
            float(batch["corridor_semantic_target_raw_inflation_weighted_mean"]),
        )

    def test_idle_corridor_clean_target_replacement_keeps_floor_takeover_when_advantage_turns_positive(self):
        device = torch.device("cpu")

        def make_loop(clean_target_floor_max: float) -> TrainingLoop:
            model = _TinyImagModel().to(device)
            with torch.no_grad():
                model.critic.weight.zero_()
                model.critic.bias.fill_(6.0)

            buffer = ReplayBuffer(capacity=8)
            buffer.start_episode(initial_vitals=np.full((4,), 0.0, dtype=np.float32))
            buffer.add_step(
                vitals=np.full((4,), 1.0, dtype=np.float32),
                action=np.array([1.0, 0.0], dtype=np.float32),
                reward=2.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 2.0, dtype=np.float32),
                action=np.array([0.0, 1.0], dtype=np.float32),
                reward=1.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 3.0, dtype=np.float32),
                action=np.array([1.0, 0.0], dtype=np.float32),
                reward=1.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 4.0, dtype=np.float32),
                action=np.array([0.0, 1.0], dtype=np.float32),
                reward=0.0,
                done=True,
                terminated=True,
                truncated=False,
                log_prob=0.0,
            )
            buffer._select_uniform = lambda valid, batch_size, required: [(0, 0)] * batch_size

            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=4,
                batch_size=1,
                seq_len=4,
                wm_seq_len=4,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                imagination_batch_size=1,
                adaptive_imag_idle_corridor_advantage_blend_max=1.0,
                adaptive_imag_idle_corridor_negative_adv_threshold=0.0,
                adaptive_imag_idle_corridor_negative_adv_tau=0.01,
                adaptive_imag_idle_corridor_quantile=0.0,
                adaptive_imag_idle_corridor_clean_target_blend_max=0.8,
                adaptive_imag_idle_corridor_clean_target_inflation_floor_max=clean_target_floor_max,
                adaptive_imag_idle_corridor_inflation_threshold=1.0,
                adaptive_imag_idle_corridor_inflation_tau=0.01,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            config.rl.use_reward_ema = False
            config.rl.use_advantage_normalization = False
            config.rl.actor_entropy_scale = 0.0
            loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
            loop.imagination_engine = _FixedImaginationEngine(
                device,
                rewards=[[3.0, 3.0]],
                values=[[8.0, 8.0, 8.0]],
                continue_probs=[[1.0, 1.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            return loop

        no_floor_loop = make_loop(0.0)
        floor_loop = make_loop(0.2)
        floor_loop.model.load_state_dict(no_floor_loop.model.state_dict())

        no_floor_batch = no_floor_loop._build_imagined_batch()
        floor_batch = floor_loop._build_imagined_batch()

        self.assertIsNotNone(no_floor_batch)
        self.assertIsNotNone(floor_batch)
        self.assertGreater(float(floor_batch["corridor_semantic_inflation_gate_mean"]), 0.0)
        self.assertGreater(
            float(floor_batch["target_actor_raw"].mean()),
            float(floor_batch["base_actor"].mean()),
        )
        self.assertAlmostEqual(
            float(no_floor_batch["corridor_semantic_clean_target_floor_mean"]),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(floor_batch["corridor_semantic_clean_target_floor_mean"]),
            0.0,
        )
        self.assertGreaterEqual(
            float(floor_batch["corridor_semantic_clean_target_blend_mean"]),
            float(no_floor_batch["corridor_semantic_clean_target_blend_mean"]),
        )

    def test_idle_corridor_clean_target_replacement_uses_stronger_takeover_for_larger_target_clean_gap_in_audit_preview(self):
        device = torch.device("cpu")

        def make_loop(value_bias: float) -> TrainingLoop:
            model = _TinyImagModel().to(device)
            with torch.no_grad():
                model.critic.weight.zero_()
                model.critic.bias.fill_(4.0)

            buffer = ReplayBuffer(capacity=8)
            buffer.start_episode(initial_vitals=np.full((4,), 0.0, dtype=np.float32))
            buffer.add_step(
                vitals=np.full((4,), 1.0, dtype=np.float32),
                action=np.array([1.0, 0.0], dtype=np.float32),
                reward=2.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 2.0, dtype=np.float32),
                action=np.array([0.0, 1.0], dtype=np.float32),
                reward=1.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 3.0, dtype=np.float32),
                action=np.array([1.0, 0.0], dtype=np.float32),
                reward=1.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 4.0, dtype=np.float32),
                action=np.array([0.0, 1.0], dtype=np.float32),
                reward=0.0,
                done=True,
                terminated=True,
                truncated=False,
                log_prob=0.0,
            )
            buffer._select_uniform = lambda valid, batch_size, required: [(0, 0)] * batch_size

            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=4,
                batch_size=1,
                seq_len=4,
                wm_seq_len=4,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                imagination_batch_size=1,
                adaptive_imag_idle_corridor_advantage_blend_max=1.0,
                adaptive_imag_idle_corridor_negative_adv_threshold=0.0,
                adaptive_imag_idle_corridor_negative_adv_tau=0.01,
                adaptive_imag_idle_corridor_quantile=0.0,
                adaptive_imag_idle_corridor_clean_target_blend_max=0.8,
                adaptive_imag_idle_corridor_clean_target_inflation_floor_max=0.0,
                adaptive_imag_idle_corridor_inflation_threshold=0.0,
                adaptive_imag_idle_corridor_inflation_tau=0.01,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            config.rl.use_reward_ema = False
            config.rl.use_advantage_normalization = False
            config.rl.actor_entropy_scale = 0.0
            loop = TrainingLoop(
                model=model,
                buffer=buffer,
                config=config,
                device=device,
                env=None,
            )
            loop.imagination_engine = _FixedImaginationEngine(
                device,
                rewards=[[3.0, 3.0]],
                values=[[value_bias, value_bias, value_bias]],
                continue_probs=[[1.0, 1.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            return loop

        mild_loop = make_loop(5.0)
        strong_loop = make_loop(11.0)
        strong_loop.model.load_state_dict(mild_loop.model.state_dict())

        mild_batch = mild_loop._build_imagined_batch()
        strong_batch = strong_loop._build_imagined_batch()

        self.assertIsNotNone(mild_batch)
        self.assertIsNotNone(strong_batch)
        self.assertGreater(float(mild_batch["corridor_semantic_inflation_gate_mean"]), 0.0)
        self.assertGreater(float(strong_batch["corridor_semantic_inflation_gate_mean"]), 0.0)
        self.assertGreater(
            float(strong_batch["corridor_semantic_clean_target_dominant_gap_mean"]),
            float(mild_batch["corridor_semantic_clean_target_dominant_gap_mean"]),
        )
        self.assertGreater(
            float(strong_batch["corridor_semantic_clean_target_dominant_alpha_mean"]),
            float(mild_batch["corridor_semantic_clean_target_dominant_alpha_mean"]),
        )
        mild_reduction = float(
            mild_batch["corridor_semantic_target_raw_inflation_weighted_mean"]
            - mild_batch["corridor_semantic_target_inflation_weighted_mean"]
        )
        strong_reduction = float(
            strong_batch["corridor_semantic_target_raw_inflation_weighted_mean"]
            - strong_batch["corridor_semantic_target_inflation_weighted_mean"]
        )
        self.assertGreater(strong_reduction, mild_reduction)
        self.assertGreater(
            float(strong_batch["corridor_semantic_target_gap_inflation_weighted_mean"]),
            0.0,
        )

    def test_build_imagined_batch_task_corridor_marks_high_value_low_task_states(self):
        device = torch.device("cpu")

        class _FeatureImaginationEngine(_FixedImaginationEngine):
            def __init__(self, device: torch.device, *, policy_features, rewards, values, continue_probs, entropies, log_probs, actions):
                super().__init__(
                    device,
                    rewards=rewards,
                    values=values,
                    continue_probs=continue_probs,
                    entropies=entropies,
                    log_probs=log_probs,
                    actions=actions,
                )
                self._policy_features = torch.tensor(
                    policy_features,
                    device=device,
                    dtype=torch.float32,
                )

            def imagine_rollout_differentiable(
                self,
                initial_state,
                initial_prev_action,
                horizon,
                actor,
                critic,
                initial_wm_state=None,
            ):
                out = super().imagine_rollout_differentiable(
                    initial_state,
                    initial_prev_action,
                    horizon,
                    actor,
                    critic,
                    initial_wm_state=initial_wm_state,
                )
                batch = initial_state.shape[0]
                out["policy_features"] = self._policy_features.repeat(batch, 1, 1)
                return out

        model = _TinyImagModel().to(device)
        with torch.no_grad():
            model.critic.weight.zero_()
            model.critic.bias.fill_(6.0)

        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 0.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 1.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        buffer._select_uniform = lambda valid, batch_size, required: [(0, 0)] * batch_size

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            adaptive_imag_idle_corridor_quantile=0.0,
            adaptive_imag_task_corridor_enabled=True,
            adaptive_imag_task_corridor_high_quantile=0.75,
            adaptive_imag_task_corridor_low_quantile=0.25,
            adaptive_imag_task_corridor_gate_tau=0.05,
            adaptive_imag_task_corridor_analytic_floor=0.25,
            adaptive_imag_task_corridor_confidence_scale=0.25,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        config.rl.use_reward_ema = False
        config.rl.use_advantage_normalization = False
        config.rl.actor_entropy_scale = 0.0
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _FeatureImaginationEngine(
            device,
            policy_features=[[[-1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
            rewards=[[0.0, 0.0]],
            values=[[6.0, 6.0, 6.0]],
            continue_probs=[[1.0, 1.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )

        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertLess(float(batch["task_corridor_gate_mean"]), 0.5)
        self.assertLess(float(batch["task_corridor_analytic_scale_mean"]), 1.0)
        self.assertLess(
            float(batch["task_corridor_analytic_scale"].mean()),
            1.0,
        )
        self.assertGreater(float(batch["high_value_but_low_task_fraction"]), 0.5)
        self.assertGreater(float(batch["task_geom_corridor_disagreement"]), 0.0)

    def test_build_imagined_batch_task_corridor_stays_neutral_when_real_prototypes_do_not_separate(self):
        device = torch.device("cpu")

        class _FeatureImaginationEngine(_FixedImaginationEngine):
            def __init__(self, device: torch.device, *, policy_features, rewards, values, continue_probs, entropies, log_probs, actions):
                super().__init__(
                    device,
                    rewards=rewards,
                    values=values,
                    continue_probs=continue_probs,
                    entropies=entropies,
                    log_probs=log_probs,
                    actions=actions,
                )
                self._policy_features = torch.tensor(
                    policy_features,
                    device=device,
                    dtype=torch.float32,
                )

            def imagine_rollout_differentiable(
                self,
                initial_state,
                initial_prev_action,
                horizon,
                actor,
                critic,
                initial_wm_state=None,
            ):
                out = super().imagine_rollout_differentiable(
                    initial_state,
                    initial_prev_action,
                    horizon,
                    actor,
                    critic,
                    initial_wm_state=initial_wm_state,
                )
                batch = initial_state.shape[0]
                out["policy_features"] = self._policy_features.repeat(batch, 1, 1)
                return out

        model = _TinyImagModel().to(device)
        with torch.no_grad():
            model.critic.weight.zero_()
            model.critic.bias.fill_(6.0)

        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 0.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 1.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        buffer._select_uniform = lambda valid, batch_size, required: [(0, 0)] * batch_size

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            adaptive_imag_idle_corridor_quantile=0.0,
            adaptive_imag_task_corridor_enabled=True,
            adaptive_imag_task_corridor_high_quantile=0.75,
            adaptive_imag_task_corridor_low_quantile=0.25,
            adaptive_imag_task_corridor_gate_tau=0.05,
            adaptive_imag_task_corridor_analytic_floor=0.25,
            adaptive_imag_task_corridor_confidence_scale=0.25,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        config.rl.use_reward_ema = False
        config.rl.use_advantage_normalization = False
        config.rl.actor_entropy_scale = 0.0
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _FeatureImaginationEngine(
            device,
            policy_features=[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            rewards=[[0.0, 0.0]],
            values=[[6.0, 6.0, 6.0]],
            continue_probs=[[1.0, 1.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )

        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[1.0, 1.0]], device=device),
        }

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(
            float(batch["task_corridor_confidence_mean"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["task_corridor_analytic_scale_mean"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["task_corridor_analytic_scale"].mean()),
            1.0,
            places=6,
        )

    def test_actor_contract_trust_drives_corridor_beta_and_reduces_weighted_actor_target_when_task_and_registry_mismatch_align(self):
        device = torch.device("cpu")

        class _FeatureImaginationEngine(_FixedImaginationEngine):
            def __init__(self, device: torch.device, *, policy_features, rewards, values, continue_probs, entropies, log_probs, actions):
                super().__init__(
                    device,
                    rewards=rewards,
                    values=values,
                    continue_probs=continue_probs,
                    entropies=entropies,
                    log_probs=log_probs,
                    actions=actions,
                )
                self._policy_features = torch.tensor(
                    policy_features,
                    device=device,
                    dtype=torch.float32,
                )

            def imagine_rollout_differentiable(
                self,
                initial_state,
                initial_prev_action,
                horizon,
                actor,
                critic,
                initial_wm_state=None,
            ):
                out = super().imagine_rollout_differentiable(
                    initial_state,
                    initial_prev_action,
                    horizon,
                    actor,
                    critic,
                    initial_wm_state=initial_wm_state,
                )
                batch = initial_state.shape[0]
                out["policy_features"] = self._policy_features.repeat(batch, 1, 1)
                return out

        def make_loop(*, drift_actor: bool) -> TrainingLoop:
            model = _TinyImagModel().to(device)
            with torch.no_grad():
                model.critic.weight.zero_()
                model.critic.bias.fill_(6.0)
                model.actor.net.weight.zero_()
                model.actor.net.bias.zero_()

            buffer = ReplayBuffer(capacity=8)
            buffer.start_episode(initial_vitals=np.full((4,), 0.0, dtype=np.float32))
            buffer.add_step(
                vitals=np.full((4,), 1.0, dtype=np.float32),
                action=np.array([1.0, 0.0], dtype=np.float32),
                reward=2.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 2.0, dtype=np.float32),
                action=np.array([0.0, 1.0], dtype=np.float32),
                reward=1.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 3.0, dtype=np.float32),
                action=np.array([1.0, 0.0], dtype=np.float32),
                reward=1.0,
                done=False,
                log_prob=0.0,
            )
            buffer.add_step(
                vitals=np.full((4,), 4.0, dtype=np.float32),
                action=np.array([0.0, 1.0], dtype=np.float32),
                reward=0.0,
                done=True,
                terminated=True,
                truncated=False,
                log_prob=0.0,
            )
            buffer._select_uniform = lambda valid, batch_size, required: [(0, 0)] * batch_size

            config = TrainingConfig(
                config_mode="strict",
                validation_mode="off",
                total_steps=1,
                num_train_steps=1,
                total_env_steps=4,
                batch_size=1,
                seq_len=4,
                wm_seq_len=4,
                wm_batch_size=1,
                rl_batch_size=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
                imagination_only=True,
                imagination_horizon=2,
                imagination_batch_size=1,
                adaptive_imag_idle_corridor_advantage_blend_max=0.0,
                adaptive_imag_idle_corridor_clean_target_blend_max=0.0,
                adaptive_imag_idle_corridor_quantile=0.0,
                adaptive_imag_task_corridor_enabled=True,
                adaptive_imag_task_corridor_high_quantile=0.75,
                adaptive_imag_task_corridor_low_quantile=0.25,
                adaptive_imag_task_corridor_gate_tau=0.05,
                adaptive_imag_task_corridor_analytic_floor=0.25,
                adaptive_imag_task_corridor_confidence_scale=0.25,
                adaptive_imag_actor_contract_trust_floor=0.0,
                log_interval=1000,
                eval_interval=1000,
                save_interval=1000,
            )
            config.rl.use_reward_ema = False
            config.rl.use_advantage_normalization = False
            config.rl.actor_entropy_scale = 0.0
            loop = TrainingLoop(
                model=model,
                buffer=buffer,
                config=config,
                device=device,
                env=None,
            )
            loop.imagination_engine = _FeatureImaginationEngine(
                device,
                policy_features=[[[-1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                rewards=[[0.0, 0.0]],
                values=[[6.0, 6.0, 6.0]],
                continue_probs=[[1.0, 0.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            registry_actor = copy.deepcopy(model.actor).to(device)
            registry_actor.eval()
            for param in registry_actor.parameters():
                param.requires_grad_(False)
            loop._real_stability_certified_anchor_registry_actors = [registry_actor]
            loop._real_stability_certified_anchor_registry_steps = [100]
            loop._real_stability_certified_anchor_registry_evals = [500.0]
            loop._real_stability_certified_anchor_registry_telemetries = [{}]
            if drift_actor:
                with torch.no_grad():
                    model.actor.net.bias.copy_(torch.tensor([3.0, -3.0], device=device))
            return loop

        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        plain_loop = make_loop(drift_actor=False)
        mismatch_loop = make_loop(drift_actor=True)

        plain_batch = plain_loop._build_imagined_batch()
        mismatch_batch = mismatch_loop._build_imagined_batch(
            reference_real_batch=reference_real_batch
        )

        self.assertIsNotNone(plain_batch)
        self.assertIsNotNone(mismatch_batch)
        self.assertAlmostEqual(
            float(mismatch_batch["behavior_policy_certified_registry_available"]),
            1.0,
            places=6,
        )
        self.assertGreater(
            float(mismatch_batch["actor_contract_certified_registry_mismatch_mean"]),
            float(plain_batch["actor_contract_certified_registry_mismatch_mean"]),
        )
        self.assertGreaterEqual(
            float(mismatch_batch["actor_contract_semantic_pressure_mean"]),
            float(plain_batch["actor_contract_semantic_pressure_mean"]),
        )
        self.assertLess(
            float(mismatch_batch["actor_contract_semantic_trust_mean"]),
            float(plain_batch["actor_contract_semantic_trust_mean"]),
        )
        self.assertGreater(
            float(mismatch_batch["corridor_semantic_blend_beta"].mean()),
            float(plain_batch["corridor_semantic_blend_beta"].mean()),
        )
        self.assertAlmostEqual(
            float(mismatch_batch["target_actor"].mean()),
            float(plain_batch["target_actor"].mean()),
            places=6,
        )

        plain_metrics = plain_loop.train_step(
            rl_batch=plain_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )
        mismatch_metrics = mismatch_loop.train_step(
            rl_batch=mismatch_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )
        self.assertAlmostEqual(
            float(mismatch_metrics["actor/corridor_semantic_blend_mean"]),
            float(mismatch_batch["corridor_semantic_blend_beta"].mean()),
            places=6,
        )
        self.assertLess(
            float(mismatch_metrics["actor/weighted_actor_target_mean"]),
            float(plain_metrics["actor/weighted_actor_target_mean"]),
        )

    def test_actor_contract_trust_activates_on_task_semantic_mismatch_even_when_geometry_registry_support_remains_high(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        plain_loop = _make_actor_contract_trust_test_loop(
            device,
            drift_actor=False,
        )
        mismatch_loop = _make_actor_contract_trust_test_loop(
            device,
            drift_actor=False,
        )
        mismatch_loop.model.load_state_dict(plain_loop.model.state_dict())
        mismatch_loop._real_stability_certified_anchor_registry_actors = [
            copy.deepcopy(actor).to(device)
            for actor in plain_loop._real_stability_certified_anchor_registry_actors
        ]
        for actor in mismatch_loop._real_stability_certified_anchor_registry_actors:
            actor.eval()
            for param in actor.parameters():
                param.requires_grad_(False)

        plain_batch = plain_loop._build_imagined_batch()
        mismatch_batch = mismatch_loop._build_imagined_batch(
            reference_real_batch=reference_real_batch
        )

        self.assertIsNotNone(plain_batch)
        self.assertIsNotNone(mismatch_batch)
        self.assertAlmostEqual(
            float(mismatch_batch["behavior_policy_certified_registry_available"]),
            1.0,
            places=6,
        )
        self.assertGreater(
            float(mismatch_batch["behavior_policy_geometry_registry_support_fraction"]),
            0.99,
        )
        self.assertLess(
            float(mismatch_batch["actor_contract_certified_registry_support_mean"]),
            0.5,
        )
        self.assertGreater(
            float(mismatch_batch["actor_contract_certified_registry_mismatch_mean"]),
            0.5,
        )
        self.assertGreater(
            float(mismatch_batch["actor_contract_task_mismatch_mean"]),
            0.99,
        )
        self.assertGreater(
            float(mismatch_batch["actor_contract_semantic_certified_mismatch_mean"]),
            0.99,
        )
        self.assertGreater(
            float(mismatch_batch["actor_contract_semantic_pressure_mean"]),
            0.0,
        )
        self.assertLess(
            float(mismatch_batch["actor_contract_semantic_trust_mean"]),
            1.0,
        )
        self.assertGreater(
            float(mismatch_batch["corridor_semantic_blend_beta"].mean()),
            float(plain_batch["corridor_semantic_blend_beta"].mean()),
        )
        self.assertAlmostEqual(
            float(mismatch_batch["target_actor"].mean()),
            float(plain_batch["target_actor"].mean()),
            places=6,
        )

        plain_metrics = plain_loop.train_step(
            rl_batch=plain_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )
        mismatch_metrics = mismatch_loop.train_step(
            rl_batch=mismatch_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )
        self.assertAlmostEqual(
            float(mismatch_metrics["actor/corridor_semantic_blend_mean"]),
            float(mismatch_batch["corridor_semantic_blend_beta"].mean()),
            places=6,
        )
        self.assertLess(
            float(mismatch_metrics["actor/weighted_actor_target_mean"]),
            float(plain_metrics["actor/weighted_actor_target_mean"]),
        )

    def test_build_imagined_batch_emits_bootstrap_rewritten_critic_target_branch_without_changing_actor_target(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        plain_loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=False,
        )
        bootstrap_loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )

        plain_batch = plain_loop._build_imagined_batch(
            reference_real_batch=reference_real_batch
        )
        bootstrap_batch = bootstrap_loop._build_imagined_batch(
            reference_real_batch=reference_real_batch
        )

        self.assertIsNotNone(plain_batch)
        self.assertIsNotNone(bootstrap_batch)
        self.assertAlmostEqual(
            float(bootstrap_batch["target_actor"].mean()),
            float(plain_batch["target_actor"].mean()),
            places=6,
        )
        self.assertAlmostEqual(
            float(bootstrap_batch["critic_contract_bootstrap_enabled"]),
            1.0,
            places=6,
        )
        self.assertGreater(
            float(bootstrap_batch["critic_contract_bootstrap_late_gate"]),
            0.99,
        )
        self.assertLess(
            float(bootstrap_batch["critic_contract_trust_mean"]),
            0.5,
        )
        self.assertGreater(
            float(bootstrap_batch["critic_contract_bootstrap_clean_mix_mean"]),
            0.5,
        )
        self.assertGreater(
            float(bootstrap_batch["critic_contract_bootstrap_semantic_debt_mean"]),
            0.0,
        )
        self.assertGreater(
            float(bootstrap_batch["critic_contract_bootstrap_raw_vs_clean_gap_mean"]),
            0.0,
        )
        self.assertGreater(
            float(bootstrap_batch["critic_contract_bootstrap_target_delta_mean"]),
            0.0,
        )
        self.assertGreater(
            float(
                (
                    bootstrap_batch["target_critic_raw"]
                    - plain_batch["target_critic_raw"]
                )
                .abs()
                .mean()
            ),
            0.0,
        )
        self.assertAlmostEqual(
            float(bootstrap_batch["target_critic_raw_mean"]),
            float(bootstrap_batch["target_critic"].mean()),
            places=6,
        )

    def test_build_imagined_batch_consumes_single_bootstrap_target_execution_result_without_post_helper_rewrite(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }
        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )

        def _forced_target_execution(**kwargs):
            reference = kwargs["raw_bootstrap_values_next"].detach()
            return BootstrapTargetExecutionResult(
                external_authority_view=torch.full_like(reference, 0.4),
                floor_authority=torch.tensor([[0.25, 0.10]], dtype=reference.dtype, device=reference.device),
                bonus_authority=torch.tensor([[0.15, 0.30]], dtype=reference.dtype, device=reference.device),
                external_value_prefinal=torch.tensor([[3.0, 4.0]], dtype=reference.dtype, device=reference.device),
                midlate_value_ratio_cap_activation=torch.ones_like(reference),
                midlate_external_value_max_ratio=torch.full_like(reference, 0.5),
                external_value_final=torch.tensor([[2.0, 3.0]], dtype=reference.dtype, device=reference.device),
                external_value_midlate_ratio_cap_delta_abs=torch.tensor([[1.0, 1.0]], dtype=reference.dtype, device=reference.device),
                internal_pessimism_gate=torch.full_like(reference, 0.8),
                internal_value_view=torch.tensor([[10.0, 12.0]], dtype=reference.dtype, device=reference.device),
                internal_value_relief_delta_abs=torch.tensor([[1.0, 2.0]], dtype=reference.dtype, device=reference.device),
                mixed_value_prefinal=torch.tensor([[6.8, 8.4]], dtype=reference.dtype, device=reference.device),
                prehold_target_cap_gate=torch.ones_like(reference),
                prehold_safe_cap=torch.tensor([[5.0, 7.0]], dtype=reference.dtype, device=reference.device),
                mixed_value_final=torch.tensor([[5.0, 7.0]], dtype=reference.dtype, device=reference.device),
                prehold_target_cap_delta_abs=torch.tensor([[1.8, 1.4]], dtype=reference.dtype, device=reference.device),
            )

        with mock.patch(
            "aletheia.aletheia_train._compute_bootstrap_target_execution_contract",
            side_effect=_forced_target_execution,
        ):
            batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertTrue(
            torch.allclose(
                batch["bootstrap_external_value_final"],
                torch.tensor([[2.0, 3.0]], dtype=torch.float32, device=device),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                batch["bootstrap_internal_value_view"],
                torch.tensor([[10.0, 12.0]], dtype=torch.float32, device=device),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                batch["mixed_bootstrap_values_next"],
                torch.tensor([[5.0, 7.0]], dtype=torch.float32, device=device),
                atol=1e-6,
            )
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_midlate_value_ratio_cap_activation_mean"]),
            0.99,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_internal_pessimism_gate_mean"]),
            0.79,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_prehold_target_cap_gate_mean"]),
            0.99,
        )

        uncapped_mixed = torch.lerp(
            batch["bootstrap_internal_value_view"],
            batch["bootstrap_external_value_final"],
            batch["bootstrap_external_authority_view"],
        )
        self.assertTrue(
            torch.any(
                uncapped_mixed > (batch["mixed_bootstrap_values_next"] + 1e-6)
            ).item()
        )

        critic_values_for_lambda = torch.cat(
            (
                batch["values"][:, :1].detach(),
                batch["mixed_bootstrap_values_next"].detach(),
            ),
            dim=1,
        )
        expected_target = loop.imagination_engine._compute_lambda_returns(
            loop.imagination_engine._rewards.repeat(batch["values"].shape[0], 1),
            critic_values_for_lambda,
            loop.imagination_engine._continue_probs.repeat(batch["values"].shape[0], 1),
            gamma=float(loop.config.rl.gamma),
            lambda_=float(loop.config.rl.gae_lambda),
        )
        self.assertTrue(
            torch.allclose(
                batch["target_critic_raw"],
                expected_target,
                atol=1e-6,
            )
        )

        uncapped_target = loop.imagination_engine._compute_lambda_returns(
            loop.imagination_engine._rewards.repeat(batch["values"].shape[0], 1),
            torch.cat((batch["values"][:, :1].detach(), uncapped_mixed.detach()), dim=1),
            loop.imagination_engine._continue_probs.repeat(batch["values"].shape[0], 1),
            gamma=float(loop.config.rl.gamma),
            lambda_=float(loop.config.rl.gae_lambda),
        )
        self.assertGreater(
            float((uncapped_target - batch["target_critic_raw"]).abs().mean()),
            0.0,
        )

    def test_bootstrap_clean_mix_tracks_critic_trust_independently_of_actor_contract_path(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertGreater(
            float(batch["critic_contract_bootstrap_late_gate"]),
            0.99,
        )
        self.assertLess(
            float(batch["critic_contract_trust_mean"]),
            0.5,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_contact_drive_mean"]),
            0.5,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_clean_mix_mean"]),
            0.5,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_mix_surface_mean"]),
            0.5,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_dense_surface_gain_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_contact_density_mean"]),
            float(batch["critic_contract_bootstrap_dense_contact_density_mean"]),
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_budget_utilization_mean"]),
            0.95,
        )

    def test_actor_consumes_unified_bootstrap_contract_trust_when_enabled(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        local_loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            actor_unified_contract_enabled=False,
        )
        unified_loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            actor_unified_contract_enabled=True,
            actor_unified_contract_blend=0.5,
        )

        local_batch = local_loop._build_imagined_batch(
            reference_real_batch=reference_real_batch
        )
        unified_batch = unified_loop._build_imagined_batch(
            reference_real_batch=reference_real_batch
        )

        self.assertIsNotNone(local_batch)
        self.assertIsNotNone(unified_batch)
        self.assertAlmostEqual(
            float(unified_batch["actor_contract_unified_trust_source_active"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(unified_batch["corridor_semantic_blend_beta"].mean()),
            0.5
            * float(local_batch["corridor_semantic_blend_beta"].mean())
            + 0.5 * float(unified_batch["critic_contract_authority_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(unified_batch["actor_contract_semantic_trust_mean"]),
            0.5 * float(local_batch["actor_contract_local_semantic_trust_mean"])
            + 0.5 * float(unified_batch["critic_contract_trust_mean"]),
            places=6,
        )
        self.assertGreater(
            float(unified_batch["corridor_semantic_blend_beta"].mean()),
            float(unified_batch["critic_contract_bootstrap_mix_surface_mean"]),
        )

        unified_metrics = unified_loop.train_step(
            rl_batch=unified_batch,
            source_tag="imag",
            imag_ratio=1.0,
        )
        self.assertAlmostEqual(
            float(unified_metrics["actor/corridor_semantic_blend_mean"]),
            float(unified_batch["corridor_semantic_blend_beta"].mean()),
            places=6,
        )

    def test_actor_unified_contract_blend_helper_preserves_soft_mapping_contract(self):
        local_trust = torch.tensor([[0.8, 0.4]], dtype=torch.float32)
        source_trust = torch.tensor([[0.2, 0.6]], dtype=torch.float32)

        blended_trust, blended_authority = _blend_actor_unified_contract(
            local_trust,
            source_trust,
            0.5,
            trust_floor=0.2,
        )

        self.assertTrue(
            torch.allclose(
                blended_trust,
                torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            )
        )

    def test_actor_unified_tail_relief_only_adds_late_negative_adv_fallback(self):
        base_authority = torch.tensor([[0.02, 0.02]], dtype=torch.float32)
        local_trust = torch.tensor([[0.90, 0.95]], dtype=torch.float32)
        advantages = torch.tensor([[-3.0, -0.2]], dtype=torch.float32)
        semantic_pressure = torch.tensor([[0.8, 0.1]], dtype=torch.float32)

        authority, trust, relief = _apply_actor_unified_tail_relief(
            base_authority,
            local_trust,
            advantages,
            semantic_pressure,
            1.0,
            relief_max=0.03,
            adv_threshold=1.0,
            adv_tau=2.0,
            trust_floor=0.2,
        )

        self.assertGreater(float(relief[0, 0]), 0.0)
        self.assertAlmostEqual(float(relief[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(
            float(authority[0, 0]),
            float(base_authority[0, 0] + relief[0, 0]),
            places=6,
        )
        self.assertAlmostEqual(
            float(trust[0, 0]),
            float(1.0 - authority[0, 0]),
            places=6,
        )
        self.assertAlmostEqual(
            float(authority[0, 1]),
            float(base_authority[0, 1]),
            places=6,
        )
        self.assertAlmostEqual(
            float(trust[0, 1]),
            float(1.0 - authority[0, 1]),
            places=6,
        )

    def test_bootstrap_surface_bonus_preserves_contract_base_authority(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertGreater(
            float(batch["critic_contract_bootstrap_late_gate"]),
            0.99,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_requested_floor_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_base_mix_surface_mean"]),
            0.0,
        )
        self.assertGreaterEqual(
            float(batch["critic_contract_bootstrap_requested_floor_mean"]),
            float(batch["critic_contract_bootstrap_base_mix_surface_mean"]),
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_surface_bonus_mean"]),
            0.0,
        )
        self.assertGreaterEqual(
            float(batch["critic_contract_bootstrap_mix_surface_mean"]),
            float(batch["critic_contract_bootstrap_base_mix_surface_mean"]),
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_surface_floor_violation_mean"]),
            0.0,
            places=6,
        )

    def test_bootstrap_clean_mix_no_longer_collapses_under_partial_anchor_coverage(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )

        def _partial_suffix_targets(self, seed_batch, horizon):
            targets = torch.tensor(
                [[9.0, 4.0]],
                device=device,
                dtype=torch.float32,
            )
            valid = torch.tensor(
                [[1.0, 1.0]],
                device=device,
                dtype=torch.float32,
            )
            return targets, valid

        loop._compute_seed_replay_suffix_targets = MethodType(
            _partial_suffix_targets,
            loop,
        )

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertLess(
            float(batch["critic_contract_bootstrap_anchor_coverage_mean"]),
            0.6,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_clean_mix_mean"]),
            0.5,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_mix_surface_mean"]),
            0.5,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_dense_surface_gain_mean"]),
            0.5,
        )
        self.assertLess(
            float(batch["critic_contract_bootstrap_effective_contact_mean"]),
            float(batch["critic_contract_bootstrap_clean_mix_mean"]),
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_dense_effective_contact_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_contact_density_mean"]),
            float(batch["critic_contract_bootstrap_dense_contact_density_mean"]),
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_budget_utilization_mean"]),
            0.95,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_dense_fallback_delta_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_dense_fallback_delta_mean"]),
            0.5,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_raw_vs_clean_gap_mean"]),
            0.0,
        )

    def test_bootstrap_focus_effective_contact_uses_valid_mask_not_fractional_coverage(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )

        def _fractional_suffix_targets(self, seed_batch, horizon):
            targets = torch.tensor(
                [[9.0, 4.0]],
                device=device,
                dtype=torch.float32,
            )
            valid = torch.tensor(
                [[1.0, 0.25]],
                device=device,
                dtype=torch.float32,
            )
            return targets, valid

        loop._compute_seed_replay_suffix_targets = MethodType(
            _fractional_suffix_targets,
            loop,
        )

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertLess(
            float(batch["critic_contract_bootstrap_anchor_coverage_mean"]),
            0.2,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_valid_mean"]),
            0.4,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_debt_focus_mean"]),
            0.5,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_valid_effective_contact_mean"]),
            0.5,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_focus_effective_contact_mean"]),
            float(batch["critic_contract_bootstrap_effective_contact_mean"]),
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_surface_state"]),
            float(batch["critic_contract_bootstrap_focus_effective_contact_mean"]),
            places=6,
        )

    def test_bootstrap_semantic_debt_persists_when_reward_alarm_recovers_but_semantic_gap_remains(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        primed_loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )
        first_batch = primed_loop._build_imagined_batch(
            reference_real_batch=reference_real_batch
        )
        self.assertIsNotNone(first_batch)
        self.assertGreater(
            float(first_batch["critic_contract_task_degradation"]),
            0.5,
        )
        self.assertGreater(
            float(first_batch["critic_contract_bootstrap_state"]),
            0.5,
        )

        primed_loop._external_eval_last_mean = 490.0
        primed_loop._real_stability_last_telemetry = {
            "real_reward_degradation": 0.02,
            "real_task_cert_gate": 0.98,
            "real_behavior_action_entropy_mean": 0.69,
            "real_behavior_action_switch_rate": 0.0,
            "real_behavior_action_oscillation_rate": 0.0,
            "real_corridor_occupancy_fraction": 1.0,
            "real_corridor_persistence_rate": 1.0,
            "real_corridor_entry_rate": 0.0,
            "real_corridor_exit_rate": 0.0,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_anchor_mean": 0.001,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.001,
        }
        primed_loop.config.adaptive_imag_critic_bootstrap_clean_mix_max = 0.2
        primed_loop.imagination_engine._values = torch.tensor(
            [[0.5, 0.5, 0.5]],
            device=device,
            dtype=torch.float32,
        )

        primed_recovered_batch = primed_loop._build_imagined_batch(
            reference_real_batch=reference_real_batch
        )

        fresh_loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            semantic_debt_decay=0.9,
        )
        fresh_loop._external_eval_last_mean = 490.0
        fresh_loop._real_stability_last_telemetry = dict(
            primed_loop._real_stability_last_telemetry
        )
        fresh_loop.config.adaptive_imag_critic_bootstrap_clean_mix_max = 0.2
        fresh_loop.imagination_engine._values = torch.tensor(
            [[0.5, 0.5, 0.5]],
            device=device,
            dtype=torch.float32,
        )
        fresh_batch = fresh_loop._build_imagined_batch(
            reference_real_batch=reference_real_batch
        )

        self.assertIsNotNone(primed_recovered_batch)
        self.assertIsNotNone(fresh_batch)
        self.assertGreater(
            float(first_batch["critic_contract_bootstrap_late_gate"]),
            0.99,
        )
        self.assertGreater(
            float(primed_recovered_batch["critic_contract_bootstrap_late_gate"]),
            0.99,
        )
        self.assertGreater(
            float(fresh_batch["critic_contract_bootstrap_late_gate"]),
            0.99,
        )
        self.assertLess(
            float(primed_recovered_batch["critic_contract_task_degradation"]),
            0.1,
        )
        self.assertLess(
            float(fresh_batch["critic_contract_task_degradation"]),
            0.1,
        )
        self.assertGreaterEqual(
            float(primed_recovered_batch["critic_contract_bootstrap_state"]),
            float(fresh_batch["critic_contract_bootstrap_state"]),
        )
        self.assertGreaterEqual(
            float(primed_recovered_batch["critic_contract_bootstrap_semantic_debt_mean"]),
            float(fresh_batch["critic_contract_bootstrap_semantic_debt_mean"]),
        )
        self.assertGreater(
            float(primed_recovered_batch["critic_contract_bootstrap_surface_state"]),
            float(fresh_batch["critic_contract_bootstrap_surface_state"]),
        )
        self.assertGreaterEqual(
            float(primed_recovered_batch["critic_contract_bootstrap_contact_drive_mean"]),
            float(primed_recovered_batch["critic_contract_authority_mean"]),
        )
        self.assertGreaterEqual(
            float(primed_recovered_batch["critic_contract_bootstrap_target_delta_mean"]),
            float(fresh_batch["critic_contract_bootstrap_target_delta_mean"]),
        )

    def test_train_step_prefers_bootstrap_rewritten_critic_target_branch_when_available(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )
        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        metrics = loop.train_step(
            rl_batch=batch,
            source_tag="imag",
            imag_ratio=1.0,
        )

        self.assertAlmostEqual(
            float(metrics["actor/target_mean"]),
            float(batch["target_actor"].mean()),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/target_mean"]),
            float(batch["target_critic"].mean()),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/target_uses_separate_branch"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/critic_contract_bootstrap_enabled"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/critic_contract_bootstrap_late_gate"]),
            float(batch["critic_contract_bootstrap_late_gate"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/critic_contract_bootstrap_clean_mix_mean"]),
            float(batch["critic_contract_bootstrap_clean_mix_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/critic_contract_bootstrap_target_delta_mean"]),
            float(batch["critic_contract_bootstrap_target_delta_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/bootstrap_trust_mean"]),
            float(metrics["critic/critic_contract_trust_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/bootstrap_late_gate"]),
            float(metrics["critic/critic_contract_bootstrap_late_gate"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/bootstrap_clean_mix_mean"]),
            float(metrics["critic/critic_contract_bootstrap_clean_mix_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/bootstrap_mix_surface_mean"]),
            float(metrics["critic/critic_contract_bootstrap_mix_surface_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/bootstrap_effective_contact_mean"]),
            float(metrics["critic/critic_contract_bootstrap_effective_contact_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["critic/bootstrap_raw_vs_clean_gap_mean"]),
            float(metrics["critic/critic_contract_bootstrap_raw_vs_clean_gap_mean"]),
            places=6,
        )

    def test_build_imagined_batch_emits_phase1b_semantic_contract_observation_summaries(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )
        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        clean_summary = batch["clean_anchor_contract_summary"]
        corridor_summary = batch["corridor_contract_summary"]
        bootstrap_internal_summary = batch["bootstrap_internal_contract_summary"]
        bootstrap_external_summary = batch["bootstrap_external_contract_summary"]

        self.assertEqual(clean_summary["source"], "replay_suffix")
        self.assertEqual(corridor_summary["source"], "task_corridor")
        self.assertEqual(bootstrap_internal_summary["source"], "critic_bootstrap")
        self.assertEqual(bootstrap_external_summary["source"], "replay_suffix")
        self.assertAlmostEqual(
            float(clean_summary["coverage_mean"]),
            float(batch["corridor_semantic_clean_target_available_fraction"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(corridor_summary["task_agreement_mean"]),
            float(batch["behavior_policy_task_cert_imag_gate_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(bootstrap_internal_summary["trust_mean"]),
            float(batch["critic_contract_trust_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(bootstrap_external_summary["authority_mean"]),
            float(batch["bootstrap_external_authority_mean"]),
            places=6,
        )

        metrics = loop.train_step(
            rl_batch=batch,
            source_tag="imag",
            imag_ratio=1.0,
        )

        self.assertAlmostEqual(
            float(metrics["contract/clean_anchor_coverage_mean"]),
            float(clean_summary["coverage_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["contract/corridor_task_agreement_mean"]),
            float(corridor_summary["task_agreement_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["contract/bootstrap_internal_trust_mean"]),
            float(bootstrap_internal_summary["trust_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["contract/bootstrap_external_authority_mean"]),
            float(bootstrap_external_summary["authority_mean"]),
            places=6,
        )

    def test_semantic_contracts_keep_raw_task_corridor_gate_when_imag_task_cert_is_inflated(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )

        def _stub_task_corridor(_self, imag_policy_features, reference_real_batch, corridor_mask):
            del imag_policy_features, reference_real_batch
            gate = torch.full_like(corridor_mask, 0.4)
            confidence = torch.full_like(corridor_mask, 0.1)
            zeros = torch.zeros_like(corridor_mask)
            ones = torch.ones_like(corridor_mask)
            return {
                "score": zeros,
                "gate": gate,
                "analytic_scale": ones,
                "confidence": confidence,
                "metrics": {
                    "task_corridor_score_mean": 0.0,
                    "task_corridor_gate_mean": 0.4,
                    "task_corridor_active_fraction": 0.0,
                    "task_corridor_confidence_mean": 0.1,
                    "task_corridor_high_return_similarity_mean": 0.0,
                    "task_corridor_low_return_similarity_mean": 0.0,
                    "task_corridor_real_separation": 0.025,
                    "task_corridor_score_threshold": 0.0,
                    "task_corridor_analytic_scale_mean": 1.0,
                    "high_value_but_low_task_fraction": 0.0,
                    "task_geom_corridor_disagreement": 0.0,
                },
            }

        loop._compute_task_corridor_signals = MethodType(  # type: ignore[method-assign]
            _stub_task_corridor,
            loop,
        )

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertLess(float(batch["task_corridor_gate_mean"]), 0.5)
        self.assertGreater(float(batch["behavior_policy_task_cert_imag_gate_mean"]), 0.9)
        self.assertAlmostEqual(
            float(batch["clean_anchor_contract_summary"]["task_agreement_mean"]),
            float(batch["task_corridor_gate_mean"]),
            places=6,
        )
        self.assertFalse(bool(batch["clean_anchor_contract_summary"]["task_certified"]))
        self.assertFalse(bool(batch["bootstrap_external_contract_summary"]["task_certified"]))

    def test_bootstrap_contract_stays_dormant_before_late_stage_gate(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            global_step=200,
            best_eval=80.0,
            last_eval=70.0,
            bootstrap_min_step=1000,
            bootstrap_step_ramp=250,
            bootstrap_eval_threshold=200.0,
            bootstrap_eval_ramp=50.0,
        )

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_late_gate"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_precontact_gate"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_state"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_clean_mix_mean"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_surface_floor_mean"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_surface_state"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_target_delta_mean"]),
            0.0,
            places=6,
        )

    def test_bootstrap_trigger_surface_can_open_without_eval_gate_once_time_ready(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            global_step=1300,
            best_eval=0.0,
            last_eval=0.0,
            bootstrap_min_step=1000,
            bootstrap_step_ramp=250,
            bootstrap_eval_threshold=200.0,
            bootstrap_eval_ramp=50.0,
        )

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_eval_gate"]),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_negative_adv_pressure_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_trigger_surface_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_trigger_gate"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_late_gate"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_precontact_gate"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_clean_mix_mean"]),
            0.0,
        )

    def test_bootstrap_precontact_floor_engages_before_late_gate_opens(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            global_step=1100,
            best_eval=500.0,
            last_eval=150.0,
            bootstrap_min_step=1200,
            bootstrap_step_ramp=200,
            bootstrap_eval_threshold=200.0,
            bootstrap_eval_ramp=50.0,
        )
        loop.config.adaptive_imag_critic_bootstrap_precontact_floor_max = 0.3

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_late_gate"]),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_precontact_gate"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_precontact_need_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_precontact_floor_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_clean_mix_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_budget_utilization_mean"]),
            0.75,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_precontact_anchor_effective_contact_mean"]),
            0.0,
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_precontact_dense_effective_contact_mean"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_dense_floor_budget_mean"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_floor_leak_to_dense_mean"]),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_precontact_anchor_effective_contact_mean"]),
            float(batch["critic_contract_bootstrap_precontact_dense_effective_contact_mean"]),
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_target_delta_mean"]),
            0.0,
        )

    def test_bootstrap_precontact_gap_drive_boosts_floor_before_late_gate_opens(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            global_step=1100,
            best_eval=500.0,
            last_eval=150.0,
            bootstrap_min_step=1200,
            bootstrap_step_ramp=200,
            bootstrap_eval_threshold=200.0,
            bootstrap_eval_ramp=50.0,
        )
        loop.config.adaptive_imag_critic_bootstrap_precontact_floor_max = 0.3
        loop.imagination_engine = _BadQualityLowPressureImaginationEngine(device)

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_late_gate"]),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_precontact_gate"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_release_guard_mean"]),
            0.0,
        )
        self.assertGreaterEqual(
            float(batch["critic_contract_bootstrap_precontact_gap_drive_mean"]),
            float(batch["critic_contract_bootstrap_contact_drive_mean"]),
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_precontact_need_mean"]),
            float(batch["critic_contract_bootstrap_precontact_gap_drive_mean"]),
            places=6,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_precontact_floor_mean"]),
            float(batch["critic_contract_bootstrap_precontact_gate"])
            * loop.config.adaptive_imag_critic_bootstrap_precontact_floor_max
            * float(batch["critic_contract_bootstrap_contact_drive_mean"]),
        )

    def test_low_real_task_cert_raises_bootstrap_external_floor_and_caps_internal_authority(self):
        device = torch.device("cpu")

        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }

        def _build_batch(value_real):
            loop = _prime_critic_bootstrap_contract_loop(
                _make_actor_contract_trust_test_loop(
                    device,
                    drift_actor=False,
                ),
                enabled=True,
                global_step=1100,
                best_eval=220.0,
                last_eval=220.0,
                bootstrap_min_step=1200,
                bootstrap_step_ramp=200,
                bootstrap_eval_threshold=200.0,
                bootstrap_eval_ramp=50.0,
            )
            loop.set_external_eval_feedback(
                220.0,
                step=100,
                telemetry=dict(telemetry),
            )
            loop.config.adaptive_imag_critic_bootstrap_precontact_floor_max = 0.3
            loop.imagination_engine = _ContractFeatureImaginationEngine(
                device,
                policy_features=[
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                ],
                rewards=[[0.0, 0.0]],
                values=[[6.0, 6.0, 6.0]],
                continue_probs=[[1.0, 1.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            reference_real_batch = {
                "vitals": torch.zeros((1, 2, 4), device=device),
                "policy_features": torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
                    device=device,
                ),
                "use_precomputed_policy_outputs": True,
                "actions": torch.tensor(
                    [[[1.0, 0.0], [0.0, 1.0]]],
                    device=device,
                ),
                "value_real": torch.tensor([value_real], device=device),
            }
            batch = loop._build_imagined_batch(
                reference_real_batch=reference_real_batch
            )
            self.assertIsNotNone(batch)
            return batch

        healthy_batch = _build_batch([220.0, 220.0])
        degraded_batch = _build_batch([20.0, 0.0])

        self.assertAlmostEqual(
            float(degraded_batch["behavior_policy_task_cert_real_eval_gate_mean"]),
            1.0,
            places=6,
        )
        self.assertGreater(
            float(healthy_batch["behavior_policy_task_cert_real_gate_mean"]),
            0.9,
        )
        self.assertLess(
            float(degraded_batch["behavior_policy_task_cert_real_reward_agreement_mean"]),
            0.1,
        )
        self.assertLess(
            float(degraded_batch["behavior_policy_task_cert_real_gate_mean"]),
            0.25,
        )
        self.assertGreater(
            float(degraded_batch["critic_contract_bootstrap_task_cert_takeover_floor_mean"]),
            float(healthy_batch["critic_contract_bootstrap_task_cert_takeover_floor_mean"]),
        )
        self.assertGreater(
            float(degraded_batch["critic_contract_bootstrap_task_cert_takeover_floor_mean"]),
            float(degraded_batch["critic_contract_bootstrap_precontact_floor_mean"]),
        )
        self.assertGreater(
            float(degraded_batch["critic_contract_bootstrap_requested_floor_mean"]),
            float(healthy_batch["critic_contract_bootstrap_requested_floor_mean"]),
        )
        self.assertGreater(
            float(
                degraded_batch[
                    "critic_contract_bootstrap_requested_floor_on_valid_mean"
                ]
            ),
            float(
                healthy_batch[
                    "critic_contract_bootstrap_requested_floor_on_valid_mean"
                ]
            ),
        )
        self.assertAlmostEqual(
            float(
                degraded_batch[
                    "critic_contract_bootstrap_floor_shortfall_on_valid_mean"
                ]
            ),
            0.0,
            places=6,
        )
        self.assertGreaterEqual(
            float(
                degraded_batch[
                    "critic_contract_bootstrap_anchor_valid_effective_contact_mean"
                ]
            ),
            float(
                degraded_batch[
                    "critic_contract_bootstrap_requested_floor_on_valid_mean"
                ]
            )
            - 1e-6,
        )

    def test_bootstrap_final_effective_contact_preserves_requested_floor_on_valid_positions(self):
        device = torch.device("cpu")

        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            global_step=1100,
            best_eval=220.0,
            last_eval=220.0,
            bootstrap_min_step=1200,
            bootstrap_step_ramp=200,
            bootstrap_eval_threshold=200.0,
            bootstrap_eval_ramp=50.0,
        )
        loop.set_external_eval_feedback(
            220.0,
            step=100,
            telemetry=dict(telemetry),
        )
        loop.config.adaptive_imag_critic_bootstrap_precontact_floor_max = 0.3
        loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ],
            rewards=[[0.0, 0.0]],
            values=[[6.0, 6.0, 6.0]],
            continue_probs=[[1.0, 1.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[20.0, 0.0]], device=device),
        }

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        valid_mask = batch["critic_contract_bootstrap_anchor_valid_mask"] > 0.0
        self.assertTrue(torch.any(valid_mask))

        requested_floor = batch["critic_contract_bootstrap_clean_mix"][valid_mask]
        effective_contact = batch["critic_contract_bootstrap_mix_surface"][valid_mask]

        self.assertGreater(
            float(requested_floor.mean().item()),
            0.2,
        )
        self.assertTrue(
            torch.all(effective_contact >= requested_floor - 1e-6)
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_floor_shortfall_on_valid_mean"]),
            0.0,
            places=6,
        )
        self.assertGreaterEqual(
            float(batch["critic_contract_bootstrap_anchor_valid_effective_contact_mean"]),
            float(batch["critic_contract_bootstrap_requested_floor_on_valid_mean"])
            - 1e-6,
        )

    def test_bootstrap_precontact_floor_allocates_to_anchor_before_dense(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            global_step=1100,
            best_eval=500.0,
            last_eval=150.0,
            bootstrap_min_step=1200,
            bootstrap_step_ramp=200,
            bootstrap_eval_threshold=200.0,
            bootstrap_eval_ramp=50.0,
        )
        loop.config.adaptive_imag_critic_bootstrap_precontact_floor_max = 0.3

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_late_gate"]),
            0.0,
            places=6,
        )
        self.assertGreaterEqual(
            float(batch["critic_contract_bootstrap_requested_floor_mean"]),
            float(batch["critic_contract_bootstrap_base_mix_surface_mean"]),
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_floor_budget_mean"]),
            0.0,
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_dense_floor_budget_mean"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["critic_contract_bootstrap_floor_leak_to_dense_mean"]),
            0.0,
            places=6,
        )

    def test_bootstrap_task_cert_revocation_raises_takeover_floor_and_caps_internal_authority(self):
        device = torch.device("cpu")
        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[20.0, 20.0]], device=device),
        }

        healthy_loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            best_eval=20.0,
            last_eval=20.0,
        )
        revoked_loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            best_eval=500.0,
            last_eval=220.0,
        )
        healthy_loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            rewards=[[0.0, 0.0]],
            values=[[6.0, 6.0, 6.0]],
            continue_probs=[[1.0, 1.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        revoked_loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            rewards=[[0.0, 0.0]],
            values=[[6.0, 6.0, 6.0]],
            continue_probs=[[1.0, 1.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        healthy_loop.set_external_eval_feedback(
            20.0,
            step=100,
            telemetry=dict(telemetry),
        )
        revoked_loop.set_external_eval_feedback(
            220.0,
            step=100,
            telemetry=dict(telemetry),
        )

        healthy_batch = healthy_loop._build_imagined_batch(
            reference_real_batch=reference_real_batch
        )
        revoked_batch = revoked_loop._build_imagined_batch(
            reference_real_batch=reference_real_batch
        )

        self.assertIsNotNone(healthy_batch)
        self.assertIsNotNone(revoked_batch)
        self.assertGreater(
            float(healthy_batch["behavior_policy_task_cert_real_gate_mean"]),
            0.95,
        )
        self.assertLess(
            float(revoked_batch["behavior_policy_task_cert_real_gate_mean"]),
            0.2,
        )
        self.assertGreater(
            float(
                revoked_batch[
                    "critic_contract_bootstrap_task_cert_revocation_mean"
                ]
            ),
            0.8,
        )
        self.assertGreater(
            float(
                revoked_batch[
                    "critic_contract_bootstrap_task_cert_takeover_floor_mean"
                ]
            ),
            float(
                healthy_batch[
                    "critic_contract_bootstrap_task_cert_takeover_floor_mean"
                ]
            )
            + 0.03,
        )
        self.assertGreaterEqual(
            float(revoked_batch["critic_contract_bootstrap_requested_floor_mean"]),
            float(
                revoked_batch[
                    "critic_contract_bootstrap_task_cert_takeover_floor_mean"
                ]
            )
            - 1e-6,
        )
        self.assertAlmostEqual(
            float(revoked_batch["bootstrap_external_authority_shortfall_on_valid_mean"]),
            0.0,
            places=6,
        )
        self.assertGreaterEqual(
            float(revoked_batch["bootstrap_external_authority_on_valid_mean"]),
            float(revoked_batch["bootstrap_external_authority_floor_on_valid_mean"])
            - 1e-6,
        )
        self.assertLess(
            float(
                revoked_batch[
                    "critic_contract_bootstrap_task_cert_internal_cap_mean"
                ]
            ),
            float(
                healthy_batch[
                    "critic_contract_bootstrap_task_cert_internal_cap_mean"
                ]
            )
            - 0.03,
        )
        self.assertGreater(
            float(
                revoked_batch[
                    "critic_contract_bootstrap_task_request_alarm_mean"
                ]
            ),
            float(
                healthy_batch[
                    "critic_contract_bootstrap_task_request_alarm_mean"
                ]
            ),
        )
        self.assertGreater(
            float(
                revoked_batch[
                    "critic_contract_bootstrap_task_request_floor_mean"
                ]
            ),
            float(
                healthy_batch[
                    "critic_contract_bootstrap_task_request_floor_mean"
                ]
            ),
        )
        self.assertGreaterEqual(
            float(
                revoked_batch[
                    "critic_contract_bootstrap_task_request_floor_on_valid_debt_high_mean"
                ]
            ),
            float(
                revoked_batch[
                    "critic_contract_bootstrap_task_request_floor_on_valid_mean"
                ]
            ),
        )

    def test_bootstrap_task_request_contract_is_formula_driven_and_coverage_free(self):
        real_task_alarm = torch.tensor([[0.2, 0.7]], dtype=torch.float32)
        task_support_mask = torch.tensor([[0.4, 1.0]], dtype=torch.float32)
        imag_task_gate = torch.tensor([[0.3, 0.9]], dtype=torch.float32)
        task_gate = torch.tensor([[0.6, 0.8]], dtype=torch.float32)
        negative_adv_pressure = torch.tensor([[0.5, 0.1]], dtype=torch.float32)
        release_guard = torch.tensor([[0.2, 0.3]], dtype=torch.float32)
        task_degradation = torch.tensor([[0.4, 0.2]], dtype=torch.float32)
        corridor_mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        semantic_debt = torch.tensor([[0.9, 0.6]], dtype=torch.float32)

        alarm, disagreement, scope, floor = _compute_bootstrap_task_request_contract(
            real_task_alarm=real_task_alarm,
            task_support_mask=task_support_mask,
            imag_task_gate=imag_task_gate,
            task_gate=task_gate,
            negative_adv_pressure=negative_adv_pressure,
            release_guard=release_guard,
            task_degradation=task_degradation,
            corridor_mask=corridor_mask,
            semantic_debt=semantic_debt,
            control_gate=0.75,
            clean_mix_max=0.8,
        )

        self.assertTrue(
            torch.allclose(
                alarm,
                torch.tensor([[0.8, 0.3]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                disagreement,
                torch.tensor([[0.7, 0.2]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                scope,
                torch.tensor([[1.0, 0.6]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                floor,
                torch.tensor([[0.54, 0.216]], dtype=torch.float32),
                atol=1e-6,
            )
        )

    def test_bootstrap_task_request_debt_high_metric_prioritizes_high_debt_valid_positions(self):
        task_request_floor = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
        anchor_valid_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        semantic_debt = torch.tensor([[0.9, 0.3]], dtype=torch.float32)
        debt_high_mask = anchor_valid_mask * (semantic_debt >= 0.5).to(
            dtype=anchor_valid_mask.dtype
        )

        on_valid = float(
            _masked_mean_tensor(task_request_floor, anchor_valid_mask).item()
        )
        on_valid_debt_high = float(
            _masked_mean_tensor(task_request_floor, debt_high_mask).item()
        )

        self.assertGreater(on_valid_debt_high, on_valid)

    def test_bootstrap_task_request_floor_mean_does_not_depend_on_anchor_coverage(self):
        device = torch.device("cpu")
        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[20.0, 20.0]], device=device),
        }

        def _build_batch(valid):
            loop = _prime_critic_bootstrap_contract_loop(
                _make_actor_contract_trust_test_loop(
                    device,
                    drift_actor=False,
                ),
                enabled=True,
                best_eval=500.0,
                last_eval=220.0,
            )
            loop.imagination_engine = _ContractFeatureImaginationEngine(
                device,
                policy_features=[
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                ],
                rewards=[[0.0, 0.0]],
                values=[[6.0, 6.0, 6.0]],
                continue_probs=[[1.0, 1.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            loop.set_external_eval_feedback(
                220.0,
                step=100,
                telemetry=dict(telemetry),
            )

            def _suffix_targets(self, seed_batch, horizon):
                del seed_batch, horizon
                targets = torch.tensor([[9.0, 4.0]], device=device, dtype=torch.float32)
                valid_mask = torch.tensor([valid], device=device, dtype=torch.float32)
                return targets, valid_mask

            loop._compute_seed_replay_suffix_targets = MethodType(
                _suffix_targets,
                loop,
            )
            batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)
            self.assertIsNotNone(batch)
            return batch

        sparse_batch = _build_batch([1.0, 0.25])
        dense_batch = _build_batch([1.0, 1.0])

        self.assertLess(
            float(sparse_batch["critic_contract_bootstrap_anchor_coverage_mean"]),
            float(dense_batch["critic_contract_bootstrap_anchor_coverage_mean"]),
        )
        self.assertAlmostEqual(
            float(sparse_batch["critic_contract_bootstrap_task_request_alarm_mean"]),
            float(dense_batch["critic_contract_bootstrap_task_request_alarm_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(sparse_batch["critic_contract_bootstrap_task_request_floor_mean"]),
            float(dense_batch["critic_contract_bootstrap_task_request_floor_mean"]),
            places=6,
        )

    def test_bootstrap_external_authority_floor_contract_uses_requested_floor_on_valid_positions(self):
        requested_floor = torch.tensor([[0.6, 0.3, 0.4]], dtype=torch.float32)
        valid_mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
        semantic_debt = torch.tensor([[0.8, 0.9, 0.2]], dtype=torch.float32)
        task_request_alarm = torch.tensor([[0.7, 0.1, 0.2]], dtype=torch.float32)
        task_request_disagreement = torch.tensor(
            [[0.5, 0.2, 0.6]],
            dtype=torch.float32,
        )

        floor, floor_valid_mask, debt_high_mask = (
            _compute_bootstrap_external_authority_floor_contract(
                critic_contract_bootstrap_requested_floor=requested_floor,
                bootstrap_anchor_valid_mask=valid_mask,
                critic_contract_bootstrap_semantic_debt=semantic_debt,
                critic_contract_bootstrap_task_request_alarm=task_request_alarm,
                critic_contract_bootstrap_task_request_disagreement=task_request_disagreement,
            )
        )

        self.assertTrue(
            torch.allclose(
                floor,
                torch.tensor([[0.6, 0.0, 0.4]], dtype=torch.float32),
            )
        )
        self.assertTrue(torch.equal(floor_valid_mask, valid_mask))
        self.assertTrue(
            torch.equal(
                debt_high_mask,
                torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            )
        )

    def test_bootstrap_external_authority_takeover_floor_contract_restores_valid_positions_only(self):
        requested_floor = torch.tensor([[0.6, 0.4, 0.5]], dtype=torch.float32)
        current_floor = torch.tensor([[0.15, 0.20, 0.0]], dtype=torch.float32)
        valid_mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

        takeover_floor, enforced_floor, enforcement_delta = (
            _compute_bootstrap_external_authority_takeover_floor_contract(
                bootstrap_external_authority_floor=current_floor,
                bootstrap_anchor_valid_mask=valid_mask,
                critic_contract_bootstrap_requested_floor=requested_floor,
                external_takeover_floor_ratio=0.5,
                max_external_authority=0.8,
            )
        )

        self.assertTrue(
            torch.allclose(
                takeover_floor,
                torch.tensor([[0.3, 0.0, 0.25]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                enforced_floor,
                torch.tensor([[0.3, 0.2, 0.25]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                enforcement_delta,
                torch.tensor([[0.15, 0.0, 0.25]], dtype=torch.float32),
            )
        )

    def test_bootstrap_external_authority_floor_mean_does_not_depend_on_anchor_coverage(self):
        device = torch.device("cpu")
        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[20.0, 20.0]], device=device),
        }

        def _build_batch(valid):
            loop = _prime_critic_bootstrap_contract_loop(
                _make_actor_contract_trust_test_loop(
                    device,
                    drift_actor=False,
                ),
                enabled=True,
                best_eval=500.0,
                last_eval=220.0,
            )
            loop.imagination_engine = _ContractFeatureImaginationEngine(
                device,
                policy_features=[
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                ],
                rewards=[[0.0, 0.0]],
                values=[[6.0, 6.0, 6.0]],
                continue_probs=[[1.0, 1.0]],
                entropies=[[0.0, 0.0]],
                log_probs=[[0.0, 0.0]],
                actions=[[[1.0, 0.0], [1.0, 0.0]]],
            )
            loop.set_external_eval_feedback(
                220.0,
                step=100,
                telemetry=dict(telemetry),
            )

            def _suffix_targets(self, seed_batch, horizon):
                del seed_batch, horizon
                targets = torch.tensor([[9.0, 4.0]], device=device, dtype=torch.float32)
                valid_mask = torch.tensor([valid], device=device, dtype=torch.float32)
                return targets, valid_mask

            loop._compute_seed_replay_suffix_targets = MethodType(
                _suffix_targets,
                loop,
            )
            batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)
            self.assertIsNotNone(batch)
            return batch

        sparse_batch = _build_batch([1.0, 0.25])
        dense_batch = _build_batch([1.0, 1.0])

        self.assertLess(
            float(sparse_batch["critic_contract_bootstrap_anchor_coverage_mean"]),
            float(dense_batch["critic_contract_bootstrap_anchor_coverage_mean"]),
        )
        self.assertAlmostEqual(
            float(sparse_batch["bootstrap_external_authority_floor_mean"]),
            float(dense_batch["bootstrap_external_authority_floor_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(sparse_batch["bootstrap_external_authority_floor_on_valid_mean"]),
            float(dense_batch["bootstrap_external_authority_floor_on_valid_mean"]),
            places=6,
        )

    def test_bootstrap_external_authority_overlay_preserves_valid_floor(self):
        device = torch.device("cpu")
        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[20.0, 20.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            best_eval=500.0,
            last_eval=220.0,
        )
        loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ],
            rewards=[[0.0, 0.0]],
            values=[[6.0, 6.0, 6.0]],
            continue_probs=[[1.0, 1.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        loop.set_external_eval_feedback(
            220.0,
            step=100,
            telemetry=dict(telemetry),
        )

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(
            float(batch["bootstrap_external_authority_shortfall_on_valid_mean"]),
            0.0,
            places=6,
        )
        self.assertGreaterEqual(
            float(batch["bootstrap_external_authority_on_valid_mean"]),
            float(batch["bootstrap_external_authority_floor_on_valid_mean"]) - 1e-6,
        )
        self.assertGreaterEqual(
            float(batch["bootstrap_external_authority_on_valid_debt_high_mean"]),
            float(batch["bootstrap_external_authority_on_valid_mean"]),
        )

    def test_bootstrap_takeover_floor_survives_live_unwind_and_reaches_mixed_target(self):
        device = torch.device("cpu")
        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[20.0, 20.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            best_eval=500.0,
            last_eval=220.0,
        )
        loop.config.adaptive_imag_critic_bootstrap_external_takeover_floor_ratio = 1.0
        loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ],
            rewards=[[0.0, 0.0]],
            values=[[6.0, 6.0, 6.0]],
            continue_probs=[[1.0, 1.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        loop.set_external_eval_feedback(
            220.0,
            step=100,
            telemetry=dict(telemetry),
        )

        def _forced_live_unwind(**kwargs):
            reference_floor = kwargs["bootstrap_external_authority_floor"].detach()
            requested_floor = kwargs["critic_contract_bootstrap_requested_floor"].detach()
            lowered_floor = (requested_floor * 0.25).clamp(0.0, 1.0)
            return (
                torch.ones_like(reference_floor),
                torch.full_like(reference_floor, 0.25),
                lowered_floor,
                (reference_floor - lowered_floor).abs(),
            )

        with mock.patch(
            "aletheia.aletheia_train._compute_bootstrap_external_authority_floor_live_unwind_contract",
            side_effect=_forced_live_unwind,
        ):
            batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertGreater(
            float(batch["bootstrap_external_authority_takeover_enforcement_delta_abs_mean"]),
            0.0,
        )
        self.assertGreaterEqual(
            float(batch["bootstrap_external_authority_floor_on_valid_mean"]),
            float(batch["critic_contract_bootstrap_requested_floor_on_valid_mean"]) - 1e-6,
        )
        self.assertAlmostEqual(
            float(batch["bootstrap_external_authority_shortfall_on_valid_mean"]),
            0.0,
            places=6,
        )
        valid_mask = batch["bootstrap_external_authority_floor_valid_mask"] > 0.0
        mixed_bootstrap = batch["mixed_bootstrap_values_next"][valid_mask]
        internal_value = batch["bootstrap_internal_value_view"][valid_mask]
        external_value = batch["bootstrap_external_value_final"][valid_mask]
        self.assertTrue(torch.any((mixed_bootstrap - internal_value).abs() > 1e-6).item())
        self.assertTrue(
            torch.all(
                (mixed_bootstrap - external_value).abs()
                <= (internal_value - external_value).abs() + 1e-6
            ).item()
        )

    def test_bootstrap_authority_source_replacement_is_neutral_before_late_gate_when_alignment_is_strong(self):
        raw_authority = torch.tensor([[0.8, 0.7]], dtype=torch.float32)
        floor = torch.tensor([[0.4, 0.3]], dtype=torch.float32)
        anchor_valid = torch.ones_like(raw_authority)
        raw_vs_clean_gap = torch.full_like(raw_authority, 8.0)
        poor_reward_truth = torch.full_like(raw_authority, 0.1)
        poor_recovery = torch.full_like(raw_authority, 0.15)
        healthy_support = torch.full_like(raw_authority, 0.9)
        healthy_imag_gate = torch.full_like(raw_authority, 0.92)
        healthy_task_gate = torch.full_like(raw_authority, 0.88)
        ones = torch.ones_like(raw_authority)

        (
            reward_truth,
            alignment,
            activation,
            cert_gate,
            source_replaced,
            bonus_multiplier_prerecert,
            bonus_source_certified,
            bonus_source_rejected,
        ) = _compute_bootstrap_authority_source_replacement_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_vs_clean_gap,
            reference_real_reward_agreement=poor_reward_truth,
            behavior_policy_task_cert_real_recovery=poor_recovery,
            behavior_policy_task_cert_support_mask=healthy_support,
            behavior_policy_task_cert_imag_gate=healthy_imag_gate,
            task_corridor_gate=healthy_task_gate,
            critic_contract_task_degradation=torch.full_like(raw_authority, 0.9),
            critic_contract_bootstrap_late_gate=0.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(reward_truth >= poor_recovery).item())
        self.assertTrue(torch.all(alignment >= healthy_task_gate).item())
        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(torch.allclose(cert_gate, torch.ones_like(cert_gate)))
        self.assertTrue(torch.allclose(source_replaced, raw_authority))
        self.assertTrue(
            torch.allclose(
                bonus_multiplier_prerecert,
                torch.ones_like(bonus_multiplier_prerecert),
            )
        )
        self.assertTrue(
            torch.allclose(
                bonus_source_certified,
                (raw_authority - floor).clamp(min=0.0),
            )
        )
        self.assertTrue(torch.allclose(bonus_source_rejected, torch.zeros_like(raw_authority)))

    def test_bootstrap_authority_source_replacement_suppresses_low_quality_source_before_late_gate(self):
        raw_authority = torch.tensor([[0.8, 0.7]], dtype=torch.float32)
        floor = torch.tensor([[0.4, 0.3]], dtype=torch.float32)
        anchor_valid = torch.zeros_like(raw_authority)
        raw_vs_clean_gap = torch.full_like(raw_authority, 8.0)
        poor_reward_truth = torch.full_like(raw_authority, 0.1)
        poor_recovery = torch.full_like(raw_authority, 0.15)
        poor_support = torch.full_like(raw_authority, 0.2)
        poor_imag_gate = torch.full_like(raw_authority, 0.2)
        poor_task_gate = torch.full_like(raw_authority, 0.25)
        ones = torch.ones_like(raw_authority)

        (
            reward_truth,
            alignment,
            activation,
            cert_gate,
            source_replaced,
            bonus_multiplier_prerecert,
            bonus_source_certified,
            bonus_source_rejected,
        ) = _compute_bootstrap_authority_source_replacement_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_vs_clean_gap,
            reference_real_reward_agreement=poor_reward_truth,
            behavior_policy_task_cert_real_recovery=poor_recovery,
            behavior_policy_task_cert_support_mask=poor_support,
            behavior_policy_task_cert_imag_gate=poor_imag_gate,
            task_corridor_gate=poor_task_gate,
            critic_contract_task_degradation=torch.full_like(raw_authority, 0.9),
            critic_contract_bootstrap_late_gate=0.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(reward_truth >= poor_recovery).item())
        self.assertTrue(torch.all(alignment >= poor_task_gate).item())
        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(torch.allclose(cert_gate, torch.ones_like(cert_gate)))
        self.assertTrue(torch.all(source_replaced >= floor).item())
        self.assertTrue(torch.all(source_replaced < raw_authority).item())
        self.assertTrue(torch.all(bonus_multiplier_prerecert < 0.4).item())
        self.assertTrue(torch.all(bonus_source_certified > 0.0).item())
        self.assertTrue(torch.all(bonus_source_rejected > 0.0).item())

    def test_bootstrap_authority_source_replacement_softly_limits_anchor_valid_bad_bonus_before_late_gate(self):
        raw_authority = torch.tensor([[0.82, 0.74]], dtype=torch.float32)
        floor = torch.tensor([[0.42, 0.34]], dtype=torch.float32)
        anchor_valid = torch.ones_like(raw_authority)
        raw_vs_clean_gap = torch.full_like(raw_authority, 12.0)
        poor_reward_truth = torch.full_like(raw_authority, 0.1)
        poor_recovery = torch.full_like(raw_authority, 0.15)
        support = torch.full_like(raw_authority, 0.85)
        poor_imag_gate = torch.full_like(raw_authority, 0.25)
        poor_task_gate = torch.full_like(raw_authority, 0.3)
        ones = torch.ones_like(raw_authority)

        (
            _reward_truth,
            _alignment,
            activation,
            cert_gate,
            source_replaced,
            bonus_multiplier_prerecert,
            bonus_source_certified,
            bonus_source_rejected,
        ) = _compute_bootstrap_authority_source_replacement_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_vs_clean_gap,
            reference_real_reward_agreement=poor_reward_truth,
            behavior_policy_task_cert_real_recovery=poor_recovery,
            behavior_policy_task_cert_support_mask=support,
            behavior_policy_task_cert_imag_gate=poor_imag_gate,
            task_corridor_gate=poor_task_gate,
            critic_contract_task_degradation=torch.full_like(raw_authority, 0.9),
            critic_contract_bootstrap_late_gate=0.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(torch.allclose(cert_gate, torch.ones_like(cert_gate)))
        self.assertTrue(torch.all(source_replaced >= floor).item())
        self.assertTrue(torch.all(source_replaced < raw_authority).item())
        self.assertTrue(torch.all(bonus_multiplier_prerecert < 0.75).item())
        self.assertTrue(torch.all(bonus_multiplier_prerecert > 0.2).item())
        self.assertTrue(torch.all(bonus_source_certified > 0.0).item())
        self.assertTrue(torch.all(bonus_source_rejected > 0.0).item())

    def test_bootstrap_authority_source_replacement_preserves_floor_under_late_rejection(self):
        raw_authority = torch.tensor([[0.78, 0.66]], dtype=torch.float32)
        floor = torch.tensor([[0.45, 0.35]], dtype=torch.float32)
        anchor_valid = torch.zeros_like(raw_authority)
        raw_vs_clean_gap = torch.full_like(raw_authority, 8.0)
        poor_reward_truth = torch.full_like(raw_authority, 0.1)
        poor_recovery = torch.full_like(raw_authority, 0.12)
        poor_support = torch.full_like(raw_authority, 0.3)
        poor_imag_gate = torch.full_like(raw_authority, 0.2)
        poor_task_gate = torch.full_like(raw_authority, 0.25)
        ones = torch.ones_like(raw_authority)

        (
            _truth,
            _alignment,
            activation,
            cert_gate,
            source_replaced,
            bonus_multiplier_prerecert,
            bonus_source_certified,
            bonus_source_rejected,
        ) = _compute_bootstrap_authority_source_replacement_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_vs_clean_gap,
            reference_real_reward_agreement=poor_reward_truth,
            behavior_policy_task_cert_real_recovery=poor_recovery,
            behavior_policy_task_cert_support_mask=poor_support,
            behavior_policy_task_cert_imag_gate=poor_imag_gate,
            task_corridor_gate=poor_task_gate,
            critic_contract_task_degradation=torch.full_like(raw_authority, 0.8),
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(torch.all(cert_gate <= 0.25).item())
        self.assertTrue(torch.all(source_replaced >= floor).item())
        self.assertTrue(torch.all(source_replaced <= raw_authority).item())
        self.assertTrue(torch.all(bonus_multiplier_prerecert <= cert_gate).item())
        self.assertTrue(torch.all(bonus_source_certified < bonus_source_rejected).item())

    def test_bootstrap_authority_source_replacement_responds_to_reward_semantic_truth(self):
        raw_authority = torch.tensor([[0.8, 0.8]], dtype=torch.float32)
        floor = torch.tensor([[0.4, 0.4]], dtype=torch.float32)
        anchor_valid = torch.ones_like(raw_authority)
        raw_vs_clean_gap = torch.full_like(raw_authority, 8.0)
        recovery = torch.full_like(raw_authority, 0.5)
        support = torch.full_like(raw_authority, 0.9)
        imag_gate = torch.full_like(raw_authority, 0.9)
        task_gate = torch.full_like(raw_authority, 0.6)
        ones = torch.ones_like(raw_authority)

        (
            low_truth,
            low_alignment,
            low_activation,
            low_gate,
            low_source_replaced,
            low_bonus_multiplier_prerecert,
            _low_bonus_source_certified,
            low_bonus_source_rejected,
        ) = _compute_bootstrap_authority_source_replacement_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_vs_clean_gap,
            reference_real_reward_agreement=torch.full_like(raw_authority, 0.2),
            behavior_policy_task_cert_real_recovery=recovery,
            behavior_policy_task_cert_support_mask=support,
            behavior_policy_task_cert_imag_gate=imag_gate,
            task_corridor_gate=task_gate,
            critic_contract_task_degradation=torch.zeros_like(raw_authority),
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )
        (
            high_truth,
            high_alignment,
            high_activation,
            high_gate,
            high_source_replaced,
            high_bonus_multiplier_prerecert,
            _high_bonus_source_certified,
            high_bonus_source_rejected,
        ) = _compute_bootstrap_authority_source_replacement_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_vs_clean_gap,
            reference_real_reward_agreement=torch.full_like(raw_authority, 0.9),
            behavior_policy_task_cert_real_recovery=recovery,
            behavior_policy_task_cert_support_mask=support,
            behavior_policy_task_cert_imag_gate=imag_gate,
            task_corridor_gate=task_gate,
            critic_contract_task_degradation=torch.zeros_like(raw_authority),
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(low_truth < high_truth).item())
        self.assertTrue(torch.allclose(low_alignment, high_alignment))
        self.assertTrue(torch.allclose(low_activation, torch.ones_like(low_activation)))
        self.assertTrue(torch.allclose(high_activation, torch.ones_like(high_activation)))
        self.assertTrue(torch.all(low_gate < high_gate).item())
        self.assertTrue(
            torch.all(low_bonus_multiplier_prerecert < high_bonus_multiplier_prerecert).item()
        )
        self.assertTrue(torch.all(low_source_replaced < high_source_replaced).item())
        self.assertTrue(
            torch.all(low_bonus_source_rejected > high_bonus_source_rejected).item()
        )

    def test_bootstrap_authority_source_replacement_preserves_bonus_for_reward_true_source(self):
        raw_authority = torch.tensor([[0.74, 0.68]], dtype=torch.float32)
        floor = torch.tensor([[0.35, 0.3]], dtype=torch.float32)
        anchor_valid = torch.ones_like(raw_authority)
        raw_vs_clean_gap = torch.full_like(raw_authority, 8.0)
        healthy_reward_truth = torch.full_like(raw_authority, 0.95)
        healthy_recovery = torch.full_like(raw_authority, 0.94)
        healthy_support = torch.full_like(raw_authority, 0.97)
        healthy_imag_gate = torch.full_like(raw_authority, 0.96)
        healthy_task_gate = torch.full_like(raw_authority, 0.92)
        ones = torch.ones_like(raw_authority)

        (
            truth,
            alignment,
            activation,
            cert_gate,
            source_replaced,
            bonus_multiplier_prerecert,
            bonus_source_certified,
            bonus_source_rejected,
        ) = _compute_bootstrap_authority_source_replacement_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_vs_clean_gap,
            reference_real_reward_agreement=healthy_reward_truth,
            behavior_policy_task_cert_real_recovery=healthy_recovery,
            behavior_policy_task_cert_support_mask=healthy_support,
            behavior_policy_task_cert_imag_gate=healthy_imag_gate,
            task_corridor_gate=healthy_task_gate,
            critic_contract_task_degradation=torch.zeros_like(raw_authority),
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth > 0.94).item())
        self.assertTrue(torch.all(alignment > 0.9).item())
        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(torch.all(cert_gate > 0.9).item())
        self.assertTrue(torch.all(bonus_multiplier_prerecert > 0.8).item())
        self.assertTrue(torch.allclose(source_replaced, raw_authority, atol=5e-2))
        self.assertTrue(torch.all(bonus_source_rejected < 5e-2).item())
        self.assertTrue(torch.all(bonus_source_certified > 0.0).item())

    def test_bootstrap_authority_source_replacement_preserves_late_coverage_with_low_truth(self):
        raw_authority = torch.tensor([[0.78, 0.72]], dtype=torch.float32)
        floor = torch.tensor([[0.13, 0.12]], dtype=torch.float32)
        anchor_valid = torch.zeros_like(raw_authority)
        raw_vs_clean_gap = torch.full_like(raw_authority, 8.0)
        late_reward_truth = torch.full_like(raw_authority, 0.04)
        late_recovery = torch.full_like(raw_authority, 0.03)
        late_support = torch.full_like(raw_authority, 0.22)
        late_imag_gate = torch.full_like(raw_authority, 0.93)
        late_task_gate = torch.full_like(raw_authority, 0.91)
        late_degradation = torch.full_like(raw_authority, 0.95)
        ones = torch.ones_like(raw_authority)

        (
            truth,
            alignment,
            activation,
            cert_gate,
            source_replaced,
            bonus_multiplier_prerecert,
            bonus_source_certified,
            bonus_source_rejected,
        ) = _compute_bootstrap_authority_source_replacement_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_vs_clean_gap,
            reference_real_reward_agreement=late_reward_truth,
            behavior_policy_task_cert_real_recovery=late_recovery,
            behavior_policy_task_cert_support_mask=late_support,
            behavior_policy_task_cert_imag_gate=late_imag_gate,
            task_corridor_gate=late_task_gate,
            critic_contract_task_degradation=late_degradation,
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )

        strict_multiplier_without_floor = (
            late_support
            * torch.minimum(torch.maximum(late_reward_truth, late_recovery), torch.maximum(late_imag_gate, late_task_gate))
            * cert_gate
        ).clamp(0.0, 1.0)

        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(torch.all(truth < 0.05).item())
        self.assertTrue(torch.all(alignment > 0.9).item())
        self.assertTrue(torch.all(cert_gate < 0.06).item())
        self.assertTrue(torch.all(bonus_multiplier_prerecert > strict_multiplier_without_floor).item())
        self.assertTrue(torch.all(bonus_multiplier_prerecert > 0.03).item())
        self.assertTrue(torch.all(source_replaced > floor + 0.02).item())
        self.assertTrue(torch.all(bonus_source_certified > 0.02).item())
        self.assertTrue(torch.all(bonus_source_rejected > bonus_source_certified).item())

    def test_bootstrap_authority_source_replacement_interpolates_in_late_window_ramp(self):
        raw_authority = torch.tensor([[0.9, 0.7]], dtype=torch.float32)
        floor = torch.tensor([[0.4, 0.3]], dtype=torch.float32)
        anchor_valid = torch.zeros_like(raw_authority)
        raw_vs_clean_gap = torch.full_like(raw_authority, 2.0)
        reward_truth = torch.full_like(raw_authority, 0.4)
        recovery = torch.full_like(raw_authority, 0.3)
        support = torch.full_like(raw_authority, 0.6)
        imag_gate = torch.full_like(raw_authority, 0.8)
        task_gate = torch.full_like(raw_authority, 0.5)
        zeros = torch.zeros_like(raw_authority)
        ones = torch.ones_like(raw_authority)

        (
            _truth,
            _alignment,
            activation,
            cert_gate,
            source_replaced,
            bonus_multiplier_prerecert,
            bonus_source_certified,
            _bonus_source_rejected,
        ) = _compute_bootstrap_authority_source_replacement_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_vs_clean_gap,
            reference_real_reward_agreement=reward_truth,
            behavior_policy_task_cert_real_recovery=recovery,
            behavior_policy_task_cert_support_mask=support,
            behavior_policy_task_cert_imag_gate=imag_gate,
            task_corridor_gate=task_gate,
            critic_contract_task_degradation=zeros,
            critic_contract_bootstrap_late_gate=0.9,
            corridor_semantic_corridor_mask=ones,
        )

        strict_multiplier = (
            support
            * torch.minimum(torch.maximum(reward_truth, recovery), torch.maximum(imag_gate, task_gate))
            * cert_gate
        ).clamp(0.0, 1.0)
        expected_activation = torch.full_like(raw_authority, 0.5)

        self.assertTrue(torch.allclose(activation, expected_activation, atol=1e-6))
        self.assertTrue(torch.all(bonus_multiplier_prerecert < 1.0).item())
        self.assertTrue(torch.all(bonus_multiplier_prerecert > strict_multiplier).item())
        self.assertTrue(torch.all(source_replaced >= floor).item())
        self.assertTrue(
            torch.allclose(
                source_replaced,
                floor + bonus_source_certified,
                atol=1e-6,
            )
        )

    def test_bootstrap_bonus_source_value_replacement_is_neutral_before_late_window(self):
        bootstrap_external_value_raw = torch.tensor(
            [[9.0, 8.5]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[1.5, 1.0]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[12.0, 11.0]], dtype=torch.float32
        )
        anchor_available = torch.ones_like(bootstrap_external_value_raw)
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            truth,
            alignment,
            activation,
            bonus_gate,
            replaced_value,
            delta_abs,
        ) = compute_bootstrap_bonus_source_value_replacement_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_bonus_source_value=raw_anchor_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.2
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.3
            ),
            behavior_policy_task_cert_support_mask=torch.full_like(
                bootstrap_external_value_raw, 0.4
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.5
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.6),
            critic_contract_task_degradation=torch.full_like(
                bootstrap_external_value_raw, 0.9
            ),
            critic_contract_bootstrap_late_gate=0.8,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth >= 0.3).item())
        self.assertTrue(torch.all(alignment >= 0.6).item())
        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(torch.allclose(bonus_gate, torch.ones_like(bonus_gate)))
        self.assertTrue(torch.allclose(replaced_value, bootstrap_external_value_raw))
        self.assertTrue(torch.allclose(delta_abs, torch.zeros_like(delta_abs), atol=1e-5))

    def test_bootstrap_bonus_source_transition_bridge_is_neutral_before_transition_window(self):
        bootstrap_external_value_raw = torch.tensor(
            [[9.0, 8.5]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[1.5, 1.0]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[12.0, 11.0]], dtype=torch.float32
        )
        anchor_available = torch.ones_like(bootstrap_external_value_raw)
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            truth,
            alignment,
            activation,
            bridge_gate,
            bridged_value,
            delta_abs,
        ) = compute_bootstrap_bonus_source_transition_bridge_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.2
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.3
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                bootstrap_external_value_raw, 0.8
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.5
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.6),
            critic_contract_task_degradation=torch.full_like(
                bootstrap_external_value_raw, 0.9
            ),
            critic_contract_bootstrap_late_gate=0.35,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth >= 0.3).item())
        self.assertTrue(torch.all(alignment >= 0.6).item())
        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(torch.allclose(bridge_gate, torch.zeros_like(bridge_gate)))
        self.assertTrue(torch.allclose(bridged_value, bootstrap_external_value_raw))
        self.assertTrue(torch.allclose(delta_abs, torch.zeros_like(delta_abs), atol=1e-5))

    def test_bootstrap_bonus_source_transition_bridge_turns_off_once_late_window_is_owned_by_phase46(self):
        bootstrap_external_value_raw = torch.tensor(
            [[8.0, 7.5]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[10.0, 11.0]], dtype=torch.float32
        )
        anchor_available = torch.ones_like(bootstrap_external_value_raw)
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            _truth,
            _alignment,
            activation,
            bridge_gate,
            bridged_value,
            delta_abs,
        ) = compute_bootstrap_bonus_source_transition_bridge_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                bootstrap_external_value_raw, 0.01
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.95),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=0.8,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(torch.allclose(bridge_gate, torch.zeros_like(bridge_gate)))
        self.assertTrue(torch.allclose(bridged_value, bootstrap_external_value_raw))
        self.assertTrue(torch.allclose(delta_abs, torch.zeros_like(delta_abs), atol=1e-5))

    def test_bootstrap_bonus_source_transition_bridge_keeps_observer_only_inside_transition_window_when_truth_is_high(self):
        bootstrap_external_value_raw = torch.tensor(
            [[9.0, 8.0]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[12.0, 12.5]], dtype=torch.float32
        )
        anchor_available = torch.ones_like(bootstrap_external_value_raw)
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            truth,
            alignment,
            activation,
            bridge_gate,
            bridged_value,
            delta_abs,
        ) = compute_bootstrap_bonus_source_transition_bridge_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.92
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                bootstrap_external_value_raw, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.96
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.97),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=0.5,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth > 0.9).item())
        self.assertTrue(torch.all(alignment > 0.95).item())
        self.assertTrue(torch.all(activation > 0.0).item())
        self.assertTrue(torch.all(bridge_gate > 0.0).item())
        self.assertTrue(
            torch.allclose(bridged_value, bootstrap_external_value_raw)
        )
        self.assertTrue(
            torch.allclose(delta_abs, torch.zeros_like(delta_abs))
        )

    def test_bootstrap_bonus_source_transition_bridge_shrinks_under_low_truth_inside_transition_window(self):
        bootstrap_external_value_raw = torch.tensor(
            [[8.0, 7.5]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[10.0, 11.0]], dtype=torch.float32
        )
        anchor_available = torch.ones_like(bootstrap_external_value_raw)
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            _high_truth,
            _high_alignment,
            _high_activation,
            high_bridge_gate,
            _high_bridged_value,
            _high_delta_abs,
        ) = compute_bootstrap_bonus_source_transition_bridge_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.9
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                bootstrap_external_value_raw, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.96),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=0.55,
            corridor_semantic_corridor_mask=ones,
        )
        (
            _low_truth,
            _low_alignment,
            activation,
            low_bridge_gate,
            low_bridged_value,
            low_delta_abs,
        ) = compute_bootstrap_bonus_source_transition_bridge_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.05
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.08
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.1
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.12),
            critic_contract_task_degradation=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            critic_contract_bootstrap_late_gate=0.55,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(activation > 0.0).item())
        self.assertTrue(torch.all(low_bridge_gate < high_bridge_gate).item())
        self.assertTrue(torch.all(low_bridge_gate < 1e-3).item())
        self.assertTrue(torch.allclose(low_delta_abs, torch.zeros_like(low_delta_abs)))
        self.assertTrue(torch.allclose(_high_delta_abs, torch.zeros_like(_high_delta_abs)))
        self.assertTrue(
            torch.allclose(
                low_bridged_value,
                bootstrap_external_value_raw,
                atol=1e-3,
            )
        )

    def test_bootstrap_bonus_source_value_replacement_preserves_carried_value_when_truth_is_low(self):
        bootstrap_external_value_raw = torch.tensor(
            [[7.5, 6.5]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[1.0, 1.2]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[4.5, 4.0]], dtype=torch.float32
        )
        anchor_available = torch.ones_like(bootstrap_external_value_raw)
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            _truth,
            _alignment,
            activation,
            bonus_gate,
            replaced_value,
            delta_abs,
        ) = compute_bootstrap_bonus_source_value_replacement_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_bonus_source_value=raw_anchor_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.05
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.08
            ),
            behavior_policy_task_cert_support_mask=torch.full_like(
                bootstrap_external_value_raw, 0.2
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.1
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.15),
            critic_contract_task_degradation=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(torch.all(bonus_gate < 0.1).item())
        self.assertTrue(
            torch.all(
                (replaced_value - bootstrap_external_value_raw).abs()
                < (replaced_value - raw_bootstrap_values_next).abs()
            ).item()
        )
        self.assertTrue(
            torch.all(
                (replaced_value - bootstrap_external_value_raw).abs()
                < (replaced_value - raw_anchor_bootstrap_values_next).abs()
            ).item()
        )
        self.assertTrue(torch.all(delta_abs < 0.01).item())

    def test_bootstrap_bonus_terminal_truth_source_preserves_carried_value_when_truth_is_low(self):
        bootstrap_external_value_raw = torch.tensor(
            [[21.0, 18.5]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[3.0, 3.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[12.0, 11.5]], dtype=torch.float32
        )
        anchor_bootstrap_values_next = raw_anchor_bootstrap_values_next.clone()
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            truth,
            alignment,
            activation,
            activation_bridge,
            relative_gap,
            anchor_preference,
            terminal_source,
            terminal_delta_abs,
        ) = compute_bootstrap_bonus_terminal_truth_source_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            anchor_bootstrap_values_next=anchor_bootstrap_values_next,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.08
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.12
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                bootstrap_external_value_raw, 0.82
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.2
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.18),
            critic_contract_task_degradation=torch.full_like(
                bootstrap_external_value_raw, 0.88
            ),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_raw_vs_clean_gap=torch.full_like(
                bootstrap_external_value_raw, 18.0
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth < 0.2).item())
        self.assertTrue(torch.all(alignment < 0.25).item())
        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(
            torch.allclose(
                activation_bridge,
                torch.zeros_like(activation_bridge),
            )
        )
        self.assertTrue(torch.all(relative_gap > 0.0).item())
        self.assertTrue(torch.all(anchor_preference < 0.05).item())
        self.assertTrue(
            torch.all(
                (terminal_source - bootstrap_external_value_raw).abs()
                < (terminal_source - raw_bootstrap_values_next).abs()
            ).item()
        )
        self.assertTrue(
            torch.all(
                (terminal_source - bootstrap_external_value_raw).abs()
                < (terminal_source - raw_anchor_bootstrap_values_next).abs()
            ).item()
        )
        self.assertTrue(torch.all(terminal_delta_abs < 0.1).item())

    def test_bootstrap_bonus_terminal_truth_source_stays_carried_when_gap_is_high(self):
        bootstrap_external_value_raw = torch.tensor(
            [[9.0, 8.0]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[12.0, 12.5]], dtype=torch.float32
        )
        anchor_bootstrap_values_next = torch.tensor(
            [[30.0, 32.0]], dtype=torch.float32
        )
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            _truth,
            _alignment,
            activation,
            activation_bridge,
            relative_gap,
            anchor_preference,
            terminal_source,
            terminal_delta_abs,
        ) = compute_bootstrap_bonus_terminal_truth_source_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            anchor_bootstrap_values_next=anchor_bootstrap_values_next,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.93
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                bootstrap_external_value_raw, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.96
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.97),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_raw_vs_clean_gap=torch.full_like(
                bootstrap_external_value_raw, 1000.0
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(
            torch.allclose(
                activation_bridge,
                torch.zeros_like(activation_bridge),
            )
        )
        self.assertTrue(torch.all(relative_gap > 0.0).item())
        self.assertTrue(torch.all(anchor_preference < 0.99).item())
        self.assertTrue(
            torch.allclose(
                terminal_source,
                bootstrap_external_value_raw,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                terminal_delta_abs,
                torch.zeros_like(terminal_delta_abs),
                atol=1e-6,
            )
        )

    def test_bootstrap_bonus_terminal_truth_source_is_neutral_before_late_window(self):
        bootstrap_external_value_raw = torch.tensor(
            [[7.5, 6.5]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[1.0, 1.2]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[4.5, 4.0]], dtype=torch.float32
        )
        anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.2]], dtype=torch.float32
        )
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            _truth,
            _alignment,
            activation,
            activation_bridge,
            relative_gap,
            anchor_preference,
            terminal_source,
            terminal_delta_abs,
        ) = compute_bootstrap_bonus_terminal_truth_source_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            anchor_bootstrap_values_next=anchor_bootstrap_values_next,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.92
            ),
            behavior_policy_task_cert_real_alarm=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.96),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=0.8,
            critic_contract_bootstrap_raw_vs_clean_gap=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(
            torch.allclose(activation_bridge, torch.zeros_like(activation_bridge))
        )
        self.assertTrue(torch.allclose(terminal_source, bootstrap_external_value_raw))
        self.assertTrue(
            torch.allclose(
                terminal_delta_abs,
                torch.zeros_like(terminal_delta_abs),
            )
        )
        self.assertTrue(torch.all(relative_gap == 0.0).item())
        self.assertTrue(torch.all(anchor_preference == 1.0).item())

    def test_bootstrap_bonus_terminal_truth_source_prefers_anchor_when_truth_is_high_and_gap_is_low(self):
        bootstrap_external_value_raw = torch.tensor(
            [[9.0, 8.0]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[12.0, 12.5]], dtype=torch.float32
        )
        anchor_bootstrap_values_next = raw_anchor_bootstrap_values_next.clone()
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            truth,
            alignment,
            activation,
            activation_bridge,
            relative_gap,
            anchor_preference,
            terminal_source,
            terminal_delta_abs,
        ) = compute_bootstrap_bonus_terminal_truth_source_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            anchor_bootstrap_values_next=anchor_bootstrap_values_next,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.97
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.94
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                bootstrap_external_value_raw, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.98
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.99),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_raw_vs_clean_gap=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth > 0.9).item())
        self.assertTrue(torch.all(alignment > 0.95).item())
        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(
            torch.allclose(
                activation_bridge,
                torch.zeros_like(activation_bridge),
            )
        )
        self.assertTrue(torch.all(relative_gap < 1e-6).item())
        self.assertTrue(torch.all(anchor_preference > 0.9).item())
        self.assertTrue(torch.all(terminal_delta_abs > 0.0).item())
        self.assertTrue(
            torch.all(
                (terminal_source - raw_anchor_bootstrap_values_next).abs()
                < (terminal_source - raw_bootstrap_values_next).abs()
            ).item()
        )

    def test_bootstrap_bonus_terminal_truth_source_uses_hold_state_to_bridge_threshold_edge(self):
        bootstrap_external_value_raw = torch.tensor(
            [[9.0, 8.0]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[12.0, 12.5]], dtype=torch.float32
        )
        anchor_bootstrap_values_next = raw_anchor_bootstrap_values_next.clone()
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            _truth,
            _alignment,
            activation_without_hold,
            activation_bridge_without_hold,
            _relative_gap_without_hold,
            _anchor_preference_without_hold,
            terminal_source_without_hold,
            terminal_delta_abs_without_hold,
        ) = compute_bootstrap_bonus_terminal_truth_source_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            anchor_bootstrap_values_next=anchor_bootstrap_values_next,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.92
            ),
            behavior_policy_task_cert_real_alarm=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.96
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.97),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=0.795,
            critic_contract_bootstrap_bonus_hold_state=0.0,
            critic_contract_bootstrap_raw_vs_clean_gap=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            corridor_semantic_corridor_mask=ones,
        )
        (
            _truth,
            _alignment,
            activation_with_hold,
            activation_bridge_with_hold,
            _relative_gap_with_hold,
            _anchor_preference_with_hold,
            terminal_source_with_hold,
            terminal_delta_abs_with_hold,
        ) = compute_bootstrap_bonus_terminal_truth_source_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            anchor_bootstrap_values_next=anchor_bootstrap_values_next,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.92
            ),
            behavior_policy_task_cert_real_alarm=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.96
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.97),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=0.795,
            critic_contract_bootstrap_bonus_hold_state=0.2,
            critic_contract_bootstrap_raw_vs_clean_gap=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(
            torch.allclose(
                activation_without_hold,
                torch.zeros_like(activation_without_hold),
            )
        )
        self.assertTrue(
            torch.allclose(
                activation_bridge_without_hold,
                torch.zeros_like(activation_bridge_without_hold),
            )
        )
        self.assertTrue(torch.all(activation_bridge_with_hold > 0.0).item())
        self.assertTrue(torch.all(activation_with_hold > 0.0).item())
        self.assertTrue(
            torch.all(activation_with_hold < 0.2 + 1e-6).item()
        )
        self.assertTrue(torch.all(terminal_delta_abs_with_hold > 0.0).item())
        self.assertTrue(
            torch.all(
                terminal_source_with_hold - bootstrap_external_value_raw
                != 0.0
            ).item()
        )
        self.assertTrue(
            torch.allclose(
                terminal_source_without_hold,
                bootstrap_external_value_raw,
            )
        )
        self.assertTrue(
            torch.allclose(
                terminal_delta_abs_without_hold,
                torch.zeros_like(terminal_delta_abs_without_hold),
            )
        )

    def test_bootstrap_bonus_source_value_replacement_prefers_anchor_when_truth_is_high(self):
        bootstrap_external_value_raw = torch.tensor(
            [[9.0, 8.0]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[12.0, 12.5]], dtype=torch.float32
        )
        anchor_available = torch.ones_like(bootstrap_external_value_raw)
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            truth,
            alignment,
            activation,
            bonus_gate,
            replaced_value,
            _delta_abs,
        ) = compute_bootstrap_bonus_source_value_replacement_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_bonus_source_value=raw_anchor_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.95
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.9
            ),
            behavior_policy_task_cert_support_mask=torch.full_like(
                bootstrap_external_value_raw, 1.0
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.97
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.98),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth > 0.9).item())
        self.assertTrue(torch.all(alignment > 0.95).item())
        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(torch.all(bonus_gate > 0.9).item())
        self.assertTrue(
            torch.all(
                (replaced_value - raw_anchor_bootstrap_values_next).abs()
                < (replaced_value - bootstrap_external_value_raw).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_value_replacement_consumes_terminal_truth_candidate(self):
        bootstrap_external_value_raw = torch.tensor(
            [[9.0, 8.0]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[12.0, 12.5]], dtype=torch.float32
        )
        terminal_truth_source = raw_bootstrap_values_next.clone()
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            _truth,
            _alignment,
            activation,
            _bonus_gate,
            replaced_value,
            _delta_abs,
        ) = compute_bootstrap_bonus_source_value_replacement_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_bonus_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.97
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.94
            ),
            behavior_policy_task_cert_support_mask=torch.full_like(
                bootstrap_external_value_raw, 1.0
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.98
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.99),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(
            torch.all(
                (replaced_value - terminal_truth_source).abs()
                < (replaced_value - raw_anchor_bootstrap_values_next).abs()
            ).item()
        )

    def test_bootstrap_bonus_main_chain_keeps_transition_bridge_as_observer_only(self):
        bootstrap_external_value_raw = torch.tensor(
            [[18.0, 16.0]], dtype=torch.float32
        )
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[11.0, 10.5]], dtype=torch.float32
        )
        anchor_bootstrap_values_next = raw_anchor_bootstrap_values_next.clone()
        ones = torch.ones_like(bootstrap_external_value_raw)

        (
            _transition_truth,
            _transition_alignment,
            transition_activation,
            transition_bridge_gate,
            transition_observed_value,
            transition_observed_delta_abs,
        ) = compute_bootstrap_bonus_source_transition_bridge_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.96
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.93
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                bootstrap_external_value_raw, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.97
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.98),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=0.55,
            corridor_semantic_corridor_mask=ones,
        )
        hypothetical_transition_candidate = torch.lerp(
            bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next,
            transition_bridge_gate,
        )
        (
            _terminal_truth,
            _terminal_alignment,
            terminal_activation,
            _terminal_activation_bridge,
            _terminal_relative_gap,
            _terminal_anchor_preference,
            terminal_truth_source,
            _terminal_delta_abs,
        ) = compute_bootstrap_bonus_terminal_truth_source_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            anchor_bootstrap_values_next=anchor_bootstrap_values_next,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.97
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.94
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                bootstrap_external_value_raw, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.98
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.99),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=0.85,
            critic_contract_bootstrap_raw_vs_clean_gap=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            corridor_semantic_corridor_mask=ones,
        )
        (
            _truth,
            _alignment,
            activation,
            _bonus_gate,
            replaced_from_carried_raw,
            _delta_abs,
        ) = compute_bootstrap_bonus_source_value_replacement_contract(
            bootstrap_external_value_raw=bootstrap_external_value_raw,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_bonus_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.97
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.94
            ),
            behavior_policy_task_cert_support_mask=torch.full_like(
                bootstrap_external_value_raw, 1.0
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.98
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.99),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=0.85,
            corridor_semantic_corridor_mask=ones,
        )
        (
            _truth,
            _alignment,
            _activation,
            _bonus_gate,
            replaced_from_transition_candidate,
            _delta_abs,
        ) = compute_bootstrap_bonus_source_value_replacement_contract(
            bootstrap_external_value_raw=hypothetical_transition_candidate,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_bonus_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(
                bootstrap_external_value_raw, 0.97
            ),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                bootstrap_external_value_raw, 0.94
            ),
            behavior_policy_task_cert_support_mask=torch.full_like(
                bootstrap_external_value_raw, 1.0
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                bootstrap_external_value_raw, 0.98
            ),
            task_corridor_gate=torch.full_like(bootstrap_external_value_raw, 0.99),
            critic_contract_task_degradation=torch.zeros_like(
                bootstrap_external_value_raw
            ),
            critic_contract_bootstrap_late_gate=0.85,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(transition_activation > 0.0).item())
        self.assertTrue(torch.all(transition_bridge_gate > 0.0).item())
        self.assertTrue(
            torch.allclose(
                transition_observed_delta_abs,
                torch.zeros_like(transition_observed_delta_abs),
            )
        )
        self.assertTrue(torch.all(terminal_activation > 0.0).item())
        self.assertTrue(torch.all(activation > 0.0).item())
        self.assertTrue(
            torch.allclose(
                transition_observed_value,
                bootstrap_external_value_raw,
            )
        )
        self.assertFalse(
            torch.allclose(
                hypothetical_transition_candidate,
                bootstrap_external_value_raw,
            )
        )
        self.assertFalse(
            torch.allclose(
                replaced_from_carried_raw,
                replaced_from_transition_candidate,
            )
        )
        self.assertTrue(
            torch.all(
                (replaced_from_carried_raw - bootstrap_external_value_raw).abs()
                < (replaced_from_transition_candidate - bootstrap_external_value_raw).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_hold_persistence_is_neutral_before_hold_window(self):
        replaced_value = torch.tensor([[8.0, 7.5]], dtype=torch.float32)
        bridged_value = torch.tensor([[7.0, 7.0]], dtype=torch.float32)
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[11.0, 10.0]], dtype=torch.float32
        )
        anchor_available = torch.ones_like(replaced_value)
        ones = torch.ones_like(replaced_value)

        (
            truth,
            alignment,
            activation,
            release_window_activation,
            release_pressure,
            release_gate,
            release_aware_state,
            persistence_gate,
            hold_persisted,
            hold_delta_abs,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_hold_source_value=raw_anchor_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.9),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.9
            ),
            behavior_policy_task_cert_real_alarm=torch.zeros_like(replaced_value),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.9),
            task_corridor_gate=torch.full_like(replaced_value, 0.9),
            critic_contract_task_degradation=torch.zeros_like(replaced_value),
            critic_contract_bootstrap_late_gate=0.55,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_bonus_hold_state=0.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth >= 0.9).item())
        self.assertTrue(torch.all(alignment >= 0.9).item())
        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(
            torch.allclose(
                release_window_activation,
                torch.zeros_like(release_window_activation),
            )
        )
        self.assertTrue(torch.allclose(release_pressure, torch.zeros_like(release_pressure)))
        self.assertTrue(torch.allclose(release_gate, torch.zeros_like(release_gate)))
        self.assertTrue(torch.allclose(release_aware_state, torch.zeros_like(release_aware_state)))
        self.assertTrue(
            torch.allclose(persistence_gate, torch.zeros_like(persistence_gate))
        )
        self.assertTrue(torch.allclose(hold_persisted, replaced_value))
        self.assertTrue(torch.allclose(hold_delta_abs, torch.zeros_like(hold_delta_abs)))

    def test_bootstrap_bonus_source_hold_persistence_moves_value_when_state_and_truth_are_high(self):
        replaced_value = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        bridged_value = torch.tensor([[7.5, 7.0]], dtype=torch.float32)
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[11.0, 10.5]], dtype=torch.float32
        )
        anchor_available = torch.ones_like(replaced_value)
        ones = torch.ones_like(replaced_value)

        (
            truth,
            alignment,
            activation,
            _release_window_activation,
            _release_pressure,
            _release_gate,
            _release_aware_state,
            persistence_gate,
            hold_persisted,
            hold_delta_abs,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_hold_source_value=raw_anchor_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.95),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.92
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.97),
            task_corridor_gate=torch.full_like(replaced_value, 0.98),
            critic_contract_task_degradation=torch.zeros_like(replaced_value),
            critic_contract_bootstrap_late_gate=0.85,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.full_like(
                replaced_value, 0.3
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.full_like(
                replaced_value, 0.4
            ),
            critic_contract_bootstrap_bonus_hold_state=0.6,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth > 0.9).item())
        self.assertTrue(torch.all(alignment > 0.95).item())
        self.assertTrue(torch.all(activation > 0.0).item())
        self.assertTrue(torch.all(persistence_gate > 0.0).item())
        self.assertTrue(torch.all(hold_delta_abs > 0.0).item())
        self.assertTrue(
            torch.all(
                (hold_persisted - raw_anchor_bootstrap_values_next).abs()
                < (replaced_value - raw_anchor_bootstrap_values_next).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_hold_persistence_keeps_bridged_target_before_handoff_window_when_only_transition_bridge_is_present(self):
        replaced_value = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        bridged_value = torch.tensor([[7.0, 6.8]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        ones = torch.ones_like(replaced_value)

        (
            _truth,
            _alignment,
            activation,
            release_window_activation,
            _release_pressure,
            _release_gate,
            _release_aware_state,
            persistence_gate,
            hold_persisted,
            hold_delta_abs,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=terminal_truth_source,
            raw_bootstrap_values_next=torch.tensor([[11.0, 10.5]], dtype=torch.float32),
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.95),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.92
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.97),
            task_corridor_gate=torch.full_like(replaced_value, 0.98),
            critic_contract_task_degradation=torch.zeros_like(replaced_value),
            critic_contract_bootstrap_late_gate=0.74,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.full_like(
                replaced_value, 0.9
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_bonus_hold_state=0.6,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(activation > 0.0).item())
        self.assertTrue(torch.allclose(release_window_activation, torch.zeros_like(release_window_activation)))
        self.assertTrue(torch.all(persistence_gate > 0.0).item())
        self.assertTrue(torch.all(hold_delta_abs > 0.0).item())
        self.assertTrue(
            torch.all(
                (hold_persisted - bridged_value).abs()
                < (hold_persisted - terminal_truth_source).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_hold_persistence_consumes_transition_bridge_inside_handoff_window(self):
        replaced_value = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        bridged_value = torch.tensor([[7.0, 6.8]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        ones = torch.ones_like(replaced_value)

        (
            _truth_off,
            _alignment_off,
            _activation_off,
            _release_window_off,
            _release_pressure_off,
            _release_gate_off,
            _release_aware_state_off,
            persistence_gate_off,
            hold_persisted_off,
            hold_delta_off,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=terminal_truth_source,
            raw_bootstrap_values_next=torch.tensor([[11.0, 10.5]], dtype=torch.float32),
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.95),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.92
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.97),
            task_corridor_gate=torch.full_like(replaced_value, 0.98),
            critic_contract_task_degradation=torch.zeros_like(replaced_value),
            critic_contract_bootstrap_late_gate=0.82,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_bonus_hold_state=0.05,
            corridor_semantic_corridor_mask=ones,
        )
        (
            _truth_on,
            _alignment_on,
            _activation_on,
            _release_window_on,
            _release_pressure_on,
            _release_gate_on,
            _release_aware_state_on,
            persistence_gate_on,
            hold_persisted_on,
            hold_delta_on,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=terminal_truth_source,
            raw_bootstrap_values_next=torch.tensor([[11.0, 10.5]], dtype=torch.float32),
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.95),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.92
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.97),
            task_corridor_gate=torch.full_like(replaced_value, 0.98),
            critic_contract_task_degradation=torch.zeros_like(replaced_value),
            critic_contract_bootstrap_late_gate=0.82,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.full_like(
                replaced_value, 0.8
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_bonus_hold_state=0.05,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(persistence_gate_on > 0.0).item())
        self.assertTrue(torch.all(hold_delta_on > 0.0).item())
        self.assertTrue(torch.all(persistence_gate_on >= persistence_gate_off).item())
        self.assertTrue(torch.all(hold_delta_on > hold_delta_off).item())
        self.assertTrue(
            torch.all(
                (hold_persisted_on - terminal_truth_source).abs()
                < (hold_persisted_off - terminal_truth_source).abs()
            ).item()
        )
        self.assertTrue(
            torch.all(
                (hold_persisted_off - bridged_value).abs()
                < (hold_persisted_off - terminal_truth_source).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_hold_persistence_consumes_terminal_truth_candidate(self):
        replaced_value = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        bridged_value = torch.tensor([[7.5, 7.0]], dtype=torch.float32)
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[11.0, 10.5]], dtype=torch.float32
        )
        terminal_truth_source = raw_bootstrap_values_next.clone()
        ones = torch.ones_like(replaced_value)

        (
            _truth,
            _alignment,
            activation,
            _release_window_activation,
            _release_pressure,
            _release_gate,
            _release_aware_state,
            persistence_gate,
            hold_persisted,
            hold_delta_abs,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.97),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.94
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.98),
            task_corridor_gate=torch.full_like(replaced_value, 0.99),
            critic_contract_task_degradation=torch.zeros_like(replaced_value),
            critic_contract_bootstrap_late_gate=0.95,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.full_like(
                replaced_value, 0.3
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.full_like(
                replaced_value, 1.0
            ),
            critic_contract_bootstrap_bonus_hold_state=0.8,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(activation > 0.0).item())
        self.assertTrue(torch.all(persistence_gate > 0.0).item())
        self.assertTrue(torch.all(hold_delta_abs > 0.0).item())
        self.assertTrue(
            torch.all(
                (hold_persisted - terminal_truth_source).abs()
                < (hold_persisted - raw_anchor_bootstrap_values_next).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_hold_persistence_rebases_to_replaced_value_once_late_window_is_active(self):
        replaced_value = torch.tensor([[30.0, 28.0]], dtype=torch.float32)
        bridged_value = torch.tensor([[4.0, 3.5]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[27.5, 26.0]], dtype=torch.float32)
        internal_value = torch.tensor([[31.0, 29.0]], dtype=torch.float32)
        ones = torch.ones_like(replaced_value)

        (
            _truth,
            _alignment,
            activation,
            _release_window_activation,
            _release_pressure,
            _release_gate,
            _release_aware_state,
            persistence_gate,
            hold_persisted,
            hold_delta_abs,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=terminal_truth_source,
            raw_bootstrap_values_next=internal_value,
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.97),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.95
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.98),
            task_corridor_gate=torch.full_like(replaced_value, 0.99),
            critic_contract_task_degradation=torch.zeros_like(replaced_value),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.ones_like(
                replaced_value
            ),
            critic_contract_bootstrap_bonus_hold_state=0.85,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(torch.all(persistence_gate > 0.0).item())
        self.assertTrue(torch.all(hold_delta_abs > 0.0).item())
        self.assertTrue(
            torch.all(
                (hold_persisted - replaced_value).abs()
                < (hold_persisted - bridged_value).abs()
            ).item()
        )
        self.assertTrue(
            torch.all(
                (hold_persisted - terminal_truth_source).abs()
                < (replaced_value - terminal_truth_source).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_hold_persistence_does_not_overconsume_single_truth_candidate_when_certified_floor_is_weak(self):
        replaced_value = torch.tensor([[18.0, 17.0]], dtype=torch.float32)
        bridged_value = torch.tensor([[16.0, 15.5]], dtype=torch.float32)
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[4.0, 4.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[24.0, 23.0]], dtype=torch.float32
        )
        anchor_available = torch.ones_like(replaced_value)
        ones = torch.ones_like(replaced_value)

        (
            truth,
            alignment,
            activation,
            _release_window_activation,
            _release_pressure,
            _release_gate,
            _release_aware_state,
            persistence_gate,
            hold_persisted,
            hold_delta_abs,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_hold_source_value=raw_anchor_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.22),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.26
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.97),
            task_corridor_gate=torch.full_like(replaced_value, 0.95),
            critic_contract_task_degradation=torch.full_like(replaced_value, 0.30),
            critic_contract_bootstrap_late_gate=0.95,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.full_like(
                replaced_value, 1.0
            ),
            critic_contract_bootstrap_bonus_hold_state=0.4,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth >= 0.26).item())
        self.assertTrue(torch.all(alignment >= 0.95).item())
        self.assertTrue(torch.all(activation > 0.0).item())
        self.assertTrue(torch.all(persistence_gate > 0.06).item())
        self.assertTrue(torch.all(hold_delta_abs > 0.0).item())
        self.assertLess(float(hold_delta_abs.mean().item()), 0.5)
        self.assertTrue(
            torch.all(
                (hold_persisted - replaced_value).abs()
                < (hold_persisted - raw_anchor_bootstrap_values_next).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_hold_persistence_uses_late_certified_floor_when_state_is_small(self):
        replaced_value = torch.tensor([[18.0, 17.0]], dtype=torch.float32)
        bridged_value = torch.tensor([[16.0, 15.5]], dtype=torch.float32)
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[4.0, 4.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[24.0, 23.0]], dtype=torch.float32
        )
        terminal_truth_source = raw_anchor_bootstrap_values_next.clone()
        ones = torch.ones_like(replaced_value)

        (
            truth,
            alignment,
            activation,
            release_window_activation,
            release_pressure,
            release_gate,
            release_aware_state,
            persistence_gate,
            hold_persisted,
            hold_delta_abs,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.9),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.88
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.05
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.95),
            task_corridor_gate=torch.full_like(replaced_value, 0.96),
            critic_contract_task_degradation=torch.full_like(replaced_value, 0.1),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.full_like(
                replaced_value, 1.0
            ),
            critic_contract_bootstrap_bonus_hold_state=0.02,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth > 0.85).item())
        self.assertTrue(torch.all(alignment > 0.9).item())
        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(torch.allclose(release_window_activation, torch.ones_like(release_window_activation)))
        self.assertTrue(torch.all(release_pressure > 0.09).item())
        self.assertTrue(torch.all(release_gate > 0.09).item())
        self.assertTrue(torch.all(release_aware_state < 0.02).item())
        self.assertTrue(torch.all(persistence_gate > 0.5).item())
        self.assertTrue(torch.all(hold_delta_abs > 1.0).item())
        self.assertTrue(
            torch.all(
                (hold_persisted - terminal_truth_source).abs()
                < (replaced_value - terminal_truth_source).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_hold_persistence_shrinks_under_high_alarm(self):
        replaced_value = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        bridged_value = torch.tensor([[7.5, 7.0]], dtype=torch.float32)
        raw_anchor_bootstrap_values_next = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float32
        )
        raw_bootstrap_values_next = torch.tensor(
            [[11.0, 10.5]], dtype=torch.float32
        )
        anchor_available = torch.ones_like(replaced_value)
        ones = torch.ones_like(replaced_value)

        (
            _truth,
            _alignment,
            activation,
            release_window_activation,
            release_pressure,
            release_gate,
            release_aware_state,
            persistence_gate,
            hold_persisted,
            hold_delta_abs,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_hold_source_value=raw_anchor_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.05),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.08
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.95
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.1),
            task_corridor_gate=torch.full_like(replaced_value, 0.15),
            critic_contract_task_degradation=torch.full_like(replaced_value, 0.95),
            critic_contract_bootstrap_late_gate=0.85,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.full_like(
                replaced_value, 0.3
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.full_like(
                replaced_value, 0.4
            ),
            critic_contract_bootstrap_bonus_hold_state=0.8,
            corridor_semantic_corridor_mask=ones,
        )
        (
            _truth_healthy,
            _alignment_healthy,
            _activation_healthy,
            _release_window_activation_healthy,
            _release_pressure_healthy,
            _release_gate_healthy,
            _release_aware_state_healthy,
            persistence_gate_healthy,
            _hold_persisted_healthy,
            hold_delta_abs_healthy,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=raw_anchor_bootstrap_values_next,
            raw_bootstrap_values_next=raw_bootstrap_values_next,
            reward_semantic_hold_source_value=raw_anchor_bootstrap_values_next,
            critic_anchor_available_next=anchor_available,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.95),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.92
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.97),
            task_corridor_gate=torch.full_like(replaced_value, 0.98),
            critic_contract_task_degradation=torch.zeros_like(replaced_value),
            critic_contract_bootstrap_late_gate=0.85,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.full_like(
                replaced_value, 0.3
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.full_like(
                replaced_value, 0.4
            ),
            critic_contract_bootstrap_bonus_hold_state=0.8,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(activation > 0.0).item())
        self.assertTrue(torch.allclose(release_window_activation, torch.zeros_like(release_window_activation)))
        self.assertTrue(torch.all(release_pressure > 0.9).item())
        self.assertTrue(torch.allclose(release_gate, torch.zeros_like(release_gate)))
        self.assertTrue(torch.allclose(release_aware_state, torch.full_like(release_aware_state, 0.8)))
        self.assertTrue(torch.all(persistence_gate > 0.0).item())
        self.assertTrue(torch.all(hold_delta_abs > 0.0).item())
        self.assertTrue(torch.all(persistence_gate < persistence_gate_healthy).item())
        self.assertTrue(torch.all(hold_delta_abs < hold_delta_abs_healthy).item())
        self.assertTrue(
            torch.all(
                (hold_persisted - replaced_value).abs()
                < (hold_persisted - raw_anchor_bootstrap_values_next).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_hold_persistence_releases_history_under_high_degradation_when_fully_active(self):
        replaced_value = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        bridged_value = torch.tensor([[7.5, 7.0]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        ones = torch.ones_like(replaced_value)

        (
            _truth,
            _alignment,
            activation,
            release_window_activation,
            release_pressure,
            release_gate,
            release_aware_state,
            persistence_gate,
            hold_persisted,
            hold_delta_abs,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=terminal_truth_source,
            raw_bootstrap_values_next=torch.tensor([[11.0, 10.5]], dtype=torch.float32),
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.92),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.91
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.85
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.93),
            task_corridor_gate=torch.full_like(replaced_value, 0.94),
            critic_contract_task_degradation=torch.full_like(replaced_value, 0.90),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.full_like(
                replaced_value, 1.0
            ),
            critic_contract_bootstrap_bonus_hold_state=0.8,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(torch.allclose(release_window_activation, torch.ones_like(release_window_activation)))
        self.assertTrue(torch.all(release_pressure >= 0.9).item())
        self.assertTrue(torch.all(release_gate >= 0.9).item())
        self.assertTrue(torch.all(release_aware_state < 0.081).item())
        self.assertTrue(torch.all(persistence_gate < 0.12).item())
        self.assertTrue(
            torch.all(
                (hold_persisted - replaced_value).abs()
                < (hold_persisted - terminal_truth_source).abs()
            ).item()
        )
        self.assertTrue(torch.all(hold_delta_abs < 1.0).item())

    def test_bootstrap_bonus_source_hold_persistence_fully_active_release_shrinks_gate_even_with_late_source_activation(self):
        replaced_value = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        bridged_value = torch.tensor([[7.5, 7.0]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        ones = torch.ones_like(replaced_value)

        (
            _truth_mid,
            _alignment_mid,
            _activation_mid,
            release_window_mid,
            _release_pressure_mid,
            release_gate_mid,
            _release_aware_state_mid,
            persistence_gate_mid,
            _hold_persisted_mid,
            _hold_delta_mid,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=terminal_truth_source,
            raw_bootstrap_values_next=torch.tensor([[11.0, 10.5]], dtype=torch.float32),
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.92),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.91
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.85
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.93),
            task_corridor_gate=torch.full_like(replaced_value, 0.94),
            critic_contract_task_degradation=torch.full_like(replaced_value, 0.90),
            critic_contract_bootstrap_late_gate=0.89,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.full_like(
                replaced_value, 1.0
            ),
            critic_contract_bootstrap_bonus_hold_state=0.8,
            corridor_semantic_corridor_mask=ones,
        )
        (
            _truth_late,
            _alignment_late,
            _activation_late,
            release_window_late,
            _release_pressure_late,
            release_gate_late,
            _release_aware_state_late,
            persistence_gate_late,
            _hold_persisted_late,
            _hold_delta_late,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=terminal_truth_source,
            raw_bootstrap_values_next=torch.tensor([[11.0, 10.5]], dtype=torch.float32),
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.92),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.91
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.85
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.93),
            task_corridor_gate=torch.full_like(replaced_value, 0.94),
            critic_contract_task_degradation=torch.full_like(replaced_value, 0.90),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.full_like(
                replaced_value, 1.0
            ),
            critic_contract_bootstrap_bonus_hold_state=0.8,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(release_window_mid, torch.zeros_like(release_window_mid)))
        self.assertTrue(torch.allclose(release_gate_mid, torch.zeros_like(release_gate_mid)))
        self.assertTrue(torch.allclose(release_window_late, torch.ones_like(release_window_late)))
        self.assertTrue(torch.all(release_gate_late >= 0.9).item())
        self.assertTrue(torch.all(persistence_gate_late < persistence_gate_mid).item())

    def test_bootstrap_bonus_source_hold_persistence_releases_retained_source_when_value_gate_collapses(self):
        replaced_value = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        bridged_value = torch.tensor([[7.5, 7.0]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        internal_value = torch.tensor([[11.0, 10.5]], dtype=torch.float32)
        ones = torch.ones_like(replaced_value)

        (
            _truth_low,
            _alignment_low,
            _activation_low,
            _release_window_low,
            _release_pressure_low,
            _release_gate_low,
            _release_aware_state_low,
            persistence_gate_low,
            hold_persisted_low,
            _hold_delta_low,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=terminal_truth_source,
            raw_bootstrap_values_next=internal_value,
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.92),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.91
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.85
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.93),
            task_corridor_gate=torch.full_like(replaced_value, 0.94),
            critic_contract_task_degradation=torch.full_like(replaced_value, 0.90),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.full_like(
                replaced_value, 1.0
            ),
            critic_contract_bootstrap_bonus_hold_state=0.8,
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                replaced_value, 0.05
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                replaced_value, 0.08
            ),
            corridor_semantic_corridor_mask=ones,
        )
        (
            _truth_healthy,
            _alignment_healthy,
            _activation_healthy,
            _release_window_healthy,
            _release_pressure_healthy,
            _release_gate_healthy,
            _release_aware_state_healthy,
            persistence_gate_healthy,
            hold_persisted_healthy,
            _hold_delta_healthy,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=terminal_truth_source,
            raw_bootstrap_values_next=internal_value,
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.92),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.91
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.85
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.93),
            task_corridor_gate=torch.full_like(replaced_value, 0.94),
            critic_contract_task_degradation=torch.full_like(replaced_value, 0.90),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.full_like(
                replaced_value, 1.0
            ),
            critic_contract_bootstrap_bonus_hold_state=0.8,
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                replaced_value, 0.92
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                replaced_value, 0.94
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(persistence_gate_low < persistence_gate_healthy).item())
        self.assertTrue(
            torch.all(
                (hold_persisted_low - internal_value).abs()
                < (hold_persisted_healthy - internal_value).abs()
            ).item()
        )
        self.assertTrue(
            torch.all(
                (hold_persisted_low - terminal_truth_source).abs()
                > (hold_persisted_healthy - terminal_truth_source).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_hold_persistence_weak_transition_bridge_still_enters_handoff_path(self):
        replaced_value = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        bridged_value = torch.tensor([[7.0, 6.8]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        ones = torch.ones_like(replaced_value)

        (
            _truth_off,
            _alignment_off,
            activation_off,
            _release_window_off,
            _release_pressure_off,
            _release_gate_off,
            _release_aware_state_off,
            persistence_gate_off,
            hold_persisted_off,
            hold_delta_off,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=terminal_truth_source,
            raw_bootstrap_values_next=torch.tensor([[11.0, 10.5]], dtype=torch.float32),
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.9),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.88
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.06
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.94),
            task_corridor_gate=torch.full_like(replaced_value, 0.95),
            critic_contract_task_degradation=torch.full_like(replaced_value, 0.12),
            critic_contract_bootstrap_late_gate=0.79,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_bonus_hold_state=0.05,
            corridor_semantic_corridor_mask=ones,
        )
        (
            _truth_on,
            _alignment_on,
            activation_on,
            _release_window_on,
            _release_pressure_on,
            _release_gate_on,
            _release_aware_state_on,
            persistence_gate_on,
            hold_persisted_on,
            hold_delta_on,
        ) = compute_bootstrap_bonus_source_hold_persistence_contract(
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            bootstrap_external_bonus_value_transition_bridged=bridged_value,
            raw_anchor_bootstrap_values_next=terminal_truth_source,
            raw_bootstrap_values_next=torch.tensor([[11.0, 10.5]], dtype=torch.float32),
            reward_semantic_hold_source_value=terminal_truth_source,
            critic_anchor_available_next=ones,
            reference_real_reward_agreement=torch.full_like(replaced_value, 0.9),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                replaced_value, 0.88
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                replaced_value, 0.06
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(replaced_value, 0.94),
            task_corridor_gate=torch.full_like(replaced_value, 0.95),
            critic_contract_task_degradation=torch.full_like(replaced_value, 0.12),
            critic_contract_bootstrap_late_gate=0.79,
            critic_contract_bootstrap_source_transition_bridge_gate=torch.full_like(
                replaced_value, 0.06
            ),
            critic_contract_bootstrap_source_value_late_window_activation=torch.zeros_like(
                replaced_value
            ),
            critic_contract_bootstrap_bonus_hold_state=0.05,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(persistence_gate_on > persistence_gate_off).item())
        self.assertTrue(torch.all(hold_delta_on > hold_delta_off).item())
        self.assertTrue(
            torch.all(
                (hold_persisted_on - terminal_truth_source).abs()
                < (hold_persisted_off - terminal_truth_source).abs()
            ).item()
        )

    def test_bootstrap_bonus_consumer_retention_is_neutral_before_fully_active_late_window(self):
        hold_persisted = torch.tensor([[8.0, 7.5]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        replaced_value = torch.tensor([[8.5, 8.0]], dtype=torch.float32)
        ones = torch.ones_like(hold_persisted)

        (
            truth,
            alignment,
            activation,
            retention_gate,
            consumer_retained,
            consumer_delta_abs,
        ) = compute_bootstrap_bonus_consumer_retention_contract(
            bootstrap_external_bonus_value_hold_persisted=hold_persisted,
            bootstrap_external_bonus_terminal_truth_source=terminal_truth_source,
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            reference_real_reward_agreement=torch.full_like(hold_persisted, 0.95),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                hold_persisted, 0.92
            ),
            behavior_policy_task_cert_real_alarm=torch.zeros_like(hold_persisted),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                hold_persisted, 0.96
            ),
            task_corridor_gate=torch.full_like(hold_persisted, 0.97),
            critic_contract_task_degradation=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_late_gate=0.9,
            critic_contract_bootstrap_bonus_hold_state=0.8,
            critic_contract_bootstrap_source_hold_persistence_gate=torch.full_like(
                hold_persisted, 0.0
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth > 0.9).item())
        self.assertTrue(torch.all(alignment > 0.95).item())
        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(
            torch.allclose(retention_gate, torch.zeros_like(retention_gate))
        )
        self.assertTrue(torch.allclose(consumer_retained, hold_persisted))
        self.assertTrue(
            torch.allclose(
                consumer_delta_abs,
                torch.zeros_like(consumer_delta_abs),
            )
        )

    def test_bootstrap_bonus_consumer_retention_stays_neutral_before_window_even_if_hold_gate_is_present(self):
        hold_persisted = torch.tensor([[8.0, 7.5]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        replaced_value = torch.tensor([[8.5, 8.0]], dtype=torch.float32)
        ones = torch.ones_like(hold_persisted)
        hold_gate = torch.full_like(hold_persisted, 0.18)

        (
            _truth,
            _alignment,
            activation,
            retention_gate,
            consumer_retained,
            consumer_delta_abs,
        ) = compute_bootstrap_bonus_consumer_retention_contract(
            bootstrap_external_bonus_value_hold_persisted=hold_persisted,
            bootstrap_external_bonus_terminal_truth_source=terminal_truth_source,
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            reference_real_reward_agreement=torch.full_like(hold_persisted, 0.95),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                hold_persisted, 0.92
            ),
            behavior_policy_task_cert_real_alarm=torch.zeros_like(hold_persisted),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                hold_persisted, 0.96
            ),
            task_corridor_gate=torch.full_like(hold_persisted, 0.97),
            critic_contract_task_degradation=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_late_gate=0.88,
            critic_contract_bootstrap_bonus_hold_state=0.8,
            critic_contract_bootstrap_source_hold_persistence_gate=hold_gate,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(torch.allclose(retention_gate, hold_gate))
        self.assertTrue(torch.allclose(consumer_retained, hold_persisted))
        self.assertTrue(
            torch.allclose(
                consumer_delta_abs,
                torch.zeros_like(consumer_delta_abs),
            )
        )

    def test_bootstrap_bonus_consumer_retention_strengthens_terminal_truth_consumption_when_late_window_is_fully_active(self):
        hold_persisted = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        replaced_value = torch.tensor([[9.0, 8.0]], dtype=torch.float32)
        ones = torch.ones_like(hold_persisted)
        hold_gate = torch.full_like(hold_persisted, 0.2)

        (
            truth,
            alignment,
            activation,
            retention_gate,
            consumer_retained,
            consumer_delta_abs,
        ) = compute_bootstrap_bonus_consumer_retention_contract(
            bootstrap_external_bonus_value_hold_persisted=hold_persisted,
            bootstrap_external_bonus_terminal_truth_source=terminal_truth_source,
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            reference_real_reward_agreement=torch.full_like(hold_persisted, 0.97),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                hold_persisted, 0.94
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                hold_persisted, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                hold_persisted, 0.98
            ),
            task_corridor_gate=torch.full_like(hold_persisted, 0.99),
            critic_contract_task_degradation=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_bonus_hold_state=0.8,
            critic_contract_bootstrap_source_hold_persistence_gate=hold_gate,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth > 0.9).item())
        self.assertTrue(torch.all(alignment > 0.95).item())
        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(torch.all(retention_gate > hold_gate).item())
        self.assertTrue(torch.all(consumer_delta_abs > 0.0).item())
        self.assertTrue(
            torch.all(
                (consumer_retained - terminal_truth_source).abs()
                < (hold_persisted - terminal_truth_source).abs()
            ).item()
        )

    def test_bootstrap_bonus_consumer_retention_can_break_phase423_hold_gate_tie(self):
        hold_persisted = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        replaced_value = torch.tensor([[9.0, 8.0]], dtype=torch.float32)
        ones = torch.ones_like(hold_persisted)
        hold_gate = torch.full_like(hold_persisted, 0.247620090842247)

        (
            truth,
            alignment,
            activation,
            retention_gate,
            consumer_retained,
            consumer_delta_abs,
        ) = compute_bootstrap_bonus_consumer_retention_contract(
            bootstrap_external_bonus_value_hold_persisted=hold_persisted,
            bootstrap_external_bonus_terminal_truth_source=terminal_truth_source,
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            reference_real_reward_agreement=torch.full_like(hold_persisted, 0.97),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                hold_persisted, 0.94
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                hold_persisted, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                hold_persisted, 0.98
            ),
            task_corridor_gate=torch.full_like(hold_persisted, 0.99),
            critic_contract_task_degradation=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_bonus_hold_state=0.247620090842247,
            critic_contract_bootstrap_source_hold_persistence_gate=hold_gate,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(truth > 0.9).item())
        self.assertTrue(torch.all(alignment > 0.95).item())
        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(torch.all(retention_gate > hold_gate).item())
        self.assertTrue(torch.all(consumer_delta_abs > 0.0).item())
        self.assertTrue(
            torch.all(
                (consumer_retained - terminal_truth_source).abs()
                < (hold_persisted - terminal_truth_source).abs()
            ).item()
        )

    def test_bootstrap_bonus_consumer_retention_shrinks_under_high_alarm_or_degradation(self):
        hold_persisted = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        replaced_value = torch.tensor([[9.0, 8.0]], dtype=torch.float32)
        ones = torch.ones_like(hold_persisted)
        hold_gate = torch.full_like(hold_persisted, 0.12)

        (
            _truth,
            _alignment,
            activation,
            retention_gate,
            consumer_retained,
            consumer_delta_abs,
        ) = compute_bootstrap_bonus_consumer_retention_contract(
            bootstrap_external_bonus_value_hold_persisted=hold_persisted,
            bootstrap_external_bonus_terminal_truth_source=terminal_truth_source,
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            reference_real_reward_agreement=torch.full_like(hold_persisted, 0.1),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                hold_persisted, 0.12
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                hold_persisted, 0.95
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                hold_persisted, 0.15
            ),
            task_corridor_gate=torch.full_like(hold_persisted, 0.18),
            critic_contract_task_degradation=torch.full_like(hold_persisted, 0.9),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_bonus_hold_state=0.8,
            critic_contract_bootstrap_source_hold_persistence_gate=hold_gate,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(activation, torch.ones_like(activation)))
        self.assertTrue(torch.allclose(retention_gate, hold_gate))
        self.assertTrue(torch.allclose(consumer_delta_abs, torch.zeros_like(consumer_delta_abs)))
        self.assertTrue(
            torch.all(
                (consumer_retained - hold_persisted).abs() < 1e-6
            ).item()
        )

    def test_bootstrap_final_external_value_assembly_keeps_consumer_layer_observer_only(self):
        hold_persisted = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        replaced_value = torch.tensor([[9.0, 8.0]], dtype=torch.float32)
        anchor_value = torch.tensor([[1.5, 1.0]], dtype=torch.float32)
        internal_value = torch.tensor([[7.0, 6.5]], dtype=torch.float32)
        anchor_valid_mask = torch.ones_like(hold_persisted)
        ones = torch.ones_like(hold_persisted)

        sequential_consumer = compute_bootstrap_bonus_consumer_retention_contract(
            bootstrap_external_bonus_value_hold_persisted=hold_persisted,
            bootstrap_external_bonus_terminal_truth_source=terminal_truth_source,
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            reference_real_reward_agreement=torch.full_like(hold_persisted, 0.97),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                hold_persisted, 0.94
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                hold_persisted, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                hold_persisted, 0.98
            ),
            task_corridor_gate=torch.full_like(hold_persisted, 0.99),
            critic_contract_task_degradation=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_bonus_hold_state=0.8,
            critic_contract_bootstrap_source_hold_persistence_gate=torch.full_like(
                hold_persisted, 0.2
            ),
            corridor_semantic_corridor_mask=ones,
        )
        sequential_quality = compute_bootstrap_final_external_value_quality_contract(
            bootstrap_external_bonus_value_consumer_retained=hold_persisted,
            anchor_bootstrap_values_next=anchor_value,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=anchor_valid_mask,
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                hold_persisted, 0.92
            ),
            critic_contract_bootstrap_raw_vs_clean_gap=torch.full_like(
                hold_persisted, 0.25
            ),
            critic_contract_bootstrap_source_reward_semantic_truth=torch.full_like(
                hold_persisted, 0.95
            ),
            critic_contract_bootstrap_source_semantic_alignment=torch.full_like(
                hold_persisted, 0.96
            ),
            actor_contract_task_mismatch=torch.full_like(hold_persisted, 0.04),
            actor_corridor_semantic_inflation_excess=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_source_hold_window_activation=torch.ones_like(
                hold_persisted
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(
                hold_persisted, 0.02
            ),
            actor_task_geom_corridor_disagreement=torch.full_like(
                hold_persisted, 0.2
            ),
        )
        assembled = compute_bootstrap_final_external_value_assembly_contract(
            bootstrap_external_bonus_value_hold_persisted=hold_persisted,
            bootstrap_external_bonus_terminal_truth_source=terminal_truth_source,
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            reference_real_reward_agreement=torch.full_like(hold_persisted, 0.97),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                hold_persisted, 0.94
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                hold_persisted, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                hold_persisted, 0.98
            ),
            task_corridor_gate=torch.full_like(hold_persisted, 0.99),
            critic_contract_task_degradation=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_bonus_hold_state=0.8,
            critic_contract_bootstrap_source_hold_persistence_gate=torch.full_like(
                hold_persisted, 0.2
            ),
            corridor_semantic_corridor_mask=ones,
            anchor_bootstrap_values_next=anchor_value,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=anchor_valid_mask,
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                hold_persisted, 0.92
            ),
            critic_contract_bootstrap_raw_vs_clean_gap=torch.full_like(
                hold_persisted, 0.25
            ),
            critic_contract_bootstrap_source_reward_semantic_truth=torch.full_like(
                hold_persisted, 0.95
            ),
            critic_contract_bootstrap_source_semantic_alignment=torch.full_like(
                hold_persisted, 0.96
            ),
            actor_contract_task_mismatch=torch.full_like(hold_persisted, 0.04),
            actor_corridor_semantic_inflation_excess=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_source_hold_window_activation=torch.ones_like(
                hold_persisted
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(
                hold_persisted, 0.02
            ),
            actor_task_geom_corridor_disagreement=torch.full_like(
                hold_persisted, 0.2
            ),
        )

        self.assertTrue(torch.allclose(assembled.source_consumer_truth, sequential_consumer[0]))
        self.assertTrue(torch.allclose(assembled.source_consumer_alignment, sequential_consumer[1]))
        self.assertTrue(
            torch.allclose(
                assembled.source_consumer_window_activation,
                torch.zeros_like(sequential_consumer[2]),
            )
        )
        self.assertTrue(
            torch.allclose(
                assembled.source_consumer_retention_gate,
                torch.zeros_like(sequential_consumer[3]),
            )
        )
        self.assertTrue(
            torch.allclose(assembled.bonus_value_consumer_retained, hold_persisted)
        )
        self.assertTrue(
            torch.allclose(
                assembled.bonus_value_consumer_delta_abs,
                torch.zeros_like(hold_persisted),
            )
        )
        self.assertTrue(torch.allclose(assembled.regime_quality_gate, sequential_quality[0]))
        self.assertTrue(torch.allclose(assembled.final_value_behavior_health, sequential_quality[1]))
        self.assertTrue(torch.allclose(assembled.prehold_value_failfast_gate, sequential_quality[2]))
        self.assertTrue(torch.allclose(assembled.floor_value_injection_gate, sequential_quality[3]))
        self.assertTrue(torch.allclose(assembled.floor_value_clamped, sequential_quality[4]))
        self.assertTrue(torch.allclose(assembled.floor_value_delta_abs, sequential_quality[5]))
        self.assertTrue(torch.allclose(assembled.final_value_injection_gate, sequential_quality[6]))
        self.assertTrue(torch.allclose(assembled.bonus_value_quality_clamped, sequential_quality[7]))
        self.assertTrue(torch.allclose(assembled.bonus_value_quality_delta_abs, sequential_quality[8]))

    def test_bootstrap_bonus_hold_state_update_only_rises_with_seed_and_health(self):
        mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        prev_state = 0.4
        source_transition_bridge_gate = torch.tensor(
            [[0.5, 0.4]], dtype=torch.float32
        )
        source_value_late_window_activation = torch.tensor(
            [[0.2, 0.6]], dtype=torch.float32
        )
        source_value_truth = torch.tensor([[0.8, 0.9]], dtype=torch.float32)
        source_value_alignment = torch.tensor([[0.9, 0.85]], dtype=torch.float32)
        real_alarm = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        task_degradation = torch.tensor([[0.15, 0.05]], dtype=torch.float32)

        seed = torch.maximum(
            source_transition_bridge_gate,
            source_value_late_window_activation,
        ).clamp(0.0, 1.0)
        health = (
            torch.minimum(source_value_truth, source_value_alignment).clamp(0.0, 1.0)
            * (1.0 - torch.maximum(real_alarm, task_degradation)).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        next_state = min(
            1.0,
            max(
                (
                    1.0
                    - float(
                        _masked_mean_tensor(
                            torch.maximum(real_alarm, task_degradation),
                            mask,
                        ).item()
                    )
                )
                * 0.9
                * prev_state,
                float(_masked_mean_tensor(seed, mask).item())
                * float(_masked_mean_tensor(health, mask).item()),
            ),
        )

        low_seed = torch.zeros_like(seed)
        low_health = torch.zeros_like(health)
        high_release = torch.ones_like(seed)
        decayed_state = min(
            1.0,
            max(
                (
                    1.0
                    - float(_masked_mean_tensor(high_release, mask).item())
                )
                * 0.9
                * prev_state,
                float(_masked_mean_tensor(low_seed, mask).item())
                * float(_masked_mean_tensor(low_health, mask).item()),
            ),
        )

        self.assertGreater(next_state, 0.0)
        self.assertLess(next_state, prev_state)
        self.assertEqual(decayed_state, 0.0)
        self.assertLess(decayed_state, next_state)
        self.assertLess(decayed_state, prev_state)

    def test_bootstrap_bonus_hold_state_update_does_not_release_prior_state_before_release_window(self):
        mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        prev_state = 0.6
        source_transition_bridge_gate = torch.zeros((1, 2), dtype=torch.float32)
        source_value_late_window_activation = torch.full(
            (1, 2), 0.84, dtype=torch.float32
        )
        source_value_truth = torch.tensor([[0.82, 0.84]], dtype=torch.float32)
        source_value_alignment = torch.tensor([[0.92, 0.94]], dtype=torch.float32)
        real_alarm = torch.tensor([[0.78, 0.82]], dtype=torch.float32)
        task_degradation = torch.tensor([[0.85, 0.88]], dtype=torch.float32)
        late_gate = torch.full((1, 2), 0.84, dtype=torch.float32)

        seed = torch.maximum(
            source_transition_bridge_gate,
            source_value_late_window_activation,
        ).clamp(0.0, 1.0)
        health = (
            torch.minimum(source_value_truth, source_value_alignment).clamp(0.0, 1.0)
            * (1.0 - torch.maximum(real_alarm, task_degradation)).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)

        seed_mean = float(_masked_mean_tensor(seed, mask).item())
        health_mean = float(_masked_mean_tensor(health, mask).item())
        release_pressure = torch.maximum(real_alarm, task_degradation).clamp(0.0, 1.0)
        release_window_activation = ((late_gate - 0.9) / 0.1).clamp(0.0, 1.0)
        effective_release_pressure_mean = float(
            _masked_mean_tensor(
                release_window_activation * release_pressure,
                mask,
            ).item()
        )
        next_state = min(
            1.0,
            max(
                (1.0 - effective_release_pressure_mean) * 0.9 * prev_state,
                seed_mean * health_mean,
            ),
        )

        self.assertEqual(effective_release_pressure_mean, 0.0)
        self.assertAlmostEqual(next_state, 0.9 * prev_state, places=6)
        self.assertGreater(next_state, seed_mean * health_mean)

    def test_bootstrap_bonus_hold_state_update_releases_prior_state_under_high_degradation(self):
        mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        prev_state = 0.6
        source_transition_bridge_gate = torch.zeros((1, 2), dtype=torch.float32)
        source_value_late_window_activation = torch.full(
            (1, 2), 0.84, dtype=torch.float32
        )
        source_value_truth = torch.tensor([[0.82, 0.84]], dtype=torch.float32)
        source_value_alignment = torch.tensor([[0.92, 0.94]], dtype=torch.float32)
        real_alarm = torch.tensor([[0.78, 0.82]], dtype=torch.float32)
        task_degradation = torch.tensor([[0.85, 0.88]], dtype=torch.float32)
        late_gate = torch.ones((1, 2), dtype=torch.float32)

        seed = torch.maximum(
            source_transition_bridge_gate,
            source_value_late_window_activation,
        ).clamp(0.0, 1.0)
        health = (
            torch.minimum(source_value_truth, source_value_alignment).clamp(0.0, 1.0)
            * (1.0 - torch.maximum(real_alarm, task_degradation)).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)

        seed_mean = float(_masked_mean_tensor(seed, mask).item())
        health_mean = float(_masked_mean_tensor(health, mask).item())
        release_pressure = torch.maximum(real_alarm, task_degradation).clamp(0.0, 1.0)
        release_window_activation = ((late_gate - 0.9) / 0.1).clamp(0.0, 1.0)
        effective_release_pressure_mean = float(
            _masked_mean_tensor(
                release_window_activation * release_pressure,
                mask,
            ).item()
        )
        next_state = min(
            1.0,
            max(
                (1.0 - effective_release_pressure_mean) * 0.9 * prev_state,
                seed_mean * health_mean,
            ),
        )
        self.assertLess(next_state, 0.2)
        self.assertLess(next_state, prev_state)

    def test_post_transition_certified_retention_floor_can_strengthen_persistence_gate(self):
        (
            activation,
            capture,
            floor_candidate,
            floor_state,
            hold_floored,
            gate_floored,
        ) = compute_post_transition_certified_retention_floor_contract(
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_trigger_gate=1.0,
            critic_contract_bootstrap_hold_state_active_transition=0.24,
            critic_contract_bootstrap_hold_state_post_transition=0.08,
            source_hold_seed_mean=0.6,
            source_hold_health_mean=0.5,
            source_hold_retention_mean=0.3,
            source_hold_release_pressure_mean=0.2,
            critic_contract_bootstrap_source_hold_persistence_gate_mean=0.02,
            prev_post_transition_certified_floor_state=0.16,
        )

        self.assertAlmostEqual(activation, 1.0)
        self.assertAlmostEqual(capture, 0.3)
        self.assertGreater(floor_candidate, 0.0)
        self.assertGreater(floor_state, 0.0)
        self.assertGreaterEqual(hold_floored, 0.08)
        self.assertGreater(gate_floored, 0.02)

    def test_post_transition_retention_capture_marks_historical_winner_when_supported(self):
        capture = compute_post_transition_retention_capture(
            late_gate_mean=1.0,
            active_hold_state=0.0,
            post_hold_state=0.08,
            source_hold_seed_mean=0.6,
            source_hold_health_mean=0.5,
            source_hold_retention_mean=0.04,
            source_hold_release_pressure_mean=0.2,
            prev_post_transition_certified_floor_state=0.4,
        )

        self.assertAlmostEqual(capture.window_activation, 1.0)
        self.assertGreater(capture.historical_authority, capture.current_authority)
        self.assertGreater(capture.certified_capture_source, capture.current_authority)
        self.assertTrue(capture.is_historical_winner)
        self.assertEqual(capture.dominant_source, "historical")
        self.assertGreater(capture.historical_takeover_gate, 0.0)
        self.assertGreater(capture.dominance_margin, 0.0)

    def test_post_transition_retention_capture_blocks_history_when_release_pressure_overwhelms_support(self):
        capture = compute_post_transition_retention_capture(
            late_gate_mean=1.0,
            active_hold_state=0.0,
            post_hold_state=0.18,
            source_hold_seed_mean=0.05,
            source_hold_health_mean=0.1,
            source_hold_retention_mean=0.16,
            source_hold_release_pressure_mean=0.95,
            prev_post_transition_certified_floor_state=0.2,
        )

        self.assertAlmostEqual(capture.window_activation, 1.0)
        self.assertEqual(capture.dominant_source, "current")
        self.assertFalse(capture.is_historical_winner)
        self.assertTrue(capture.release_blocked)
        self.assertAlmostEqual(
            capture.certified_capture_source,
            capture.current_authority,
            places=6,
        )

    def test_post_transition_retention_capture_keeps_supported_small_historical_gap(self):
        capture = compute_post_transition_retention_capture(
            late_gate_mean=1.0,
            active_hold_state=0.0,
            post_hold_state=0.09775253547712914,
            source_hold_seed_mean=1.0,
            source_hold_health_mean=0.09775253547712914,
            source_hold_retention_mean=0.09775253547712914,
            source_hold_release_pressure_mean=0.630566418170929,
            prev_post_transition_certified_floor_state=0.08028087585663083,
        )

        self.assertAlmostEqual(capture.window_activation, 1.0)
        self.assertGreater(capture.historical_authority, capture.current_authority)
        self.assertGreater(capture.certified_capture_source, capture.current_authority)
        self.assertTrue(capture.is_historical_winner)
        self.assertGreater(capture.dominance_margin, 0.0)

    def test_post_transition_certified_retention_floor_uses_support_bounded_history_uplift(self):
        (
            activation,
            capture,
            floor_candidate,
            floor_state,
            hold_floored,
            gate_floored,
        ) = compute_post_transition_certified_retention_floor_contract(
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_trigger_gate=1.0,
            critic_contract_bootstrap_hold_state_active_transition=0.0,
            critic_contract_bootstrap_hold_state_post_transition=0.08,
            source_hold_seed_mean=0.6,
            source_hold_health_mean=0.5,
            source_hold_retention_mean=0.04,
            source_hold_release_pressure_mean=0.2,
            critic_contract_bootstrap_source_hold_persistence_gate_mean=0.02,
            prev_post_transition_certified_floor_state=0.4,
        )

        self.assertAlmostEqual(activation, 1.0)
        self.assertGreater(capture, 0.08)
        self.assertAlmostEqual(capture, 0.38)
        self.assertGreater(floor_candidate, 0.38 - 1e-6)
        self.assertGreater(floor_state, 0.08)
        self.assertGreaterEqual(hold_floored, floor_state)
        self.assertAlmostEqual(gate_floored, floor_state)

    def test_post_transition_historical_winner_can_raise_persistence_gate_above_raw_gate(self):
        (
            _activation,
            _capture,
            _floor_candidate,
            floor_state,
            _hold_floored,
            gate_floored,
        ) = compute_post_transition_certified_retention_floor_contract(
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_trigger_gate=1.0,
            critic_contract_bootstrap_hold_state_active_transition=0.0,
            critic_contract_bootstrap_hold_state_post_transition=0.08,
            source_hold_seed_mean=0.7,
            source_hold_health_mean=0.8,
            source_hold_retention_mean=0.04,
            source_hold_release_pressure_mean=0.1,
            critic_contract_bootstrap_source_hold_persistence_gate_mean=0.02,
            prev_post_transition_certified_floor_state=0.45,
        )

        self.assertGreater(floor_state, 0.08)
        self.assertGreater(gate_floored, 0.02)
        self.assertGreaterEqual(gate_floored, floor_state)

    def test_post_transition_certified_floor_preserves_supported_small_historical_gap(self):
        (
            activation,
            capture,
            floor_candidate,
            floor_state,
            _hold_floored,
            gate_floored,
        ) = compute_post_transition_certified_retention_floor_contract(
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_trigger_gate=1.0,
            critic_contract_bootstrap_hold_state_active_transition=0.0,
            critic_contract_bootstrap_hold_state_post_transition=0.09775253547712914,
            source_hold_seed_mean=1.0,
            source_hold_health_mean=0.09775253547712914,
            source_hold_retention_mean=0.09775253547712914,
            source_hold_release_pressure_mean=0.630566418170929,
            critic_contract_bootstrap_source_hold_persistence_gate_mean=0.09775253547712914,
            prev_post_transition_certified_floor_state=0.08028087585663083,
        )

        self.assertAlmostEqual(activation, 1.0)
        self.assertGreater(capture, 0.09775253547712914)
        self.assertGreater(floor_candidate, 0.09775253547712914)
        self.assertGreater(floor_state, 0.09775253547712914)
        self.assertGreater(gate_floored, 0.09775253547712914)

    def test_adaptive_compensation_state_roundtrip_restores_post_transition_certified_floor_state(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=1,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop._adaptive_imag_critic_bootstrap_post_transition_certified_floor_state = 0.37

        exported = loop._export_adaptive_compensation_state()
        minimal_payload = exported["minimal_compensation_state"]

        restored = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        restored._restore_adaptive_compensation_state(exported)

        self.assertEqual(minimal_payload["schema_version"], 2)
        for removed_key in (
            "adaptive_imag_continue_cap_state",
            "adaptive_imag_compensation_trigger_actor_scale_state",
            "adaptive_imag_compensation_phase",
            "adaptive_imag_critic_bootstrap_contact_surface_state",
            "adaptive_imag_critic_bootstrap_bonus_hold_state",
            "adaptive_imag_critic_bootstrap_post_transition_certified_floor_state",
            "adaptive_imag_compensation_persistence_hold_until_step",
            "adaptive_imag_compensation_persistence_armed_step",
            "adaptive_imag_compensation_persistence_entry_protect_until_step",
            "adaptive_imag_compensation_persistence_release_hold_until_step",
            "adaptive_imag_compensation_post_solved_hold_until_step",
        ):
            self.assertNotIn(removed_key, exported)
        self.assertAlmostEqual(
            restored._adaptive_imag_critic_bootstrap_post_transition_certified_floor_state,
            0.37,
        )

    def test_bootstrap_trigger_entry_contract_uses_real_reward_degradation_even_without_kinematic_gap(self):
        mask = torch.ones((1, 2), dtype=torch.float32)
        zeros = torch.zeros_like(mask)

        (
            eval_drawdown,
            kinematic_degradation,
            real_reward_degradation,
            real_task_degradation,
            task_degradation,
            release_guard,
            negative_adv_pressure,
            trigger_surface,
            regime_quality_gate,
            trigger_gate,
            late_gate,
            precontact_gate,
        ) = compute_bootstrap_trigger_entry_contract(
            external_eval_best_mean=500.0,
            external_eval_last_mean=500.0,
            real_reward_degradation=0.8,
            real_task_cert_gate=1.0,
            real_behavior_action_switch_rate=0.1,
            real_behavior_action_oscillation_rate=0.1,
            behavior_action_switch_rate_preinflation=0.1,
            behavior_action_oscillation_rate_preinflation=0.1,
            critic_contract_bootstrap_eval_gate=0.0,
            bootstrap_step_gate=1.0,
            bootstrap_precontact_step_gate=1.0,
            critic_contract_semantic_pressure=zeros,
            critic_anchor_confidence=zeros,
            critic_contract_bootstrap_raw_vs_clean_gap=zeros,
            anchor_bootstrap_values_next=zeros,
            online_advantages=zeros,
            returns=torch.ones_like(mask),
            corridor_semantic_corridor_mask=mask,
        )

        self.assertEqual(eval_drawdown, 0.0)
        self.assertEqual(kinematic_degradation, 0.0)
        self.assertAlmostEqual(real_reward_degradation, 0.8, places=6)
        self.assertEqual(real_task_degradation, 0.0)
        self.assertGreater(task_degradation, 0.0)
        self.assertTrue(torch.allclose(release_guard, zeros))
        self.assertTrue(torch.allclose(negative_adv_pressure, zeros))
        self.assertTrue(torch.all(trigger_surface > 0.0).item())
        self.assertTrue(torch.all(regime_quality_gate < 0.5).item())
        self.assertGreater(trigger_gate, 0.0)
        self.assertGreater(late_gate, 0.0)
        self.assertGreater(precontact_gate, 0.0)

    def test_resolve_bootstrap_external_eval_feedback_snapshot_defaults_when_feedback_is_missing(self):
        mask = torch.ones((1, 2), dtype=torch.float32)

        (
            real_reward_degradation,
            real_task_cert_gate,
            real_task_cert_alarm,
            real_task_cert_recovery,
            real_behavior_action_switch_rate,
            real_behavior_action_oscillation_rate,
            external_feedback_available,
            external_feedback_age_steps,
        ) = _resolve_bootstrap_external_eval_feedback_snapshot(
            bootstrap_external_eval_feedback_telemetry={},
            bootstrap_external_eval_feedback_last_step=-1,
            global_step=500,
            corridor_semantic_corridor_mask=mask,
        )

        self.assertEqual(external_feedback_available, 0.0)
        self.assertEqual(external_feedback_age_steps, -1.0)
        self.assertEqual(real_reward_degradation, 0.0)
        self.assertEqual(real_task_cert_gate, 1.0)
        self.assertEqual(real_task_cert_alarm, 0.0)
        self.assertEqual(real_task_cert_recovery, 1.0)
        self.assertEqual(real_behavior_action_switch_rate, 0.0)
        self.assertEqual(real_behavior_action_oscillation_rate, 0.0)

    def test_resolve_bootstrap_external_eval_feedback_snapshot_uses_dedicated_feedback_values(self):
        mask = torch.ones((1, 2), dtype=torch.float32)
        telemetry = {
            "real_reward_degradation": 0.6,
            "real_task_cert_gate": 0.35,
            "real_task_cert_alarm": 0.4,
            "real_task_cert_recovery": 0.3,
            "real_behavior_action_switch_rate": 0.7,
            "real_behavior_action_oscillation_rate": 0.45,
        }

        (
            real_reward_degradation,
            real_task_cert_gate,
            real_task_cert_alarm,
            real_task_cert_recovery,
            real_behavior_action_switch_rate,
            real_behavior_action_oscillation_rate,
            external_feedback_available,
            external_feedback_age_steps,
        ) = _resolve_bootstrap_external_eval_feedback_snapshot(
            bootstrap_external_eval_feedback_telemetry=telemetry,
            bootstrap_external_eval_feedback_last_step=100,
            global_step=160,
            corridor_semantic_corridor_mask=mask,
        )

        self.assertEqual(external_feedback_available, 1.0)
        self.assertEqual(external_feedback_age_steps, 60.0)
        self.assertAlmostEqual(real_reward_degradation, 0.6, places=6)
        self.assertAlmostEqual(real_task_cert_gate, 0.35, places=6)
        self.assertAlmostEqual(real_task_cert_alarm, 0.4, places=6)
        self.assertAlmostEqual(real_task_cert_recovery, 0.3, places=6)
        self.assertAlmostEqual(real_behavior_action_switch_rate, 0.7, places=6)
        self.assertAlmostEqual(real_behavior_action_oscillation_rate, 0.45, places=6)

    def test_bootstrap_trigger_entry_contract_uses_real_task_degradation_even_without_reward_drop(self):
        mask = torch.ones((1, 2), dtype=torch.float32)
        zeros = torch.zeros_like(mask)

        (
            _eval_drawdown,
            _kinematic_degradation,
            real_reward_degradation,
            real_task_degradation,
            task_degradation,
            release_guard,
            negative_adv_pressure,
            trigger_surface,
            regime_quality_gate,
            trigger_gate,
            late_gate,
            precontact_gate,
        ) = compute_bootstrap_trigger_entry_contract(
            external_eval_best_mean=500.0,
            external_eval_last_mean=500.0,
            real_reward_degradation=0.0,
            real_task_cert_gate=0.2,
            real_behavior_action_switch_rate=0.1,
            real_behavior_action_oscillation_rate=0.1,
            behavior_action_switch_rate_preinflation=0.1,
            behavior_action_oscillation_rate_preinflation=0.1,
            critic_contract_bootstrap_eval_gate=0.0,
            bootstrap_step_gate=1.0,
            bootstrap_precontact_step_gate=1.0,
            critic_contract_semantic_pressure=zeros,
            critic_anchor_confidence=zeros,
            critic_contract_bootstrap_raw_vs_clean_gap=zeros,
            anchor_bootstrap_values_next=zeros,
            online_advantages=zeros,
            returns=torch.ones_like(mask),
            corridor_semantic_corridor_mask=mask,
        )

        self.assertEqual(real_reward_degradation, 0.0)
        self.assertAlmostEqual(real_task_degradation, 0.8, places=6)
        self.assertGreater(task_degradation, 0.0)
        self.assertTrue(torch.allclose(release_guard, zeros))
        self.assertTrue(torch.allclose(negative_adv_pressure, zeros))
        self.assertTrue(torch.all(trigger_surface > 0.0).item())
        self.assertTrue(torch.all(regime_quality_gate < 0.5).item())
        self.assertGreater(trigger_gate, 0.0)
        self.assertGreater(late_gate, 0.0)
        self.assertGreater(precontact_gate, 0.0)

    def test_bootstrap_trigger_entry_contract_stays_idle_when_all_degradation_sources_are_low(self):
        mask = torch.ones((1, 2), dtype=torch.float32)
        zeros = torch.zeros_like(mask)

        (
            eval_drawdown,
            kinematic_degradation,
            real_reward_degradation,
            real_task_degradation,
            task_degradation,
            release_guard,
            negative_adv_pressure,
            trigger_surface,
            regime_quality_gate,
            trigger_gate,
            late_gate,
            precontact_gate,
        ) = compute_bootstrap_trigger_entry_contract(
            external_eval_best_mean=500.0,
            external_eval_last_mean=500.0,
            real_reward_degradation=0.0,
            real_task_cert_gate=1.0,
            real_behavior_action_switch_rate=0.1,
            real_behavior_action_oscillation_rate=0.1,
            behavior_action_switch_rate_preinflation=0.1,
            behavior_action_oscillation_rate_preinflation=0.1,
            critic_contract_bootstrap_eval_gate=0.0,
            bootstrap_step_gate=1.0,
            bootstrap_precontact_step_gate=1.0,
            critic_contract_semantic_pressure=zeros,
            critic_anchor_confidence=zeros,
            critic_contract_bootstrap_raw_vs_clean_gap=zeros,
            anchor_bootstrap_values_next=zeros,
            online_advantages=zeros,
            returns=torch.ones_like(mask),
            corridor_semantic_corridor_mask=mask,
        )

        self.assertEqual(eval_drawdown, 0.0)
        self.assertEqual(kinematic_degradation, 0.0)
        self.assertEqual(real_reward_degradation, 0.0)
        self.assertEqual(real_task_degradation, 0.0)
        self.assertEqual(task_degradation, 0.0)
        self.assertTrue(torch.allclose(release_guard, zeros))
        self.assertTrue(torch.allclose(negative_adv_pressure, zeros))
        self.assertTrue(torch.allclose(trigger_surface, zeros))
        self.assertTrue(torch.all(regime_quality_gate > 0.6).item())
        self.assertEqual(trigger_gate, 0.0)
        self.assertEqual(late_gate, 0.0)
        self.assertEqual(precontact_gate, 0.0)

    def test_bootstrap_trigger_entry_contract_respects_step_gates(self):
        mask = torch.ones((1, 2), dtype=torch.float32)
        zeros = torch.zeros_like(mask)

        (
            _eval_drawdown,
            _kinematic_degradation,
            _real_reward_degradation,
            _real_task_degradation,
            task_degradation,
            _release_guard,
            _negative_adv_pressure,
            trigger_surface,
            regime_quality_gate,
            trigger_gate,
            late_gate,
            precontact_gate,
        ) = compute_bootstrap_trigger_entry_contract(
            external_eval_best_mean=500.0,
            external_eval_last_mean=500.0,
            real_reward_degradation=0.9,
            real_task_cert_gate=1.0,
            real_behavior_action_switch_rate=0.1,
            real_behavior_action_oscillation_rate=0.1,
            behavior_action_switch_rate_preinflation=0.1,
            behavior_action_oscillation_rate_preinflation=0.1,
            critic_contract_bootstrap_eval_gate=0.0,
            bootstrap_step_gate=0.0,
            bootstrap_precontact_step_gate=0.5,
            critic_contract_semantic_pressure=zeros,
            critic_anchor_confidence=zeros,
            critic_contract_bootstrap_raw_vs_clean_gap=zeros,
            anchor_bootstrap_values_next=zeros,
            online_advantages=zeros,
            returns=torch.ones_like(mask),
            corridor_semantic_corridor_mask=mask,
        )

        self.assertGreater(task_degradation, 0.0)
        self.assertTrue(torch.all(trigger_surface > 0.0).item())
        self.assertTrue(torch.all(regime_quality_gate < 0.5).item())
        self.assertGreater(trigger_gate, 0.0)
        self.assertEqual(late_gate, 0.0)
        self.assertGreater(precontact_gate, 0.0)

    def test_bootstrap_trigger_entry_contract_can_keep_trigger_active_while_regime_quality_veto_is_low(self):
        mask = torch.ones((1, 2), dtype=torch.float32)
        semantic_pressure = torch.full_like(mask, 0.9)
        anchor_confidence = torch.full_like(mask, 0.95)
        raw_vs_clean_gap = torch.full_like(mask, 18.0)
        anchor_bootstrap = torch.full_like(mask, 3.0)
        returns = torch.full_like(mask, 22.0)
        online_adv = torch.full_like(mask, -6.0)

        (
            _eval_drawdown,
            _kinematic_degradation,
            _real_reward_degradation,
            _real_task_degradation,
            task_degradation,
            _release_guard,
            _negative_adv_pressure,
            trigger_surface,
            regime_quality_gate,
            trigger_gate,
            late_gate,
            precontact_gate,
        ) = compute_bootstrap_trigger_entry_contract(
            external_eval_best_mean=500.0,
            external_eval_last_mean=500.0,
            real_reward_degradation=0.0,
            real_task_cert_gate=0.45,
            real_behavior_action_switch_rate=0.8,
            real_behavior_action_oscillation_rate=0.7,
            behavior_action_switch_rate_preinflation=0.1,
            behavior_action_oscillation_rate_preinflation=0.1,
            critic_contract_bootstrap_eval_gate=1.0,
            bootstrap_step_gate=1.0,
            bootstrap_precontact_step_gate=1.0,
            critic_contract_semantic_pressure=semantic_pressure,
            critic_anchor_confidence=anchor_confidence,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_vs_clean_gap,
            anchor_bootstrap_values_next=anchor_bootstrap,
            online_advantages=online_adv,
            returns=returns,
            corridor_semantic_corridor_mask=mask,
        )

        self.assertGreater(task_degradation, 0.0)
        self.assertTrue(torch.all(trigger_surface > 0.0).item())
        self.assertGreater(trigger_gate, 0.0)
        self.assertGreater(precontact_gate, 0.0)
        self.assertGreater(late_gate, 0.0)
        self.assertTrue(torch.all(regime_quality_gate < 0.35).item())

    def test_bootstrap_bonus_source_seed_quality_contract_suppresses_bad_dense_seed_before_late_window(self):
        external_seed = torch.tensor([[22.0, 18.0]], dtype=torch.float32)
        raw_anchor = torch.tensor([[6.0, 6.0]], dtype=torch.float32)
        internal_value = torch.tensor([[4.0, 5.0]], dtype=torch.float32)
        dense_only = torch.zeros_like(external_seed)
        low_regime = torch.full_like(external_seed, 0.18)
        high_gap = torch.full_like(external_seed, 14.0)
        low_truth = torch.full_like(external_seed, 0.12)
        low_alignment = torch.full_like(external_seed, 0.24)

        (
            gap_health,
            source_quality,
            seed_gate,
            seeded_value,
            delta_abs,
        ) = compute_bootstrap_bonus_source_seed_quality_contract(
            bootstrap_external_value_seed=external_seed,
            raw_anchor_bootstrap_values_next=raw_anchor,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=dense_only,
            critic_contract_bootstrap_regime_quality_gate=low_regime,
            critic_contract_bootstrap_raw_vs_clean_gap=high_gap,
            critic_contract_bootstrap_source_reward_semantic_truth=low_truth,
            critic_contract_bootstrap_source_semantic_alignment=low_alignment,
            critic_contract_task_degradation=0.75,
            critic_contract_bootstrap_late_gate=0.0,
        )

        self.assertTrue(torch.all(gap_health < 0.4).item())
        self.assertTrue(torch.all(source_quality < 0.2).item())
        self.assertTrue(torch.all(seed_gate < 0.1).item())
        self.assertTrue(torch.all(seeded_value < external_seed).item())
        self.assertTrue(torch.allclose(seeded_value, internal_value, atol=1.5))
        self.assertTrue(torch.all(delta_abs > 10.0).item())

    def test_bootstrap_external_seed_assembly_contract_extends_sparse_anchor_signal_across_row(self):
        raw_anchor = torch.tensor([[8.0, 0.0]], dtype=torch.float32)
        internal_value = torch.tensor([[4.0, 4.0]], dtype=torch.float32)
        anchor_available = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        anchor_confidence = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        assembled = compute_bootstrap_external_seed_assembly_contract(
            raw_anchor_bootstrap_values_next=raw_anchor,
            raw_bootstrap_values_next=internal_value,
            critic_anchor_available_next=anchor_available,
            critic_anchor_confidence_next=anchor_confidence,
        )

        self.assertTrue(
            torch.allclose(
                assembled.anchor_bootstrap_values_next,
                torch.tensor([[8.0, 8.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                assembled.raw_vs_clean_gap,
                torch.tensor([[4.0, 4.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                assembled.dense_surface_gain,
                torch.tensor([[2.0**-0.5]], dtype=torch.float32),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                assembled.dense_seed_support,
                torch.full_like(internal_value, 2.0**-0.5),
                atol=1e-5,
            )
        )

    def test_bootstrap_bonus_source_seed_quality_contract_preserves_healthy_anchor_valid_seed(self):
        external_seed = torch.tensor([[8.0, 7.0]], dtype=torch.float32)
        raw_anchor = external_seed.clone()
        internal_value = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        anchor_valid = torch.ones_like(external_seed)
        regime_quality = torch.full_like(external_seed, 0.92)
        raw_gap = torch.full_like(external_seed, 2.0)
        truth = torch.full_like(external_seed, 0.88)
        alignment = torch.full_like(external_seed, 0.93)

        (
            gap_health,
            source_quality,
            seed_gate,
            seeded_value,
            delta_abs,
        ) = compute_bootstrap_bonus_source_seed_quality_contract(
            bootstrap_external_value_seed=external_seed,
            raw_anchor_bootstrap_values_next=raw_anchor,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_regime_quality_gate=regime_quality,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_gap,
            critic_contract_bootstrap_source_reward_semantic_truth=truth,
            critic_contract_bootstrap_source_semantic_alignment=alignment,
            critic_contract_task_degradation=0.05,
            critic_contract_bootstrap_late_gate=0.0,
        )

        self.assertTrue(torch.all(gap_health > 0.7).item())
        self.assertTrue(torch.all(source_quality > 0.85).item())
        self.assertTrue(torch.allclose(seed_gate, torch.ones_like(seed_gate)))
        self.assertTrue(torch.allclose(seeded_value, external_seed))
        self.assertTrue(
            torch.allclose(delta_abs, torch.zeros_like(delta_abs), atol=1e-5)
        )

    def test_bootstrap_bonus_source_seed_quality_contract_keeps_sparse_row_support_subordinate_to_quality_gate(self):
        external_seed = torch.tensor([[8.0, 8.0]], dtype=torch.float32)
        raw_anchor = torch.tensor([[8.0, 0.0]], dtype=torch.float32)
        internal_value = torch.tensor([[4.0, 4.0]], dtype=torch.float32)
        anchor_valid = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        dense_seed_support = torch.tensor([[0.7, 0.7]], dtype=torch.float32)
        regime_quality = torch.full_like(external_seed, 0.65)
        raw_gap = torch.full_like(external_seed, 4.0)
        truth = torch.full_like(external_seed, 0.82)
        alignment = torch.full_like(external_seed, 0.90)

        (
            _gap_health,
            _source_quality,
            seed_gate,
            seeded_value,
            delta_abs,
        ) = compute_bootstrap_bonus_source_seed_quality_contract(
            bootstrap_external_value_seed=external_seed,
            raw_anchor_bootstrap_values_next=raw_anchor,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=anchor_valid,
            bootstrap_dense_seed_support=dense_seed_support,
            critic_contract_bootstrap_regime_quality_gate=regime_quality,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_gap,
            critic_contract_bootstrap_source_reward_semantic_truth=truth,
            critic_contract_bootstrap_source_semantic_alignment=alignment,
            critic_contract_task_degradation=0.05,
            critic_contract_bootstrap_late_gate=0.0,
        )

        self.assertTrue(torch.allclose(seed_gate[:, :1], torch.ones_like(seed_gate[:, :1])))
        self.assertGreater(float(seed_gate[:, 1:].mean().item()), 0.40)
        self.assertLess(float(seed_gate[:, 1:].mean().item()), 0.50)
        self.assertGreater(float(seeded_value[:, 1:].mean().item()), 5.5)
        self.assertLess(float(seeded_value[:, 1:].mean().item()), 6.0)
        self.assertGreater(float(delta_abs[:, 1:].mean().item()), 2.0)

    def test_bootstrap_bonus_source_seed_quality_contract_hands_off_dense_seed_inside_fully_late_window(self):
        external_seed = torch.tensor([[22.0, 18.0]], dtype=torch.float32)
        raw_anchor = torch.tensor([[6.0, 6.0]], dtype=torch.float32)
        internal_value = torch.tensor([[4.0, 5.0]], dtype=torch.float32)
        dense_only = torch.zeros_like(external_seed)
        low_regime = torch.full_like(external_seed, 0.18)
        high_gap = torch.full_like(external_seed, 14.0)
        low_truth = torch.full_like(external_seed, 0.12)
        low_alignment = torch.full_like(external_seed, 0.24)

        (
            _gap_health,
            _source_quality,
            seed_gate,
            seeded_value,
            delta_abs,
        ) = compute_bootstrap_bonus_source_seed_quality_contract(
            bootstrap_external_value_seed=external_seed,
            raw_anchor_bootstrap_values_next=raw_anchor,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=dense_only,
            critic_contract_bootstrap_regime_quality_gate=low_regime,
            critic_contract_bootstrap_raw_vs_clean_gap=high_gap,
            critic_contract_bootstrap_source_reward_semantic_truth=low_truth,
            critic_contract_bootstrap_source_semantic_alignment=low_alignment,
            critic_contract_task_degradation=0.75,
            critic_contract_bootstrap_late_gate=1.0,
        )

        self.assertTrue(torch.allclose(seed_gate, torch.ones_like(seed_gate)))
        self.assertTrue(torch.allclose(seeded_value, external_seed))
        self.assertTrue(
            torch.allclose(delta_abs, torch.zeros_like(delta_abs), atol=1e-5)
        )

    def test_bootstrap_bonus_source_seed_quality_contract_relaxes_moderately_healthy_dense_seed_inside_late_window_without_full_override(self):
        external_seed = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
        raw_anchor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        internal_value = torch.tensor([[4.0, 4.0]], dtype=torch.float32)
        dense_only = torch.zeros_like(external_seed)
        regime_quality = torch.full_like(external_seed, 0.72)
        raw_gap = torch.full_like(external_seed, 3.0)
        truth = torch.full_like(external_seed, 0.76)
        alignment = torch.full_like(external_seed, 0.82)

        (
            _gap_health,
            _source_quality,
            prelate_gate,
            _prelate_seeded_value,
            _prelate_delta_abs,
        ) = compute_bootstrap_bonus_source_seed_quality_contract(
            bootstrap_external_value_seed=external_seed,
            raw_anchor_bootstrap_values_next=raw_anchor,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=dense_only,
            critic_contract_bootstrap_regime_quality_gate=regime_quality,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_gap,
            critic_contract_bootstrap_source_reward_semantic_truth=truth,
            critic_contract_bootstrap_source_semantic_alignment=alignment,
            critic_contract_task_degradation=0.05,
            critic_contract_bootstrap_late_gate=0.0,
        )
        (
            _gap_health,
            _source_quality,
            late_gate,
            late_seeded_value,
            late_delta_abs,
        ) = compute_bootstrap_bonus_source_seed_quality_contract(
            bootstrap_external_value_seed=external_seed,
            raw_anchor_bootstrap_values_next=raw_anchor,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=dense_only,
            critic_contract_bootstrap_regime_quality_gate=regime_quality,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_gap,
            critic_contract_bootstrap_source_reward_semantic_truth=truth,
            critic_contract_bootstrap_source_semantic_alignment=alignment,
            critic_contract_task_degradation=0.05,
            critic_contract_bootstrap_late_gate=0.9,
        )

        self.assertTrue(torch.all(late_gate > prelate_gate).item())
        self.assertTrue(torch.all(late_gate < 1.0).item())
        self.assertTrue(torch.all(late_seeded_value > internal_value).item())
        self.assertTrue(torch.all(late_seeded_value < external_seed).item())
        self.assertTrue(torch.all(late_delta_abs < _prelate_delta_abs).item())

    def test_prelate_terminal_truth_source_consumes_seed_vetoed_live_value(self):
        external_seed = torch.tensor([[22.0, 18.0]], dtype=torch.float32)
        raw_anchor = torch.tensor([[6.0, 6.0]], dtype=torch.float32)
        internal_value = torch.tensor([[4.0, 5.0]], dtype=torch.float32)
        dense_only = torch.zeros_like(external_seed)
        low_regime = torch.full_like(external_seed, 0.18)
        high_gap = torch.full_like(external_seed, 14.0)
        low_truth = torch.full_like(external_seed, 0.12)
        low_alignment = torch.full_like(external_seed, 0.24)

        (
            _gap_health,
            _source_quality,
            _seed_gate,
            seeded_value,
            _seed_delta_abs,
        ) = compute_bootstrap_bonus_source_seed_quality_contract(
            bootstrap_external_value_seed=external_seed,
            raw_anchor_bootstrap_values_next=raw_anchor,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=dense_only,
            critic_contract_bootstrap_regime_quality_gate=low_regime,
            critic_contract_bootstrap_raw_vs_clean_gap=high_gap,
            critic_contract_bootstrap_source_reward_semantic_truth=low_truth,
            critic_contract_bootstrap_source_semantic_alignment=low_alignment,
            critic_contract_task_degradation=0.75,
            critic_contract_bootstrap_late_gate=0.0,
        )
        (
            _truth,
            _alignment,
            activation,
            activation_bridge,
            _relative_gap,
            _anchor_preference,
            terminal_source,
            terminal_delta_abs,
        ) = compute_bootstrap_bonus_terminal_truth_source_contract(
            bootstrap_external_value_raw=seeded_value,
            raw_anchor_bootstrap_values_next=raw_anchor,
            raw_bootstrap_values_next=internal_value,
            anchor_bootstrap_values_next=raw_anchor,
            critic_anchor_available_next=dense_only,
            reference_real_reward_agreement=low_truth,
            behavior_policy_task_cert_real_recovery=low_truth,
            behavior_policy_task_cert_real_alarm=torch.full_like(external_seed, 0.8),
            behavior_policy_task_cert_imag_gate=low_alignment,
            task_corridor_gate=low_alignment,
            critic_contract_task_degradation=torch.full_like(external_seed, 0.75),
            critic_contract_bootstrap_late_gate=0.0,
            critic_contract_bootstrap_raw_vs_clean_gap=high_gap,
            corridor_semantic_corridor_mask=torch.ones_like(external_seed),
        )

        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(
            torch.allclose(activation_bridge, torch.zeros_like(activation_bridge))
        )
        self.assertTrue(torch.allclose(terminal_source, seeded_value))
        self.assertTrue(torch.allclose(terminal_delta_abs, torch.zeros_like(terminal_delta_abs)))
        self.assertFalse(torch.allclose(seeded_value, external_seed))

    def test_bootstrap_final_external_value_quality_contract_suppresses_bad_dense_regime(self):
        retained = torch.tensor([[22.0, 20.0]], dtype=torch.float32)
        anchor_value = torch.tensor([[8.0, 7.5]], dtype=torch.float32)
        dense_only = torch.zeros_like(retained)
        low_regime = torch.full_like(retained, 0.18)
        high_gap = torch.full_like(retained, 14.0)
        low_truth = torch.full_like(retained, 0.12)
        low_alignment = torch.full_like(retained, 0.24)

        (
            regime_quality_gate,
            behavior_health,
            prehold_value_failfast_gate,
            floor_gate,
            floor_quality_clamped,
            floor_delta_abs,
            final_gate,
            quality_clamped,
            delta_abs,
        ) = compute_bootstrap_final_external_value_quality_contract(
            bootstrap_external_bonus_value_consumer_retained=retained,
            anchor_bootstrap_values_next=anchor_value,
            raw_bootstrap_values_next=torch.tensor([[4.0, 5.0]], dtype=torch.float32),
            bootstrap_anchor_valid_mask=dense_only,
            critic_contract_bootstrap_regime_quality_gate=low_regime,
            critic_contract_bootstrap_raw_vs_clean_gap=high_gap,
            critic_contract_bootstrap_source_reward_semantic_truth=low_truth,
            critic_contract_bootstrap_source_semantic_alignment=low_alignment,
            actor_contract_task_mismatch=0.29,
            actor_corridor_semantic_inflation_excess=11.0,
            critic_contract_bootstrap_late_gate=torch.full_like(retained, 0.45),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                retained
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(retained, 0.28),
            actor_task_geom_corridor_disagreement=torch.full_like(retained, 0.58),
        )

        self.assertTrue(torch.allclose(regime_quality_gate, low_regime))
        self.assertTrue(torch.all(behavior_health < 0.1).item())
        self.assertTrue(torch.all(prehold_value_failfast_gate < 0.3).item())
        self.assertTrue(torch.all(floor_gate < 0.05).item())
        self.assertTrue(torch.all(final_gate < 0.1).item())
        self.assertTrue(torch.all(floor_quality_clamped < anchor_value).item())
        self.assertTrue(
            torch.allclose(
                floor_quality_clamped,
                torch.tensor([[4.0, 5.0]], dtype=torch.float32),
                atol=1.5,
            )
        )
        self.assertTrue(torch.all(floor_delta_abs > 2.0).item())
        self.assertTrue(torch.all(quality_clamped < retained).item())
        self.assertTrue(
            torch.allclose(
                quality_clamped,
                floor_quality_clamped,
                atol=0.3,
            )
        )
        self.assertTrue(torch.all(delta_abs > 5.0).item())

    def test_bootstrap_final_external_value_quality_contract_preserves_healthy_anchor_valid_source(self):
        retained = torch.tensor([[12.0, 11.0]], dtype=torch.float32)
        anchor_value = torch.tensor([[7.0, 6.0]], dtype=torch.float32)
        anchor_valid = torch.ones_like(retained)
        regime_quality = torch.full_like(retained, 0.92)
        raw_gap = torch.full_like(retained, 3.0)
        truth = torch.full_like(retained, 0.72)
        alignment = torch.full_like(retained, 0.93)

        (
            _regime_quality_gate,
            behavior_health,
            prehold_value_failfast_gate,
            floor_gate,
            floor_quality_clamped,
            floor_delta_abs,
            final_gate,
            quality_clamped,
            delta_abs,
        ) = compute_bootstrap_final_external_value_quality_contract(
            bootstrap_external_bonus_value_consumer_retained=retained,
            anchor_bootstrap_values_next=anchor_value,
            raw_bootstrap_values_next=torch.tensor([[2.5, 2.0]], dtype=torch.float32),
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_regime_quality_gate=regime_quality,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_gap,
            critic_contract_bootstrap_source_reward_semantic_truth=truth,
            critic_contract_bootstrap_source_semantic_alignment=alignment,
            actor_contract_task_mismatch=0.03,
            actor_corridor_semantic_inflation_excess=1.0,
            critic_contract_bootstrap_late_gate=torch.full_like(retained, 0.45),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                retained
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(retained, 0.02),
            actor_task_geom_corridor_disagreement=torch.full_like(retained, 0.32),
        )

        self.assertTrue(torch.all(behavior_health > 0.85).item())
        self.assertTrue(torch.all(prehold_value_failfast_gate >= 0.95).item())
        self.assertTrue(torch.all(floor_gate > 0.85).item())
        self.assertTrue(torch.all(final_gate >= 0.85).item())
        self.assertTrue(torch.allclose(floor_quality_clamped, anchor_value, atol=1.0))
        self.assertTrue(torch.all(floor_delta_abs < 1.0).item())
        self.assertTrue(torch.allclose(quality_clamped, retained, atol=1.0))
        self.assertTrue(torch.all(delta_abs < 1.0).item())

    def test_bootstrap_final_external_value_quality_contract_preserves_anchor_floor_when_bonus_gate_is_lower(self):
        retained = torch.tensor([[12.0, 10.0]], dtype=torch.float32)
        anchor_value = torch.tensor([[7.0, 6.0]], dtype=torch.float32)
        internal_value = torch.tensor([[3.5, 3.0]], dtype=torch.float32)
        anchor_valid = torch.ones_like(retained)
        regime_quality = torch.full_like(retained, 0.55)
        raw_gap = torch.full_like(retained, 4.0)
        truth = torch.full_like(retained, 0.82)
        alignment = torch.full_like(retained, 0.91)

        (
            _regime_quality_gate,
            behavior_health,
            prehold_value_failfast_gate,
            floor_gate,
            floor_quality_clamped,
            floor_delta_abs,
            final_gate,
            quality_clamped,
            delta_abs,
        ) = compute_bootstrap_final_external_value_quality_contract(
            bootstrap_external_bonus_value_consumer_retained=retained,
            anchor_bootstrap_values_next=anchor_value,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_regime_quality_gate=regime_quality,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_gap,
            critic_contract_bootstrap_source_reward_semantic_truth=truth,
            critic_contract_bootstrap_source_semantic_alignment=alignment,
            actor_contract_task_mismatch=0.12,
            actor_corridor_semantic_inflation_excess=4.0,
            critic_contract_bootstrap_late_gate=torch.full_like(retained, 0.45),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                retained
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(retained, 0.03),
            actor_task_geom_corridor_disagreement=torch.full_like(retained, 0.36),
        )

        self.assertTrue(torch.all(behavior_health >= 0.6).item())
        self.assertTrue(torch.all(prehold_value_failfast_gate > 0.9).item())
        self.assertTrue(torch.all(floor_gate >= 0.9).item())
        self.assertTrue(torch.all(final_gate < floor_gate).item())
        self.assertTrue(torch.allclose(floor_quality_clamped, anchor_value, atol=0.8))
        self.assertTrue(torch.all(floor_delta_abs < 0.8).item())
        self.assertTrue(torch.all(quality_clamped >= anchor_value).item())
        self.assertTrue(torch.all(delta_abs > 0.5).item())

    def test_bootstrap_final_external_value_quality_contract_clamps_low_quality_anchor_valid_source(self):
        retained = torch.tensor([[18.0, 16.0]], dtype=torch.float32)
        anchor_value = torch.tensor([[10.0, 9.0]], dtype=torch.float32)
        internal_value = torch.tensor([[5.0, 5.5]], dtype=torch.float32)
        anchor_valid = torch.ones_like(retained)

        (
            _regime_quality_gate,
            _behavior_health,
            prehold_value_failfast_gate,
            floor_gate,
            floor_quality_clamped,
            floor_delta_abs,
            final_gate,
            quality_clamped,
            delta_abs,
        ) = compute_bootstrap_final_external_value_quality_contract(
            bootstrap_external_bonus_value_consumer_retained=retained,
            anchor_bootstrap_values_next=anchor_value,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                retained, 0.24
            ),
            critic_contract_bootstrap_raw_vs_clean_gap=torch.full_like(retained, 11.0),
            critic_contract_bootstrap_source_reward_semantic_truth=torch.full_like(
                retained, 0.22
            ),
            critic_contract_bootstrap_source_semantic_alignment=torch.full_like(
                retained, 0.30
            ),
            actor_contract_task_mismatch=0.26,
            actor_corridor_semantic_inflation_excess=8.0,
            critic_contract_bootstrap_late_gate=torch.full_like(retained, 0.35),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                retained
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(retained, 0.24),
            actor_task_geom_corridor_disagreement=torch.full_like(retained, 0.56),
        )

        self.assertTrue(torch.all(prehold_value_failfast_gate < 0.35).item())
        self.assertTrue(torch.all(floor_gate < 0.25).item())
        self.assertTrue(torch.all(final_gate <= 0.35 + 1e-6).item())
        self.assertTrue(torch.all(final_gate > floor_gate).item())
        self.assertTrue(torch.all(floor_quality_clamped < anchor_value).item())
        self.assertTrue(torch.allclose(floor_quality_clamped, internal_value, atol=1.2))
        self.assertTrue(torch.all(quality_clamped > floor_quality_clamped).item())
        self.assertTrue(torch.all(quality_clamped < retained).item())
        self.assertTrue(torch.all(floor_delta_abs >= 0.0).item())
        self.assertTrue(torch.all(delta_abs >= 0.0).item())

    def test_bootstrap_final_external_value_quality_contract_caps_late_hold_floor_above_retained_signal(self):
        retained = torch.tensor([[18.0, 16.0]], dtype=torch.float32)
        anchor_value = torch.tensor([[70.0, 68.0]], dtype=torch.float32)
        internal_value = torch.tensor([[64.0, 62.0]], dtype=torch.float32)
        anchor_valid = torch.ones_like(retained)

        (
            _regime_quality_gate,
            _behavior_health,
            prehold_value_failfast_gate,
            floor_gate,
            floor_quality_clamped,
            _floor_delta_abs,
            final_gate,
            quality_clamped,
            _delta_abs,
        ) = compute_bootstrap_final_external_value_quality_contract(
            bootstrap_external_bonus_value_consumer_retained=retained,
            anchor_bootstrap_values_next=anchor_value,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                retained, 0.20
            ),
            critic_contract_bootstrap_raw_vs_clean_gap=torch.full_like(retained, 10.0),
            critic_contract_bootstrap_source_reward_semantic_truth=torch.full_like(
                retained, 0.16
            ),
            critic_contract_bootstrap_source_semantic_alignment=torch.full_like(
                retained, 0.95
            ),
            actor_contract_task_mismatch=0.28,
            actor_corridor_semantic_inflation_excess=10.5,
            critic_contract_bootstrap_late_gate=torch.ones_like(retained),
            critic_contract_bootstrap_source_hold_window_activation=torch.ones_like(
                retained
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(retained, 0.25),
            actor_task_geom_corridor_disagreement=torch.full_like(retained, 0.56),
        )

        retained_floor_cap = retained + 0.25 * (internal_value - retained)

        self.assertTrue(torch.all(prehold_value_failfast_gate >= 0.95).item())
        self.assertTrue(torch.all(floor_gate < 0.2).item())
        self.assertTrue(torch.all(final_gate <= 0.3 + 1e-6).item())
        self.assertTrue(
            torch.all(
                floor_quality_clamped
                <= retained_floor_cap + 1e-6
            ).item()
        )
        self.assertTrue(
            torch.all(
                (floor_quality_clamped - retained).abs()
                < (floor_quality_clamped - internal_value).abs()
            ).item()
        )
        self.assertTrue(
            torch.all(
                (quality_clamped - retained).abs()
                < (quality_clamped - internal_value).abs()
            ).item()
        )

    def test_bootstrap_bonus_source_value_final_mix_applies_floor_quality_gate(self):
        raw_bootstrap_values_next = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
        anchor_bootstrap_values_next = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
        floor_value_quality_clamped = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
        bonus_source_value = torch.tensor([[7.0, 8.0]], dtype=torch.float32)
        floor_authority = torch.tensor([[0.5, 0.4]], dtype=torch.float32)
        final_external_authority = torch.tensor([[0.8, 0.7]], dtype=torch.float32)
        bonus_authority = (final_external_authority - floor_authority).clamp(min=0.0)

        external_value_final = torch.where(
            final_external_authority > 1e-6,
            (
                floor_authority * floor_value_quality_clamped
                + bonus_authority * bonus_source_value
            )
            / final_external_authority.clamp(min=1e-6),
            raw_bootstrap_values_next,
        )
        mixed_bootstrap_values_next = torch.lerp(
            raw_bootstrap_values_next,
            external_value_final,
            final_external_authority,
        )

        floor_only_external = torch.where(
            final_external_authority > 1e-6,
            (
                floor_authority * floor_value_quality_clamped
                + bonus_authority * floor_value_quality_clamped
            )
            / final_external_authority.clamp(min=1e-6),
            raw_bootstrap_values_next,
        )
        mixed_floor_only = torch.lerp(
            raw_bootstrap_values_next,
            floor_only_external,
            final_external_authority,
        )

        observed_floor_component = final_external_authority * (
            mixed_bootstrap_values_next - mixed_floor_only
        )

        self.assertTrue(torch.all(external_value_final >= 0.0).item())
        self.assertTrue(
            torch.allclose(
                observed_floor_component,
                final_external_authority
                * bonus_authority
                * (bonus_source_value - floor_value_quality_clamped),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.all(floor_value_quality_clamped >= anchor_bootstrap_values_next).item()
        )

    def test_bootstrap_authority_source_recertification_is_neutral_before_late_gate(self):
        raw_authority = torch.tensor([[0.7, 0.5]], dtype=torch.float32)
        floor = torch.tensor([[0.4, 0.3]], dtype=torch.float32)
        ones = torch.ones_like(raw_authority)
        zeros = torch.zeros_like(raw_authority)

        (
            _alarm,
            _persistence,
            gate_pure,
            gate_composite,
            prefinal,
            bonus_prefinal,
            _bonus_midlate_aligned,
            bonus_postrecert,
            _midlate_activation,
            _midlate_gate,
            _bonus_value_coupling_gate,
            _prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=ones,
            behavior_policy_task_cert_real_alarm=ones,
            behavior_policy_task_cert_real_recovery=ones,
            behavior_policy_task_cert_imag_gate=ones,
            behavior_policy_task_cert_support_mask=ones,
            task_corridor_gate=ones,
            critic_contract_task_degradation=zeros,
            critic_contract_bootstrap_late_gate=0.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(gate_pure, torch.ones_like(gate_pure)))
        self.assertTrue(torch.allclose(gate_composite, torch.ones_like(gate_composite)))
        self.assertTrue(torch.allclose(prefinal, raw_authority))
        self.assertTrue(torch.allclose(bonus_postrecert, bonus_prefinal))

    def test_bootstrap_authority_source_recertification_preserves_floor(self):
        raw_authority = torch.tensor([[0.75, 0.65]], dtype=torch.float32)
        floor = torch.tensor([[0.45, 0.35]], dtype=torch.float32)
        poor_gate = torch.full_like(raw_authority, 0.2)
        poor_support = torch.full_like(raw_authority, 0.1)
        poor_recovery = torch.full_like(raw_authority, 0.15)
        ones = torch.ones_like(raw_authority)

        (
            _alarm,
            _persistence,
            _gate_pure,
            _gate_composite,
            _prefinal,
            _bonus_prefinal,
            _bonus_midlate_aligned,
            bonus_postrecert,
            _midlate_activation,
            _midlate_gate,
            _bonus_value_coupling_gate,
            _prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=poor_gate,
            behavior_policy_task_cert_real_alarm=1.0 - poor_gate,
            behavior_policy_task_cert_real_recovery=poor_recovery,
            behavior_policy_task_cert_imag_gate=poor_gate,
            behavior_policy_task_cert_support_mask=poor_support,
            task_corridor_gate=poor_gate,
            critic_contract_task_degradation=torch.full_like(raw_authority, 0.8),
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )
        final_authority = (floor + bonus_postrecert).clamp(0.0, 0.8)

        self.assertTrue(torch.all(final_authority >= floor).item())
        self.assertTrue(torch.all(final_authority <= raw_authority).item())

    def test_bootstrap_authority_source_recertification_suppresses_bonus_under_late_window_task_drift(self):
        raw_authority = torch.tensor([[0.72, 0.68]], dtype=torch.float32)
        floor = torch.tensor([[0.4, 0.4]], dtype=torch.float32)
        poor_gate = torch.full_like(raw_authority, 0.2)
        poor_support = torch.full_like(raw_authority, 0.3)
        poor_recovery = torch.full_like(raw_authority, 0.25)
        ones = torch.ones_like(raw_authority)

        (
            alarm,
            persistence,
            gate_pure,
            gate_composite,
            _prefinal,
            bonus_prefinal,
            bonus_midlate_aligned,
            bonus_postrecert,
            _midlate_activation,
            _midlate_gate,
            bonus_value_coupling_gate,
            _prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=poor_gate,
            behavior_policy_task_cert_real_alarm=1.0 - poor_gate,
            behavior_policy_task_cert_real_recovery=poor_recovery,
            behavior_policy_task_cert_imag_gate=poor_gate,
            behavior_policy_task_cert_support_mask=poor_support,
            task_corridor_gate=poor_gate,
            critic_contract_task_degradation=torch.full_like(raw_authority, 0.9),
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(alarm > 0.0).item())
        self.assertTrue(torch.all(persistence < 0.3).item())
        self.assertTrue(torch.all(gate_pure < 1.0).item())
        self.assertTrue(torch.all(gate_composite <= gate_pure).item())
        self.assertTrue(torch.all(bonus_midlate_aligned >= bonus_postrecert).item())
        self.assertTrue(torch.all(bonus_value_coupling_gate <= 1.0).item())
        self.assertTrue(torch.all(bonus_postrecert < bonus_prefinal).item())

    def test_bootstrap_authority_source_recertification_keeps_floor_supported_bonus_above_raw_recert_gate(self):
        raw_authority = torch.tensor([[0.72, 0.68]], dtype=torch.float32)
        floor = torch.tensor([[0.4, 0.4]], dtype=torch.float32)
        poor_gate = torch.full_like(raw_authority, 0.2)
        poor_support = torch.full_like(raw_authority, 0.3)
        poor_recovery = torch.full_like(raw_authority, 0.25)
        ones = torch.ones_like(raw_authority)

        (
            alarm,
            persistence,
            gate_pure,
            gate_composite,
            _prefinal,
            bonus_prefinal,
            _bonus_midlate_aligned,
            bonus_postrecert,
            _midlate_activation,
            _midlate_gate,
            _bonus_value_coupling_gate,
            _prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=poor_gate,
            behavior_policy_task_cert_real_alarm=1.0 - poor_gate,
            behavior_policy_task_cert_real_recovery=poor_recovery,
            behavior_policy_task_cert_imag_gate=poor_gate,
            behavior_policy_task_cert_support_mask=poor_support,
            task_corridor_gate=poor_gate,
            critic_contract_task_degradation=torch.full_like(raw_authority, 0.9),
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )

        raw_recert_gate = (
            1.0
            - torch.maximum(
                alarm,
                (1.0 - persistence).clamp(0.0, 1.0),
            )
        ).clamp(0.0, 1.0)

        self.assertTrue(torch.all(gate_pure > raw_recert_gate).item())
        self.assertTrue(torch.all(gate_composite <= gate_pure).item())
        self.assertTrue(torch.all(bonus_postrecert > raw_recert_gate * bonus_prefinal).item())
        self.assertTrue(torch.all(bonus_postrecert < bonus_prefinal).item())

    def test_bootstrap_authority_source_recertification_recovers_headroom_above_floor_supported_bonus(self):
        raw_authority = torch.tensor([[0.82, 0.74]], dtype=torch.float32)
        floor = torch.tensor([[0.38, 0.34]], dtype=torch.float32)
        partial_gate = torch.full_like(raw_authority, 0.35)
        partial_support = torch.full_like(raw_authority, 0.8)
        partial_recovery = torch.full_like(raw_authority, 0.36)
        partial_imag = torch.full_like(raw_authority, 0.45)
        partial_task = torch.full_like(raw_authority, 0.42)
        ones = torch.ones_like(raw_authority)

        (
            _alarm,
            persistence,
            gate_pure,
            _gate_composite,
            prefinal,
            bonus_prefinal,
            _bonus_midlate_aligned,
            bonus_postrecert,
            _midlate_activation,
            _midlate_gate,
            _bonus_value_coupling_gate,
            _prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=partial_gate,
            behavior_policy_task_cert_real_alarm=1.0 - partial_gate,
            behavior_policy_task_cert_real_recovery=partial_recovery,
            behavior_policy_task_cert_imag_gate=partial_imag,
            behavior_policy_task_cert_support_mask=partial_support,
            task_corridor_gate=partial_task,
            critic_contract_task_degradation=torch.full_like(raw_authority, 0.35),
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )

        floor_supported_gate_floor = torch.sqrt((floor * persistence).clamp(0.0, 1.0))
        floor_supported_bonus = floor_supported_gate_floor * bonus_prefinal

        self.assertTrue(torch.all(gate_pure > floor_supported_gate_floor).item())
        self.assertTrue(torch.all(bonus_postrecert > floor_supported_bonus).item())
        self.assertTrue(torch.all(bonus_postrecert < bonus_prefinal).item())
        self.assertTrue(torch.all((prefinal - floor) > 0.0).item())

    def test_bootstrap_authority_source_recertification_preserves_bonus_under_healthy_task_cert(self):
        raw_authority = torch.tensor([[0.7, 0.6]], dtype=torch.float32)
        floor = torch.tensor([[0.35, 0.3]], dtype=torch.float32)
        healthy_gate = torch.full_like(raw_authority, 0.95)
        healthy_support = torch.full_like(raw_authority, 1.0)
        healthy_recovery = torch.full_like(raw_authority, 0.98)
        healthy_imag = torch.full_like(raw_authority, 0.96)
        healthy_task = torch.full_like(raw_authority, 0.94)
        near_zero = torch.full_like(raw_authority, 0.02)
        ones = torch.ones_like(raw_authority)

        (
            _alarm,
            _persistence,
            _gate_pure,
            gate_composite,
            _prefinal,
            bonus_prefinal,
            _bonus_midlate_aligned,
            bonus_postrecert,
            _midlate_activation,
            _midlate_gate,
            _bonus_value_coupling_gate,
            _prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=healthy_gate,
            behavior_policy_task_cert_real_alarm=near_zero,
            behavior_policy_task_cert_real_recovery=healthy_recovery,
            behavior_policy_task_cert_imag_gate=healthy_imag,
            behavior_policy_task_cert_support_mask=healthy_support,
            task_corridor_gate=healthy_task,
            critic_contract_task_degradation=near_zero,
            critic_contract_bootstrap_late_gate=1.0,
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(gate_composite > 0.9).item())
        self.assertTrue(torch.allclose(bonus_postrecert, bonus_prefinal, atol=5e-2))

    def test_bootstrap_authority_source_recertification_couples_bonus_to_value_side_health(self):
        raw_authority = torch.tensor([[0.82, 0.76]], dtype=torch.float32)
        floor = torch.tensor([[0.38, 0.34]], dtype=torch.float32)
        healthy_gate = torch.full_like(raw_authority, 0.95)
        healthy_support = torch.full_like(raw_authority, 1.0)
        healthy_recovery = torch.full_like(raw_authority, 0.97)
        healthy_imag = torch.full_like(raw_authority, 0.96)
        healthy_task = torch.full_like(raw_authority, 0.95)
        near_zero = torch.full_like(raw_authority, 0.02)
        ones = torch.ones_like(raw_authority)

        (
            _alarm_low,
            _persistence_low,
            _gate_pure_low,
            gate_composite_low,
            _prefinal_low,
            bonus_prefinal_low,
            bonus_midlate_aligned_low,
            bonus_postrecert_low,
            _midlate_activation_low,
            _midlate_gate_low,
            bonus_value_coupling_gate_low,
            _prehold_consistency_gate_low,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=healthy_gate,
            behavior_policy_task_cert_real_alarm=near_zero,
            behavior_policy_task_cert_real_recovery=healthy_recovery,
            behavior_policy_task_cert_imag_gate=healthy_imag,
            behavior_policy_task_cert_support_mask=healthy_support,
            task_corridor_gate=healthy_task,
            critic_contract_task_degradation=near_zero,
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                raw_authority, 0.06
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                raw_authority, 0.08
            ),
            critic_contract_bootstrap_source_hold_release_pressure=torch.full_like(
                raw_authority, 0.92
            ),
            critic_contract_bootstrap_source_hold_persistence_gate=torch.full_like(
                raw_authority, 0.12
            ),
            corridor_semantic_corridor_mask=ones,
        )
        (
            _alarm_healthy,
            _persistence_healthy,
            _gate_pure_healthy,
            gate_composite_healthy,
            _prefinal_healthy,
            bonus_prefinal_healthy,
            bonus_midlate_aligned_healthy,
            bonus_postrecert_healthy,
            _midlate_activation_healthy,
            _midlate_gate_healthy,
            bonus_value_coupling_gate_healthy,
            _prehold_consistency_gate_healthy,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=healthy_gate,
            behavior_policy_task_cert_real_alarm=near_zero,
            behavior_policy_task_cert_real_recovery=healthy_recovery,
            behavior_policy_task_cert_imag_gate=healthy_imag,
            behavior_policy_task_cert_support_mask=healthy_support,
            task_corridor_gate=healthy_task,
            critic_contract_task_degradation=near_zero,
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                raw_authority, 0.94
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                raw_authority, 0.96
            ),
            critic_contract_bootstrap_source_hold_release_pressure=torch.full_like(
                raw_authority, 0.05
            ),
            critic_contract_bootstrap_source_hold_persistence_gate=torch.full_like(
                raw_authority, 0.92
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.allclose(bonus_prefinal_low, bonus_prefinal_healthy))
        self.assertTrue(torch.all(gate_composite_low < gate_composite_healthy).item())
        self.assertTrue(
            torch.all(bonus_midlate_aligned_low >= bonus_postrecert_low).item()
        )
        self.assertTrue(
            torch.all(bonus_value_coupling_gate_low < bonus_value_coupling_gate_healthy).item()
        )
        self.assertTrue(
            torch.allclose(
                bonus_midlate_aligned_healthy,
                bonus_postrecert_healthy,
                atol=5e-2,
            )
        )
        self.assertTrue(torch.all(bonus_postrecert_low < bonus_postrecert_healthy).item())
        self.assertTrue(
            torch.all(bonus_postrecert_low < 0.5 * bonus_prefinal_low).item()
        )

    def test_bootstrap_authority_source_recertification_preserves_bonus_when_value_side_health_is_healthy(self):
        raw_authority = torch.tensor([[0.7, 0.6]], dtype=torch.float32)
        floor = torch.tensor([[0.35, 0.3]], dtype=torch.float32)
        healthy_gate = torch.full_like(raw_authority, 0.95)
        healthy_support = torch.full_like(raw_authority, 1.0)
        healthy_recovery = torch.full_like(raw_authority, 0.98)
        healthy_imag = torch.full_like(raw_authority, 0.96)
        healthy_task = torch.full_like(raw_authority, 0.94)
        near_zero = torch.full_like(raw_authority, 0.02)
        ones = torch.ones_like(raw_authority)

        (
            _alarm,
            _persistence,
            _gate_pure,
            gate_composite,
            _prefinal,
            bonus_prefinal,
            bonus_midlate_aligned,
            bonus_postrecert,
            _midlate_activation,
            _midlate_gate,
            bonus_value_coupling_gate,
            _prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=healthy_gate,
            behavior_policy_task_cert_real_alarm=near_zero,
            behavior_policy_task_cert_real_recovery=healthy_recovery,
            behavior_policy_task_cert_imag_gate=healthy_imag,
            behavior_policy_task_cert_support_mask=healthy_support,
            task_corridor_gate=healthy_task,
            critic_contract_task_degradation=near_zero,
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                raw_authority, 0.95
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                raw_authority, 0.97
            ),
            critic_contract_bootstrap_source_hold_release_pressure=torch.full_like(
                raw_authority, 0.04
            ),
            critic_contract_bootstrap_source_hold_persistence_gate=torch.full_like(
                raw_authority, 0.93
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(gate_composite > 0.9).item())
        self.assertTrue(torch.all(bonus_value_coupling_gate > 0.9).item())
        self.assertTrue(torch.allclose(bonus_midlate_aligned, bonus_postrecert, atol=5e-2))
        self.assertTrue(torch.allclose(bonus_postrecert, bonus_prefinal, atol=5e-2))

    def test_bootstrap_authority_source_recertification_midlate_gate_clamps_bonus_before_hold_arms(self):
        raw_authority = torch.tensor([[0.82, 0.76]], dtype=torch.float32)
        floor = torch.tensor([[0.38, 0.34]], dtype=torch.float32)
        healthy_gate = torch.full_like(raw_authority, 0.96)
        healthy_support = torch.full_like(raw_authority, 1.0)
        weak_recovery = torch.full_like(raw_authority, 0.28)
        healthy_imag = torch.full_like(raw_authority, 0.95)
        healthy_task = torch.full_like(raw_authority, 0.93)
        near_zero = torch.full_like(raw_authority, 0.02)
        ones = torch.ones_like(raw_authority)

        (
            _alarm,
            persistence,
            gate_pure,
            gate_composite,
            _prefinal,
            bonus_prefinal,
            bonus_midlate_aligned,
            bonus_postrecert,
            midlate_activation,
            midlate_gate,
            bonus_value_coupling_gate,
            _prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=healthy_gate,
            behavior_policy_task_cert_real_alarm=near_zero,
            behavior_policy_task_cert_real_recovery=weak_recovery,
            behavior_policy_task_cert_imag_gate=healthy_imag,
            behavior_policy_task_cert_support_mask=healthy_support,
            task_corridor_gate=healthy_task,
            critic_contract_task_degradation=near_zero,
            critic_contract_bootstrap_late_gate=torch.full_like(raw_authority, 0.5),
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                raw_authority, 0.02
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                raw_authority, 0.03
            ),
            critic_contract_bootstrap_source_hold_release_pressure=torch.full_like(
                raw_authority, 0.12
            ),
            critic_contract_bootstrap_source_hold_persistence_gate=torch.zeros_like(
                raw_authority
            ),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                raw_authority
            ),
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                raw_authority, 0.22
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(midlate_activation > 0.6).item())
        self.assertTrue(torch.all(persistence < 0.35).item())
        self.assertTrue(torch.all(midlate_gate < 0.5).item())
        self.assertTrue(torch.all(gate_pure > gate_composite).item())
        self.assertTrue(
            torch.allclose(
                gate_composite,
                gate_pure * midlate_gate * bonus_value_coupling_gate,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                bonus_postrecert,
                bonus_midlate_aligned * bonus_value_coupling_gate,
                atol=1e-5,
            )
        )
        self.assertTrue(torch.all(bonus_value_coupling_gate > 0.8).item())
        self.assertTrue(torch.all(gate_composite < 0.5).item())
        self.assertTrue(torch.all(bonus_postrecert < 0.5 * bonus_prefinal).item())
        final_authority = (floor + bonus_postrecert).clamp(0.0, 1.0)
        self.assertTrue(torch.all(final_authority >= floor).item())

    def test_bootstrap_authority_source_recertification_midlate_gate_preserves_healthy_half_open_bonus(self):
        raw_authority = torch.tensor([[0.82, 0.76]], dtype=torch.float32)
        floor = torch.tensor([[0.38, 0.34]], dtype=torch.float32)
        healthy_gate = torch.full_like(raw_authority, 0.96)
        healthy_support = torch.full_like(raw_authority, 1.0)
        healthy_recovery = torch.full_like(raw_authority, 0.92)
        healthy_imag = torch.full_like(raw_authority, 0.95)
        healthy_task = torch.full_like(raw_authority, 0.93)
        near_zero = torch.full_like(raw_authority, 0.02)
        ones = torch.ones_like(raw_authority)

        (
            _alarm,
            _persistence,
            gate_pure,
            gate_composite,
            _prefinal,
            bonus_prefinal,
            bonus_midlate_aligned,
            bonus_postrecert,
            midlate_activation,
            midlate_gate,
            bonus_value_coupling_gate,
            _prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=healthy_gate,
            behavior_policy_task_cert_real_alarm=near_zero,
            behavior_policy_task_cert_real_recovery=healthy_recovery,
            behavior_policy_task_cert_imag_gate=healthy_imag,
            behavior_policy_task_cert_support_mask=healthy_support,
            task_corridor_gate=healthy_task,
            critic_contract_task_degradation=near_zero,
            critic_contract_bootstrap_late_gate=torch.full_like(raw_authority, 0.5),
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                raw_authority, 0.94
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                raw_authority, 0.95
            ),
            critic_contract_bootstrap_source_hold_release_pressure=torch.full_like(
                raw_authority, 0.08
            ),
            critic_contract_bootstrap_source_hold_persistence_gate=torch.zeros_like(
                raw_authority
            ),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                raw_authority
            ),
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                raw_authority, 0.93
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(midlate_activation > 0.6).item())
        self.assertTrue(torch.all(midlate_gate > 0.9).item())
        self.assertTrue(torch.all(gate_composite > 0.8).item())
        self.assertTrue(
            torch.allclose(
                gate_composite,
                gate_pure * midlate_gate * bonus_value_coupling_gate,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                bonus_postrecert,
                bonus_midlate_aligned * bonus_value_coupling_gate,
                atol=5e-2,
            )
        )
        self.assertTrue(torch.all(bonus_value_coupling_gate > 0.8).item())
        self.assertTrue(torch.all(bonus_postrecert > 0.8 * bonus_prefinal).item())

    def test_bootstrap_authority_source_recertification_starts_clamping_in_early_midlate_when_value_gates_collapse(self):
        raw_authority = torch.tensor([[0.78, 0.74]], dtype=torch.float32)
        floor = torch.tensor([[0.46, 0.44]], dtype=torch.float32)
        ones = torch.ones_like(raw_authority)

        (
            _alarm,
            _persistence,
            gate_pure,
            gate_composite,
            _prefinal,
            bonus_prefinal,
            bonus_midlate_aligned,
            bonus_postrecert,
            midlate_activation,
            midlate_gate,
            bonus_value_coupling_gate,
            _prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=ones,
            behavior_policy_task_cert_real_alarm=torch.zeros_like(raw_authority),
            behavior_policy_task_cert_real_recovery=torch.full_like(raw_authority, 0.28),
            behavior_policy_task_cert_imag_gate=torch.full_like(raw_authority, 0.92),
            behavior_policy_task_cert_support_mask=ones,
            task_corridor_gate=torch.full_like(raw_authority, 0.91),
            critic_contract_task_degradation=torch.zeros_like(raw_authority),
            critic_contract_bootstrap_late_gate=torch.full_like(raw_authority, 0.12),
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(raw_authority, 0.05),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(raw_authority, 0.08),
            critic_contract_bootstrap_source_hold_release_pressure=torch.zeros_like(raw_authority),
            critic_contract_bootstrap_source_hold_persistence_gate=torch.zeros_like(raw_authority),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(raw_authority),
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(raw_authority, 0.26),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(midlate_activation > 0.1).item())
        self.assertTrue(torch.all(midlate_gate < 0.9).item())
        self.assertTrue(torch.all(bonus_midlate_aligned < bonus_prefinal).item())
        self.assertTrue(torch.allclose(gate_composite, gate_pure * midlate_gate * bonus_value_coupling_gate, atol=1e-6))
        self.assertTrue(torch.all(bonus_postrecert < bonus_prefinal).item())

    def test_bootstrap_authority_source_recertification_reports_prehold_mismatch_without_live_clamp(self):
        raw_authority = torch.tensor([[0.78, 0.74]], dtype=torch.float32)
        floor = torch.tensor([[0.46, 0.44]], dtype=torch.float32)
        ones = torch.ones_like(raw_authority)

        (
            _alarm,
            _persistence,
            gate_pure,
            gate_composite,
            _prefinal,
            bonus_prefinal,
            bonus_midlate_aligned,
            bonus_postrecert,
            _midlate_activation,
            midlate_gate,
            bonus_value_coupling_gate,
            prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=ones,
            behavior_policy_task_cert_real_alarm=torch.zeros_like(raw_authority),
            behavior_policy_task_cert_real_recovery=torch.full_like(raw_authority, 0.92),
            behavior_policy_task_cert_imag_gate=torch.full_like(raw_authority, 0.94),
            behavior_policy_task_cert_support_mask=ones,
            task_corridor_gate=torch.full_like(raw_authority, 0.92),
            critic_contract_task_degradation=torch.zeros_like(raw_authority),
            critic_contract_bootstrap_late_gate=torch.full_like(raw_authority, 0.2),
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                raw_authority, 0.04
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                raw_authority, 0.07
            ),
            critic_contract_bootstrap_source_hold_release_pressure=torch.zeros_like(
                raw_authority
            ),
            critic_contract_bootstrap_source_hold_persistence_gate=torch.zeros_like(
                raw_authority
            ),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                raw_authority
            ),
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                raw_authority, 0.28
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(raw_authority, 0.24),
            actor_task_geom_corridor_disagreement=torch.full_like(raw_authority, 0.54),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(prehold_consistency_gate < 0.5).item())
        self.assertTrue(
            torch.allclose(
                gate_composite,
                gate_pure * midlate_gate * bonus_value_coupling_gate,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                bonus_postrecert,
                bonus_midlate_aligned * bonus_value_coupling_gate,
                atol=5e-2,
            )
        )
        self.assertTrue(torch.all(bonus_postrecert > 0.5 * bonus_prefinal).item())
        final_authority = (floor + bonus_postrecert).clamp(0.0, 1.0)
        self.assertTrue(torch.all(final_authority >= floor).item())

    def test_bootstrap_authority_source_recertification_preserves_healthy_prehold_bonus(self):
        raw_authority = torch.tensor([[0.78, 0.74]], dtype=torch.float32)
        floor = torch.tensor([[0.46, 0.44]], dtype=torch.float32)
        ones = torch.ones_like(raw_authority)

        (
            _alarm,
            _persistence,
            gate_pure,
            gate_composite,
            _prefinal,
            bonus_prefinal,
            bonus_midlate_aligned,
            bonus_postrecert,
            _midlate_activation,
            _midlate_gate,
            _bonus_value_coupling_gate,
            prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=ones,
            behavior_policy_task_cert_real_alarm=torch.zeros_like(raw_authority),
            behavior_policy_task_cert_real_recovery=torch.full_like(raw_authority, 0.95),
            behavior_policy_task_cert_imag_gate=torch.full_like(raw_authority, 0.95),
            behavior_policy_task_cert_support_mask=ones,
            task_corridor_gate=torch.full_like(raw_authority, 0.94),
            critic_contract_task_degradation=torch.zeros_like(raw_authority),
            critic_contract_bootstrap_late_gate=torch.full_like(raw_authority, 0.2),
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                raw_authority, 0.92
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                raw_authority, 0.95
            ),
            critic_contract_bootstrap_source_hold_release_pressure=torch.zeros_like(
                raw_authority
            ),
            critic_contract_bootstrap_source_hold_persistence_gate=torch.zeros_like(
                raw_authority
            ),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                raw_authority
            ),
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                raw_authority, 0.92
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(raw_authority, 0.02),
            actor_task_geom_corridor_disagreement=torch.full_like(raw_authority, 0.32),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(prehold_consistency_gate > 0.95).item())
        self.assertTrue(torch.allclose(gate_composite, gate_pure, atol=5e-2))
        self.assertTrue(torch.allclose(bonus_postrecert, bonus_midlate_aligned, atol=5e-2))
        self.assertTrue(torch.allclose(bonus_postrecert, bonus_prefinal, atol=5e-2))

    def test_bootstrap_authority_source_recertification_midlate_gate_yields_to_late_hold_phase(self):
        raw_authority = torch.tensor([[0.82, 0.76]], dtype=torch.float32)
        floor = torch.tensor([[0.38, 0.34]], dtype=torch.float32)
        healthy_gate = torch.full_like(raw_authority, 0.96)
        healthy_support = torch.full_like(raw_authority, 1.0)
        healthy_recovery = torch.full_like(raw_authority, 0.92)
        healthy_imag = torch.full_like(raw_authority, 0.95)
        healthy_task = torch.full_like(raw_authority, 0.93)
        near_zero = torch.full_like(raw_authority, 0.02)
        ones = torch.ones_like(raw_authority)

        (
            _alarm,
            _persistence,
            gate_pure,
            gate_composite,
            _prefinal,
            bonus_prefinal,
            bonus_midlate_aligned,
            bonus_postrecert,
            midlate_activation,
            midlate_gate,
            bonus_value_coupling_gate,
            _prehold_consistency_gate,
        ) = _compute_bootstrap_authority_source_recertification_contract(
            bootstrap_external_authority_raw=raw_authority,
            bootstrap_external_authority_floor=floor,
            behavior_policy_task_cert_real_gate=healthy_gate,
            behavior_policy_task_cert_real_alarm=near_zero,
            behavior_policy_task_cert_real_recovery=healthy_recovery,
            behavior_policy_task_cert_imag_gate=healthy_imag,
            behavior_policy_task_cert_support_mask=healthy_support,
            task_corridor_gate=healthy_task,
            critic_contract_task_degradation=near_zero,
            critic_contract_bootstrap_late_gate=torch.full_like(raw_authority, 1.0),
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                raw_authority, 0.03
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                raw_authority, 0.04
            ),
            critic_contract_bootstrap_source_hold_release_pressure=torch.full_like(
                raw_authority, 0.7
            ),
            critic_contract_bootstrap_source_hold_persistence_gate=torch.full_like(
                raw_authority, 0.8
            ),
            critic_contract_bootstrap_source_hold_window_activation=torch.ones_like(
                raw_authority
            ),
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                raw_authority, 0.22
            ),
            corridor_semantic_corridor_mask=ones,
        )

        self.assertTrue(torch.all(midlate_activation < 0.05).item())
        self.assertTrue(torch.all(midlate_gate > 0.95).item())
        self.assertTrue(torch.all(gate_composite > 0.9).item())
        self.assertTrue(torch.all(gate_composite <= gate_pure).item())
        self.assertTrue(torch.all(bonus_value_coupling_gate > 0.95).item())
        self.assertTrue(torch.allclose(bonus_midlate_aligned, bonus_postrecert, atol=5e-2))
        self.assertTrue(torch.allclose(bonus_postrecert, bonus_prefinal, atol=5e-2))

    def test_bootstrap_external_authority_floor_live_unwind_contract_unwinds_stale_midlate_floor(self):
        floor = torch.tensor([[0.78, 0.74]], dtype=torch.float32)

        (
            activation,
            unwind_gate,
            unwound_floor,
            delta_abs,
        ) = compute_bootstrap_external_authority_floor_live_unwind_contract(
            bootstrap_external_authority_floor=floor,
            critic_contract_bootstrap_requested_floor=floor,
            critic_contract_bootstrap_late_gate=torch.full_like(floor, 0.5),
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                floor, 0.04
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                floor, 0.07
            ),
            critic_contract_bootstrap_source_recert_persistence=torch.full_like(
                floor, 0.18
            ),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                floor
            ),
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(floor, 0.24),
        )

        self.assertTrue(torch.all(activation > 0.85).item())
        self.assertTrue(torch.all(unwind_gate < 0.6).item())
        self.assertTrue(torch.all(unwound_floor < floor).item())
        self.assertTrue(torch.all(delta_abs > 0.0).item())

    def test_bootstrap_external_authority_floor_live_unwind_contract_preserves_healthy_pre_midlate_floor(self):
        floor = torch.tensor([[0.78, 0.74]], dtype=torch.float32)

        (
            activation,
            unwind_gate,
            unwound_floor,
            delta_abs,
        ) = compute_bootstrap_external_authority_floor_live_unwind_contract(
            bootstrap_external_authority_floor=floor,
            critic_contract_bootstrap_requested_floor=floor,
            critic_contract_bootstrap_late_gate=torch.full_like(floor, 0.02),
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                floor, 0.92
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                floor, 0.95
            ),
            critic_contract_bootstrap_source_recert_persistence=torch.full_like(
                floor, 0.94
            ),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                floor
            ),
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(floor, 0.93),
        )

        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(torch.all(unwind_gate > 0.95).item())
        self.assertTrue(torch.allclose(unwound_floor, floor, atol=5e-2))
        self.assertTrue(torch.all(delta_abs < 0.05).item())

    def test_bootstrap_internal_value_relief_contract_clamps_midlate_internal_branch_when_value_gates_collapse(self):
        raw_internal = torch.tensor([[14.0, 12.0]], dtype=torch.float32)
        external_final = torch.tensor([[8.0, 7.0]], dtype=torch.float32)

        relief_gate, relieved_value, delta_abs = compute_bootstrap_internal_value_relief_contract(
            raw_bootstrap_values_next=raw_internal,
            bootstrap_external_value_final=external_final,
            critic_contract_bootstrap_requested_floor=0.78,
            critic_contract_bootstrap_late_gate=0.45,
            critic_contract_bootstrap_floor_value_injection_gate=0.04,
            critic_contract_bootstrap_final_value_injection_gate=0.07,
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(raw_internal),
            critic_contract_bootstrap_regime_quality_gate=0.22,
            critic_contract_bootstrap_real_value_cap=6.0,
            actor_high_value_but_low_task_fraction=0.31,
            actor_task_geom_corridor_disagreement=0.63,
        )

        self.assertTrue(torch.all(relief_gate > 0.2).item())
        self.assertTrue(torch.all(relieved_value < raw_internal).item())
        self.assertTrue(torch.all(delta_abs > 0.0).item())
        self.assertTrue(torch.all(relieved_value > 5.5).item())

    def test_bootstrap_internal_value_relief_contract_preserves_healthy_branch(self):
        raw_internal = torch.tensor([[11.0, 10.0]], dtype=torch.float32)
        external_final = torch.tensor([[8.0, 7.0]], dtype=torch.float32)

        relief_gate, relieved_value, delta_abs = compute_bootstrap_internal_value_relief_contract(
            raw_bootstrap_values_next=raw_internal,
            bootstrap_external_value_final=external_final,
            critic_contract_bootstrap_requested_floor=0.78,
            critic_contract_bootstrap_late_gate=0.45,
            critic_contract_bootstrap_floor_value_injection_gate=0.92,
            critic_contract_bootstrap_final_value_injection_gate=0.95,
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(raw_internal),
            critic_contract_bootstrap_regime_quality_gate=0.93,
            critic_contract_bootstrap_real_value_cap=5.0,
            actor_high_value_but_low_task_fraction=0.0,
            actor_task_geom_corridor_disagreement=0.30,
        )

        self.assertTrue(torch.all(relief_gate <= 0.05).item())
        self.assertTrue(torch.all(relieved_value > (raw_internal - 0.6)).item())
        self.assertTrue(torch.all(delta_abs < 0.6).item())

    def test_bootstrap_internal_value_relief_contract_uses_real_cap_as_extra_grounding(self):
        raw_internal = torch.tensor([[14.0, 12.0]], dtype=torch.float32)
        external_final = torch.tensor([[8.0, 7.5]], dtype=torch.float32)
        common_kwargs = {
            "raw_bootstrap_values_next": raw_internal,
            "bootstrap_external_value_final": external_final,
            "critic_contract_bootstrap_requested_floor": 0.82,
            "critic_contract_bootstrap_late_gate": 0.62,
            "critic_contract_bootstrap_floor_value_injection_gate": 0.05,
            "critic_contract_bootstrap_final_value_injection_gate": 0.08,
            "critic_contract_bootstrap_source_hold_window_activation": torch.zeros_like(
                raw_internal
            ),
            "critic_contract_bootstrap_regime_quality_gate": 0.18,
            "actor_high_value_but_low_task_fraction": 0.32,
            "actor_task_geom_corridor_disagreement": 0.61,
        }

        _, relieved_without_cap, _ = compute_bootstrap_internal_value_relief_contract(
            **common_kwargs,
            critic_contract_bootstrap_real_value_cap=None,
        )
        relief_gate_with_cap, relieved_with_cap, delta_abs_with_cap = (
            compute_bootstrap_internal_value_relief_contract(
                **common_kwargs,
                critic_contract_bootstrap_real_value_cap=6.2,
            )
        )

        self.assertTrue(torch.all(relief_gate_with_cap > 0.35).item())
        self.assertTrue(torch.all(relieved_with_cap < relieved_without_cap).item())
        self.assertTrue(torch.all(delta_abs_with_cap > 0.0).item())

    def test_bootstrap_prehold_target_cap_contract_compresses_hot_tail_after_internal_pessimism_activates(self):
        mixed_target = torch.tensor([[19.0, 18.0]], dtype=torch.float32)
        internal_value = torch.tensor([[15.0, 14.0]], dtype=torch.float32)

        cap_gate, safe_cap, capped_target, delta_abs = (
            compute_bootstrap_prehold_target_cap_contract(
                mixed_bootstrap_values_next=mixed_target,
                bootstrap_internal_value_view=internal_value,
                critic_contract_bootstrap_late_gate=0.40,
                critic_contract_bootstrap_source_hold_window_activation=0.0,
                critic_contract_bootstrap_post_transition_window_activation=0.0,
                critic_contract_bootstrap_floor_value_injection_gate=0.05,
                critic_contract_bootstrap_final_value_injection_gate=0.08,
                critic_contract_bootstrap_regime_quality_gate=0.20,
                critic_contract_bootstrap_internal_pessimism_gate=0.55,
                critic_contract_bootstrap_real_value_cap=14.5,
            )
        )

        self.assertTrue(torch.all(cap_gate > 0.4).item())
        self.assertTrue(torch.allclose(safe_cap, torch.tensor([[15.0, 14.5]]), atol=1e-6))
        self.assertTrue(torch.all(capped_target < mixed_target).item())
        self.assertTrue(torch.all(capped_target >= safe_cap).item())
        self.assertTrue(torch.all(delta_abs > 0.0).item())

    def test_bootstrap_prehold_target_cap_contract_stays_idle_when_prehold_window_is_closed(self):
        mixed_target = torch.tensor([[16.0, 15.5]], dtype=torch.float32)
        internal_value = torch.tensor([[14.0, 13.5]], dtype=torch.float32)

        cap_gate, safe_cap, capped_target, delta_abs = (
            compute_bootstrap_prehold_target_cap_contract(
                mixed_bootstrap_values_next=mixed_target,
                bootstrap_internal_value_view=internal_value,
                critic_contract_bootstrap_late_gate=1.0,
                critic_contract_bootstrap_source_hold_window_activation=1.0,
                critic_contract_bootstrap_post_transition_window_activation=1.0,
                critic_contract_bootstrap_floor_value_injection_gate=0.05,
                critic_contract_bootstrap_final_value_injection_gate=0.08,
                critic_contract_bootstrap_regime_quality_gate=0.20,
                critic_contract_bootstrap_internal_pessimism_gate=0.70,
                critic_contract_bootstrap_real_value_cap=13.0,
            )
        )

        self.assertTrue(torch.allclose(cap_gate, torch.zeros_like(cap_gate), atol=1e-6))
        self.assertTrue(torch.allclose(safe_cap, internal_value, atol=1e-6))
        self.assertTrue(torch.allclose(capped_target, mixed_target, atol=1e-6))
        self.assertTrue(torch.all(delta_abs < 1e-6).item())

    def test_bootstrap_midlate_external_value_ratio_cap_contract_clamps_hot_external_value_during_floor_unwind(self):
        external_value = torch.tensor([[10.5, 10.0]], dtype=torch.float32)
        internal_value = torch.tensor([[14.5, 14.0]], dtype=torch.float32)

        (
            activation,
            max_ratio,
            capped_value,
            delta_abs,
        ) = compute_bootstrap_midlate_external_value_ratio_cap_contract(
            bootstrap_external_value_final=external_value,
            raw_bootstrap_values_next=internal_value,
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                external_value, 0.03
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                external_value, 0.07
            ),
            critic_contract_bootstrap_midlate_floor_unwind_activation=torch.full_like(
                external_value, 0.8
            ),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                external_value
            ),
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                external_value, 0.24
            ),
        )

        self.assertTrue(torch.all(activation > 0.7).item())
        self.assertTrue(torch.all(max_ratio < 0.6).item())
        self.assertTrue(torch.all(capped_value < external_value).item())
        self.assertTrue(torch.all(delta_abs > 0.0).item())

    def test_bootstrap_midlate_external_value_ratio_cap_contract_stays_idle_when_hold_is_armed(self):
        external_value = torch.tensor([[6.0, 5.5]], dtype=torch.float32)
        internal_value = torch.tensor([[14.0, 13.0]], dtype=torch.float32)

        (
            activation,
            max_ratio,
            capped_value,
            delta_abs,
        ) = compute_bootstrap_midlate_external_value_ratio_cap_contract(
            bootstrap_external_value_final=external_value,
            raw_bootstrap_values_next=internal_value,
            critic_contract_bootstrap_floor_value_injection_gate=torch.full_like(
                external_value, 0.03
            ),
            critic_contract_bootstrap_final_value_injection_gate=torch.full_like(
                external_value, 0.07
            ),
            critic_contract_bootstrap_midlate_floor_unwind_activation=torch.full_like(
                external_value, 0.8
            ),
            critic_contract_bootstrap_source_hold_window_activation=torch.ones_like(
                external_value
            ),
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                external_value, 0.24
            ),
        )

        self.assertTrue(torch.allclose(activation, torch.zeros_like(activation)))
        self.assertTrue(torch.allclose(capped_value, external_value, atol=1e-6))
        self.assertTrue(torch.all(delta_abs < 1e-6).item())
        self.assertTrue(torch.all(max_ratio > 0.65).item())

    def test_bootstrap_target_execution_contract_enforces_single_ordered_execution_path(self):
        result = compute_bootstrap_target_execution_contract(
            bootstrap_external_authority_view=torch.tensor([[0.6, 0.2]], dtype=torch.float32),
            bootstrap_external_authority_floor=torch.tensor([[0.4, 0.1]], dtype=torch.float32),
            bootstrap_external_floor_value_clamped=torch.tensor([[9.0, 8.0]], dtype=torch.float32),
            bootstrap_external_bonus_value_quality_clamped=torch.tensor([[12.0, 11.0]], dtype=torch.float32),
            raw_bootstrap_values_next=torch.tensor([[15.0, 14.0]], dtype=torch.float32),
            critic_contract_bootstrap_floor_value_injection_gate=torch.tensor([[0.04, 0.05]], dtype=torch.float32),
            critic_contract_bootstrap_final_value_injection_gate=torch.tensor([[0.07, 0.08]], dtype=torch.float32),
            critic_contract_bootstrap_midlate_floor_unwind_activation=torch.tensor([[0.8, 0.8]], dtype=torch.float32),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros((1, 2), dtype=torch.float32),
            critic_contract_bootstrap_regime_quality_gate=torch.tensor([[0.24, 0.24]], dtype=torch.float32),
            critic_contract_bootstrap_requested_floor=torch.tensor([[0.78, 0.78]], dtype=torch.float32),
            critic_contract_bootstrap_late_gate=0.4,
            critic_contract_bootstrap_post_transition_window_activation=0.0,
            critic_contract_bootstrap_real_value_cap=torch.tensor([[14.5, 13.5]], dtype=torch.float32),
            actor_high_value_but_low_task_fraction=0.31,
            actor_task_geom_corridor_disagreement=0.63,
        )

        expected_prefinal = torch.tensor([[10.0, 9.5]], dtype=torch.float32)
        self.assertTrue(
            torch.allclose(result.external_value_prefinal, expected_prefinal, atol=1e-6)
        )
        self.assertTrue(torch.all(result.external_value_final <= result.external_value_prefinal).item())
        self.assertTrue(torch.all(result.internal_value_view <= torch.tensor([[15.0, 14.0]])).item())

        expected_mixed_prefinal = torch.lerp(
            result.internal_value_view,
            result.external_value_final,
            torch.tensor([[0.6, 0.2]], dtype=torch.float32),
        )
        self.assertTrue(
            torch.allclose(result.mixed_value_prefinal, expected_mixed_prefinal, atol=1e-6)
        )
        self.assertTrue(torch.all(result.mixed_value_final <= result.mixed_value_prefinal).item())
        self.assertTrue(torch.all(result.prehold_target_cap_gate > 0.0).item())
        self.assertTrue(torch.all(result.external_value_midlate_ratio_cap_delta_abs > 0.0).item())
        self.assertTrue(torch.all(result.internal_value_relief_delta_abs > 0.0).item())

    def test_bootstrap_precontact_floor_frontloads_anchor_contact_to_prefix(self):
        priority = torch.ones((1, 4), dtype=torch.float32)
        capacity = torch.full((1, 4), 0.5, dtype=torch.float32)
        budget_sum = torch.tensor([[1.0]], dtype=torch.float32)

        neutral = _allocate_priority_budget_with_capacity(
            priority,
            capacity,
            budget_sum,
        )
        frontload = 0.8
        positions = torch.linspace(0.0, 1.0, steps=4, dtype=torch.float32).view(1, -1)
        head_profile = (1.0 - frontload * positions).clamp(
            min=1.0 - frontload,
            max=1.0,
        )
        frontloaded = _allocate_priority_budget_with_capacity(
            priority * head_profile,
            capacity,
            budget_sum,
        )

        self.assertGreater(
            float(frontloaded[:, 0].mean().item()),
            float(neutral[:, 0].mean().item()),
        )
        self.assertLess(
            float(frontloaded[:, -1].mean().item()),
            float(neutral[:, -1].mean().item()),
        )
        self.assertGreater(
            float(frontloaded[:, :2].sum(dim=1).mean().item()),
            float(neutral[:, :2].sum(dim=1).mean().item()),
        )

    def test_bootstrap_surface_bonus_stays_local_to_realized_base_contact(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )
        loop.config.adaptive_imag_critic_bootstrap_precontact_frontload = 0.9
        loop.config.adaptive_imag_critic_bootstrap_precontact_floor_max = 0.15

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertGreater(
            float(batch["critic_contract_bootstrap_surface_bonus_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_bonus_on_base_fraction_mean"]),
            0.9,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_effective_prefix_fraction_mean"]),
            0.6,
        )
        self.assertGreaterEqual(
            float(batch["critic_contract_bootstrap_anchor_effective_prefix_fraction_mean"]),
            float(batch["critic_contract_bootstrap_precontact_anchor_prefix_fraction_mean"])
            - 0.2,
        )

    def test_bootstrap_anchor_budget_saturates_valid_positions_before_dense_spillover(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
        )

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_coverage_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_budget_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_dense_budget_mean"]),
            0.0,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_budget_utilization_mean"]),
            0.95,
        )
        self.assertGreater(
            float(batch["critic_contract_bootstrap_anchor_contact_density_mean"]),
            0.95,
        )

    def test_bootstrap_surface_memory_raises_sustained_floor_before_late_gate_opens(self):
        device = torch.device("cpu")
        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        fresh_loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            global_step=1100,
            best_eval=500.0,
            last_eval=150.0,
            bootstrap_min_step=1200,
            bootstrap_step_ramp=200,
            bootstrap_eval_threshold=200.0,
            bootstrap_eval_ramp=50.0,
        )
        fresh_loop.config.adaptive_imag_critic_bootstrap_precontact_floor_max = 0.25

        primed_loop = _prime_critic_bootstrap_contract_loop(
            _make_actor_contract_trust_test_loop(
                device,
                drift_actor=False,
            ),
            enabled=True,
            global_step=1100,
            best_eval=500.0,
            last_eval=150.0,
            bootstrap_min_step=1200,
            bootstrap_step_ramp=200,
            bootstrap_eval_threshold=200.0,
            bootstrap_eval_ramp=50.0,
        )
        primed_loop.config.adaptive_imag_critic_bootstrap_precontact_floor_max = 0.25
        primed_loop._adaptive_imag_critic_bootstrap_contact_surface_state = 0.75

        fresh_batch = fresh_loop._build_imagined_batch(reference_real_batch=reference_real_batch)
        primed_batch = primed_loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(fresh_batch)
        self.assertIsNotNone(primed_batch)
        self.assertAlmostEqual(
            float(primed_batch["critic_contract_bootstrap_late_gate"]),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(primed_batch["critic_contract_bootstrap_precontact_gate"]),
            0.0,
        )
        self.assertGreater(
            float(primed_batch["critic_contract_bootstrap_surface_memory_mean"]),
            0.0,
        )
        self.assertGreater(
            float(primed_batch["critic_contract_bootstrap_persistent_bonus_mean"]),
            float(fresh_batch["critic_contract_bootstrap_persistent_bonus_mean"]),
        )
        self.assertGreater(
            float(primed_batch["critic_contract_bootstrap_sustained_floor_mean"]),
            float(primed_batch["critic_contract_bootstrap_precontact_floor_mean"]),
        )
        self.assertGreater(
            float(primed_batch["critic_contract_bootstrap_sustained_floor_mean"]),
            float(fresh_batch["critic_contract_bootstrap_sustained_floor_mean"]),
        )
        self.assertGreater(
            float(primed_batch["critic_contract_bootstrap_clean_mix_mean"]),
            float(fresh_batch["critic_contract_bootstrap_clean_mix_mean"]),
        )

    def test_post_trigger_confirmation_steps_delay_guard_activation(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_trigger_min_pressure=0.01,
            adaptive_imag_compensation_trigger_actor_scale_gain=3.0,
            adaptive_imag_compensation_trigger_actor_scale_floor=0.25,
            adaptive_imag_compensation_trigger_confirmation_steps=3,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _AdaptiveCapImaginationEngine(device)

        first = loop._build_imagined_batch()
        self.assertIsNotNone(first)
        self.assertAlmostEqual(float(loop.metrics["imag/compensation_trigger_confirmation_ready"]), 0.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/compensation_trigger_active"]), 0.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/post_trigger_scale"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_cap_pressure_confirmation_scale"]), 0.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/compensation_trigger_pressure_streak"]), 1.0, places=6)
        self.assertGreater(float(loop.metrics["imag/continue_cap_pressure_raw"]), float(loop.metrics["imag/continue_cap_pressure"]))
        self.assertGreaterEqual(float(loop.metrics["imag/continue_cap_pressure_unscaled"]), float(loop.metrics["imag/continue_cap_pressure_raw"]))
        first_cap = float(loop.metrics["imag/continue_prob_cap_dynamic"])

        second = loop._build_imagined_batch()
        self.assertIsNotNone(second)
        self.assertAlmostEqual(float(loop.metrics["imag/compensation_trigger_confirmation_ready"]), 0.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/compensation_trigger_active"]), 0.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/post_trigger_scale"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_cap_pressure_confirmation_scale"]), 0.5, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/compensation_trigger_pressure_streak"]), 2.0, places=6)
        self.assertLess(float(loop.metrics["imag/continue_prob_cap_dynamic"]), first_cap)

        third = loop._build_imagined_batch()
        self.assertIsNotNone(third)
        self.assertAlmostEqual(float(loop.metrics["imag/compensation_trigger_confirmation_ready"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/compensation_trigger_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/continue_cap_pressure_confirmation_scale"]), 1.0, places=6)
        self.assertLess(float(loop.metrics["actor/post_trigger_scale"]), 1.0)


    def test_post_trigger_quality_gate_requires_good_streak_before_release(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_trigger_min_pressure=0.01,
            adaptive_imag_compensation_trigger_actor_scale_gain=3.0,
            adaptive_imag_compensation_trigger_actor_scale_floor=0.25,
            adaptive_imag_compensation_trigger_hysteresis=True,
            adaptive_imag_compensation_trigger_release_ratio=0.5,
            adaptive_imag_compensation_trigger_attack_ema=0.0,
            adaptive_imag_compensation_trigger_release_ema=0.0,
            adaptive_imag_compensation_trigger_quality_gate=True,
            adaptive_imag_compensation_trigger_quality_gap_threshold=2.0,
            adaptive_imag_compensation_trigger_quality_actor_threshold=4.5,
            adaptive_imag_compensation_trigger_quality_streak=2,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _AdaptiveCapImaginationEngine(device)
        engaged = loop._build_imagined_batch()
        self.assertIsNotNone(engaged)
        engaged_scale = float(loop.metrics["actor/post_trigger_scale"])
        self.assertLess(engaged_scale, 1.0)

        loop.imagination_engine = _BadQualityLowPressureImaginationEngine(device)
        blocked = loop._build_imagined_batch()
        self.assertIsNotNone(blocked)
        self.assertAlmostEqual(float(loop.metrics["actor/post_trigger_scale"]), engaged_scale, places=6)
        self.assertEqual(float(loop.metrics["actor/post_trigger_quality_good"]), 0.0)
        self.assertEqual(float(loop.metrics["actor/post_trigger_quality_good_streak"]), 0.0)

        loop.imagination_engine = _FakeImaginationEngine(device)
        first_good = loop._build_imagined_batch()
        self.assertIsNotNone(first_good)
        self.assertAlmostEqual(float(loop.metrics["actor/post_trigger_scale"]), engaged_scale, places=6)
        self.assertEqual(float(loop.metrics["actor/post_trigger_quality_good_streak"]), 1.0)

        second_good = loop._build_imagined_batch()
        self.assertIsNotNone(second_good)
        self.assertGreater(float(loop.metrics["actor/post_trigger_scale"]), engaged_scale)
        self.assertEqual(float(loop.metrics["actor/post_trigger_quality_good_streak"]), 2.0)

    def test_build_imagined_batch_emits_compensation_entry_audit_when_adaptive_continue_cap_disabled(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            adaptive_imag_continue_cap=False,
            imag_continue_prob_cap=0.95,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _AdaptiveCapImaginationEngine(device)

        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(
            loop.metrics["imag/compensation_entry_root_veto_reason"],
            "adaptive_continue_cap_disabled",
        )
        self.assertAlmostEqual(
            float(loop.metrics["imag/compensation_entry_adaptive_continue_cap_enabled"]),
            0.0,
            places=6,
        )
        self.assertEqual(
            loop.metrics["imag/compensation_entry_trigger_veto_reason"],
            "adaptive_continue_cap_disabled",
        )
        self.assertEqual(
            loop.metrics["imag/compensation_entry_handoff_veto_reason"],
            "handoff_disabled",
        )

    def test_build_imagined_batch_emits_post_entry_veto_when_trigger_can_activate_but_source_stays_idle(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_trigger_min_pressure=0.01,
            adaptive_imag_compensation_trigger_confirmation_steps=1,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _AdaptiveCapImaginationEngine(device)

        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertAlmostEqual(
            float(loop.metrics["imag/compensation_entry_trigger_gate_pass"]),
            1.0,
            places=6,
        )
        self.assertEqual(
            loop.metrics["imag/compensation_entry_post_entry_veto_reason"],
            "entry_eval_disabled",
        )
        self.assertEqual(
            loop.metrics["imag/compensation_entry_release_veto_reason"],
            "post_entry_source_idle",
        )
        self.assertEqual(
            loop.metrics["imag/compensation_entry_root_veto_reason"],
            "phase_active",
        )


    def test_quality_scale_further_reduces_actor_scale_under_bad_quality(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)

        common = dict(
            config_mode="strict", validation_mode="off", total_steps=1, num_train_steps=1, total_env_steps=2,
            batch_size=1, seq_len=4, wm_seq_len=4, wm_batch_size=1, rl_batch_size=1, wm_pretrain_steps=0,
            warmup_steps=0, imagination_only=True, imagination_horizon=2, imagination_batch_size=1,
            imag_continue_prob_cap=0.95, adaptive_imag_continue_cap=True, adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95, adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97, adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02, adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002, adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_trigger_min_pressure=0.01, adaptive_imag_compensation_trigger_actor_scale_gain=3.0,
            adaptive_imag_compensation_trigger_actor_scale_floor=0.25, adaptive_imag_compensation_trigger_quality_gap_threshold=2.0,
            adaptive_imag_compensation_trigger_quality_actor_threshold=4.5, log_interval=1000, eval_interval=1000, save_interval=1000,
        )
        config_plain = TrainingConfig(**common)
        loop_plain = TrainingLoop(model=model, buffer=buffer, config=config_plain, device=device, env=None)
        loop_plain.imagination_engine = _BadQualityLowPressureImaginationEngine(device)
        plain = loop_plain._build_imagined_batch()
        self.assertIsNotNone(plain)
        plain_scale = float(loop_plain.metrics["actor/post_trigger_scale"])

        config_scaled = TrainingConfig(**common, adaptive_imag_compensation_trigger_quality_scale_gain=1.0, adaptive_imag_compensation_trigger_quality_scale_floor=0.5)
        loop_scaled = TrainingLoop(model=model, buffer=buffer, config=config_scaled, device=device, env=None)
        loop_scaled.imagination_engine = _BadQualityLowPressureImaginationEngine(device)
        scaled = loop_scaled._build_imagined_batch()
        self.assertIsNotNone(scaled)
        self.assertLess(float(loop_scaled.metrics["actor/post_trigger_scale"]), plain_scale)
        self.assertGreater(float(loop_scaled.metrics["actor/post_trigger_quality_scale_pressure"]), 0.0)


    def test_piecewise_quality_scale_stays_inactive_below_mild_threshold(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict", validation_mode="off", total_steps=1, num_train_steps=1, total_env_steps=2,
            batch_size=1, seq_len=4, wm_seq_len=4, wm_batch_size=1, rl_batch_size=1, wm_pretrain_steps=0,
            warmup_steps=0, imagination_only=True, imagination_horizon=2, imagination_batch_size=1,
            imag_continue_prob_cap=0.95, adaptive_imag_continue_cap=True, adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95, adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97, adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02, adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002, adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_trigger_quality_piecewise=True,
            adaptive_imag_compensation_trigger_quality_mild_threshold=2.0,
            adaptive_imag_compensation_trigger_quality_severe_threshold=8.0,
            adaptive_imag_compensation_trigger_quality_mild_floor=0.85,
            adaptive_imag_compensation_trigger_quality_severe_floor=0.55,
            log_interval=1000, eval_interval=1000, save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FakeImaginationEngine(device)
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(loop.metrics["actor/post_trigger_quality_scale_multiplier"]), 1.0, places=6)

    def test_piecewise_quality_scale_bounds_bad_quality_without_full_collapse(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(vitals=np.full((4,), 2.0, dtype=np.float32), action=np.array([1.0, 0.0], dtype=np.float32), reward=1.0, done=False, log_prob=0.0)
        buffer.add_step(vitals=np.full((4,), 3.0, dtype=np.float32), action=np.array([0.0, 1.0], dtype=np.float32), reward=0.0, done=True, terminated=True, truncated=False, log_prob=0.0)
        config = TrainingConfig(
            config_mode="strict", validation_mode="off", total_steps=1, num_train_steps=1, total_env_steps=2,
            batch_size=1, seq_len=4, wm_seq_len=4, wm_batch_size=1, rl_batch_size=1, wm_pretrain_steps=0,
            warmup_steps=0, imagination_only=True, imagination_horizon=2, imagination_batch_size=1,
            imag_continue_prob_cap=0.95, adaptive_imag_continue_cap=True, adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95, adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97, adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02, adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002, adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_trigger_quality_piecewise=True,
            adaptive_imag_compensation_trigger_quality_mild_threshold=1.0,
            adaptive_imag_compensation_trigger_quality_severe_threshold=4.0,
            adaptive_imag_compensation_trigger_quality_mild_floor=0.85,
            adaptive_imag_compensation_trigger_quality_severe_floor=0.55,
            log_interval=1000, eval_interval=1000, save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _BadQualityLowPressureImaginationEngine(device)
        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        mult = float(loop.metrics["actor/post_trigger_quality_scale_multiplier"])
        self.assertLess(mult, 1.0)
        self.assertGreaterEqual(mult, 0.55)


    def test_imagination_only_resume_restores_buffer_and_loop_counters(self):
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=6,
            num_train_steps=6,
            total_env_steps=12,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )

        seed_buffer = ReplayBuffer(capacity=8, store_obs=False)
        seed_buffer.start_episode(np.zeros((4,), dtype=np.float32))
        seed_buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        seed_buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        source_loop = TrainingLoop(
            model=_TinyImagModel(),
            buffer=seed_buffer,
            config=config,
            device=device,
            env=None,
        )
        source_loop.imagination_engine = _FakeImaginationEngine(device)

        from aletheia.aletheia_train import TrainingStateManager
        state = TrainingStateManager.create(config=config, device=device)
        state.global_step = 5
        state.episode_count = 1
        state.total_samples = 160
        state.best_eval_return = 222.0
        state.best_step = 4
        state.episode_return_ema = 12.5
        state.episode_return_initialized = True
        state.last_episode_count = seed_buffer._episode_counter
        state.last_return_episode_count = 0
        state.last_seen_episode_idx = seed_buffer.episodes[-1]["idx"]
        source_loop._external_eval_best_mean = 234.5
        source_loop._external_eval_last_mean = 210.0
        source_loop._bootstrap_external_eval_feedback_last_step = 5
        source_loop._bootstrap_external_eval_feedback_last_mean = 210.0
        source_loop._bootstrap_external_eval_feedback_best_mean = 234.5
        source_loop._bootstrap_external_eval_feedback_telemetry = {
            "real_reward_degradation": 0.25,
            "real_task_cert_gate": 0.75,
        }
        source_loop._adaptive_imag_compensation_trigger_pressure_streak = 7
        source_loop._adaptive_imag_post_entry_eval_confirmation_streak = 1
        source_loop._adaptive_imag_compensation_persistence_eval_confirmation_streak = 2
        source_loop._adaptive_imag_compensation_post_solved_eval_confirmation_streak = 3
        source_loop._adaptive_imag_compensation_persistence_confirmed_best_mean = 180.0
        source_loop._adaptive_imag_compensation_post_solved_confirmed_best_mean = 240.0
        source_loop._adaptive_imag_compensation_persistence_hold_until_step = 12
        source_loop._adaptive_imag_compensation_trigger_persistence_handoff_blocked_until_step = 18
        source_loop._adaptive_imag_compensation_trigger_persistence_handoff_confirmation_pending_after_late_trigger_base_cap = True
        source_loop._adaptive_imag_compensation_trigger_persistence_handoff_confirmation_streak_after_late_trigger_base_cap = 1
        source_loop._adaptive_imag_compensation_persistence_ever_armed = True
        source_loop._adaptive_imag_compensation_persistence_ever_active = True
        source_loop._adaptive_imag_compensation_post_solved_hold_until_step = 20
        source_loop._capture_post_solved_actor_anchor(step=5, eval_mean=230.0)
        source_loop._capture_behavior_policy_eval_anchor(step=5, eval_mean=230.0)
        source_loop._store_real_stability_telemetry(
            step=4,
            telemetry={
                "real_behavior_action_entropy_mean": 0.2,
                "real_policy_certified_registry_available": 1.0,
                "real_policy_certified_registry_size": 1.0,
                "real_policy_kl_to_certified_registry_mean": 0.01,
            },
        )
        source_loop._capture_real_stability_certified_anchor(step=4, eval_mean=210.0)
        with torch.no_grad():
            source_loop.model.actor.net.bias.copy_(
                torch.tensor([0.5, -0.5], device=device)
            )
        source_loop._store_real_stability_telemetry(
            step=5,
            telemetry={
                "real_behavior_action_entropy_mean": 0.3,
                "real_policy_certified_registry_available": 1.0,
                "real_policy_certified_registry_size": 2.0,
                "real_policy_kl_to_certified_registry_mean": 0.02,
            },
        )
        source_loop._capture_real_stability_certified_anchor(step=5, eval_mean=230.0)
        state.adaptive_compensation_state = source_loop._export_adaptive_compensation_state()
        self.assertEqual(
            state.adaptive_compensation_state["minimal_compensation_state"]["schema_version"],
            2,
        )
        for removed_key in (
            "adaptive_imag_continue_cap_state",
            "adaptive_imag_compensation_trigger_actor_scale_state",
            "adaptive_imag_compensation_phase",
            "adaptive_imag_critic_bootstrap_contact_surface_state",
            "adaptive_imag_critic_bootstrap_bonus_hold_state",
            "adaptive_imag_critic_bootstrap_post_transition_certified_floor_state",
            "adaptive_imag_compensation_persistence_hold_until_step",
            "adaptive_imag_compensation_persistence_armed_step",
            "adaptive_imag_compensation_persistence_entry_protect_until_step",
            "adaptive_imag_compensation_persistence_release_hold_until_step",
            "adaptive_imag_compensation_post_solved_hold_until_step",
        ):
            self.assertNotIn(removed_key, state.adaptive_compensation_state)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(Path(tmpdir) / "resume.pt")
            from aletheia.aletheia_train import TrainingStateManager
            TrainingStateManager.save(
                state,
                ckpt_path,
                model=source_loop.model,
                opt_bundle=source_loop.opt_bundle,
                buffer=source_loop.buffer,
            )

            resumed_loop = TrainingLoop(
                model=_TinyImagModel(),
                buffer=ReplayBuffer(capacity=8, store_obs=False),
                config=config,
                device=device,
                env=None,
            )
            resumed_loop.imagination_engine = _FakeImaginationEngine(device)
            resumed_loop._build_real_batch = lambda: None

            warnings = []

            def _capture_warning(msg, *args, **kwargs):
                del kwargs
                warnings.append(msg % args if args else msg)

            with mock.patch("aletheia.aletheia_train.logger.warning", side_effect=_capture_warning):
                resumed_loop.run(num_steps=6, data_collector=None, resume_from=ckpt_path)

        self.assertEqual(resumed_loop.global_step, 6)
        self.assertEqual(resumed_loop.env_steps_collected, 160)
        self.assertEqual(len(resumed_loop.buffer.episodes), 1)
        self.assertAlmostEqual(resumed_loop._episode_return_ema, 12.5, places=6)
        self.assertTrue(resumed_loop._episode_return_initialized)
        self.assertAlmostEqual(resumed_loop._external_eval_best_mean, 234.5, places=6)
        self.assertAlmostEqual(resumed_loop._external_eval_last_mean, 210.0, places=6)
        self.assertEqual(resumed_loop._bootstrap_external_eval_feedback_last_step, 5)
        self.assertAlmostEqual(
            resumed_loop._bootstrap_external_eval_feedback_last_mean,
            210.0,
            places=6,
        )
        self.assertAlmostEqual(
            resumed_loop._bootstrap_external_eval_feedback_best_mean,
            234.5,
            places=6,
        )
        self.assertAlmostEqual(
            resumed_loop._bootstrap_external_eval_feedback_telemetry["real_reward_degradation"],
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            resumed_loop._bootstrap_external_eval_feedback_telemetry["real_task_cert_gate"],
            0.75,
            places=6,
        )
        self.assertEqual(resumed_loop._adaptive_imag_compensation_trigger_pressure_streak, 7)
        self.assertEqual(resumed_loop._adaptive_imag_post_entry_eval_confirmation_streak, 1)
        self.assertEqual(resumed_loop._adaptive_imag_compensation_persistence_eval_confirmation_streak, 2)
        self.assertEqual(resumed_loop._adaptive_imag_compensation_post_solved_eval_confirmation_streak, 3)
        self.assertAlmostEqual(resumed_loop._adaptive_imag_compensation_persistence_confirmed_best_mean, 180.0, places=6)
        self.assertAlmostEqual(resumed_loop._adaptive_imag_compensation_post_solved_confirmed_best_mean, 240.0, places=6)
        self.assertEqual(resumed_loop._adaptive_imag_compensation_persistence_hold_until_step, 12)
        self.assertEqual(resumed_loop._adaptive_imag_compensation_trigger_persistence_handoff_blocked_until_step, 18)
        self.assertTrue(
            resumed_loop._adaptive_imag_compensation_trigger_persistence_handoff_confirmation_pending_after_late_trigger_base_cap
        )
        self.assertEqual(
            resumed_loop._adaptive_imag_compensation_trigger_persistence_handoff_confirmation_streak_after_late_trigger_base_cap,
            1,
        )
        self.assertTrue(resumed_loop._adaptive_imag_compensation_persistence_ever_armed)
        self.assertTrue(resumed_loop._adaptive_imag_compensation_persistence_ever_active)
        self.assertEqual(resumed_loop._adaptive_imag_compensation_post_solved_hold_until_step, 20)
        self.assertEqual(
            resumed_loop._adaptive_imag_compensation_post_solved_actor_anchor_step,
            5,
        )
        self.assertAlmostEqual(
            resumed_loop._adaptive_imag_compensation_post_solved_actor_anchor_eval,
            230.0,
            places=6,
        )
        self.assertIsNotNone(
            resumed_loop._adaptive_imag_compensation_post_solved_actor_anchor_actor
        )
        self.assertTrue(
            bool(
                resumed_loop._adaptive_imag_compensation_post_solved_actor_anchor_params
            )
        )
        self.assertEqual(
            resumed_loop._adaptive_imag_compensation_post_solved_critic_anchor_step,
            5,
        )
        self.assertAlmostEqual(
            resumed_loop._adaptive_imag_compensation_post_solved_critic_anchor_eval,
            230.0,
            places=6,
        )
        self.assertIsNotNone(
            resumed_loop._adaptive_imag_compensation_post_solved_critic_anchor_critic
        )
        self.assertEqual(resumed_loop._behavior_policy_eval_anchor_step, 5)
        self.assertAlmostEqual(
            resumed_loop._behavior_policy_eval_anchor_eval,
            230.0,
            places=6,
        )
        self.assertIsNotNone(resumed_loop._behavior_policy_eval_anchor_actor)
        self.assertEqual(
            resumed_loop.training_step._adaptive_imag_compensation_post_solved_actor_anchor_step,
            5,
        )
        self.assertAlmostEqual(
            resumed_loop.training_step._adaptive_imag_compensation_post_solved_actor_anchor_eval,
            230.0,
            places=6,
        )
        self.assertEqual(
            len(resumed_loop._real_stability_certified_anchor_registry_actors),
            2,
        )
        self.assertIsNotNone(resumed_loop._real_stability_certified_anchor_actor)
        self.assertEqual(resumed_loop._real_stability_certified_anchor_step, 5)
        self.assertAlmostEqual(
            resumed_loop._real_stability_certified_anchor_eval,
            230.0,
            places=6,
        )
        self.assertFalse(any("no seed data available" in w for w in warnings))

    def test_imagination_only_resume_rejects_legacy_post_solved_anchor_backfill_by_default(self):
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=6,
            num_train_steps=6,
            total_env_steps=12,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            adaptive_imag_compensation_post_solved_eval_threshold=200.0,
            adaptive_imag_compensation_post_solved_eval_confirmation_count=1,
            adaptive_imag_compensation_post_solved_hold_steps=8,
            adaptive_imag_compensation_post_solved_min_step=4,
            adaptive_imag_compensation_post_solved_actor_anchor_eval_threshold=200.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )

        seed_buffer = ReplayBuffer(capacity=8, store_obs=False)
        seed_buffer.start_episode(np.zeros((4,), dtype=np.float32))
        seed_buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        seed_buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        source_loop = TrainingLoop(
            model=_TinyImagModel(),
            buffer=seed_buffer,
            config=config,
            device=device,
            env=None,
        )
        source_loop.imagination_engine = _FakeImaginationEngine(device)
        with torch.no_grad():
            source_loop.model.actor.net.bias.copy_(
                torch.tensor([0.25, -0.25], device=device)
            )

        from aletheia.aletheia_train import TrainingStateManager
        state = TrainingStateManager.create(config=config, device=device)
        state.global_step = 5
        state.episode_count = 1
        state.total_samples = 160
        state.best_eval_return = 234.5
        state.best_step = 5
        state.episode_return_ema = 12.5
        state.episode_return_initialized = True
        state.last_episode_count = seed_buffer._episode_counter
        state.last_return_episode_count = 0
        state.last_seen_episode_idx = seed_buffer.episodes[-1]["idx"]
        source_loop._external_eval_best_mean = 234.5
        source_loop._external_eval_last_mean = 234.5
        source_loop._external_eval_last_step = 5
        state.adaptive_compensation_state = source_loop._export_adaptive_compensation_state()
        state.adaptive_compensation_state.pop("post_solved_anchor_state", None)
        state.adaptive_compensation_state.pop("behavior_policy_eval_anchor_state", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(Path(tmpdir) / "resume.pt")
            TrainingStateManager.save(
                state,
                ckpt_path,
                model=source_loop.model,
                opt_bundle=source_loop.opt_bundle,
                buffer=source_loop.buffer,
            )

            resumed_loop = TrainingLoop(
                model=_TinyImagModel(),
                buffer=ReplayBuffer(capacity=8, store_obs=False),
                config=config,
                device=device,
                env=None,
            )
            resumed_loop.imagination_engine = _FakeImaginationEngine(device)
            resumed_loop._build_real_batch = lambda: None
            with self.assertRaisesRegex(
                RuntimeError,
                "Adaptive compensation restore degraded",
            ):
                resumed_loop.run(num_steps=6, data_collector=None, resume_from=ckpt_path)

    def test_resume_reports_compensation_anchor_degradation_explicitly(self):
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=6,
            num_train_steps=6,
            total_env_steps=12,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            adaptive_imag_compensation_post_solved_eval_threshold=200.0,
            adaptive_imag_compensation_post_solved_eval_confirmation_count=1,
            adaptive_imag_compensation_post_solved_hold_steps=8,
            adaptive_imag_compensation_post_solved_min_step=4,
            adaptive_imag_compensation_post_solved_actor_anchor_eval_threshold=200.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )

        seed_buffer = ReplayBuffer(capacity=8, store_obs=False)
        seed_buffer.start_episode(np.zeros((4,), dtype=np.float32))
        seed_buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        seed_buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        source_loop = TrainingLoop(
            model=_TinyImagModel(),
            buffer=seed_buffer,
            config=config,
            device=device,
            env=None,
        )
        source_loop.imagination_engine = _FakeImaginationEngine(device)
        source_loop._capture_post_solved_actor_anchor(step=5, eval_mean=230.0)
        source_loop._capture_behavior_policy_eval_anchor(step=5, eval_mean=230.0)
        source_loop._store_real_stability_telemetry(
            step=4,
            telemetry={"real_policy_certified_registry_available": 1.0},
        )
        source_loop._capture_real_stability_certified_anchor(step=4, eval_mean=210.0)
        source_loop._store_real_stability_telemetry(
            step=5,
            telemetry={"real_policy_certified_registry_available": 1.0},
        )
        source_loop._capture_real_stability_certified_anchor(step=5, eval_mean=230.0)

        exported = source_loop._export_adaptive_compensation_state()
        exported["post_solved_anchor_state"]["actor_anchor_params"] = {}
        exported["post_solved_anchor_state"]["actor_anchor_actor_state_dict"] = {
            "net.weight": torch.zeros((3, 3)),
        }
        exported["post_solved_anchor_state"]["critic_anchor_critic_state_dict"] = {
            "weight": torch.zeros((3, 3)),
        }
        exported["behavior_policy_eval_anchor_state"]["actor_state_dict"] = {
            "net.weight": torch.zeros((3, 3)),
        }
        registry_entries = list(exported["real_stability_certified_registry_entries"])
        registry_entries[0]["actor_state_dict"] = {"net.weight": torch.zeros((3, 3))}
        exported["real_stability_certified_registry_entries"] = registry_entries

        restored_loop = TrainingLoop(
            model=_TinyImagModel(),
            buffer=ReplayBuffer(capacity=8, store_obs=False),
            config=config,
            device=device,
            env=None,
        )
        restored_loop.imagination_engine = _FakeImaginationEngine(device)
        restored_loop._external_eval_best_mean = 234.5
        restored_loop._external_eval_last_mean = 234.5
        restored_loop._external_eval_last_step = 5

        warnings = []

        def _capture_warning(msg, *args, **kwargs):
            del kwargs
            warnings.append(msg % args if args else msg)

        with mock.patch(
            "aletheia.training.compensation_kernel.logger.warning",
            side_effect=_capture_warning,
        ):
            restored_loop._restore_adaptive_compensation_state(
                exported,
                fallback_best_step=5,
                fallback_best_eval=234.5,
            )

        report = restored_loop._adaptive_compensation_restore_report
        self.assertEqual(report["status"], "degraded")
        self.assertEqual(report["post_solved_anchor"]["status"], "fallback_recovered")
        self.assertEqual(report["behavior_policy_anchor"]["status"], "fallback_recovered")
        self.assertEqual(
            int(report["real_stability_registry"]["skipped_entries"]),
            1,
        )
        self.assertEqual(
            int(report["real_stability_registry"]["restored_entries"]),
            1,
        )
        self.assertTrue(any("post_solved_anchor" in item for item in warnings))
        self.assertTrue(any("behavior_policy_anchor" in item for item in warnings))
        self.assertTrue(any("real_stability_registry" in item for item in warnings))

    def test_resume_from_with_optimizer_drift_auto_skips_incompatible_optimizer_state(self):
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=6,
            num_train_steps=6,
            total_env_steps=12,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )

        seed_buffer = ReplayBuffer(capacity=8, store_obs=False)
        seed_buffer.start_episode(np.zeros((4,), dtype=np.float32))
        seed_buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        seed_buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        source_loop = TrainingLoop(
            model=_TinyImagModel(),
            buffer=seed_buffer,
            config=config,
            device=device,
            env=None,
        )
        source_loop.imagination_engine = _FakeImaginationEngine(device)
        source_loop._build_real_batch = lambda: None

        from aletheia.aletheia_train import TrainingStateManager

        state = TrainingStateManager.create(config=config, device=device)
        state.global_step = 5
        state.episode_count = 1
        state.total_samples = 160
        state.last_episode_count = seed_buffer._episode_counter
        state.last_return_episode_count = 0
        state.last_seen_episode_idx = seed_buffer.episodes[-1]["idx"]

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(Path(tmpdir) / "resume.pt")
            TrainingStateManager.save(
                state,
                ckpt_path,
                model=source_loop.model,
                opt_bundle=source_loop.opt_bundle,
                buffer=source_loop.buffer,
            )

            resumed_loop = TrainingLoop(
                model=_TinyImagModelWithOptimizerDrift(),
                buffer=ReplayBuffer(capacity=8, store_obs=False),
                config=config,
                device=device,
                env=None,
            )
            resumed_loop.imagination_engine = _FakeImaginationEngine(device)
            resumed_loop._build_real_batch = lambda: None

            warnings = []

            def _capture_warning(msg, *args, **kwargs):
                del kwargs
                warnings.append(msg % args if args else msg)

            with mock.patch(
                "aletheia._training_checkpoint_schema.logger.warning",
                side_effect=_capture_warning,
            ):
                resumed_loop.run(num_steps=6, data_collector=None, resume_from=ckpt_path)

        self.assertEqual(resumed_loop.global_step, 6)
        self.assertEqual(len(resumed_loop.buffer.episodes), 1)
        self.assertTrue(
            any("non-strict model load" in message for message in warnings)
        )
        self.assertTrue(
            any(
                "optimizer_bundle" in message and "compatible mode" in message
                for message in warnings
            )
        )

    def test_build_real_batch_injects_runtime_compensation_context(self):
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            wm_pretrain_steps=0,
            warmup_steps=0,
        )

        buffer = ReplayBuffer(capacity=8, store_obs=False)
        buffer.start_episode(np.zeros((4,), dtype=np.float32))
        buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        loop = TrainingLoop(
            model=_TinyImagModel(),
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.global_step = 10
        loop._adaptive_imag_compensation_phase = "idle"
        loop._adaptive_imag_compensation_post_solved_hold_until_step = 20
        loop._adaptive_imag_compensation_persistence_hold_until_step = 15
        loop._adaptive_imag_external_post_entry_pending_commit = True
        loop._adaptive_imag_post_entry_pending_recovery_latched = True
        loop._external_eval_last_mean = 210.0
        loop._external_eval_best_mean = 234.5
        loop._compute_gae_for_batch = lambda batch: {
            **batch,
            "advantages": torch.ones((1, 2), device=device),
            "returns": torch.ones((1, 2), device=device),
            "values": torch.zeros((1, 2), device=device),
        }

        batch = loop._build_real_batch()

        self.assertIsNotNone(batch)
        self.assertEqual(batch["compensation_phase"], "post_solved")
        self.assertFalse(batch["compensation_post_entry_commit_ready"])
        self.assertTrue(batch["compensation_post_entry_pending_active"])
        self.assertTrue(batch["compensation_post_entry_pending_recovery_latched"])
        self.assertAlmostEqual(batch["external_eval_last_mean"], 210.0, places=6)
        self.assertAlmostEqual(batch["external_eval_best_mean"], 234.5, places=6)

    def test_train_step_logs_real_runtime_compensation_phase(self):
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            wm_pretrain_steps=0,
            warmup_steps=0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )

        buffer = ReplayBuffer(capacity=8, store_obs=False)
        buffer.start_episode(np.zeros((4,), dtype=np.float32))
        buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        loop = TrainingLoop(
            model=_TinyImagModel(),
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.global_step = 10
        loop._adaptive_imag_compensation_phase = "idle"
        loop._adaptive_imag_compensation_post_solved_hold_until_step = 20
        loop._adaptive_imag_compensation_persistence_hold_until_step = 15
        loop._compute_gae_for_batch = lambda batch: {
            **batch,
            "advantages": torch.ones((1, 2), device=device),
            "returns": torch.ones((1, 2), device=device),
            "values": torch.zeros((1, 2), device=device),
        }

        real_batch = loop._build_real_batch()

        self.assertIsNotNone(real_batch)
        metrics = loop.train_step(
            rl_batch=real_batch,
            wm_batch=real_batch,
            source_tag="real",
            imag_ratio=0.0,
        )

        self.assertEqual(metrics["real/runtime_compensation_phase"], "post_solved")
        self.assertEqual(metrics["train/runtime_compensation_phase"], "post_solved")
        self.assertEqual(metrics["train/compensation_restore_status"], "not_restored")
        self.assertAlmostEqual(
            float(metrics["train/compensation_restore_degraded"]),
            0.0,
            places=6,
        )

    def test_train_step_warns_once_when_post_solved_guards_are_disabled(self):
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            wm_pretrain_steps=0,
            warmup_steps=0,
            adaptive_imag_compensation_post_solved_critic_anchor_weight=0.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )

        buffer = ReplayBuffer(capacity=8, store_obs=False)
        buffer.start_episode(np.zeros((4,), dtype=np.float32))
        buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        loop = TrainingLoop(
            model=_TinyImagModel(),
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.global_step = 10
        loop._adaptive_imag_compensation_post_solved_hold_until_step = 20
        loop._compute_gae_for_batch = lambda batch: {
            **batch,
            "advantages": torch.ones((1, 2), device=device),
            "returns": torch.ones((1, 2), device=device),
            "values": torch.zeros((1, 2), device=device),
        }

        warnings = []

        def _capture_warning(msg, *args, **kwargs):
            del kwargs
            warnings.append(msg % args if args else msg)

        with mock.patch("aletheia.aletheia_train.logger.warning", side_effect=_capture_warning):
            first_metrics = loop.train_step(
                rl_batch=loop._build_real_batch(),
                wm_batch=loop._build_real_batch(),
                source_tag="real",
                imag_ratio=0.0,
            )
            second_metrics = loop.train_step(
                rl_batch=loop._build_real_batch(),
                wm_batch=loop._build_real_batch(),
                source_tag="real",
                imag_ratio=0.0,
            )

        self.assertEqual(
            first_metrics["train/runtime_compensation_guard_disabled_reason"],
            "post_solved_guards_disabled",
        )
        self.assertAlmostEqual(
            float(first_metrics["train/runtime_compensation_guard_disabled"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(first_metrics["real/runtime_compensation_guard_disabled"]),
            1.0,
            places=6,
        )
        self.assertEqual(
            second_metrics["train/runtime_compensation_guard_disabled_reason"],
            "post_solved_guards_disabled",
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("Runtime compensation phase is post_solved", warnings[0])

    def test_train_step_does_not_warn_when_runtime_phase_is_idle(self):
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            wm_pretrain_steps=0,
            warmup_steps=0,
        )

        buffer = ReplayBuffer(capacity=8, store_obs=False)
        buffer.start_episode(np.zeros((4,), dtype=np.float32))
        buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        loop = TrainingLoop(
            model=_TinyImagModel(),
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.global_step = 10
        loop._compute_gae_for_batch = lambda batch: {
            **batch,
            "advantages": torch.ones((1, 2), device=device),
            "returns": torch.ones((1, 2), device=device),
            "values": torch.zeros((1, 2), device=device),
        }

        warnings = []

        def _capture_warning(msg, *args, **kwargs):
            del kwargs
            warnings.append(msg % args if args else msg)

        with mock.patch("aletheia.aletheia_train.logger.warning", side_effect=_capture_warning):
            metrics = loop.train_step(
                rl_batch=loop._build_real_batch(),
                wm_batch=loop._build_real_batch(),
                source_tag="real",
                imag_ratio=0.0,
            )

        self.assertEqual(metrics["train/runtime_compensation_phase"], "idle")
        self.assertAlmostEqual(
            float(metrics["train/runtime_compensation_guard_disabled"]),
            0.0,
            places=6,
        )
        self.assertEqual(
            metrics["train/runtime_compensation_guard_disabled_reason"],
            "none",
        )
        self.assertEqual(warnings, [])

    def test_train_step_does_not_warn_when_post_solved_guard_is_configured(self):
        device = torch.device("cpu")
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            wm_pretrain_steps=0,
            warmup_steps=0,
            adaptive_imag_compensation_post_solved_actor_anchor_pull=0.1,
        )

        buffer = ReplayBuffer(capacity=8, store_obs=False)
        buffer.start_episode(np.zeros((4,), dtype=np.float32))
        buffer.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        loop = TrainingLoop(
            model=_TinyImagModel(),
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.global_step = 10
        loop._adaptive_imag_compensation_post_solved_hold_until_step = 20
        loop._compute_gae_for_batch = lambda batch: {
            **batch,
            "advantages": torch.ones((1, 2), device=device),
            "returns": torch.ones((1, 2), device=device),
            "values": torch.zeros((1, 2), device=device),
        }

        warnings = []

        def _capture_warning(msg, *args, **kwargs):
            del kwargs
            warnings.append(msg % args if args else msg)

        with mock.patch("aletheia.aletheia_train.logger.warning", side_effect=_capture_warning):
            metrics = loop.train_step(
                rl_batch=loop._build_real_batch(),
                wm_batch=loop._build_real_batch(),
                source_tag="real",
                imag_ratio=0.0,
            )

        self.assertEqual(metrics["train/runtime_compensation_phase"], "post_solved")
        self.assertAlmostEqual(
            float(metrics["train/runtime_compensation_guard_disabled"]),
            0.0,
            places=6,
        )
        self.assertEqual(
            metrics["train/runtime_compensation_guard_disabled_reason"],
            "none",
        )
        self.assertEqual(warnings, [])

    def test_training_state_load_materializes_lazy_world_model_components(self):
        from aletheia import aletheia_train as train_mod

        config = TrainingConfig()
        device = torch.device("cpu")
        state = train_mod.TrainingStateManager.create(config=config, device=device)

        source_world_model = _LazyV45WorldModel()
        source_world_model._ensure_v45_components()
        source_model = build_training_model(
            actor=_TinyImagActor(),
            critic=_FlatContractCritic(4),
            world_model=source_world_model,
            router=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(Path(tmpdir) / "resume_lazy_world_model.pt")
            train_mod.TrainingStateManager.save(state, ckpt_path, model=source_model)

            target_world_model = _LazyV45WorldModel()
            target_model = build_training_model(
                actor=_TinyImagActor(),
                critic=_FlatContractCritic(4),
                world_model=target_world_model,
                router=None,
            )

            warnings = []

            def _capture_warning(msg, *args, **kwargs):
                del kwargs
                warnings.append(msg % args if args else msg)

            with mock.patch("aletheia.aletheia_train.logger.warning", side_effect=_capture_warning):
                train_mod.TrainingStateManager.load(
                    ckpt_path,
                    config,
                    device,
                    model=target_model,
                )

        self.assertFalse(any("non-strict model restore" in w for w in warnings))
        self.assertIsNotNone(target_world_model._control_head)
        self.assertIsNotNone(target_world_model._projection)
        self.assertTrue(torch.equal(
            target_world_model._control_head.weight.detach(),
            source_world_model._control_head.weight.detach(),
        ))
        self.assertTrue(torch.equal(
            target_world_model._control_head.bias.detach(),
            source_world_model._control_head.bias.detach(),
        ))
        self.assertTrue(torch.equal(
            target_world_model._projection.weight.detach(),
            source_world_model._projection.weight.detach(),
        ))
        self.assertTrue(torch.equal(
            target_world_model._projection.bias.detach(),
            source_world_model._projection.bias.detach(),
        ))

    def test_persistence_actor_base_return_cap_applies_when_configured(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            imag_continue_prob_cap=0.95,
            adaptive_imag_continue_cap=True,
            adaptive_imag_continue_cap_min=0.80,
            adaptive_imag_continue_cap_max=0.95,
            adaptive_imag_continue_cap_target_gap=4.0,
            adaptive_imag_continue_cap_target_continue=0.97,
            adaptive_imag_continue_cap_target_actor=20.0,
            adaptive_imag_continue_cap_gap_gain=0.02,
            adaptive_imag_continue_cap_continue_gain=0.5,
            adaptive_imag_continue_cap_actor_gain=0.002,
            adaptive_imag_continue_cap_ema=0.0,
            adaptive_imag_compensation_persistence_eval_threshold=150.0,
            adaptive_imag_compensation_persistence_cap=0.86,
            adaptive_imag_compensation_persistence_actor_scale=0.87,
            adaptive_imag_compensation_persistence_hold_steps=250,
            adaptive_imag_compensation_persistence_actor_base_return_cap_margin=0.5,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _FixedImaginationEngine(
            device,
            rewards=[[0.0, 0.0]],
            values=[[5.0, 5.0, 5.0]],
            continue_probs=[[0.0, 0.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )
        loop._external_eval_best_mean = 155.0
        loop._adaptive_imag_compensation_persistence_confirmed_best_mean = 155.0
        loop._adaptive_imag_compensation_persistence_ever_active = True
        loop.global_step = 800

        batch = loop._build_imagined_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(batch["compensation_phase"], "persistence")
        self.assertAlmostEqual(float(loop.metrics["actor/base_return_cap_margin"]), 0.5, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/base_return_cap_active"]), 1.0, places=6)
        self.assertEqual(loop.metrics["actor/base_source"], "return_cap")

    def test_build_imagined_batch_emits_behavior_observability_metrics(self):
        device = torch.device("cpu")

        class _BehaviorFeatureImaginationEngine(_FixedImaginationEngine):
            def __init__(
                self,
                device: torch.device,
                *,
                policy_features,
                rewards,
                values,
                continue_probs,
                entropies,
                log_probs,
                actions,
            ):
                super().__init__(
                    device,
                    rewards=rewards,
                    values=values,
                    continue_probs=continue_probs,
                    entropies=entropies,
                    log_probs=log_probs,
                    actions=actions,
                )
                self._policy_features = torch.tensor(
                    policy_features,
                    device=device,
                    dtype=torch.float32,
                )

            def imagine_rollout_differentiable(
                self,
                initial_state,
                initial_prev_action,
                horizon,
                actor,
                critic,
                initial_wm_state=None,
            ):
                out = super().imagine_rollout_differentiable(
                    initial_state,
                    initial_prev_action,
                    horizon,
                    actor,
                    critic,
                    initial_wm_state=initial_wm_state,
                )
                batch = initial_state.shape[0]
                out["policy_features"] = self._policy_features.repeat(batch, 1, 1)
                return out

        model = _TinyImagModel().to(device)
        with torch.no_grad():
            model.actor.net.weight.zero_()
            model.actor.net.bias.zero_()
            model.critic.weight.zero_()
            model.critic.bias.zero_()

        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=0.5,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.5,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=3,
            imagination_batch_size=1,
            imag_continue_prob_cap=1.0,
            adaptive_imag_critic_bootstrap_contract_enabled=True,
            adaptive_imag_continue_cap=False,
            adaptive_imag_idle_corridor_advantage_blend_max=0.1,
            adaptive_imag_idle_corridor_quantile=0.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]],
            rewards=[[0.1, 0.2, 0.3]],
            values=[[0.0, 0.0, 0.0, 0.0]],
            continue_probs=[[1.0, 1.0, 1.0]],
            entropies=[[0.2, 0.4, 0.6]],
            log_probs=[[-0.2, -0.3, -0.4]],
            actions=[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
        )

        anchor_actor = _TinyImagActor().to(device)
        with torch.no_grad():
            anchor_actor.net.weight.zero_()
            anchor_actor.net.bias.copy_(torch.tensor([2.0, -2.0], device=device))
        anchor_actor.eval()
        for param in anchor_actor.parameters():
            param.requires_grad_(False)
        loop._behavior_policy_eval_anchor_actor = anchor_actor
        loop._behavior_policy_eval_anchor_step = 700
        loop._behavior_policy_eval_anchor_eval = 250.0
        loop.global_step = 900

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(batch["behavior_policy_eval_anchor_available"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_policy_eval_anchor_age_steps"]), 200.0, places=6)
        self.assertGreater(float(batch["behavior_policy_kl_to_eval_anchor_mean"]), 0.0)
        self.assertGreater(float(batch["behavior_policy_kl_to_eval_anchor_corridor_mean"]), 0.0)
        self.assertGreater(float(batch["behavior_policy_kl_to_eval_anchor_preinflation_mean"]), 0.0)
        self.assertAlmostEqual(float(batch["behavior_action_entropy_mean"]), 0.4, places=6)
        self.assertAlmostEqual(float(batch["behavior_action_entropy_preinflation_mean"]), 0.4, places=6)
        self.assertAlmostEqual(float(batch["behavior_action_switch_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_action_switch_rate_preinflation"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_action_oscillation_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_action_oscillation_rate_preinflation"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_corridor_occupancy_fraction"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_corridor_persistence_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_corridor_exit_rate"]), 0.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_preinflation_fraction"]), 1.0, places=6)
        self.assertGreater(float(batch["behavior_f_policy_velocity_mean"]), 0.0)
        self.assertGreater(float(batch["behavior_f_policy_velocity_corridor_mean"]), 0.0)
        self.assertGreater(float(batch["behavior_f_policy_velocity_preinflation_mean"]), 0.0)
        self.assertGreater(float(batch["behavior_f_policy_acceleration_mean"]), 0.0)
        self.assertGreater(float(batch["behavior_f_policy_acceleration_preinflation_mean"]), 0.0)
        self.assertGreater(float(batch["behavior_f_policy_curvature_preinflation_mean"]), 0.0)
        self.assertAlmostEqual(float(loop.metrics["actor/behavior_action_switch_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/behavior_corridor_persistence_rate"]), 1.0, places=6)
        self.assertGreater(float(loop.metrics["actor/behavior_policy_kl_to_eval_anchor_mean"]), 0.0)

    def test_build_imagined_batch_emits_real_stability_observability_metrics(self):
        device = torch.device("cpu")

        class _BehaviorFeatureImaginationEngine(_FixedImaginationEngine):
            def __init__(
                self,
                device: torch.device,
                *,
                policy_features,
                rewards,
                values,
                continue_probs,
                entropies,
                log_probs,
                actions,
            ):
                super().__init__(
                    device,
                    rewards=rewards,
                    values=values,
                    continue_probs=continue_probs,
                    entropies=entropies,
                    log_probs=log_probs,
                    actions=actions,
                )
                self._policy_features = torch.tensor(
                    policy_features,
                    device=device,
                    dtype=torch.float32,
                )

            def imagine_rollout_differentiable(
                self,
                initial_state,
                initial_prev_action,
                horizon,
                actor,
                critic,
                initial_wm_state=None,
            ):
                out = super().imagine_rollout_differentiable(
                    initial_state,
                    initial_prev_action,
                    horizon,
                    actor,
                    critic,
                    initial_wm_state=initial_wm_state,
                )
                batch = initial_state.shape[0]
                out["policy_features"] = self._policy_features.repeat(batch, 1, 1)
                return out

        model = _TinyImagModel().to(device)
        with torch.no_grad():
            model.actor.net.weight.zero_()
            model.actor.net.bias.zero_()
            model.critic.weight.zero_()
            model.critic.bias.zero_()

        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=0.5,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.5,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=3,
            imagination_batch_size=1,
            imag_continue_prob_cap=1.0,
            adaptive_imag_continue_cap=False,
            adaptive_imag_idle_corridor_advantage_blend_max=0.1,
            adaptive_imag_idle_corridor_quantile=0.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]],
            rewards=[[0.1, 0.2, 0.3]],
            values=[[0.0, 0.0, 0.0, 0.0]],
            continue_probs=[[1.0, 1.0, 1.0]],
            entropies=[[0.2, 0.4, 0.6]],
            log_probs=[[-0.2, -0.3, -0.4]],
            actions=[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
        )

        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }
        loop.set_external_eval_feedback(220.0, step=100, telemetry=telemetry)
        with torch.no_grad():
            model.actor.net.bias.copy_(torch.tensor([1.5, -1.5], device=device))
        registry_actor = copy.deepcopy(model.actor).to(device)
        registry_actor.eval()
        for param in registry_actor.parameters():
            param.requires_grad_(False)
        loop._real_stability_certified_anchor_registry_actors = [registry_actor]
        loop._real_stability_certified_anchor_registry_steps = [100]
        loop._real_stability_certified_anchor_registry_evals = [220.0]
        loop._real_stability_certified_anchor_registry_telemetries = [dict(telemetry)]
        loop.global_step = 160

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(batch["real_behavior_available"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["real_behavior_age_steps"]), 60.0, places=6)
        self.assertAlmostEqual(float(batch["real_behavior_action_entropy_mean"]), 0.25, places=6)
        self.assertAlmostEqual(float(batch["real_behavior_action_switch_rate"]), 0.75, places=6)
        self.assertAlmostEqual(float(batch["real_behavior_action_oscillation_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(batch["real_reward_health"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["real_reward_degradation"]), 0.0, places=6)
        self.assertAlmostEqual(float(batch["real_task_cert_gate"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["real_corridor_occupancy_fraction"]), 0.4, places=6)
        self.assertAlmostEqual(float(batch["real_corridor_persistence_rate"]), 0.25, places=6)
        self.assertAlmostEqual(float(batch["real_corridor_entry_rate"]), 0.1, places=6)
        self.assertAlmostEqual(float(batch["real_corridor_exit_rate"]), 0.75, places=6)
        self.assertAlmostEqual(float(batch["real_policy_certified_anchor_available"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["real_policy_certified_anchor_age_steps"]), 60.0, places=6)
        self.assertAlmostEqual(float(batch["real_policy_kl_to_certified_anchor_mean"]), 0.8, places=6)
        self.assertAlmostEqual(float(batch["real_policy_certified_registry_available"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["real_policy_certified_registry_size"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["real_policy_kl_to_certified_registry_mean"]), 0.3, places=6)
        self.assertAlmostEqual(float(batch["behavior_policy_certified_registry_available"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_policy_certified_registry_size"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_policy_kl_to_certified_registry_mean"]), 0.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_policy_certified_registry_support_fraction"]), 1.0, places=6)
        self.assertAlmostEqual(
            float(batch["behavior_policy_geometry_registry_support_fraction"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_reward_semantic_registry_gate_mean"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_reward_semantic_registry_support_fraction"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_task_cert_imag_gate_mean"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_task_cert_real_gate_mean"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_task_cert_gate_mean"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_task_cert_support_fraction"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["real_imag_corridor_occupancy_gap"]),
            float(batch["behavior_corridor_occupancy_fraction"]) - 0.4,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["real_imag_corridor_persistence_gap"]),
            float(batch["behavior_corridor_persistence_rate"]) - 0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["real_imag_policy_anchor_drift_gap"]),
            0.8 - float(batch["behavior_policy_kl_to_eval_anchor_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["real_imag_policy_registry_drift_gap"]),
            0.3,
            places=6,
        )
        self.assertGreaterEqual(float(batch["real_imag_stability_disagreement"]), 0.0)
        self.assertAlmostEqual(float(loop.metrics["actor/real_behavior_action_entropy_mean"]), 0.25, places=6)
        self.assertAlmostEqual(float(loop.metrics["actor/real_corridor_persistence_rate"]), 0.25, places=6)
        self.assertAlmostEqual(
            float(loop.metrics["actor/real_policy_certified_registry_size"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(loop.metrics["actor/real_reward_health"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(loop.metrics["actor/real_imag_corridor_occupancy_gap"]),
            float(batch["real_imag_corridor_occupancy_gap"]),
            places=6,
        )

    def test_build_imagined_batch_can_use_pre_eval_certified_registry_support(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        with torch.no_grad():
            model.actor.net.weight.zero_()
            model.actor.net.bias.zero_()
            model.critic.weight.zero_()
            model.critic.bias.zero_()

        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=0.5,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.5,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=3,
            imagination_batch_size=1,
            imag_continue_prob_cap=1.0,
            adaptive_imag_continue_cap=False,
            adaptive_imag_idle_corridor_advantage_blend_max=0.1,
            adaptive_imag_idle_corridor_quantile=0.0,
            adaptive_imag_real_stability_use_pre_eval_registry_support=True,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(model=model, buffer=buffer, config=config, device=device, env=None)
        loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]],
            rewards=[[0.1, 0.2, 0.3]],
            values=[[0.0, 0.0, 0.0, 0.0]],
            continue_probs=[[1.0, 1.0, 1.0]],
            entropies=[[0.2, 0.4, 0.6]],
            log_probs=[[-0.2, -0.3, -0.4]],
            actions=[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
        )

        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.2,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.1,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 2.0,
        }
        loop.set_external_eval_feedback(220.0, step=100, telemetry=dict(telemetry))
        older_registry_actor = copy.deepcopy(loop._behavior_policy_eval_anchor_actor).to(device)
        older_registry_actor.eval()
        for param in older_registry_actor.parameters():
            param.requires_grad_(False)
        with torch.no_grad():
            model.actor.net.bias.copy_(torch.tensor([1.5, -1.5], device=device))
        current_eval_registry_actor = copy.deepcopy(model.actor).to(device)
        current_eval_registry_actor.eval()
        for param in current_eval_registry_actor.parameters():
            param.requires_grad_(False)
        loop._real_stability_certified_anchor_registry_actors = [
            older_registry_actor,
            current_eval_registry_actor,
        ]
        loop._real_stability_certified_anchor_registry_steps = [50, 100]
        loop._real_stability_certified_anchor_registry_evals = [180.0, 220.0]
        loop._real_stability_certified_anchor_registry_telemetries = [
            dict(telemetry),
            dict(telemetry),
        ]
        loop.global_step = 160

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(batch["behavior_policy_certified_registry_available"]), 1.0, places=6)
        self.assertAlmostEqual(float(batch["behavior_policy_certified_registry_size"]), 1.0, places=6)
        self.assertGreater(float(batch["behavior_policy_kl_to_certified_registry_mean"]), 0.0)
        self.assertLess(float(batch["behavior_policy_certified_registry_support_fraction"]), 1.0)

    def test_external_eval_feedback_state_is_separate_from_runtime_task_cert_telemetry(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=3,
            imagination_batch_size=1,
            imag_continue_prob_cap=1.0,
            adaptive_imag_continue_cap=False,
            adaptive_imag_idle_corridor_advantage_blend_max=0.1,
            adaptive_imag_idle_corridor_quantile=0.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]],
            rewards=[[0.1, 0.2, 0.3]],
            values=[[0.0, 0.0, 0.0, 0.0]],
            continue_probs=[[1.0, 1.0, 1.0]],
            entropies=[[0.2, 0.4, 0.6]],
            log_probs=[[-0.2, -0.3, -0.4]],
            actions=[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
        )
        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }
        loop.set_external_eval_feedback(220.0, step=100, telemetry=dict(telemetry))
        loop.set_external_eval_feedback(110.0, step=120, telemetry=dict(telemetry))
        loop._bootstrap_runtime_task_cert_telemetry = {
            "real_task_cert_state": 0.9,
            "real_task_cert_alarm": 0.1,
            "real_task_cert_recovery": 0.95,
        }
        loop.global_step = 160

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(
            float(loop._bootstrap_external_eval_feedback_telemetry["real_reward_degradation"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(loop._bootstrap_external_eval_feedback_telemetry["real_task_cert_gate"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(float(batch["real_reward_degradation"]), 0.5, places=6)
        self.assertAlmostEqual(float(batch["real_task_cert_gate"]), 0.5, places=6)
        self.assertAlmostEqual(float(batch["real_behavior_age_steps"]), 40.0, places=6)
        self.assertAlmostEqual(
            float(loop._bootstrap_runtime_task_cert_telemetry["real_task_cert_state"]),
            float(batch["behavior_policy_task_cert_real_gate_mean"]),
            places=6,
        )

    def test_external_eval_feedback_can_clamp_real_task_cert_gate_by_policy_anchor_drift(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=4)
        config = TrainingConfig(
            total_steps=1,
            num_train_steps=1,
            total_env_steps=1,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            imagination_only=True,
            adaptive_imag_task_cert_real_policy_anchor_gate_enabled=True,
            adaptive_imag_task_cert_real_policy_anchor_gate_kl_scale=0.2,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop._external_eval_best_mean = 220.0
        telemetry = {
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
        }

        loop.set_external_eval_feedback(110.0, step=120, telemetry=dict(telemetry))

        self.assertAlmostEqual(
            float(
                loop._bootstrap_external_eval_feedback_telemetry[
                    "real_policy_certified_anchor_health"
                ]
            ),
            0.2,
            places=6,
        )
        self.assertAlmostEqual(
            float(loop._bootstrap_external_eval_feedback_telemetry["real_task_cert_gate"]),
            0.2,
            places=6,
        )

    def test_registry_support_is_recertified_by_real_reward_health(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 1.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 3.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=3,
            imagination_batch_size=1,
            imag_continue_prob_cap=1.0,
            adaptive_imag_continue_cap=False,
            adaptive_imag_idle_corridor_advantage_blend_max=0.1,
            adaptive_imag_idle_corridor_quantile=0.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]],
            rewards=[[0.1, 0.2, 0.3]],
            values=[[0.0, 0.0, 0.0, 0.0]],
            continue_probs=[[1.0, 1.0, 1.0]],
            entropies=[[0.2, 0.4, 0.6]],
            log_probs=[[-0.2, -0.3, -0.4]],
            actions=[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
        )

        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }
        loop.set_external_eval_feedback(220.0, step=100, telemetry=dict(telemetry))
        loop.set_external_eval_feedback(110.0, step=120, telemetry=dict(telemetry))
        with torch.no_grad():
            model.actor.net.bias.copy_(torch.tensor([1.5, -1.5], device=device))
        registry_actor = copy.deepcopy(model.actor).to(device)
        registry_actor.eval()
        for param in registry_actor.parameters():
            param.requires_grad_(False)
        loop._real_stability_certified_anchor_registry_actors = [registry_actor]
        loop._real_stability_certified_anchor_registry_steps = [100]
        loop._real_stability_certified_anchor_registry_evals = [220.0]
        loop._real_stability_certified_anchor_registry_telemetries = [dict(telemetry)]
        loop.global_step = 160

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(batch["real_reward_health"]), 0.5, places=6)
        self.assertAlmostEqual(float(batch["real_reward_degradation"]), 0.5, places=6)
        self.assertAlmostEqual(float(batch["real_task_cert_gate"]), 0.5, places=6)
        self.assertAlmostEqual(
            float(batch["behavior_policy_geometry_registry_support_fraction"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_reward_semantic_registry_gate_mean"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_reward_semantic_registry_support_fraction"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_task_cert_imag_gate_mean"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_task_cert_real_gate_mean"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_task_cert_gate_mean"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_task_cert_support_fraction"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_certified_registry_support_fraction"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["actor_contract_certified_registry_support_mean"]),
            0.5,
            places=6,
        )

    def test_registry_support_requires_explicit_task_cert_channel_when_task_corridor_disagrees(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 0.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 1.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        buffer._select_uniform = lambda valid, batch_size, required: [(0, 0)] * batch_size

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            adaptive_imag_idle_corridor_advantage_blend_max=0.0,
            adaptive_imag_idle_corridor_clean_target_blend_max=0.0,
            adaptive_imag_idle_corridor_quantile=0.0,
            adaptive_imag_task_corridor_enabled=True,
            adaptive_imag_task_corridor_high_quantile=0.75,
            adaptive_imag_task_corridor_low_quantile=0.25,
            adaptive_imag_task_corridor_gate_tau=0.05,
            adaptive_imag_task_corridor_analytic_floor=0.25,
            adaptive_imag_task_corridor_confidence_scale=0.25,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        config.rl.use_reward_ema = False
        config.rl.use_advantage_normalization = False
        config.rl.actor_entropy_scale = 0.0
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[[[-1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
            rewards=[[0.0, 0.0]],
            values=[[6.0, 6.0, 6.0]],
            continue_probs=[[1.0, 1.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )

        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }
        loop.set_external_eval_feedback(220.0, step=100, telemetry=dict(telemetry))
        registry_actor = copy.deepcopy(model.actor).to(device)
        registry_actor.eval()
        for param in registry_actor.parameters():
            param.requires_grad_(False)
        loop._real_stability_certified_anchor_registry_actors = [registry_actor]
        loop._real_stability_certified_anchor_registry_steps = [100]
        loop._real_stability_certified_anchor_registry_evals = [220.0]
        loop._real_stability_certified_anchor_registry_telemetries = [dict(telemetry)]
        loop.global_step = 160

        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[10.0, 0.0]], device=device),
        }

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(batch["real_task_cert_gate"]), 1.0, places=6)
        self.assertAlmostEqual(
            float(batch["behavior_policy_geometry_registry_support_fraction"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_reward_semantic_registry_support_fraction"]),
            1.0,
            places=6,
        )
        self.assertLess(float(batch["task_corridor_gate_mean"]), 0.5)
        self.assertLess(
            float(batch["behavior_policy_task_cert_imag_gate_mean"]),
            0.5,
        )
        self.assertLess(
            float(batch["behavior_policy_task_cert_gate_mean"]),
            0.5,
        )
        self.assertLess(
            float(batch["behavior_policy_certified_registry_support_fraction"]),
            0.5,
        )
        self.assertLess(
            float(batch["actor_contract_certified_registry_support_mean"]),
            0.5,
        )

    def test_registry_support_uses_immediate_real_batch_reward_agreement_before_next_eval(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 0.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 1.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        buffer._select_uniform = lambda valid, batch_size, required: [(0, 0)] * batch_size

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            adaptive_imag_idle_corridor_advantage_blend_max=0.0,
            adaptive_imag_idle_corridor_clean_target_blend_max=0.0,
            adaptive_imag_idle_corridor_quantile=0.0,
            adaptive_imag_task_corridor_enabled=True,
            adaptive_imag_task_corridor_high_quantile=0.75,
            adaptive_imag_task_corridor_low_quantile=0.25,
            adaptive_imag_task_corridor_gate_tau=0.05,
            adaptive_imag_task_corridor_analytic_floor=0.25,
            adaptive_imag_task_corridor_confidence_scale=0.25,
            adaptive_imag_task_cert_reward_agreement_quantile=0.75,
            adaptive_imag_task_cert_recovery_rate=0.5,
            adaptive_imag_task_cert_alarm_rate=0.7,
            adaptive_imag_task_cert_state_floor=0.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        config.rl.use_reward_ema = False
        config.rl.use_advantage_normalization = False
        config.rl.actor_entropy_scale = 0.0
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            rewards=[[0.0, 0.0]],
            values=[[6.0, 6.0, 6.0]],
            continue_probs=[[1.0, 1.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )

        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }
        loop.set_external_eval_feedback(220.0, step=100, telemetry=dict(telemetry))
        registry_actor = copy.deepcopy(model.actor).to(device)
        registry_actor.eval()
        for param in registry_actor.parameters():
            param.requires_grad_(False)
        loop._real_stability_certified_anchor_registry_actors = [registry_actor]
        loop._real_stability_certified_anchor_registry_steps = [100]
        loop._real_stability_certified_anchor_registry_evals = [220.0]
        loop._real_stability_certified_anchor_registry_telemetries = [dict(telemetry)]
        loop.global_step = 160

        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[20.0, 0.0]], device=device),
        }

        batch = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(batch["real_task_cert_gate"]), 1.0, places=6)
        self.assertAlmostEqual(
            float(batch["behavior_policy_geometry_registry_support_fraction"]),
            1.0,
            places=6,
        )
        self.assertGreater(
            float(batch["behavior_policy_task_cert_imag_gate_mean"]),
            0.95,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_task_cert_real_eval_gate_mean"]),
            1.0,
            places=6,
        )
        self.assertLess(
            float(batch["behavior_policy_task_cert_real_reward_agreement_mean"]),
            0.1,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_task_cert_real_alarm_mean"]),
            float(batch["behavior_policy_task_cert_real_reward_agreement_mean"]),
            places=6,
        )
        self.assertGreater(
            float(batch["behavior_policy_task_cert_real_recovery_mean"]),
            float(batch["behavior_policy_task_cert_real_alarm_mean"]),
        )
        self.assertGreater(
            float(batch["behavior_policy_task_cert_real_gate_mean"]),
            float(batch["behavior_policy_task_cert_real_reward_agreement_mean"]),
        )
        self.assertLess(
            float(batch["behavior_policy_task_cert_real_gate_mean"]),
            0.25,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_task_cert_gate_mean"]),
            float(batch["behavior_policy_task_cert_real_gate_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(batch["behavior_policy_certified_registry_support_fraction"]),
            float(batch["behavior_policy_task_cert_real_gate_mean"]),
            places=6,
        )

    def test_task_cert_state_can_recover_above_alarm_floor_under_persistent_positive_task_agreement(self):
        device = torch.device("cpu")
        model = _TinyImagModel().to(device)
        buffer = ReplayBuffer(capacity=8)
        buffer.start_episode(initial_vitals=np.full((4,), 0.0, dtype=np.float32))
        buffer.add_step(
            vitals=np.full((4,), 1.0, dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=0.0,
        )
        buffer.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=1.0,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.0,
        )
        buffer._select_uniform = lambda valid, batch_size, required: [(0, 0)] * batch_size

        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            total_steps=1,
            num_train_steps=1,
            total_env_steps=2,
            batch_size=1,
            seq_len=4,
            wm_seq_len=4,
            wm_batch_size=1,
            rl_batch_size=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
            imagination_horizon=2,
            imagination_batch_size=1,
            adaptive_imag_idle_corridor_advantage_blend_max=0.0,
            adaptive_imag_idle_corridor_clean_target_blend_max=0.0,
            adaptive_imag_idle_corridor_quantile=0.0,
            adaptive_imag_task_corridor_enabled=True,
            adaptive_imag_task_corridor_high_quantile=0.75,
            adaptive_imag_task_corridor_low_quantile=0.25,
            adaptive_imag_task_corridor_gate_tau=0.05,
            adaptive_imag_task_corridor_analytic_floor=0.25,
            adaptive_imag_task_corridor_confidence_scale=0.25,
            adaptive_imag_task_cert_reward_agreement_quantile=0.75,
            adaptive_imag_task_cert_recovery_rate=0.5,
            adaptive_imag_task_cert_alarm_rate=0.7,
            adaptive_imag_task_cert_state_floor=0.0,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        config.rl.use_reward_ema = False
        config.rl.use_advantage_normalization = False
        config.rl.actor_entropy_scale = 0.0
        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=device,
            env=None,
        )
        loop.imagination_engine = _ContractFeatureImaginationEngine(
            device,
            policy_features=[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            rewards=[[0.0, 0.0]],
            values=[[6.0, 6.0, 6.0]],
            continue_probs=[[1.0, 1.0]],
            entropies=[[0.0, 0.0]],
            log_probs=[[0.0, 0.0]],
            actions=[[[1.0, 0.0], [1.0, 0.0]]],
        )

        telemetry = {
            "real_behavior_action_entropy_mean": 0.25,
            "real_behavior_action_switch_rate": 0.75,
            "real_behavior_action_oscillation_rate": 0.5,
            "real_corridor_occupancy_fraction": 0.4,
            "real_corridor_persistence_rate": 0.25,
            "real_corridor_entry_rate": 0.1,
            "real_corridor_exit_rate": 0.75,
            "real_policy_kl_to_certified_anchor_mean": 0.8,
            "real_policy_certified_anchor_available": 1.0,
            "real_policy_kl_to_certified_registry_mean": 0.3,
            "real_policy_certified_registry_available": 1.0,
            "real_policy_certified_registry_size": 1.0,
        }
        loop.set_external_eval_feedback(220.0, step=100, telemetry=dict(telemetry))
        loop._real_stability_last_telemetry["real_task_cert_state"] = 0.0
        registry_actor = copy.deepcopy(model.actor).to(device)
        registry_actor.eval()
        for param in registry_actor.parameters():
            param.requires_grad_(False)
        loop._real_stability_certified_anchor_registry_actors = [registry_actor]
        loop._real_stability_certified_anchor_registry_steps = [100]
        loop._real_stability_certified_anchor_registry_evals = [220.0]
        loop._real_stability_certified_anchor_registry_telemetries = [dict(telemetry)]
        loop.global_step = 160

        reference_real_batch = {
            "vitals": torch.zeros((1, 2, 4), device=device),
            "policy_features": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
                device=device,
            ),
            "use_precomputed_policy_outputs": True,
            "actions": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]],
                device=device,
            ),
            "value_real": torch.tensor([[20.0, 0.0]], device=device),
        }

        batch1 = loop._build_imagined_batch(reference_real_batch=reference_real_batch)
        batch2 = loop._build_imagined_batch(reference_real_batch=reference_real_batch)

        self.assertIsNotNone(batch1)
        self.assertIsNotNone(batch2)
        self.assertLess(
            float(batch1["behavior_policy_task_cert_real_reward_agreement_mean"]),
            0.1,
        )
        self.assertGreater(
            float(batch1["behavior_policy_task_cert_real_gate_mean"]),
            float(batch1["behavior_policy_task_cert_real_reward_agreement_mean"]),
        )
        self.assertGreater(
            float(batch2["behavior_policy_task_cert_real_gate_mean"]),
            float(batch1["behavior_policy_task_cert_real_gate_mean"]),
        )
        self.assertGreater(
            float(batch2["behavior_policy_task_cert_real_recovery_mean"]),
            float(batch2["behavior_policy_task_cert_real_alarm_mean"]),
        )
        self.assertAlmostEqual(
            float(batch2["behavior_policy_task_cert_gate_mean"]),
            float(batch2["behavior_policy_task_cert_real_gate_mean"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(batch2["behavior_policy_certified_registry_support_fraction"]),
            float(batch2["behavior_policy_task_cert_real_gate_mean"]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
