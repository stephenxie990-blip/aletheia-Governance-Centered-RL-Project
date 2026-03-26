import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore

import torch
import torch.nn as nn
import numpy as np

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.aletheia_actor_critic import CategoricalStraightThrough
from aletheia.aletheia_api import create_agent
from aletheia.aletheia_config import TrainingConfig
from aletheia.aletheia_train import ImaginationEngine, ReplayBuffer, TrainingLoop


class _DummyWMState:
    def __init__(self, x_t, h_shared, h_pred, z, s_ctrl):
        self.x_t = x_t
        self.h_shared = h_shared
        self.h_pred = h_pred
        self.z = z
        self.s_ctrl = s_ctrl

    def isolate_gradients(self, context: str, wall_strength: float = 1.0):
        if context == "all":
            return self.detach_all()
        if context == "rl":
            return _DummyWMState(
                self.x_t,
                self.h_shared.detach(),
                self.h_pred.detach(),
                self.z.detach(),
                self.s_ctrl,
            )
        if context == "policy":
            if wall_strength >= 1.0:
                x_t = self.x_t.detach()
            elif wall_strength <= 0.0:
                x_t = self.x_t
            else:
                x_t = wall_strength * self.x_t.detach() + (1.0 - wall_strength) * self.x_t
            return _DummyWMState(
                x_t,
                self.h_shared.detach(),
                self.h_pred.detach(),
                self.z.detach(),
                self.s_ctrl,
            )
        raise ValueError(f"unknown isolation context: {context}")

    def detach_all(self):
        return _DummyWMState(
            self.x_t.detach(),
            self.h_shared.detach(),
            self.h_pred.detach(),
            self.z.detach(),
            self.s_ctrl.detach(),
        )


class _DummyRSSM:
    def get_features(self, h_shared, z):
        return torch.cat([h_shared, z], dim=-1)


class _DummyWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_transition = _DummyRSSM()
        self._wm_config = SimpleNamespace(is_discrete_action=True)
        self.action_to_h = nn.Linear(2, 2, bias=False)
        self.reward_head = nn.Linear(4, 1)
        self.continue_head = nn.Linear(4, 1)
        self.context_call_count = 0

    def init_wm_state(self, batch_size, device):
        zeros2 = torch.zeros(batch_size, 2, device=device)
        zeros4 = torch.zeros(batch_size, 4, device=device)
        zeros1 = torch.zeros(batch_size, 1, device=device)
        return _DummyWMState(zeros4, zeros2, zeros2, zeros2, zeros1)

    def forward_context(self, obs, action_prev, prev_state):
        self.context_call_count += 1
        h_shared = obs[:, :2] + self.action_to_h(action_prev.float())
        z = obs[:, 2:4]
        return _DummyWMState(obs, h_shared, h_shared, z, prev_state.s_ctrl), SimpleNamespace()

    def forward_imagination(self, action_prev, prev_state):
        h_shared = prev_state.h_shared + self.action_to_h(action_prev.float())
        z = prev_state.z + 0.1
        feat = torch.cat([h_shared, z], dim=-1)
        next_state = _DummyWMState(prev_state.x_t, h_shared, h_shared, z, prev_state.s_ctrl)
        return next_state, SimpleNamespace(features_prior=feat)


class _DummyDiscreteActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, feat, temperature: float = 1.0, intent=None):
        logits = self.fc(feat)
        return CategoricalStraightThrough(logits=logits, temperature=temperature, unimix_ratio=0.0)


class _DummyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = nn.Linear(4, 1)

    def forward(self, feat, use_target: bool = False):
        if feat.dim() == 3:
            batch, steps, feat_dim = feat.shape
            out = self.v(feat.reshape(batch * steps, feat_dim)).squeeze(-1)
            return out.reshape(batch, steps)
        return self.v(feat).squeeze(-1)


class _DummyImagModel(nn.Module):
    def __init__(self, world_model=None, actor=None, critic=None):
        super().__init__()
        self.world_model = world_model or _DummyWorldModel()
        self.actor = actor or _DummyDiscreteActor()
        self.critic = critic or _DummyCritic()
        self.router = None

    def policy_from_wm_state(self, wm_state):
        return torch.cat([wm_state.h_shared, wm_state.z], dim=-1)


class TestPureImaginationPath(unittest.TestCase):
    def test_imagination_rollout_backprops_actor_but_not_world_model_or_critic(self):
        torch.manual_seed(0)
        model = _DummyImagModel()
        engine = ImaginationEngine(model=model, device=torch.device("cpu"), max_horizon=3)
        engine._gamma_buf = 0.99
        engine.gae_lambda = 0.95

        initial_state = torch.randn(4, 4)
        initial_prev_action = torch.zeros(4, 2)
        initial_prev_action[:, 0] = 1.0

        rollout = engine.imagine_rollout_differentiable(
            initial_state=initial_state,
            initial_prev_action=initial_prev_action,
            horizon=2,
            actor=model.actor,
            critic=model.critic,
        )
        loss = rollout["returns"].mean()
        loss.backward()

        actor_grad = sum(
            p.grad.abs().sum().item() if p.grad is not None else 0.0
            for p in model.actor.parameters()
        )
        world_model_grad = sum(
            p.grad.abs().sum().item() if p.grad is not None else 0.0
            for p in model.world_model.parameters()
        )
        critic_grad = sum(
            p.grad.abs().sum().item() if p.grad is not None else 0.0
            for p in model.critic.parameters()
        )

        self.assertGreater(actor_grad, 0.0)
        self.assertEqual(world_model_grad, 0.0)
        self.assertEqual(critic_grad, 0.0)

    def test_imagination_log_probs_do_not_backprop_through_sampled_actions(self):
        torch.manual_seed(2)
        model = _DummyImagModel()
        engine = ImaginationEngine(model=model, device=torch.device("cpu"), max_horizon=3)
        engine._gamma_buf = 0.99
        engine.gae_lambda = 0.95

        initial_state = torch.randn(3, 4)
        initial_prev_action = torch.zeros(3, 2)
        initial_prev_action[:, 0] = 1.0

        rollout = engine.imagine_rollout_differentiable(
            initial_state=initial_state,
            initial_prev_action=initial_prev_action,
            horizon=2,
            actor=model.actor,
            critic=model.critic,
        )
        self.assertTrue(bool(rollout["log_probs"].requires_grad))
        grad = torch.autograd.grad(
            rollout["log_probs"].sum(),
            rollout["actions"],
            allow_unused=True,
        )[0]
        if grad is not None:
            self.assertAlmostEqual(float(grad.abs().sum().item()), 0.0, places=7)

    def test_seed_state_uses_full_context_sequence(self):
        torch.manual_seed(1)
        model = _DummyImagModel(world_model=_DummyWorldModel())
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            imagination_only=False,
            total_steps=1,
            num_train_steps=1,
            total_env_steps=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            batch_size=1,
            seq_len=2,
            wm_seq_len=2,
            wm_batch_size=1,
            rl_batch_size=1,
            imagination_horizon=2,
        )
        loop = TrainingLoop(
            model=model,
            buffer=ReplayBuffer(capacity=4, store_obs=False),
            config=config,
            device=torch.device("cpu"),
            env=None,
        )

        vitals = torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0, 2.0]]])
        actions = torch.tensor([[[1.0, 0.0],
                                 [0.0, 1.0]]])

        seed_state = loop._build_imagination_seed_state(vitals, actions)
        self.assertIsNotNone(seed_state)
        self.assertEqual(model.world_model.context_call_count, vitals.shape[1])
        self.assertTrue(torch.allclose(seed_state.x_t, vitals[:, -1]))

    def test_world_model_imagination_advances_projected_embedding(self):
        env = gym.make("CartPole-v1")
        try:
            agent = create_agent(env, device="cpu", seed=0)
            wm = agent.world_model
            device = torch.device("cpu")

            obs, _ = env.reset(seed=0)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            prev_action = torch.zeros(1, env.action_space.n, dtype=torch.float32, device=device)
            prev_action[:, 0] = 1.0

            wm_state = wm.init_wm_state(1, device)
            wm_state, _ = wm.forward_context(obs_t, prev_action, wm_state)

            action = torch.zeros_like(prev_action)
            action[:, 1] = 1.0
            next_state, _ = wm.forward_imagination(action, wm_state)
            self.assertIsNotNone(next_state.x_proj)
            self.assertTrue(torch.allclose(next_state.x_t, next_state.x_proj))
            self.assertFalse(torch.allclose(next_state.x_t, wm_state.x_t))

            next_state_2, _ = wm.forward_imagination(action, next_state)
            self.assertTrue(torch.allclose(next_state_2.x_t, next_state_2.x_proj))
            self.assertFalse(torch.allclose(next_state_2.x_t, next_state.x_t))
            self.assertFalse(torch.allclose(next_state_2.s_ctrl, next_state.s_ctrl))
        finally:
            env.close()

    def test_cartpole_no_longer_forces_fixed_gamma_for_imagination(self):
        model = _DummyImagModel()
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            imagination_only=False,
            total_steps=1,
            num_train_steps=1,
            total_env_steps=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_horizon=2,
        )
        config.env_name = "CartPole-v1"
        loop = TrainingLoop(
            model=model,
            buffer=ReplayBuffer(capacity=4, store_obs=False),
            config=config,
            device=torch.device("cpu"),
            env=None,
        )
        self.assertIsNotNone(loop.imagination_engine)
        self.assertFalse(bool(loop.imagination_engine.use_fixed_gamma_for_imag))

    def test_imagined_batch_emits_open_loop_audit_metrics(self):
        torch.manual_seed(3)
        model = _DummyImagModel()
        config = TrainingConfig(
            config_mode="strict",
            validation_mode="off",
            imagination_only=True,
            total_steps=1,
            num_train_steps=1,
            total_env_steps=16,
            wm_pretrain_steps=0,
            warmup_steps=0,
            batch_size=1,
            seq_len=16,
            wm_seq_len=16,
            wm_batch_size=1,
            rl_batch_size=1,
            imagination_horizon=2,
            imagination_batch_size=1,
            log_interval=1000,
            eval_interval=1000,
            save_interval=1000,
        )
        buffer = ReplayBuffer(capacity=8, store_obs=False)
        buffer.start_episode(np.zeros((4,), dtype=np.float32))
        for step in range(16):
            action = np.array([1.0, 0.0], dtype=np.float32) if step % 2 == 0 else np.array([0.0, 1.0], dtype=np.float32)
            done = step == 15
            buffer.add_step(
                vitals=np.full((4,), float(step + 1), dtype=np.float32),
                action=action,
                reward=1.0,
                done=done,
                terminated=done,
                truncated=False,
                log_prob=0.0,
            )

        loop = TrainingLoop(
            model=model,
            buffer=buffer,
            config=config,
            device=torch.device("cpu"),
            env=None,
        )

        batch = loop._build_imagined_batch()

        self.assertIsNotNone(batch)
        self.assertAlmostEqual(float(loop.metrics["imag/open_loop_audit_active"]), 1.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/open_loop_audit_horizon"]), 2.0, places=6)
        self.assertAlmostEqual(float(loop.metrics["imag/open_loop_audit_context_len"]), 2.0, places=6)
        for key in (
            "imag/open_loop_audit_feature_l1_step1",
            "imag/open_loop_audit_feature_l1_step2",
            "imag/open_loop_audit_feature_l1_mean",
            "imag/open_loop_audit_reward_gap_mean",
            "imag/open_loop_audit_continue_gap_mean",
            "imag/open_loop_audit_value_gap_mean",
            "imag/open_loop_audit_teacher_reward_abs_to_real_mean",
            "imag/open_loop_audit_imag_continue_abs_to_real_mean",
            "imag/open_loop_audit_teacher_value_abs_to_short_return_mean",
        ):
            self.assertIn(key, loop.metrics)
            self.assertGreaterEqual(float(loop.metrics[key]), 0.0)


if __name__ == "__main__":
    unittest.main()
