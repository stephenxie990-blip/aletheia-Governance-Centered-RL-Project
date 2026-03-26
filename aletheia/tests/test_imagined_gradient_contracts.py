import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.aletheia_actor_critic import CategoricalStraightThrough
from aletheia.aletheia_train import ImaginationEngine, isolate_policy_wm_state
from aletheia.aletheia_world_model import WMState


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

    def init_wm_state(self, batch_size, device):
        return WMState(
            x_t=torch.zeros(batch_size, 4, device=device),
            h_shared=torch.zeros(batch_size, 2, device=device),
            h_pred=torch.zeros(batch_size, 2, device=device),
            z=torch.zeros(batch_size, 2, device=device),
            s_ctrl=torch.zeros(batch_size, 1, device=device),
        )

    def forward_context(self, obs, action_prev, prev_state):
        h_shared = obs[:, :2] + self.action_to_h(action_prev.float())
        z = obs[:, 2:4]
        next_state = WMState(
            x_t=obs,
            h_shared=h_shared,
            h_pred=h_shared,
            z=z,
            s_ctrl=prev_state.s_ctrl,
        )
        return next_state, SimpleNamespace()

    def forward_imagination(self, action_prev, prev_state):
        h_shared = prev_state.h_shared + self.action_to_h(action_prev.float())
        z = prev_state.z + 0.1
        features = torch.cat([h_shared, z], dim=-1)
        next_state = WMState(
            x_t=prev_state.x_t,
            h_shared=h_shared,
            h_pred=h_shared,
            z=z,
            s_ctrl=prev_state.s_ctrl,
        )
        return next_state, SimpleNamespace(features_prior=features)


class _DummyDiscreteActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, feat, temperature: float = 1.0, intent=None):
        del intent
        logits = self.fc(feat)
        return CategoricalStraightThrough(
            logits=logits,
            temperature=temperature,
            unimix_ratio=0.0,
        )


class _DummyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = nn.Linear(4, 1)

    def forward(self, feat, use_target: bool = False):
        del use_target
        if feat.dim() == 3:
            batch, steps, feat_dim = feat.shape
            out = self.v(feat.reshape(batch * steps, feat_dim)).squeeze(-1)
            return out.reshape(batch, steps)
        return self.v(feat).squeeze(-1)


class _DummyImagModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.world_model = _DummyWorldModel()
        self.actor = _DummyDiscreteActor()
        self.critic = _DummyCritic()
        self.router = None

    def policy_from_wm_state(self, wm_state):
        return torch.cat([wm_state.h_shared, wm_state.z], dim=-1)


def _sum_grad(module: nn.Module) -> float:
    return sum(
        p.grad.abs().sum().item() if p.grad is not None else 0.0
        for p in module.parameters()
    )


class TestImaginedGradientContracts(unittest.TestCase):
    def test_wm_state_coerce_legacy_tuple_respects_explicit_obs_embed_dim(self):
        h = torch.zeros(3, 5)
        z = torch.zeros(3, 4)

        state = WMState.coerce(
            (h, z),
            ctrl_dim=2,
            obs_embed_dim=7,
        )

        self.assertEqual(tuple(state.x_t.shape), (3, 7))
        self.assertEqual(tuple(state.s_ctrl.shape), (3, 2))
        self.assertTrue(torch.equal(state.h_shared, h))
        self.assertTrue(torch.equal(state.z, z))

    def _run_rollout(self, horizon: int):
        torch.manual_seed(0)
        model = _DummyImagModel()
        engine = ImaginationEngine(
            model=model,
            device=torch.device("cpu"),
            max_horizon=max(3, horizon),
        )
        engine._gamma_buf = 0.99
        engine.gae_lambda = 0.95

        initial_state = torch.randn(4, 4)
        initial_prev_action = torch.zeros(4, 2)
        initial_prev_action[:, 0] = 1.0

        self.assertTrue(all(p.requires_grad for p in model.actor.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.world_model.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.critic.parameters()))

        rollout = engine.imagine_rollout_differentiable(
            initial_state=initial_state,
            initial_prev_action=initial_prev_action,
            horizon=horizon,
            actor=model.actor,
            critic=model.critic,
        )

        self.assertTrue(bool(rollout["returns"].requires_grad))
        self.assertTrue(all(p.requires_grad for p in model.actor.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.world_model.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.critic.parameters()))

        rollout["returns"].mean().backward()
        return model

    def test_single_step_imagined_rollout_keeps_actor_only_gradient_contract(self):
        model = self._run_rollout(horizon=1)

        self.assertGreater(_sum_grad(model.actor), 0.0)
        self.assertAlmostEqual(_sum_grad(model.world_model), 0.0, places=7)
        self.assertAlmostEqual(_sum_grad(model.critic), 0.0, places=7)

    def test_multi_step_imagined_rollout_keeps_actor_only_gradient_contract(self):
        model = self._run_rollout(horizon=3)

        self.assertGreater(_sum_grad(model.actor), 0.0)
        self.assertAlmostEqual(_sum_grad(model.world_model), 0.0, places=7)
        self.assertAlmostEqual(_sum_grad(model.critic), 0.0, places=7)

    def test_wmstate_rl_isolation_only_keeps_xt_and_sctrl_live(self):
        state = WMState(
            x_t=torch.randn(2, 4, requires_grad=True),
            h_shared=torch.randn(2, 3, requires_grad=True),
            h_pred=torch.randn(2, 3, requires_grad=True),
            z=torch.randn(2, 2, requires_grad=True),
            s_ctrl=torch.randn(2, 1, requires_grad=True),
            x_proj=torch.randn(2, 2, requires_grad=True),
            z_task=torch.randn(2, 2, requires_grad=True),
        )

        isolated = state.isolate_gradients(context="rl")
        self.assertTrue(bool(isolated.x_t.requires_grad))
        self.assertTrue(bool(isolated.s_ctrl.requires_grad))
        self.assertFalse(bool(isolated.h_shared.requires_grad))
        self.assertFalse(bool(isolated.h_pred.requires_grad))
        self.assertFalse(bool(isolated.z.requires_grad))
        self.assertFalse(bool(isolated.x_proj.requires_grad))
        self.assertFalse(bool(isolated.z_task.requires_grad))

        (isolated.x_t.sum() + isolated.s_ctrl.sum()).backward()
        self.assertGreater(float(state.x_t.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(state.s_ctrl.grad.abs().sum().item()), 0.0)
        self.assertIsNone(state.h_shared.grad)
        self.assertIsNone(state.h_pred.grad)
        self.assertIsNone(state.z.grad)
        self.assertIsNone(state.x_proj.grad)
        self.assertIsNone(state.z_task.grad)

    def test_wmstate_all_isolation_detaches_everything(self):
        state = WMState(
            x_t=torch.randn(2, 4, requires_grad=True),
            h_shared=torch.randn(2, 3, requires_grad=True),
            h_pred=torch.randn(2, 3, requires_grad=True),
            z=torch.randn(2, 2, requires_grad=True),
            s_ctrl=torch.randn(2, 1, requires_grad=True),
            x_proj=torch.randn(2, 2, requires_grad=True),
            z_task=torch.randn(2, 2, requires_grad=True),
        )

        isolated = state.isolate_gradients(context="all")
        self.assertFalse(bool(isolated.x_t.requires_grad))
        self.assertFalse(bool(isolated.h_shared.requires_grad))
        self.assertFalse(bool(isolated.h_pred.requires_grad))
        self.assertFalse(bool(isolated.z.requires_grad))
        self.assertFalse(bool(isolated.s_ctrl.requires_grad))
        self.assertFalse(bool(isolated.x_proj.requires_grad))
        self.assertFalse(bool(isolated.z_task.requires_grad))

    def test_wmstate_coerce_materializes_legacy_tuple_state(self):
        h_shared = torch.randn(2, 3)
        z = torch.randn(2, 2)

        coerced = WMState.coerce((h_shared, z), ctrl_dim=5)

        self.assertTrue(isinstance(coerced, WMState))
        self.assertEqual(tuple(coerced.h_shared.shape), (2, 3))
        self.assertEqual(tuple(coerced.h_pred.shape), (2, 3))
        self.assertEqual(tuple(coerced.z.shape), (2, 2))
        self.assertEqual(tuple(coerced.s_ctrl.shape), (2, 5))
        self.assertEqual(tuple(coerced.x_t.shape), (2, 3))

    def test_wmstate_coerce_accepts_legacy_tuple_state(self):
        h = torch.randn(2, 3)
        z = torch.randn(2, 2)

        coerced = WMState.coerce((h, z), ctrl_dim=4)

        self.assertTrue(torch.equal(coerced.h_shared, h))
        self.assertTrue(torch.equal(coerced.h_pred, h))
        self.assertTrue(torch.equal(coerced.z, z))
        self.assertEqual(tuple(coerced.x_t.shape), (2, 3))
        self.assertEqual(tuple(coerced.s_ctrl.shape), (2, 4))

    def test_wmstate_coerce_rejects_invalid_legacy_state_shape(self):
        with self.assertRaisesRegex(TypeError, "WMState.coerce expects"):
            WMState.coerce(torch.randn(2, 3), ctrl_dim=4)

    def test_isolate_policy_wm_state_rejects_noncanonical_adapter_objects(self):
        class _InvalidState:
            def __init__(self, feat: torch.Tensor):
                self.x_t = feat
                self.x_proj = feat
                self.s_ctrl = torch.zeros(feat.shape[0], 1, device=feat.device)
                self.z_task = None

        legacy = _InvalidState(torch.randn(2, 4))

        with self.assertRaisesRegex(TypeError, "Legacy policy WMState adapters are no longer supported"):
            isolate_policy_wm_state(legacy, wall_strength=0.5)
