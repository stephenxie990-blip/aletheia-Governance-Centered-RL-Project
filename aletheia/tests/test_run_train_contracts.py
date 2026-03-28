import sys
import json
import random
import tempfile
import unittest
import copy
import io
from dataclasses import asdict
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import numpy as np

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import aletheia.aletheia_api as api
import aletheia.aletheia_train as train_mod
from aletheia.aletheia_foundation import (
    make_config_policy_overrides_for_env,
    get_activation_class,
    set_activation_validation_mode,
)
from aletheia._agent_checkpoint_schema import (
    build_agent_checkpoint_payload,
    restore_agent_checkpoint_modules,
)
from aletheia._training_checkpoint_schema import (
    TrainingCheckpointRestorePolicy,
    build_training_checkpoint_payload,
    restore_training_checkpoint_payload,
)
from aletheia._checkpoint_metadata import (
    read_agent_creation_overrides_from_checkpoint,
)
from aletheia._run_artifact_schema import (
    RunArtifactPaths,
    build_eval_record,
    build_resolved_config_payload,
    build_summary_payload,
    build_train_metric_record,
)
from aletheia.tests._artifact_test_helpers import (
    agent_bootstrap_bundle_of as _agent_bootstrap_bundle_of,
    effective_training_config_of as _effective_training_config_of,
    load_resolved_config_payload as _load_resolved_config_payload,
    load_summary_payload as _load_summary_payload,
    make_minimal_agent_checkpoint as _make_minimal_agent_checkpoint,
    optimizer_bundle_payload_of as _optimizer_bundle_payload_of,
    replay_buffer_payload_of as _replay_buffer_payload_of,
    run_artifact_paths_for as _run_artifact_paths_for,
    training_state_payload_of as _training_state_payload_of,
)
from aletheia.aletheia_config import (
    coerce_config_policy_overrides,
    ConfigPolicyOverrides,
    ConfigBundle,
    AgentBootstrapTrainParams,
    TrainParams,
    AGENT_BOOTSTRAP_TRAIN_FIELDS,
    TRAIN_PARAMS_BOOTSTRAP_METADATA_FIELDS,
    TrainingConfig,
    export_config_policy_wire_overrides,
    export_factory_bridge_overrides,
    FactoryBridgeOverrides,
    parse_factory_bridge_overrides,
    parse_config_policy_overrides,
)
from aletheia.aletheia_actor_critic import TanhTransformedDistribution


class _CountingEnv:
    def __init__(self, done_after: int = 2):
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(n=2)
        self.done_after = int(done_after)
        self.step_count = 0
        self.reset_count = 0
        self.reset_seeds = []

    def reset(self, seed=None):
        self.step_count = 0
        self.reset_count += 1
        self.reset_seeds.append(seed)
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        del action
        self.step_count += 1
        obs = np.full((4,), float(self.step_count), dtype=np.float32)
        reward = 1.0
        terminated = self.step_count >= self.done_after
        truncated = False
        return obs, reward, terminated, truncated, {}

    def close(self):
        return None


class _AlternatingObsEnv:
    def __init__(self, observations):
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(n=2)
        self._observations = [
            np.asarray(obs, dtype=np.float32) for obs in observations
        ]
        self._idx = 0
        self.reset_count = 0

    def reset(self, seed=None):
        del seed
        self._idx = 0
        self.reset_count += 1
        return self._observations[0].copy(), {}

    def step(self, action):
        del action
        self._idx += 1
        obs = self._observations[min(self._idx, len(self._observations) - 1)].copy()
        terminated = self._idx >= len(self._observations) - 1
        return obs, 1.0, terminated, False, {}

    def close(self):
        return None


class _NeverDoneEnv:
    def __init__(self):
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(n=2)
        self.step_count = 0
        self.reset_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        self.reset_count += 1
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        del action
        self.step_count += 1
        obs = np.full((4,), float(self.step_count), dtype=np.float32)
        reward = float(self.step_count)
        return obs, reward, False, False, {}

    def close(self):
        return None


class _ExplodingEnv:
    def __init__(self):
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(n=2)
        self.reset_count = 0

    def reset(self, seed=None):
        del seed
        self.reset_count += 1
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        del action
        raise RuntimeError("env boom")

    def close(self):
        return None


class _DummyAgent:
    def __init__(self):
        self.profile = SimpleNamespace(is_discrete=True, action_dim=2)
        self.config = ConfigBundle(train=AgentBootstrapTrainParams())
        self.device = torch.device("cpu")
        self.reset_calls = 0
        self.act_calls = 0
        self.deterministic_flags = []
        self._prev_action = None
        self.buffer = None
        self.world_model = nn.Linear(1, 1)
        self.actor = nn.Linear(1, 1)
        self.critic = nn.Linear(1, 1)
        self.router = None
        self.will = None
        self.saved_checkpoints = []

    def reset(self):
        self.reset_calls += 1
        self._prev_action = None

    def act(self, obs, deterministic=False):
        del obs
        idx = self.act_calls % 2
        one_hot = np.zeros((1, 2), dtype=np.float32)
        one_hot[0, idx] = 1.0
        self._prev_action = torch.from_numpy(one_hot)
        self.deterministic_flags.append(bool(deterministic))
        info = {
            "value": np.array([10.0 + self.act_calls], dtype=np.float32),
            "log_prob": np.array([-0.5 - self.act_calls], dtype=np.float32),
        }
        self.act_calls += 1
        return idx, info

    def save(
        self,
        path: str,
        *,
        effective_training_config=None,
        agent_bootstrap_bundle=None,
    ):
        self.saved_checkpoints.append(
            {
                "path": str(path),
                "effective_training_config": copy.deepcopy(effective_training_config),
                "agent_bootstrap_bundle": copy.deepcopy(agent_bootstrap_bundle),
            }
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("checkpoint", encoding="utf-8")


class _RuntimeBuffer:
    def __init__(self):
        self.current = {"marker": ["live"]}
        self._needs_initial_obs = False
        self.start_episode_calls = 0

    def start_episode(self, initial_obs=None):
        del initial_obs
        self.start_episode_calls += 1
        self.current = {"marker": [f"eval-{self.start_episode_calls}"]}
        self._needs_initial_obs = True


class _ResettingDummyAgent(_DummyAgent):
    def __init__(self):
        super().__init__()
        self.buffer = _RuntimeBuffer()
        self._step_count = 17
        self._needs_initial_obs = False

    def reset(self):
        self.reset_calls += 1
        self._prev_action = None
        if self.buffer is not None:
            self.buffer.start_episode()
        self._needs_initial_obs = True
        self._step_count = 0


class _RngConsumingAgent(_DummyAgent):
    def reset(self):
        random.random()
        np.random.rand()
        torch.rand(1)
        super().reset()

    def act(self, obs, deterministic=False):
        random.random()
        np.random.rand()
        torch.rand(1)
        return super().act(obs, deterministic=deterministic)


class _DummyBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)

    def __len__(self):
        return 0


class _DummyOptBundle:
    def state_dict(self):
        return {"dummy": 1}


class _FakeTrainingLoop:
    last_instance = None

    def __init__(self, *args, **kwargs):
        del args
        self.kwargs = kwargs
        self.model = nn.Linear(1, 1)
        self.opt_bundle = _DummyOptBundle()
        self.global_step = 0
        self.episode_count = 0
        self.env_steps_collected = 0
        self.log_fn = kwargs.get("logger_fn")
        self.external_eval_feedback_calls = []
        _FakeTrainingLoop.last_instance = self

    def set_external_eval_feedback(self, mean_return, step=None, telemetry=None):
        self.external_eval_feedback_calls.append(
            {
                "mean_return": float(mean_return),
                "step": None if step is None else int(step),
                "telemetry": dict(telemetry or {}),
            }
        )

    def run(
        self,
        num_steps,
        data_collector,
        resume_from=None,
        resume_restore_policy=None,
        eval_fn=None,
        save_fn=None,
    ):
        del data_collector
        self.global_step = int(num_steps)
        self.episode_count = 3
        self.env_steps_collected = 17
        self.resume_from = resume_from
        self.resume_restore_policy = resume_restore_policy
        if self.log_fn is not None:
            self.log_fn(
                {
                    "train/global_step": float(self.global_step),
                    "train/env_steps_collected": float(self.env_steps_collected),
                    "train/batch_source": "imag",
                    "train/imag_ratio": 1.0,
                    "loss_actor": 1.25,
                    "loss_critic": 2.5,
                    "critic/continue_prob": 0.8,
                    "actor/target_mean": 3.5,
                },
                self.global_step,
            )
        if eval_fn is not None:
            eval_fn(self.model, self.global_step)
        if save_fn is not None:
            save_fn(self.model, self.global_step)


class _ModeDist:
    def __init__(self, logits: torch.Tensor):
        self._cat = torch.distributions.Categorical(logits=logits)
        self._num_classes = logits.shape[-1]

    @property
    def mode(self) -> torch.Tensor:
        idx = self._cat.probs.argmax(dim=-1)
        return torch.nn.functional.one_hot(idx, num_classes=self._num_classes).float()

    def sample(self) -> torch.Tensor:
        idx = self._cat.sample()
        return torch.nn.functional.one_hot(idx, num_classes=self._num_classes).float()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() >= 1 and action.shape[-1] == self._num_classes:
            idx = action.argmax(dim=-1)
        else:
            idx = action.long()
        return self._cat.log_prob(idx)

    def entropy(self) -> torch.Tensor:
        return self._cat.entropy()


class _ModeAwareWMState:
    def __init__(self, feat: torch.Tensor):
        self.x_proj = feat
        self.x_t = feat
        self.s_ctrl = None
        self.z_task = None

    def isolate_gradients(self, context: str, wall_strength: float = 1.0):
        self.last_context = context
        self.last_wall_strength = wall_strength
        return self


class _InvalidModeAwareWMState:
    def __init__(self, feat: torch.Tensor):
        self.x_proj = feat
        self.x_t = feat
        self.s_ctrl = None
        self.z_task = None

class _ModeAwareWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(4))
        self.seen_training = []

    def init_wm_state(self, batch_size: int, device: torch.device, deterministic: bool = False):
        del deterministic
        return _ModeAwareWMState(torch.zeros(batch_size, 4, device=device))

    def forward_context(self, obs_t, a_prev, wm_state, deterministic_state=False):
        del a_prev, wm_state, deterministic_state
        self.seen_training.append(bool(self.training))
        feat = self.linear(obs_t)
        return _ModeAwareWMState(feat), SimpleNamespace(decoder_logvar=None)


class _InvalidModeAwareWorldModel(_ModeAwareWorldModel):
    def init_wm_state(self, batch_size: int, device: torch.device, deterministic: bool = False):
        del deterministic
        return _InvalidModeAwareWMState(torch.zeros(batch_size, 4, device=device))

    def forward_context(self, obs_t, a_prev, wm_state, deterministic_state=False):
        del a_prev, wm_state, deterministic_state
        self.seen_training.append(bool(self.training))
        feat = self.linear(obs_t)
        return _InvalidModeAwareWMState(feat), SimpleNamespace(decoder_logvar=None)


class _ModeAwareRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(4))
        self.seen_training = []

    def forward_components(self, x_rl, s_ctrl, z_task, logvar, x_t, x_proj, wall_strength):
        del s_ctrl, z_task, logvar, x_t, x_proj, wall_strength
        self.seen_training.append(bool(self.training))
        return SimpleNamespace(f_policy=self.linear(x_rl))


class _ModeAwareActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]))
        self.seen_training = []

    def forward(self, feat):
        self.seen_training.append(bool(self.training))
        return _ModeDist(self.linear(feat))


class _ContinuousModeAwareActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor([[0.25, 0.0, 0.0, 0.0], [-0.25, 0.0, 0.0, 0.0]])
            )
        self.seen_training = []

    def forward(self, feat):
        self.seen_training.append(bool(self.training))
        mean = self.linear(feat)
        scale = torch.full_like(mean, 0.5)
        return TanhTransformedDistribution(mean, scale, entropy_samples=8)


class _FailingModeAwareActor(nn.Module):
    def forward(self, feat):
        del feat
        raise RuntimeError("telemetry actor boom")


class _BadEntropyModeDist(_ModeDist):
    def entropy(self):
        return object()


class _BadEntropyModeAwareActor(_ModeAwareActor):
    def forward(self, feat):
        self.seen_training.append(bool(self.training))
        return _BadEntropyModeDist(self.linear(feat))


class _MissingPolicyStatsDist:
    def entropy(self):
        return torch.tensor([0.0], dtype=torch.float32)


class _BadKlModeAwareActor(nn.Module):
    def forward(self, feat):
        del feat
        return _MissingPolicyStatsDist()


class _ModeAwareCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(0.25)
        self.seen_training = []

    def forward(self, feat):
        self.seen_training.append(bool(self.training))
        value = self.linear(feat)
        return SimpleNamespace(values_real_main=value, uncertainty_raw=torch.zeros_like(value))


class _TinyCollectorModel(nn.Module):
    def forward(self, vitals, temperature: float = 1.0, intent=None):
        del temperature, intent
        batch = vitals.shape[0]
        loc = torch.zeros(batch, 2, device=vitals.device)
        dist = torch.distributions.Normal(loc, torch.ones_like(loc))
        value = torch.zeros(batch, 1, device=vitals.device)
        return dist, value


class TestRunTrainContracts(unittest.TestCase):
    def test_agent_rollout_collector_preserves_agent_outputs_and_forced_truncation(self):
        agent = _DummyAgent()
        env = _NeverDoneEnv()
        collector = api.AgentRolloutCollector(agent=agent, env=env, max_steps=2)

        batch = collector.collect(num_steps=3, deterministic=True)

        self.assertIsNotNone(batch)
        self.assertEqual(batch["actions"].shape, (3, 2))
        self.assertTrue(np.allclose(batch["actions"][0], np.array([1.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(batch["actions"][1], np.array([0.0, 1.0], dtype=np.float32)))
        self.assertEqual(float(batch["truncated"][1]), 1.0)
        self.assertEqual(float(batch["dones"][1]), 1.0)
        self.assertEqual(float(batch["terminated"][1]), 0.0)
        self.assertAlmostEqual(float(batch["values"][0]), 10.0, places=5)
        self.assertAlmostEqual(float(batch["log_probs"][2]), -2.5, places=5)
        self.assertEqual(agent.reset_calls, 2)
        self.assertEqual(len(collector.episode_returns), 1)
        self.assertAlmostEqual(float(collector.episode_returns[0]), 3.0, places=5)

    def test_agent_rollout_collector_collect_propagates_errors(self):
        agent = _DummyAgent()
        collector = api.AgentRolloutCollector(agent=agent, env=_ExplodingEnv(), max_steps=2)

        with self.assertRaisesRegex(RuntimeError, "env boom"):
            collector.collect(num_steps=1, deterministic=True)

    def test_rollout_collector_collect_propagates_errors(self):
        collector = train_mod.RolloutCollector(
            model=_TinyCollectorModel(),
            env=_ExplodingEnv(),
            device=torch.device("cpu"),
            max_steps=2,
        )

        with self.assertRaisesRegex(RuntimeError, "env boom"):
            collector.collect(num_steps=1, deterministic=True)

    def test_evaluate_agent_resets_agent_every_episode(self):
        agent = _DummyAgent()
        env = _CountingEnv(done_after=2)

        results = api._evaluate_agent(env, agent, episodes=3, max_steps=5)

        self.assertEqual(agent.reset_calls, 3)
        self.assertEqual(env.reset_count, 3)
        self.assertEqual(agent.deterministic_flags, [True] * 6)
        self.assertAlmostEqual(results["mean"], 2.0, places=5)
        self.assertAlmostEqual(results["mean_length"], 2.0, places=5)

    def test_evaluate_agent_restores_runtime_state(self):
        agent = _DummyAgent()
        env = _CountingEnv(done_after=2)
        agent._wm_state = {"latent": torch.tensor([[3.0]])}
        agent._prev_action = torch.tensor([[0.0, 1.0]])

        results = api._evaluate_agent(env, agent, episodes=2, max_steps=5)

        self.assertEqual(env.reset_count, 2)
        self.assertAlmostEqual(results["mean"], 2.0, places=5)
        self.assertIsInstance(agent._wm_state, dict)
        self.assertTrue(torch.equal(agent._wm_state["latent"], torch.tensor([[3.0]])))
        self.assertTrue(torch.equal(agent._prev_action, torch.tensor([[0.0, 1.0]])))

    def test_evaluate_agent_restores_buffer_runtime_state(self):
        agent = _ResettingDummyAgent()
        env = _CountingEnv(done_after=2)
        agent._wm_state = {"latent": torch.tensor([[5.0]])}
        agent._prev_action = torch.tensor([[1.0, 0.0]])

        results = api._evaluate_agent(env, agent, episodes=2, max_steps=5)

        self.assertAlmostEqual(results["mean"], 2.0, places=5)
        self.assertEqual(agent.reset_calls, 2)
        self.assertEqual(agent.buffer.start_episode_calls, 2)
        self.assertEqual(agent.buffer.current, {"marker": ["live"]})
        self.assertFalse(agent.buffer._needs_initial_obs)
        self.assertEqual(agent._step_count, 17)
        self.assertFalse(agent._needs_initial_obs)
        self.assertIsInstance(agent._wm_state, dict)
        self.assertTrue(torch.equal(agent._wm_state["latent"], torch.tensor([[5.0]])))
        self.assertTrue(torch.equal(agent._prev_action, torch.tensor([[1.0, 0.0]])))

    def test_evaluate_agent_restores_global_random_state(self):
        agent = _RngConsumingAgent()
        env = _CountingEnv(done_after=2)

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()

        api._evaluate_agent(env, agent, episodes=2, max_steps=5)

        self.assertEqual(random.getstate(), python_state)
        restored_numpy_state = np.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_state[0])
        self.assertTrue(np.array_equal(restored_numpy_state[1], numpy_state[1]))
        self.assertEqual(restored_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_evaluate_agent_preserves_seed_progression_for_raw_env(self):
        agent = _DummyAgent()
        env = _CountingEnv(done_after=1)

        results = api._evaluate_agent(env, agent, episodes=3, max_steps=5)

        self.assertEqual(env.reset_seeds, [42, None, None])
        self.assertEqual(env.reset_count, 3)
        self.assertAlmostEqual(results["mean"], 1.0, places=5)
        self.assertAlmostEqual(results["std"], 0.0, places=5)

    def test_evaluate_agent_emits_real_stability_telemetry(self):
        handle = self._make_mode_handle()
        anchor_actor = copy.deepcopy(handle.actor)
        anchor_actor.eval()
        for param in anchor_actor.parameters():
            param.requires_grad_(False)
        handle._real_stability_eval_anchor_actor = anchor_actor
        handle._real_stability_eval_anchor_registry_actors = [anchor_actor]
        env = _AlternatingObsEnv(
            observations=[
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ]
        )

        results = api._evaluate_agent(env, handle, episodes=1, max_steps=8)

        telemetry = results["telemetry"]
        self.assertGreater(float(telemetry["real_behavior_action_entropy_mean"]), 0.0)
        self.assertAlmostEqual(float(telemetry["real_behavior_action_switch_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(telemetry["real_behavior_action_oscillation_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(telemetry["real_policy_certified_anchor_available"]), 1.0, places=6)
        self.assertAlmostEqual(float(telemetry["real_policy_kl_to_certified_anchor_mean"]), 0.0, places=6)
        self.assertAlmostEqual(float(telemetry["real_policy_certified_registry_available"]), 1.0, places=6)
        self.assertAlmostEqual(float(telemetry["real_policy_certified_registry_size"]), 1.0, places=6)
        self.assertAlmostEqual(float(telemetry["real_policy_kl_to_certified_registry_mean"]), 0.0, places=6)
        self.assertAlmostEqual(float(telemetry["real_corridor_occupancy_fraction"]), 1.0, places=6)
        self.assertAlmostEqual(float(telemetry["real_corridor_persistence_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(telemetry["real_corridor_entry_rate"]), 0.0, places=6)
        self.assertAlmostEqual(float(telemetry["real_corridor_exit_rate"]), 0.0, places=6)

    def test_evaluate_agent_raises_when_anchor_telemetry_forward_fails(self):
        handle = self._make_mode_handle()
        handle._real_stability_eval_anchor_actor = _FailingModeAwareActor()
        env = _AlternatingObsEnv(
            observations=[
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "anchor policy forward"):
            api._evaluate_agent(env, handle, episodes=1, max_steps=4)

    def test_evaluate_agent_raises_when_registry_telemetry_forward_fails(self):
        handle = self._make_mode_handle()
        handle._real_stability_eval_anchor_registry_actors = [_FailingModeAwareActor()]
        env = _AlternatingObsEnv(
            observations=[
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "registry policy forward"):
            api._evaluate_agent(env, handle, episodes=1, max_steps=4)

    def test_evaluate_agent_raises_when_entropy_telemetry_stat_is_malformed(self):
        handle = self._make_mode_handle()
        handle.actor = _BadEntropyModeAwareActor()
        env = _AlternatingObsEnv(
            observations=[
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "action entropy"):
            api._evaluate_agent(env, handle, episodes=1, max_steps=4)

    def test_evaluate_agent_raises_when_anchor_telemetry_kl_is_unavailable(self):
        handle = self._make_mode_handle()
        handle._real_stability_eval_anchor_actor = _BadKlModeAwareActor()
        env = _AlternatingObsEnv(
            observations=[
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "anchor policy KL"):
            api._evaluate_agent(env, handle, episodes=1, max_steps=4)

    def test_evaluate_agent_supports_tanh_transformed_policy_kl_telemetry(self):
        handle = self._make_continuous_mode_handle()
        anchor_actor = _ContinuousModeAwareActor()
        anchor_actor.load_state_dict(handle.actor.state_dict())
        for param in anchor_actor.parameters():
            param.requires_grad_(False)
        handle._real_stability_eval_anchor_actor = anchor_actor
        handle._real_stability_eval_anchor_registry_actors = [anchor_actor]
        env = _AlternatingObsEnv(
            observations=[
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ]
        )

        results = api._evaluate_agent(env, handle, episodes=1, max_steps=6)

        telemetry = results["telemetry"]
        self.assertGreaterEqual(
            float(telemetry["real_behavior_action_entropy_mean"]),
            0.0,
        )
        self.assertAlmostEqual(
            float(telemetry["real_policy_kl_to_certified_anchor_mean"]),
            0.0,
            places=4,
        )
        self.assertAlmostEqual(
            float(telemetry["real_policy_kl_to_certified_registry_mean"]),
            0.0,
            places=4,
        )

    def test_training_loop_run_stops_when_eval_requests_early_stop(self):
        events = []
        loop = SimpleNamespace()
        loop.config = SimpleNamespace(total_steps=3, num_train_steps=3, total_env_steps=3, wm_pretrain_steps=0, warmup_steps=0, imagination_only=False)
        loop.device = torch.device("cpu")
        loop.global_step = 0
        loop.episode_count = 0
        loop.env_steps_collected = 0
        loop._steps_since_collect = 1
        loop.train_steps_per_cycle = 1
        loop.collect_steps_per_cycle = 1
        loop.model = object()
        loop.logger_fn = lambda metrics, step=0: events.append(("log", step, dict(metrics)))
        loop._should_collect = lambda: False
        loop._add_to_buffer = lambda result: None
        loop._sync_episode_counts = lambda collector: None
        loop._build_real_batch = lambda: {"real": True}
        loop._build_imagined_batch = lambda: {"imag": True}
        loop._build_wm_batch = lambda: {"wm": True}
        loop._select_rl_batch = lambda real_batch, imag_batch: (real_batch, "real", 0.0)
        loop.should_log = lambda: False
        loop.should_eval = lambda: True
        loop.should_save = lambda: True

        def train_step(**kwargs):
            del kwargs
            loop.global_step += 1
            return {"loss_actor": 0.0}

        loop.train_step = train_step

        def eval_fn(model, step):
            events.append(("eval", step, model))
            return {"mean_return": 500.0, "stop_training": True}

        def save_fn(model, step):
            events.append(("save", step, model))

        train_mod.TrainingLoop.run(loop, num_steps=3, data_collector=None, eval_fn=eval_fn, save_fn=save_fn)

        self.assertEqual(loop.global_step, 1)
        self.assertEqual([kind for kind, *_ in events], ["eval", "log", "save"])

    def test_training_loop_run_rejects_collectors_returning_none(self):
        loop = SimpleNamespace()
        loop.config = SimpleNamespace(total_steps=1, num_train_steps=1, total_env_steps=1, wm_pretrain_steps=0, warmup_steps=0, imagination_only=False)
        loop.device = torch.device("cpu")
        loop.global_step = 0
        loop.episode_count = 0
        loop.env_steps_collected = 0
        loop._steps_since_collect = 1
        loop.train_steps_per_cycle = 1
        loop.collect_steps_per_cycle = 1
        loop.model = object()
        loop.logger_fn = lambda metrics, step=0: None
        loop._should_collect = lambda: True
        loop._add_to_buffer = lambda result: (_ for _ in ()).throw(AssertionError("should not add None batch"))
        loop._sync_episode_counts = lambda collector: None
        loop._build_real_batch = lambda: {"real": True}
        loop._build_imagined_batch = lambda: {"imag": True}
        loop._build_wm_batch = lambda: {"wm": True}
        loop._select_rl_batch = lambda real_batch, imag_batch: (real_batch, "real", 0.0)
        loop.should_log = lambda: False
        loop.should_eval = lambda: False
        loop.should_save = lambda: False

        def train_step(**kwargs):
            del kwargs
            loop.global_step += 1
            return {"loss_actor": 0.0}

        loop.train_step = train_step

        class _NullCollector:
            episode_returns = []

            def collect(self, num_steps, deterministic=False):
                del num_steps, deterministic
                return None

        with self.assertRaisesRegex(RuntimeError, "Data collector returned None"):
            train_mod.TrainingLoop.run(loop, num_steps=1, data_collector=_NullCollector())

    def test_training_loop_run_limits_final_collection_to_remaining_env_steps(self):
        collect_requests = []
        loop = SimpleNamespace()
        loop.config = SimpleNamespace(
            total_steps=4,
            num_train_steps=4,
            total_env_steps=5,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=False,
        )
        loop.device = torch.device("cpu")
        loop.global_step = 0
        loop.episode_count = 0
        loop.env_steps_collected = 0
        loop._steps_since_collect = 2
        loop.train_steps_per_cycle = 2
        loop.collect_steps_per_cycle = 3
        loop.model = object()
        loop.imagination_engine = object()
        loop.logger_fn = lambda metrics, step=0: None
        loop._should_collect = lambda: loop._steps_since_collect >= loop.train_steps_per_cycle
        loop._add_to_buffer = lambda result: setattr(
            loop,
            "env_steps_collected",
            int(loop.env_steps_collected) + int(len(result["actions"])),
        )
        loop._sync_episode_counts = lambda collector: None
        loop._build_real_batch = lambda: {"real": True}
        loop._build_imagined_batch = lambda reference_real_batch=None: {"imag": True}
        loop._build_wm_batch = lambda: {"wm": True}
        loop._select_rl_batch = lambda real_batch, imag_batch: (real_batch, "real", 0.0)
        loop.should_log = lambda: False
        loop.should_eval = lambda: False
        loop.should_save = lambda: False

        def train_step(**kwargs):
            del kwargs
            loop.global_step += 1
            return {"loss_actor": 0.0}

        loop.train_step = train_step

        class _Collector:
            episode_returns = []

            def collect(self, num_steps, deterministic=False):
                del deterministic
                collect_requests.append(int(num_steps))
                return {
                    "actions": np.zeros((int(num_steps), 1), dtype=np.float32),
                }

        train_mod.TrainingLoop.run(loop, num_steps=4, data_collector=_Collector())

        self.assertEqual(collect_requests, [3, 2])
        self.assertEqual(loop.global_step, 4)
        self.assertEqual(loop.env_steps_collected, 5)

    def test_training_loop_run_resume_preserves_saved_collect_phase(self):
        collect_requests = []
        loop = SimpleNamespace()
        loop.config = SimpleNamespace(
            total_steps=8,
            num_train_steps=8,
            total_env_steps=6,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=False,
        )
        loop.device = torch.device("cpu")
        loop.global_step = 0
        loop.episode_count = 0
        loop.env_steps_collected = 0
        loop._steps_since_collect = 0
        loop.train_steps_per_cycle = 4
        loop.collect_steps_per_cycle = 3
        loop.model = object()
        loop.opt_bundle = None
        loop.buffer = None
        loop.imagination_engine = object()
        loop._episode_return_ema = 0.0
        loop._episode_return_initialized = False
        loop._last_episode_count = 0
        loop._last_return_episode_count = 0
        loop._last_seen_episode_idx = -1
        loop._restore_adaptive_compensation_state = lambda *args, **kwargs: None
        loop.logger_fn = lambda metrics, step=0: None
        loop._should_collect = lambda: loop._steps_since_collect >= loop.train_steps_per_cycle
        loop._add_to_buffer = lambda result: setattr(
            loop,
            "env_steps_collected",
            int(loop.env_steps_collected) + int(len(result["actions"])),
        )
        loop._sync_episode_counts = lambda collector: None
        loop._build_real_batch = lambda: {"real": True}
        loop._build_imagined_batch = lambda reference_real_batch=None: {"imag": True}
        loop._build_wm_batch = lambda: {"wm": True}
        loop._select_rl_batch = lambda real_batch, imag_batch: (real_batch, "real", 0.0)
        loop.should_log = lambda: False
        loop.should_eval = lambda: False
        loop.should_save = lambda: False

        def train_step(**kwargs):
            del kwargs
            loop.global_step += 1
            return {"loss_actor": 0.0}

        loop.train_step = train_step

        class _Collector:
            episode_returns = []

            def collect(self, num_steps, deterministic=False):
                del deterministic
                collect_requests.append(int(num_steps))
                return {
                    "actions": np.zeros((int(num_steps), 1), dtype=np.float32),
                }

        state = train_mod.TrainingState(
            config=TrainingConfig(),
            device=torch.device("cpu"),
        )
        state.global_step = 5
        state.total_samples = 6
        state.steps_since_collect = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = Path(tmpdir) / "resume.pt"
            resume_path.write_text("resume", encoding="utf-8")
            with mock.patch.object(
                train_mod.TrainingStateManager,
                "load",
                return_value=state,
            ):
                train_mod.TrainingLoop.run(
                    loop,
                    num_steps=8,
                    data_collector=_Collector(),
                    resume_from=str(resume_path),
                )

        self.assertEqual(collect_requests, [])
        self.assertEqual(loop.global_step, 8)
        self.assertEqual(loop.env_steps_collected, 6)
        self.assertEqual(loop._steps_since_collect, 4)

    def test_training_loop_run_resume_infers_collect_phase_for_legacy_checkpoints(self):
        collect_requests = []
        loop = SimpleNamespace()
        loop.config = SimpleNamespace(
            total_steps=8,
            num_train_steps=8,
            total_env_steps=6,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=False,
        )
        loop.device = torch.device("cpu")
        loop.global_step = 0
        loop.episode_count = 0
        loop.env_steps_collected = 0
        loop._steps_since_collect = 0
        loop.train_steps_per_cycle = 4
        loop.collect_steps_per_cycle = 3
        loop.model = object()
        loop.opt_bundle = None
        loop.buffer = None
        loop.imagination_engine = object()
        loop._episode_return_ema = 0.0
        loop._episode_return_initialized = False
        loop._last_episode_count = 0
        loop._last_return_episode_count = 0
        loop._last_seen_episode_idx = -1
        loop._restore_adaptive_compensation_state = lambda *args, **kwargs: None
        loop.logger_fn = lambda metrics, step=0: None
        loop._should_collect = lambda: loop._steps_since_collect >= loop.train_steps_per_cycle
        loop._add_to_buffer = lambda result: setattr(
            loop,
            "env_steps_collected",
            int(loop.env_steps_collected) + int(len(result["actions"])),
        )
        loop._sync_episode_counts = lambda collector: None
        loop._build_real_batch = lambda: {"real": True}
        loop._build_imagined_batch = lambda reference_real_batch=None: {"imag": True}
        loop._build_wm_batch = lambda: {"wm": True}
        loop._select_rl_batch = lambda real_batch, imag_batch: (real_batch, "real", 0.0)
        loop.should_log = lambda: False
        loop.should_eval = lambda: False
        loop.should_save = lambda: False

        def train_step(**kwargs):
            del kwargs
            loop.global_step += 1
            return {"loss_actor": 0.0}

        loop.train_step = train_step

        class _Collector:
            episode_returns = []

            def collect(self, num_steps, deterministic=False):
                del deterministic
                collect_requests.append(int(num_steps))
                return {
                    "actions": np.zeros((int(num_steps), 1), dtype=np.float32),
                }

        legacy_state = SimpleNamespace(
            global_step=5,
            episode_count=0,
            total_samples=6,
            episode_return_ema=0.0,
            episode_return_initialized=False,
            last_episode_count=0,
            last_return_episode_count=0,
            last_seen_episode_idx=-1,
            adaptive_compensation_state={},
            best_step=0,
            best_eval_return=-float("inf"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = Path(tmpdir) / "resume.pt"
            resume_path.write_text("resume", encoding="utf-8")
            with mock.patch.object(
                train_mod.TrainingStateManager,
                "load",
                return_value=legacy_state,
            ):
                train_mod.TrainingLoop.run(
                    loop,
                    num_steps=8,
                    data_collector=_Collector(),
                    resume_from=str(resume_path),
                )

        self.assertEqual(collect_requests, [])
        self.assertEqual(loop.global_step, 8)
        self.assertEqual(loop.env_steps_collected, 6)
        self.assertEqual(loop._steps_since_collect, 4)

    def test_training_loop_run_rejects_imagination_only_without_engine(self):
        loop = SimpleNamespace()
        loop.config = SimpleNamespace(
            total_steps=1,
            num_train_steps=1,
            total_env_steps=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=True,
        )
        loop.device = torch.device("cpu")
        loop.global_step = 0
        loop.episode_count = 0
        loop.env_steps_collected = 0
        loop._steps_since_collect = 1
        loop.train_steps_per_cycle = 1
        loop.collect_steps_per_cycle = 1
        loop.model = object()
        loop.imagination_engine = None
        loop.logger_fn = lambda metrics, step=0: None
        loop._should_collect = lambda: False
        loop._add_to_buffer = lambda result: None
        loop._sync_episode_counts = lambda collector: None
        loop._build_real_batch = lambda: {"real": True}
        loop._build_imagined_batch = lambda reference_real_batch=None: None
        loop._build_wm_batch = lambda: {"wm": True}
        loop._select_rl_batch = lambda real_batch, imag_batch: (real_batch, "real", 1.0)
        loop.should_log = lambda: False
        loop.should_eval = lambda: False
        loop.should_save = lambda: False
        loop.train_step = lambda **kwargs: {"loss_actor": 0.0}

        with self.assertRaisesRegex(RuntimeError, "imagination_engine is None"):
            train_mod.TrainingLoop.run(loop, num_steps=1, data_collector=None)

    def test_training_loop_run_rejects_missing_batches_without_collector(self):
        loop = SimpleNamespace()
        loop.config = SimpleNamespace(
            total_steps=1,
            num_train_steps=1,
            total_env_steps=1,
            wm_pretrain_steps=0,
            warmup_steps=0,
            imagination_only=False,
        )
        loop.device = torch.device("cpu")
        loop.global_step = 0
        loop.episode_count = 0
        loop.env_steps_collected = 0
        loop._steps_since_collect = 1
        loop.train_steps_per_cycle = 1
        loop.collect_steps_per_cycle = 1
        loop.model = object()
        loop.imagination_engine = object()
        loop.logger_fn = lambda metrics, step=0: None
        loop._should_collect = lambda: False
        loop._add_to_buffer = lambda result: None
        loop._sync_episode_counts = lambda collector: None
        loop._build_real_batch = lambda: None
        loop._build_imagined_batch = lambda reference_real_batch=None: None
        loop._build_wm_batch = lambda: None
        loop._select_rl_batch = lambda real_batch, imag_batch: (None, "none", 0.0)
        loop.should_log = lambda: False
        loop.should_eval = lambda: False
        loop.should_save = lambda: False
        loop.train_step = lambda **kwargs: {"loss_actor": 0.0}

        with self.assertRaisesRegex(RuntimeError, "No available batch \\(real/imag\\)"):
            train_mod.TrainingLoop.run(loop, num_steps=1, data_collector=None)

    def test_training_loop_run_rejects_missing_wm_batch_without_collector(self):
        loop = SimpleNamespace()
        loop.config = SimpleNamespace(
            total_steps=1,
            num_train_steps=1,
            total_env_steps=1,
            wm_pretrain_steps=1,
            warmup_steps=0,
            imagination_only=False,
        )
        loop.device = torch.device("cpu")
        loop.global_step = 0
        loop.episode_count = 0
        loop.env_steps_collected = 0
        loop._steps_since_collect = 1
        loop.train_steps_per_cycle = 1
        loop.collect_steps_per_cycle = 1
        loop.model = object()
        loop.imagination_engine = object()
        loop.logger_fn = lambda metrics, step=0: None
        loop._should_collect = lambda: False
        loop._add_to_buffer = lambda result: None
        loop._sync_episode_counts = lambda collector: None
        loop._build_real_batch = lambda: None
        loop._build_imagined_batch = lambda reference_real_batch=None: {"imag": True}
        loop._build_wm_batch = lambda: None
        loop._select_rl_batch = lambda real_batch, imag_batch: (imag_batch, "imag", 1.0)
        loop.should_log = lambda: False
        loop.should_eval = lambda: False
        loop.should_save = lambda: False
        loop.train_step = lambda **kwargs: {"loss_actor": 0.0}

        with self.assertRaisesRegex(RuntimeError, "Skip-RL phase requires WM batch"):
            train_mod.TrainingLoop.run(loop, num_steps=1, data_collector=None)

    def test_training_config_strict_allows_distinct_env_step_budget(self):
        cfg = TrainingConfig(
            config_mode="strict",
            validation_mode="strict",
            total_steps=4,
            num_train_steps=4,
            total_env_steps=32,
            wm_pretrain_steps=0,
            warmup_steps=0,
        )

        self.assertEqual(int(cfg.total_steps), 4)
        self.assertEqual(int(cfg.num_train_steps), 4)
        self.assertEqual(int(cfg.total_env_steps), 32)

    def test_normalize_training_config_keeps_total_env_steps_distinct_in_strict_mode(self):
        cfg = train_mod._normalize_training_config(
            {
                "config_mode": "strict",
                "validation_mode": "strict",
                "total_steps": 4,
                "total_env_steps": 32,
                "wm_pretrain_steps": 0,
                "warmup_steps": 0,
            }
        )

        self.assertEqual(int(cfg.total_steps), 4)
        self.assertEqual(int(cfg.num_train_steps), 4)
        self.assertEqual(int(cfg.total_env_steps), 32)

    def test_normalize_training_config_strict_rejects_legacy_field_aliases(self):
        with self.assertRaisesRegex(
            ValueError,
            "Legacy TrainingConfig aliases are no longer supported",
        ):
            train_mod._normalize_training_config(
                {
                    "config_mode": "strict",
                    "validation_mode": "strict",
                    "sequence_length": 8,
                    "wm_pretrain_steps": 0,
                    "warmup_steps": 0,
                }
            )

    def test_training_config_preserves_explicit_strict_validation_in_compat_mode(self):
        cfg = TrainingConfig(
            config_mode="compat",
            validation_mode="strict",
            total_steps=4,
            num_train_steps=4,
            total_env_steps=32,
            wm_pretrain_steps=0,
            warmup_steps=0,
        )

        self.assertEqual(str(cfg.validation_mode), "strict")

    def test_unknown_activation_raises_in_explicit_strict_validation_mode(self):
        set_activation_validation_mode("strict")
        try:
            with self.assertRaisesRegex(ValueError, "Unknown activation"):
                get_activation_class("definitely_not_a_real_activation")
        finally:
            set_activation_validation_mode("warn")

    def test_normalize_training_config_compat_rejects_legacy_field_aliases_without_opt_in(self):
        with self.assertRaisesRegex(
            ValueError,
            "Legacy TrainingConfig aliases are no longer supported",
        ):
            train_mod._normalize_training_config(
                {
                    "config_mode": "compat",
                    "validation_mode": "warn",
                    "sequence_length": 8,
                    "buffer_size": 64,
                    "num_train_steps": 4,
                    "total_env_steps": 32,
                    "wm_pretrain_steps": 0,
                    "warmup_steps": 0,
                }
            )

    def test_normalize_training_config_rejects_legacy_alias_flag_inside_payload(self):
        with self.assertRaisesRegex(
            ValueError,
            "allow_legacy_field_aliases is no longer supported",
        ):
            train_mod._normalize_training_config(
                {
                    "config_mode": "compat",
                    "validation_mode": "warn",
                    "allow_legacy_field_aliases": True,
                    "sequence_length": 8,
                    "buffer_size": 64,
                    "num_train_steps": 4,
                    "total_env_steps": 32,
                    "wm_pretrain_steps": 0,
                    "warmup_steps": 0,
                }
            )

    def test_training_config_from_dict_completes_canonical_sibling_fields(self):
        cfg = TrainingConfig.from_dict(
            {
                "config_mode": "compat",
                "validation_mode": "warn",
                "seq_len": 8,
                "batch_size": 64,
                "total_steps": 4,
                "total_env_steps": 32,
                "wm_pretrain_steps": 0,
                "warmup_steps": 0,
            }
        )

        self.assertEqual(int(cfg.seq_len), 8)
        self.assertEqual(int(cfg.wm_seq_len), 8)
        self.assertEqual(int(cfg.batch_size), 64)
        self.assertEqual(int(cfg.wm_batch_size), 64)
        self.assertEqual(int(cfg.total_steps), 4)
        self.assertEqual(int(cfg.num_train_steps), 4)

    def test_training_config_from_dict_rejects_removed_deprecated_fields(self):
        with self.assertRaisesRegex(
            ValueError,
            "Deprecated TrainingConfig fields are no longer supported",
        ):
            TrainingConfig.from_dict(
                {
                    "adaptive_imag_critic_trust_target_enabled": True,
                }
            )

    def test_load_agent_bridge_accepts_only_bootstrap_train_payload(self):
        env = _CountingEnv(done_after=2)
        create_calls = []

        class _LoadedAgent:
            def __init__(self):
                self.load_calls = []

            def load(self, path, strict=True, allow_unsafe_fallback=False):
                self.load_calls.append((path, strict, allow_unsafe_fallback))

        loaded_agent = _LoadedAgent()

        def fake_create_agent(env_obj, config_overrides=None, device=None, seed=None):
            create_calls.append((env_obj, config_overrides, device, seed))
            return loaded_agent

        checkpoint = _make_minimal_agent_checkpoint(
            agent_bootstrap_bundle={
                "train": {
                    "buffer_size": 123,
                    "batch_size": 8,
                    "use_compile": True,
                },
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            torch.save(checkpoint, ckpt_path)
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent):
                restored = api.load_agent(str(ckpt_path), env=env, device="cpu")

        self.assertIs(restored, loaded_agent)
        _, config_overrides, device, seed = create_calls[0]
        self.assertEqual(device, "cpu")
        self.assertIsNone(seed)
        self.assertEqual(config_overrides["buffer_size"], 123)
        self.assertEqual(config_overrides["batch_size"], 8)
        self.assertTrue(config_overrides["use_compile"])

    def test_load_agent_rejects_checkpoint_train_payload_with_buffer_capacity_alias(self):
        env = _CountingEnv(done_after=2)

        checkpoint = _make_minimal_agent_checkpoint(
            agent_bootstrap_bundle={
                "train": {"buffer_capacity": 123, "batch_size": 8, "sequence_length": 11},
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            torch.save(checkpoint, ckpt_path)
            with self.assertRaisesRegex(
                ValueError,
                "unsupported fields .*buffer_capacity",
            ):
                api.load_agent(str(ckpt_path), env=env, device="cpu")

    def test_load_agent_rejects_checkpoint_train_payload_with_deprecated_aliases(self):
        env = _CountingEnv(done_after=2)

        checkpoint = {
            "world_model": {},
            "actor": {},
            "critic": {},
            "agent_bootstrap_bundle": {
                "train": {"batch_length": 11, "wm_seq_len": 11, "num_train_steps": 4},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            torch.save(checkpoint, ckpt_path)
            with self.assertRaisesRegex(
                ValueError,
                "unsupported fields .*batch_length.*num_train_steps.*wm_seq_len",
            ):
                api.load_agent(str(ckpt_path), env=env, device="cpu")

    def test_load_agent_rejects_checkpoint_train_payload_with_seq_len_alias(self):
        env = _CountingEnv(done_after=2)

        checkpoint = {
            "world_model": {},
            "actor": {},
            "critic": {},
            "agent_bootstrap_bundle": {
                "train": {"buffer_size": 123, "batch_size": 8, "seq_len": 11},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            torch.save(checkpoint, ckpt_path)
            with self.assertRaisesRegex(
                ValueError,
                "unsupported fields .*seq_len",
            ):
                api.load_agent(str(ckpt_path), env=env, device="cpu")

    def test_training_loop_derives_components_from_model(self):
        actor = nn.Linear(4, 2)
        critic = nn.Linear(4, 1)
        world_model = nn.Linear(4, 4)
        model = train_mod.build_training_model(
            actor=actor,
            critic=critic,
            world_model=world_model,
        )

        loop = train_mod.TrainingLoop(
            model=model,
            buffer=_DummyBuffer(capacity=8),
            config=TrainingConfig(
                total_steps=1,
                num_train_steps=1,
                wm_pretrain_steps=0,
                warmup_steps=0,
            ),
            env=None,
            device=torch.device("cpu"),
        )

        self.assertIs(loop.model, model)
        self.assertIs(loop.actor, actor)
        self.assertIs(loop.critic, critic)
        self.assertIs(loop.world_model, world_model)

    def test_run_train_rejects_unimplemented_dataset_path(self):
        args = SimpleNamespace(env="Dummy-v0", device="cpu", seed=0, render=False, verbose=False)
        with self.assertRaises(NotImplementedError):
            api.run_train(
                args,
                input_spec={
                    "env": _CountingEnv(),
                    "eval_env": _CountingEnv(),
                    "dataset_path": "/tmp/offline.npz",
                },
            )

    def test_parse_overrides_rejects_invalid_json_payload(self):
        with self.assertRaisesRegex(ValueError, "override JSON string"):
            api._parse_overrides('{"rl": ')

        with tempfile.TemporaryDirectory() as tmpdir:
            overrides_path = Path(tmpdir) / "bad_overrides.json"
            overrides_path.write_text('{"rl": ', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "override file"):
                api._parse_overrides(str(overrides_path))

    def test_parse_overrides_rejects_non_mapping_payload(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            api._parse_overrides('["not", "a", "mapping"]')

    def test_run_train_uses_agent_collector_and_writes_resume_artifacts(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)
        collector_sentinel = object()
        created_collectors = []
        eval_calls = []
        best_agent = _DummyAgent()

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        def fake_create_replay_buffer(capacity, store_obs=False):
            del store_obs
            return _DummyBuffer(capacity=capacity)

        def fake_create_agent_rollout_collector(agent, env, max_steps=1000):
            created_collectors.append((agent, env, max_steps))
            return collector_sentinel

        def fake_evaluate_agent(env_obj, eval_agent, episodes, max_steps=1000):
            eval_calls.append((env_obj, eval_agent, episodes, max_steps))
            mean = 123.0 if eval_agent is agent else 111.0
            return {
                "mean": mean,
                "std": 0.0,
                "min": mean,
                "max": mean,
                "mean_length": 2.0,
                "telemetry": {
                    "real_behavior_action_entropy_mean": 0.4,
                    "real_behavior_action_switch_rate": 0.25,
                    "real_behavior_action_oscillation_rate": 0.1,
                    "real_corridor_occupancy_fraction": 0.8,
                    "real_corridor_persistence_rate": 0.6,
                    "real_corridor_entry_rate": 0.2,
                    "real_corridor_exit_rate": 0.4,
                    "real_policy_kl_to_certified_anchor_mean": 0.05,
                    "real_policy_certified_anchor_available": 1.0,
                },
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = str(Path(tmpdir) / "resume_in.pt")
            Path(resume_path).write_text("resume", encoding="utf-8")
            resume_state = train_mod.TrainingStateManager.create(
                config=TrainingConfig(),
                device=torch.device("cpu"),
            )
            args = SimpleNamespace(
                env="Dummy-v0",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=resume_path,
                steps=64,
                update_steps=2,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.0,
                warmup_ratio=0.0,
                imagination_only=False,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=1,
                save_interval=1,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=True,
                overrides=None,
                argv=["scripts/cartpole_train.py", "--steps", "64"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", side_effect=fake_create_agent_rollout_collector), \
                 mock.patch.object(api, "_evaluate_agent", side_effect=fake_evaluate_agent), \
                 mock.patch.object(api, "load_agent", return_value=best_agent), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=fake_create_replay_buffer), \
                 mock.patch.object(
                     train_mod.TrainingStateManager,
                     "load",
                     return_value=resume_state,
                 ):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

            self.assertEqual(result["status"], 0)
            self.assertEqual(created_collectors, [(agent, train_env, 5)])
            cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
            self.assertEqual(int(cfg.total_steps), 2)
            self.assertEqual(int(cfg.num_train_steps), 2)
            self.assertEqual(int(cfg.total_env_steps), 64)
            self.assertEqual(_FakeTrainingLoop.last_instance.resume_from, resume_path)
            policy = _FakeTrainingLoop.last_instance.resume_restore_policy
            self.assertTrue(bool(policy.restore_training_state))
            self.assertTrue(bool(policy.restore_model))
            self.assertTrue(bool(policy.restore_optimizers))
            self.assertTrue(bool(policy.restore_buffer))
            self.assertEqual(str(policy.model_restore_mode), "strict")
            self.assertEqual(str(policy.optimizer_restore_mode), "auto")
            self.assertTrue((Path(tmpdir) / "resume_latest.pt").exists())
            self.assertTrue((Path(tmpdir) / "trainer_state_final.pt").exists())
            self.assertTrue((Path(tmpdir) / "best.pt").exists())
            self.assertTrue((Path(tmpdir) / "trainer_state_best.pt").exists())
            self.assertGreaterEqual(len(agent.saved_checkpoints), 2)
            saved_best = next(
                item for item in agent.saved_checkpoints if item["path"].endswith("best.pt")
            )
            self.assertEqual(
                int(_effective_training_config_of(saved_best)["total_steps"]),
                2,
            )
            self.assertEqual(
                int(_effective_training_config_of(saved_best)["buffer_capacity"]),
                8,
            )
            self.assertIn("train", _agent_bootstrap_bundle_of(saved_best))
            metrics_path = Path(tmpdir) / "train_metrics.jsonl"
            self.assertTrue(metrics_path.exists())
            records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["step"], 2)
            self.assertEqual(records[0]["update_step"], 2)
            self.assertEqual(records[0]["global_step"], 2)
            self.assertEqual(records[0]["env_steps_collected"], 17)
            self.assertEqual(records[0]["batch_source"], "imag")
            self.assertAlmostEqual(float(records[0]["imag_ratio"]), 1.0, places=6)
            self.assertAlmostEqual(records[0]["metrics"]["critic/continue_prob"], 0.8, places=6)
            summary = _load_summary_payload(tmpdir)
            self.assertEqual(summary["artifacts"]["train_metrics_jsonl"], str(metrics_path))
            self.assertEqual(int(_effective_training_config_of(summary)["total_steps"]), 2)
            self.assertEqual(int(_effective_training_config_of(summary)["buffer_capacity"]), 8)
            self.assertIn("train", _agent_bootstrap_bundle_of(summary))
            self.assertGreaterEqual(len(eval_calls), 2)
            self.assertIs(eval_calls[0][1], agent)
            self.assertIs(eval_calls[-1][1], best_agent)
            self.assertGreaterEqual(len(_FakeTrainingLoop.last_instance.external_eval_feedback_calls), 1)
            feedback = _FakeTrainingLoop.last_instance.external_eval_feedback_calls[0]
            self.assertAlmostEqual(
                float(feedback["telemetry"]["real_behavior_action_entropy_mean"]),
                0.4,
                places=6,
            )
            eval_history_path = Path(tmpdir) / "eval_history.jsonl"
            eval_records = [
                json.loads(line)
                for line in eval_history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertAlmostEqual(
                float(eval_records[0]["real_corridor_persistence_rate"]),
                0.6,
                places=6,
            )
            self.assertEqual(eval_records[0]["update_step"], 2)
            self.assertEqual(eval_records[0]["global_step"], 2)
            self.assertAlmostEqual(float(eval_records[0]["mean_reward"]), 123.0, places=6)
            self.assertAlmostEqual(float(eval_records[0]["eval_reward"]), 123.0, places=6)
            self.assertAlmostEqual(float(eval_records[0]["current_checkpoint_reward"]), 123.0, places=6)

    def test_run_train_maps_env_step_budget_to_full_update_cycles(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        def fake_create_replay_buffer(capacity, store_obs=False):
            del store_obs
            return _DummyBuffer(capacity=capacity)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="Dummy-v0",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=3500,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=4,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.0,
                warmup_ratio=0.0,
                imagination_only=False,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides=None,
                argv=["scripts/cartpole_train.py", "--steps", "3500"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=fake_create_replay_buffer):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertEqual(int(cfg.total_steps), 440)
        self.assertEqual(int(cfg.num_train_steps), 440)
        self.assertEqual(int(cfg.total_env_steps), 3500)
        self.assertEqual(int(_FakeTrainingLoop.last_instance.global_step), 440)

    def test_run_train_derives_expected_env_steps_from_update_budget_and_cycle_ratio(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        def fake_create_replay_buffer(capacity, store_obs=False):
            del store_obs
            return _DummyBuffer(capacity=capacity)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="Dummy-v0",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=999,
                update_steps=10,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=4,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.0,
                warmup_ratio=0.0,
                imagination_only=False,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides=None,
                argv=["scripts/cartpole_train.py", "--update-steps", "10"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=fake_create_replay_buffer):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertEqual(int(cfg.total_steps), 10)
        self.assertEqual(int(cfg.num_train_steps), 10)
        self.assertEqual(int(cfg.total_env_steps), 96)
        self.assertEqual(int(_FakeTrainingLoop.last_instance.global_step), 10)

    def test_run_train_rejects_resume_metadata_preload_failures(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        def fake_create_replay_buffer(capacity, store_obs=False):
            del store_obs
            return _DummyBuffer(capacity=capacity)

        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = str(Path(tmpdir) / "resume_in.pt")
            Path(resume_path).write_text("resume", encoding="utf-8")
            args = SimpleNamespace(
                env="Dummy-v0",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=resume_path,
                steps=64,
                update_steps=2,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.0,
                warmup_ratio=0.0,
                imagination_only=False,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=1,
                save_interval=1,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=True,
                overrides=None,
                argv=["scripts/cartpole_train.py", "--resume-from", resume_path],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=fake_create_replay_buffer), \
                 mock.patch.object(
                     train_mod.TrainingStateManager,
                     "load",
                     side_effect=RuntimeError("resume metadata boom"),
                 ):
                with self.assertRaisesRegex(RuntimeError, "resume metadata"):
                    api.run_train(
                        args,
                        input_spec={"env": train_env, "eval_env": eval_env},
                    )

    def test_run_train_console_log_disambiguates_update_and_env_steps(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        def fake_create_replay_buffer(capacity, store_obs=False):
            del store_obs
            return _DummyBuffer(capacity=capacity)

        time_counter = {"value": 0.0}

        def fake_time():
            time_counter["value"] += 1.0
            return time_counter["value"]

        args = SimpleNamespace(
            env="Dummy-v0",
            device="cpu",
            seed=7,
            render=False,
            verbose=False,
            load=None,
            save=None,
            resume_from=None,
            steps=64,
            update_steps=2,
            collect_steps_per_cycle=32,
            train_steps_per_cycle=1,
            wm_seq_len=1,
            wm_batch_size=1,
            imagination_horizon=3,
            imagination_batch_size=1,
            rl_batch_size=1,
            buffer_capacity=8,
            pretrain_ratio=0.05,
            warmup_ratio=0.0,
            imagination_only=True,
            imag_ratio_start=0.0,
            imag_ratio_end=0.0,
            imag_ratio_max=0.0,
            imag_ratio_ramp_steps=1,
            imag_gradient="dynamics",
            imag_gradient_mix=0.0,
            actor_analytic_weight=1.0,
            actor_reinforce_aux_weight_discrete=0.1,
            use_reward_ema=True,
            log_interval=1000,
            eval_interval=0,
            save_interval=0,
            eval_episodes=2,
            eval_max_steps=5,
            eval_only=False,
            enable_eval=False,
            overrides=None,
            argv=["scripts/cartpole_train.py", "--update-steps", "2"],
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=fake_create_replay_buffer), \
                 mock.patch.object(api.time, "time", side_effect=fake_time):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        output = stdout.getvalue()
        self.assertIn("STEP upd=", output)
        self.assertIn("env=", output)
        self.assertIn("upd=     2", output)
        self.assertIn("env=    17", output)

    def test_run_train_resume_restores_best_eval_and_truncates_history(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        def fake_create_replay_buffer(capacity: int, store_obs: bool = False):
            del store_obs
            return _DummyBuffer(capacity)

        def fake_evaluate_agent(env_obj, eval_agent, episodes, max_steps=1000):
            del env_obj, episodes, max_steps
            mean = 123.0 if eval_agent is agent else 111.0
            return {
                "mean": mean,
                "std": 0.0,
                "min": mean,
                "max": mean,
                "mean_length": 2.0,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = str(Path(tmpdir) / "resume_in.pt")
            resume_state = train_mod.TrainingStateManager.create(
                config=TrainingConfig(),
                device=torch.device("cpu"),
            )
            resume_state.global_step = 20
            resume_state.best_eval_return = 456.0
            resume_state.best_step = 12
            train_mod.TrainingStateManager.save(resume_state, resume_path)

            train_metrics_path = Path(tmpdir) / "train_metrics.jsonl"
            train_metrics_path.write_text(
                "\n".join([
                    json.dumps({"step": 10, "metrics": {"keep": 1}}),
                    json.dumps({"step": 30, "metrics": {"drop": 1}}),
                ]) + "\n",
                encoding="utf-8",
            )
            eval_history_path = Path(tmpdir) / "eval_history.jsonl"
            eval_history_path.write_text(
                "\n".join([
                    json.dumps({"step": 10, "mean": 400.0, "std": 0.0, "min": 400.0, "max": 400.0, "mean_length": 10.0}),
                    json.dumps({"step": 30, "mean": 999.0, "std": 0.0, "min": 999.0, "max": 999.0, "mean_length": 10.0}),
                ]) + "\n",
                encoding="utf-8",
            )

            args = SimpleNamespace(
                env="Dummy-v0",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=resume_path,
                steps=64,
                update_steps=25,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.0,
                warmup_ratio=0.0,
                imagination_only=False,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=1,
                save_interval=1,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=True,
                overrides=None,
                argv=["scripts/cartpole_train.py", "--steps", "64", "--resume-from", resume_path],
            )

            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(api, "_evaluate_agent", side_effect=fake_evaluate_agent), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=fake_create_replay_buffer):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

            self.assertEqual(result["status"], 0)
            summary = _load_summary_payload(tmpdir)
            self.assertAlmostEqual(summary["eval"]["best_eval_mean_during_train"], 456.0, places=6)
            self.assertEqual(summary["eval"]["best_eval_step_during_train"], 12)

            metric_records = [json.loads(line) for line in train_metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([rec["step"] for rec in metric_records], [10, 25])
            self.assertEqual([rec["update_step"] for rec in metric_records], [10, 25])
            self.assertEqual([rec["global_step"] for rec in metric_records], [10, 25])
            eval_records = [json.loads(line) for line in eval_history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([rec["step"] for rec in eval_records], [10, 25])
            self.assertEqual([rec["update_step"] for rec in eval_records], [10, 25])
            self.assertEqual([rec["global_step"] for rec in eval_records], [10, 25])


    def test_run_train_caps_cartpole_imag_warmup_for_long_runs(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=100000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.1,
                warmup_ratio=0.1,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides=None,
                argv=["scripts/cartpole_train.py", "--steps", "100000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertEqual(cfg.wm_pretrain_steps, 32)
        self.assertEqual(cfg.warmup_steps, 32)
        self.assertAlmostEqual(float(cfg.imag_continue_prob_cap), 0.95, places=6)
        self.assertAlmostEqual(float(cfg.lambda_value_real_anchor), 0.0, places=6)
        self.assertTrue(bool(cfg.rl.detach_critic_features_on_imagination))
        self.assertTrue(bool(cfg.rl.use_actor_drift_guard))
        self.assertAlmostEqual(float(cfg.rl.slow_value_reg_drift_gain), 2.0, places=6)

    def test_run_train_pure_imag_cartpole_defaults_enable_consistency_modules(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)
        create_calls = []

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, device, seed
            create_calls.append(dict(config_overrides or {}))
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=15,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides=None,
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertEqual(int(cfg.imagination_horizon), 15)
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_policy_open_loop_consistency_weight),
            0.15,
            places=6,
        )
        self.assertTrue(bool(cfg.adaptive_imag_actor_use_target_value_ruler_enabled))
        self.assertTrue(bool(cfg.adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled))
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_actor_use_target_value_ruler_gap_margin),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_advantage_blend_max),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_clean_target_blend_max),
            0.8,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_clean_target_inflation_floor_max),
            0.2,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_inflation_threshold),
            5.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_inflation_tau),
            3.0,
            places=6,
        )
        self.assertEqual(len(create_calls), 1)
        overrides = create_calls[0]
        self.assertEqual(overrides["rssm_msc"]["enabled"], True)
        self.assertEqual(tuple(overrides["rssm_msc"]["horizons"]), (1, 4, 8))
        self.assertAlmostEqual(float(overrides["rssm_msc"]["loss_scale"]), 0.25, places=6)
        self.assertEqual(overrides["rssm_shortcut_consistency"]["enabled"], True)
        self.assertEqual(tuple(overrides["rssm_shortcut_consistency"]["horizons"]), (2, 4))
        self.assertAlmostEqual(float(overrides["rssm_shortcut_consistency"]["loss_scale"]), 0.25, places=6)
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_policy_open_loop_consistency_weight"]),
            0.15,
            places=6,
        )
        self.assertEqual(
            int(overrides["adaptive_imag_policy_open_loop_consistency_horizon"]),
            3,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_policy_open_loop_consistency_delta"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_policy_open_loop_consistency_high_value_boost"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_policy_open_loop_consistency_high_value_quantile"]),
            0.75,
            places=6,
        )
        self.assertTrue(bool(overrides["adaptive_imag_actor_use_target_value_ruler_enabled"]))
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_actor_use_target_value_ruler_blend"]),
            1.0,
            places=6,
        )
        self.assertTrue(
            bool(overrides["adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled"])
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_actor_use_target_value_ruler_gap_margin"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(
                overrides[
                    "adaptive_imag_actor_use_target_value_ruler_critic_distill_weight"
                ]
            ),
            0.15,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_idle_corridor_advantage_blend_max"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_idle_corridor_negative_adv_threshold"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_idle_corridor_negative_adv_tau"]),
            2.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_idle_corridor_quantile"]),
            0.75,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_idle_corridor_adv_term_clamp_min"]),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_idle_corridor_clean_target_blend_max"]),
            0.8,
            places=6,
        )
        self.assertAlmostEqual(
            float(
                overrides[
                    "adaptive_imag_idle_corridor_clean_target_inflation_floor_max"
                ]
            ),
            0.2,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_idle_corridor_inflation_threshold"]),
            5.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_idle_corridor_inflation_tau"]),
            3.0,
            places=6,
        )

    def test_explicit_cartpole_consistency_overrides_win_over_defaults(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)
        create_calls = []

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, device, seed
            create_calls.append(dict(config_overrides or {}))
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=15,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"rssm_msc":{"enabled":false},"rssm_shortcut_consistency":{"enabled":false},"adaptive_imag_policy_open_loop_consistency_weight":0.05,"adaptive_imag_policy_open_loop_consistency_horizon":5,"adaptive_imag_policy_open_loop_consistency_delta":0.75,"adaptive_imag_policy_open_loop_consistency_high_value_boost":2.0,"adaptive_imag_policy_open_loop_consistency_high_value_quantile":0.6,"adaptive_imag_actor_use_target_value_ruler_enabled":false,"adaptive_imag_actor_use_target_value_ruler_blend":0.4,"adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled":false,"adaptive_imag_actor_use_target_value_ruler_gap_margin":2.0,"adaptive_imag_actor_use_target_value_ruler_critic_distill_weight":0.0}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertEqual(int(cfg.imagination_horizon), 15)
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_policy_open_loop_consistency_weight),
            0.05,
            places=6,
        )
        self.assertFalse(bool(cfg.adaptive_imag_actor_use_target_value_ruler_enabled))
        self.assertFalse(bool(cfg.adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled))
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_actor_use_target_value_ruler_gap_margin),
            2.0,
            places=6,
        )
        self.assertEqual(len(create_calls), 1)
        overrides = create_calls[0]
        self.assertEqual(overrides["rssm_msc"]["enabled"], False)
        self.assertEqual(overrides["rssm_shortcut_consistency"]["enabled"], False)
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_policy_open_loop_consistency_weight"]),
            0.05,
            places=6,
        )
        self.assertEqual(
            int(overrides["adaptive_imag_policy_open_loop_consistency_horizon"]),
            5,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_policy_open_loop_consistency_delta"]),
            0.75,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_policy_open_loop_consistency_high_value_boost"]),
            2.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_policy_open_loop_consistency_high_value_quantile"]),
            0.6,
            places=6,
        )
        self.assertFalse(bool(overrides["adaptive_imag_actor_use_target_value_ruler_enabled"]))
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_actor_use_target_value_ruler_blend"]),
            0.4,
            places=6,
        )
        self.assertFalse(
            bool(overrides["adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled"])
        )
        self.assertAlmostEqual(
            float(overrides["adaptive_imag_actor_use_target_value_ruler_gap_margin"]),
            2.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(
                overrides[
                    "adaptive_imag_actor_use_target_value_ruler_critic_distill_weight"
                ]
            ),
            0.0,
            places=6,
        )

    def test_explicit_wm_pretrain_and_warmup_overrides_win_over_cartpole_cap(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"wm_pretrain_steps":128,"warmup_steps":64}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertEqual(cfg.wm_pretrain_steps, 128)
        self.assertEqual(cfg.warmup_steps, 64)

    def test_explicit_value_real_anchor_override_wins_over_cartpole_default(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"lambda_value_real_anchor":0.02}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.lambda_value_real_anchor), 0.02, places=6)

    def test_explicit_mc_real_anchor_overrides_propagate_to_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"value_real_anchor_use_mc_returns":true,"value_real_anchor_corridor_quantile":0.8}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertTrue(bool(cfg.value_real_anchor_use_mc_returns))
        self.assertAlmostEqual(
            float(cfg.value_real_anchor_corridor_quantile), 0.8, places=6
        )

    def test_explicit_policy_open_loop_corridor_maintenance_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_policy_open_loop_consistency_value_scale":0.35,"adaptive_imag_policy_open_loop_consistency_late_step_boost":1.25}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_policy_open_loop_consistency_value_scale),
            0.35,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_policy_open_loop_consistency_late_step_boost),
            1.25,
            places=6,
        )

    def test_explicit_imag_continue_cap_override_wins_over_cartpole_default(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"imag_continue_prob_cap":0.9}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.imag_continue_prob_cap), 0.9, places=6)

    def test_explicit_rl_stability_overrides_win_over_cartpole_defaults(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"rl":{"detach_critic_features_on_imagination":false,"use_actor_drift_guard":false,"slow_value_reg_drift_gain":0.5}}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertFalse(bool(cfg.rl.detach_critic_features_on_imagination))
        self.assertFalse(bool(cfg.rl.use_actor_drift_guard))
        self.assertAlmostEqual(float(cfg.rl.slow_value_reg_drift_gain), 0.5, places=6)


    def test_run_train_rejects_legacy_post_entry_highwater_commit_overrides(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_post_entry_commit_highwater_gap_max":6.0,"adaptive_imag_post_entry_commit_signed_adv_threshold":0.25,"adaptive_imag_post_entry_commit_highwater_hold_steps":100,"adaptive_imag_post_entry_commit_confirmation_steps":64}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                with self.assertRaisesRegex(
                    ValueError,
                    "Legacy phase overrides are no longer supported",
                ):
                    api.run_train(
                        args,
                        input_spec={"env": train_env, "eval_env": eval_env},
                    )

    def test_explicit_eval_confirmation_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_eval_confirmation_count":2,"adaptive_imag_compensation_persistence_eval_confirmation_count":2,"adaptive_imag_compensation_post_solved_eval_confirmation_count":2,"adaptive_imag_compensation_trigger_confirmation_steps":64,"adaptive_imag_compensation_trigger_quality_gate":true,"adaptive_imag_critic_bootstrap_contract_enabled":true,"adaptive_imag_critic_bootstrap_clean_mix_max":0.7,"adaptive_imag_critic_bootstrap_semantic_debt_decay":0.85,"adaptive_imag_critic_bootstrap_anchor_confidence_floor":0.15,"adaptive_imag_critic_bootstrap_min_step":1200,"adaptive_imag_critic_bootstrap_step_ramp":300,"adaptive_imag_critic_bootstrap_eval_threshold":230.0,"adaptive_imag_critic_bootstrap_eval_ramp":40.0}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

            config_payload = _load_resolved_config_payload(tmpdir)

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertEqual(int(cfg.adaptive_imag_eval_confirmation_count), 2)
        self.assertEqual(int(cfg.adaptive_imag_compensation_persistence_eval_confirmation_count), 2)
        self.assertEqual(int(cfg.adaptive_imag_compensation_post_solved_eval_confirmation_count), 2)
        self.assertEqual(int(cfg.adaptive_imag_compensation_trigger_confirmation_steps), 64)
        self.assertEqual(str(cfg.config_mode), "strict")
        self.assertEqual(str(cfg.validation_mode), "strict")
        self.assertEqual(int(cfg.total_steps), 625)
        self.assertEqual(int(cfg.num_train_steps), 625)
        self.assertEqual(int(cfg.total_env_steps), 20000)
        self.assertTrue(bool(cfg.adaptive_imag_compensation_trigger_quality_gate))
        self.assertTrue(bool(cfg.adaptive_imag_critic_bootstrap_contract_enabled))
        self.assertAlmostEqual(float(cfg.adaptive_imag_critic_bootstrap_clean_mix_max), 0.7, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_critic_bootstrap_semantic_debt_decay), 0.85, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_critic_bootstrap_anchor_confidence_floor), 0.15, places=6)
        self.assertEqual(int(cfg.adaptive_imag_critic_bootstrap_min_step), 1200)
        self.assertEqual(int(cfg.adaptive_imag_critic_bootstrap_step_ramp), 300)
        self.assertAlmostEqual(float(cfg.adaptive_imag_critic_bootstrap_eval_threshold), 230.0, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_critic_bootstrap_eval_ramp), 40.0, places=6)
        config_effective = _effective_training_config_of(config_payload)
        self.assertEqual(int(config_effective["adaptive_imag_eval_confirmation_count"]), 2)
        self.assertEqual(int(config_effective["adaptive_imag_compensation_persistence_eval_confirmation_count"]), 2)
        self.assertEqual(int(config_effective["adaptive_imag_compensation_post_solved_eval_confirmation_count"]), 2)
        self.assertEqual(int(config_effective["adaptive_imag_compensation_trigger_confirmation_steps"]), 64)
        self.assertEqual(str(config_effective["config_mode"]), "strict")
        self.assertEqual(str(config_effective["validation_mode"]), "strict")
        self.assertEqual(int(config_effective["total_steps"]), 625)
        self.assertEqual(int(config_effective["num_train_steps"]), 625)
        self.assertEqual(int(config_effective["total_env_steps"]), 20000)
        self.assertTrue(bool(config_effective["adaptive_imag_compensation_trigger_quality_gate"]))
        self.assertTrue(bool(config_effective["adaptive_imag_critic_bootstrap_contract_enabled"]))
        self.assertAlmostEqual(float(config_effective["adaptive_imag_critic_bootstrap_clean_mix_max"]), 0.7, places=6)
        self.assertAlmostEqual(float(config_effective["adaptive_imag_critic_bootstrap_semantic_debt_decay"]), 0.85, places=6)
        self.assertAlmostEqual(float(config_effective["adaptive_imag_critic_bootstrap_anchor_confidence_floor"]), 0.15, places=6)
        self.assertEqual(int(config_effective["adaptive_imag_critic_bootstrap_min_step"]), 1200)
        self.assertEqual(int(config_effective["adaptive_imag_critic_bootstrap_step_ramp"]), 300)
        self.assertAlmostEqual(float(config_effective["adaptive_imag_critic_bootstrap_eval_threshold"]), 230.0, places=6)
        self.assertAlmostEqual(float(config_effective["adaptive_imag_critic_bootstrap_eval_ramp"]), 40.0, places=6)
        self.assertNotIn("train_config", config_payload)
        self.assertIn("train", _agent_bootstrap_bundle_of(config_payload))

    def test_anchor_authority_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_real_stability_use_pre_eval_registry_support":true,"adaptive_imag_task_cert_real_policy_anchor_gate_enabled":true,"adaptive_imag_task_cert_real_policy_anchor_gate_kl_scale":0.15}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

            config_payload = _load_resolved_config_payload(tmpdir)

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertTrue(bool(cfg.adaptive_imag_real_stability_use_pre_eval_registry_support))
        self.assertTrue(bool(cfg.adaptive_imag_task_cert_real_policy_anchor_gate_enabled))
        self.assertAlmostEqual(float(cfg.adaptive_imag_task_cert_real_policy_anchor_gate_kl_scale), 0.15, places=6)
        config_effective = _effective_training_config_of(config_payload)
        self.assertTrue(bool(config_effective["adaptive_imag_real_stability_use_pre_eval_registry_support"]))
        self.assertTrue(bool(config_effective["adaptive_imag_task_cert_real_policy_anchor_gate_enabled"]))
        self.assertAlmostEqual(
            float(config_effective["adaptive_imag_task_cert_real_policy_anchor_gate_kl_scale"]),
            0.15,
            places=6,
        )
        self.assertNotIn("train_config", config_payload)
        self.assertIn("train", _agent_bootstrap_bundle_of(config_payload))

    def test_run_train_rejects_mixed_legacy_compensation_phase_overrides(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_eval_confirmation_count":2,"adaptive_imag_post_entry_eval_confirmation_count":2,"adaptive_imag_late_trigger_rescue_enabled":true}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                with self.assertRaisesRegex(
                    ValueError,
                    "Legacy phase overrides are no longer supported",
                ):
                    api.run_train(
                        args,
                        input_spec={"env": train_env, "eval_env": eval_env},
                    )

    def test_explicit_post_solved_negative_adv_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_post_solved_negative_adv_threshold":0.25,"adaptive_imag_compensation_post_solved_negative_adv_actor_scale":0.35,"adaptive_imag_compensation_post_solved_negative_adv_critic_boost":1.5,"adaptive_imag_compensation_post_solved_negative_adv_latch_steps":120,"adaptive_imag_compensation_post_solved_negative_adv_latch_actor_scale":0.75,"adaptive_imag_compensation_post_solved_negative_adv_latch_critic_boost":0.5}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_negative_adv_threshold), 0.25, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_negative_adv_actor_scale), 0.35, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_negative_adv_critic_boost), 1.5, places=6)
        self.assertEqual(int(cfg.adaptive_imag_compensation_post_solved_negative_adv_latch_steps), 120)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_negative_adv_latch_actor_scale), 0.75, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_negative_adv_latch_critic_boost), 0.5, places=6)

    def test_explicit_post_solved_actor_anchor_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_post_solved_actor_anchor_eval_threshold":220.0,"adaptive_imag_compensation_post_solved_actor_anchor_pull":0.02,"adaptive_imag_compensation_post_solved_actor_anchor_hard_pull":0.05,"adaptive_imag_compensation_post_solved_actor_anchor_latched_pull":0.08,"adaptive_imag_compensation_post_solved_actor_anchor_kl":0.04,"adaptive_imag_compensation_post_solved_actor_anchor_latched_kl":0.09,"adaptive_imag_compensation_post_solved_actor_anchor_highwater_eval_threshold":450.0,"adaptive_imag_compensation_post_solved_actor_anchor_highwater_pull":0.07,"adaptive_imag_compensation_post_solved_actor_anchor_highwater_kl":0.11,"adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_actor_scale":0.55,"adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_critic_boost":0.5}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_anchor_eval_threshold), 220.0, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_anchor_pull), 0.02, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_anchor_hard_pull), 0.05, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_anchor_latched_pull), 0.08, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_anchor_kl), 0.04, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_anchor_latched_kl), 0.09, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_anchor_highwater_eval_threshold), 450.0, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_anchor_highwater_pull), 0.07, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_anchor_highwater_kl), 0.11, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_actor_scale), 0.55, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_critic_boost), 0.5, places=6)

    def test_explicit_post_solved_real_actor_anchor_override_reaches_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_post_solved_real_actor_anchor_kl":0.02}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_real_actor_anchor_kl), 0.02, places=6)

    def test_explicit_post_solved_real_advantage_correction_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_post_solved_real_advantage_threshold":0.0,"adaptive_imag_compensation_post_solved_real_advantage_actor_scale":0.2,"adaptive_imag_compensation_post_solved_real_advantage_critic_boost":1.0,"adaptive_imag_compensation_post_solved_real_value_anchor_scale":3.0}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_real_advantage_threshold), 0.0, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_real_advantage_actor_scale), 0.2, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_real_advantage_critic_boost), 1.0, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_real_value_anchor_scale), 3.0, places=6)

    def test_explicit_post_solved_real_advantage_latch_veto_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_post_solved_real_advantage_latch_veto":true,"adaptive_imag_compensation_post_solved_real_advantage_latch_veto_threshold":0.5}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertTrue(bool(cfg.adaptive_imag_compensation_post_solved_real_advantage_latch_veto))
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_real_advantage_latch_veto_threshold), 0.5, places=6)

    def test_explicit_post_entry_negative_online_adv_analytic_scale_override_reaches_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)
        default_cfg = TrainingConfig()

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_post_entry_negative_online_adv_threshold":0.0,"adaptive_imag_post_entry_negative_online_adv_eval_threshold":220.0,"adaptive_imag_post_entry_negative_online_adv_analytic_scale":0.25}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_post_entry_negative_online_adv_threshold),
            float(default_cfg.adaptive_imag_post_entry_negative_online_adv_threshold),
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_post_entry_negative_online_adv_eval_threshold),
            float(default_cfg.adaptive_imag_post_entry_negative_online_adv_eval_threshold),
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_post_entry_negative_online_adv_analytic_scale),
            float(default_cfg.adaptive_imag_post_entry_negative_online_adv_analytic_scale),
            places=6,
        )

    def test_explicit_post_entry_actor_base_return_cap_override_reaches_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)
        default_cfg = TrainingConfig()

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_post_entry_actor_base_return_cap_margin":0.0}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_post_entry_actor_base_return_cap_margin),
            float(default_cfg.adaptive_imag_post_entry_actor_base_return_cap_margin),
            places=6,
        )

    def test_legacy_bootstrap_takeover_floor_defaults_to_half_floor(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)
        default_cfg = TrainingConfig()

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_critic_bootstrap_contract_enabled":true}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(
                     train_mod,
                     "create_replay_buffer",
                     side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity),
                 ):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_critic_bootstrap_external_takeover_floor_ratio),
            float(default_cfg.adaptive_imag_critic_bootstrap_external_takeover_floor_ratio),
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_critic_bootstrap_external_takeover_floor_ratio),
            0.5,
            places=6,
        )

    def test_explicit_persistence_actor_base_return_cap_override_reaches_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_persistence_actor_base_return_cap_margin":0.0}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_persistence_actor_base_return_cap_margin), 0.0, places=6)

    def test_explicit_persistence_base_return_cap_correction_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_persistence_base_return_cap_critic_boost":1.25,"adaptive_imag_compensation_persistence_base_return_cap_value_real_anchor_scale":2.5}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_persistence_base_return_cap_critic_boost), 1.25, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_persistence_base_return_cap_value_real_anchor_scale), 2.5, places=6)

    def test_explicit_persistence_commit_gate_override_reaches_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                steps=2,
                save=tmpdir,
                load=None,
                resume_from=None,
                collect_steps_per_cycle=1,
                train_steps_per_cycle=1,
                batch_size=2,
                seq_len=2,
                wm_seq_len=2,
                wm_batch_size=2,
                num_train_steps=1,
                warmup_steps=0,
                wm_pretrain_steps=0,
                total_env_steps=2,
                buffer_capacity=16,
                learning_rate=1e-3,
                lr_actor=1e-3,
                lr_critic=1e-3,
                gamma=0.99,
                gae_lambda=0.95,
                entropy_coef=0.0,
                value_coef=0.5,
                kl_coef=0.0,
                free_nats=0.0,
                imagination_horizon=3,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_persistence_requires_post_entry_commit":false}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertFalse(bool(cfg.adaptive_imag_compensation_persistence_requires_post_entry_commit))

    def test_run_train_rejects_legacy_post_entry_highwater_eval_threshold_override(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                steps=2,
                save=tmpdir,
                load=None,
                resume_from=None,
                collect_steps_per_cycle=1,
                train_steps_per_cycle=1,
                batch_size=2,
                seq_len=2,
                wm_seq_len=2,
                wm_batch_size=2,
                num_train_steps=1,
                warmup_steps=0,
                wm_pretrain_steps=0,
                total_env_steps=2,
                buffer_capacity=16,
                learning_rate=1e-3,
                lr_actor=1e-3,
                lr_critic=1e-3,
                gamma=0.99,
                gae_lambda=0.95,
                entropy_coef=0.0,
                value_coef=0.5,
                kl_coef=0.0,
                free_nats=0.0,
                imagination_horizon=3,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_post_entry_commit_highwater_eval_threshold":150.0}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                with self.assertRaisesRegex(
                    ValueError,
                    "Legacy phase overrides are no longer supported",
                ):
                    api.run_train(
                        args,
                        input_spec={"env": train_env, "eval_env": eval_env},
                    )

    def test_explicit_post_solved_actor_base_return_cap_override_reaches_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_post_solved_actor_base_return_cap_margin":0.0}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_base_return_cap_margin), 0.0, places=6)

    def test_explicit_post_solved_min_step_override_reaches_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_post_solved_min_step":1200}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertEqual(int(cfg.adaptive_imag_compensation_post_solved_min_step), 1200)

    def test_explicit_post_solved_highwater_relax_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_post_solved_highwater_eval_threshold":430.0,"adaptive_imag_compensation_post_solved_highwater_cap":0.845,"adaptive_imag_compensation_post_solved_highwater_actor_scale":0.9}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_highwater_eval_threshold), 430.0, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_highwater_cap), 0.845, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_highwater_actor_scale), 0.9, places=6)

    def test_explicit_post_solved_target_base_blend_override_reaches_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_post_solved_actor_target_base_blend":1.0}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_actor_target_base_blend), 1.0, places=6)

    def test_explicit_semantic_alignment_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_global_target_base_blend":0.35,"adaptive_imag_target_value_consistency_weight":0.2,"adaptive_imag_target_value_consistency_horizon":4,"adaptive_imag_target_value_consistency_delta":0.75,"adaptive_imag_target_value_consistency_high_value_boost":1.5,"adaptive_imag_target_value_consistency_high_value_quantile":0.8,"adaptive_imag_target_value_consistency_high_value_feature_scale":0.25,"adaptive_imag_policy_open_loop_consistency_weight":0.15,"adaptive_imag_policy_open_loop_consistency_horizon":5,"adaptive_imag_policy_open_loop_consistency_delta":0.6,"adaptive_imag_policy_open_loop_consistency_high_value_boost":1.25,"adaptive_imag_policy_open_loop_consistency_high_value_quantile":0.7}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_global_target_base_blend), 0.35, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_target_value_consistency_weight), 0.2, places=6)
        self.assertEqual(int(cfg.adaptive_imag_target_value_consistency_horizon), 4)
        self.assertAlmostEqual(float(cfg.adaptive_imag_target_value_consistency_delta), 0.75, places=6)
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_target_value_consistency_high_value_boost),
            1.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_target_value_consistency_high_value_quantile),
            0.8,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_target_value_consistency_high_value_feature_scale),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(float(cfg.adaptive_imag_policy_open_loop_consistency_weight), 0.15, places=6)
        self.assertEqual(int(cfg.adaptive_imag_policy_open_loop_consistency_horizon), 5)
        self.assertAlmostEqual(float(cfg.adaptive_imag_policy_open_loop_consistency_delta), 0.6, places=6)
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_policy_open_loop_consistency_high_value_boost),
            1.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_policy_open_loop_consistency_high_value_quantile),
            0.7,
            places=6,
        )

    def test_explicit_actor_use_target_value_ruler_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_actor_use_target_value_ruler_enabled":true,"adaptive_imag_actor_use_target_value_ruler_blend":0.65,"adaptive_imag_actor_use_target_value_ruler_critic_distill_weight":0.4}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertTrue(bool(cfg.adaptive_imag_actor_use_target_value_ruler_enabled))
        self.assertAlmostEqual(float(cfg.adaptive_imag_actor_use_target_value_ruler_blend), 0.65, places=6)
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_actor_use_target_value_ruler_critic_distill_weight),
            0.4,
            places=6,
        )

    def test_explicit_actor_use_target_value_ruler_soft_gate_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_actor_use_target_value_ruler_enabled":true,"adaptive_imag_actor_use_target_value_ruler_blend":0.65,"adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled":true,"adaptive_imag_actor_use_target_value_ruler_gap_margin":0.75}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertTrue(bool(cfg.adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled))
        self.assertAlmostEqual(float(cfg.adaptive_imag_actor_use_target_value_ruler_gap_margin), 0.75, places=6)

    def test_explicit_idle_corridor_semantic_calibration_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_idle_corridor_advantage_blend_max":0.5,"adaptive_imag_idle_corridor_negative_adv_threshold":1.0,"adaptive_imag_idle_corridor_negative_adv_tau":2.0,"adaptive_imag_idle_corridor_quantile":0.8,"adaptive_imag_idle_corridor_adv_term_clamp_scale":0.4,"adaptive_imag_idle_corridor_adv_term_clamp_min":0.25,"adaptive_imag_idle_corridor_clean_target_blend_max":0.65,"adaptive_imag_idle_corridor_clean_target_inflation_floor_max":0.15,"adaptive_imag_idle_corridor_inflation_threshold":3.0,"adaptive_imag_idle_corridor_inflation_tau":1.5,"adaptive_imag_task_corridor_enabled":true,"adaptive_imag_task_corridor_high_quantile":0.7,"adaptive_imag_task_corridor_low_quantile":0.2,"adaptive_imag_task_corridor_gate_tau":0.3,"adaptive_imag_task_corridor_analytic_floor":0.4,"adaptive_imag_task_corridor_confidence_scale":0.6,"adaptive_imag_actor_contract_trust_floor":0.15}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_advantage_blend_max),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_negative_adv_threshold),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_negative_adv_tau),
            2.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_quantile),
            0.8,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_adv_term_clamp_scale),
            0.4,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_adv_term_clamp_min),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_clean_target_blend_max),
            0.65,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_clean_target_inflation_floor_max),
            0.15,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_inflation_threshold),
            3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_idle_corridor_inflation_tau),
            1.5,
            places=6,
        )
        self.assertTrue(bool(cfg.adaptive_imag_task_corridor_enabled))
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_task_corridor_high_quantile),
            0.7,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_task_corridor_low_quantile),
            0.2,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_task_corridor_gate_tau),
            0.3,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_task_corridor_analytic_floor),
            0.4,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_task_corridor_confidence_scale),
            0.6,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_actor_contract_trust_floor),
            0.15,
            places=6,
        )

    def test_cartpole_pureimag_defaults_enable_task_corridor_trust_contract(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides="{}",
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent), \
                 mock.patch.object(api, "create_agent_rollout_collector", return_value=object()), \
                 mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop), \
                 mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertTrue(bool(cfg.value_real_anchor_use_mc_returns))
        self.assertTrue(bool(cfg.adaptive_imag_task_corridor_enabled))
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_task_corridor_high_quantile),
            0.75,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_task_corridor_low_quantile),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_task_corridor_gate_tau),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_task_corridor_analytic_floor),
            0.35,
            places=6,
        )
        self.assertAlmostEqual(
            float(cfg.adaptive_imag_actor_contract_trust_floor),
            0.2,
            places=6,
        )

    def test_explicit_post_solved_critic_anchor_override_reaches_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_post_solved_critic_anchor_weight":0.05}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_critic_anchor_weight), 0.05, places=6)

    def test_explicit_post_solved_drift_damping_overrides_reach_training_config(self):
        agent = _DummyAgent()
        train_env = _CountingEnv(done_after=2)
        eval_env = _CountingEnv(done_after=2)

        def fake_create_agent(env, config_overrides=None, device=None, seed=None):
            del env, config_overrides, device, seed
            return agent

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                env="CartPole-v1",
                device="cpu",
                seed=7,
                render=False,
                verbose=False,
                load=None,
                save=tmpdir,
                resume_from=None,
                steps=20000,
                update_steps=None,
                collect_steps_per_cycle=32,
                train_steps_per_cycle=1,
                wm_seq_len=1,
                wm_batch_size=1,
                imagination_horizon=3,
                imagination_batch_size=1,
                rl_batch_size=1,
                buffer_capacity=8,
                pretrain_ratio=0.05,
                warmup_ratio=0.0,
                imagination_only=True,
                imag_ratio_start=0.0,
                imag_ratio_end=0.0,
                imag_ratio_max=0.0,
                imag_ratio_ramp_steps=1,
                imag_gradient="dynamics",
                imag_gradient_mix=0.0,
                actor_analytic_weight=1.0,
                actor_reinforce_aux_weight_discrete=0.1,
                use_reward_ema=True,
                log_interval=1000,
                eval_interval=0,
                save_interval=0,
                eval_episodes=2,
                eval_max_steps=5,
                eval_only=False,
                enable_eval=False,
                overrides='{"adaptive_imag_compensation_post_solved_drift_damping_eval_threshold":440.0,"adaptive_imag_compensation_post_solved_drift_damping_wm_scale":0.5,"adaptive_imag_compensation_post_solved_drift_damping_critic_scale":0.6}',
                argv=["scripts/cartpole_train.py", "--steps", "20000", "--imagination-only"],
            )
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent),                  mock.patch.object(api, "create_agent_rollout_collector", return_value=object()),                  mock.patch.object(train_mod, "TrainingLoop", _FakeTrainingLoop),                  mock.patch.object(train_mod, "create_replay_buffer", side_effect=lambda capacity, store_obs=False: _DummyBuffer(capacity)):
                result = api.run_train(
                    args,
                    input_spec={"env": train_env, "eval_env": eval_env},
                )

        self.assertEqual(result["status"], 0)
        cfg = _FakeTrainingLoop.last_instance.kwargs["config"]
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_drift_damping_eval_threshold), 440.0, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_drift_damping_wm_scale), 0.5, places=6)
        self.assertAlmostEqual(float(cfg.adaptive_imag_compensation_post_solved_drift_damping_critic_scale), 0.6, places=6)

    def test_training_state_manager_load_requires_explicit_compatible_model_restore(self):
        class _SaveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.core = nn.Linear(2, 2)
                self.extra = nn.Linear(2, 2)

        class _LoadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.core = nn.Linear(2, 2)

        device = torch.device("cpu")
        config = TrainingConfig()
        save_model = _SaveModel()
        load_model = _LoadModel()

        with torch.no_grad():
            save_model.core.weight.fill_(1.5)
            save_model.core.bias.fill_(0.25)

        state = train_mod.TrainingStateManager.create(config=config, device=device)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(Path(tmpdir) / "trainer_state.pt")
            train_mod.TrainingStateManager.save(state, ckpt_path, model=save_model)
            raw_checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            with self.assertRaisesRegex(RuntimeError, "model state is incompatible"):
                train_mod.TrainingStateManager.load(
                    ckpt_path,
                    config=config,
                    device=device,
                    model=load_model,
                )
            restored = train_mod.TrainingStateManager.load(
                ckpt_path,
                config=config,
                device=device,
                model=load_model,
                restore_policy=TrainingCheckpointRestorePolicy(
                    model_restore_mode="compatible"
                ),
            )

        self.assertEqual(restored.global_step, 0)
        self.assertNotIn("config", raw_checkpoint)
        self.assertEqual(_effective_training_config_of(raw_checkpoint), config.to_dict())
        self.assertTrue(torch.allclose(load_model.core.weight, save_model.core.weight))
        self.assertTrue(torch.allclose(load_model.core.bias, save_model.core.bias))

    def test_build_training_checkpoint_payload_uses_canonical_schema_keys(self):
        config = TrainingConfig()
        state = train_mod.TrainingState(config=config, device=torch.device("cpu"))
        model = nn.Linear(1, 1)

        payload = build_training_checkpoint_payload(
            state,
            model=model,
            effective_training_config={"total_steps": 12},
        )

        self.assertEqual(_training_state_payload_of(payload), state.state_dict())
        self.assertIn("model", payload)
        self.assertEqual(int(payload["global_step"]), 0)
        self.assertEqual(payload["config_hash"], config.compute_hash())
        self.assertEqual(int(_effective_training_config_of(payload)["total_steps"]), 12)

    def test_run_artifact_paths_cover_directory_and_file_save_modes(self):
        dir_paths = RunArtifactPaths.from_save_arg("/tmp/run-output")
        file_paths = RunArtifactPaths.from_save_arg("/tmp/final-custom.pt")

        self.assertEqual(dir_paths.save_dir, Path("/tmp/run-output"))
        self.assertEqual(dir_paths.final_ckpt_path, Path("/tmp/run-output/final.pt"))
        self.assertEqual(dir_paths.periodic_checkpoint_path(7), Path("/tmp/run-output/checkpoint_step7.pt"))
        self.assertEqual(file_paths.save_dir, Path("/tmp"))
        self.assertEqual(file_paths.final_ckpt_path, Path("/tmp/final-custom.pt"))
        self.assertEqual(file_paths.periodic_checkpoint_path(7), Path("/tmp/final-custom_step7.pt"))

    def test_build_train_metric_record_keeps_canonical_fields(self):
        record = build_train_metric_record(
            {
                "train/update_step": 2,
                "train/global_step": 3,
                "train/env_steps_collected": 17,
                "train/batch_source": "imag",
                "train/imag_ratio": 1.0,
                "train/runtime_compensation_phase": "post_solved",
                "train/compensation_restore_status": "degraded",
                "train/compensation_restore_degraded": 1.0,
                "train/compensation_restore_post_solved_anchor_status": "fallback_recovered",
                "train/compensation_restore_behavior_policy_anchor_status": "restored",
                "train/compensation_restore_real_stability_registry_status": "partial_restore",
                "train/compensation_restore_real_stability_registry_restored_entries": 2.0,
                "train/compensation_restore_real_stability_registry_skipped_entries": 1.0,
                "train/runtime_compensation_guard_disabled": 1.0,
                "train/runtime_compensation_guard_disabled_reason": "post_solved_guards_disabled",
                "critic/continue_prob": torch.tensor(0.8),
            },
            2,
            train_t0=10.0,
            now=12.5,
        )

        self.assertEqual(record["step"], 2)
        self.assertEqual(record["update_step"], 2)
        self.assertEqual(record["global_step"], 3)
        self.assertEqual(record["env_steps_collected"], 17)
        self.assertEqual(record["batch_source"], "imag")
        self.assertEqual(record["runtime_compensation_phase"], "post_solved")
        self.assertEqual(record["compensation_restore_status"], "degraded")
        self.assertAlmostEqual(float(record["compensation_restore_degraded"]), 1.0, places=6)
        self.assertEqual(
            record["compensation_restore_post_solved_anchor_status"],
            "fallback_recovered",
        )
        self.assertEqual(
            record["compensation_restore_behavior_policy_anchor_status"],
            "restored",
        )
        self.assertEqual(
            record["compensation_restore_real_stability_registry_status"],
            "partial_restore",
        )
        self.assertEqual(
            record["compensation_restore_real_stability_registry_restored_entries"],
            2,
        )
        self.assertEqual(
            record["compensation_restore_real_stability_registry_skipped_entries"],
            1,
        )
        self.assertAlmostEqual(float(record["imag_ratio"]), 1.0, places=6)
        self.assertAlmostEqual(float(record["elapsed_sec"]), 2.5, places=6)
        self.assertAlmostEqual(
            float(record["metrics"]["train/runtime_compensation_guard_disabled"]),
            1.0,
            places=6,
        )
        self.assertEqual(
            record["metrics"]["train/runtime_compensation_guard_disabled_reason"],
            "post_solved_guards_disabled",
        )
        self.assertAlmostEqual(float(record["metrics"]["critic/continue_prob"]), 0.8, places=6)

    def test_build_eval_record_keeps_canonical_fields(self):
        record = build_eval_record(
            results={
                "mean": 123.0,
                "std": 4.0,
                "min": 120.0,
                "max": 126.0,
                "mean_length": 5.0,
            },
            step=2,
            telemetry={"real_corridor_persistence_rate": 0.6},
            agent_step_count=17,
        )

        self.assertEqual(record["step"], 2)
        self.assertEqual(record["update_step"], 2)
        self.assertEqual(record["global_step"], 2)
        self.assertEqual(record["env_step_count"], 17)
        self.assertAlmostEqual(float(record["mean_reward"]), 123.0, places=6)
        self.assertAlmostEqual(float(record["eval_reward"]), 123.0, places=6)
        self.assertAlmostEqual(float(record["current_checkpoint_reward"]), 123.0, places=6)
        self.assertAlmostEqual(float(record["real_corridor_persistence_rate"]), 0.6, places=6)

    def test_run_artifact_payload_builders_keep_canonical_fields(self):
        profile = SimpleNamespace(vitals_dim=4, action_dim=2, action_embed_dim=8)
        resolved = build_resolved_config_payload(
            saved_at_utc="2026-03-22T00:00:00+00:00",
            project_root=Path("/tmp/project"),
            git_commit="abc123",
            argv=["scripts/cartpole_train.py", "--steps", "64"],
            args={"steps": 64},
            overrides_effective={"foo": "bar"},
            effective_training_config={"total_steps": 2},
            agent_bootstrap_bundle={"train": {"buffer_size": 8}},
            profile=profile,
        )

        paths = RunArtifactPaths.from_save_arg("/tmp/run-output")

        class _Loop:
            _bootstrap_external_eval_feedback_last_step = 2
            _bootstrap_external_eval_feedback_last_mean = 123.0
            _bootstrap_external_eval_feedback_best_mean = 124.0
            _bootstrap_external_eval_feedback_telemetry = {"x": 1.0}
            _adaptive_compensation_restore_report = {
                "status": "degraded",
                "issues": ["post_solved_anchor fallback"],
                "post_solved_anchor": {
                    "status": "fallback_recovered",
                    "issues": ["actor anchor missing"],
                },
                "behavior_policy_anchor": {
                    "status": "restored",
                    "issues": [],
                },
                "real_stability_registry": {
                    "status": "partial_restore",
                    "restored_entries": 2,
                    "skipped_entries": 1,
                    "issues": ["one registry entry skipped"],
                },
            }

        summary = build_summary_payload(
            saved_at_utc="2026-03-22T00:00:00+00:00",
            env_id="Dummy-v0",
            args=SimpleNamespace(steps=64),
            seed=7,
            device="cpu",
            collector_steps=32,
            train_steps_per_cycle=1,
            total_updates=2,
            expected_env_steps=64,
            effective_training_config={"total_steps": 2},
            agent_bootstrap_bundle={"train": {"buffer_size": 8}},
            paths=paths,
            final_current={"mean": 123.0},
            final_best=None,
            final_model_source="current",
            best_eval_mean=123.0,
            best_eval_step=2,
            eval_history=[{"step": 2}],
            stats={"total_steps": 2},
            latest_train_metric_record={
                "step": 2,
                "update_step": 2,
                "global_step": 2,
                "metrics": {"train/foo": 1.0, "bootstrap/x": 2.0},
            },
            bootstrap_contract_summary_keys=("bootstrap/x", "bootstrap/y"),
            loop=_Loop(),
        )

        self.assertEqual(resolved["project_root"], "/tmp/project")
        self.assertEqual(int(_effective_training_config_of(resolved)["total_steps"]), 2)
        self.assertEqual(int(resolved["env_expected_dims"]["vitals_dim"]), 4)
        self.assertEqual(summary["artifacts"]["final_ckpt"], "/tmp/run-output/final.pt")
        self.assertEqual(summary["artifacts"]["resume_latest"], "/tmp/run-output/resume_latest.pt")
        self.assertEqual(summary["bootstrap_contract_last"]["metrics"], {"bootstrap/x": 2.0})
        self.assertEqual(summary["eval"]["num_eval_records"], 1)
        self.assertAlmostEqual(float(summary["bootstrap_external_eval_feedback"]["best_mean"]), 124.0, places=6)
        self.assertEqual(summary["compensation_restore"]["status"], "degraded")
        self.assertTrue(bool(summary["compensation_restore"]["degraded"]))
        self.assertEqual(
            summary["compensation_restore"]["post_solved_anchor_status"],
            "fallback_recovered",
        )
        self.assertEqual(
            summary["compensation_restore"]["real_stability_registry_restored_entries"],
            2,
        )
        self.assertEqual(
            summary["compensation_restore"]["real_stability_registry_skipped_entries"],
            1,
        )

    def test_restore_training_checkpoint_payload_uses_canonical_schema_keys(self):
        config = TrainingConfig()
        state = train_mod.TrainingState(config=config, device=torch.device("cpu"))
        model = mock.Mock()
        opt_bundle = mock.Mock()
        buffer = mock.Mock()

        restored = restore_training_checkpoint_payload(
            {
                "training_state": state.state_dict() | {
                    "global_step": 5,
                    "best_eval_return": 1.5,
                    "best_step": 4,
                },
                "effective_training_config": config.to_dict(),
                "model": {"weight": 1},
                "optimizer_bundle": {"optimizer": 2},
                "replay_buffer": {"buffer": 3},
            },
            state=state,
            model=model,
            opt_bundle=opt_bundle,
            buffer=buffer,
            checkpoint_path="trainer_state.pt",
            current_effective_training_config=config.to_dict(),
        )

        self.assertIs(restored, state)
        self.assertEqual(state.global_step, 5)
        self.assertAlmostEqual(state.best_eval_return, 1.5, places=6)
        self.assertEqual(state.best_step, 4)
        model.load_state_dict.assert_called_once_with({"weight": 1})
        opt_bundle.load_state_dict.assert_called_once_with(
            {"optimizer": 2},
            restore_mode="strict",
        )
        buffer.load_state_dict.assert_called_once_with({"buffer": 3})

    def test_restore_training_checkpoint_payload_raises_on_model_mismatch_by_default(self):
        config = TrainingConfig()
        state = train_mod.TrainingState(config=config, device=torch.device("cpu"))
        model = mock.Mock()
        model.load_state_dict.side_effect = RuntimeError("shape mismatch")
        opt_bundle = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "model state is incompatible"):
            restore_training_checkpoint_payload(
                {
                    "training_state": state.state_dict(),
                    "effective_training_config": config.to_dict(),
                    "model": {"weight": 1},
                    "optimizer_bundle": {"optimizer": 2},
                },
                state=state,
                model=model,
                opt_bundle=opt_bundle,
                checkpoint_path="trainer_state.pt",
                current_effective_training_config=config.to_dict(),
            )

        model.load_state_dict.assert_called_once_with({"weight": 1})
        opt_bundle.load_state_dict.assert_not_called()

    def test_restore_training_checkpoint_payload_compatible_model_restore_mode_uses_non_strict_load(self):
        config = TrainingConfig()
        state = train_mod.TrainingState(config=config, device=torch.device("cpu"))
        model = mock.Mock()

        restore_training_checkpoint_payload(
            {
                "training_state": state.state_dict(),
                "effective_training_config": config.to_dict(),
                "model": {"weight": 1},
            },
            state=state,
            model=model,
            checkpoint_path="trainer_state.pt",
            restore_policy=TrainingCheckpointRestorePolicy(model_restore_mode="compatible"),
            current_effective_training_config=config.to_dict(),
        )

        model.load_state_dict.assert_called_once_with({"weight": 1}, strict=False)

    def test_restore_training_checkpoint_payload_raises_when_component_bootstrap_fails(self):
        config = TrainingConfig()
        state = train_mod.TrainingState(config=config, device=torch.device("cpu"))
        model = mock.Mock()
        model.world_model = SimpleNamespace(
            _ensure_v45_components=mock.Mock(side_effect=RuntimeError("ensure boom"))
        )

        with self.assertRaisesRegex(RuntimeError, "ensure_v45_components failed"):
            restore_training_checkpoint_payload(
                {
                    "training_state": state.state_dict(),
                    "effective_training_config": config.to_dict(),
                    "model": {"weight": 1},
                },
                state=state,
                model=model,
                checkpoint_path="trainer_state.pt",
                current_effective_training_config=config.to_dict(),
            )

    def test_restore_training_checkpoint_payload_auto_downgrades_optimizer_restore_on_config_drift(self):
        config = TrainingConfig(
            total_steps=12,
            num_train_steps=12,
            total_env_steps=12,
            wm_pretrain_steps=0,
            warmup_steps=0,
        )
        state = train_mod.TrainingState(config=config, device=torch.device("cpu"))
        opt_bundle = mock.Mock()
        buffer = mock.Mock()

        with mock.patch("aletheia._training_checkpoint_schema.logger.warning") as warning_mock:
            restore_training_checkpoint_payload(
                {
                    "training_state": state.state_dict(),
                    "effective_training_config": {"total_steps": 12},
                    "optimizer_bundle": {"optimizer": 2},
                    "replay_buffer": {"buffer": 3},
                },
                state=state,
                opt_bundle=opt_bundle,
                buffer=buffer,
                checkpoint_path="trainer_state.pt",
                current_effective_training_config={"total_steps": 24},
            )

        opt_bundle.load_state_dict.assert_called_once_with(
            {"optimizer": 2},
            restore_mode="compatible",
        )
        buffer.load_state_dict.assert_called_once_with({"buffer": 3})
        warning_text = " ".join(
            " ".join(str(arg) for arg in call.args) for call in warning_mock.call_args_list
        )
        self.assertIn("effective training config drift detected", warning_text)

    def test_restore_training_checkpoint_payload_can_skip_optimizer_and_buffer_layers(self):
        config = TrainingConfig()
        state = train_mod.TrainingState(config=config, device=torch.device("cpu"))
        opt_bundle = mock.Mock()
        buffer = mock.Mock()
        policy = TrainingCheckpointRestorePolicy(
            restore_optimizers=False,
            restore_buffer=False,
        )

        restore_training_checkpoint_payload(
            {
                "training_state": state.state_dict(),
                "optimizer_bundle": {"optimizer": 2},
                "replay_buffer": {"buffer": 3},
            },
            state=state,
            opt_bundle=opt_bundle,
            buffer=buffer,
            checkpoint_path="trainer_state.pt",
            restore_policy=policy,
        )

        opt_bundle.load_state_dict.assert_not_called()
        buffer.load_state_dict.assert_not_called()

    def test_optimizer_bundle_compatible_restore_skips_incompatible_optimizer_groups(self):
        source_param = nn.Parameter(torch.tensor([1.0]))
        target_param_a = nn.Parameter(torch.tensor([1.0]))
        target_param_b = nn.Parameter(torch.tensor([2.0]))
        source = train_mod.OptimizerBundle(
            rl_optimizer=torch.optim.Adam([source_param], lr=1e-3),
        )
        target = train_mod.OptimizerBundle(
            rl_optimizer=torch.optim.Adam([target_param_a, target_param_b], lr=1e-3),
        )

        with self.assertRaises(ValueError):
            target.load_state_dict(source.state_dict(), restore_mode="strict")

        report = target.load_state_dict(
            source.state_dict(),
            restore_mode="compatible",
        )

        self.assertEqual(report["restored"], [])
        self.assertEqual(report["skipped"], ["rl"])
        self.assertEqual(len(report["issues"]), 1)
        self.assertEqual(report["issues"][0]["optimizer"], "rl")

    def test_training_state_manager_load_uses_unified_training_checkpoint_reader(self):
        config = TrainingConfig()
        device = torch.device("cpu")
        model = mock.Mock()
        payload = {
            "training_state": train_mod.TrainingState(config=config, device=device).state_dict(),
        }

        with mock.patch("aletheia.aletheia_train.os.path.exists", return_value=True), \
             mock.patch.object(train_mod, "read_training_checkpoint", return_value=payload) as read_mock:
            restored = train_mod.TrainingStateManager.load(
                "/tmp/trainer_state.pt",
                config=config,
                device=device,
                model=model,
                allow_unsafe_fallback=True,
                trusted_source=True,
            )

        read_mock.assert_called_once_with(
            "/tmp/trainer_state.pt",
            map_location=device,
            allow_unsafe_fallback=True,
            trusted_source=True,
        )
        self.assertIsInstance(restored, train_mod.TrainingState)

    def test_training_state_sync_from_loop_captures_runtime_fields(self):
        config = TrainingConfig()
        state = train_mod.TrainingState(config=config, device=torch.device("cpu"))

        class _Loop:
            global_step = 12
            episode_count = 3
            env_steps_collected = 99
            _steps_since_collect = 2
            _episode_return_ema = 1.25
            _episode_return_initialized = True
            _last_episode_count = 4
            _last_return_episode_count = 5
            _last_seen_episode_idx = 6

            @staticmethod
            def _export_adaptive_compensation_state():
                return {"post_transition": 0.75}

        state.sync_from_loop(_Loop(), best_eval_return=7.5, best_step=12)

        self.assertEqual(state.global_step, 12)
        self.assertEqual(state.episode_count, 3)
        self.assertEqual(state.total_samples, 99)
        self.assertEqual(state.steps_since_collect, 2)
        self.assertAlmostEqual(state.episode_return_ema, 1.25, places=6)
        self.assertTrue(state.episode_return_initialized)
        self.assertEqual(state.last_episode_count, 4)
        self.assertEqual(state.last_return_episode_count, 5)
        self.assertEqual(state.last_seen_episode_idx, 6)
        self.assertAlmostEqual(state.best_eval_return, 7.5, places=6)
        self.assertEqual(state.best_step, 12)
        self.assertEqual(state.adaptive_compensation_state, {"post_transition": 0.75})

    def test_agent_save_persists_agent_bootstrap_bundle_metadata(self):
        handle = api.AgentHandle.__new__(api.AgentHandle)
        handle.profile = SimpleNamespace(
            env_id="Dummy-v0",
            obs_shape=(4,),
            action_dim=2,
            is_discrete=True,
        )
        handle.config = ConfigBundle(
            train=AgentBootstrapTrainParams(buffer_size=321, batch_size=16),
            custom_overrides={"rssm_deter_dim": 96, "hard_fallback_enabled": True},
            router_overrides={"router_mode": "hard"},
            perceptor_overrides={"activation": "relu"},
            bootstrap_env_profile={"obs_shape": [4]},
        )
        handle.world_model = nn.Linear(1, 1)
        handle.actor = nn.Linear(1, 1)
        handle.critic = nn.Linear(1, 1)
        handle.router = None
        handle.will = None
        handle._step_count = 7
        effective_training_config = TrainingConfig(
            total_steps=12,
            num_train_steps=12,
            buffer_capacity=777,
            wm_pretrain_steps=0,
            warmup_steps=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "agent.pt"
            handle.save(str(ckpt_path), effective_training_config=effective_training_config)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.assertIsInstance(_agent_bootstrap_bundle_of(checkpoint), dict)
        self.assertNotIn("config_bundle", checkpoint)
        self.assertNotIn("profile", checkpoint)
        self.assertEqual(_agent_bootstrap_bundle_of(checkpoint)["train"]["buffer_size"], 321)
        self.assertEqual(_agent_bootstrap_bundle_of(checkpoint)["custom_overrides"]["rssm_deter_dim"], 96)
        self.assertEqual(int(_effective_training_config_of(checkpoint)["total_steps"]), 12)
        self.assertEqual(int(_effective_training_config_of(checkpoint)["buffer_capacity"]), 777)

    def test_build_agent_checkpoint_payload_uses_canonical_schema_keys(self):
        handle = api.AgentHandle.__new__(api.AgentHandle)
        handle.world_model = nn.Linear(1, 1)
        handle.actor = nn.Linear(1, 1)
        handle.critic = nn.Linear(1, 1)
        handle.router = None
        handle.will = None
        handle._step_count = 7

        payload = build_agent_checkpoint_payload(
            handle,
            version="test-version",
            bootstrap_bundle={"train": {"buffer_size": 321}},
            effective_training_config={"total_steps": 12},
        )

        self.assertEqual(payload["version"], "test-version")
        self.assertEqual(_agent_bootstrap_bundle_of(payload), {"train": {"buffer_size": 321}})
        self.assertIn("world_model", payload)
        self.assertIn("actor", payload)
        self.assertIn("critic", payload)
        self.assertNotIn("router", payload)
        self.assertNotIn("will", payload)
        self.assertEqual(int(payload["step_count"]), 7)
        self.assertEqual(int(_effective_training_config_of(payload)["total_steps"]), 12)

    def test_restore_agent_checkpoint_modules_uses_canonical_schema_keys(self):
        handle = api.AgentHandle.__new__(api.AgentHandle)
        handle.world_model = mock.Mock()
        handle.actor = mock.Mock()
        handle.critic = mock.Mock()
        handle.router = mock.Mock()
        handle.will = None

        step_count = restore_agent_checkpoint_modules(
            handle,
            {
                "world_model": {"wm": 1},
                "actor": {"actor": 2},
                "critic": {"critic": 3},
                "router": {"router": 4},
                "step_count": 7,
            },
            strict=False,
        )

        handle.world_model.load_state_dict.assert_called_once_with({"wm": 1}, strict=False)
        handle.actor.load_state_dict.assert_called_once_with({"actor": 2}, strict=False)
        handle.critic.load_state_dict.assert_called_once_with({"critic": 3}, strict=False)
        handle.router.load_state_dict.assert_called_once_with({"router": 4}, strict=False)
        self.assertEqual(step_count, 7)

    def test_restore_agent_checkpoint_modules_compatible_mode_skips_incompatible_required_component(self):
        handle = api.AgentHandle.__new__(api.AgentHandle)
        handle.world_model = mock.Mock()
        handle.actor = mock.Mock()
        handle.critic = mock.Mock()
        handle.router = None
        handle.will = None
        handle.world_model.load_state_dict.side_effect = RuntimeError("shape mismatch")

        with mock.patch("aletheia._agent_checkpoint_schema.logger.warning") as warning_mock:
            step_count = restore_agent_checkpoint_modules(
                handle,
                {
                    "world_model": {"wm": 1},
                    "actor": {"actor": 2},
                    "critic": {"critic": 3},
                    "step_count": 7,
                },
                strict=False,
                checkpoint_path="agent.pt",
            )

        handle.world_model.load_state_dict.assert_called_once_with({"wm": 1}, strict=False)
        handle.actor.load_state_dict.assert_called_once_with({"actor": 2}, strict=False)
        handle.critic.load_state_dict.assert_called_once_with({"critic": 3}, strict=False)
        warning_text = " ".join(
            " ".join(str(arg) for arg in call.args) for call in warning_mock.call_args_list
        )
        self.assertIn("world_model", warning_text)
        self.assertIn("shape mismatch", warning_text)
        self.assertIn("compatible", warning_text)
        self.assertEqual(step_count, 7)

    def test_agent_load_uses_unified_checkpoint_reader(self):
        handle = api.AgentHandle.__new__(api.AgentHandle)
        handle.device = torch.device("cpu")
        handle.world_model = mock.Mock()
        handle.actor = mock.Mock()
        handle.critic = mock.Mock()
        handle.router = mock.Mock()
        handle.will = mock.Mock()
        handle._step_count = 0

        checkpoint = {
            "world_model": {"wm": 1},
            "actor": {"actor": 2},
            "critic": {"critic": 3},
            "router": {"router": 4},
            "will": {"will": 5},
            "step_count": 7,
        }

        with mock.patch.object(api, "read_checkpoint", return_value=checkpoint) as read_mock:
            handle.load("/tmp/fake.pt", strict=False, allow_unsafe_fallback=True)

        read_mock.assert_called_once_with(
            "/tmp/fake.pt",
            map_location=handle.device,
            allow_unsafe_fallback=True,
        )
        handle.world_model.load_state_dict.assert_called_once_with({"wm": 1}, strict=False)
        handle.actor.load_state_dict.assert_called_once_with({"actor": 2}, strict=False)
        handle.critic.load_state_dict.assert_called_once_with({"critic": 3}, strict=False)
        handle.router.load_state_dict.assert_called_once_with({"router": 4}, strict=False)
        handle.will.load_state_dict.assert_called_once_with({"will": 5}, strict=False)
        self.assertEqual(handle._step_count, 7)

    def test_load_agent_uses_checkpoint_agent_bootstrap_bundle_when_present(self):
        env = _CountingEnv(done_after=2)
        create_calls = []

        class _LoadedAgent:
            def __init__(self):
                self.load_calls = []

            def load(self, path, strict=True, allow_unsafe_fallback=False):
                self.load_calls.append((path, strict, allow_unsafe_fallback))

        loaded_agent = _LoadedAgent()

        def fake_create_agent(env_obj, config_overrides=None, device=None, seed=None):
            create_calls.append((env_obj, config_overrides, device, seed))
            return loaded_agent

        checkpoint = _make_minimal_agent_checkpoint(
            agent_bootstrap_bundle={
                "train": {"buffer_size": 123, "batch_size": 8, "use_compile": True},
                "custom_overrides": {"rssm_deter_dim": 96},
                "router_overrides": {"router_mode": "hard"},
                "perceptor_overrides": {"activation": "relu"},
                "bootstrap_env_profile": {"obs_shape": [4]},
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            torch.save(checkpoint, ckpt_path)
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent):
                restored = api.load_agent(str(ckpt_path), env=env, device="cpu")

        self.assertIs(restored, loaded_agent)
        self.assertEqual(len(create_calls), 1)
        _, config_overrides, device, seed = create_calls[0]
        expected_train = AgentBootstrapTrainParams(
            buffer_size=123,
            batch_size=8,
            use_compile=True,
        )
        expected_bridge = ConfigBundle(
            train=AgentBootstrapTrainParams(),
            custom_overrides={"rssm_deter_dim": 96},
            router_overrides={"router_mode": "hard"},
            perceptor_overrides={"activation": "relu"},
            bootstrap_env_profile={"obs_shape": [4]},
        ).to_factory_overrides()
        self.assertEqual(device, "cpu")
        self.assertIsNone(seed)
        self.assertEqual(config_overrides["buffer_size"], 123)
        self.assertEqual(config_overrides["batch_size"], 8)
        self.assertTrue(config_overrides["use_compile"])
        self.assertEqual(config_overrides["rssm_deter_dim"], 96)
        self.assertEqual(config_overrides, {
            **asdict(expected_train),
            **expected_bridge,
        })
        self.assertEqual(loaded_agent.load_calls, [(str(ckpt_path), True, False)])

    def test_load_agent_rejects_legacy_config_bundle_payload(self):
        env = _CountingEnv(done_after=2)

        checkpoint = _make_minimal_agent_checkpoint(
            config_bundle={
                "train": {"buffer_size": 123, "batch_size": 8, "sequence_length": 11},
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            torch.save(checkpoint, ckpt_path)
            with self.assertRaisesRegex(
                ValueError,
                "Legacy checkpoint bootstrap payload 'config_bundle' is no longer supported",
            ):
                api.load_agent(str(ckpt_path), env=env, device="cpu")

    def test_load_agent_rejects_legacy_checkpoint_env_profile_override_field(self):
        env = _CountingEnv(done_after=2)

        checkpoint = _make_minimal_agent_checkpoint(
            agent_bootstrap_bundle={
                "train": {"buffer_size": 123, "batch_size": 8},
                "env_profile_override": {"obs_shape": [4]},
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            torch.save(checkpoint, ckpt_path)
            with self.assertRaisesRegex(
                ValueError,
                "Legacy checkpoint bootstrap payload field 'env_profile_override' is no longer supported",
            ):
                api.load_agent(str(ckpt_path), env=env, device="cpu")

    def test_load_agent_without_agent_bootstrap_bundle_uses_default_creation_overrides(self):
        env = _CountingEnv(done_after=2)
        create_calls = []

        class _LoadedAgent:
            def __init__(self):
                self.load_calls = []

            def load(self, path, strict=True, allow_unsafe_fallback=False):
                self.load_calls.append((path, strict, allow_unsafe_fallback))

        loaded_agent = _LoadedAgent()

        def fake_create_agent(env_obj, config_overrides=None, device=None, seed=None):
            create_calls.append((env_obj, config_overrides, device, seed))
            return loaded_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            torch.save({"world_model": {}, "actor": {}, "critic": {}}, ckpt_path)
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent):
                restored = api.load_agent(str(ckpt_path), env=env, device="cpu")

        self.assertIs(restored, loaded_agent)
        self.assertEqual(len(create_calls), 1)
        _, config_overrides, device, seed = create_calls[0]
        self.assertEqual(device, "cpu")
        self.assertIsNone(seed)
        self.assertIsNone(config_overrides)
        self.assertEqual(loaded_agent.load_calls, [(str(ckpt_path), True, False)])

    def test_load_agent_raises_before_create_when_checkpoint_metadata_read_fails(self):
        env = _CountingEnv(done_after=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "broken.pt"
            ckpt_path.write_text("not a checkpoint", encoding="utf-8")
            with mock.patch.object(api, "create_agent") as create_agent_mock:
                with self.assertRaises(Exception):
                    api.load_agent(str(ckpt_path), env=env, device="cpu")

        create_agent_mock.assert_not_called()

    def test_read_agent_creation_overrides_from_checkpoint_can_soft_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "broken.pt"
            ckpt_path.write_text("not a checkpoint", encoding="utf-8")

            overrides = read_agent_creation_overrides_from_checkpoint(
                ckpt_path,
                fail_soft=True,
            )

        self.assertIsNone(overrides)

    def test_load_agent_allows_explicit_non_strict_opt_out(self):
        env = _CountingEnv(done_after=2)
        create_calls = []

        class _LoadedAgent:
            def __init__(self):
                self.load_calls = []

            def load(self, path, strict=True, allow_unsafe_fallback=False):
                self.load_calls.append((path, strict, allow_unsafe_fallback))

        loaded_agent = _LoadedAgent()

        def fake_create_agent(env_obj, config_overrides=None, device=None, seed=None):
            create_calls.append((env_obj, config_overrides, device, seed))
            return loaded_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            torch.save({"world_model": {}, "actor": {}, "critic": {}}, ckpt_path)
            with mock.patch.object(api, "create_agent", side_effect=fake_create_agent):
                restored = api.load_agent(str(ckpt_path), env=env, device="cpu", strict=False)

        self.assertIs(restored, loaded_agent)
        self.assertEqual(len(create_calls), 1)
        self.assertEqual(loaded_agent.load_calls, [(str(ckpt_path), False, False)])

    def test_config_bundle_for_env_rejects_noncanonical_policy_aliases(self):
        env = _CountingEnv(done_after=2)

        with mock.patch(
            "aletheia.aletheia_foundation.make_config_policy_overrides_for_env",
            return_value={"router_overrides": {"router_mode": "hard"}},
        ):
            with self.assertRaisesRegex(ValueError, "Non-canonical ConfigPolicy"):
                ConfigBundle.for_env(env)

    def test_config_bundle_for_profile_uses_bootstrap_defaults(self):
        profile = api.extract_env_profile(_CountingEnv(done_after=2))
        bundle = ConfigBundle.for_profile(profile)

        train = bundle.train

        self.assertIs(bundle.profile, profile)
        self.assertEqual(int(train.buffer_size), 1000)
        self.assertEqual(int(train.batch_size), 32)
        self.assertFalse(bool(train.use_compile))
        self.assertNotIn("sequence_length", AGENT_BOOTSTRAP_TRAIN_FIELDS)
        self.assertFalse(hasattr(train, "sequence_length"))
        self.assertFalse(hasattr(train, "imagination_horizon"))

    def test_config_bundle_for_env_applies_config_policy_on_top_of_profile_defaults(self):
        env = _CountingEnv(done_after=2)
        expected_profile = api.extract_env_profile(env)

        bundle = ConfigBundle.for_env(env)

        self.assertIsNotNone(bundle.profile)
        self.assertEqual(bundle.profile.env_id, expected_profile.env_id)
        self.assertEqual(int(bundle.train.buffer_size), 1000)
        self.assertEqual(int(bundle.train.batch_size), 32)
        self.assertEqual(int(bundle.custom_overrides["rssm_deter_dim"]), 96)
        self.assertTrue(bool(bundle.custom_overrides["curiosity_enabled"]))
        self.assertFalse(bool(bundle.custom_overrides["mastery_enabled"]))
        self.assertIn("uncertainty_quantile", bundle.router_overrides)
        self.assertIn("vitals_dim", bundle.perceptor_overrides)
        self.assertIsNotNone(bundle.bootstrap_env_profile)

    def test_agent_handle_profile_uses_config_profile_as_single_source_of_truth(self):
        profile = api.extract_env_profile(_CountingEnv(done_after=2))
        handle = api.AgentHandle.__new__(api.AgentHandle)
        handle.config = ConfigBundle(
            train=AgentBootstrapTrainParams(),
            profile=profile,
        )
        handle._profile = None

        self.assertIs(handle.profile, profile)

        replacement = SimpleNamespace(
            env_id="Dummy-v1",
            obs_shape=(8,),
            action_dim=3,
            is_discrete=False,
        )
        handle.profile = replacement

        self.assertIs(handle.profile, replacement)
        self.assertIs(handle.config.profile, replacement)

    def test_config_bundle_to_factory_overrides_keeps_reserved_keys_at_boundary(self):
        bundle = ConfigBundle(
            train=AgentBootstrapTrainParams(buffer_size=321, batch_size=16),
            custom_overrides={"rssm_deter_dim": 96},
            router_overrides={"router_mode": "hard"},
            perceptor_overrides={"activation": "relu"},
            bootstrap_env_profile={"obs_shape": [4]},
        )

        payload = bundle.to_factory_overrides()

        self.assertEqual(payload["rssm_deter_dim"], 96)
        self.assertEqual(payload["_router_overrides"], {"router_mode": "hard"})
        self.assertEqual(payload["_perceptor_overrides"], {"activation": "relu"})
        self.assertEqual(payload["_env_profile"], {"obs_shape": [4]})
        self.assertNotIn("train", payload)

    def test_parse_factory_bridge_overrides_decodes_wire_boundary(self):
        parsed = parse_factory_bridge_overrides(
            {
                "rssm_deter_dim": 96,
                "_router_overrides": {"router_mode": "hard"},
                "_perceptor_overrides": {"activation": "relu"},
                "_env_profile": {"obs_shape": [4]},
            }
        )

        self.assertEqual(parsed.custom_overrides, {"rssm_deter_dim": 96})
        self.assertEqual(parsed.router_overrides, {"router_mode": "hard"})
        self.assertEqual(parsed.perceptor_overrides, {"activation": "relu"})
        self.assertEqual(parsed.bridge_env_profile, {"obs_shape": [4]})

    def test_export_factory_bridge_overrides_encodes_wire_boundary(self):
        payload = export_factory_bridge_overrides(
            FactoryBridgeOverrides(
                custom_overrides={"rssm_deter_dim": 96},
                router_overrides={"router_mode": "hard"},
                perceptor_overrides={"activation": "relu"},
                bridge_env_profile={"obs_shape": [4]},
            )
        )

        self.assertEqual(payload["rssm_deter_dim"], 96)
        self.assertEqual(payload["_router_overrides"], {"router_mode": "hard"})
        self.assertEqual(payload["_perceptor_overrides"], {"activation": "relu"})
        self.assertEqual(payload["_env_profile"], {"obs_shape": [4]})

    def test_config_bundle_from_factory_bridge_overrides_decodes_boundary_payload(self):
        bundle = ConfigBundle.from_factory_bridge_overrides(
            {
                "rssm_deter_dim": 96,
                "_router_overrides": {"router_mode": "hard"},
                "_perceptor_overrides": {"activation": "relu"},
                "_env_profile": {"obs_shape": [4]},
            }
        )

        self.assertEqual(int(bundle.train.buffer_size), 1000)
        self.assertEqual(int(bundle.train.batch_size), 32)
        self.assertEqual(bundle.custom_overrides, {"rssm_deter_dim": 96})
        self.assertEqual(bundle.router_overrides, {"router_mode": "hard"})
        self.assertEqual(bundle.perceptor_overrides, {"activation": "relu"})
        self.assertEqual(bundle.bootstrap_env_profile, {"obs_shape": [4]})

    def test_config_bundle_to_agent_bootstrap_bundle_keeps_bootstrap_sections_explicit(self):
        bundle = ConfigBundle(
            train=AgentBootstrapTrainParams(buffer_size=321, batch_size=16, use_compile=True),
            custom_overrides={"rssm_deter_dim": 96},
            router_overrides={"router_mode": "hard"},
            perceptor_overrides={"activation": "relu"},
            bootstrap_env_profile={"obs_shape": [4]},
        )

        payload = bundle.to_agent_bootstrap_bundle()

        self.assertEqual(payload["train"]["buffer_size"], 321)
        self.assertEqual(payload["train"]["batch_size"], 16)
        self.assertTrue(payload["train"]["use_compile"])
        self.assertEqual(payload["custom_overrides"]["rssm_deter_dim"], 96)
        self.assertEqual(payload["router_overrides"]["router_mode"], "hard")
        self.assertEqual(payload["perceptor_overrides"]["activation"], "relu")
        self.assertEqual(payload["bootstrap_env_profile"], {"obs_shape": [4]})

    def test_config_bundle_from_agent_bootstrap_bundle_decodes_checkpoint_payload(self):
        bundle = ConfigBundle.from_agent_bootstrap_bundle(
            {
                "train": {
                    "buffer_size": 321,
                    "batch_size": 16,
                    "use_compile": True,
                },
                "custom_overrides": {"rssm_deter_dim": 96},
                "router_overrides": {"router_mode": "hard"},
                "perceptor_overrides": {"activation": "relu"},
                "bootstrap_env_profile": {"obs_shape": [4]},
            }
        )

        self.assertEqual(int(bundle.train.buffer_size), 321)
        self.assertEqual(int(bundle.train.batch_size), 16)
        self.assertTrue(bundle.train.use_compile)
        self.assertEqual(bundle.custom_overrides, {"rssm_deter_dim": 96})
        self.assertEqual(bundle.router_overrides, {"router_mode": "hard"})
        self.assertEqual(bundle.perceptor_overrides, {"activation": "relu"})
        self.assertEqual(bundle.bootstrap_env_profile, {"obs_shape": [4]})

    def test_config_bundle_from_checkpoint_metadata_decodes_agent_bootstrap_bundle(self):
        bundle = ConfigBundle.from_checkpoint_metadata(
            {
                "agent_bootstrap_bundle": {
                    "train": {
                        "buffer_size": 321,
                        "batch_size": 16,
                        "use_compile": True,
                    },
                    "custom_overrides": {"rssm_deter_dim": 96},
                    "router_overrides": {"router_mode": "hard"},
                    "perceptor_overrides": {"activation": "relu"},
                    "bootstrap_env_profile": {"obs_shape": [4]},
                }
            }
        )

        assert bundle is not None
        self.assertEqual(int(bundle.train.buffer_size), 321)
        self.assertEqual(int(bundle.train.batch_size), 16)
        self.assertTrue(bundle.train.use_compile)
        self.assertEqual(bundle.custom_overrides, {"rssm_deter_dim": 96})
        self.assertEqual(bundle.router_overrides, {"router_mode": "hard"})
        self.assertEqual(bundle.perceptor_overrides, {"activation": "relu"})
        self.assertEqual(bundle.bootstrap_env_profile, {"obs_shape": [4]})

    def test_config_bundle_from_checkpoint_metadata_rejects_legacy_config_bundle(self):
        with self.assertRaisesRegex(
            ValueError,
            "Legacy checkpoint bootstrap payload 'config_bundle' is no longer supported",
        ):
            ConfigBundle.from_checkpoint_metadata(
                {
                    "config_bundle": {
                        "train": {"buffer_size": 123, "batch_size": 8},
                    }
                }
            )

    def test_config_bundle_agent_creation_overrides_from_checkpoint_metadata(self):
        payload = ConfigBundle.agent_creation_overrides_from_checkpoint_metadata(
            {
                "agent_bootstrap_bundle": {
                    "train": {
                        "buffer_size": 321,
                        "batch_size": 16,
                        "use_compile": True,
                    },
                    "custom_overrides": {"rssm_deter_dim": 96},
                    "router_overrides": {"router_mode": "hard"},
                    "perceptor_overrides": {"activation": "relu"},
                    "bootstrap_env_profile": {"obs_shape": [4]},
                }
            }
        )

        self.assertEqual(int(payload["buffer_size"]), 321)
        self.assertEqual(int(payload["batch_size"]), 16)
        self.assertTrue(payload["use_compile"])
        self.assertEqual(payload["rssm_deter_dim"], 96)
        self.assertEqual(payload["_router_overrides"], {"router_mode": "hard"})
        self.assertEqual(payload["_perceptor_overrides"], {"activation": "relu"})
        self.assertEqual(payload["_env_profile"], {"obs_shape": [4]})

    def test_config_bundle_to_agent_creation_overrides_merges_train_and_bridge_payloads(self):
        bundle = ConfigBundle(
            train=AgentBootstrapTrainParams(buffer_size=321, batch_size=16, use_compile=True),
            custom_overrides={"rssm_deter_dim": 96},
            router_overrides={"router_mode": "hard"},
            perceptor_overrides={"activation": "relu"},
            bootstrap_env_profile={"obs_shape": [4]},
        )

        payload = bundle.to_agent_creation_overrides()

        self.assertEqual(int(payload["buffer_size"]), 321)
        self.assertEqual(int(payload["batch_size"]), 16)
        self.assertTrue(payload["use_compile"])
        self.assertEqual(payload["rssm_deter_dim"], 96)
        self.assertEqual(payload["_router_overrides"], {"router_mode": "hard"})
        self.assertEqual(payload["_perceptor_overrides"], {"activation": "relu"})
        self.assertEqual(payload["_env_profile"], {"obs_shape": [4]})

    def test_train_params_bootstrap_field_audit_stays_partitioned(self):
        train_field_names = set(TrainParams.__dataclass_fields__)

        self.assertTrue(AGENT_BOOTSTRAP_TRAIN_FIELDS)
        self.assertTrue(TRAIN_PARAMS_BOOTSTRAP_METADATA_FIELDS)
        self.assertTrue(
            AGENT_BOOTSTRAP_TRAIN_FIELDS.isdisjoint(
                TRAIN_PARAMS_BOOTSTRAP_METADATA_FIELDS
            )
        )
        self.assertEqual(
            train_field_names,
            set(AGENT_BOOTSTRAP_TRAIN_FIELDS)
            | set(TRAIN_PARAMS_BOOTSTRAP_METADATA_FIELDS),
        )
        self.assertIn("buffer_size", AGENT_BOOTSTRAP_TRAIN_FIELDS)
        self.assertIn("batch_size", AGENT_BOOTSTRAP_TRAIN_FIELDS)
        self.assertNotIn("sequence_length", AGENT_BOOTSTRAP_TRAIN_FIELDS)
        self.assertIn("sequence_length", TRAIN_PARAMS_BOOTSTRAP_METADATA_FIELDS)
        self.assertIn("imagination_horizon", TRAIN_PARAMS_BOOTSTRAP_METADATA_FIELDS)

    def test_parse_config_policy_overrides_rejects_partial_reserved_payload(self):
        with self.assertRaisesRegex(ValueError, "Unexpected ConfigPolicy reserved overrides"):
            parse_config_policy_overrides(
                {
                    "_policy_batch_length": 8,
                    "_unknown_policy_key": 1,
                    "rssm_deter_dim": 64,
                }
            )

    def test_parse_config_policy_overrides_accepts_policy_will_flags(self):
        parsed = parse_config_policy_overrides(
            {
                "_policy_batch_length": 8,
                "_policy_horizon": 12,
                "_policy_curiosity_enabled": False,
                "_policy_mastery_enabled": True,
                "rssm_deter_dim": 64,
            }
        )

        self.assertEqual(int(parsed.sequence_length), 8)
        self.assertEqual(int(parsed.imagination_horizon), 12)
        self.assertFalse(bool(parsed.curiosity_enabled))
        self.assertTrue(bool(parsed.mastery_enabled))
        self.assertEqual(int(parsed.custom_overrides["rssm_deter_dim"]), 64)

    def test_coerce_config_policy_overrides_accepts_typed_boundary(self):
        typed = ConfigPolicyOverrides(
            sequence_length=8,
            imagination_horizon=12,
            custom_overrides={"rssm_deter_dim": 64},
        )

        parsed = coerce_config_policy_overrides(typed)

        self.assertIs(parsed, typed)

    def test_coerce_config_policy_overrides_decodes_wire_mapping(self):
        parsed = coerce_config_policy_overrides(
            {
                "_policy_batch_length": 8,
                "_policy_horizon": 12,
                "rssm_deter_dim": 64,
            }
        )

        self.assertEqual(int(parsed.sequence_length), 8)
        self.assertEqual(int(parsed.imagination_horizon), 12)
        self.assertEqual(int(parsed.custom_overrides["rssm_deter_dim"]), 64)

    def test_export_config_policy_wire_overrides_encodes_reserved_keys_at_boundary(self):
        payload = export_config_policy_wire_overrides(
            ConfigPolicyOverrides(
                sequence_length=8,
                imagination_horizon=12,
                curiosity_enabled=False,
                mastery_enabled=True,
                custom_overrides={"rssm_deter_dim": 64},
            )
        )

        self.assertEqual(int(payload["_policy_batch_length"]), 8)
        self.assertEqual(int(payload["_policy_horizon"]), 12)
        self.assertFalse(bool(payload["_policy_curiosity_enabled"]))
        self.assertTrue(bool(payload["_policy_mastery_enabled"]))
        self.assertEqual(int(payload["rssm_deter_dim"]), 64)

    def test_make_config_policy_overrides_for_env_stays_typed_until_wire_export(self):
        env = _CountingEnv(done_after=2)

        typed = make_config_policy_overrides_for_env(env, include_env_profile=True)
        wire = export_config_policy_wire_overrides(typed)

        self.assertEqual(wire, export_config_policy_wire_overrides(typed))
        self.assertIsInstance(typed, ConfigPolicyOverrides)
        self.assertEqual(int(typed.sequence_length), 32)
        self.assertEqual(int(typed.imagination_horizon), 15)
        self.assertIsNotNone(typed.policy_env_profile)
        self.assertIs(wire["_env_profile"], typed.policy_env_profile)
        self.assertEqual(int(typed.custom_overrides["rssm_deter_dim"]), 96)
        self.assertEqual(int(typed.custom_overrides["rssm_hidden_dim"]), 96)
        self.assertNotIn("sequence_length", typed.custom_overrides)
        self.assertNotIn("imagination_horizon", typed.custom_overrides)

    def test_create_agent_rejects_noncanonical_reserved_override_aliases(self):
        env = _CountingEnv(done_after=2)
        with self.assertRaisesRegex(ValueError, "Non-canonical reserved override"):
            api.create_agent(
                env,
                config_overrides={"router_overrides": {"router_mode": "hard"}},
                device="cpu",
                seed=0,
            )

    def test_real_agent_save_load_roundtrip_preserves_deterministic_action(self):
        env = _CountingEnv(done_after=2)
        obs = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        overrides = {
            "rssm_deter_dim": 96,
            "rssm_hidden_dim": 128,
            "hard_fallback_enabled": True,
        }

        agent = api.create_agent(env, config_overrides=overrides, device="cpu", seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "agent.pt"
            agent.save(str(ckpt_path))
            restored = api.load_agent(str(ckpt_path), env=env, device="cpu")

            agent.reset()
            restored.reset()
            action_a, info_a = agent.act(obs.copy(), deterministic=True)
            action_b, info_b = restored.act(obs.copy(), deterministic=True)

        self.assertEqual(int(action_a), int(action_b))
        self.assertAlmostEqual(
            float(np.asarray(info_a["value"]).reshape(-1)[0]),
            float(np.asarray(info_b["value"]).reshape(-1)[0]),
            places=6,
        )
        self.assertAlmostEqual(
            float(np.asarray(info_a["log_prob"]).reshape(-1)[0]),
            float(np.asarray(info_b["log_prob"]).reshape(-1)[0]),
            places=6,
        )

    def test_real_agent_deterministic_action_is_seed_invariant(self):
        env = _CountingEnv(done_after=2)
        obs = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        overrides = {
            "rssm_deter_dim": 96,
            "rssm_hidden_dim": 128,
            "hard_fallback_enabled": True,
        }

        agent = api.create_agent(env, config_overrides=overrides, device="cpu", seed=0)
        agent.reset()
        torch.manual_seed(1)
        action_a, info_a = agent.act(obs.copy(), deterministic=True)

        agent.reset()
        torch.manual_seed(999)
        action_b, info_b = agent.act(obs.copy(), deterministic=True)

        self.assertEqual(int(action_a), int(action_b))
        self.assertAlmostEqual(
            float(np.asarray(info_a["value"]).reshape(-1)[0]),
            float(np.asarray(info_b["value"]).reshape(-1)[0]),
            places=6,
        )
        self.assertAlmostEqual(
            float(np.asarray(info_a["log_prob"]).reshape(-1)[0]),
            float(np.asarray(info_b["log_prob"]).reshape(-1)[0]),
            places=6,
        )

    def test_real_agent_rssm_continue_overrides_reach_world_model(self):
        env = _CountingEnv(done_after=2)
        overrides = {
            "rssm_continue_temperature": 0.8,
            "rssm_continue_loss_type": "focal",
            "rssm_continue_focal_alpha": 0.4,
            "rssm_continue_focal_gamma": 1.5,
            "rssm_continue_optimistic_bias": 1.25,
            "rssm_continue_positive_weight": 0.9,
            "rssm_continue_negative_weight": 4.0,
            "hard_fallback_enabled": True,
        }

        agent = api.create_agent(env, config_overrides=overrides, device="cpu", seed=0)
        continue_head = agent.world_model.continue_head
        cfg = continue_head.config

        self.assertAlmostEqual(float(cfg.temperature), 0.8, places=6)
        self.assertEqual(str(cfg.loss_type), "focal")
        self.assertAlmostEqual(float(cfg.focal_alpha), 0.4, places=6)
        self.assertAlmostEqual(float(cfg.focal_gamma), 1.5, places=6)
        self.assertAlmostEqual(float(cfg.optimistic_bias), 1.25, places=6)
        self.assertAlmostEqual(float(cfg.positive_weight), 0.9, places=6)
        self.assertAlmostEqual(float(cfg.negative_weight), 4.0, places=6)

    def test_real_agent_rssm_consistency_overrides_reach_world_model(self):
        env = _CountingEnv(done_after=2)
        overrides = {
            "rssm_msc": {
                "enabled": True,
                "horizons": [1, 2, 4],
                "loss_scale": 0.3,
            },
            "rssm_nst": {
                "enabled": True,
                "n_steps": [1, 2, 4],
                "loss_scale": 0.2,
            },
            "rssm_shortcut_consistency": {
                "enabled": True,
                "horizons": [2, 4],
                "loss_scale": 0.4,
                "sample_ratio": 0.5,
                "feature_loss_scale": 0.25,
            },
            "hard_fallback_enabled": True,
        }

        agent = api.create_agent(env, config_overrides=overrides, device="cpu", seed=0)
        world_model = agent.world_model

        self.assertIsNotNone(world_model.msc_head)
        self.assertEqual(tuple(world_model.msc_head.config.horizons), (1, 2, 4))
        self.assertAlmostEqual(float(world_model.msc_head.config.loss_scale), 0.3, places=6)
        self.assertIsNotNone(world_model._nst_loss_module)
        self.assertEqual(tuple(world_model._nst_loss_module.config.n_steps), (1, 2, 4))
        self.assertAlmostEqual(float(world_model._nst_loss_module.config.loss_scale), 0.2, places=6)
        self.assertIsNotNone(world_model._sc_loss_module)
        self.assertEqual(tuple(world_model._sc_loss_module.config.horizons), (2, 4))
        self.assertAlmostEqual(float(world_model._sc_loss_module.config.loss_scale), 0.4, places=6)
        self.assertAlmostEqual(float(world_model._sc_loss_module.config.sample_ratio), 0.5, places=6)
        self.assertAlmostEqual(float(world_model._sc_loss_module.config.feature_loss_scale), 0.25, places=6)

    def test_real_agent_critic_overrides_reach_critic_module(self):
        env = _CountingEnv(done_after=2)
        overrides = {
            "critic_mode": "adaptive",
            "critic_gammas": (0.9, 0.95, 0.99, 0.997),
            "critic_n_ensemble": 3,
            "critic_hidden_dim": 96,
            "critic_primary_gamma_index": 2,
            "critic_use_twohot": True,
            "critic_use_adaptive_routing": True,
            "critic_pessimism": 0.15,
            "critic_target_update_tau": 0.01,
            "hard_fallback_enabled": True,
        }

        agent = api.create_agent(env, config_overrides=overrides, device="cpu", seed=0)
        critic = agent.critic

        self.assertEqual(str(critic.cfg.mode), "adaptive")
        self.assertEqual(tuple(float(g) for g in critic.gammas), (0.9, 0.95, 0.99, 0.997))
        self.assertEqual(int(critic.cfg.n_ensemble), 3)
        self.assertEqual(int(critic.cfg.hidden_dim), 96)
        self.assertEqual(int(critic.cfg.primary_gamma_index), 2)
        self.assertTrue(bool(critic.cfg.use_twohot))
        self.assertTrue(bool(critic.cfg.use_adaptive_routing))
        self.assertAlmostEqual(float(critic.cfg.pessimism), 0.15, places=6)
        self.assertAlmostEqual(float(critic.cfg.target_update_tau), 0.01, places=6)
        self.assertIsNotNone(critic.router)

    def _make_mode_handle(self):
        handle = api.AgentHandle.__new__(api.AgentHandle)
        handle.device = torch.device("cpu")
        handle.profile = SimpleNamespace(obs_shape=(4,), action_dim=2, is_discrete=True)
        handle.world_model = _ModeAwareWorldModel()
        handle.router = _ModeAwareRouter()
        handle.actor = _ModeAwareActor()
        handle.critic = _ModeAwareCritic()
        handle._compiled_router = None
        handle._compiled_actor = None
        handle._compiled_critic = None
        handle.buffer = None
        handle._wm_state = None
        handle._prev_action = None
        return handle

    def _make_invalid_mode_handle(self):
        handle = self._make_mode_handle()
        handle.world_model = _InvalidModeAwareWorldModel()
        return handle

    def _make_continuous_mode_handle(self):
        handle = self._make_mode_handle()
        handle.profile = SimpleNamespace(obs_shape=(4,), action_dim=2, is_discrete=False)
        handle.actor = _ContinuousModeAwareActor()
        return handle

    def test_agent_act_keeps_train_mode_for_stochastic_collection(self):
        handle = self._make_mode_handle()
        for module in (handle.world_model, handle.router, handle.actor, handle.critic):
            module.train(True)

        action, info = handle.act(np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32), deterministic=False)

        self.assertIn(int(action), (0, 1))
        self.assertAlmostEqual(float(np.asarray(info["value"]).reshape(-1)[0]), 0.05, places=6)
        self.assertEqual(handle.world_model.seen_training, [True])
        self.assertEqual(handle.router.seen_training, [True])
        self.assertEqual(handle.actor.seen_training, [True])
        self.assertEqual(handle.critic.seen_training, [True])
        self.assertEqual(handle._wm_state.last_context, "policy")
        self.assertAlmostEqual(float(handle._wm_state.last_wall_strength), 1.0, places=6)
        for module in (handle.world_model, handle.router, handle.actor, handle.critic):
            self.assertTrue(module.training)

    def test_agent_act_forces_eval_mode_for_deterministic_rollouts(self):
        handle = self._make_mode_handle()
        for module in (handle.world_model, handle.router, handle.actor, handle.critic):
            module.train(True)

        action, info = handle.act(np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32), deterministic=True)

        self.assertEqual(int(action), 0)
        self.assertAlmostEqual(float(np.asarray(info["value"]).reshape(-1)[0]), 0.05, places=6)
        self.assertEqual(handle.world_model.seen_training, [False])
        self.assertEqual(handle.router.seen_training, [False])
        self.assertEqual(handle.actor.seen_training, [False])
        self.assertEqual(handle.critic.seen_training, [False])
        self.assertEqual(handle._wm_state.last_context, "policy")
        self.assertAlmostEqual(float(handle._wm_state.last_wall_strength), 1.0, places=6)
        for module in (handle.world_model, handle.router, handle.actor, handle.critic):
            self.assertTrue(module.training)

    def test_agent_act_rejects_noncanonical_policy_wm_state(self):
        handle = self._make_invalid_mode_handle()
        for module in (handle.world_model, handle.router, handle.actor, handle.critic):
            module.train(True)

        with self.assertRaisesRegex(TypeError, "Legacy policy WMState adapters are no longer supported"):
            handle.act(
                np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
                deterministic=True,
            )

    def test_main_delegates_to_run_train(self):
        with mock.patch.object(api, "run_train", return_value={"status": 0}) as run_train_mock:
            status = api.main([
                "--env",
                "CartPole-v1",
                "--steps",
                "7",
                "--resume-from",
                "resume.pt",
                "--resume-model-restore-mode",
                "compatible",
                "--resume-optimizer-restore-mode",
                "skip",
                "--resume-restore-layers",
                "model,buffer",
            ])

        self.assertEqual(status, 0)
        parsed_args = run_train_mock.call_args.args[0]
        self.assertEqual(parsed_args.env, "CartPole-v1")
        self.assertEqual(parsed_args.steps, 7)
        self.assertEqual(parsed_args.resume_from, "resume.pt")
        self.assertEqual(parsed_args.resume_model_restore_mode, "compatible")
        self.assertEqual(parsed_args.resume_optimizer_restore_mode, "skip")
        self.assertEqual(parsed_args.resume_restore_layers, "model,buffer")


if __name__ == "__main__":
    unittest.main()
