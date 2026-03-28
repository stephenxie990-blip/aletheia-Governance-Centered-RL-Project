import os
import sys
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.aletheia_train import (
    ReplayBuffer,
    RolloutCollector,
    compute_discounted_returns_to_go,
)
from aletheia.aletheia_api import BufferAdapter, resolve_device
from aletheia.aletheia_config import EnvProfile

from aletheia.aletheia_foundation import (
    DistributionConfig,
    DistributionFactory,
    check_imagine_trajectory_contract,
    check_temporal_contract,
    safe_torch_load,
)


class TestReplayBufferTemporalContract(unittest.TestCase):
    def test_requires_initial_vitals(self):
        buf = ReplayBuffer(capacity=10, store_obs=False)
        buf.start_episode()
        with self.assertRaises(ValueError):
            buf.add_step(
                vitals=np.zeros((4,), dtype=np.float32),
                action=np.zeros((2,), dtype=np.float32),
                reward=0.0,
                done=False,
            )

    def test_finish_episode_enforces_lengths(self):
        buf = ReplayBuffer(capacity=10, store_obs=False)
        buf.start_episode(initial_vitals=np.zeros((4,), dtype=np.float32))
        for t in range(3):
            buf.add_step(
                vitals=np.full((4,), t + 1, dtype=np.float32),
                action=np.full((2,), t, dtype=np.float32),
                reward=float(t),
                done=(t == 2),
            )
        self.assertEqual(len(buf.episodes), 1)
        ep = buf.episodes[0]
        self.assertEqual(ep["vitals"].shape[0], 4)
        self.assertEqual(ep["actions"].shape[0], 3)
        self.assertEqual(ep["rewards"].shape[0], 3)
        self.assertEqual(ep["dones"].shape[0], 3)
        self.assertEqual(ep["log_probs"].shape[0], 3)

    def test_danger_alignment_shift(self):
        buf = ReplayBuffer(capacity=10, store_obs=False)
        buf.start_episode(initial_vitals=np.zeros((4,), dtype=np.float32))
        for t in range(3):
            buf.add_step(
                vitals=np.full((4,), t + 1, dtype=np.float32),
                action=np.zeros((2,), dtype=np.float32),
                reward=0.0,
                done=(t == 2),
            )
        ep = buf.episodes[0]
        danger = ep["danger"]
        self.assertEqual(danger.shape[0], 4)
        self.assertAlmostEqual(float(danger[0]), 0.0)
        self.assertTrue(np.allclose(danger[1:], 0.5, atol=1e-6))

    def test_verify_remaining_steps_uses_truncated(self):
        buf = ReplayBuffer(capacity=10, store_obs=False)
        buf.start_episode(initial_vitals=np.zeros((4,), dtype=np.float32))
        buf.add_step(vitals=np.ones((4,), dtype=np.float32), action=np.zeros((2,), dtype=np.float32), reward=0.0, done=False, terminated=False, truncated=False)
        buf.add_step(vitals=np.ones((4,), dtype=np.float32), action=np.zeros((2,), dtype=np.float32), reward=0.0, done=False, terminated=False, truncated=False)
        buf.add_step(vitals=np.ones((4,), dtype=np.float32), action=np.zeros((2,), dtype=np.float32), reward=0.0, done=True, terminated=False, truncated=True)
        report = buf.verify_remaining_steps()
        self.assertTrue(report["valid"])

    def test_compute_discounted_returns_to_go_matches_full_mc_return(self):
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        returns = compute_discounted_returns_to_go(
            rewards=rewards,
            dones=dones,
            gamma=0.9,
        )
        expected = np.array(
            [
                1.0 + 0.9 * 2.0 + 0.9 * 0.9 * 3.0,
                2.0 + 0.9 * 3.0,
                3.0,
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(returns, expected, atol=1e-6))

    def test_sample_sequences_with_remaining_exposes_absolute_episode_ids(self):
        buf = ReplayBuffer(capacity=10, store_obs=False)

        buf.start_episode(initial_vitals=np.zeros((4,), dtype=np.float32))
        buf.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.zeros((2,), dtype=np.float32),
            reward=0.0,
            done=True,
            log_prob=0.0,
        )

        buf.start_episode(initial_vitals=np.full((4,), 2.0, dtype=np.float32))
        for step in range(3):
            buf.add_step(
                vitals=np.full((4,), step + 3.0, dtype=np.float32),
                action=np.zeros((2,), dtype=np.float32),
                reward=float(step + 1),
                done=(step == 2),
                log_prob=0.0,
            )

        batch = buf.sample_sequences_with_remaining(batch_size=1, seq_len=3)
        self.assertIsNotNone(batch)
        self.assertEqual(int(batch["episode_ids"][0]), 1)
        self.assertTrue(
            np.array_equal(
                batch["step_indices"][0],
                np.array([0, 1, 2], dtype=np.int32),
            )
        )

    def test_load_state_dict_migrates_legacy_episode_payload(self):
        buf = ReplayBuffer(capacity=10, store_obs=False)
        legacy_state = {
            "episodes": [
                {
                    "vitals": np.array(
                        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                        dtype=np.float32,
                    ),
                    "actions": np.array(
                        [[1.0, 0.0], [0.0, 1.0]],
                        dtype=np.float32,
                    ),
                    "rewards": np.array([1.0, 0.5], dtype=np.float32),
                    "dones": np.array([0.0, 1.0], dtype=np.float32),
                }
            ],
            "current": None,
            "capacity": 10,
            "store_obs": False,
        }

        buf.load_state_dict(legacy_state)

        self.assertEqual(len(buf.episodes), 1)
        ep = buf.episodes[0]
        self.assertTrue(np.array_equal(ep["terminated"], ep["dones"]))
        self.assertTrue(np.array_equal(ep["truncated"], np.zeros_like(ep["dones"])))
        self.assertTrue(np.array_equal(ep["dones_src"], ep["dones"]))
        self.assertEqual(int(ep["idx"]), 0)
        self.assertEqual(int(ep["length"]), 3)
        self.assertEqual(int(buf._episode_counter), 1)
        self.assertEqual(int(ep["danger"].shape[0]), int(ep["vitals"].shape[0]))

    def test_load_state_dict_rejects_invalid_temporal_contract(self):
        buf = ReplayBuffer(capacity=10, store_obs=False)
        invalid_state = {
            "episodes": [
                {
                    "vitals": np.array([[0.0], [1.0]], dtype=np.float32),
                    "actions": np.array([[1.0], [0.0]], dtype=np.float32),
                    "rewards": np.array([1.0], dtype=np.float32),
                    "dones": np.array([1.0], dtype=np.float32),
                }
            ],
            "current": None,
        }

        with self.assertRaisesRegex(ValueError, "temporal contract"):
            buf.load_state_dict(invalid_state)


class TestWeightedPointsDangerShift(unittest.TestCase):
    def test_weighted_points_use_state_danger_shift(self):
        buf = ReplayBuffer(capacity=10, store_obs=False)
        ep = {
            "vitals": np.zeros((5, 4), dtype=np.float32),
            "actions": np.zeros((4, 2), dtype=np.float32),
            "rewards": np.zeros((4,), dtype=np.float32),
            "dones": np.zeros((4,), dtype=np.float32),
            "log_probs": np.zeros((4,), dtype=np.float32),
            "danger": np.array([100.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        }
        points = buf._build_weighted_points([ep], seq_len=2, required_ep_len=3)
        self.assertIsNotNone(points)
        _, _, dangers = points
        self.assertTrue(np.allclose(dangers, np.ones_like(dangers), atol=1e-6))


class TestTemporalContracts(unittest.TestCase):
    def test_check_temporal_contract_accepts_valid_batch(self):
        batch = {
            "vitals": np.zeros((2, 4, 3), dtype=np.float32),
            "actions": np.zeros((2, 3, 2), dtype=np.float32),
            "rewards": np.zeros((2, 3), dtype=np.float32),
            "dones": np.zeros((2, 3), dtype=np.float32),
        }
        report = check_temporal_contract(batch, strict=True)
        self.assertTrue(report["valid"])

    def test_check_imagine_trajectory_contract_strict_raises(self):
        class Dummy:
            pass

        imagine_out = Dummy()
        imagine_out.f_policy = torch.zeros((3, 2, 8))
        imagine_out.actions = torch.zeros((3, 2, 2))
        imagine_out.log_probs = torch.zeros((3, 2))
        imagine_out.old_log_probs = torch.zeros((3, 2))
        imagine_out.values = torch.zeros((3, 2))
        imagine_out.rewards_pred = torch.zeros((3, 2))
        imagine_out.continues_pred = torch.zeros((3, 2))
        imagine_out.final_value = torch.zeros((2,))

        report = check_imagine_trajectory_contract(imagine_out, strict=True)
        self.assertTrue(report["valid"])

        imagine_out.old_log_probs = torch.zeros((3, 1))
        with self.assertRaises(ValueError):
            check_imagine_trajectory_contract(imagine_out, strict=True)


class TestSafeTorchLoad(unittest.TestCase):
    def test_safe_torch_load_loads_saved_object(self):
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            torch.save({"a": 1}, path)
            obj = safe_torch_load(path, map_location="cpu", weights_only=True)
            self.assertEqual(obj["a"], 1)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_safe_torch_load_loads_legacy_numpy_array_payload(self):
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        legacy_payload = {
            "arr": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "mask": np.array([True, False], dtype=np.bool_),
        }
        try:
            torch.save(legacy_payload, path)
            obj = safe_torch_load(path, map_location="cpu", weights_only=True)
            self.assertTrue(np.array_equal(np.asarray(obj["arr"]), legacy_payload["arr"]))
            self.assertTrue(np.array_equal(np.asarray(obj["mask"]), legacy_payload["mask"]))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_safe_torch_load_trusted_fallback_suppresses_warning(self):
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            torch.save({"a": 1}, path)
            with patch(
                "aletheia.aletheia_foundation.torch.load",
                side_effect=[
                    RuntimeError("weights-only load failed"),
                    {"a": 1},
                ],
            ) as load_mock:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    obj = safe_torch_load(
                        path,
                        map_location="cpu",
                        weights_only=True,
                        allow_unsafe_fallback=True,
                        trusted_source=True,
                    )
            self.assertEqual(obj["a"], 1)
            self.assertEqual(len(caught), 0)
            self.assertEqual(load_mock.call_count, 2)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class TestReplayBufferCheckpointSchema(unittest.TestCase):
    def test_state_dict_roundtrip_is_weights_only_safe_for_completed_episode(self):
        buf = ReplayBuffer(capacity=10, store_obs=False)
        buf.start_episode(initial_vitals=np.zeros((4,), dtype=np.float32))
        buf.add_step(
            vitals=np.ones((4,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            log_prob=0.1,
        )
        buf.add_step(
            vitals=np.full((4,), 2.0, dtype=np.float32),
            action=np.array([0.0, 1.0], dtype=np.float32),
            reward=0.5,
            done=True,
            terminated=True,
            truncated=False,
            log_prob=0.2,
        )

        state = buf.state_dict()
        self.assertIsInstance(state["episodes"][0]["vitals"], torch.Tensor)
        self.assertIsNone(state["current"])

        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            torch.save({"replay_buffer": state}, path)
            payload = safe_torch_load(path, map_location="cpu", weights_only=True)
            restored = ReplayBuffer(capacity=10, store_obs=False)
            restored.load_state_dict(payload["replay_buffer"])
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        self.assertEqual(len(restored.episodes), 1)
        episode = restored.episodes[0]
        self.assertTrue(
            np.array_equal(
                episode["vitals"],
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )
        self.assertTrue(
            np.array_equal(
                episode["actions"],
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.array_equal(
                episode["log_probs"],
                np.array([0.1, 0.2], dtype=np.float32),
            )
        )

    def test_state_dict_roundtrip_is_weights_only_safe_for_inflight_episode(self):
        buf = ReplayBuffer(capacity=10, store_obs=True)
        buf.start_episode(
            initial_vitals=np.zeros((2,), dtype=np.float32),
            initial_obs={"sensor": np.zeros((3,), dtype=np.float32)},
        )
        buf.add_step(
            vitals=np.ones((2,), dtype=np.float32),
            action=np.array([1.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            terminated=False,
            truncated=False,
            obs={"sensor": np.ones((3,), dtype=np.float32)},
            log_prob=0.3,
        )

        state = buf.state_dict()
        self.assertIsNotNone(state["current"])
        self.assertIsInstance(state["current"]["vitals"][0], torch.Tensor)
        self.assertIsInstance(state["current"]["obs"][0]["sensor"], torch.Tensor)

        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            torch.save({"replay_buffer": state}, path)
            payload = safe_torch_load(path, map_location="cpu", weights_only=True)
            restored = ReplayBuffer(capacity=10, store_obs=True)
            restored.load_state_dict(payload["replay_buffer"])
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        self.assertIsNotNone(restored.current)
        self.assertEqual(len(restored.current["vitals"]), 2)
        self.assertTrue(
            np.array_equal(
                restored.current["vitals"][0],
                np.zeros((2,), dtype=np.float32),
            )
        )
        self.assertTrue(
            np.array_equal(
                restored.current["vitals"][1],
                np.ones((2,), dtype=np.float32),
            )
        )
        self.assertEqual(restored.current["rewards"], [1.0])
        self.assertEqual(restored.current["dones"], [0.0])
        self.assertEqual(restored.current["terminated"], [0.0])
        self.assertEqual(restored.current["truncated"], [0.0])
        self.assertEqual(len(restored.current["log_probs"]), 1)
        self.assertAlmostEqual(restored.current["log_probs"][0], 0.3, places=6)
        self.assertTrue(
            np.array_equal(
                restored.current["obs"][0]["sensor"],
                np.zeros((3,), dtype=np.float32),
            )
        )
        self.assertTrue(
            np.array_equal(
                restored.current["obs"][1]["sensor"],
                np.ones((3,), dtype=np.float32),
            )
        )


class TestDistributionFactoryKL(unittest.TestCase):
    def test_compute_kl_categorical_runs(self):
        cfg = DistributionConfig(
            dist_type="categorical",
            num_distributions=2,
            num_classes=3,
            use_unimix=False,
        )
        factory = DistributionFactory(config=cfg)
        q_logits = torch.randn(4, cfg.num_distributions, cfg.num_classes)
        p_logits = torch.randn(4, cfg.num_distributions, cfg.num_classes)
        kl = factory.compute_kl(q_logits, p_logits, balance=True, alpha=0.8)
        self.assertEqual(tuple(kl.shape), (4,))
        self.assertTrue(torch.isfinite(kl).all().item())


class _NoDoneEnv:
    def __init__(self):
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(n=2)
        self._count = 0

    def reset(self, seed=None):
        self._count = 0
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        self._count += 1
        obs = np.full((4,), float(self._count), dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def close(self):
        return None


class _ConstantPolicyModel(nn.Module):
    def forward(self, obs):
        logits = torch.zeros((obs.shape[0], 2), device=obs.device)
        dist = torch.distributions.Categorical(logits=logits)
        value = torch.zeros((obs.shape[0],), device=obs.device)
        return dist, value


class TestRolloutCollectorTimeout(unittest.TestCase):
    def test_max_steps_boundary_marks_done_and_truncated(self):
        collector = RolloutCollector(
            model=_ConstantPolicyModel(),
            env=_NoDoneEnv(),
            device=torch.device("cpu"),
            max_steps=2,
        )
        batch = collector.collect(num_steps=3)
        self.assertIsNotNone(batch)
        self.assertTrue(np.allclose(batch["dones"], np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(batch["terminated"], np.array([0.0, 0.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(batch["truncated"], np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        self.assertEqual(collector.episode_lengths, [2])


class TestBufferAdapterTerminationSemantics(unittest.TestCase):
    def _make_profile(self) -> EnvProfile:
        return EnvProfile(
            env_id="DummyEnv-v0",
            obs_shape=(4,),
            obs_modality="vector",
            obs_dim=4,
            action_dim=2,
            is_discrete_action=True,
            env_category="SIMPLE_VECTOR",
            temporal_dependency="low",
            is_pomdp=False,
            reward_variance_level="low",
            max_episode_steps=100,
        )

    def test_preserves_explicit_terminated_and_truncated(self):
        buf = ReplayBuffer(capacity=10, store_obs=False)
        adapter = BufferAdapter(buf, self._make_profile())
        obs = np.zeros((4,), dtype=np.float32)
        next_obs = np.ones((4,), dtype=np.float32)

        adapter.start_episode(initial_obs=obs)
        adapter.add_transition(
            obs=obs,
            action=1,
            reward=0.0,
            next_obs=next_obs,
            done=True,
            info={"terminated": False, "truncated": True},
        )

        ep = buf.episodes[0]
        self.assertEqual(float(ep["terminated"][0]), 0.0)
        self.assertEqual(float(ep["truncated"][0]), 1.0)

    def test_supports_legacy_timelimit_truncated_flag(self):
        buf = ReplayBuffer(capacity=10, store_obs=False)
        adapter = BufferAdapter(buf, self._make_profile())
        obs = np.zeros((4,), dtype=np.float32)
        next_obs = np.ones((4,), dtype=np.float32)

        adapter.start_episode(initial_obs=obs)
        adapter.add_transition(
            obs=obs,
            action=1,
            reward=0.0,
            next_obs=next_obs,
            done=True,
            info={"TimeLimit.truncated": True},
        )

        ep = buf.episodes[0]
        self.assertEqual(float(ep["terminated"][0]), 0.0)
        self.assertEqual(float(ep["truncated"][0]), 1.0)


class TestResolveDevice(unittest.TestCase):
    def test_resolve_mps_when_available(self):
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not hasattr(mps_backend, "is_available"):
            self.skipTest("PyTorch build has no mps backend API")
        with patch.object(mps_backend, "is_available", return_value=True):
            dev = resolve_device("mps")
        self.assertEqual(dev.type, "mps")




if __name__ == "__main__":
    unittest.main()
