"""Synchronous vectorized environment wrapper.

Provides :class:`SyncVecEnv` which runs N gym‑like environments in
lockstep within a single process.  Useful for increasing data
throughput without the complexity of sub‑process parallelism.

Usage::

    env_fns = [lambda: gym.make("CartPole-v1") for _ in range(4)]
    vec_env = SyncVecEnv(env_fns)
    obs = vec_env.reset()            # (4, obs_dim)
    obs, rewards, terminated, truncated, infos = vec_env.step(actions)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class SyncVecEnv:
    """Synchronous vectorized environment.

    Parameters
    ----------
    env_fns : list of callables
        Each callable returns a fresh gym‑like environment instance.
    auto_reset : bool
        If True, automatically reset individual environments when they
        report ``terminated or truncated``.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Any]],
        auto_reset: bool = True,
    ):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.auto_reset = auto_reset
        self._dones = np.zeros(self.num_envs, dtype=bool)

    # ── Core API ─────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset all environments and return stacked observations ``(N, D)``."""
        obs_list: List[np.ndarray] = []
        for env in self.envs:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]  # gymnasium compat (obs, info)
            obs_list.append(np.asarray(obs, dtype=np.float32))
        self._dones[:] = False
        return np.stack(obs_list)

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments with ``actions`` shape ``(N, ...) or (N,)``.

        Returns
        -------
        obs : np.ndarray, shape ``(N, D)``
        rewards : np.ndarray, shape ``(N,)``
        terminated : np.ndarray, shape ``(N,)``  — bool
        truncated : np.ndarray, shape ``(N,)``  — bool
        infos : list of dict
        """
        obs_list: List[np.ndarray] = []
        rewards = np.empty(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict] = []

        for i, env in enumerate(self.envs):
            action = actions[i]
            if isinstance(action, np.ndarray) and action.ndim == 0:
                action = action.item()

            result = env.step(action)
            if len(result) == 5:
                obs, rew, term, trunc, info = result
            else:
                # old gym API: (obs, reward, done, info)
                obs, rew, done, info = result
                term = done
                trunc = False

            rewards[i] = float(rew)
            terminated[i] = bool(term)
            truncated[i] = bool(trunc)
            infos.append(info if isinstance(info, dict) else {})

            if self.auto_reset and (term or trunc):
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

            obs_list.append(np.asarray(obs, dtype=np.float32))

        return np.stack(obs_list), rewards, terminated, truncated, infos

    # ── Utilities ────────────────────────────────────────────────────────

    @property
    def observation_space(self):
        """Observation space of the first (representative) environment."""
        return self.envs[0].observation_space if self.envs else None

    @property
    def action_space(self):
        """Action space of the first (representative) environment."""
        return self.envs[0].action_space if self.envs else None

    def close(self):
        """Close all sub‑environments."""
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    def __len__(self) -> int:
        return self.num_envs

    def __del__(self):
        self.close()
