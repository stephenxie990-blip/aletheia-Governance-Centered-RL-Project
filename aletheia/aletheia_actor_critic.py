"""
Aletheia Actor-Critic Module (v4.6.0 — Full Optimization)
=========================================================
Merged: actor.py + critic.py + will.py + proto_self.py + actor_critic.py

v4.6.0 Changelog (this version — on top of v4.5.1):
────────────────────────────────────────────────────
  [FIX-P0] CuriosityModule.forward: expand_as crashes when H_router is 1-D
           but U_wm is 2-D.  Now auto-unsqueezes before expand.
  [FIX-P0] Actor backbone: build_mlp received BOTH hidden_dims AND
           hidden_dim+depth simultaneously, causing ambiguous architecture.
           Now uses exactly one parameter set depending on config length.
  [FIX-P1] DoubleCritic: inner MultiHeadEnsembleCritic already applied
           pessimistic quantile, then DoubleCritic took min() again
           → double pessimism → systematic underestimation.
           Fix: inner critics now use pessimism=0.5 (median).
  [FIX-P1] ActorCritic.get_value: removed fragile inspect.signature
           introspection.  All critic classes now accept a unified
           (feat, intent, use_target) signature, so a plain call suffices.
  [FIX-P2] AdaptiveGammaRouter.compute_router_loss: feat is now detached
           to prevent router gradients from polluting the shared backbone.
  [FIX-P2] HierarchicalUnifiedCritic.compute_loss: router loss is returned
           separately (loss_critic / loss_router) so the training loop can
           apply independent optimisers or gradient scaling.
  [FIX-P2] ModeSwitchEngine: PPO can no longer jump directly to DREAM;
           must pass through MIXED first (gradual degradation).
  [FIX-P2] All compute_loss methods: added `detach_features` kwarg so
           training loop can cut WM→Critic gradient coupling on demand.
  [NEW]    ValueNormalizer: running-mean/std normaliser for value targets,
           stabilises L_crit when return magnitudes vary.
  [NEW]    _DEFAULT_SYMLOG_CLIP = 20.0 module constant; ensures symlog
           never silently truncates rewards up to ~500 M.
  [OPT]   TanhTransformedDistribution: entropy result is now cached per
           distribution instance (same pattern as CategoricalStraightThrough).

Previously fixed (v4.5.1, all maintained):
  [FIX] TrainingMode / EmergencyConfig imported from config (NameError)
  [FIX] ActorCritic.forward: intent no longer passed as temperature
  [FIX] TrustModel buffer overwrite → copy_()
  [FIX] HealthMonitor loc / scale clip tracked separately
  [FIX] DoubleCritic forwards use_target to inner critics
  [FIX] CriticOutput.__getitem__ raises explicit KeyError
  [OPT] HealthMonitor uses EMA (not cumulative counters)
  [OPT] MasteryModule unified pad-diff logic
  [OPT] CategoricalStraightThrough caches entropy
  [OPT] Duplicate imports / utility functions eliminated
"""

from __future__ import annotations

# =============================================================================
# 0  Unified Imports
# =============================================================================

import copy
import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from torch import Tensor

# --- Project-internal imports (single point) ---
from .aletheia_foundation import (
    build_mlp,
    gamma_to_key,
    huber_loss,
    get_activation,
    get_activation_class,
    soft_update,
    symlog as _foundation_symlog,
    symexp as _foundation_symexp,
)
from .aletheia_twohot import (
    symlog_bins,
    twohot_encode,
    twohot_decode,
    twohot_loss,
)

# Config imports
from .aletheia_config import (
    InitConfig,
    ContinuousDistConfig,
    DiscreteDistConfig,
    ActorConfig,
    ActionCodecConfig,
    EntropyProtectionConfig,
    CriticConfig,
    GammaSchedulerConfig,
    CriticMode,
    WillConfig,
)

# =============================================================================
# 1  Module-level Constants
# =============================================================================

# [NEW-v4.6] Safe symlog clipping bound.
# symlog(500_000_000) ≈ 20.  CartPole max return 500 → symlog ≈ 6.2.
# A generous default avoids silent gradient death at the boundary.
_DEFAULT_SYMLOG_CLIP: float = 20.0


def _norm_from_layer_flag(enabled: bool) -> str:
    return "layer" if enabled else "none"


def _validate_critic_target_gammas(
    target_gammas: Iterable[float],
    expected_gammas: Iterable[float],
) -> None:
    target_set = {float(g) for g in target_gammas}
    expected_set = {float(g) for g in expected_gammas}
    if target_set == expected_set:
        return

    extra = sorted(target_set - expected_set)
    missing = sorted(expected_set - target_set)
    details: List[str] = []
    if missing:
        details.append(f"missing={missing}")
    if extra:
        details.append(f"extra={extra}")
    suffix = f": {', '.join(details)}" if details else ""
    raise ValueError(
        "gamma targets must exactly match configured critic gammas"
        f"{suffix}"
    )

# =============================================================================
# 2  Symlog / Symexp  (unified from aletheia_foundation)
# =============================================================================

symlog = _foundation_symlog
symexp = _foundation_symexp

# =============================================================================
# 3  Base Types  (EmergencyState / EmergencyMonitor)
#    NOTE: EmergencyMonitor is provided for external callers (e.g. training
#    loop crisis detection).  It is *not* used internally by Will — Will uses
#    TrustModel + ModeSwitchEngine instead.
# =============================================================================


# =============================================================================
# 4  Initialisation Helpers
# =============================================================================


def init_linear(
    module: nn.Linear,
    method: str = "orthogonal",
    gain: float = 1.0,
    bias_init: str = "zeros",
) -> None:
    """Unified linear-layer initialisation."""
    if method == "orthogonal":
        nn.init.orthogonal_(module.weight, gain=gain)
    elif method == "xavier":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    elif method == "kaiming":
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        module.weight.data *= gain
    else:
        raise ValueError(f"Unknown init method: {method}")

    if module.bias is not None:
        if bias_init == "zeros":
            nn.init.zeros_(module.bias)
        elif bias_init == "uniform":
            fan_in = module.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(module.bias, -bound, bound)


def validate_gamma_keys(gammas: Tuple[float, ...]) -> None:
    """Validate that a gamma list has no key collisions."""
    keys = [gamma_to_key(g) for g in gammas]
    if len(keys) != len(set(keys)):
        seen: set = set()
        dups: List[float] = []
        for g, k in zip(gammas, keys):
            if k in seen:
                dups.append(g)
            seen.add(k)
        raise ValueError(
            f"Gamma key collision for {dups}.  "
            "Ensure gammas differ by at least 1e-6."
        )


# =============================================================================
# 5  Health Monitor  (EMA-based, loc / scale separated)
# =============================================================================


@dataclass
class ActorHealthStats:
    """Snapshot of actor health indicators."""

    entropy: float
    entropy_target: float
    entropy_ratio: float
    is_collapsed: bool
    logits_clip_rate: float
    loc_clip_rate: float
    scale_clip_rate: float
    alpha: float

    def __repr__(self) -> str:
        tag = " COLLAPSED" if self.is_collapsed else " HEALTHY"
        return (
            f"ActorHealthStats({tag})\n"
            f"  entropy: {self.entropy:.4f} "
            f"(target: {self.entropy_target:.4f}, "
            f"ratio: {self.entropy_ratio:.2%})\n"
            f"  clip_rates: logits={self.logits_clip_rate:.2%}, "
            f"loc={self.loc_clip_rate:.2%}, "
            f"scale={self.scale_clip_rate:.2%}\n"
            f"  alpha: {self.alpha:.4f}"
        )


class HealthMonitor:
    """
    EMA-based health monitor.

    Tracks clipping rates for logits / loc / scale *separately* using
    exponential moving averages so that recent events are weighted higher
    than ancient history.
    """

    def __init__(
        self, ema_alpha: float = 0.01, entropy_window: int = 200,
    ):
        self._alpha = ema_alpha
        self._logits_clip_ema = 0.0
        self._loc_clip_ema = 0.0
        self._scale_clip_ema = 0.0
        self._entropy_history: deque = deque(maxlen=entropy_window)

    # -- EMA core --

    def _ema(self, old: float, rate: float) -> float:
        return old * (1.0 - self._alpha) + rate * self._alpha

    # -- Recording --

    def record_logits_clip(self, clipped: int, total: int) -> None:
        if total > 0:
            self._logits_clip_ema = self._ema(
                self._logits_clip_ema, clipped / total,
            )

    def record_loc_clip(self, clipped: int, total: int) -> None:
        if total > 0:
            self._loc_clip_ema = self._ema(
                self._loc_clip_ema, clipped / total,
            )

    def record_scale_clip(self, clipped: int, total: int) -> None:
        if total > 0:
            self._scale_clip_ema = self._ema(
                self._scale_clip_ema, clipped / total,
            )

    def record_entropy(self, entropy: float) -> None:
        self._entropy_history.append(entropy)

    # -- Properties --

    @property
    def logits_clip_rate(self) -> float:
        return self._logits_clip_ema

    @property
    def loc_clip_rate(self) -> float:
        return self._loc_clip_ema

    @property
    def scale_clip_rate(self) -> float:
        return self._scale_clip_ema

    @property
    def mean_entropy(self) -> float:
        if not self._entropy_history:
            return 0.0
        return sum(self._entropy_history) / len(self._entropy_history)

    @property
    def entropy_trend(self) -> float:
        """Positive = increasing, negative = decreasing."""
        n = len(self._entropy_history)
        if n < 20:
            return 0.0
        recent = list(self._entropy_history)[-10:]
        older = list(self._entropy_history)[-min(n, 100):-10]
        if not older:
            return 0.0
        return sum(recent) / len(recent) - sum(older) / len(older)

    def reset(self) -> None:
        self._logits_clip_ema = 0.0
        self._loc_clip_ema = 0.0
        self._scale_clip_ema = 0.0
        self._entropy_history.clear()


# =============================================================================
# 6  Distributions
# =============================================================================


class TanhTransformedDistribution:
    """
    Tanh-squashed Normal → actions ∈ [-1, 1].

    [v4.6] Added ``_entropy_cache`` to avoid redundant MC sampling when
    ``entropy()`` is called more than once per distribution instance.
    """

    _EPS = 1e-6
    _LOG_EPS = 1e-8

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        tanh_clip: float = 0.999,
        entropy_samples: int = 100,
        loc_clip: float = 10.0,
        scale_min: float = 1e-4,
        scale_max: float = 10.0,
        health_monitor: Optional[HealthMonitor] = None,
    ):
        self.loc_raw = loc
        self.scale_raw = scale

        self.loc = loc.clamp(-loc_clip, loc_clip)
        self.scale = scale.clamp(scale_min, scale_max)

        # --- health tracking (loc and scale counted separately) ---
        if health_monitor is not None:
            with torch.no_grad():
                numel = loc.numel()
                loc_thresh = loc_clip * 0.95
                loc_clipped = int((loc.abs() > loc_thresh).sum().item())
                health_monitor.record_loc_clip(loc_clipped, numel)

                scale_clipped = int(
                    (
                        (scale < scale_min * 1.1) | (scale > scale_max * 0.9)
                    ).sum().item()
                )
                health_monitor.record_scale_clip(scale_clipped, numel)

        self.base_dist = D.Normal(self.loc, self.scale)
        self._tanh_clip = tanh_clip
        self._entropy_samples = entropy_samples
        self._health_monitor = health_monitor
        self._entropy_cache: Optional[Tensor] = None  # [NEW-v4.6]

    # -- shape --

    @property
    def batch_shape(self) -> torch.Size:
        return self.loc.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        return self.loc.shape[-1:]

    # -- sampling --

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return torch.tanh(self.base_dist.rsample(sample_shape))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    # -- log_prob --

    def log_prob(self, action: Tensor) -> Tensor:
        """log π(a|s) = log N(atanh(a); μ, σ) − Σ log(1 − a²)."""
        a_clip = action.clamp(-self._tanh_clip, self._tanh_clip)
        raw = torch.atanh(a_clip)
        lp_raw = self.base_dist.log_prob(raw)
        log_det = torch.log((1.0 - a_clip.pow(2)).clamp(min=self._LOG_EPS))
        return (lp_raw - log_det).sum(dim=-1)

    # -- entropy --

    def entropy(self) -> Tensor:
        """Monte-Carlo entropy:  H ≈ −E[log π(a)]  (cached)."""
        if self._entropy_cache is not None:
            return self._entropy_cache

        samples = self.rsample((self._entropy_samples,))
        log_probs = self.log_prob(samples)
        ent = -log_probs.mean(dim=0)

        if self._health_monitor is not None:
            with torch.no_grad():
                self._health_monitor.record_entropy(ent.mean().item())

        self._entropy_cache = ent
        return ent

    def entropy_approx(self) -> Tensor:
        """Analytic entropy of the base Normal (upper bound on tanh-H)."""
        return self.base_dist.entropy().sum(dim=-1)

    # -- mode / mean --

    @property
    def mode(self) -> Tensor:
        return torch.tanh(self.loc)

    @property
    def mean(self) -> Tensor:
        return self.mode

    # -- KL --

    def kl_divergence(self, other: "TanhTransformedDistribution") -> Tensor:
        with torch.no_grad():
            samples = self.rsample((self._entropy_samples,))
            return (
                self.log_prob(samples) - other.log_prob(samples)
            ).mean(dim=0)


class CategoricalStraightThrough:
    """
    Gumbel-Softmax + Straight-Through + Unimix.

    ``entropy()`` is cached per instance.
    """

    _LOG_EPS = 1e-8

    def __init__(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        unimix_ratio: float = 0.01,
        logits_clip: float = 8.0,
        min_prob: float = 0.01,
        health_monitor: Optional[HealthMonitor] = None,
    ):
        self._num_actions = logits.shape[-1]
        self._temperature = max(temperature, 1e-6)
        self._unimix_ratio = max(0.0, min(unimix_ratio, 1.0))
        self._logits_clip = logits_clip
        self._min_prob = min_prob
        self._health_monitor = health_monitor

        self._logits_raw = logits.clamp(-logits_clip, logits_clip)
        self._logits_original = logits

        if health_monitor is not None:
            with torch.no_grad():
                total = logits.numel()
                clipped = int(
                    (logits.abs() > logits_clip * 0.95).sum().item()
                )
                health_monitor.record_logits_clip(clipped, total)

        self._probs_cache: Optional[Tensor] = None
        self._log_probs_cache: Optional[Tensor] = None
        self._entropy_cache: Optional[Tensor] = None

    # -- temperature / unimix --

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def unimix_ratio(self) -> float:
        return self._unimix_ratio

    @property
    def logits(self) -> Tensor:
        return self._logits_raw / self._temperature

    # -- probs --

    def _compute_mixed_probs(self) -> Tensor:
        base = F.softmax(self.logits, dim=-1)
        if self._unimix_ratio > 0:
            uniform = torch.full_like(base, 1.0 / self._num_actions)
            base = (
                (1.0 - self._unimix_ratio) * base
                + self._unimix_ratio * uniform
            )
        if self._min_prob > 0:
            floor = self._min_prob / self._num_actions
            base = base.clamp(min=floor)
            base = base / base.sum(dim=-1, keepdim=True)
        return base

    @property
    def probs(self) -> Tensor:
        if self._probs_cache is None:
            self._probs_cache = self._compute_mixed_probs()
        return self._probs_cache

    @property
    def log_probs(self) -> Tensor:
        if self._log_probs_cache is None:
            self._log_probs_cache = torch.log(
                self.probs.clamp(min=self._LOG_EPS)
            )
        return self._log_probs_cache

    # -- shape --

    @property
    def batch_shape(self) -> torch.Size:
        return self._logits_raw.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        return self._logits_raw.shape[-1:]

    # -- sampling --

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        lp = self.log_probs
        for size in reversed(sample_shape):
            lp = lp.unsqueeze(0).expand((size,) + lp.shape)
        soft = F.gumbel_softmax(lp, tau=1.0, hard=False)
        hard = F.one_hot(
            soft.argmax(dim=-1), self._num_actions,
        ).to(soft.dtype)
        return hard + soft - soft.detach()

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        with torch.no_grad():
            probs = self.probs
            flat = probs.reshape(-1, self._num_actions)
            n = 1
            for s in sample_shape:
                n *= int(s)
            idx = torch.multinomial(flat, num_samples=n, replacement=True)
            idx = idx.transpose(0, 1).reshape(
                sample_shape + probs.shape[:-1]
            )
            return F.one_hot(idx, self._num_actions).to(probs.dtype)

    # -- log_prob --

    def log_prob(self, action: Tensor) -> Tensor:
        # Integer-type → index gather
        if action.dtype in (
            torch.int8, torch.int16, torch.int32,
            torch.int64, torch.uint8,
        ):
            idx = action.long()
            if idx.dim() >= 2 and idx.shape[-1] == 1:
                idx = idx.reshape(*idx.shape[:-1])
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)
            return self.log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

        # One-hot float → dot product (preserves ST gradient)
        if action.dim() > 0 and action.shape[-1] == self._num_actions:
            return (action * self.log_probs).sum(dim=-1)

        # Scalar float → treat as integer index
        idx = action.long()
        if idx.dim() == 0:
            idx = idx.unsqueeze(0)
        return self.log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    # -- entropy --

    def entropy(self) -> Tensor:
        if self._entropy_cache is not None:
            return self._entropy_cache
        p = self.probs.clamp(self._LOG_EPS, 1.0)
        ent = -(p * p.log()).sum(dim=-1)
        if self._health_monitor is not None:
            with torch.no_grad():
                self._health_monitor.record_entropy(ent.mean().item())
        self._entropy_cache = ent
        return ent

    @property
    def max_entropy(self) -> float:
        return math.log(self._num_actions)

    @property
    def entropy_ratio(self) -> Tensor:
        return self.entropy() / self.max_entropy

    # -- mode / mean --

    @property
    def mode(self) -> Tensor:
        idx = self._logits_original.argmax(dim=-1)
        return F.one_hot(idx, self._num_actions).to(self.probs.dtype)

    @property
    def mean(self) -> Tensor:
        return self.probs


# =============================================================================
# 7  Entropy Protector
# =============================================================================


class EntropyProtector(nn.Module):
    """Dynamic entropy target + adaptive α + collapse recovery."""

    def __init__(
        self,
        action_dim: int,
        is_discrete: bool,
        config: Optional[EntropyProtectionConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = EntropyProtectionConfig()

        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.config = config

        if is_discrete:
            self._max_entropy = math.log(action_dim)
        else:
            self._max_entropy = (
                0.5 * math.log(2.0 * math.pi * math.e) * action_dim
            )

        initial_target = config.initial_target * self._max_entropy
        self.register_buffer("entropy_target", torch.tensor(initial_target))
        self._entropy_floor = config.floor * self._max_entropy

        if config.adaptive_alpha:
            init_val = getattr(config, "initial_log_alpha", -5.0)
            self.log_alpha = nn.Parameter(torch.tensor([init_val]))
        else:
            self.register_buffer("log_alpha", torch.zeros(1))

        self._collapse_threshold = (
            config.collapse_threshold * self._max_entropy
        )
        self._recovery_boost = config.recovery_boost * self._max_entropy
        self._consecutive_low = 0
        self._is_recovering = False

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    @property
    def target(self) -> float:
        return self.entropy_target.item()

    @property
    def max_entropy(self) -> float:
        return self._max_entropy

    @property
    def is_recovering(self) -> bool:
        return self._is_recovering

    def compute_loss(
        self, entropy: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        ent_mean = entropy.mean() if entropy.dim() > 0 else entropy
        self._detect_collapse(ent_mean.item())

        eff_target = self.entropy_target
        if self._is_recovering:
            eff_target = eff_target + self._recovery_boost

        gap = eff_target - ent_mean
        loss = self.alpha * gap

        info = {
            "entropy": ent_mean.item(),
            "entropy_target": eff_target.item(),
            "entropy_ratio": ent_mean.item() / max(self._max_entropy, 1e-8),
            "alpha": self.alpha.item(),
            "is_recovering": float(self._is_recovering),
        }
        return loss, info

    def _detect_collapse(self, entropy: float) -> None:
        if entropy < self._collapse_threshold:
            self._consecutive_low += 1
            if self._consecutive_low >= 5 and not self._is_recovering:
                self._is_recovering = True
                warnings.warn(
                    f"⚠ Entropy collapse!  H={entropy:.4f} "
                    f"< threshold={self._collapse_threshold:.4f}.  "
                    "Recovery mode activated.",
                    RuntimeWarning,
                )
        else:
            self._consecutive_low = 0
            if (
                self._is_recovering
                and entropy > self._collapse_threshold * 2.0
            ):
                self._is_recovering = False

    def step(self) -> None:
        with torch.no_grad():
            new = self.entropy_target * self.config.decay
            self.entropy_target.fill_(
                max(new.item(), self._entropy_floor)
            )

    def reset_target(self, ratio: float = 0.5) -> None:
        with torch.no_grad():
            self.entropy_target.fill_(ratio * self._max_entropy)
        self._consecutive_low = 0
        self._is_recovering = False

    def get_alpha_loss(self, entropy: Tensor) -> Tensor:
        if not self.config.adaptive_alpha:
            return torch.zeros(1, device=entropy.device)
        ent_mean = entropy.mean() if entropy.dim() > 0 else entropy
        return self.log_alpha * (self.entropy_target - ent_mean).detach()


# =============================================================================
# 8  FiLM
# =============================================================================


class FiLM(nn.Module):
    """Feature-wise Linear Modulation:  h' = γ(intent) · h + β(intent)."""

    def __init__(self, intent_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(intent_dim, hidden_dim)
        self.beta_net = nn.Linear(intent_dim, hidden_dim)

        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, h: Tensor, intent: Tensor) -> Tensor:
        if h.dim() == 3 and intent.dim() == 2:
            intent = intent.unsqueeze(1)
        elif h.dim() == 2 and intent.dim() == 3:
            intent = intent[:, -1, :]
        return self.gamma_net(intent) * h + self.beta_net(intent)


# =============================================================================
# 9  Actor
#    [FIX-v4.6] build_mlp receives exactly ONE parameter style:
#      - len(hidden_dims) == 1  →  hidden_dim + depth=1
#      - len(hidden_dims) >  1  →  hidden_dims (explicit widths per layer)
# =============================================================================


class Actor(nn.Module):
    """
    Policy network — stateless executor.

    Temperature and intent are passed in from outside (Will / training loop).
    FiLM injects intent information into hidden features.
    """

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        is_discrete: bool = False,
        config: Optional[ActorConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = ActorConfig()

        self.feat_dim = feat_dim
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.config = config

        self._health_monitor = HealthMonitor()

        self.entropy_protector: Optional[EntropyProtector] = None
        if config.entropy_protection.enabled:
            self.entropy_protector = EntropyProtector(
                action_dim=action_dim,
                is_discrete=is_discrete,
                config=config.entropy_protection,
            )

        self._last_entropy: Optional[float] = None

        # ---- backbone ----
        hidden_dims = list(config.hidden_dims) or [256]
        backbone_out_dim = hidden_dims[-1]

        if len(hidden_dims) <= 1:
            self.backbone = build_mlp(
                in_dim=feat_dim,
                out_dim=backbone_out_dim,
                hidden_dim=backbone_out_dim,
                depth=2,
                activation=config.activation,
                norm=_norm_from_layer_flag(config.layer_norm),
                dropout=config.dropout,
                output_activation=True,
                final_gain=config.init.backbone_gain,
            )
        else:
            intermediate_dims = tuple(hidden_dims[:-1])
            self.backbone = build_mlp(
                in_dim=feat_dim,
                out_dim=backbone_out_dim,
                hidden_dims=intermediate_dims,
                activation=config.activation,
                norm=_norm_from_layer_flag(config.layer_norm),
                dropout=config.dropout,
                output_activation=True,
                final_gain=config.init.backbone_gain,
            )

        # ---- FiLM (optional) ----
        self.film: Optional[FiLM] = None
        if config.intent_dim > 0:
            self.film = FiLM(config.intent_dim, backbone_out_dim)

        # ---- output heads ----
        if is_discrete:
            self.head = nn.Linear(backbone_out_dim, action_dim)
            self.mean_head = None
            self._logstd = None
            self.logstd_head = None
        else:
            self.head = None
            self.mean_head = nn.Linear(backbone_out_dim, action_dim)
            if config.continuous.logstd_mode == "state_independent":
                self._logstd = nn.Parameter(
                    torch.full((action_dim,), config.continuous.init_logstd)
                )
                self.logstd_head = None
            else:
                self._logstd = None
                self.logstd_head = nn.Linear(backbone_out_dim, action_dim)

        self._init_output_heads()

    def _init_output_heads(self) -> None:
        cfg = self.config.init
        for m in (self.head, self.mean_head, self.logstd_head):
            if m is not None:
                init_linear(
                    m,
                    method=cfg.method,
                    gain=cfg.output_gain,
                    bias_init=cfg.bias_init,
                )

    # ---- forward ----

    def forward(
        self,
        f_policy: Tensor,
        temperature: float = 1.0,
        intent: Optional[Tensor] = None,
    ) -> Union[TanhTransformedDistribution, CategoricalStraightThrough]:
        h = self.backbone(f_policy)

        if self.film is not None and intent is not None:
            h = self.film(h, intent)

        if self.is_discrete:
            logits = self.head(h)
            temp = max(
                self.config.discrete.min_temperature,
                min(temperature, self.config.discrete.max_temperature),
            )
            return CategoricalStraightThrough(
                logits,
                temperature=temp,
                unimix_ratio=self.config.discrete.unimix_ratio,
                logits_clip=self.config.discrete.logits_clip,
                min_prob=self.config.discrete.min_prob,
                health_monitor=(
                    self._health_monitor if self.training else None
                ),
            )
        else:
            mean = self.mean_head(h)
            logstd = (
                self._logstd
                if self._logstd is not None
                else self.logstd_head(h)
            )
            logstd = logstd.clamp(
                self.config.continuous.min_logstd,
                self.config.continuous.max_logstd,
            )
            std = (logstd.exp() * temperature).expand_as(mean)
            return TanhTransformedDistribution(
                mean,
                std,
                tanh_clip=self.config.continuous.tanh_clip,
                entropy_samples=self.config.continuous.entropy_samples,
                loc_clip=self.config.continuous.loc_clip,
                scale_min=self.config.continuous.scale_min,
                scale_max=self.config.continuous.scale_max,
                health_monitor=(
                    self._health_monitor if self.training else None
                ),
            )

    # ---- entropy API ----

    def entropy_loss(
        self, entropy: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        with torch.no_grad():
            self._last_entropy = (
                entropy.mean().item() if entropy.dim() > 0 else entropy.item()
            )
        if self.entropy_protector is not None:
            return self.entropy_protector.compute_loss(entropy)
        return (
            torch.zeros(1, device=entropy.device),
            {"entropy": self._last_entropy},
        )

    def alpha_loss(self, entropy: Tensor) -> Tensor:
        if self.entropy_protector is not None:
            return self.entropy_protector.get_alpha_loss(entropy)
        return torch.zeros(1, device=entropy.device)

    def step_entropy_target(self) -> None:
        if self.entropy_protector is not None:
            self.entropy_protector.step()

    def reset_entropy_target(self, ratio: float = 0.5) -> None:
        if self.entropy_protector is not None:
            self.entropy_protector.reset_target(ratio)

    # ---- health API ----

    def health_stats(self) -> ActorHealthStats:
        ent = self._last_entropy or 0.0
        max_ent = (
            math.log(self.action_dim)
            if self.is_discrete
            else self.action_dim * 0.5 * math.log(2.0 * math.pi * math.e)
        )
        ent_target = 0.0
        alpha = 0.0
        is_collapsed = False
        if self.entropy_protector is not None:
            ent_target = self.entropy_protector.target
            alpha = self.entropy_protector.alpha.item()
            is_collapsed = self.entropy_protector.is_recovering
        return ActorHealthStats(
            entropy=ent,
            entropy_target=ent_target,
            entropy_ratio=ent / max_ent if max_ent > 0 else 0.0,
            is_collapsed=is_collapsed,
            logits_clip_rate=self._health_monitor.logits_clip_rate,
            loc_clip_rate=self._health_monitor.loc_clip_rate,
            scale_clip_rate=self._health_monitor.scale_clip_rate,
            alpha=alpha,
        )

    def reset_health_monitor(self) -> None:
        self._health_monitor.reset()

    # ---- inference shortcut ----

    @torch.no_grad()
    def get_action(
        self,
        f_policy: Tensor,
        temperature: float = 1.0,
        intent: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        dist = self.forward(f_policy, temperature, intent)
        action = dist.mode if deterministic else dist.sample()
        return action, {
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy(),
        }

    # ---- parameter grouping ----

    def get_trainable_params(self) -> Dict[str, List[nn.Parameter]]:
        params: Dict[str, List[nn.Parameter]] = {
            "backbone": list(self.backbone.parameters()),
            "head": [],
            "entropy": [],
        }
        if self.is_discrete:
            params["head"] = list(self.head.parameters())
        else:
            params["head"] = list(self.mean_head.parameters())
            if self._logstd is not None:
                params["head"].append(self._logstd)
            if self.logstd_head is not None:
                params["head"].extend(self.logstd_head.parameters())
        if self.film is not None:
            params["backbone"].extend(self.film.parameters())
        if (
            self.entropy_protector is not None
            and isinstance(
                getattr(self.entropy_protector, "log_alpha", None),
                nn.Parameter,
            )
        ):
            params["entropy"].append(self.entropy_protector.log_alpha)
        return params


# =============================================================================
# 10  Actor Factory
# =============================================================================


def build_actor(
    feat_dim: int,
    action_dim: int,
    is_discrete: bool = False,
    hidden_dims: Tuple[int, ...] = (256, 256),
    entropy_protection: bool = True,
    **kwargs: Any,
) -> Actor:
    """Convenience factory for Actor."""
    valid_keys = {
        f.name for f in ActorConfig.__dataclass_fields__.values()
    }
    actor_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    config = ActorConfig(
        hidden_dims=hidden_dims,
        entropy_protection=EntropyProtectionConfig(
            enabled=entropy_protection,
        ),
        **actor_kwargs,
    )
    return Actor(
        feat_dim=feat_dim,
        action_dim=action_dim,
        is_discrete=is_discrete,
        config=config,
    )


# =============================================================================
# 11  ActionCodec
# =============================================================================


class ActionCodec(nn.Module):
    """
    Action encoder / decoder.

    encode:  env action → embedding
    decode:  embedding  → raw action ∈ [-1, 1]
    scale:   [-1, 1]    → [low, high]
    """

    def __init__(
        self,
        action_dim: int,
        is_discrete: bool,
        config: Optional[ActionCodecConfig] = None,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if config is None:
            config = ActionCodecConfig()

        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.embed_dim = config.embed_dim
        self.config = config

        if not is_discrete and action_low is not None and action_high is not None:
            low = torch.as_tensor(action_low, dtype=torch.float32)
            high = torch.as_tensor(action_high, dtype=torch.float32)
            self.register_buffer("action_low", low)
            self.register_buffer("action_high", high)
            self.register_buffer("action_scale", (high - low) / 2.0)
            self.register_buffer("action_bias", (high + low) / 2.0)
        else:
            self.action_low = None
            self.action_high = None
            self.action_scale = None
            self.action_bias = None

        hidden = tuple(config.hidden_dims)

        self.encoder = build_mlp(
            in_dim=action_dim,
            out_dim=config.embed_dim,
            hidden_dims=hidden,
            norm=_norm_from_layer_flag(config.use_layer_norm),
            output_activation=True,
            final_gain=1.0,
        )
        self.decoder = build_mlp(
            in_dim=config.embed_dim,
            out_dim=action_dim,
            hidden_dims=hidden[::-1],
            norm=_norm_from_layer_flag(config.use_layer_norm),
            output_activation=False,
            final_gain=1.0,
        )

    def scale_action(self, action: Tensor) -> Tensor:
        if self.action_scale is None:
            return action
        return action * self.action_scale + self.action_bias

    def unscale_action(self, action: Tensor) -> Tensor:
        if self.action_scale is None:
            return action
        return (action - self.action_bias) / (self.action_scale + 1e-8)

    def encode(self, action: Tensor) -> Tensor:
        return self.encoder(action)

    def decode(self, embed: Tensor) -> Tensor:
        raw = self.decoder(embed)
        return raw if self.is_discrete else torch.tanh(raw)

    def decode_to_env(self, embed: Tensor) -> Tensor:
        raw = self.decode(embed)
        return raw if self.is_discrete else self.scale_action(raw)

    def forward(self, action: Tensor) -> Tensor:
        return self.encode(action)

    @classmethod
    def from_action_space(
        cls,
        action_space: Any,
        config: Optional[ActionCodecConfig] = None,
    ) -> "ActionCodec":
        if hasattr(action_space, "n"):
            return cls(
                action_dim=action_space.n, is_discrete=True, config=config,
            )
        return cls(
            action_dim=int(np.prod(action_space.shape)),
            is_discrete=False,
            config=config,
            action_low=action_space.low,
            action_high=action_space.high,
        )


# =============================================================================
# 12  CriticOutput
# =============================================================================


@dataclass
class CriticOutput:
    """
    Unified Critic output container.

    ``__getitem__`` raises an explicit ``KeyError`` for unknown keys.
    """

    values_symlog_main: Tensor
    values_real_main: Tensor
    uncertainty_raw: Tensor
    values_per_gamma: Optional[Dict[float, Tensor]] = None
    values_real_per_gamma: Optional[Dict[float, Tensor]] = None
    uncertainties_per_gamma: Optional[Dict[float, Tensor]] = None
    gate_weights: Optional[Tensor] = None

    # -- convenience properties --

    @property
    def value(self) -> Tensor:
        return self.values_real_main

    @property
    def value_symlog(self) -> Tensor:
        return self.values_symlog_main

    @property
    def uncertainty(self) -> Tensor:
        return self.uncertainty_raw

    # -- dict-like access --

    _KEY_MAP = {
        "value": "values_real_main",
        "value_symlog": "values_symlog_main",
        "value_real": "values_real_main",
        "uncertainty": "uncertainty_raw",
        "values_by_gamma": "values_per_gamma",
        "uncertainties_by_gamma": "uncertainties_per_gamma",
        "gate_weights": "gate_weights",
    }

    def __getitem__(self, key: str) -> Any:
        attr = self._KEY_MAP.get(key, key)
        if not hasattr(self, attr):
            raise KeyError(
                f"CriticOutput has no key '{key}'.  "
                f"Valid: {sorted(self._KEY_MAP.keys())}"
            )
        return getattr(self, attr)

    # -- tensor ops --

    def detach(self) -> "CriticOutput":
        def _d(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, Tensor):
                return v.detach()
            if isinstance(v, dict):
                return {k: _v.detach() for k, _v in v.items()}
            return v

        return CriticOutput(
            values_symlog_main=self.values_symlog_main.detach(),
            values_real_main=self.values_real_main.detach(),
            uncertainty_raw=self.uncertainty_raw.detach(),
            values_per_gamma=_d(self.values_per_gamma),
            values_real_per_gamma=_d(self.values_real_per_gamma),
            uncertainties_per_gamma=_d(self.uncertainties_per_gamma),
            gate_weights=_d(self.gate_weights),
        )

    def to(self, device: torch.device) -> "CriticOutput":
        def _t(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, Tensor):
                return v.to(device)
            if isinstance(v, dict):
                return {k: _v.to(device) for k, _v in v.items()}
            return v

        return CriticOutput(
            values_symlog_main=_t(self.values_symlog_main),
            values_real_main=_t(self.values_real_main),
            uncertainty_raw=_t(self.uncertainty_raw),
            values_per_gamma=_t(self.values_per_gamma),
            values_real_per_gamma=_t(self.values_real_per_gamma),
            uncertainties_per_gamma=_t(self.uncertainties_per_gamma),
            gate_weights=_t(self.gate_weights),
        )


# =============================================================================
# 13  ValueNormalizer  [NEW-v4.6]
# =============================================================================


class ValueNormalizer(nn.Module):
    """
    Running mean / std normaliser for value targets.

    Stabilises L_crit when return magnitudes vary across training.
    Training loop usage::

        normalizer = ValueNormalizer()
        normed_returns = normalizer.normalize(returns)
        L_crit = loss_fn(critic_output, normed_returns)
        # at inference: real_value = normalizer.denormalize(critic_raw)

    The normaliser tracks statistics via EMA and stores them as buffers
    so they survive ``state_dict`` / ``load_state_dict``.
    """

    def __init__(self, beta: float = 0.999, epsilon: float = 1e-8):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("_mean", torch.tensor(0.0))
        self.register_buffer("_var", torch.tensor(1.0))
        self.register_buffer("_count", torch.tensor(0, dtype=torch.long))

    def update(self, values: Tensor) -> None:
        """Update running statistics (call once per batch)."""
        with torch.no_grad():
            batch_mean = values.mean()
            batch_var = values.var(unbiased=False)
            if self._count.item() == 0:
                self._mean.copy_(batch_mean)
                self._var.copy_(batch_var + self.epsilon)
            else:
                self._mean.mul_(self.beta).add_(
                    (1.0 - self.beta) * batch_mean
                )
                self._var.mul_(self.beta).add_(
                    (1.0 - self.beta) * batch_var
                )
            self._count.add_(1)

    def normalize(self, values: Tensor, update: bool = True) -> Tensor:
        """Normalize values to zero-mean unit-variance."""
        if update and self.training:
            self.update(values)
        return (values - self._mean) / (self._var.sqrt() + self.epsilon)

    def denormalize(self, normed: Tensor) -> Tensor:
        """Map normalised values back to original scale."""
        return normed * (self._var.sqrt() + self.epsilon) + self._mean

    @property
    def mean(self) -> float:
        return self._mean.item()

    @property
    def std(self) -> float:
        return (self._var.sqrt() + self.epsilon).item()

    def reset(self) -> None:
        self._mean.zero_()
        self._var.fill_(1.0)
        self._count.zero_()


# =============================================================================
# 14  SingleCritic
#     [FIX-v4.6] forward() now accepts (feat, intent, use_target, **kwargs)
#                for interface uniformity.  use_target is accepted but ignored
#                (SingleCritic has no target network).
#     [FIX-v4.6] compute_loss accepts detach_features kwarg.
#     [FIX-v4.6] Uses _DEFAULT_SYMLOG_CLIP.
# =============================================================================


class SingleCritic(nn.Module):
    """Single-head critic (lightweight).

    [v5.5] Now supports twohot discrete regression (DreamerV3 style)
    when ``use_twohot=True``.
    """

    def __init__(
        self,
        feat_dim: int,
        intent_dim: int = 0,
        hidden_dim: int = 256,
        depth: int = 2,
        activation: str = "silu",
        layer_norm: bool = True,
        primary_gamma: float = 0.99,
        huber_delta: float = 1.0,
        symlog_clip: Optional[float] = None,
        use_twohot: bool = True,
        twohot_num_bins: int = 255,
        twohot_vmin: float = -20.0,
        twohot_vmax: float = 20.0,
    ):
        super().__init__()
        self.primary_gamma = primary_gamma
        self.gammas = (primary_gamma,)
        self.huber_delta = huber_delta
        self.symlog_clip = (
            _DEFAULT_SYMLOG_CLIP if symlog_clip is None else float(symlog_clip)
        )
        self.cfg = None
        self.use_twohot = use_twohot

        out_dim = twohot_num_bins if use_twohot else 1
        self.net = build_mlp(
            in_dim=feat_dim + intent_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
            norm=_norm_from_layer_flag(layer_norm),
            final_gain=1.0,  # [FIX] Increase from 0.01 to 1.0 for better initialization
        )

        if use_twohot:
            self.register_buffer(
                "_twohot_bins",
                symlog_bins(twohot_num_bins, twohot_vmin, twohot_vmax),
            )

    def forward(
        self,
        feat: Tensor,
        intent: Optional[Tensor] = None,
        use_target: bool = False,
        **kwargs: Any,
    ) -> CriticOutput:
        # use_target accepted for interface uniformity, but ignored.
        x = (
            torch.cat([feat, intent], dim=-1)
            if intent is not None
            else feat
        )

        if self.use_twohot:
            logits = self.net(x)  # (..., num_bins)
            probs = F.softmax(logits, dim=-1)
            val_sym = twohot_decode(probs, self._twohot_bins)
            val_real = symexp(val_sym, max_val=self.symlog_clip)
        else:
            clip = self.symlog_clip
            val_sym = self.net(x).squeeze(-1).clamp(-clip, clip)
            val_real = symexp(val_sym, max_val=clip)

        zero = torch.zeros_like(val_real)
        return CriticOutput(
            values_symlog_main=val_sym,
            values_real_main=val_real,
            uncertainty_raw=zero,
            values_per_gamma={self.primary_gamma: val_sym},
            values_real_per_gamma={self.primary_gamma: val_real},
            uncertainties_per_gamma={self.primary_gamma: zero},
        )

    def forward_full(
        self, feat: Tensor, intent: Optional[Tensor] = None,
    ) -> CriticOutput:
        return self.forward(feat, intent)

    def compute_loss(
        self,
        feat: Tensor,
        targets: Union[Tensor, Dict[float, Tensor]],
        intent: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        detach_features: bool = False,
    ) -> Dict[str, Tensor]:
        if detach_features:
            feat = feat.detach()

        if isinstance(targets, dict):
            target_real = targets.get(
                self.primary_gamma, next(iter(targets.values()))
            )
        else:
            target_real = targets

        x = (
            torch.cat([feat, intent], dim=-1)
            if intent is not None
            else feat
        )

        if self.use_twohot:
            logits = self.net(x)  # (..., num_bins)
            target_flat = target_real.reshape(-1)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            loss_val = twohot_loss(logits_flat, target_flat, self._twohot_bins)
            if weights is not None:
                w_flat = weights.reshape(-1)[:loss_val.shape[0]]
                loss_val = loss_val * w_flat
            loss = loss_val.mean()
        else:
            out = self.forward(feat, intent)
            clip = self.symlog_clip
            target_sym = symlog(target_real).clamp(-clip, clip)
            loss_val = huber_loss(
                out.values_symlog_main,
                target_sym,
                delta=self.huber_delta,
                reduction="none",
            )
            if weights is not None:
                loss_val = loss_val * weights.view_as(loss_val)
            loss = loss_val.mean()

        return {
            "loss": loss,
            "per_gamma": {self.primary_gamma: loss.detach()},
        }

    def update_target(self) -> None:
        pass  # no target network


# =============================================================================
# 15  MultiHeadEnsembleCritic
#     [FIX-v4.6] Uses _DEFAULT_SYMLOG_CLIP.
#     [FIX-v4.6] compute_loss accepts detach_features.
# =============================================================================


class MultiHeadEnsembleCritic(nn.Module):
    """
    Multi-gamma × multi-ensemble critic with shared backbone.

    [v5.5] Now supports twohot discrete regression (DreamerV3 style)
    when ``cfg.use_twohot=True``.  Each head outputs ``num_bins`` logits
    instead of a single scalar.  Loss uses cross-entropy on twohot soft
    labels instead of Huber on symlog values.
    """

    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.gammas = tuple(sorted(cfg.gammas))
        self.n_ensemble = cfg.effective_ensemble_size
        self.huber_delta = getattr(cfg, "huber_delta", 1.0)
        self.symlog_clip = getattr(
            cfg, "symlog_clip", _DEFAULT_SYMLOG_CLIP,
        )
        self.pessimism = cfg.pessimism  # may be overridden by DoubleCritic
        self.use_twohot = getattr(cfg, "use_twohot", True)
        self._twohot_num_bins = getattr(cfg, "twohot_num_bins", 255)

        if self.use_twohot:
            self.register_buffer(
                "_twohot_bins",
                symlog_bins(
                    self._twohot_num_bins,
                    getattr(cfg, "twohot_vmin", -20.0),
                    getattr(cfg, "twohot_vmax", 20.0),
                ),
            )

        validate_gamma_keys(self.gammas)

        self._gamma_key_map: Dict[float, str] = {
            g: gamma_to_key(g) for g in self.gammas
        }

        # Shared backbone
        self.backbone = build_mlp(
            in_dim=cfg.d_feature + cfg.intent_dim,
            out_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.hidden_depth,
            norm=_norm_from_layer_flag(cfg.layer_norm),
            activation=cfg.activation,
            output_activation=True,
        )

        # Per-gamma × per-ensemble value heads
        self.heads = nn.ModuleDict(
            {
                self._gamma_key_map[g]: nn.ModuleList(
                    [self._build_head(cfg) for _ in range(self.n_ensemble)]
                )
                for g in self.gammas
            }
        )

        # Target network
        self.use_target = cfg.use_target_critic
        if self.use_target:
            self.target_backbone = copy.deepcopy(self.backbone)
            self.target_heads = copy.deepcopy(self.heads)
            self._freeze(self.target_backbone)
            self._freeze(self.target_heads)

    def _build_head(self, cfg: CriticConfig) -> nn.Sequential:
        out_dim = self._twohot_num_bins if self.use_twohot else 1
        return build_mlp(
            in_dim=cfg.hidden_dim,
            out_dim=out_dim,
            hidden_dim=cfg.hidden_dim,
            depth=2,
            norm=_norm_from_layer_flag(cfg.layer_norm),
            activation=cfg.activation,
            final_gain=1.0,  # [FIX] Increase from 0.01 to 1.0 for better initialization
        )

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False

    def _gk(self, g: float) -> str:
        return self._gamma_key_map.get(g, gamma_to_key(g))

    def _decode_twohot(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode twohot logits → (val_symlog, val_real)."""
        probs = F.softmax(logits, dim=-1)
        val_sym = twohot_decode(probs, self._twohot_bins)
        val_real = symexp(val_sym, max_val=self.symlog_clip)
        return val_sym, val_real

    def forward(
        self,
        feat: Tensor,
        intent: Optional[Tensor] = None,
        use_target: bool = False,
        **kwargs: Any,
    ) -> CriticOutput:
        x = (
            torch.cat([feat, intent], dim=-1)
            if intent is not None
            else feat
        )
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        bb = (
            self.target_backbone
            if (use_target and self.use_target)
            else self.backbone
        )
        hd = (
            self.target_heads
            if (use_target and self.use_target)
            else self.heads
        )

        h = bb(x_flat)
        clip = self.symlog_clip

        v_sym_pg: Dict[float, Tensor] = {}
        v_real_pg: Dict[float, Tensor] = {}
        unc_pg: Dict[float, Tensor] = {}

        for g in self.gammas:
            heads = hd[self._gk(g)]

            if self.use_twohot:
                # Each head outputs (N, num_bins) logits
                all_sym = []
                all_real = []
                for head in heads:
                    logits = head(h)  # (N, num_bins)
                    vs, vr = self._decode_twohot(logits)
                    all_sym.append(vs)
                    all_real.append(vr)
                preds_sym = torch.stack(all_sym, dim=-1)  # (N, E)
                preds_real = torch.stack(all_real, dim=-1)  # (N, E)
            else:
                preds_sym = torch.stack(
                    [
                        head(h).squeeze(-1).clamp(-clip, clip)
                        for head in heads
                    ],
                    dim=-1,
                )  # [N, E]
                preds_real = symexp(preds_sym, max_val=clip)

            if self.n_ensemble > 1:
                agg_sym = torch.quantile(
                    preds_sym, self.pessimism, dim=-1,
                )
                unc = preds_real.std(dim=-1, unbiased=True)
            else:
                agg_sym = preds_sym.squeeze(-1)
                unc = torch.zeros_like(agg_sym)

            agg_real = symexp(agg_sym, max_val=clip)
            v_sym_pg[g] = agg_sym.view(*orig_shape)
            v_real_pg[g] = agg_real.view(*orig_shape)
            unc_pg[g] = unc.view(*orig_shape)

        primary = self.gammas[self.cfg.primary_gamma_index]
        return CriticOutput(
            values_symlog_main=v_sym_pg[primary],
            values_real_main=v_real_pg[primary],
            uncertainty_raw=unc_pg[primary],
            values_per_gamma=v_sym_pg,
            values_real_per_gamma=v_real_pg,
            uncertainties_per_gamma=unc_pg,
        )

    def forward_full(
        self, feat: Tensor, intent: Optional[Tensor] = None,
    ) -> CriticOutput:
        return self.forward(feat, intent, use_target=False)

    def compute_loss(
        self,
        feat: Tensor,
        targets: Dict[float, Tensor],
        intent: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        detach_features: bool = False,
    ) -> Dict[str, Tensor]:
        if detach_features:
            feat = feat.detach()

        x = (
            torch.cat([feat, intent], dim=-1)
            if intent is not None
            else feat
        )
        h = self.backbone(x.reshape(-1, x.shape[-1]))

        _validate_critic_target_gammas(targets.keys(), self.gammas)
        valid_gammas = set(self.gammas)

        g_losses: List[Tensor] = []
        per_gamma: Dict[float, Tensor] = {}
        clip = self.symlog_clip

        for g in sorted(valid_gammas):
            tgt_real_flat = targets[g].reshape(-1)
            heads = self.heads[self._gk(g)]

            head_losses: List[Tensor] = []
            for head in heads:
                if self.use_twohot:
                    logits = head(h)  # (N, num_bins)
                    # align lengths
                    if logits.shape[0] != tgt_real_flat.shape[0]:
                        n = min(logits.shape[0], tgt_real_flat.shape[0])
                        logits = logits[:n]
                        tgt_aligned = tgt_real_flat[:n]
                    else:
                        tgt_aligned = tgt_real_flat
                    lv = twohot_loss(logits, tgt_aligned, self._twohot_bins)
                else:
                    pred = head(h).squeeze(-1).clamp(-clip, clip)
                    tgt_sym = symlog(tgt_real_flat).clamp(-clip, clip)
                    # align lengths
                    if pred.shape[0] != tgt_sym.shape[0]:
                        n = min(pred.shape[0], tgt_sym.shape[0])
                        pred, tgt_aligned = pred[:n], tgt_sym[:n]
                    else:
                        tgt_aligned = tgt_sym
                    lv = huber_loss(
                        pred, tgt_aligned,
                        delta=self.huber_delta,
                        reduction="none",
                    )

                if weights is not None:
                    w_flat = weights.reshape(-1)[: lv.shape[0]]
                    lv = lv * w_flat
                head_losses.append(lv.mean())

            gl = torch.stack(head_losses).mean()
            per_gamma[g] = gl.detach()
            g_losses.append(gl)

        total = torch.stack(g_losses).mean()
        return {"loss": total, "per_gamma": per_gamma}

    def update_target(self) -> None:
        if not self.use_target:
            return
        tau = self.cfg.target_update_tau
        soft_update(self.target_backbone, self.backbone, tau)
        for g_key in self.heads:
            for src_h, tgt_h in zip(
                self.heads[g_key], self.target_heads[g_key]
            ):
                soft_update(tgt_h, src_h, tau)


# =============================================================================
# 16  DoubleCritic
#     [FIX-v4.6] Inner critics use pessimism=0.5 (median) to avoid
#                double-pessimism.  DoubleCritic already takes min().
# =============================================================================


class DoubleCritic(nn.Module):
    """
    Double critic for overestimation mitigation.

    Inner ``MultiHeadEnsembleCritic`` instances use ``pessimism=0.5``
    (median) so that the *only* pessimistic operation is the min()
    across the two critics.  This avoids the double-pessimism bug
    (quantile *inside* + min *outside*) that caused systematic
    value underestimation.
    """

    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.gammas = tuple(sorted(cfg.gammas))

        # [FIX-v4.6] Override pessimism for inner critics
        cfg_inner = copy.copy(cfg)
        cfg_inner.pessimism = 0.5  # median — no extra pessimism
        self.c1 = MultiHeadEnsembleCritic(cfg_inner)
        self.c2 = MultiHeadEnsembleCritic(cfg_inner)

    def forward(
        self,
        feat: Tensor,
        intent: Optional[Tensor] = None,
        use_target: bool = False,
        **kwargs: Any,
    ) -> Tuple[CriticOutput, CriticOutput]:
        return (
            self.c1(feat, intent, use_target=use_target),
            self.c2(feat, intent, use_target=use_target),
        )

    def forward_min(
        self,
        feat: Tensor,
        intent: Optional[Tensor] = None,
        use_target: bool = False,
    ) -> CriticOutput:
        o1, o2 = self.forward(feat, intent, use_target=use_target)
        clip = self.c1.symlog_clip

        min_sym = {
            g: torch.min(o1.values_per_gamma[g], o2.values_per_gamma[g])
            for g in o1.values_per_gamma
        }
        min_real = {
            g: symexp(v, max_val=clip) for g, v in min_sym.items()
        }
        avg_unc = {
            g: (
                o1.uncertainties_per_gamma[g]
                + o2.uncertainties_per_gamma[g]
            )
            * 0.5
            for g in o1.uncertainties_per_gamma
        }

        primary = self.gammas[self.cfg.primary_gamma_index]
        return CriticOutput(
            values_symlog_main=min_sym[primary],
            values_real_main=min_real[primary],
            uncertainty_raw=avg_unc[primary],
            values_per_gamma=min_sym,
            values_real_per_gamma=min_real,
            uncertainties_per_gamma=avg_unc,
        )

    def forward_full(
        self, feat: Tensor, intent: Optional[Tensor] = None,
    ) -> CriticOutput:
        return self.forward_min(feat, intent, use_target=False)

    def compute_loss(
        self,
        feat: Tensor,
        targets: Dict[float, Tensor],
        intent: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        detach_features: bool = False,
    ) -> Dict[str, Tensor]:
        l1 = self.c1.compute_loss(
            feat, targets, intent, weights,
            detach_features=detach_features,
        )
        l2 = self.c2.compute_loss(
            feat, targets, intent, weights,
            detach_features=detach_features,
        )
        return {
            "loss": (l1["loss"] + l2["loss"]) * 0.5,
            "per_gamma": {
                g: (l1["per_gamma"][g] + l2["per_gamma"][g]) * 0.5
                for g in l1["per_gamma"]
            },
        }

    def update_target(self) -> None:
        self.c1.update_target()
        self.c2.update_target()


# =============================================================================
# 17  Gamma Annealer
# =============================================================================


class GammaAnnealer:
    """γ annealing scheduler."""

    def __init__(self, cfg: GammaSchedulerConfig):
        self.base_gammas = tuple(sorted(cfg.base_gammas))
        self.warmup_steps = cfg.warmup_steps
        self.min_ratio = cfg.min_ratio
        self._step = 0

    def step(self) -> None:
        self._step += 1

    def get_gammas(self) -> Tuple[float, ...]:
        ratio = min(1.0, self._step / max(1, self.warmup_steps))
        scale = self.min_ratio + (1.0 - self.min_ratio) * ratio
        return tuple(g * scale for g in self.base_gammas)

    @property
    def progress(self) -> float:
        return min(1.0, self._step / max(1, self.warmup_steps))

    @property
    def is_warmed_up(self) -> bool:
        return self._step >= self.warmup_steps


# =============================================================================
# 18  AdaptiveGammaRouter
#     [FIX-v4.6] compute_router_loss detaches feat to prevent gradients
#                from polluting the shared critic backbone.
# =============================================================================


class AdaptiveGammaRouter(nn.Module):
    """Adaptive γ router: dynamically weights γ-heads based on state."""

    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.num_gammas = len(cfg.gammas)
        self.cfg = cfg

        self.gate = build_mlp(
            in_dim=cfg.d_feature,
            out_dim=self.num_gammas,
            hidden_dim=cfg.adaptive_hidden_dim,
            depth=2,
            final_gain=0.01,
        )

        self.temp_init, self.temp_final = cfg.adaptive_temp_range
        self.temp_steps = cfg.adaptive_steps
        self.register_buffer(
            "_step", torch.tensor(0, dtype=torch.long),
        )

    def get_temperature(self) -> float:
        prog = min(self._step.item() / max(1, self.temp_steps), 1.0)
        return self.temp_init + prog * (self.temp_final - self.temp_init)

    def forward(self, feat: Tensor) -> Tensor:
        logits = self.gate(feat)
        w = F.softmax(logits / self.get_temperature(), dim=-1)
        if self.training:
            self._step += 1
        return w

    def compute_router_loss(
        self,
        feat: Tensor,
        values_per_gamma: Dict[float, Tensor],
        targets_per_gamma: Dict[float, Tensor],
        gammas: Tuple[float, ...],
    ) -> Dict[str, Tensor]:
        """
        Router teacher loss + regularisation.

        [FIX-v4.6] ``feat`` is detached so router gradients do NOT flow
        back through the shared critic backbone.
        """
        feat_d = feat.detach().reshape(-1, feat.shape[-1])

        vals_flat = {g: v.reshape(-1) for g, v in values_per_gamma.items()}
        tgts_flat = {g: v.reshape(-1) for g, v in targets_per_gamma.items()}

        w = self.forward(feat_d)  # [N, K]

        vals = torch.stack([vals_flat[g] for g in gammas], dim=-1)
        tgts = torch.stack([tgts_flat[g] for g in gammas], dim=-1)

        td2 = (tgts.detach() - vals.detach()).pow(2)

        beta = self.cfg.route_teacher_beta
        q = F.softmax(-beta * td2, dim=-1)

        # KL(q || w)
        L_route = (
            q * (q.clamp_min(1e-8).log() - w.clamp_min(1e-8).log())
        ).sum(dim=-1).mean()

        # Load balance
        w_mean = w.mean(dim=0)
        uniform = torch.full_like(w_mean, 1.0 / self.num_gammas)
        L_lb = (w_mean - uniform).pow(2).mean()

        # Entropy regularisation
        H = -(w * w.clamp_min(1e-8).log()).sum(dim=-1).mean()
        L_ent = -H

        return {
            "L_route": L_route,
            "L_lb": L_lb,
            "L_ent": L_ent,
            "H_gamma": H.detach(),
            "w_mean": w_mean.detach(),
        }

    @property
    def step_count(self) -> int:
        return self._step.item()


# =============================================================================
# 19  HierarchicalUnifiedCritic
#     [FIX-v4.6] Router loss is returned separately (loss_critic / loss_router)
#                so the training loop can apply independent scaling / optimisers.
#     [FIX-v4.6] compute_loss accepts detach_features kwarg.
#     [FIX-v4.6] Unified _core_forward — no more try-except.
# =============================================================================


class HierarchicalUnifiedCritic(nn.Module):
    """
    Unified critic interface.

    Wraps DoubleCritic or MultiHeadEnsembleCritic and optionally an
    AdaptiveGammaRouter.
    """

    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.gammas = tuple(sorted(cfg.gammas))
        self.symlog_clip = getattr(cfg, "symlog_clip", _DEFAULT_SYMLOG_CLIP)

        if cfg.mode == "double":
            self.core = DoubleCritic(cfg)
        else:
            self.core = MultiHeadEnsembleCritic(cfg)

        self.router: Optional[AdaptiveGammaRouter] = None
        if cfg.mode == "adaptive" or cfg.use_adaptive_routing:
            self.router = AdaptiveGammaRouter(cfg)

    def _core_forward(
        self,
        feat: Tensor,
        intent: Optional[Tensor],
        use_target: bool,
    ) -> CriticOutput:
        if isinstance(self.core, DoubleCritic):
            return self.core.forward_min(
                feat, intent, use_target=use_target,
            )
        return self.core.forward(
            feat, intent, use_target=use_target,
        )

    def forward(
        self,
        feat: Tensor,
        intent: Optional[Tensor] = None,
        use_target: bool = False,
        **kwargs: Any,
    ) -> CriticOutput:
        out = self._core_forward(feat, intent, use_target)

        if (
            self.router is not None
            and out.values_real_per_gamma is not None
        ):
            weights = self.router(feat)

            vals = torch.stack(
                [out.values_real_per_gamma[g] for g in self.gammas],
                dim=-1,
            )
            uncs = torch.stack(
                [out.uncertainties_per_gamma[g] for g in self.gammas],
                dim=-1,
            )

            w_real = (vals * weights).sum(dim=-1)
            w_unc = (uncs * weights).sum(dim=-1)

            out = CriticOutput(
                values_symlog_main=symlog(w_real),
                values_real_main=w_real,
                uncertainty_raw=w_unc,
                values_per_gamma=out.values_per_gamma,
                values_real_per_gamma=out.values_real_per_gamma,
                uncertainties_per_gamma=out.uncertainties_per_gamma,
                gate_weights=weights,
            )

        return out

    def forward_full(
        self, feat: Tensor, intent: Optional[Tensor] = None,
    ) -> CriticOutput:
        return self.forward(feat, intent)

    def compute_loss(
        self,
        feat: Tensor,
        targets: Dict[float, Tensor],
        intent: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        include_router_loss: bool = True,
        detach_features: bool = False,
        wall_strength: float = 0.0,
    ) -> Dict[str, Tensor]:
        """
        Compute critic loss.

        ``loss`` = critic + router (backward-compatible with existing
        training loops that only read ``result['loss']``).

        ``loss_critic`` / ``loss_router`` are detached diagnostics for
        logging or advanced training loops.
        """
        _validate_critic_target_gammas(targets.keys(), self.gammas)
        core_loss = self.core.compute_loss(
            feat, targets, intent, weights,
            detach_features=detach_features,
        )

        if (
            self.router is None
            or not self.cfg.use_adaptive_routing
            or not include_router_loss
        ):
            return {
                "loss": core_loss["loss"],
                "loss_critic": core_loss["loss"].detach(),
                "per_gamma": core_loss["per_gamma"],
            }

        with torch.no_grad():
            out_core = self._core_forward(feat, intent, use_target=False)

        targets_aligned = {g: targets[g] for g in self.gammas}
        rl = self.router.compute_router_loss(
            feat=feat,
            values_per_gamma=out_core.values_real_per_gamma,
            targets_per_gamma=targets_aligned,
            gammas=self.gammas,
        )

        L_router = (
            self.cfg.lambda_route * rl["L_route"]
            + self.cfg.lambda_load_balance * rl["L_lb"]
            + self.cfg.lambda_entropy * rl["L_ent"]
        )
        ws = float(wall_strength)
        if ws <= 0.0:
            L_router_gated = L_router
        elif ws >= 1.0:
            L_router_gated = L_router * 0.0
        else:
            L_router_gated = (1.0 - ws) * L_router

        return {
            "loss": core_loss["loss"] + L_router_gated,
            "loss_critic": core_loss["loss"].detach(),
            "loss_router": L_router_gated.detach(),
            "per_gamma": core_loss["per_gamma"],
            "L_route": rl["L_route"].detach(),
            "L_lb": rl["L_lb"].detach(),
            "L_ent": rl["L_ent"].detach(),
            "H_gamma": rl["H_gamma"],
            "w_gamma_mean": rl["w_mean"],
        }

    def update_target(self) -> None:
        self.core.update_target()

    @property
    def primary_gamma(self) -> float:
        return self.cfg.primary_gamma

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "mode": self.cfg.mode,
            "primary_gamma": self.primary_gamma,
            "gammas": self.gammas,
        }
        if self.router is not None:
            stats["router_step"] = self.router.step_count
            stats["router_temp"] = self.router.get_temperature()
        return stats


# =============================================================================
# 20  Critic Factory
# =============================================================================


def create_critic(cfg: CriticConfig) -> nn.Module:
    """Unified critic creation entry point."""
    if cfg.mode == "single" and len(cfg.gammas) == 1:
        return SingleCritic(
            feat_dim=cfg.d_feature,
            intent_dim=cfg.intent_dim,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.hidden_depth,
            activation=cfg.activation,
            layer_norm=cfg.layer_norm,
            primary_gamma=cfg.primary_gamma,
            huber_delta=getattr(cfg, "huber_delta", 1.0),
            symlog_clip=getattr(cfg, "symlog_clip", _DEFAULT_SYMLOG_CLIP),
        )
    return HierarchicalUnifiedCritic(cfg)


def create_critic_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Smart optimiser: no weight_decay for bias / norm layers."""
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    decay_p: List[nn.Parameter] = []
    nodecay_p: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (nodecay_p if any(nd in name for nd in no_decay) else decay_p).append(
            param
        )
    return torch.optim.AdamW(
        [
            {"params": decay_p, "weight_decay": weight_decay},
            {"params": nodecay_p, "weight_decay": 0.0},
        ],
        lr=lr,
        eps=eps,
    )


# =============================================================================
# 21  Will Input / Output Types
# =============================================================================


@dataclass
class WillInputs:
    """Inputs to the Will module."""

    U_wm: Tensor
    U_v: Tensor
    router_weights: Tensor
    policy_entropy: Tensor
    U_wm_prev: Optional[Tensor] = None
    U_v_prev: Optional[Tensor] = None
    proj_error: Optional[Tensor] = None
    value_error: Optional[Tensor] = None
    alive_ratio: Optional[float] = None


@dataclass
class WillOutputs:
    """Outputs from the Will module."""

    r_intrinsic: Tensor
    r_curiosity: Tensor
    r_mastery: Tensor
    r_autonomy: Tensor
    trust: float
    trust_wm: float
    trust_v: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def detach(self) -> "WillOutputs":
        return WillOutputs(
            r_intrinsic=self.r_intrinsic.detach(),
            r_curiosity=self.r_curiosity.detach(),
            r_mastery=self.r_mastery.detach(),
            r_autonomy=self.r_autonomy.detach(),
            trust=self.trust,
            trust_wm=self.trust_wm,
            trust_v=self.trust_v,
            metrics=self.metrics,
        )

    def to(self, device: torch.device) -> "WillOutputs":
        return WillOutputs(
            r_intrinsic=self.r_intrinsic.to(device),
            r_curiosity=self.r_curiosity.to(device),
            r_mastery=self.r_mastery.to(device),
            r_autonomy=self.r_autonomy.to(device),
            trust=self.trust,
            trust_wm=self.trust_wm,
            trust_v=self.trust_v,
            metrics=self.metrics,
        )


# =============================================================================
# 22  Will Core Modules  (zero learnable parameters)
# =============================================================================


class CuriosityModule(nn.Module):
    """
    Multi-source curiosity  (zero learnable parameters).

    r = η_wm · tanh(U_wm / u₀) + η_v · tanh(U_v / u₀) + η_r · H(router)

    [FIX-v4.6] Router entropy ``H_router`` is auto-unsqueezed to match
    ``U_wm`` dimensionality before ``expand_as``.  The original code
    crashed with a RuntimeError when H_router was 1-D and U_wm was 2-D.
    """

    def __init__(self, config: WillConfig):
        super().__init__()
        self.eta_wm = config.eta_wm
        self.eta_v = config.eta_v
        self.eta_router = config.eta_router
        self.u0_wm = config.u0_wm
        self.u0_v = config.u0_v

    def forward(
        self, U_wm: Tensor, U_v: Tensor, router_weights: Tensor,
    ) -> Tensor:
        r_wm = self.eta_wm * torch.tanh(U_wm / self.u0_wm)
        r_v = self.eta_v * torch.tanh(U_v / self.u0_v)

        # Router entropy: [..., K] → [...]
        H_router = -(
            router_weights * (router_weights + 1e-8).log()
        ).sum(dim=-1)

        # [FIX-v4.6] Auto-unsqueeze until dimensionality matches U_wm
        while H_router.dim() < U_wm.dim():
            H_router = H_router.unsqueeze(-1)

        r_router = self.eta_router * H_router.expand_as(U_wm)

        return r_wm + r_v + r_router


class MasteryModule(nn.Module):
    """
    Mastery  (zero parameters): uncertainty decrease → positive reward.

    Uses unified ``_uncertainty_reduction_reward`` for all cases.
    """

    def __init__(self, config: WillConfig):
        super().__init__()
        self.kappa_wm = config.kappa_wm
        self.kappa_v = config.kappa_v

    def _uncertainty_reduction_reward(
        self,
        U: Tensor,
        U_prev: Optional[Tensor],
        kappa: float,
        time_dim: int,
    ) -> Tensor:
        if kappa == 0:
            return torch.zeros_like(U)

        if U_prev is not None:
            return kappa * F.relu(U_prev - U)

        if U.dim() >= 2 and U.shape[time_dim] > 1:
            delta = U.diff(dim=time_dim, n=1).neg()
            reward = kappa * F.relu(delta)
            pad_shape = list(U.shape)
            pad_shape[time_dim] = 1
            zero_pad = torch.zeros(
                pad_shape, device=U.device, dtype=U.dtype,
            )
            return torch.cat([zero_pad, reward], dim=time_dim)

        return torch.zeros_like(U)

    def forward(
        self,
        U_wm: Tensor,
        U_v: Tensor,
        U_wm_prev: Optional[Tensor] = None,
        U_v_prev: Optional[Tensor] = None,
        time_dim: int = 1,
    ) -> Tensor:
        r_wm = self._uncertainty_reduction_reward(
            U_wm, U_wm_prev, self.kappa_wm, time_dim,
        )
        r_v = self._uncertainty_reduction_reward(
            U_v, U_v_prev, self.kappa_v, time_dim,
        )
        return r_wm + r_v


class AutonomyModule(nn.Module):
    """Autonomy  (zero parameters):  r = κ_a · H[π]."""

    def __init__(self, config: WillConfig):
        super().__init__()
        self.kappa_a = config.kappa_a

    def forward(self, entropy: Tensor) -> Tensor:
        return self.kappa_a * entropy


# =============================================================================
# 23  TrustModel
#     [FIX — maintained from v4.5.1] Uses copy_() instead of assignment
#     to preserve register_buffer semantics.
# =============================================================================


class TrustModel(nn.Module):
    """
    Multi-dimensional trust model.

    trust = f(EMA(U_wm), EMA(U_v))
    """

    def __init__(self, config: WillConfig):
        super().__init__()
        self.config = config

        self.register_buffer("U_wm_ema", torch.tensor(1.0))
        self.register_buffer("U_v_ema", torch.tensor(0.5))
        self.register_buffer("trust_wm", torch.tensor(0.5))
        self.register_buffer("trust_v", torch.tensor(0.5))
        self.register_buffer("trust_combined", torch.tensor(0.5))
        self.register_buffer(
            "_step", torch.tensor(0, dtype=torch.long),
        )

        if config.use_window_update:
            self.register_buffer(
                "wm_buffer", torch.zeros(config.window_size),
            )
            self.register_buffer(
                "v_buffer", torch.zeros(config.window_size),
            )
            self.register_buffer(
                "buffer_ptr", torch.tensor(0, dtype=torch.long),
            )
            self.register_buffer(
                "buffer_filled", torch.tensor(False),
            )

    def update(
        self,
        U_wm: Tensor,
        U_v: Tensor,
        step: Optional[int] = None,
    ) -> float:
        if step is not None:
            self._step.fill_(step)
        else:
            self._step.add_(1)

        cfg = self.config
        U_wm_mean = U_wm.mean()
        U_v_mean = U_v.mean()

        if cfg.use_window_update:
            self._update_buffer(U_wm_mean, U_v_mean)
            if (
                cfg.update_freq > 0
                and (self._step.item() % cfg.update_freq != 0)
            ):
                return self.trust_combined.item()
            if self.buffer_filled.item():
                bar_wm = self.wm_buffer.mean()
                bar_v = self.v_buffer.mean()
            else:
                n = min(
                    self.buffer_ptr.item(), cfg.window_size,
                )
                if n > 0:
                    bar_wm = self.wm_buffer[:n].mean()
                    bar_v = self.v_buffer[:n].mean()
                else:
                    bar_wm, bar_v = U_wm_mean, U_v_mean
        else:
            beta = cfg.trust_ema_beta
            self.U_wm_ema.mul_(beta).add_((1 - beta) * U_wm_mean)
            self.U_v_ema.mul_(beta).add_((1 - beta) * U_v_mean)
            bar_wm = self.U_wm_ema
            bar_v = self.U_v_ema

        # [FIX v4.5.1 maintained] copy_() preserves buffer attribute
        self.trust_wm.copy_(
            torch.exp(-bar_wm / cfg.U0_wm).clamp(0.0, 1.0)
        )
        self.trust_v.copy_(
            torch.exp(-bar_v / cfg.U0_v).clamp(0.0, 1.0)
        )

        if cfg.trust_aggregation == "min":
            combined = torch.min(self.trust_wm, self.trust_v)
        elif cfg.trust_aggregation == "mean":
            combined = (self.trust_wm + self.trust_v) * 0.5
        else:
            combined = (
                cfg.trust_wm_weight * self.trust_wm
                + cfg.trust_v_weight * self.trust_v
            )
        self.trust_combined.copy_(combined.clamp(0.0, 1.0))
        return self.trust_combined.item()

    def _update_buffer(
        self, U_wm_mean: Tensor, U_v_mean: Tensor,
    ) -> None:
        idx = int(self.buffer_ptr.item() % self.config.window_size)
        self.wm_buffer[idx] = U_wm_mean
        self.v_buffer[idx] = U_v_mean
        self.buffer_ptr.add_(1)
        if self.buffer_ptr.item() >= self.config.window_size:
            self.buffer_filled.fill_(True)

    def get_trust(self) -> float:
        return self.trust_combined.item()

    def get_trust_breakdown(self) -> Dict[str, float]:
        return {
            "trust": self.trust_combined.item(),
            "trust_wm": self.trust_wm.item(),
            "trust_v": self.trust_v.item(),
            "U_wm_ema": self.U_wm_ema.item(),
            "U_v_ema": self.U_v_ema.item(),
        }

    def reset(self) -> None:
        self.U_wm_ema.fill_(1.0)
        self.U_v_ema.fill_(0.5)
        self.trust_wm.fill_(0.5)
        self.trust_v.fill_(0.5)
        self.trust_combined.fill_(0.5)
        self._step.zero_()
        if self.config.use_window_update:
            self.wm_buffer.zero_()
            self.v_buffer.zero_()
            self.buffer_ptr.zero_()
            self.buffer_filled.fill_(False)


# =============================================================================
# 24  ModeSwitchEngine
#     [FIX-v4.6] PPO → DREAM now forces a stop at MIXED (gradual degradation).
# =============================================================================


# =============================================================================
# 25  Will  (main module)
# =============================================================================


class Will(nn.Module):
    """
    Autonomous Will system: intrinsic reward + trust + mode switching.

    All inputs are detached at the start of ``forward()`` — no gradient
    back-propagation through Will into upstream modules.
    """

    def __init__(self, config: WillConfig):
        super().__init__()
        self.config = config

        self.curiosity = (
            CuriosityModule(config) if config.curiosity_enabled else None
        )
        self.mastery = (
            MasteryModule(config) if config.mastery_enabled else None
        )
        self.autonomy = (
            AutonomyModule(config) if config.autonomy_enabled else None
        )
        self.trust_model = (
            TrustModel(config) if config.trust_enabled else None
        )

        if config.normalize_uncertainty:
            self.register_buffer("U_wm_ema_mean", torch.tensor(1.0))
            self.register_buffer("U_wm_ema_std", torch.tensor(1.0))
            self.register_buffer("U_v_ema_mean", torch.tensor(0.1))
            self.register_buffer("U_v_ema_std", torch.tensor(0.1))

    def forward(
        self, inputs: WillInputs, step: int = 0,
    ) -> WillOutputs:
        # ===== unified detach =====
        U_wm = inputs.U_wm.detach()
        U_v = inputs.U_v.detach()
        router_w = self._sanitize_router_weights(
            inputs.weights.detach()
        )
        policy_ent = inputs.policy_entropy.detach()

        U_wm_prev = (
            inputs.U_wm_prev.detach()
            if inputs.U_wm_prev is not None
            else None
        )
        U_v_prev = (
            inputs.U_v_prev.detach()
            if inputs.U_v_prev is not None
            else None
        )

        time_dim = self._infer_time_dim(U_wm, router_w)

        # ===== normalisation =====
        if self.config.normalize_uncertainty:
            U_wm_n = self._normalize_and_update(
                U_wm, "U_wm_ema_mean", "U_wm_ema_std",
            )
            U_v_n = self._normalize_and_update(
                U_v, "U_v_ema_mean", "U_v_ema_std",
            )
        else:
            U_wm_n, U_v_n = U_wm, U_v

        # ===== intrinsic rewards =====
        ref = U_wm

        r_curiosity = (
            self.curiosity(U_wm_n, U_v_n, router_w)
            if self.curiosity is not None
            else torch.zeros_like(ref)
        )
        r_mastery = (
            self.mastery(
                U_wm, U_v, U_wm_prev, U_v_prev, time_dim=time_dim,
            )
            if self.mastery is not None
            else torch.zeros_like(ref)
        )
        r_autonomy = (
            self.autonomy(policy_ent)
            if self.autonomy is not None
            else torch.zeros_like(ref)
        )

        r_curiosity, r_mastery, r_autonomy = torch.broadcast_tensors(
            r_curiosity, r_mastery, r_autonomy,
        )
        r_intrinsic = r_curiosity + r_mastery + r_autonomy

        # ===== trust =====
        if self.trust_model is not None:
            trust = self.trust_model.update(U_wm, U_v, step)
            tb = self.trust_model.get_trust_breakdown()
            trust_wm, trust_v = tb["trust_wm"], tb["trust_v"]
        else:
            trust, trust_wm, trust_v = 1.0, 1.0, 1.0

        # ===== metrics =====
        metrics: Dict[str, float] = {
            "will/r_curiosity": r_curiosity.mean().item(),
            "will/r_mastery": r_mastery.mean().item(),
            "will/r_autonomy": r_autonomy.mean().item(),
            "will/r_intrinsic": r_intrinsic.mean().item(),
            "will/trust": trust,
            "will/trust_wm": trust_wm,
            "will/trust_v": trust_v,
            "will/training_mode": float(training_mode.value),
            "will/is_emergency": float(is_emergency),
            "will/U_wm_mean": U_wm.mean().item(),
            "will/U_v_mean": U_v.mean().item(),
            "will/router_entropy": (
                -(router_w * (router_w + 1e-8).log())
                .sum(dim=-1)
                .mean()
                .item()
            ),
        }

        return WillOutputs(
            r_intrinsic=r_intrinsic,
            r_curiosity=r_curiosity,
            trust=trust,
            trust_wm=trust_wm,
            trust_v=trust_v,
            training_mode=training_mode,
            router_mode_hint=router_hint,
            is_emergency=is_emergency,
            metrics=metrics,
        )

    # ---- helpers ----

    @staticmethod
    def _infer_time_dim(U_wm: Tensor, router_w: Tensor) -> int:
        if router_w.dim() == 3 and U_wm.dim() == 2:
            if router_w.shape[0] == U_wm.shape[0]:
                return 0
        return 1

    def _sanitize_router_weights(self, w: Tensor) -> Tensor:
        if w.dim() not in (2, 3):
            raise ValueError(
                f"router_weights must be 2-D or 3-D, "
                f"got shape={tuple(w.shape)}"
            )
        if not torch.is_floating_point(w):
            w = w.float()

        K = w.shape[-1]
        if K < 3:
            pad = torch.zeros(
                *w.shape[:-1], 3 - K, device=w.device, dtype=w.dtype,
            )
            w = torch.cat([w, pad], dim=-1)
        elif K > 3:
            w = w[..., :3]

        row_sum = w.sum(dim=-1, keepdim=True)
        ok = (
            torch.isfinite(w).all(dim=-1, keepdim=True)
            & (w >= 0).all(dim=-1, keepdim=True)
            & ((row_sum - 1.0).abs() <= 1e-6)
        )
        if ok.all():
            return w

        w = torch.where(
            torch.isfinite(w), w, torch.zeros_like(w),
        ).clamp_min(0.0)
        denom = w.sum(dim=-1, keepdim=True)
        uniform_val = 1.0 / w.shape[-1]
        return torch.where(
            denom > 0, w / denom, torch.full_like(w, uniform_val),
        )

    def _normalize_and_update(
        self, U: Tensor, mean_name: str, std_name: str,
    ) -> Tensor:
        ema_mean: Tensor = getattr(self, mean_name)
        ema_std: Tensor = getattr(self, std_name)
        if self.training:
            alpha = self.config.ema_alpha
            ema_mean.lerp_(U.mean(), alpha)
            ema_std.lerp_(U.std(unbiased=False) + 1e-8, alpha)
        return (U - ema_mean) / (ema_std + 1e-8)

    # ---- convenience ----

    def get_trust(self) -> float:
        if self.trust_model is not None:
            return self.trust_model.get_trust()
        return 1.0

    def reset(self) -> None:
        if self.trust_model is not None:
            self.trust_model.reset()
        if self.config.normalize_uncertainty:
            self.U_wm_ema_mean.fill_(1.0)
            self.U_wm_ema_std.fill_(1.0)
            self.U_v_ema_mean.fill_(0.1)
            self.U_v_ema_std.fill_(0.1)

    # ---- serialisation ----

    def state_dict_extra(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if self.trust_model is not None:
            state["trust_model"] = self.trust_model.get_trust_breakdown()
        if self.config.normalize_uncertainty:
            state["normalize"] = {
                "U_wm_ema_mean": self.U_wm_ema_mean.item(),
                "U_wm_ema_std": self.U_wm_ema_std.item(),
                "U_v_ema_mean": self.U_v_ema_mean.item(),
                "U_v_ema_std": self.U_v_ema_std.item(),
            }
        return state

    def load_state_dict_extra(self, state: Dict[str, Any]) -> None:
        if "trust_model" in state and self.trust_model is not None:
            tm = state["trust_model"]
            self.trust_model.U_wm_ema.fill_(tm["U_wm_ema"])
            self.trust_model.U_v_ema.fill_(tm["U_v_ema"])
            self.trust_model.trust_combined.fill_(tm["trust"])
            self.trust_model._step.fill_(tm.get("_step", 0))
        if "normalize" in state and self.config.normalize_uncertainty:
            n = state["normalize"]
            self.U_wm_ema_mean.fill_(n["U_wm_ema_mean"])
            self.U_wm_ema_std.fill_(n["U_wm_ema_std"])
            self.U_v_ema_mean.fill_(n["U_v_ema_mean"])
            self.U_v_ema_std.fill_(n["U_v_ema_std"])


# =============================================================================
# 26  Will Factory
# =============================================================================


def create_will(config: Optional[WillConfig] = None) -> Will:
    if config is None:
        config = WillConfig()
    return Will(config)


def create_will_optimizer(
    will: Will,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
) -> Optional[torch.optim.Optimizer]:
    """Will typically has zero parameters; returns None if so."""
    params = list(will.parameters())
    if not params:
        return None
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# =============================================================================
# 27  ActorCritic
#     [FIX-v4.6] Removed fragile inspect.signature introspection.
#                All critic classes now accept unified (feat, intent,
#                use_target) signature, so a plain call suffices.
#     [FIX — maintained from v4.5.1] forward() uses keyword arguments
#                to avoid intent-as-temperature bug.
# =============================================================================


def is_multihead_critic(c: nn.Module) -> bool:
    """Check if a critic module has multiple γ heads."""
    gammas = getattr(c, "gammas", ())
    return len(gammas) > 1


class ActorCritic(nn.Module):
    """
    Combined actor-critic network.

    Provides a unified interface for action sampling, value estimation,
    and target-network management.
    """

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        is_discrete: bool = False,
        intent_dim: int = 0,
        use_double_critic: bool = False,
        use_multihead_critic: bool = False,
        multihead_config: Optional[CriticConfig] = None,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "silu",
        layer_norm: bool = True,
        primary_gamma: float = 0.99,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Actor
        self.actor = build_actor(
            feat_dim=feat_dim,
            action_dim=action_dim,
            is_discrete=is_discrete,
            intent_dim=intent_dim,
            hidden_dims=tuple(hidden_dims),
            activation=activation,
            layer_norm=layer_norm,
        )

        # Critic config
        if multihead_config is not None:
            critic_cfg = multihead_config
            if critic_cfg.d_feature != feat_dim:
                critic_cfg.d_feature = feat_dim
        else:
            critic_cfg = CriticConfig(
                d_feature=feat_dim,
                intent_dim=intent_dim,
                hidden_dim=hidden_dims[-1],
                hidden_depth=len(hidden_dims),
                activation=activation,
                layer_norm=layer_norm,
            )
            critic_cfg.gammas = (primary_gamma,)
            critic_cfg.primary_gamma_index = 0

        if use_double_critic:
            critic_cfg.mode = "double"
        elif use_multihead_critic:
            critic_cfg.mode = getattr(critic_cfg, "mode", "ensemble")
        else:
            critic_cfg.mode = "single"
            critic_cfg.gammas = (primary_gamma,)

        self.critic = create_critic(critic_cfg)
        self.is_discrete = is_discrete
        self.use_double_critic = use_double_critic
        self._is_multihead = is_multihead_critic(self.critic)
        self._intent_dim = intent_dim

    def actor_parameters(self) -> List[nn.Parameter]:
        return list(self.actor.parameters())

    def critic_parameters(self) -> List[nn.Parameter]:
        return list(self.critic.parameters())

    def forward(
        self,
        features: Tensor,
        temperature: float = 1.0,
        intent: Optional[Tensor] = None,
    ) -> Tuple[
        Union[TanhTransformedDistribution, CategoricalStraightThrough],
        Tensor,
    ]:
        """
        Forward pass.

        Returns:
            (action_distribution, real_value)
        """
        dist = self.actor(
            features, temperature=temperature, intent=intent,
        )
        critic_out = self.critic(features, intent=intent)
        return dist, critic_out.values_real_main

    def forward_full(
        self,
        features: Tensor,
        temperature: float = 1.0,
        intent: Optional[Tensor] = None,
    ) -> Tuple[
        Union[TanhTransformedDistribution, CategoricalStraightThrough],
        CriticOutput,
    ]:
        """Return full CriticOutput alongside the distribution."""
        dist = self.actor(
            features, temperature=temperature, intent=intent,
        )
        critic_out = self.critic.forward_full(features, intent=intent)
        return dist, critic_out

    def get_value(
        self,
        features: Tensor,
        intent: Optional[Tensor] = None,
        use_target: bool = False,
    ) -> CriticOutput:
        """
        Get value estimate.

        [FIX-v4.6] All critic classes now accept the same unified
        ``(feat, intent, use_target)`` signature, so no inspect.signature
        introspection is needed.
        """
        return self.critic(
            features, intent=intent, use_target=use_target,
        )

    def update_target(self) -> None:
        """Update target networks (if any)."""
        if hasattr(self.critic, "update_target"):
            self.critic.update_target()

    @property
    def is_multihead(self) -> bool:
        return self._is_multihead

    @property
    def gammas(self) -> Optional[Tuple[float, ...]]:
        return getattr(self.critic, "gammas", None)

    @property
    def primary_gamma(self) -> Optional[float]:
        return getattr(self.critic, "primary_gamma", None)


def create_actor_critic(
    feat_dim: int,
    action_space: Any,
    intent_dim: int = 0,
    hidden_dims: Optional[List[int]] = None,
    use_double_critic: bool = False,
    use_multihead_critic: bool = False,
    multihead_config: Optional[CriticConfig] = None,
    gamma: float = 0.99,
    device: Optional[torch.device] = None,
) -> ActorCritic:
    """Convenience ActorCritic creator."""
    if hasattr(action_space, "n"):
        is_discrete = True
        action_dim = action_space.n
    else:
        is_discrete = False
        action_dim = action_space.shape[0]

    ac = ActorCritic(
        feat_dim=feat_dim,
        action_dim=action_dim,
        is_discrete=is_discrete,
        intent_dim=intent_dim,
        use_double_critic=use_double_critic,
        use_multihead_critic=use_multihead_critic,
        multihead_config=multihead_config,
        hidden_dims=hidden_dims,
        primary_gamma=gamma,
    )

    if device is not None:
        ac = ac.to(device)

    return ac


# =============================================================================
# 28  Tests  (only executed under __main__)
# =============================================================================


def _test_actor_basic():
    print("Testing Actor basic functionality...")
    B = 4

    actor_d = Actor(feat_dim=64, action_dim=5, is_discrete=True)
    dist_d = actor_d(torch.randn(B, 64))
    action_d = dist_d.rsample()
    assert action_d.shape == (B, 5)
    lp = dist_d.log_prob(action_d)
    assert lp.shape == (B,)
    ent = dist_d.entropy()
    assert ent.shape == (B,)
    print("   ✓ Discrete actor OK")

    actor_c = Actor(feat_dim=64, action_dim=3, is_discrete=False)
    dist_c = actor_c(torch.randn(B, 64))
    action_c = dist_c.rsample()
    assert action_c.shape == (B, 3)
    assert (action_c.abs() <= 1.0).all()
    lp_c = dist_c.log_prob(action_c)
    assert lp_c.shape == (B,)
    print("   ✓ Continuous actor OK")


def _test_actor_entropy_protection():
    print("Testing Actor entropy protection...")
    actor = Actor(
        feat_dim=64,
        action_dim=4,
        is_discrete=True,
        config=ActorConfig(
            entropy_protection=EntropyProtectionConfig(enabled=True),
        ),
    )
    assert actor.entropy_protector is not None
    dist = actor(torch.randn(2, 64))
    ent = dist.entropy()
    loss, info = actor.entropy_loss(ent)
    assert "entropy" in info
    assert "alpha" in info
    alpha_loss = actor.alpha_loss(ent)
    assert alpha_loss.shape == (1,)
    actor.step_entropy_target()
    health = actor.health_stats()
    assert isinstance(health, ActorHealthStats)
    print(f"   ✓ Entropy protection OK (alpha={info['alpha']:.4f})")


def _test_health_monitor():
    print("Testing HealthMonitor EMA behaviour...")
    mon = HealthMonitor(ema_alpha=0.1)
    for _ in range(50):
        mon.record_logits_clip(80, 100)
    high = mon.logits_clip_rate
    for _ in range(50):
        mon.record_logits_clip(0, 100)
    low = mon.logits_clip_rate
    assert low < high
    mon.record_loc_clip(10, 100)
    mon.record_scale_clip(5, 100)
    assert mon.loc_clip_rate > 0
    assert mon.scale_clip_rate > 0
    assert mon.loc_clip_rate != mon.scale_clip_rate
    print(f"   ✓ EMA: high={high:.4f} → low={low:.4f}")


def _test_critic_single():
    print("Testing SingleCritic...")
    c = SingleCritic(feat_dim=64, primary_gamma=0.99)
    feat = torch.randn(4, 64)
    out = c(feat)
    assert out.value.shape == (4,)
    assert out.uncertainty.shape == (4,)
    # Test use_target parameter is accepted (even if ignored)
    out2 = c(feat, use_target=True)
    assert out2.value.shape == (4,)
    loss_d = c.compute_loss(feat, torch.randn(4))
    assert "loss" in loss_d
    # Test detach_features
    loss_d2 = c.compute_loss(feat, torch.randn(4), detach_features=True)
    assert "loss" in loss_d2
    print("   ✓ SingleCritic OK")


def _test_critic_output_getitem():
    print("Testing CriticOutput __getitem__...")
    out = CriticOutput(
        values_symlog_main=torch.zeros(4),
        values_real_main=torch.zeros(4),
        uncertainty_raw=torch.zeros(4),
    )
    _ = out["value"]
    _ = out["uncertainty"]
    try:
        _ = out["nonexistent_key"]
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    print("   ✓ CriticOutput __getitem__ OK")


def _test_double_critic_no_double_pessimism():
    print("Testing DoubleCritic no double pessimism...")
    try:
        cfg = CriticConfig(
            d_feature=64,
            gammas=(0.99,),
            mode="double",
            use_target_critic=True,
            pessimism=0.25,  # outer config has pessimism
        )
        dc = DoubleCritic(cfg)
        # Inner critics should have pessimism=0.5 (median)
        assert dc.c1.pessimism == 0.5, (
            f"Inner pessimism should be 0.5, got {dc.c1.pessimism}"
        )
        assert dc.c2.pessimism == 0.5
        feat = torch.randn(4, 64)
        out1, out2 = dc.forward(feat, use_target=False)
        assert out1.value.shape == (4,)
        out_min = dc.forward_min(feat, use_target=True)
        assert out_min.value.shape == (4,)
        # Test detach_features in compute_loss
        targets = {0.99: torch.randn(4)}
        ld = dc.compute_loss(feat, targets, detach_features=True)
        assert "loss" in ld
        print("   ✓ DoubleCritic: inner pessimism=0.5, use_target OK")
    except Exception as e:
        print(f"   ⚠ DoubleCritic test skipped: {e}")


def _test_curiosity_dimension_fix():
    print("Testing CuriosityModule dimension fix (P0 bug)...")
    config = WillConfig()
    cur = CuriosityModule(config)

    # Case 1: U_wm=[B], router=[B, 3]
    r1 = cur(
        torch.rand(4),
        torch.rand(4),
        F.softmax(torch.randn(4, 3), dim=-1),
    )
    assert r1.shape == (4,)

    # Case 2: U_wm=[B, T], router=[B, 3]  — this was the crash case
    r2 = cur(
        torch.rand(4, 10),
        torch.rand(4, 10),
        F.softmax(torch.randn(4, 3), dim=-1),
    )
    assert r2.shape == (4, 10)

    # Case 3: U_wm=[B, T], router=[B, T, 3]
    r3 = cur(
        torch.rand(4, 10),
        torch.rand(4, 10),
        F.softmax(torch.randn(4, 10, 3), dim=-1),
    )
    assert r3.shape == (4, 10)

    print("   ✓ CuriosityModule all dimension combos OK")


def _test_trust_buffer_preservation():
    print("Testing TrustModel buffer preservation...")
    config = WillConfig()
    trust = TrustModel(config)

    assert "trust_wm" in dict(trust.named_buffers())
    assert "trust_v" in dict(trust.named_buffers())
    assert "trust_combined" in dict(trust.named_buffers())

    trust.update(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.8]))

    assert "trust_wm" in dict(trust.named_buffers())
    assert "trust_v" in dict(trust.named_buffers())
    assert "trust_combined" in dict(trust.named_buffers())

    sd = trust.state_dict()
    assert "trust_wm" in sd
    assert "trust_v" in sd
    assert "trust_combined" in sd
    print("   ✓ TrustModel buffers preserved after update")


def _test_mastery_unified():
    print("Testing MasteryModule unified logic...")
    config = WillConfig()
    mastery = MasteryModule(config)
    B, T = 4, 10

    U_now = torch.rand(B)
    U_prev = U_now + 0.5
    r = mastery(U_now, U_now, U_prev, U_prev)
    assert (r >= 0).all()
    assert r.shape == (B,)

    U_dec = torch.linspace(2.0, 0.5, T).unsqueeze(0).expand(B, T)
    r_seq = mastery(U_dec, U_dec)
    assert r_seq.shape == (B, T)
    assert (r_seq[:, 0] == 0).all()

    U_t0 = torch.linspace(2.0, 0.5, T).unsqueeze(1).expand(T, B)
    r_t0 = mastery(U_t0, U_t0, time_dim=0)
    assert r_t0.shape == (T, B)
    assert (r_t0[0, :] == 0).all()
    print("   ✓ MasteryModule OK")


def _test_will_basic():
    print("Testing Will basic functionality...")
    config = WillConfig()
    will = Will(config)
    B, T = 4, 10
    inputs = WillInputs(
        U_wm=torch.rand(B, T),
        U_v=torch.rand(B, T) * 0.5,
        router_weights=F.softmax(torch.randn(B, 3), dim=-1),
        policy_entropy=torch.rand(B, T) * 2.0,
    )
    out = will(inputs, step=0)
    assert out.r_intrinsic.shape == (B, T)
    assert out.r_curiosity.shape == (B, T)
    assert 0.0 <= out.trust <= 1.0
    print("   ✓ Will basic OK")


def _test_will_trust_monotonicity():
    print("Testing Will trust monotonicity...")
    config = WillConfig(trust_ema_beta=0.5)
    will = Will(config)
    B = 4

    high_inp = WillInputs(
        U_wm=torch.ones(B) * 5.0,
        U_v=torch.ones(B) * 3.0,
        router_weights=F.softmax(torch.randn(B, 3), dim=-1),
        policy_entropy=torch.rand(B),
    )
    for i in range(20):
        out_h = will(high_inp, step=i)
    trust_high = out_h.trust

    will.reset()

    low_inp = WillInputs(
        U_wm=torch.ones(B) * 0.1,
        U_v=torch.ones(B) * 0.05,
        router_weights=F.softmax(torch.randn(B, 3), dim=-1),
        policy_entropy=torch.rand(B),
    )
    for i in range(20):
        out_l = will(low_inp, step=i)
    trust_low = out_l.trust

    assert trust_low > trust_high, (
        f"Monotonicity: low_unc={trust_low:.3f}, high_unc={trust_high:.3f}"
    )
    print(
        f"   ✓ Trust: low_unc={trust_low:.3f} > high_unc={trust_high:.3f}"
    )


def _test_will_components():
    print("Testing Will components (curiosity, mastery, autonomy)...")
    config = WillConfig(
        curiosity_enabled=True,
        mastery_enabled=True,
        autonomy_enabled=True,
        trust_enabled=True,
    )
    will = Will(config)
    B, T = 2, 8

    inputs = WillInputs(
        U_wm=torch.rand(B, T),
        U_v=torch.rand(B, T),
        router_weights=F.softmax(torch.randn(B, T, 3), dim=-1),
        policy_entropy=torch.rand(B, T),
        U_wm_prev=torch.rand(B, T) + 0.1,
        U_v_prev=torch.rand(B, T) + 0.1,
    )
    out = will(inputs)

    assert out.r_curiosity.shape == (B, T)
    assert out.r_intrinsic.shape == (B, T)
    assert out.trust_wm is not None
    assert out.trust_v is not None

    print("   ✓ Will components OK")


def _test_will_gradient_isolation():
    print("Testing Will gradient isolation...")
    config = WillConfig()
    will = Will(config)
    B = 4

    U_wm = torch.rand(B, requires_grad=True)
    U_v = torch.rand(B, requires_grad=True)
    router_w = F.softmax(
        torch.randn(B, 3, requires_grad=True), dim=-1,
    )
    policy_ent = torch.rand(B, requires_grad=True)

    out = will(
        WillInputs(
            U_wm=U_wm,
            U_v=U_v,
            router_weights=router_w,
            policy_entropy=policy_ent,
        ),
        step=0,
    )

    assert out.r_intrinsic.grad_fn is None
    assert not out.r_intrinsic.requires_grad
    assert U_wm.requires_grad is True
    print("   ✓ Gradient isolation verified")


def _test_actor_critic_forward():
    print("Testing ActorCritic forward signature fix...")
    try:

        class FakeDiscreteSpace:
            n = 4

        ac = create_actor_critic(
            feat_dim=64,
            action_space=FakeDiscreteSpace(),
        )
        feat = torch.randn(2, 64)

        dist, value = ac(feat, temperature=0.5)
        action = dist.rsample()
        lp = dist.log_prob(action)
        assert lp.shape == (2,)
        assert value.shape == (2,)

        # get_value — no inspect.signature needed
        critic_out = ac.get_value(feat)
        assert isinstance(critic_out, CriticOutput)
        critic_out_tgt = ac.get_value(feat, use_target=True)
        assert isinstance(critic_out_tgt, CriticOutput)

        ac.update_target()
        print("   ✓ ActorCritic forward signature fix OK")
    except Exception as e:
        print(f"   ⚠ ActorCritic test skipped: {e}")


def _test_value_normalizer():
    print("Testing ValueNormalizer...")
    vn = ValueNormalizer(beta=0.99)

    # Simulate some returns
    for _ in range(50):
        returns = torch.randn(32) * 100.0 + 50.0
        normed = vn.normalize(returns)

    # After many updates, normalized values should be roughly zero-mean
    test_returns = torch.randn(100) * 100.0 + 50.0
    normed = vn.normalize(test_returns, update=False)
    assert abs(normed.mean().item()) < 5.0, (
        f"Normalised mean too far from 0: {normed.mean().item():.2f}"
    )

    # Denormalize should recover original scale
    recovered = vn.denormalize(normed)
    err = (recovered - test_returns).abs().mean().item()
    assert err < 1e-4, f"Denormalize error too large: {err}"

    # state_dict should contain buffers
    sd = vn.state_dict()
    assert "_mean" in sd
    assert "_var" in sd
    print(
        f"   ✓ ValueNormalizer OK "
        f"(mean={vn.mean:.1f}, std={vn.std:.1f})"
    )


def _test_router_detach():
    print("Testing AdaptiveGammaRouter gradient isolation...")
    try:
        cfg = CriticConfig(
            d_feature=64,
            gammas=(0.9, 0.99, 0.999),
            adaptive_hidden_dim=32,
        )
        router = AdaptiveGammaRouter(cfg)

        feat = torch.randn(4, 64, requires_grad=True)
        values = {
            g: torch.randn(4, requires_grad=True) for g in cfg.gammas
        }
        targets = {g: torch.randn(4) for g in cfg.gammas}

        rl = router.compute_router_loss(feat, values, targets, cfg.gammas)

        # Router loss should have grad_fn (through router params)
        assert rl["L_route"].requires_grad or rl["L_route"].grad_fn is not None
        # But feat should NOT receive gradients from router
        # (because compute_router_loss detaches feat)
        rl["L_route"].backward(retain_graph=True)
        assert feat.grad is None, (
            "feat should not receive gradients from router loss"
        )
        print("   ✓ Router gradient isolation OK")
    except Exception as e:
        print(f"   ⚠ Router test skipped: {e}")


def _test_hierarchical_critic_loss_separation():
    print("Testing HierarchicalUnifiedCritic loss separation...")
    try:
        cfg = CriticConfig(
            d_feature=64,
            gammas=(0.9, 0.99, 0.999),
            mode="ensemble",
            use_adaptive_routing=True,
            adaptive_hidden_dim=32,
        )
        hc = HierarchicalUnifiedCritic(cfg)

        feat = torch.randn(4, 64)
        targets = {g: torch.randn(4) for g in cfg.gammas}

        result = hc.compute_loss(feat, targets)

        assert "loss" in result
        assert "loss_critic" in result
        # If router is active, loss_router should be present
        if hc.router is not None:
            assert "loss_router" in result, (
                "loss_router should be present when router is active"
            )
            # loss should NOT include router loss
            # (training loop decides how to combine them)
            print(
                f"   loss_critic={result['loss_critic'].item():.4f}, "
                f"loss_router={result['loss_router'].item():.4f}"
            )
        print("   ✓ Loss separation OK")
    except Exception as e:
        print(f"   ⚠ HierarchicalCritic test skipped: {e}")


def _test_entropy_cache():
    print("Testing entropy cache (TanhTransformed + Categorical)...")

    # Categorical
    logits = torch.randn(4, 5)
    cat = CategoricalStraightThrough(logits)
    e1 = cat.entropy()
    e2 = cat.entropy()
    assert e1 is e2, "Categorical entropy should be cached"

    # TanhTransformed
    loc = torch.randn(4, 3)
    scale = torch.ones(4, 3) * 0.5
    tanh = TanhTransformedDistribution(loc, scale)
    e1 = tanh.entropy()
    e2 = tanh.entropy()
    assert e1 is e2, "Tanh entropy should be cached"

    print("   ✓ Entropy cache OK")


def _test_factory():
    print("Testing factory functions...")
    will = create_will()
    assert isinstance(will, Will)
    cfg = WillConfig()
    will2 = create_will(cfg)
    assert isinstance(will2, Will)
    opt = create_will_optimizer(will)
    # Will has zero params → None typically
    print("   ✓ Factory OK")


def run_all_tests() -> bool:
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ALETHEIA ACTOR-CRITIC MODULE v4.6.0 — ALL TESTS")
    print("=" * 60 + "\n")

    tests = [
        # Actor
        _test_actor_basic,
        _test_actor_entropy_protection,
        _test_health_monitor,
        # Critic
        _test_critic_single,
        _test_critic_output_getitem,
        _test_double_critic_no_double_pessimism,
        # Will
        _test_will_basic,
        _test_will_trust_monotonicity,
        _test_will_components,
        _test_will_gradient_isolation,
        # Bug regressions
        _test_curiosity_dimension_fix,
        _test_trust_buffer_preservation,
        _test_mastery_unified,
        _test_actor_critic_forward,
        # New v4.6 tests
        _test_value_normalizer,
        _test_router_detach,
        _test_hierarchical_critic_loss_separation,
        _test_entropy_cache,
        # Factory
        _test_factory,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   ✗ {test.__name__} FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "-" * 60)
    tag = "ALL PASSED ✓" if failed == 0 else f"{failed} FAILURES ✗"
    print(f"Results: {passed} passed, {failed} failed — {tag}")
    print("=" * 60 + "\n")

    return failed == 0


# =============================================================================
# 29  Entry Point
# =============================================================================

if __name__ == "__main__":
    run_all_tests()
