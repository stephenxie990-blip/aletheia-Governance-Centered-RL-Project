from __future__ import annotations

import copy
import json
import logging
import shutil
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
import numpy as np
from torch.distributions.kl import kl_divergence as torch_kl_divergence

from .aletheia_config import (
    DEFAULT_STEPS,
    DEFAULT_SEED,
    DEFAULT_ENV,
    DEFAULT_BATCH_LENGTH,
    DEFAULT_HORIZON,
    PRETRAIN_RATIO,
    WARMUP_RATIO,
    LOG_INTERVAL,
    EVAL_INTERVAL,
    SAVE_INTERVAL,
    BUFFER_CAPACITY,
    EVAL_SEED_OFFSET,
    EnvProfile,
    extract_env_profile,
    TrainParams,
    ConfigBundle,
    parse_factory_bridge_overrides,
    RouterConfig,
    is_legacy_compensation_phase_field,
)
from ._checkpoint_metadata import (
    read_agent_creation_overrides_from_checkpoint,
    read_checkpoint,
)
from ._agent_checkpoint_schema import (
    AgentCheckpointRestorePolicy,
    build_agent_checkpoint_payload,
    restore_agent_checkpoint_modules,
)
from ._run_artifact_schema import (
    RunArtifactPaths,
    append_jsonl_record,
    build_eval_record,
    build_resolved_config_payload,
    build_summary_payload,
    build_train_metric_record,
    normalize_eval_record,
    normalize_train_metric_record,
    read_jsonl_records,
    rewrite_jsonl_records,
    trim_history_records_for_resume,
)
# =============================================================================
# Version Information
# =============================================================================

try:
    from .aletheia_foundation import __version__, VERSION_NAME
except ImportError:
    __version__ = "5.0.0"
    VERSION_NAME = "Aletheia Unified API"

logger = logging.getLogger("aletheia.api")


# =============================================================================
# Compilation Helpers
# =============================================================================

def _try_compile(module: nn.Module, mode: str) -> nn.Module:
    """Attempt to compile a module with ``torch.compile``; return original on failure."""
    if not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(module, mode=mode)
    except Exception as exc:
        logger.debug("torch.compile failed for %s: %s", type(module).__name__, exc)
        return module


def _try_script(module: nn.Module) -> nn.Module:
    """Attempt to JIT-script a module; return original on failure."""
    if sys.version_info >= (3, 14):
        return module
    try:
        return torch.jit.script(module)
    except Exception as exc:
        logger.debug("torch.jit.script failed for %s: %s", type(module).__name__, exc)
        return module


class _RouterForwardWrapper(nn.Module):
    """Thin wrapper that delegates to ``router.forward_components`` so the
    module can be compiled / scripted independently."""

    def __init__(self, router: nn.Module):
        super().__init__()
        self.router = router

    def forward(
        self,
        x_rl: torch.Tensor,
        s_ctrl: Optional[torch.Tensor] = None,
        z_task: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        x_proj: Optional[torch.Tensor] = None,
        wall_strength: float = 1.0,
    ):
        return self.router.forward_components(
            x_rl=x_rl,
            s_ctrl=s_ctrl,
            z_task=z_task,
            logvar=logvar,
            x_t=x_t,
            x_proj=x_proj,
            wall_strength=wall_strength,
        )


# =============================================================================
# Path Configuration
# =============================================================================

_package_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_package_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# =============================================================================
# Architecture Constants (Centralized)
# =============================================================================

# RSSM capacity range
RSSM_DETER_MIN: int = 48
RSSM_DETER_MAX: int = 320
RSSM_STOCH_MIN: int = 8
RSSM_STOCH_MAX: int = 32

# Hidden layer dimension range
HIDDEN_DIM_MIN: int = 48
HIDDEN_DIM_MAX: int = 256

# Control dimension range
CTRL_DIM_MIN: int = 32
CTRL_DIM_MAX: int = 128

# Router dimensions
ROUTER_Z_EMBED_DIM: int = 64
ROUTER_PROJ_DIM_MIN: int = 96
ROUTER_PROJ_DIM_MAX: int = 512

# Capacity calculation weights
OBS_COMPLEXITY_WEIGHT: float = 0.6
ACT_COMPLEXITY_WEIGHT: float = 0.4

# Logarithmic calculation constants
LOG10_SCALE: float = 3.0
LOG2_SCALE: float = 5.0
CONTINUOUS_ACT_SCALE: float = 12.0


_ROUTER_OVERRIDE_WHITELIST: frozenset = frozenset({
    "uncertainty_alpha",
    "uncertainty_quantile",
    "uncertainty_ema_decay",
    "uncertainty_hysteresis",
    "uncertainty_window_size",
    "uncertainty_warmup_steps",
    "router_mode",
    "proj_dim",
    "temp_range",
    "anneal_steps",
    "ema_blend_train",
    "ema_blend_eval",
    "mask_x_factor",
    "mask_z_factor",
    "hard_fallback_enabled",
    "hard_fallback_weights_3branch",
    "hard_fallback_weights_2branch",
})

_PERCEPTOR_OVERRIDE_WHITELIST: frozenset = frozenset({
    "vitals_dim",
    "modality_embed_dim",
    "fusion_type",
    "fusion_hidden_dims",
    "image_depth",
    "image_kernels",
    "vector_hidden_dims",
    "vision_backbone",
    "vision_backbone_min_resolution",
    "vision_backbone_patch_size",
    "vision_backbone_min_patches",
    "use_layernorm",
    "activation",
    "dropout",
})


# =============================================================================
# Profile Data Classes
# =============================================================================

@dataclass(frozen=True)
class RSSMProfile:
    """Configuration profile for the RSSM world-model backbone."""
    deter_dim: int
    stoch_dim: int
    num_classes: int
    hidden_dim: int = 512


@dataclass(frozen=True)
class PerceptorProfile:
    """Configuration profile for the perceptor (observation encoder)."""
    type: str
    hidden_dim: int
    depth: int


@dataclass(frozen=True)
class CriticProfile:
    """Configuration profile for the critic network."""
    type: str
    gammas: Tuple[float, ...]
    n_ensemble: int
    hidden_dim: int
    mode: Optional[str] = None
    primary_gamma_index: int = -1
    use_twohot: bool = True
    use_adaptive_routing: bool = False
    pessimism: float = 0.25
    target_update_tau: float = 0.005


@dataclass(frozen=True)
class WillProfile:
    """Configuration profile for the intrinsic-motivation (Will) module."""
    enabled: bool
    version: str = "v4.5"
    curiosity_type: Optional[str] = None
    trust_enabled: bool = False
    mastery_enabled: bool = False
    autonomy_enabled: bool = False


@dataclass(frozen=True)
class TrainProfile:
    """High-level training configuration."""
    algorithm: str
    imagination: bool
    horizon: int
    batch_length: int = 50


@dataclass
class AgentProfile:
    """Composite profile describing a full agent configuration."""
    name: str
    rssm: RSSMProfile
    perceptor: PerceptorProfile
    critic: CriticProfile
    will: WillProfile
    train: TrainProfile
    use_msc: bool = False
    use_nst: bool = False
    use_sc: bool = False
    use_gap_monitor: bool = False

    def __hash__(self) -> int:
        return hash(self.name)


_BASE_PROFILE: AgentProfile = AgentProfile(
    name="Base",
    rssm=RSSMProfile(deter_dim=256, stoch_dim=32, num_classes=16, hidden_dim=256),
    perceptor=PerceptorProfile(type="Vector", hidden_dim=128, depth=2),
    critic=CriticProfile(
        type="multi-gamma",
        gammas=(0.95, 0.99),
        n_ensemble=2,
        hidden_dim=128,
    ),
    will=WillProfile(enabled=False),
    train=TrainProfile(
        algorithm="Dream+PPO",
        imagination=True,
        horizon=10,
        batch_length=32,
    ),
    use_sc=True,
    use_gap_monitor=False,
)


def get_base_profile() -> AgentProfile:
    """Return a deep copy of the default base profile."""
    return copy.deepcopy(_BASE_PROFILE)


# =============================================================================
# Component Registry
# =============================================================================

class ComponentRegistry:
    """Lazy-initialized registry that maps ``(category, name)`` to concrete
    component classes.

    The lazy initialization is *intentional* to avoid circular imports between
    ``aletheia_api`` and sub-modules like ``aletheia_actor_critic``.
    """

    _registry: Dict[str, Dict[str, Type]] = {
        "perceptor": {},
        "rssm": {},
        "critic": {},
        "actor_critic": {},
        "will": {},
        "agent": {},
    }
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        cls._register_standard_components()
        cls._initialized = True

    @classmethod
    def _register_standard_components(cls) -> None:
        try:
            from .aletheia_actor_critic import ActorCritic, Will
            from .aletheia_world_model import UniversalPerceptor, WorldModel

            cls._registry["perceptor"].update({
                "Universal": UniversalPerceptor,
                "Vector": UniversalPerceptor,
                "Vector+": UniversalPerceptor,
            })
            cls._registry["rssm"]["Standard"] = WorldModel
            cls._registry["critic"]["Standard"] = ActorCritic
            cls._registry["actor_critic"]["Standard"] = ActorCritic
            cls._registry["will"]["v4.5"] = Will

        except ImportError as e:
            raise RuntimeError(
                f"ComponentRegistry: failed to import standard components.\n"
                f"Cause: {e}\n"
                "This usually means a missing dependency or a circular import. "
                "Resolve the import error before instantiating the agent."
            ) from e

    @classmethod
    def register(cls, category: str, name: str, component_cls: Type) -> None:
        """Register a component class under *category/name*."""
        if category not in cls._registry:
            cls._registry[category] = {}
        cls._registry[category][name] = component_cls

    @classmethod
    def get(cls, category: str, name: str) -> Type:
        """Retrieve a registered component class.

        .. note:: ``lru_cache`` is intentionally **not** used here so that
           components registered *after* the first ``get`` call are visible.
        """
        cls._ensure_initialized()

        cat = cls._registry.get(category)
        if cat is None:
            raise ValueError(f"Category '{category}' not found")
        comp = cat.get(name)
        if comp is None:
            raise ValueError(
                f"Component '{name}' in '{category}' not found. "
                f"Available: {list(cat.keys())}"
            )
        return comp

    @classmethod
    def list_components(cls, category: str) -> List[str]:
        """Return the names of all components in *category*."""
        cls._ensure_initialized()
        return list(cls._registry.get(category, {}).keys())


# =============================================================================
# Agent Container
# =============================================================================

class AgentContainer:
    """Thin container that groups world-model, actor-critic, router, and
    optional modules into a single unit with convenience methods for
    ``to`` / ``train`` / ``eval``."""

    __slots__ = (
        "world_model",
        "actor_critic",
        "perceptor",
        "router_system",
        "will",
        "meta_will",
        "_device",
        "_components",
        "env_profile",
    )

    def __init__(
        self,
        world_model: Any,
        actor_critic: Any,
        perceptor: Optional[Any],
        router_system: Any,
        will: Optional[Any] = None,
        meta_will: Optional[Any] = None,
        env_profile: Optional[Any] = None,
    ):
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.perceptor = perceptor
        self.router_system = router_system
        self.will = will
        self.meta_will = meta_will
        self.env_profile = env_profile
        self._device: Optional[torch.device] = None

        self._components: Tuple[Any, ...] = tuple(
            c
            for c in (world_model, actor_critic, router_system, perceptor, will, meta_will)
            if c is not None
        )

    # -- Device / mode helpers ------------------------------------------------

    def to(self, device: torch.device) -> "AgentContainer":
        self._device = device
        for component in self._components:
            component.to(device)
        return self

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def train(self, mode: bool = True) -> "AgentContainer":
        for component in self._components:
            component.train(mode)
        return self

    def eval(self) -> "AgentContainer":
        return self.train(False)

    def parameters(self) -> Iterator[nn.Parameter]:
        from itertools import chain

        return chain.from_iterable(c.parameters() for c in self._components)

    # -- Environment profile accessors ----------------------------------------

    def get_env_category(self) -> Optional[str]:
        if self.env_profile is None:
            return None
        return getattr(self.env_profile, "env_category", None)

    def get_env_profile_info(self) -> Dict[str, Any]:
        if self.env_profile is None:
            return {"available": False}

        return {
            "available": True,
            "env_id": getattr(self.env_profile, "env_id", "unknown"),
            "env_category": getattr(self.env_profile, "env_category", "unknown"),
            "obs_modality": getattr(self.env_profile, "obs_modality", "unknown"),
            "obs_dim": getattr(self.env_profile, "obs_dim", 0),
            "action_dim": getattr(self.env_profile, "action_dim", 0),
            "is_discrete_action": getattr(self.env_profile, "is_discrete_action", True),
        }


# =============================================================================
# Agent Factory
# =============================================================================

class AgentFactory:
    """Builds a complete :class:`AgentContainer` from observation/action spaces."""

    MIN_DETER: int = RSSM_DETER_MIN
    MAX_DETER: int = RSSM_DETER_MAX
    MIN_STOCH: int = RSSM_STOCH_MIN
    MAX_STOCH: int = RSSM_STOCH_MAX

    # -- Space inspection helpers ---------------------------------------------

    @staticmethod
    @lru_cache(maxsize=128)
    def _detect_env_type_cached(shape: Tuple[int, ...]) -> str:
        if len(shape) == 1:
            return "vector"
        if len(shape) == 3:
            return "image"
        return "hybrid"

    @staticmethod
    def _detect_env_type(obs_space: Any) -> str:
        shape = getattr(obs_space, "shape", None)
        if shape is not None:
            return AgentFactory._detect_env_type_cached(tuple(shape))
        return "hybrid"

    @staticmethod
    def _is_vector_obs_space(space: Any) -> bool:
        shape = getattr(space, "shape", None)
        return shape is not None and len(shape) == 1

    @staticmethod
    def _infer_vector_obs_dim(space: Any) -> Optional[int]:
        shape = getattr(space, "shape", None)
        if shape is not None and len(shape) == 1:
            return int(shape[0])
        return None

    @staticmethod
    def _infer_action_info(action_space: Any) -> Tuple[int, bool]:
        """Return ``(action_dim, is_discrete)`` for the given space."""
        # MultiDiscrete
        nvec = getattr(action_space, "nvec", None)
        if nvec is not None:
            nvec_arr = np.asarray(nvec, dtype=np.int64)
            if nvec_arr.size == 0:
                raise TypeError(
                    f"Unsupported action space (empty nvec): {action_space}"
                )
            return int(np.prod(nvec_arr)), True

        # Discrete
        n = getattr(action_space, "n", None)
        shape = getattr(action_space, "shape", None)
        if n is not None and (shape is None or shape == () or len(shape) == 0):
            return int(n), True

        # Continuous (Box)
        if shape is not None and len(shape) == 1:
            return int(shape[0]), False

        raise TypeError(
            f"Unsupported action space type: {type(action_space)}. "
            "Supported: Discrete(n), Box(shape=(n,)), MultiDiscrete(nvec)."
        )

    # -- Adaptive capacity computation ----------------------------------------

    @classmethod
    def _compute_adaptive_capacity(
        cls,
        obs_space: Any,
        action_space: Any,
        profile: AgentProfile,
        *,
        adaptive_enabled: bool = True,
        min_deter: int = RSSM_DETER_MIN,
        max_deter: int = RSSM_DETER_MAX,
        min_stoch: int = RSSM_STOCH_MIN,
        max_stoch: int = RSSM_STOCH_MAX,
        respect_profile_minimum: bool = True,
    ) -> Tuple[int, int, int, int, float, float]:
        """Compute adaptive model dimensions based on env complexity.

        Returns
        -------
        deter_dim, num_distributions, hidden_dim, ctrl_dim, free_nats,
        initial_log_alpha
        """
        defaults = (
            profile.rssm.deter_dim,
            profile.rssm.stoch_dim,
            profile.rssm.hidden_dim,
            128,
            1.0,
            -2.0,
        )

        if not adaptive_enabled:
            return defaults

        env_type = cls._detect_env_type(obs_space)
        if env_type != "vector":
            return defaults

        # Compute complexity score from observation and action spaces
        obs_shape = getattr(obs_space, "shape", (1,))
        o_dim = 1
        for d in obs_shape:
            o_dim *= d

        obs_score = min(1.0, math.log10(o_dim + 1) / LOG10_SCALE)

        n = getattr(action_space, "n", None)
        if n is not None:
            act_score = min(1.0, math.log2(max(n, 2)) / LOG2_SCALE)
        else:
            act_shape = getattr(action_space, "shape", (1,))
            a_dim = 1
            for d in act_shape:
                a_dim *= d
            act_score = min(1.0, a_dim / CONTINUOUS_ACT_SCALE)

        total_score = OBS_COMPLEXITY_WEIGHT * obs_score + ACT_COMPLEXITY_WEIGHT * act_score

        # Map score to dimension ranges (align to 8)
        deter_range = max(0, max_deter - min_deter)
        stoch_range = max(0, max_stoch - min_stoch)

        deter = int(min_deter + total_score * deter_range) & ~7
        deter = max(min_deter, min(max_deter, deter))

        num_dist = int(min_stoch + total_score * stoch_range)
        num_dist = max(min_stoch, min(max_stoch, num_dist))

        if respect_profile_minimum:
            deter = max(deter, min(profile.rssm.deter_dim, max_deter))
            num_dist = max(num_dist, min(profile.rssm.stoch_dim, max_stoch))

        hidden = int(HIDDEN_DIM_MIN + total_score * (HIDDEN_DIM_MAX - HIDDEN_DIM_MIN)) & ~7
        hidden = max(HIDDEN_DIM_MIN, min(HIDDEN_DIM_MAX, hidden))

        ctrl = int(CTRL_DIM_MIN + total_score * (CTRL_DIM_MAX - CTRL_DIM_MIN)) & ~7
        ctrl = max(CTRL_DIM_MIN, min(CTRL_DIM_MAX, ctrl))

        free_nats = 1.0
        if total_score < 0.3:
            initial_log_alpha = -2.0
        elif total_score < 0.6:
            initial_log_alpha = -1.0
        else:
            initial_log_alpha = 0.0

        return deter, num_dist, hidden, ctrl, free_nats, initial_log_alpha

    # -- Override validation --------------------------------------------------

    @classmethod
    def _validate_overrides(
        cls,
        overrides: Dict[str, Any],
        whitelist: frozenset,
        component_name: str,
    ) -> Dict[str, Any]:
        valid: Dict[str, Any] = {}
        invalid: List[str] = []

        for key, value in overrides.items():
            if key in whitelist:
                valid[key] = value
            else:
                invalid.append(key)

        if invalid:
            raise ValueError(
                "[%s] Unknown override keys: %s. Valid: %s"
                % (
                    component_name,
                    invalid,
                    sorted(whitelist),
                )
            )

        return valid

    # -- Component constructors -----------------------------------------------

    @classmethod
    def _create_perceptor(
        cls,
        profile: AgentProfile,
        obs_space: Any,
        vitals_dim: int,
        device: torch.device,
        perceptor_overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        from .aletheia_foundation import PerceptorConfig

        perceptor_type = profile.perceptor.type.split("+")[0]
        PerceptorCls = ComponentRegistry.get("perceptor", perceptor_type)

        depth = max(1, profile.perceptor.depth)
        hidden_dims = [profile.perceptor.hidden_dim] * depth

        config_kwargs: Dict[str, Any] = {
            "vitals_dim": vitals_dim,
            "vector_hidden_dims": hidden_dims,
        }

        if perceptor_overrides:
            validated = cls._validate_overrides(
                perceptor_overrides,
                _PERCEPTOR_OVERRIDE_WHITELIST,
                "Perceptor",
            )
            config_kwargs.update(validated)
            if validated:
                logger.info("[Perceptor] Applied overrides: %s", list(validated.keys()))

        perceptor_config = PerceptorConfig(**config_kwargs)
        return PerceptorCls.from_obs_space(obs_space, perceptor_config).to(device)

    @classmethod
    def _create_will(
        cls,
        profile: AgentProfile,
        device: torch.device,
    ) -> Optional[Any]:
        if not profile.will.enabled:
            return None

        from .aletheia_foundation import WillConfig

        WillCls = ComponentRegistry.get("will", profile.will.version)
        will_config = WillConfig(
            curiosity_enabled=(profile.will.curiosity_type is not None),
            trust_enabled=profile.will.trust_enabled,
            mastery_enabled=profile.will.mastery_enabled,
            autonomy_enabled=profile.will.autonomy_enabled,
        )
        return WillCls(will_config).to(device)

    @classmethod
    def _create_world_model(
        cls,
        profile: AgentProfile,
        deter_dim: int,
        num_distributions: int,
        vitals_dim: int,
        obs_space: Any,
        action_space: Any,
        perceptor: Optional[Any],
        device: torch.device,
        rssm_overrides: Optional[Dict[str, Any]] = None,
        adaptive_free_nats: float = 1.0,
        adaptive_ctrl_dim: int = 128,
    ) -> Any:
        from .aletheia_foundation import (
            ContinueConfig,
            DistributionConfig,
            MHCConfig,
            MSCConfig,
            NSTConfig,
            RSSMConfig,
            KLConfig,
            SCConfig,
            WorldModelConfig,
            ControlConfig,
        )
        from .aletheia_world_model import WorldModel

        # Parse config overrides
        hidden_dim = profile.rssm.hidden_dim
        num_classes = profile.rssm.num_classes
        ctrl_dim = adaptive_ctrl_dim
        free_nats = adaptive_free_nats
        continue_defaults = getattr(profile.rssm, "continue_config", ContinueConfig())
        continue_kwargs: Dict[str, Any] = {
            "temperature": float(getattr(continue_defaults, "temperature", 1.0)),
            "loss_type": str(getattr(continue_defaults, "loss_type", "bce")),
            "focal_alpha": float(getattr(continue_defaults, "focal_alpha", 0.25)),
            "focal_gamma": float(getattr(continue_defaults, "focal_gamma", 2.0)),
            "optimistic_bias": float(getattr(continue_defaults, "optimistic_bias", 5.0)),
            "positive_weight": float(getattr(continue_defaults, "positive_weight", 1.0)),
            "negative_weight": float(getattr(continue_defaults, "negative_weight", 1.0)),
        }
        default_hidden_dims = getattr(continue_defaults, "hidden_dims", None)
        if default_hidden_dims:
            continue_kwargs["hidden_dims"] = list(default_hidden_dims)

        if rssm_overrides:
            hidden_dim = rssm_overrides.get("hidden_dim", hidden_dim)
            num_classes = rssm_overrides.get("num_classes", num_classes)
            ctrl_dim = rssm_overrides.get("ctrl_dim", ctrl_dim)

            continue_override_map = {
                "continue_temperature": ("temperature", float),
                "continue_loss_type": ("loss_type", str),
                "continue_focal_alpha": ("focal_alpha", float),
                "continue_focal_gamma": ("focal_gamma", float),
                "continue_optimistic_bias": ("optimistic_bias", float),
                "continue_positive_weight": ("positive_weight", float),
                "continue_negative_weight": ("negative_weight", float),
            }
            for key, (target_key, caster) in continue_override_map.items():
                if key in rssm_overrides:
                    continue_kwargs[target_key] = caster(rssm_overrides[key])
            continue_hidden_dims = rssm_overrides.get("continue_hidden_dims")
            if continue_hidden_dims is not None:
                continue_kwargs["hidden_dims"] = list(continue_hidden_dims)

            # Safely parse nested KL config
            kl_config = rssm_overrides.get("kl", {})
            if isinstance(kl_config, dict):
                free_nats = kl_config.get("free_nats", free_nats)

        def _coerce_optional_config(
            value: Any,
            config_cls: Type,
            *,
            aliases: Optional[Dict[str, str]] = None,
        ) -> Any:
            if value is None:
                return None
            if isinstance(value, config_cls):
                return copy.deepcopy(value)
            if isinstance(value, dict):
                kwargs = dict(value)
            elif hasattr(value, "__dict__"):
                kwargs = dict(vars(value))
            else:
                return value
            if aliases:
                for src, dst in aliases.items():
                    if src in kwargs and dst not in kwargs:
                        kwargs[dst] = kwargs.pop(src)
            return config_cls(**kwargs)

        mhc_config = _coerce_optional_config(
            rssm_overrides.get("mhc") if rssm_overrides else None,
            MHCConfig,
        )
        msc_config = _coerce_optional_config(
            rssm_overrides.get("msc") if rssm_overrides else None,
            MSCConfig,
        )
        nst_config = _coerce_optional_config(
            rssm_overrides.get("nst") if rssm_overrides else None,
            NSTConfig,
            aliases={"horizons": "n_steps"},
        )
        sc_config = _coerce_optional_config(
            rssm_overrides.get("shortcut_consistency") if rssm_overrides else None,
            SCConfig,
            aliases={"shortcut_horizons": "horizons"},
        )

        # Distribution config
        rssm_distribution = DistributionConfig(
            dist_type="categorical",
            num_distributions=num_distributions,
            num_classes=num_classes,
        )

        z_embed_dim = max(32, min(256, deter_dim >> 2))

        rssm_config = RSSMConfig(
            vitals_dim=vitals_dim,
            deter_dim=deter_dim,
            hidden_dim=hidden_dim,
            obs_embed_dim=hidden_dim,
            z_embed_dim=z_embed_dim,
            distribution=rssm_distribution,
            kl=KLConfig(free_nats=free_nats),
            continue_config=ContinueConfig(**continue_kwargs),
            mhc=mhc_config,
            msc=msc_config,
            nst=nst_config,
            shortcut_consistency=sc_config,
        )

        wm_config = WorldModelConfig(
            obs_embed_dim=hidden_dim,
            deter_dim=deter_dim,
            stoch_dim=num_distributions * num_classes,
            action_dim=1,
            num_classes=num_classes,
            control=ControlConfig(
                mode="static",
                ctrl_dim=ctrl_dim,
                hidden_dims=(hidden_dim,),
            ),
        )

        world_model = WorldModel(rssm_config, wm_config=wm_config).to(device)
        world_model.setup_action_space(action_space)

        if perceptor is not None:
            world_model.setup_perceptor(obs_space=obs_space, perceptor=perceptor)

        return world_model

    @classmethod
    def _create_router(
        cls,
        rssm_hidden_dim: int,
        ctrl_dim: int,
        device: torch.device,
        hard_fallback_enabled: bool = False,
        router_overrides: Optional[Dict[str, Any]] = None,
    ) -> Any:
        from .aletheia_world_model import create_router

        router_defaults = RouterConfig()
        adaptive_proj_dim = max(
            ROUTER_PROJ_DIM_MIN,
            min(ROUTER_PROJ_DIM_MAX, rssm_hidden_dim * 2),
        )
        base_kwargs: Dict[str, Any] = {
            "d_x": rssm_hidden_dim,
            "d_c": ctrl_dim,
            "d_z": ROUTER_Z_EMBED_DIM,
            "mode": "attention",
            "proj_dim": adaptive_proj_dim,
            "hard_fallback_enabled": hard_fallback_enabled,
            "uncertainty_alpha": router_defaults.uncertainty_alpha,
            "uncertainty_quantile": router_defaults.uncertainty_quantile,
            "uncertainty_ema_decay": router_defaults.uncertainty_ema_decay,
            "uncertainty_hysteresis": router_defaults.uncertainty_hysteresis,
            "uncertainty_window_size": router_defaults.uncertainty_window_size,
            "uncertainty_warmup_steps": router_defaults.uncertainty_warmup_steps,
            "temp_range": router_defaults.temp_range,
            "anneal_steps": router_defaults.anneal_steps,
            "ema_blend_train": router_defaults.ema_blend_train,
            "ema_blend_eval": router_defaults.ema_blend_eval,
            "mask_x_factor": router_defaults.mask_x_factor,
            "mask_z_factor": router_defaults.mask_z_factor,
            "hard_fallback_weights": router_defaults.hard_fallback_weights,
            "hard_fallback_weights_3branch": router_defaults.hard_fallback_weights_3branch,
            "hard_fallback_weights_2branch": router_defaults.hard_fallback_weights_2branch,
        }

        if router_overrides:
            validated = cls._validate_overrides(
                router_overrides,
                _ROUTER_OVERRIDE_WHITELIST,
                "Router",
            )

            if "router_mode" in validated:
                validated["mode"] = validated.pop("router_mode")

            base_kwargs.update(validated)
            if validated:
                logger.info("[Router] Applied overrides: %s", list(validated.keys()))

        return create_router(**base_kwargs).to(device)

    @classmethod
    def _create_actor_critic(
        cls,
        profile: AgentProfile,
        feat_dim: int,
        action_dim: int,
        is_discrete: bool,
        device: torch.device,
        adaptive_hidden_dim: Optional[int] = None,
        adaptive_initial_log_alpha: Optional[float] = None,
    ) -> Any:
        # Try multiple registry categories for ActorCritic
        ACCls = None
        for category in ("actor_critic", "critic", "agent"):
            try:
                ACCls = ComponentRegistry.get(category, "Standard")
                break
            except ValueError:
                continue

        if ACCls is None:
            raise RuntimeError(
                "ActorCritic component not found in registry. "
                "Ensure ComponentRegistry is properly initialized."
            )

        hidden = (
            adaptive_hidden_dim if adaptive_hidden_dim is not None else profile.critic.hidden_dim
        )

        gammas = tuple(getattr(profile.critic, "gammas", (0.99,))) or (0.99,)
        gammas = tuple(sorted(float(g) for g in gammas))
        primary_gamma = float(gammas[-1])
        explicit_mode = getattr(profile.critic, "mode", None)
        explicit_mode = str(explicit_mode).lower() if explicit_mode is not None else None

        critic_type = str(getattr(profile.critic, "type", "")).lower()
        if explicit_mode is not None:
            use_double_critic = explicit_mode == "double"
            use_multihead_critic = explicit_mode in ("double", "ensemble", "adaptive")
        else:
            use_double_critic = "double" in critic_type
            use_multihead_critic = (
                len(gammas) > 1
                or int(getattr(profile.critic, "n_ensemble", 1)) > 1
                or "multi" in critic_type
            )

        multihead_config = None
        if use_multihead_critic or use_double_critic:
            from .aletheia_foundation import CriticConfig

            mode = explicit_mode or ("double" if use_double_critic else "ensemble")
            if explicit_mode is None and "adaptive" in critic_type:
                mode = "adaptive"

            multihead_config = CriticConfig(
                d_feature=int(feat_dim),
                intent_dim=0,
                mode=mode,
                gammas=gammas,
                n_ensemble=int(getattr(profile.critic, "n_ensemble", 1)),
                hidden_dim=int(hidden),
                hidden_depth=2,
                activation="silu",
                layer_norm=True,
                primary_gamma_index=int(getattr(profile.critic, "primary_gamma_index", -1)),
                use_target_critic=True,
                target_update_tau=float(getattr(profile.critic, "target_update_tau", 0.005)),
                use_twohot=bool(getattr(profile.critic, "use_twohot", True)),
                use_adaptive_routing=bool(
                    getattr(profile.critic, "use_adaptive_routing", False)
                ),
                pessimism=float(getattr(profile.critic, "pessimism", 0.25)),
            )

        ac = ACCls(
            feat_dim=feat_dim,
            action_dim=action_dim,
            is_discrete=is_discrete,
            hidden_dims=[hidden, hidden],
            use_double_critic=use_double_critic,
            use_multihead_critic=use_multihead_critic and not use_double_critic,
            multihead_config=multihead_config,
            primary_gamma=primary_gamma,
        ).to(device)

        # Optionally initialize EntropyProtector log_alpha
        if adaptive_initial_log_alpha is not None:
            actor = getattr(ac, "actor", None)
            if actor is not None:
                ep = getattr(actor, "entropy_protector", None)
                if ep is not None and hasattr(ep, "log_alpha"):
                    with torch.no_grad():
                        ep.log_alpha.fill_(adaptive_initial_log_alpha)
                    logger.debug(
                        "Set EntropyProtector.initial_log_alpha = %s",
                        adaptive_initial_log_alpha,
                    )

        return ac

    # -- Main entry point -----------------------------------------------------

    @classmethod
    def create_agent(
        cls,
        env_obs_space: Any,
        env_action_space: Any,
        device: torch.device,
        custom_overrides: Optional[Dict[str, Any]] = None,
    ) -> AgentContainer:
        """Build a complete :class:`AgentContainer` from raw spaces."""

        bridge_overrides = parse_factory_bridge_overrides(custom_overrides)
        cleaned_overrides = bridge_overrides.custom_overrides
        router_overrides = bridge_overrides.router_overrides
        perceptor_overrides = bridge_overrides.perceptor_overrides
        bridge_env_profile = bridge_overrides.bridge_env_profile

        profile = get_base_profile()

        # Parse ``rssm_*`` prefixed overrides
        rssm_overrides: Dict[str, Any] = {}
        if cleaned_overrides:
            keys_to_remove: List[str] = []
            for key in cleaned_overrides:
                if key.startswith("rssm_"):
                    if key == "rssm_kl_free_nats":
                        rssm_overrides.setdefault("kl", {})["free_nats"] = cleaned_overrides[key]
                    else:
                        rssm_overrides[key[5:]] = cleaned_overrides[key]
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del cleaned_overrides[key]

        # Apply remaining overrides to profile attributes
        if cleaned_overrides:
            critic_field_map = {
                "critic_type": "type",
                "critic_mode": "mode",
                "critic_gammas": "gammas",
                "critic_n_ensemble": "n_ensemble",
                "critic_hidden_dim": "hidden_dim",
                "critic_primary_gamma_index": "primary_gamma_index",
                "critic_use_twohot": "use_twohot",
                "critic_use_adaptive_routing": "use_adaptive_routing",
                "critic_pessimism": "pessimism",
                "critic_target_update_tau": "target_update_tau",
            }
            capacity_tuning_keys = {
                "adaptive_capacity",
                "min_deter",
                "max_deter",
                "min_stoch",
                "max_stoch",
                "respect_profile_minimum",
                "hard_fallback_enabled",
            }
            compatibility_override_keys = {
                "curiosity_enabled",
                "mastery_enabled",
            }
            unknown_override_keys: List[str] = []
            for key, value in cleaned_overrides.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
                elif key in critic_field_map:
                    critic_attr = critic_field_map[key]
                    if critic_attr == "gammas":
                        value = tuple(sorted(float(g) for g in value))
                    elif critic_attr in {"n_ensemble", "hidden_dim", "primary_gamma_index"}:
                        value = int(value)
                    elif critic_attr in {"use_twohot", "use_adaptive_routing"}:
                        value = bool(value)
                    elif critic_attr in {"pessimism", "target_update_tau"}:
                        value = float(value)
                    elif critic_attr == "mode":
                        value = str(value).lower()
                    profile.critic = replace(profile.critic, **{critic_attr: value})
                elif hasattr(profile.will, key):
                    profile.will = replace(profile.will, **{key: value})
                elif key in capacity_tuning_keys:
                    continue
                elif key in compatibility_override_keys:
                    continue
                else:
                    unknown_override_keys.append(key)
            if unknown_override_keys:
                raise ValueError(
                    "Unknown create_agent overrides: "
                    + ", ".join(sorted(unknown_override_keys))
                )

        # Read capacity tuning knobs
        adaptive_enabled = True
        min_deter = RSSM_DETER_MIN
        max_deter = RSSM_DETER_MAX
        min_stoch = RSSM_STOCH_MIN
        max_stoch = RSSM_STOCH_MAX
        respect_profile_minimum = False
        hard_fallback_enabled = False

        if cleaned_overrides:
            adaptive_enabled = bool(cleaned_overrides.get("adaptive_capacity", True))
            min_deter = int(cleaned_overrides.get("min_deter", min_deter))
            max_deter = int(cleaned_overrides.get("max_deter", max_deter))
            min_stoch = int(cleaned_overrides.get("min_stoch", min_stoch))
            max_stoch = int(cleaned_overrides.get("max_stoch", max_stoch))
            respect_profile_minimum = bool(
                cleaned_overrides.get("respect_profile_minimum", respect_profile_minimum)
            )
            hard_fallback_enabled = bool(
                cleaned_overrides.get("hard_fallback_enabled", False)
            )

        # Compute adaptive capacity
        (
            deter_dim,
            num_distributions,
            adaptive_hidden,
            adaptive_ctrl,
            adaptive_free_nats,
            adaptive_initial_log_alpha,
        ) = cls._compute_adaptive_capacity(
            env_obs_space,
            env_action_space,
            profile,
            adaptive_enabled=adaptive_enabled,
            min_deter=min_deter,
            max_deter=max_deter,
            min_stoch=min_stoch,
            max_stoch=max_stoch,
            respect_profile_minimum=respect_profile_minimum,
        )

        logger.info(
            "Adaptive Capacity: deter=%d, stoch=%dx16, hidden=%d, ctrl=%d, "
            "free_nats=%.1f, initial_log_alpha=%.1f",
            deter_dim,
            num_distributions,
            adaptive_hidden,
            adaptive_ctrl,
            adaptive_free_nats,
            adaptive_initial_log_alpha,
        )

        # Apply rssm overrides (explicit values take precedence)
        if rssm_overrides:
            if "deter_dim" in rssm_overrides:
                deter_dim = int(rssm_overrides["deter_dim"])
            if "num_distributions" in rssm_overrides:
                num_distributions = int(rssm_overrides["num_distributions"])
            elif "stoch_dim" in rssm_overrides:
                num_distributions = int(rssm_overrides["stoch_dim"])
            if "hidden_dim" in rssm_overrides:
                adaptive_hidden = int(rssm_overrides["hidden_dim"])
            if "ctrl_dim" in rssm_overrides:
                adaptive_ctrl = int(rssm_overrides["ctrl_dim"])

        rssm_overrides.setdefault("hidden_dim", adaptive_hidden)
        rssm_overrides.setdefault("ctrl_dim", adaptive_ctrl)

        # Determine observation dimensionality
        obs_dim = cls._infer_vector_obs_dim(env_obs_space)
        use_perceptor = obs_dim is None
        vitals_dim = obs_dim if obs_dim is not None else 64
        if use_perceptor and perceptor_overrides and "vitals_dim" in perceptor_overrides:
            vitals_dim = int(perceptor_overrides["vitals_dim"])

        # Create perceptor
        perceptor = None
        if use_perceptor:
            perceptor = cls._create_perceptor(
                profile,
                env_obs_space,
                vitals_dim,
                device,
                perceptor_overrides=perceptor_overrides,
            )

        # Create Will module
        will_mod = cls._create_will(profile, device)

        # Create world model
        world_model = cls._create_world_model(
            profile,
            deter_dim,
            num_distributions,
            vitals_dim,
            env_obs_space,
            env_action_space,
            perceptor,
            device,
            rssm_overrides=rssm_overrides,
            adaptive_free_nats=adaptive_free_nats,
            adaptive_ctrl_dim=adaptive_ctrl,
        )

        effective_hidden_dim = adaptive_hidden

        # Create router
        router_system = cls._create_router(
            effective_hidden_dim,
            adaptive_ctrl,
            device,
            hard_fallback_enabled=hard_fallback_enabled,
            router_overrides=router_overrides,
        )

        # Infer action space
        action_dim, is_discrete = cls._infer_action_info(env_action_space)

        # Create actor-critic
        actor_critic = cls._create_actor_critic(
            profile,
            router_system.feature_dim,
            action_dim,
            is_discrete,
            device,
            adaptive_hidden_dim=adaptive_hidden,
            adaptive_initial_log_alpha=adaptive_initial_log_alpha,
        )

        # Assemble container
        container = AgentContainer(
            world_model=world_model,
            actor_critic=actor_critic,
            perceptor=perceptor,
            router_system=router_system,
            will=will_mod,
            meta_will=None,
            env_profile=bridge_env_profile,
        )
        container._device = device

        # Log summary
        log_parts = [
            f"deter={deter_dim}",
            f"stochN={num_distributions}",
            f"action_dim={action_dim}",
            f"discrete={is_discrete}",
        ]
        if router_overrides:
            log_parts.append(f"router_overrides={len(router_overrides)} keys")
        if perceptor_overrides:
            log_parts.append(f"perceptor_overrides={len(perceptor_overrides)} keys")
        if bridge_env_profile is not None:
            env_cat = getattr(bridge_env_profile, "env_category", "unknown")
            log_parts.append(f"env_category={env_cat}")

        logger.info("Created agent: %s", ", ".join(log_parts))

        return container


# =============================================================================
# Environment Wrapper
# =============================================================================

class EnvWrapper:
    """Unified wrapper around *gymnasium* / *gym* environments, handling
    action-space conversion, seeding, and API differences."""

    __slots__ = (
        "env",
        "env_name",
        "_seed",
        "_first_reset",
        "observation_space",
        "action_space",
        "is_discrete",
        "action_dim",
        "_action_low",
        "_action_high",
        "_action_scale",
        "_action_bias",
        "_gym_version",
        "_action_processor",
        "_action_buffer",
    )

    def __init__(self, env_name: str, seed: int = DEFAULT_SEED):
        try:
            import gymnasium as gym
            self._gym_version = "gymnasium"
        except ImportError:
            import gym  # type: ignore[no-redef]
            self._gym_version = "gym"

        env = gym.make(env_name)
        self._setup_from_env(env=env, env_name=env_name, seed=seed)

    @classmethod
    def from_env(cls, env: Any, seed: int = DEFAULT_SEED) -> "EnvWrapper":
        """Wrap an already-instantiated environment."""
        self = cls.__new__(cls)
        is_gymnasium = False
        try:
            import gymnasium  # noqa: F401

            is_gymnasium = env.__class__.__module__.startswith("gymnasium")
        except ImportError:
            pass
        self._gym_version = "gymnasium" if is_gymnasium else "gym"
        env_name = getattr(getattr(env, "spec", None), "id", None) or env.__class__.__name__
        self._setup_from_env(env=env, env_name=env_name, seed=seed)
        return self

    def _setup_from_env(self, env: Any, env_name: str, seed: int) -> None:
        self.env = env
        self.env_name = env_name
        self._seed = seed
        self._first_reset = True

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        n = getattr(self.action_space, "n", None)
        self.is_discrete = n is not None

        if self.is_discrete:
            self.action_dim = n
            self._action_low = None
            self._action_high = None
            self._action_scale = None
            self._action_bias = None
            self._action_processor = self._process_discrete_action
            self._action_buffer = None
        else:
            self.action_dim = int(np.prod(self.action_space.shape))
            self._action_low = np.asarray(self.action_space.low, dtype=np.float32)
            self._action_high = np.asarray(self.action_space.high, dtype=np.float32)
            # Pre-compute scale/bias for tanh-to-bounds rescaling (cached)
            self._action_scale = (self._action_high - self._action_low) * 0.5
            self._action_bias = (self._action_high + self._action_low) * 0.5
            self._action_processor = self._process_continuous_action
            self._action_buffer = np.empty(self.action_dim, dtype=np.float32)

    # -- Core API -------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        use_seed = seed if seed is not None else (self._seed if self._first_reset else None)
        self._first_reset = False

        try:
            result = self.env.reset(seed=use_seed) if use_seed is not None else self.env.reset()
        except TypeError:
            if use_seed is not None and hasattr(self.env, "seed"):
                self.env.seed(use_seed)
            result = self.env.reset()

        obs = result[0] if isinstance(result, tuple) else result
        return self._convert_obs(obs)

    def step(
        self, action: Any
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float, bool, bool, Dict[str, Any]]:
        processed_action = self._action_processor(action)
        result = self.env.step(processed_action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False

        return self._convert_obs(obs), float(reward), bool(terminated), bool(truncated), info

    # -- Observation conversion -----------------------------------------------

    @staticmethod
    def _convert_obs(obs: Any) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(obs, dict):
            return {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}
        return np.asarray(obs, dtype=np.float32)

    # -- Action processing ----------------------------------------------------

    def _process_discrete_action(self, action: Any) -> int:
        """Convert *action* to a valid integer index."""
        if isinstance(action, (int, np.integer)):
            action_int = int(action)
        elif isinstance(action, np.ndarray):
            if action.ndim > 0 and action.shape[-1] == self.action_dim:
                action_int = int(action.argmax())
            else:
                action_int = int(action.flat[0])
        elif isinstance(action, torch.Tensor):
            if action.numel() == 1:
                action_int = int(action.item())
            elif action.shape[-1] == self.action_dim:
                action_int = int(action.argmax().item())
            else:
                action_int = int(action.flatten()[0].item())
        else:
            try:
                action_int = int(action)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Cannot convert action to int: {action}") from exc

        # Boundary check
        if not 0 <= action_int < self.action_dim:
            logger.warning(
                "Action %d out of bounds [0, %d), clamping to valid range",
                action_int,
                self.action_dim,
            )
            action_int = max(0, min(action_int, self.action_dim - 1))

        return action_int

    def _process_continuous_action(self, action: Any) -> np.ndarray:
        """Rescale *action* from [-1, 1] to the environment's bounds.

        .. warning:: The returned array is an *internal buffer* that will be
           overwritten on the next call.  Copy it if you need persistence.
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        buf = self._action_buffer
        np.copyto(buf, np.asarray(action, dtype=np.float32).flat[: self.action_dim])

        # tanh-to-bounds rescaling using pre-computed scale/bias
        buf[:] = buf * self._action_scale + self._action_bias
        np.clip(buf, self._action_low, self._action_high, out=buf)

        return buf

    # -- Lifecycle ------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        try:
            return self.env.render(mode=mode)
        except TypeError:
            return self.env.render()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception as exc:
            logger.warning("Error closing environment: %s", exc)

    def __enter__(self) -> "EnvWrapper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def __repr__(self) -> str:
        return (
            f"EnvWrapper({self.env_name}, discrete={self.is_discrete}, "
            f"action_dim={self.action_dim})"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def resolve_device(device_arg: Optional[str] = None) -> torch.device:
    """Resolve a device string (``"auto"`` / ``"cuda"`` / ``"cpu"`` / …) to a
    :class:`torch.device`."""
    if device_arg is None or device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_lower = device_arg.lower()

    if device_lower in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    if device_lower == "cpu":
        return torch.device("cpu")

    if device_lower.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but not available: {device_arg}")
        return torch.device(device_lower)

    if device_lower == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(
            mps_backend is not None
            and hasattr(mps_backend, "is_available")
            and mps_backend.is_available()
        )
        if not mps_available:
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    raise ValueError(
        f"Unknown device '{device_arg}'. Expected auto/cuda/cpu/mps or cuda:<id>."
    )


def set_global_seed(seed: int) -> None:
    """Set random seeds for ``random``, ``numpy``, and ``torch``."""
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)


def _prepare_obs_for_buffer(obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
    """Flatten an observation (or dict of observations) into a 1-D float32 array."""
    if isinstance(obs, dict):
        sorted_keys = sorted(obs.keys())
        arrays = [np.asarray(obs[k], dtype=np.float32).ravel() for k in sorted_keys]
        return np.concatenate(arrays)
    return np.asarray(obs, dtype=np.float32)


# =============================================================================
# Buffer Adapter
# =============================================================================

class BufferAdapter:
    """Thin adapter that unifies multiple replay-buffer interfaces behind a
    single ``add_transition`` API."""

    def __init__(self, buffer: Any, profile: EnvProfile):
        self._buffer = buffer
        self._profile = profile
        self._needs_initial_obs = True

        # Pre-detect supported interface methods
        self._has_add_step = hasattr(buffer, "add_step")
        self._has_add_transition = hasattr(buffer, "add_transition")
        self._has_add = hasattr(buffer, "add")
        self._has_push = hasattr(buffer, "push")
        self._has_start_episode = hasattr(buffer, "start_episode")
        self._has_add_initial = hasattr(buffer, "add_initial")

        # Pre-allocate one-hot buffer for discrete actions
        if profile.is_discrete:
            self._onehot_buffer = np.zeros(profile.action_dim, dtype=np.float32)
        else:
            self._onehot_buffer = None

    def start_episode(self, initial_obs: Optional[np.ndarray] = None) -> None:
        """Signal the start of a new episode, optionally providing the initial
        observation."""
        if self._has_start_episode:
            try:
                if initial_obs is not None:
                    self._buffer.start_episode(
                        initial_vitals=_prepare_obs_for_buffer(initial_obs),
                        initial_obs=None,
                    )
                    self._needs_initial_obs = False
                else:
                    self._buffer.start_episode()
                    self._needs_initial_obs = bool(self._has_add_initial)
            except Exception:
                self._needs_initial_obs = True
        else:
            self._needs_initial_obs = True

    def add_transition(
        self,
        obs: np.ndarray,
        action: Any,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a single transition using whatever interface the underlying
        buffer supports."""
        if info is None:
            info = {}
        done = bool(done)
        log_prob = float(np.asarray(info.get("log_prob", 0.0)).reshape(-1)[0])
        value = float(np.asarray(info.get("value", 0.0)).reshape(-1)[0])
        terminated = info.get("terminated")
        truncated = info.get("truncated")

        if terminated is None and truncated is None and "TimeLimit.truncated" in info:
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done and not truncated)
        elif terminated is None and truncated is not None:
            truncated = bool(truncated)
            terminated = bool(done and not truncated)
        elif terminated is not None and truncated is None:
            terminated = bool(terminated)
            truncated = bool(done and not terminated)
        elif terminated is not None and truncated is not None:
            terminated = bool(terminated)
            truncated = bool(truncated)

        action_arr = self._prepare_action(action)
        obs_arr = _prepare_obs_for_buffer(obs)
        next_obs_arr = _prepare_obs_for_buffer(next_obs)

        # Handle initial observation if needed
        if self._needs_initial_obs:
            if self._has_add_initial:
                try:
                    self._buffer.add_initial(vitals=obs_arr, obs=None)
                    self._needs_initial_obs = False
                except Exception:
                    pass
            elif self._has_start_episode:
                try:
                    self._buffer.start_episode(initial_vitals=obs_arr, initial_obs=None)
                    self._needs_initial_obs = False
                except Exception:
                    pass

        if self._has_add_step:
            try:
                self._buffer.add_step(
                    vitals=next_obs_arr,
                    action=action_arr,
                    reward=reward,
                    done=done,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=log_prob,
                )
            except TypeError:
                self._buffer.add_step(
                    vitals=next_obs_arr,
                    action=action_arr,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                )
        elif self._has_add_transition:
            try:
                self._buffer.add_transition(
                    action=action_arr,
                    reward=reward,
                    done=done,
                    next_vitals=next_obs_arr,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=log_prob,
                )
            except TypeError:
                self._buffer.add_transition(
                    action=action_arr,
                    reward=reward,
                    done=done,
                    next_vitals=next_obs_arr,
                    log_prob=log_prob,
                )
        elif self._has_add:
            self._buffer.add(
                observation=obs_arr,
                action=action_arr,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
            )
        elif self._has_push:
            self._buffer.push(
                obs=obs_arr,
                action=action_arr,
                reward=reward,
                done=done,
            )
        else:
            raise RuntimeError(f"Unsupported buffer type: {type(self._buffer)}")

    def _prepare_action(self, action: Union[int, np.ndarray]) -> np.ndarray:
        """Convert *action* to a float32 array (one-hot for discrete)."""
        if self._profile.is_discrete and isinstance(action, (int, np.integer)):
            buf = self._onehot_buffer
            buf[:] = 0.0
            buf[int(action)] = 1.0
            # Return a copy to avoid aliasing the internal buffer
            return buf.copy()
        return np.asarray(action, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._buffer)

    def __getattr__(self, name: str) -> Any:
        # Guard against infinite recursion during __init__
        try:
            buffer = object.__getattribute__(self, "_buffer")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(buffer, name)


# =============================================================================
# Agent Handle
# =============================================================================

class AgentHandle:
    """High-level interface for acting, observing, saving, and loading.

    This is the primary object returned by :func:`create_agent`.
    """

    def __init__(
        self,
        config: ConfigBundle,
        device: torch.device,
        agent_container: AgentContainer,
        buffer: Any,
        profile: Optional[EnvProfile] = None,
    ):
        self.config = config
        self._profile: Optional[EnvProfile] = None
        resolved_profile = profile if profile is not None else config.profile
        if resolved_profile is None:
            raise ValueError("AgentHandle requires ConfigBundle.profile or an explicit profile")
        self.profile = resolved_profile
        self.device = device

        self._container = agent_container
        self.world_model = agent_container.world_model
        self.actor = agent_container.actor_critic.actor
        self.critic = agent_container.actor_critic.critic
        self.router = agent_container.router_system
        self.will = agent_container.will

        self.buffer = BufferAdapter(buffer, self.profile) if buffer is not None else None

        self._wm_state: Any = None
        self._prev_action: Optional[torch.Tensor] = None
        self._step_count: int = 0
        self._needs_initial_obs: bool = True

        self._optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None
        self._compiled_router: Optional[nn.Module] = None
        self._compiled_actor: Optional[nn.Module] = None
        self._compiled_critic: Optional[nn.Module] = None
        self._compiled_wm_encoder: Optional[nn.Module] = None
        self._compiled_wm_projection: Optional[nn.Module] = None
        self._setup_inference_compilation()

        logger.info("AgentHandle initialized on %s", self.device)
        logger.info(
            "  Profile: %s | obs=%s | act=%d",
            self.profile.env_id,
            self.profile.obs_shape,
            self.profile.action_dim,
        )

    @property
    def profile(self) -> EnvProfile:
        config = getattr(self, "config", None)
        config_profile = getattr(config, "profile", None) if config is not None else None
        if config_profile is not None:
            return config_profile
        profile = getattr(self, "_profile", None)
        if profile is None:
            raise AttributeError("AgentHandle profile is not initialized")
        return profile

    @profile.setter
    def profile(self, value: EnvProfile) -> None:
        self._profile = value
        config = getattr(self, "config", None)
        if config is not None and hasattr(config, "profile"):
            config.profile = value

    # -- Episode lifecycle ----------------------------------------------------

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self._wm_state = None
        self._prev_action = None

        if self.world_model is not None and hasattr(self.world_model, "reset"):
            self.world_model.reset()

        if self.buffer is not None:
            try:
                self.buffer.start_episode()
            except Exception:
                pass

        self._needs_initial_obs = True

    # -- Compilation setup ----------------------------------------------------

    def _setup_inference_compilation(self) -> None:
        """Optionally compile / JIT-script sub-modules for faster inference."""
        cfg = getattr(self.config, "train", None)
        if cfg is None:
            return
        use_compile = bool(getattr(cfg, "use_compile", False))
        use_jit = bool(getattr(cfg, "use_jit", False))
        if not use_compile and not use_jit:
            return

        compile_mode = getattr(cfg, "compile_mode", "default")

        # Actor
        if self.actor is not None and bool(getattr(cfg, "compile_actor", True)):
            self._compiled_actor = self._apply_compilation(
                self.actor, use_jit, use_compile, compile_mode
            )

        # Critic
        if self.critic is not None and bool(getattr(cfg, "compile_critic", True)):
            self._compiled_critic = self._apply_compilation(
                self.critic, use_jit, use_compile, compile_mode
            )

        # Router
        if self.router is not None and bool(getattr(cfg, "compile_router", True)):
            if hasattr(self.router, "forward_components"):
                wrapper = _RouterForwardWrapper(self.router)
                compiled = self._apply_compilation(wrapper, use_jit, use_compile, compile_mode)
                if compiled is not None:
                    self._compiled_router = compiled

        # World-model encoder & projection
        if self.world_model is not None:
            if bool(getattr(cfg, "compile_wm_encoder", True)):
                encoder = getattr(self.world_model, "encoder", None)
                if isinstance(encoder, nn.Module):
                    compiled = self._apply_compilation(
                        encoder, use_jit, use_compile, compile_mode
                    )
                    if compiled is not None:
                        self._compiled_wm_encoder = compiled
                        self.world_model.encoder = compiled

            if bool(getattr(cfg, "compile_wm_projection", True)):
                proj = getattr(self.world_model, "_projection", None)
                if isinstance(proj, nn.Module):
                    compiled = self._apply_compilation(
                        proj, use_jit, use_compile, compile_mode
                    )
                    if compiled is not None:
                        self._compiled_wm_projection = compiled
                        self.world_model._projection = compiled

        if bool(getattr(cfg, "compile_warmup", False)):
            self._warmup_inference_compilation()

    @staticmethod
    def _apply_compilation(
        module: nn.Module, use_jit: bool, use_compile: bool, compile_mode: str
    ) -> Optional[nn.Module]:
        """Apply JIT / torch.compile to *module*; return *None* if unchanged."""
        result = module
        if use_jit:
            result = _try_script(result)
        if use_compile:
            result = _try_compile(result, compile_mode)
        return result if result is not module else None

    def _warmup_inference_compilation(self) -> None:
        """Run dummy forward passes so JIT / compiled graphs are traced."""
        cfg = getattr(self.config, "train", None)
        if cfg is None:
            return
        batch = int(getattr(cfg, "compile_warmup_batch_size", 2))
        if batch <= 0:
            return

        device = self.device
        feat_dim = self._infer_feat_dim()
        dummy_feat = torch.zeros(batch, feat_dim, device=device)

        self._warmup_single(self._compiled_actor or self.actor, dummy_feat)
        self._warmup_single(self._compiled_critic or self.critic, dummy_feat)

        # Router warmup
        if self._compiled_router is not None:
            ctrl_dim = self._infer_ctrl_dim()
            try:
                self._compiled_router(
                    x_rl=torch.zeros(1, feat_dim, device=device),
                    s_ctrl=torch.zeros(1, ctrl_dim, device=device),
                    z_task=None,
                    logvar=None,
                    x_t=torch.zeros(1, feat_dim, device=device),
                    x_proj=torch.zeros(1, feat_dim, device=device),
                    wall_strength=1.0,
                )
            except Exception:
                pass

        # Encoder warmup
        if self.world_model is not None:
            encoder = self._compiled_wm_encoder or getattr(self.world_model, "encoder", None)
            if isinstance(encoder, nn.Module):
                obs_shape = self.profile.obs_shape if self.profile else (4,)
                self._warmup_single(encoder, torch.zeros((1, *obs_shape), device=device))

        # Projection warmup
        if self.world_model is not None:
            proj = self._compiled_wm_projection or getattr(self.world_model, "_projection", None)
            if isinstance(proj, nn.Module):
                in_dim = self._infer_projection_in_dim(proj, feat_dim)
                self._warmup_single(proj, torch.zeros(1, in_dim, device=device))

    def _infer_feat_dim(self) -> int:
        if self.actor is not None and hasattr(self.actor, "feat_dim"):
            return int(self.actor.feat_dim)
        if self.profile is not None:
            return int(np.prod(self.profile.obs_shape))
        return 4

    def _infer_ctrl_dim(self) -> int:
        if self.world_model is not None and hasattr(self.world_model, "_wm_config"):
            val = getattr(self.world_model._wm_config, "ctrl_dim", None)
            if val is not None:
                return int(val)
        return 64

    @staticmethod
    def _infer_projection_in_dim(proj: nn.Module, fallback: int) -> int:
        net = getattr(proj, "net", None)
        if net is not None:
            try:
                return int(net[0].in_features)
            except Exception:
                pass
        return fallback

    @staticmethod
    def _warmup_single(module: Optional[nn.Module], dummy: torch.Tensor) -> None:
        if module is None:
            return
        try:
            module(dummy)
        except Exception:
            pass

    # -- Action selection -----------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select an action given *obs* and return ``(action_np, info_dict)``."""
        # Only force eval-mode for deterministic evaluation rollouts.
        # During stochastic data collection, router/world-model training-mode
        # behavior (EMA smoothing, uncertainty statistics, etc.) is part of the
        # live policy and should remain active.
        infer_modules: List[nn.Module] = []
        infer_prev_modes: List[bool] = []
        force_inference_eval = bool(deterministic)
        if force_inference_eval:
            for module in (
                self.world_model,
                self.router,
                self.actor,
                self.critic,
                self._compiled_router,
                self._compiled_actor,
                self._compiled_critic,
            ):
                if isinstance(module, nn.Module):
                    infer_modules.append(module)
                    infer_prev_modes.append(bool(module.training))
                    module.eval()

        wall_strength = 1.0
        x_rl: Optional[torch.Tensor] = None
        s_ctrl: Optional[torch.Tensor] = None
        z_task: Optional[torch.Tensor] = None
        logvar: Optional[torch.Tensor] = None
        x_proj: Optional[torch.Tensor] = None
        x_t: Optional[torch.Tensor] = None

        try:
            # Convert observation to tensor (batch dim = 1)
            obs_t = self._obs_to_tensor(obs)

            # World-model forward
            if self.world_model is not None:
                from .aletheia_train import isolate_policy_wm_state, resolve_policy_wall_strength

                a_prev = self._prev_action
                if a_prev is None:
                    a_prev = torch.zeros(1, self.profile.action_dim, device=self.device)
                    if self.profile.is_discrete:
                        a_prev[:, 0] = 1.0

                if self._wm_state is None and hasattr(self.world_model, "init_wm_state"):
                    try:
                        self._wm_state = self.world_model.init_wm_state(
                            1,
                            self.device,
                            deterministic=deterministic,
                        )
                    except Exception:
                        self._wm_state = None

                self._wm_state, outputs = self.world_model.forward_context(
                    obs_t, a_prev, self._wm_state, deterministic_state=deterministic
                )

                wall_strength = resolve_policy_wall_strength(self.world_model)

                sfp = isolate_policy_wm_state(
                    self._wm_state,
                    wall_strength=wall_strength,
                )

                x_rl = sfp.x_proj if sfp.x_proj is not None else sfp.x_t
                s_ctrl = sfp.s_ctrl
                z_task = sfp.z_task
                logvar = getattr(outputs, "decoder_logvar", None)
                x_proj = sfp.x_proj
                x_t = sfp.x_t
            else:
                # Fallback: use raw observation as features
                x_t = next(iter(obs_t.values())) if isinstance(obs_t, dict) else obs_t
                x_rl = x_t

            # Router forward
            f_policy = self._forward_router(
                x_rl, s_ctrl, z_task, logvar, x_t, x_proj, wall_strength
            )

            # Actor forward
            actor_mod = self._compiled_actor or self.actor
            action_dist = actor_mod(f_policy)

            if self.profile.is_discrete:
                action_one_hot = action_dist.mode if deterministic else action_dist.sample()
                log_prob = action_dist.log_prob(action_one_hot)
                action_index = action_one_hot.argmax(dim=-1)
                action_for_prev = action_one_hot
                action_for_env = action_index
            else:
                action_cont = action_dist.mean if deterministic else action_dist.sample()
                log_prob = action_dist.log_prob(action_cont)
                action_for_prev = action_cont
                action_for_env = action_cont

            # Critic forward
            critic_mod = self._compiled_critic or self.critic
            critic_out = critic_mod(f_policy)
            value = critic_out.values_real_main

            # Update internal state
            self._prev_action = self._to_action_tensor(action_for_prev)

            # Convert action to numpy
            action_np = self._action_tensor_to_numpy(action_for_env)

            info = {
                "log_prob": log_prob.squeeze().cpu().numpy(),
                "value": value.squeeze().cpu().numpy(),
                "uncertainty": critic_out.uncertainty_raw.squeeze().cpu().numpy(),
                "f_policy": f_policy.detach().squeeze(0).cpu().numpy().astype(np.float32),
            }

            return action_np, info
        finally:
            for module, was_training in zip(infer_modules, infer_prev_modes):
                module.train(was_training)

    def _obs_to_tensor(
        self, obs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert observation to tensor(s) with a leading batch dimension."""
        obs_ndim = len(self.profile.obs_shape)
        if isinstance(obs, dict):
            return {
                k: self._ensure_batch_dim(
                    torch.as_tensor(v, device=self.device, dtype=torch.float32), obs_ndim
                )
                for k, v in obs.items()
            }
        t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self._ensure_batch_dim(t, obs_ndim)

    @staticmethod
    def _ensure_batch_dim(t: torch.Tensor, expected_ndim: int) -> torch.Tensor:
        if t.dim() == expected_ndim:
            return t.unsqueeze(0)
        return t

    def _forward_router(
        self,
        x_rl: Optional[torch.Tensor],
        s_ctrl: Optional[torch.Tensor],
        z_task: Optional[torch.Tensor],
        logvar: Optional[torch.Tensor],
        x_t: Optional[torch.Tensor],
        x_proj: Optional[torch.Tensor],
        wall_strength: float,
    ) -> torch.Tensor:
        """Run the router (compiled or interpreted) and return ``f_policy``."""
        router_mod = self._compiled_router
        if router_mod is not None:
            pf = router_mod(
                x_rl=x_rl,
                s_ctrl=s_ctrl,
                z_task=z_task,
                logvar=logvar,
                x_t=x_t,
                x_proj=x_proj,
                wall_strength=wall_strength,
            )
            return pf.f_policy

        if self.router is not None and hasattr(self.router, "forward_components"):
            pf = self.router.forward_components(
                x_rl=x_rl,
                s_ctrl=s_ctrl,
                z_task=z_task,
                logvar=logvar,
                x_t=x_t,
                x_proj=x_proj,
                wall_strength=wall_strength,
            )
            return pf.f_policy

        return x_rl

    def _action_tensor_to_numpy(self, action_for_env: torch.Tensor) -> Any:
        """Convert the action tensor to a numpy value suitable for ``env.step``."""
        if action_for_env.dim() == 0:
            val = action_for_env.cpu().item()
        else:
            val = action_for_env.squeeze(0).cpu().numpy()

        if self.profile.is_discrete and isinstance(val, np.ndarray):
            val = int(val.item()) if val.size == 1 else val

        return val

    # -- Action tensor conversion ---------------------------------------------

    def _to_action_tensor(
        self,
        action: Union[torch.Tensor, np.ndarray, int, float],
    ) -> torch.Tensor:
        """Unify *action* into a ``(1, action_dim)`` float tensor."""
        if not isinstance(action, torch.Tensor):
            if isinstance(action, (int, float, np.integer, np.floating)):
                action = torch.tensor(action, device=self.device, dtype=torch.float32)
            elif isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(device=self.device, dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported action type: {type(action)}")

        if action.device != self.device:
            action = action.to(self.device)

        if self.profile.is_discrete:
            return self._discrete_to_onehot(action)

        # Continuous: ensure shape is (1, action_dim)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return action.float()

    def _discrete_to_onehot(self, action: torch.Tensor) -> torch.Tensor:
        """Convert a discrete action (index or one-hot) to one-hot ``(1, A)``."""
        adim = self.profile.action_dim

        # Already one-hot shaped
        if action.dim() >= 1 and action.shape[-1] == adim:
            return action.unsqueeze(0) if action.dim() == 1 else action

        # Convert index to one-hot
        if action.dim() == 0:
            indices = action.view(1)
        elif action.dim() == 2 and action.shape[-1] == 1:
            indices = action.squeeze(-1)
        else:
            indices = action.reshape(-1)

        indices = indices.long().view(-1, 1)
        one_hot = torch.zeros(indices.shape[0], adim, device=self.device)
        one_hot.scatter_(1, indices, 1.0)
        return one_hot

    # -- Transition storage ---------------------------------------------------

    def observe(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a transition in the replay buffer."""
        if self.buffer is not None:
            self.buffer.add_transition(obs, action, reward, next_obs, done, info)
        self._step_count += 1

    def ready_to_train(self) -> bool:
        """Return True when the buffer has enough data for training."""
        if self.buffer is None:
            return False
        min_size = self.config.train.batch_size * 2
        return len(self.buffer) >= min_size

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def buffer_size(self) -> int:
        if self.buffer is None:
            return 0
        return len(self.buffer)

    # -- Save / Load ----------------------------------------------------------

    def save(
        self,
        path: str,
        *,
        effective_training_config: Optional[Any] = None,
        agent_bootstrap_bundle: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a checkpoint to *path*."""
        if self.world_model is not None and hasattr(self.world_model, "_ensure_v45_components"):
            self.world_model._ensure_v45_components()

        bootstrap_bundle = (
            copy.deepcopy(agent_bootstrap_bundle)
            if isinstance(agent_bootstrap_bundle, dict)
            else self.config.to_agent_bootstrap_bundle()
        )
        checkpoint = build_agent_checkpoint_payload(
            self,
            version=__version__,
            bootstrap_bundle=bootstrap_bundle,
            effective_training_config=effective_training_config,
        )

        torch.save(checkpoint, path)
        logger.info("Model saved to %s", path)

    def load(
        self,
        path: str,
        strict: bool = True,
        *,
        allow_unsafe_fallback: bool = False,
    ) -> None:
        """Load a checkpoint from *path*.

        ``strict=False`` only relaxes optional component restores. Required
        components still restore strictly by default.
        """
        if allow_unsafe_fallback:
            raise ValueError(
                "allow_unsafe_fallback is no longer supported for agent checkpoint loads"
            )
        checkpoint = read_checkpoint(
            path,
            map_location=self.device,
            weights_only=True,
        )

        if self.world_model is not None and hasattr(self.world_model, "_ensure_v45_components"):
            self.world_model._ensure_v45_components()

        self._step_count = restore_agent_checkpoint_modules(
            self,
            checkpoint,
            strict=strict,
            checkpoint_path=path,
        )
        logger.info("Model loaded from %s", path)

    @property
    def actor_critic(self) -> Any:
        return self._container.actor_critic


class AgentRolloutCollector:
    """Collect trajectories using the standard stateful ``AgentHandle`` path.

    This collector preserves the same output contract as ``RolloutCollector``
    while delegating action selection to ``agent.act(...)`` so training and
    evaluation share the same world-model state semantics.
    """

    def __init__(
        self,
        agent: AgentHandle,
        env: Any,
        max_steps: int = 1000,
    ):
        self.agent = agent
        self.env = env if isinstance(env, EnvWrapper) else EnvWrapper.from_env(env)
        self.max_steps = int(max_steps)
        self.current_obs: Optional[np.ndarray] = None
        self.current_step = 0
        self.episode_return = 0.0
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self.agent.reset()
        self.current_obs = obs
        self.current_step = 0
        self.episode_return = 0.0
        return obs

    def _action_array(self, action_for_env: Any) -> np.ndarray:
        prev_action = getattr(self.agent, '_prev_action', None)
        if isinstance(prev_action, torch.Tensor):
            return prev_action.detach().cpu().numpy()[0].astype(np.float32, copy=False)
        if self.agent.profile.is_discrete:
            one_hot = np.zeros(self.agent.profile.action_dim, dtype=np.float32)
            one_hot[int(action_for_env)] = 1.0
            return one_hot
        return np.asarray(action_for_env, dtype=np.float32)

    @staticmethod
    def _scalar(info: Dict[str, Any], key: str) -> float:
        val = info.get(key, 0.0)
        arr = np.asarray(val, dtype=np.float32).reshape(-1)
        return float(arr[0]) if arr.size > 0 else 0.0

    def _collect_impl(
        self,
        num_steps: int,
        deterministic: bool = False,
    ) -> Dict[str, np.ndarray]:
        observations: List[np.ndarray] = []
        next_observations: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        rewards_list: List[float] = []
        dones_list: List[float] = []
        terminated_list: List[float] = []
        truncated_list: List[float] = []
        values_list: List[float] = []
        log_probs_list: List[float] = []

        if self.current_obs is None:
            self.reset()

        for _ in range(int(num_steps)):
            obs_np = np.asarray(self.current_obs, dtype=np.float32)
            action_env, act_info = self.agent.act(obs_np, deterministic=deterministic)
            next_obs_np, reward, terminated, truncated, info = self.env.step(action_env)
            next_obs_np = np.asarray(next_obs_np, dtype=np.float32)
            next_step = self.current_step + 1
            forced_truncate = bool(next_step >= self.max_steps)
            terminated_step = bool(terminated)
            truncated_step = bool(truncated) or forced_truncate
            done = terminated_step or truncated_step

            observations.append(obs_np)
            next_observations.append(next_obs_np)
            actions.append(self._action_array(action_env))
            rewards_list.append(float(reward))
            dones_list.append(float(done))
            terminated_list.append(float(terminated_step))
            truncated_list.append(float(truncated_step))
            values_list.append(self._scalar(act_info, 'value'))
            log_probs_list.append(self._scalar(act_info, 'log_prob'))

            self.episode_return += float(reward)
            self.current_step = next_step

            if done:
                self.episode_returns.append(self.episode_return)
                self.episode_lengths.append(self.current_step)
                self.reset()
            else:
                self.current_obs = next_obs_np

        return {
            'observations': np.array(observations, dtype=np.float32),
            'next_observations': np.array(next_observations, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards_list, dtype=np.float32),
            'dones': np.array(dones_list, dtype=np.float32),
            'terminated': np.array(terminated_list, dtype=np.float32),
            'truncated': np.array(truncated_list, dtype=np.float32),
            'values': np.array(values_list, dtype=np.float32),
            'log_probs': np.array(log_probs_list, dtype=np.float32),
        }

    def collect(
        self,
        num_steps: int,
        deterministic: bool = False,
    ) -> Dict[str, np.ndarray]:
        try:
            return self._collect_impl(num_steps=num_steps, deterministic=deterministic)
        except Exception as exc:
            logger.exception('Error during agent data collection: %s', exc)
            raise

    def get_statistics(self) -> Dict[str, float]:
        if not self.episode_returns:
            return {'mean_return': 0.0, 'mean_length': 0.0, 'num_episodes': 0}
        return {
            'mean_return': float(np.mean(self.episode_returns)),
            'std_return': float(np.std(self.episode_returns)),
            'mean_length': float(np.mean(self.episode_lengths)),
            'num_episodes': len(self.episode_returns),
            'min_return': float(np.min(self.episode_returns)),
            'max_return': float(np.max(self.episode_returns)),
        }

    def reset_statistics(self) -> None:
        self.episode_returns.clear()
        self.episode_lengths.clear()


def create_agent_rollout_collector(
    agent: AgentHandle,
    env: Any,
    max_steps: int = 1000,
) -> AgentRolloutCollector:
    return AgentRolloutCollector(agent=agent, env=env, max_steps=max_steps)


_REAL_STABILITY_CORRIDOR_KL_THRESHOLD = 0.05


def _reduce_dist_stat(
    value: Any,
    *,
    strict: bool = False,
    failure_context: str = "distribution statistic",
) -> Optional[float]:
    if value is None:
        if strict:
            raise RuntimeError(f"{failure_context} failed: value is unavailable")
        return None
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value, dtype=torch.float32)
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"{failure_context} failed: could not coerce value to tensor ({exc})"
                ) from exc
            return None
    if value.numel() <= 0:
        if strict:
            raise RuntimeError(f"{failure_context} failed: value is empty")
        return None
    value_t = value.detach().to(dtype=torch.float32)
    if value_t.dim() > 1:
        value_t = value_t.reshape(value_t.shape[0], -1).mean(dim=-1)
    return float(value_t.mean().item())


def _resolve_dist_probs_and_log_probs(
    dist: Any,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    probs = getattr(dist, "probs", None)
    log_probs = getattr(dist, "log_probs", None)
    logits = getattr(dist, "logits", None)
    inner = getattr(dist, "_cat", None)

    if inner is not None:
        if probs is None:
            probs = getattr(inner, "probs", None)
        if logits is None:
            logits = getattr(inner, "logits", None)

    if probs is None and logits is not None:
        probs = torch.softmax(logits, dim=-1)
    if log_probs is None:
        if probs is not None:
            log_probs = torch.log(probs.clamp(min=1e-8))
        elif logits is not None:
            log_probs = torch.log_softmax(logits, dim=-1)

    return probs, log_probs


def _compute_dist_kl(
    anchor_dist: Any,
    current_dist: Any,
    *,
    strict: bool = False,
    failure_context: str = "policy KL",
) -> Optional[torch.Tensor]:
    kl_method = getattr(anchor_dist, "kl_divergence", None)
    if callable(kl_method):
        try:
            return kl_method(current_dist)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"{failure_context} failed: {exc}") from exc
            return None
    try:
        return torch_kl_divergence(anchor_dist, current_dist)
    except Exception:
        pass
    anchor_probs, anchor_log_probs = _resolve_dist_probs_and_log_probs(anchor_dist)
    _, current_log_probs = _resolve_dist_probs_and_log_probs(current_dist)
    if anchor_probs is not None and anchor_log_probs is not None and current_log_probs is not None:
        return (
            anchor_probs.detach()
            * (anchor_log_probs.detach() - current_log_probs)
        ).sum(dim=-1)
    if strict:
        raise RuntimeError(
            f"{failure_context} failed: could not resolve probability/log-prob tensors"
        )
    return None


def _forward_policy_dist(
    actor: Any,
    feat_t: torch.Tensor,
    *,
    strict: bool = False,
    failure_context: str = "policy distribution forward",
) -> Optional[Any]:
    if actor is None or not callable(actor):
        if strict:
            raise RuntimeError(f"{failure_context} failed: actor is unavailable")
        return None
    was_training: Optional[bool] = None
    if isinstance(actor, nn.Module):
        was_training = bool(actor.training)
        actor.eval()
    try:
        with torch.no_grad():
            return actor(feat_t)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"{failure_context} failed: {exc}") from exc
        return None
    finally:
        if was_training is not None:
            actor.train(was_training)


def _actions_equal(lhs: np.ndarray, rhs: np.ndarray, atol: float = 1e-6) -> bool:
    lhs_arr = np.asarray(lhs, dtype=np.float32).reshape(-1)
    rhs_arr = np.asarray(rhs, dtype=np.float32).reshape(-1)
    if lhs_arr.shape != rhs_arr.shape:
        return False
    return bool(np.allclose(lhs_arr, rhs_arr, atol=atol, rtol=0.0))


def _run_episode_agent_with_telemetry(
    env_ref: EnvWrapper,
    agent: AgentHandle,
    deterministic: bool,
    max_steps: int = 1000,
) -> Tuple[float, int, Dict[str, float]]:
    obs = env_ref.reset()
    agent.reset()
    total_reward = 0.0
    steps = 0
    done = False

    action_trace: List[np.ndarray] = []
    entropy_values: List[float] = []
    policy_kl_values: List[float] = []
    policy_registry_kl_values: List[float] = []
    corridor_flags: List[float] = []

    current_actor = getattr(agent, "_compiled_actor", None) or getattr(agent, "actor", None)
    anchor_actor = getattr(agent, "_real_stability_eval_anchor_actor", None)
    registry_actors = getattr(agent, "_real_stability_eval_anchor_registry_actors", None)
    device = getattr(agent, "device", torch.device("cpu"))

    while not done and steps < max_steps:
        action, info = agent.act(obs, deterministic=deterministic)
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 0:
            action_arr = action_arr.reshape(1)
        action_trace.append(action_arr.reshape(-1))

        feat_np = info.get("f_policy") if isinstance(info, dict) else None
        if feat_np is not None:
            feat_t = torch.as_tensor(feat_np, device=device, dtype=torch.float32)
            if feat_t.dim() == 1:
                feat_t = feat_t.unsqueeze(0)
            current_dist = _forward_policy_dist(
                current_actor,
                feat_t,
                strict=True,
                failure_context="real-stability telemetry current policy forward",
            )
            entropy_fn = getattr(current_dist, "entropy", None)
            entropy_val = _reduce_dist_stat(
                entropy_fn() if callable(entropy_fn) else None,
                strict=True,
                failure_context="real-stability telemetry action entropy",
            )
            if entropy_val is not None:
                entropy_values.append(entropy_val)
            anchor_kl_val: Optional[float] = None
            if anchor_actor is not None:
                anchor_dist = _forward_policy_dist(
                    anchor_actor,
                    feat_t,
                    strict=True,
                    failure_context="real-stability telemetry anchor policy forward",
                )
                anchor_kl_val = _reduce_dist_stat(
                    _compute_dist_kl(
                        anchor_dist,
                        current_dist,
                        strict=True,
                        failure_context="real-stability telemetry anchor policy KL",
                    ),
                    strict=True,
                    failure_context="real-stability telemetry anchor policy KL",
                )
                if anchor_kl_val is not None:
                    policy_kl_values.append(anchor_kl_val)
            registry_kl_val: Optional[float] = None
            valid_registry_actors = [
                actor
                for actor in list(registry_actors or [])
                if actor is not None
            ]
            for registry_actor in valid_registry_actors:
                anchor_dist = _forward_policy_dist(
                    registry_actor,
                    feat_t,
                    strict=True,
                    failure_context="real-stability telemetry registry policy forward",
                )
                kl_val = _reduce_dist_stat(
                    _compute_dist_kl(
                        anchor_dist,
                        current_dist,
                        strict=True,
                        failure_context="real-stability telemetry registry policy KL",
                    ),
                    strict=True,
                    failure_context="real-stability telemetry registry policy KL",
                )
                if kl_val is None:
                    continue
                registry_kl_val = (
                    kl_val
                    if registry_kl_val is None
                    else min(registry_kl_val, kl_val)
                )
            if registry_kl_val is not None:
                policy_registry_kl_values.append(registry_kl_val)
                corridor_flags.append(
                    float(
                        registry_kl_val
                        <= _REAL_STABILITY_CORRIDOR_KL_THRESHOLD
                    )
                )
            elif anchor_kl_val is not None:
                corridor_flags.append(
                    float(
                        anchor_kl_val
                        <= _REAL_STABILITY_CORRIDOR_KL_THRESHOLD
                    )
                )

        obs, reward, terminated, truncated, _ = env_ref.step(action)
        total_reward += float(reward)
        steps += 1
        done = bool(terminated) or bool(truncated)

    switch_sum = 0.0
    switch_count = 0.0
    oscillation_sum = 0.0
    oscillation_count = 0.0
    if len(action_trace) > 1:
        for idx in range(1, len(action_trace)):
            switch_sum += float(not _actions_equal(action_trace[idx], action_trace[idx - 1]))
            switch_count += 1.0
    if len(action_trace) > 2:
        for idx in range(2, len(action_trace)):
            oscillation_sum += float(
                _actions_equal(action_trace[idx], action_trace[idx - 2])
                and not _actions_equal(action_trace[idx], action_trace[idx - 1])
            )
            oscillation_count += 1.0

    occupancy_sum = float(sum(corridor_flags))
    occupancy_count = float(len(corridor_flags))
    persistence_num = 0.0
    persistence_den = 0.0
    entry_num = 0.0
    entry_den = 0.0
    exit_num = 0.0
    exit_den = 0.0
    if len(corridor_flags) > 1:
        for prev_flag, next_flag in zip(corridor_flags[:-1], corridor_flags[1:]):
            if prev_flag > 0.0:
                persistence_num += float(next_flag > 0.0)
                persistence_den += 1.0
                exit_num += float(next_flag <= 0.0)
                exit_den += 1.0
            else:
                entry_num += float(next_flag > 0.0)
                entry_den += 1.0

    telemetry = {
        "real_behavior_action_entropy_sum": float(sum(entropy_values)),
        "real_behavior_action_entropy_count": float(len(entropy_values)),
        "real_behavior_action_switch_sum": float(switch_sum),
        "real_behavior_action_switch_count": float(switch_count),
        "real_behavior_action_oscillation_sum": float(oscillation_sum),
        "real_behavior_action_oscillation_count": float(oscillation_count),
        "real_corridor_occupancy_sum": float(occupancy_sum),
        "real_corridor_occupancy_count": float(occupancy_count),
        "real_corridor_persistence_num": float(persistence_num),
        "real_corridor_persistence_den": float(persistence_den),
        "real_corridor_entry_num": float(entry_num),
        "real_corridor_entry_den": float(entry_den),
        "real_corridor_exit_num": float(exit_num),
        "real_corridor_exit_den": float(exit_den),
        "real_policy_kl_to_certified_anchor_sum": float(sum(policy_kl_values)),
        "real_policy_kl_to_certified_anchor_count": float(len(policy_kl_values)),
        "real_policy_certified_anchor_available": float(anchor_actor is not None and len(policy_kl_values) > 0),
        "real_policy_kl_to_certified_registry_sum": float(sum(policy_registry_kl_values)),
        "real_policy_kl_to_certified_registry_count": float(len(policy_registry_kl_values)),
        "real_policy_certified_registry_available": float(
            bool(registry_actors) and len(policy_registry_kl_values) > 0
        ),
        "real_policy_certified_registry_size_sum": float(
            len(list(registry_actors or []))
        ),
        "real_policy_certified_registry_size_count": float(
            bool(registry_actors)
        ),
    }
    return total_reward, steps, telemetry


def _evaluate_agent(
    env_obj: Any,
    agent: AgentHandle,
    episodes: int,
    max_steps: int = 1000,
) -> Dict[str, float]:
    def _safe_div(aggregate: Dict[str, float], num_key: str, den_key: str) -> float:
        denom = float(aggregate.get(den_key, 0.0))
        if denom <= 0.0:
            return 0.0
        return float(aggregate.get(num_key, 0.0)) / denom

    rewards: List[float] = []
    lengths: List[int] = []
    env_ref = env_obj if isinstance(env_obj, EnvWrapper) else EnvWrapper.from_env(env_obj)
    wm_state = getattr(agent, "_wm_state", None)
    if hasattr(wm_state, "detach_all"):
        wm_state = wm_state.detach_all()
    else:
        try:
            wm_state = copy.deepcopy(wm_state)
        except Exception:
            pass

    prev_action = getattr(agent, "_prev_action", None)
    if isinstance(prev_action, torch.Tensor):
        prev_action = prev_action.detach().clone()

    runtime_state: Dict[str, Any] = {
        "wm_state": wm_state,
        "prev_action": prev_action,
        "step_count": int(getattr(agent, "_step_count", 0)),
        "needs_initial_obs": bool(getattr(agent, "_needs_initial_obs", True)),
    }

    buffer = getattr(agent, "buffer", None)
    if buffer is not None:
        if hasattr(buffer, "_needs_initial_obs"):
            runtime_state["buffer_needs_initial_obs"] = bool(
                getattr(buffer, "_needs_initial_obs", True)
            )
        raw_buffer = getattr(buffer, "_buffer", buffer)
        if hasattr(raw_buffer, "current"):
            try:
                runtime_state["buffer_current"] = copy.deepcopy(
                    getattr(raw_buffer, "current", None)
                )
            except Exception:
                runtime_state["buffer_current"] = getattr(raw_buffer, "current", None)

    rng_state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        rng_state["torch_cpu"] = torch.random.get_rng_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass

    telemetry_totals: Dict[str, float] = {}
    try:
        for _ in range(int(episodes)):
            reward, steps, telemetry = _run_episode_agent_with_telemetry(
                env_ref=env_ref,
                agent=agent,
                deterministic=True,
                max_steps=max_steps,
            )
            rewards.append(float(reward))
            lengths.append(int(steps))
            for key, value in telemetry.items():
                telemetry_totals[key] = float(telemetry_totals.get(key, 0.0)) + float(value)
    finally:
        agent._wm_state = runtime_state.get("wm_state")
        agent._prev_action = runtime_state.get("prev_action")
        agent._step_count = int(
            runtime_state.get("step_count", getattr(agent, "_step_count", 0))
        )
        agent._needs_initial_obs = bool(
            runtime_state.get(
                "needs_initial_obs",
                getattr(agent, "_needs_initial_obs", True),
            )
        )

        buffer = getattr(agent, "buffer", None)
        if buffer is not None:
            if "buffer_needs_initial_obs" in runtime_state and hasattr(
                buffer, "_needs_initial_obs"
            ):
                buffer._needs_initial_obs = bool(
                    runtime_state["buffer_needs_initial_obs"]
                )
            raw_buffer = getattr(buffer, "_buffer", buffer)
            if "buffer_current" in runtime_state and hasattr(raw_buffer, "current"):
                raw_buffer.current = runtime_state.get("buffer_current")

        python_state = rng_state.get("python")
        if python_state is not None:
            random.setstate(python_state)
        numpy_state = rng_state.get("numpy")
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        torch_cpu_state = rng_state.get("torch_cpu")
        if torch_cpu_state is not None:
            torch.random.set_rng_state(torch_cpu_state)
        torch_cuda_state = rng_state.get("torch_cuda")
        if torch_cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(torch_cuda_state)

    telemetry = {
        "real_behavior_action_entropy_mean": _safe_div(
            telemetry_totals,
            "real_behavior_action_entropy_sum",
            "real_behavior_action_entropy_count",
        ),
        "real_behavior_action_switch_rate": _safe_div(
            telemetry_totals,
            "real_behavior_action_switch_sum",
            "real_behavior_action_switch_count",
        ),
        "real_behavior_action_oscillation_rate": _safe_div(
            telemetry_totals,
            "real_behavior_action_oscillation_sum",
            "real_behavior_action_oscillation_count",
        ),
        "real_corridor_occupancy_fraction": _safe_div(
            telemetry_totals,
            "real_corridor_occupancy_sum",
            "real_corridor_occupancy_count",
        ),
        "real_corridor_persistence_rate": _safe_div(
            telemetry_totals,
            "real_corridor_persistence_num",
            "real_corridor_persistence_den",
        ),
        "real_corridor_entry_rate": _safe_div(
            telemetry_totals,
            "real_corridor_entry_num",
            "real_corridor_entry_den",
        ),
        "real_corridor_exit_rate": _safe_div(
            telemetry_totals,
            "real_corridor_exit_num",
            "real_corridor_exit_den",
        ),
        "real_policy_kl_to_certified_anchor_mean": _safe_div(
            telemetry_totals,
            "real_policy_kl_to_certified_anchor_sum",
            "real_policy_kl_to_certified_anchor_count",
        ),
        "real_policy_certified_anchor_available": float(
            telemetry_totals.get("real_policy_certified_anchor_available", 0.0) > 0.0
        ),
        "real_policy_kl_to_certified_registry_mean": _safe_div(
            telemetry_totals,
            "real_policy_kl_to_certified_registry_sum",
            "real_policy_kl_to_certified_registry_count",
        ),
        "real_policy_certified_registry_available": float(
            telemetry_totals.get("real_policy_certified_registry_available", 0.0) > 0.0
        ),
        "real_policy_certified_registry_size": _safe_div(
            telemetry_totals,
            "real_policy_certified_registry_size_sum",
            "real_policy_certified_registry_size_count",
        ),
    }
    if not rewards:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'mean_length': 0.0,
            'telemetry': telemetry,
        }
    return {
        'mean': float(np.mean(rewards)),
        'std': float(np.std(rewards)),
        'min': float(np.min(rewards)),
        'max': float(np.max(rewards)),
        'mean_length': float(np.mean(lengths)),
        'telemetry': telemetry,
    }


# =============================================================================
# Factory Functions (Main Entry Points)
# =============================================================================

def create_agent(
    env: Any,
    config_overrides: Optional[Dict[str, Any]] = None,
    device: Union[str, torch.device] = "auto",
    seed: int = DEFAULT_SEED,
) -> AgentHandle:
    """Create an :class:`AgentHandle` for the given environment.

    Parameters
    ----------
    env : gym.Env or gymnasium.Env
        The environment to build an agent for.
    config_overrides : dict, optional
        Optional key-value overrides for training / architecture configuration.
    device : str or torch.device
        Target compute device (``"auto"`` selects CUDA if available).
    seed : int
        Global random seed.

    Returns
    -------
    AgentHandle
    """
    from .aletheia_train import create_replay_buffer

    set_global_seed(seed)

    if isinstance(device, str):
        device = resolve_device(device)

    # Generate config bundle
    config = ConfigBundle.for_env(env, config_overrides, include_env_profile=True)
    profile = config.profile

    # Create agent components
    agent_container = AgentFactory.create_agent(
        env_obs_space=env.observation_space,
        env_action_space=env.action_space,
        device=device,
        custom_overrides=config.to_factory_overrides() or None,
    )

    # Create replay buffer
    buffer = create_replay_buffer(
        capacity=config.train.buffer_size,
        store_obs=False,
    )

    return AgentHandle(
        config=config,
        device=device,
        agent_container=agent_container,
        buffer=buffer,
    )


def load_agent(
    path: str,
    env: Any = None,
    device: Union[str, torch.device] = "auto",
    strict: bool = True,
    allow_unsafe_fallback: bool = False,
) -> AgentHandle:
    """Create an agent from *env* and restore weights from *path*.

    Raises
    ------
    ValueError
        If *env* is ``None``.
    """
    if env is None:
        raise ValueError("env is required to load agent")
    if allow_unsafe_fallback:
        raise ValueError(
            "allow_unsafe_fallback is no longer supported for agent checkpoint loads"
        )

    checkpoint_path = Path(path)
    config_overrides = read_agent_creation_overrides_from_checkpoint(
        checkpoint_path,
    )
    if config_overrides:
        logger.info(
            "Restoring agent config overrides from checkpoint metadata: %s",
            checkpoint_path,
        )

    agent = create_agent(env, config_overrides=config_overrides, device=device)
    agent.load(path, strict=strict)

    return agent


def _normalize_args(args: Any) -> Any:
    if isinstance(args, dict):
        return SimpleNamespace(**args)
    return args


def _parse_overrides(overrides_arg: Optional[str]) -> Dict[str, Any]:
    if not overrides_arg:
        return {}
    path = Path(overrides_arg)
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        source = f"override file '{path}'"
    else:
        raw = overrides_arg
        source = "override JSON string"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Overrides from {source} must decode to a JSON object, got {type(data).__name__}"
        )
    rl = data.setdefault("rl", {})
    rl.setdefault("use_alpha_adaptive", True)
    rl.setdefault("use_entropy_protection", False)
    return data


@lru_cache(maxsize=1)
def _run_train_training_only_override_keys() -> frozenset[str]:
    from .aletheia_config import AGENT_BOOTSTRAP_TRAIN_FIELDS, TrainingConfig

    return frozenset(TrainingConfig.__dataclass_fields__) - frozenset(
        AGENT_BOOTSTRAP_TRAIN_FIELDS
    )


def _extract_run_train_factory_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only create_agent bootstrap/factory overrides at the run_train boundary."""
    if not overrides:
        return {}

    training_only_keys = _run_train_training_only_override_keys()
    return {
        key: copy.deepcopy(value)
        for key, value in overrides.items()
        if key not in training_only_keys
    }


def _resolve_adapter_fn(adapter: Any, name: str) -> Optional[Callable[[Any], Any]]:
    if adapter is None:
        return None
    if isinstance(adapter, dict):
        fn = adapter.get(name)
        return fn if callable(fn) else None
    if hasattr(adapter, name):
        fn = getattr(adapter, name)
        return fn if callable(fn) else None
    if name == "obs" and callable(adapter):
        return adapter
    return None


def _wrap_env_with_adapter(env: Any, adapter: Any) -> Any:
    obs_adapter = _resolve_adapter_fn(adapter, "obs")
    action_adapter = _resolve_adapter_fn(adapter, "action")
    reward_adapter = _resolve_adapter_fn(adapter, "reward")
    if obs_adapter is None and action_adapter is None and reward_adapter is None:
        return env
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    class _WrappedEnv(gym.Wrapper):
        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple):
                obs, info = result
                if obs_adapter is not None:
                    obs = obs_adapter(obs)
                return obs, info
            obs = result
            if obs_adapter is not None:
                obs = obs_adapter(obs)
            return obs

        def step(self, action):
            if action_adapter is not None:
                action = action_adapter(action)
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                if obs_adapter is not None:
                    obs = obs_adapter(obs)
                if reward_adapter is not None:
                    reward = reward_adapter(reward)
                return obs, reward, terminated, truncated, info
            obs, reward, done, info = result
            if obs_adapter is not None:
                obs = obs_adapter(obs)
            if reward_adapter is not None:
                reward = reward_adapter(reward)
            return obs, reward, done, info

    return _WrappedEnv(env)


def _get_git_commit(project_root: Path) -> Optional[str]:
    head_path = project_root / ".git" / "HEAD"
    if not head_path.exists():
        return None
    content = head_path.read_text(encoding="utf-8").strip()
    if content.startswith("ref: "):
        ref_path = project_root / ".git" / content.split(" ", 1)[1].strip()
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip() or None
    return content or None


def _resolve_training_budgets(
    *,
    steps: int,
    update_steps: Optional[int],
    collect_steps_per_cycle: int,
    train_steps_per_cycle: int,
) -> Tuple[int, int]:
    collector_steps = max(1, int(collect_steps_per_cycle))
    updates_per_cycle = max(1, int(train_steps_per_cycle))

    if update_steps is not None and int(update_steps) > 0:
        total_updates = int(update_steps)
        collect_cycles = max(
            1,
            int(math.ceil(float(total_updates) / float(updates_per_cycle))),
        )
        expected_env_steps = int(collect_cycles * collector_steps)
        return int(total_updates), int(expected_env_steps)

    requested_env_steps = max(1, int(steps))
    collect_cycles = max(
        1,
        int(math.ceil(float(requested_env_steps) / float(collector_steps))),
    )
    total_updates = int(collect_cycles * updates_per_cycle)
    expected_env_steps = int(requested_env_steps)
    return int(total_updates), int(expected_env_steps)


def run_train(
    args: Any,
    input_spec: Optional[Dict[str, Any]] = None,
    adapter: Any = None,
) -> Dict[str, Any]:
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return {
                key: _json_safe(val)
                for key, val in asdict(value).items()
            }
        if isinstance(value, SimpleNamespace):
            return {
                key: _json_safe(val)
                for key, val in vars(value).items()
            }
        if isinstance(value, dict):
            return {
                str(key): _json_safe(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return _json_safe(value.to_dict())
        if hasattr(value, "value"):
            try:
                return _json_safe(value.value)
            except Exception:
                pass
        return repr(value)

    args = _normalize_args(args)
    input_spec = input_spec or {}

    log_level = logging.DEBUG if bool(getattr(args, "verbose", False)) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    logging.getLogger("aletheia").setLevel(logging.DEBUG if bool(getattr(args, "verbose", False)) else logging.INFO)

    env_id = input_spec.get("env_id", getattr(args, "env", DEFAULT_ENV))
    device = getattr(args, "device", "auto")
    seed = int(getattr(args, "seed", DEFAULT_SEED))
    render = bool(getattr(args, "render", False))
    env_factory = input_spec.get("env_factory")
    eval_env_factory = input_spec.get("eval_env_factory", env_factory)
    env = input_spec.get("env")
    eval_env = input_spec.get("eval_env")

    if env is None:
        if env_factory is not None:
            env = env_factory()
        else:
            try:
                import gymnasium as gym
            except ImportError:
                import gym
            env = gym.make(env_id, render_mode="human" if render else None)

    if eval_env is None:
        if eval_env_factory is not None:
            eval_env = eval_env_factory()
        else:
            try:
                import gymnasium as gym
            except ImportError:
                import gym
            eval_env = gym.make(env_id)

    if input_spec.get("dataset_path"):
        raise NotImplementedError(
            "input_spec['dataset_path'] is not implemented yet; run_train currently supports only online env inputs"
        )

    if adapter is not None:
        env = _wrap_env_with_adapter(env, adapter)
        if eval_env is not env:
            eval_env = _wrap_env_with_adapter(eval_env, adapter)

    overrides = _parse_overrides(getattr(args, "overrides", None))
    overrides.setdefault("rl", {})
    overrides["rl"].setdefault("use_alpha_adaptive", True)
    overrides["rl"].setdefault("use_entropy_protection", False)
    if "CartPole" in str(env_id) and bool(getattr(args, "imagination_only", False)):
        overrides.setdefault("imag_continue_prob_cap", 0.95)
        overrides["rl"].setdefault("detach_critic_features_on_imagination", True)
        overrides["rl"].setdefault("use_actor_drift_guard", True)
        overrides["rl"].setdefault("slow_value_reg_drift_gain", 2.0)
        overrides.setdefault(
            "rssm_msc",
            {
                "enabled": True,
                "horizons": [1, 4, 8],
                "loss_scale": 0.25,
            },
        )
        overrides.setdefault(
            "rssm_shortcut_consistency",
            {
                "enabled": True,
                "horizons": [2, 4],
                "loss_scale": 0.25,
                "sample_ratio": 0.5,
                "max_starts": 4,
            },
        )
        # Keep the early imagined manifold locally aligned to replay teacher space.
        overrides.setdefault("adaptive_imag_policy_open_loop_consistency_weight", 0.15)
        overrides.setdefault("adaptive_imag_policy_open_loop_consistency_horizon", 3)
        overrides.setdefault("adaptive_imag_policy_open_loop_consistency_delta", 0.5)
        overrides.setdefault("adaptive_imag_policy_open_loop_consistency_high_value_boost", 1.0)
        overrides.setdefault("adaptive_imag_policy_open_loop_consistency_high_value_quantile", 0.75)
        # Reuse the existing target-ruler path as a late-stage single-ruler default
        # so online critic / actor-use imagined states keep a shared value scale.
        overrides.setdefault("adaptive_imag_actor_use_target_value_ruler_enabled", True)
        overrides.setdefault("adaptive_imag_actor_use_target_value_ruler_blend", 1.0)
        overrides.setdefault("adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled", True)
        overrides.setdefault("adaptive_imag_actor_use_target_value_ruler_gap_margin", 1.0)
        overrides.setdefault(
            "adaptive_imag_actor_use_target_value_ruler_critic_distill_weight",
            0.15,
        )
        # Default actor-side corridor maintenance now uses a clean replay-grounded
        # target in inflation zones instead of only damping the polluted ruler.
        overrides.setdefault("adaptive_imag_idle_corridor_advantage_blend_max", 0.5)
        overrides.setdefault("adaptive_imag_idle_corridor_negative_adv_threshold", 1.0)
        overrides.setdefault("adaptive_imag_idle_corridor_negative_adv_tau", 2.0)
        overrides.setdefault("adaptive_imag_idle_corridor_quantile", 0.75)
        overrides.setdefault("adaptive_imag_idle_corridor_adv_term_clamp_min", 0.25)
        overrides.setdefault("adaptive_imag_idle_corridor_clean_target_blend_max", 0.8)
        overrides.setdefault(
            "adaptive_imag_idle_corridor_clean_target_inflation_floor_max", 0.2
        )
        overrides.setdefault("adaptive_imag_idle_corridor_inflation_threshold", 5.0)
        overrides.setdefault("adaptive_imag_idle_corridor_inflation_tau", 3.0)
        overrides.setdefault("value_real_anchor_use_mc_returns", True)
        overrides.setdefault("adaptive_imag_critic_semantic_anchor_enabled", True)
        overrides.setdefault("adaptive_imag_critic_semantic_anchor_weight", 0.1)
        overrides.setdefault("adaptive_imag_critic_semantic_anchor_clean_scale", 1.0)
        overrides.setdefault("adaptive_imag_critic_semantic_anchor_external_scale", 1.0)
        overrides.setdefault("adaptive_imag_critic_semantic_anchor_authority_floor", 0.05)
        overrides.setdefault("adaptive_imag_task_corridor_enabled", True)
        overrides.setdefault("adaptive_imag_task_corridor_high_quantile", 0.75)
        overrides.setdefault("adaptive_imag_task_corridor_low_quantile", 0.25)
        overrides.setdefault("adaptive_imag_task_corridor_gate_tau", 0.25)
        overrides.setdefault("adaptive_imag_task_corridor_analytic_floor", 0.35)
        overrides.setdefault("adaptive_imag_task_corridor_confidence_scale", 0.5)
        overrides.setdefault("adaptive_imag_actor_contract_trust_floor", 0.2)
        overrides.setdefault("adaptive_imag_actor_unified_contract_enabled", False)
        overrides.setdefault("adaptive_imag_actor_unified_contract_blend", 1.0)
        overrides.setdefault("adaptive_imag_actor_unified_contract_tail_relief_max", 0.0)
        overrides.setdefault(
            "adaptive_imag_actor_unified_contract_tail_relief_adv_threshold", 1.0
        )
        overrides.setdefault(
            "adaptive_imag_actor_unified_contract_tail_relief_adv_tau", 2.0
        )
        overrides.setdefault(
            "adaptive_imag_critic_bootstrap_contact_persistence_decay", 0.95
        )
        overrides.setdefault(
            "adaptive_imag_critic_bootstrap_contact_persistence_floor", 0.35
        )
        overrides.setdefault(
            "adaptive_imag_critic_bootstrap_external_takeover_floor_ratio", 0.5
        )
        overrides.setdefault(
            "adaptive_imag_critic_bootstrap_anchor_contact_priority_bonus", 2.0
        )
        overrides.setdefault(
            "adaptive_imag_critic_bootstrap_dense_contact_priority_floor", 0.35
        )
        overrides.setdefault(
            "adaptive_imag_critic_bootstrap_anchor_contact_cap", 1.0
        )

    factory_overrides = _extract_run_train_factory_overrides(overrides)

    try:
        if getattr(args, "load", None):
            agent = load_agent(getattr(args, "load"), env=env, device=device)
            logger.info("Loaded agent from %s", getattr(args, "load"))
        else:
            agent = create_agent(
                env,
                config_overrides=factory_overrides,
                device=device,
                seed=seed,
            )
            logger.info("Created new agent")
    except Exception as exc:
        logger.error("Failed to create/load agent: %s", exc)
        import traceback
        traceback.print_exc()
        try:
            env.close()
        except Exception:
            pass
        if eval_env is not env:
            try:
                eval_env.close()
            except Exception:
                pass
        return {"status": 1, "error": str(exc)}

    from .aletheia_train import (
        TrainingCheckpointRestorePolicy,
        TrainingLoop,
        TrainingStateManager,
        build_training_model,
        create_replay_buffer,
        isolate_policy_wm_state,
    )
    from .contracts.telemetry import BOOTSTRAP_CONTRACT_SUMMARY_KEYS

    if bool(getattr(args, "eval_only", False)):
        results = _evaluate_agent(
            eval_env,
            agent,
            episodes=int(getattr(args, "eval_episodes", 10)),
            max_steps=int(getattr(args, "eval_max_steps", 1000)),
        )
        print(f"eval_reward={results['mean']:.2f}  {results['std']:.2f}", flush=True)
        try:
            env.close()
        except Exception:
            pass
        if eval_env is not env:
            try:
                eval_env.close()
            except Exception:
                pass
        return {"status": 0, "eval": results}

    collector_steps = max(1, int(getattr(args, "collect_steps_per_cycle", 32)))
    train_steps_per_cycle = max(1, int(getattr(args, "train_steps_per_cycle", 4)))
    from .aletheia_config import TrainingConfig

    total_updates, expected_env_steps = _resolve_training_budgets(
        steps=int(getattr(args, "steps", DEFAULT_STEPS)),
        update_steps=getattr(args, "update_steps", None),
        collect_steps_per_cycle=collector_steps,
        train_steps_per_cycle=train_steps_per_cycle,
    )

    wm_pretrain = int(total_updates * float(getattr(args, "pretrain_ratio", PRETRAIN_RATIO)))
    warmup = int(total_updates * float(getattr(args, "warmup_ratio", WARMUP_RATIO)))
    imag_ratio_start = float(getattr(args, "imag_ratio_start", 0.0))
    imag_ratio_end = float(getattr(args, "imag_ratio_end", 0.4))
    imag_ratio_max = float(getattr(args, "imag_ratio_max", 0.6))
    imag_ratio_ramp_steps = int(getattr(args, "imag_ratio_ramp_steps", 2000))

    if "CartPole" in str(env_id):
        if bool(getattr(args, "imagination_only", False)):
            wm_pretrain = min(wm_pretrain, 32)
            warmup = min(warmup, 32)
            overrides.setdefault("imag_continue_prob_cap", 0.95)
            overrides["rl"].setdefault("detach_critic_features_on_imagination", True)
            overrides["rl"].setdefault("use_actor_drift_guard", True)
            overrides["rl"].setdefault("slow_value_reg_drift_gain", 2.0)
        else:
            wm_pretrain = min(wm_pretrain, 60)
            imag_ratio_end = min(imag_ratio_end, 0.25)
            imag_ratio_max = min(imag_ratio_max, 0.30)
            imag_ratio_ramp_steps = max(imag_ratio_ramp_steps, 5000)

    if wm_pretrain + warmup > total_updates:
        wm_pretrain = max(0, total_updates // 3)
        warmup = max(0, total_updates // 3)

    seq_len = max(1, int(getattr(args, "wm_seq_len", 16)))
    train_config = TrainingConfig(
        config_mode="strict",
        validation_mode="strict",
        total_steps=int(total_updates),
        num_train_steps=int(total_updates),
        total_env_steps=int(expected_env_steps),
        wm_pretrain_steps=int(wm_pretrain),
        warmup_steps=int(warmup),
        batch_size=int(getattr(args, "wm_batch_size", 64)),
        seq_len=int(seq_len),
        wm_seq_len=int(seq_len),
        wm_batch_size=int(getattr(args, "wm_batch_size", 64)),
        imagination_horizon=int(getattr(args, "imagination_horizon", 15)),
        imagination_batch_size=int(getattr(args, "imagination_batch_size", 8)),
        rl_batch_size=int(getattr(args, "rl_batch_size", 64)),
        lambda_value_real_anchor=0.0,
        log_interval=int(getattr(args, "log_interval", LOG_INTERVAL)),
        eval_interval=int(getattr(args, "eval_interval", EVAL_INTERVAL)),
        save_interval=int(getattr(args, "save_interval", SAVE_INTERVAL)),
        disable_eval=not bool(getattr(args, "enable_eval", False)),
        disable_checkpoint=getattr(args, "save", None) is None,
        seed=int(getattr(args, "seed", DEFAULT_SEED)),
        device=str(getattr(args, "device", "auto")),
        imagination_only=bool(getattr(args, "imagination_only", False)),
        imag_ratio_start=imag_ratio_start,
        imag_ratio_end=imag_ratio_end,
        imag_ratio_ramp_steps=imag_ratio_ramp_steps,
        imag_ratio_max=imag_ratio_max,
        wm_real_only=True,
        buffer_capacity=int(getattr(args, "buffer_capacity", BUFFER_CAPACITY)),
    )
    total_updates = int(train_config.total_steps)
    expected_env_steps = int(train_config.total_env_steps)

    rejected_legacy_override_keys: list[str] = []
    ignored_phase_override_keys = {
        "adaptive_imag_post_entry_negative_online_adv_threshold",
        "adaptive_imag_post_entry_negative_online_adv_eval_threshold",
        "adaptive_imag_post_entry_negative_online_adv_analytic_scale",
        "adaptive_imag_post_entry_actor_base_return_cap_margin",
    }
    rejected_phase_override_keys = {
        "adaptive_imag_post_entry_commit_highwater_gap_max",
        "adaptive_imag_post_entry_commit_signed_adv_threshold",
        "adaptive_imag_post_entry_commit_highwater_hold_steps",
        "adaptive_imag_post_entry_commit_confirmation_steps",
        "adaptive_imag_post_entry_commit_highwater_eval_threshold",
    }
    for key in (
        "imagination_only",
        "critic_mode",
        "critic_gammas",
        "critic_n_ensemble",
        "critic_hidden_dim",
        "critic_primary_gamma_index",
        "critic_use_twohot",
        "critic_use_adaptive_routing",
        "critic_pessimism",
        "critic_target_update_tau",
        "wm_pretrain_steps",
        "warmup_steps",
        "enable_mixed_scheduler",
        "wm_real_only",
        "imagination_continue_threshold",
        "imagination_hard_truncate",
        "imag_ratio_start",
        "imag_ratio_end",
        "imag_ratio_max",
        "imag_ratio_ramp_steps",
        "imag_ratio_wm_guard",
        "imag_ratio_critic_guard",
        "imag_ratio_guard_floor",
        "imag_return_delta_clip",
        "imag_continue_prob_cap",
        "rssm_continue_temperature",
        "rssm_continue_loss_type",
        "rssm_continue_focal_alpha",
        "rssm_continue_focal_gamma",
        "rssm_continue_optimistic_bias",
        "rssm_continue_positive_weight",
        "rssm_continue_negative_weight",
        "adaptive_imag_continue_cap",
        "adaptive_imag_continue_cap_min",
        "adaptive_imag_continue_cap_max",
        "adaptive_imag_continue_cap_target_gap",
        "adaptive_imag_continue_cap_target_continue",
        "adaptive_imag_continue_cap_target_actor",
        "adaptive_imag_continue_cap_gap_gain",
        "adaptive_imag_continue_cap_continue_gain",
        "adaptive_imag_continue_cap_actor_gain",
        "adaptive_imag_continue_cap_ema",
        "adaptive_imag_continue_cap_warmup_steps",
        "adaptive_imag_continue_cap_ramp_steps",
        "adaptive_imag_eval_confirmation_count",
        "adaptive_imag_entry_window_steps",
        "adaptive_imag_entry_target_continue",
        "adaptive_imag_entry_continue_gain_scale",
        "adaptive_imag_compensation_trigger_return_delta_clip",
        "adaptive_imag_compensation_trigger_min_pressure",
        "adaptive_imag_compensation_trigger_actor_scale_gain",
        "adaptive_imag_compensation_trigger_actor_scale_floor",
        "adaptive_imag_compensation_trigger_hysteresis",
        "adaptive_imag_compensation_trigger_release_ratio",
        "adaptive_imag_compensation_trigger_attack_ema",
        "adaptive_imag_compensation_trigger_release_ema",
        "adaptive_imag_compensation_trigger_confirmation_steps",
        "adaptive_imag_compensation_trigger_quality_gate",
        "adaptive_imag_compensation_trigger_quality_gap_threshold",
        "adaptive_imag_compensation_trigger_quality_actor_threshold",
        "adaptive_imag_compensation_trigger_quality_streak",
        "adaptive_imag_compensation_trigger_quality_scale_gain",
        "adaptive_imag_compensation_trigger_quality_scale_floor",
        "adaptive_imag_compensation_trigger_quality_piecewise",
        "adaptive_imag_compensation_trigger_quality_mild_threshold",
        "adaptive_imag_compensation_trigger_quality_severe_threshold",
        "adaptive_imag_compensation_trigger_quality_mild_floor",
        "adaptive_imag_compensation_trigger_quality_severe_floor",
        "adaptive_imag_late_trigger_rescue_enabled",
        "adaptive_imag_late_trigger_rescue_min_release_progress",
        "adaptive_imag_late_trigger_rescue_negative_adv_threshold",
        "adaptive_imag_late_trigger_rescue_value_gap_threshold",
        "adaptive_imag_late_trigger_rescue_actor_scale",
        "adaptive_imag_late_trigger_rescue_critic_boost",
        "adaptive_imag_late_trigger_rescue_latch_steps",
        "adaptive_imag_late_trigger_target_base_blend",
        "adaptive_imag_late_trigger_target_base_min_release_progress",
        "adaptive_imag_late_trigger_base_return_cap_margin",
        "adaptive_imag_late_trigger_base_return_cap_min_release_progress",
        "adaptive_imag_late_trigger_base_return_cap_prebuild_real_adv_threshold",
        "adaptive_imag_compensation_persistence_release_tail_late_trigger_prebuild_gate_bypass_steps",
        "adaptive_imag_compensation_persistence_release_tail_late_trigger_prebuild_gate_bypass_negative_online_adv_threshold",
        "adaptive_imag_compensation_persistence_release_tail_late_trigger_prebuild_gate_bypass_sustain_latch_steps",
        "adaptive_imag_late_trigger_base_return_cap_critic_boost",
        "adaptive_imag_late_trigger_base_return_cap_value_real_anchor_scale",
        "adaptive_imag_" + "".join(("con", "troller")) + "_staging",
        "adaptive_imag_pretrigger_continue_threshold",
        "adaptive_imag_pretrigger_gap_threshold",
        "adaptive_imag_pretrigger_return_ema_threshold",
        "adaptive_imag_pretrigger_cap",
        "adaptive_imag_pretrigger_actor_scale",
        "adaptive_imag_post_entry_eval_threshold",
        "adaptive_imag_real_stability_use_pre_eval_registry_support",
        "adaptive_imag_task_cert_real_policy_anchor_gate_enabled",
        "adaptive_imag_task_cert_real_policy_anchor_gate_kl_scale",
        "adaptive_imag_post_entry_internal_return_ema_threshold",
        "adaptive_imag_post_entry_internal_min_step",
        "adaptive_imag_post_entry_internal_continue_threshold",
        "adaptive_imag_post_entry_internal_gap_max",
        "adaptive_imag_post_entry_internal_hold_steps",
        "adaptive_imag_post_entry_internal_cap",
        "adaptive_imag_post_entry_internal_actor_scale",
        "adaptive_imag_handoff_hold_steps",
        "adaptive_imag_handoff_cap",
        "adaptive_imag_handoff_actor_scale",
        "adaptive_imag_post_entry_cap",
        "adaptive_imag_post_entry_actor_scale",
        "adaptive_imag_post_entry_soft_cap",
        "adaptive_imag_post_entry_soft_cap_floor",
        "adaptive_imag_post_entry_soft_actor_scale",
        "adaptive_imag_post_entry_soft_actor_scale_floor",
        "adaptive_imag_post_entry_soft_trigger_release_steps",
        "adaptive_imag_post_entry_soft_trigger_bad_quality_max_progress",
        "adaptive_imag_standard_soft_fallback_trigger_release_steps",
        "adaptive_imag_standard_soft_fallback_trigger_release_quality_clamp",
        "adaptive_imag_post_entry_pending_cap",
        "adaptive_imag_post_entry_pending_actor_scale",
        "adaptive_imag_post_entry_pending_negative_adv_threshold",
        "adaptive_imag_post_entry_pending_negative_adv_actor_scale",
        "adaptive_imag_post_entry_pending_negative_adv_critic_boost",
        "adaptive_imag_post_entry_soft_negative_adv_actor_scale",
        "adaptive_imag_post_entry_soft_negative_adv_critic_boost",
        "adaptive_imag_post_entry_soft_negative_adv_midwater_eval_threshold",
        "adaptive_imag_post_entry_soft_negative_adv_midwater_min_step",
        "adaptive_imag_post_entry_soft_negative_adv_midwater_ramp_steps",
        "adaptive_imag_post_entry_soft_negative_adv_midwater_actor_scale",
        "adaptive_imag_post_entry_soft_negative_adv_midwater_critic_boost",
        "adaptive_imag_post_entry_soft_negative_adv_highwater_eval_threshold",
        "adaptive_imag_post_entry_soft_negative_adv_highwater_actor_scale",
        "adaptive_imag_post_entry_soft_negative_adv_highwater_critic_boost",
        "adaptive_imag_compensation_persistence_negative_adv_threshold",
        "adaptive_imag_compensation_persistence_negative_adv_actor_scale",
        "adaptive_imag_compensation_persistence_negative_adv_critic_boost",
        "adaptive_imag_compensation_persistence_escape_negative_adv_actor_scale",
        "adaptive_imag_compensation_persistence_escape_negative_adv_critic_boost",
        "adaptive_imag_post_entry_hold_steps",
        "adaptive_imag_post_entry_rearm_delta",
        "adaptive_imag_post_entry_eval_confirmation_count",
        "adaptive_imag_post_entry_preview_min_step",
        "adaptive_imag_post_entry_preview_hold_steps",
        "adaptive_imag_post_entry_preview_improve_margin",
        "adaptive_imag_post_entry_allow_persistence_escape",
        "adaptive_imag_post_entry_commit_continue_threshold",
        "adaptive_imag_post_entry_commit_gap_max",
        "adaptive_imag_post_entry_commit_return_threshold",
        "adaptive_imag_post_entry_commit_highwater_eval_threshold",
        "adaptive_imag_post_entry_commit_highwater_gap_max",
        "adaptive_imag_post_entry_commit_signed_adv_threshold",
        "adaptive_imag_post_entry_commit_highwater_hold_steps",
        "adaptive_imag_post_entry_commit_confirmation_steps",
        "adaptive_imag_compensation_persistence_escape_band_release_steps",
        "adaptive_imag_compensation_persistence_eval_threshold",
        "adaptive_imag_compensation_persistence_requires_post_entry_commit",
        "adaptive_imag_compensation_persistence_min_step",
        "adaptive_imag_compensation_persistence_eval_drop",
        "adaptive_imag_compensation_persistence_continue_threshold",
        "adaptive_imag_compensation_persistence_adv_threshold",
        "adaptive_imag_compensation_persistence_use_relative",
        "adaptive_imag_compensation_persistence_ema",
        "adaptive_imag_compensation_persistence_gap_scale",
        "adaptive_imag_compensation_persistence_continue_delta",
        "adaptive_imag_compensation_persistence_adv_delta",
        "adaptive_imag_compensation_persistence_cap",
        "adaptive_imag_compensation_persistence_actor_scale",
        "adaptive_imag_compensation_persistence_highwater_eval_threshold",
        "adaptive_imag_compensation_persistence_highwater_actor_scale",
        "adaptive_imag_compensation_persistence_highwater_cap",
        "adaptive_imag_compensation_persistence_highwater_tail_target_gap_cap",
        "adaptive_imag_compensation_persistence_highwater_tail_weight_floor",
        "adaptive_imag_compensation_persistence_highwater_tail_weight_floor_ratio",
        "adaptive_imag_compensation_persistence_highwater_tail_critic_boost",
        "adaptive_imag_compensation_persistence_highwater_tail_critic_continue_margin",
        "adaptive_imag_compensation_persistence_highwater_tail_critic_real_adv_threshold",
        "adaptive_imag_compensation_persistence_highwater_tail_critic_real_adv_margin",
        "adaptive_imag_compensation_persistence_hold_steps",
        "adaptive_imag_compensation_persistence_eval_confirmation_count",
        "adaptive_imag_compensation_persistence_drop_confirmation_count",
        "adaptive_imag_compensation_persistence_entry_protect_steps",
        "adaptive_imag_compensation_persistence_entry_protect_eval_ratio",
        "adaptive_imag_compensation_persistence_release_eval_ratio",
        "adaptive_imag_compensation_persistence_release_steps",
        "adaptive_imag_compensation_persistence_release_cap",
        "adaptive_imag_compensation_persistence_release_actor_scale",
        "adaptive_imag_compensation_persistence_real_advantage_threshold",
        "adaptive_imag_compensation_persistence_real_advantage_imag_threshold",
        "adaptive_imag_compensation_persistence_real_advantage_critic_boost",
        "adaptive_imag_compensation_persistence_real_value_anchor_scale",
        "adaptive_imag_compensation_persistence_tail_window",
        "adaptive_imag_compensation_persistence_tail_gap_threshold",
        "adaptive_imag_compensation_persistence_tail_continue_threshold",
        "adaptive_imag_compensation_persistence_tail_weight_threshold",
        "adaptive_imag_compensation_persistence_tail_target_gap_cap",
        "adaptive_imag_compensation_persistence_tail_weight_floor",
        "adaptive_imag_compensation_persistence_tail_weight_floor_ratio",
        "adaptive_imag_compensation_persistence_tail_critic_boost",
        "adaptive_imag_compensation_post_solved_eval_threshold",
        "adaptive_imag_compensation_post_solved_min_step",
        "adaptive_imag_compensation_post_solved_requires_persistence",
        "adaptive_imag_compensation_post_solved_cap",
        "adaptive_imag_compensation_post_solved_actor_scale",
        "adaptive_imag_compensation_post_solved_hold_steps",
        "adaptive_imag_compensation_post_solved_eval_confirmation_count",
        "adaptive_imag_compensation_post_solved_highwater_eval_threshold",
        "adaptive_imag_compensation_post_solved_highwater_cap",
        "adaptive_imag_compensation_post_solved_highwater_actor_scale",
        "adaptive_imag_compensation_post_solved_actor_anchor_eval_threshold",
        "adaptive_imag_compensation_post_solved_actor_anchor_pull",
        "adaptive_imag_compensation_post_solved_actor_anchor_hard_pull",
        "adaptive_imag_compensation_post_solved_actor_anchor_latched_pull",
        "adaptive_imag_compensation_post_solved_actor_anchor_kl",
        "adaptive_imag_compensation_post_solved_actor_anchor_latched_kl",
        "adaptive_imag_compensation_post_solved_actor_anchor_highwater_eval_threshold",
        "adaptive_imag_compensation_post_solved_actor_anchor_highwater_pull",
        "adaptive_imag_compensation_post_solved_actor_anchor_highwater_kl",
        "adaptive_imag_compensation_post_solved_real_actor_anchor_kl",
        "adaptive_imag_compensation_post_solved_real_advantage_threshold",
        "adaptive_imag_compensation_post_solved_real_advantage_latch_veto",
        "adaptive_imag_compensation_post_solved_real_advantage_latch_veto_threshold",
        "adaptive_imag_compensation_post_solved_real_advantage_actor_scale",
        "adaptive_imag_compensation_post_solved_real_advantage_critic_boost",
        "adaptive_imag_compensation_post_solved_real_value_anchor_scale",
        "adaptive_imag_compensation_post_solved_actor_target_base_blend",
        "adaptive_imag_global_target_base_blend",
        "adaptive_imag_target_value_consistency_weight",
        "adaptive_imag_target_value_consistency_horizon",
        "adaptive_imag_target_value_consistency_delta",
        "adaptive_imag_target_value_consistency_high_value_boost",
        "adaptive_imag_target_value_consistency_high_value_quantile",
        "adaptive_imag_target_value_consistency_high_value_feature_scale",
        "adaptive_imag_policy_open_loop_consistency_weight",
        "adaptive_imag_policy_open_loop_consistency_horizon",
        "adaptive_imag_policy_open_loop_consistency_delta",
        "adaptive_imag_policy_open_loop_consistency_high_value_boost",
        "adaptive_imag_policy_open_loop_consistency_high_value_quantile",
        "adaptive_imag_actor_use_target_value_ruler_enabled",
        "adaptive_imag_actor_use_target_value_ruler_blend",
        "adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled",
        "adaptive_imag_actor_use_target_value_ruler_gap_margin",
        "adaptive_imag_actor_use_target_value_ruler_critic_distill_weight",
        "adaptive_imag_post_entry_actor_base_return_cap_margin",
        "adaptive_imag_post_entry_base_return_cap_critic_boost",
        "adaptive_imag_post_entry_base_return_cap_value_real_anchor_scale",
        "adaptive_imag_compensation_persistence_actor_base_return_cap_margin",
        "adaptive_imag_compensation_persistence_base_return_cap_critic_boost",
        "adaptive_imag_compensation_persistence_base_return_cap_value_real_anchor_scale",
        "adaptive_imag_compensation_persistence_release_actor_base_return_cap_margin",
        "adaptive_imag_compensation_persistence_release_base_return_cap_critic_boost",
        "adaptive_imag_compensation_persistence_release_base_return_cap_value_real_anchor_scale",
        "adaptive_imag_compensation_post_solved_actor_base_return_cap_margin",
        "adaptive_imag_compensation_post_solved_base_return_cap_critic_boost",
        "adaptive_imag_compensation_post_solved_base_return_cap_value_real_anchor_scale",
        "adaptive_imag_idle_corridor_advantage_blend_max",
        "adaptive_imag_idle_corridor_negative_adv_threshold",
        "adaptive_imag_idle_corridor_negative_adv_tau",
        "adaptive_imag_idle_corridor_quantile",
        "adaptive_imag_idle_corridor_adv_term_clamp_scale",
        "adaptive_imag_idle_corridor_adv_term_clamp_min",
        "adaptive_imag_idle_corridor_clean_target_blend_max",
        "adaptive_imag_idle_corridor_clean_target_inflation_floor_max",
        "adaptive_imag_idle_corridor_inflation_threshold",
        "adaptive_imag_idle_corridor_inflation_tau",
        "adaptive_imag_task_corridor_enabled",
        "adaptive_imag_task_corridor_high_quantile",
        "adaptive_imag_task_corridor_low_quantile",
        "adaptive_imag_task_corridor_gate_tau",
        "adaptive_imag_task_corridor_analytic_floor",
        "adaptive_imag_task_corridor_confidence_scale",
        "adaptive_imag_actor_contract_trust_floor",
        "adaptive_imag_actor_unified_contract_enabled",
        "adaptive_imag_actor_unified_contract_blend",
        "adaptive_imag_actor_unified_contract_tail_relief_max",
        "adaptive_imag_actor_unified_contract_tail_relief_adv_threshold",
        "adaptive_imag_actor_unified_contract_tail_relief_adv_tau",
        "adaptive_imag_critic_bootstrap_contract_enabled",
        "adaptive_imag_critic_bootstrap_clean_mix_max",
        "adaptive_imag_critic_bootstrap_external_takeover_floor_ratio",
        "adaptive_imag_critic_bootstrap_semantic_debt_decay",
        "adaptive_imag_critic_bootstrap_contact_persistence_decay",
        "adaptive_imag_critic_bootstrap_contact_persistence_floor",
        "adaptive_imag_critic_bootstrap_anchor_confidence_floor",
        "adaptive_imag_critic_bootstrap_anchor_contact_priority_bonus",
        "adaptive_imag_critic_bootstrap_dense_contact_priority_floor",
        "adaptive_imag_critic_bootstrap_anchor_contact_cap",
        "adaptive_imag_critic_bootstrap_min_step",
        "adaptive_imag_critic_bootstrap_step_ramp",
        "adaptive_imag_critic_bootstrap_eval_threshold",
        "adaptive_imag_critic_bootstrap_eval_ramp",
        "adaptive_imag_post_entry_negative_online_adv_threshold",
        "adaptive_imag_post_entry_negative_online_adv_eval_threshold",
        "adaptive_imag_post_entry_negative_online_adv_analytic_scale",
        "adaptive_imag_late_trigger_negative_online_adv_threshold",
        "adaptive_imag_late_trigger_negative_online_adv_analytic_scale",
        "adaptive_imag_late_trigger_negative_online_adv_min_release_progress",
        "adaptive_imag_compensation_trigger_persistence_handoff_enabled",
        "adaptive_imag_compensation_trigger_persistence_handoff_min_release_progress",
        "adaptive_imag_compensation_trigger_persistence_handoff_gap_threshold",
        "adaptive_imag_compensation_trigger_persistence_handoff_continue_threshold",
        "adaptive_imag_compensation_trigger_persistence_handoff_eval_threshold",
        "adaptive_imag_compensation_trigger_persistence_handoff_block_after_late_trigger_base_cap_steps",
        "adaptive_imag_compensation_trigger_persistence_handoff_confirmation_steps_after_late_trigger_base_cap",
        "adaptive_imag_compensation_post_solved_negative_online_adv_threshold",
        "adaptive_imag_compensation_post_solved_negative_online_adv_analytic_scale",
        "adaptive_imag_compensation_post_solved_critic_anchor_weight",
        "adaptive_imag_compensation_post_solved_drift_damping_eval_threshold",
        "adaptive_imag_compensation_post_solved_drift_damping_wm_scale",
        "adaptive_imag_compensation_post_solved_drift_damping_critic_scale",
        "adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_actor_scale",
        "adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_critic_boost",
        "adaptive_imag_compensation_post_solved_negative_adv_threshold",
        "adaptive_imag_compensation_post_solved_negative_adv_actor_scale",
        "adaptive_imag_compensation_post_solved_negative_adv_critic_boost",
        "adaptive_imag_compensation_post_solved_negative_adv_latch_steps",
        "adaptive_imag_compensation_post_solved_negative_adv_latch_actor_scale",
        "adaptive_imag_compensation_post_solved_negative_adv_latch_critic_boost",
        "adaptive_imag_negative_adv_guard_enabled",
        "adaptive_imag_negative_adv_guard_threshold",
        "adaptive_imag_negative_adv_actor_scale",
        "adaptive_imag_negative_adv_critic_boost",
        "early_stop_eval_mean",
        "early_stop_eval_patience",
        "lambda_value_real_anchor",
        "value_real_anchor_use_mc_returns",
        "value_real_anchor_corridor_quantile",
        "adaptive_imag_critic_semantic_anchor_enabled",
        "adaptive_imag_critic_semantic_anchor_weight",
        "adaptive_imag_critic_semantic_anchor_clean_scale",
        "adaptive_imag_critic_semantic_anchor_external_scale",
        "adaptive_imag_critic_semantic_anchor_authority_floor",
        "adaptive_imag_policy_open_loop_consistency_value_scale",
        "adaptive_imag_policy_open_loop_consistency_late_step_boost",
        "imagination_horizon",
        "imagination_horizon_min",
    ):
        if key in overrides:
            if key in ignored_phase_override_keys:
                continue
            if key in rejected_phase_override_keys:
                rejected_legacy_override_keys.append(key)
                continue
            if is_legacy_compensation_phase_field(key):
                rejected_legacy_override_keys.append(key)
                continue
            setattr(train_config, key, overrides[key])

    if rejected_legacy_override_keys:
        rejected_legacy_override_keys = sorted(set(rejected_legacy_override_keys))
        raise ValueError(
            "Legacy phase overrides are no longer supported: "
            + ", ".join(rejected_legacy_override_keys)
        )

    opt_overrides = overrides.get("optimizer", {})
    for key in ("learning_rate", "weight_decay"):
        if key in opt_overrides:
            setattr(train_config.optimizer, key, opt_overrides[key])

    rl_overrides = overrides.get("rl", {})
    train_config.rl.imag_gradient = str(getattr(args, "imag_gradient", getattr(train_config.rl, "imag_gradient", "dynamics")))
    train_config.rl.imag_gradient_mix = float(getattr(args, "imag_gradient_mix", 0.0))
    train_config.rl.actor_analytic_weight = float(getattr(args, "actor_analytic_weight", getattr(train_config.rl, "actor_analytic_weight", 1.0)))
    train_config.rl.actor_reinforce_aux_weight_discrete = float(getattr(args, "actor_reinforce_aux_weight_discrete", getattr(train_config.rl, "actor_reinforce_aux_weight_discrete", 0.1)))
    train_config.rl.use_reward_ema = bool(getattr(args, "use_reward_ema", True))
    for key in (
        "use_alpha_adaptive",
        "use_entropy_protection",
        "target_entropy",
        "entropy_coef",
        "actor_entropy_scale",
        "actor_analytic_weight",
        "actor_reinforce_aux_weight_discrete",
        "use_value_normalization",
        "use_advantage_normalization",
        "imag_gradient",
        "imag_gradient_mix",
        "use_reward_ema",
        "use_fixed_gamma",
        "detach_critic_features_on_imagination",
        "slow_value_reg_weight",
        "slow_value_reg_drift_threshold",
        "slow_value_reg_drift_gain",
        "use_actor_drift_guard",
        "actor_drift_guard_threshold",
        "actor_drift_guard_gain",
        "actor_drift_guard_floor",
        "actor_drift_guard_imag_only",
    ):
        if key in rl_overrides:
            setattr(train_config.rl, key, rl_overrides[key])

    if "use_alpha_adaptive" not in rl_overrides:
        train_config.rl.use_alpha_adaptive = True
    if "use_entropy_protection" not in rl_overrides:
        train_config.rl.use_entropy_protection = False

    if (
        getattr(agent, "buffer", None) is None
        or int(getattr(agent.buffer, "capacity", 0))
        != int(train_config.buffer_capacity)
    ):
        agent.buffer = create_replay_buffer(
            capacity=int(train_config.buffer_capacity),
            store_obs=False,
        )
    effective_training_config = copy.deepcopy(train_config.to_dict())
    agent_bootstrap_bundle = agent.config.to_agent_bootstrap_bundle()

    artifact_paths = RunArtifactPaths.from_save_arg(getattr(args, "save", None))
    save_dir = artifact_paths.save_dir
    final_ckpt_path = artifact_paths.final_ckpt_path
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    eval_history: List[Dict[str, Any]] = []
    eval_history_path = artifact_paths.eval_history_path
    train_metrics_path = artifact_paths.train_metrics_path
    config_path = artifact_paths.config_path
    summary_path = artifact_paths.summary_path
    best_trainer_path = artifact_paths.best_trainer_path
    if config_path is not None:
        project_root = Path(__file__).resolve().parents[1]
        config_payload = build_resolved_config_payload(
            saved_at_utc=datetime.now(timezone.utc).isoformat(),
            project_root=project_root,
            git_commit=_get_git_commit(project_root),
            argv=list(getattr(args, "argv", None) or sys.argv),
            args=vars(args),
            overrides_effective=overrides,
            effective_training_config=copy.deepcopy(effective_training_config),
            agent_bootstrap_bundle=copy.deepcopy(agent_bootstrap_bundle),
            profile=getattr(agent, "profile", None),
        )
        config_path.write_text(
            json.dumps(
                _json_safe(config_payload),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    last_print_time = [time.time()]
    train_t0 = time.time()
    latest_train_metric_record: Dict[str, Any] = {}

    def log_fn(metrics: Dict[str, float], step: int = 0) -> None:
        if "eval" not in metrics and train_metrics_path is not None:
            normalized_record = build_train_metric_record(
                metrics,
                int(step),
                train_t0=train_t0,
                now=time.time(),
            )
            latest_train_metric_record.clear()
            latest_train_metric_record.update(normalized_record)
            append_jsonl_record(train_metrics_path, normalized_record)

        now = time.time()
        if now - last_print_time[0] < 0.1:
            return
        last_print_time[0] = now
        if "eval" in metrics and isinstance(metrics["eval"], dict):
            ev = metrics["eval"]
            msg = (
                f"EVAL step={int(step):6d} | "
                f"mean={float(ev.get('mean_return', 0.0)):7.2f} | "
                f"std={float(ev.get('std_return', 0.0)):7.2f} | "
                f"min={float(ev.get('min_return', 0.0)):7.2f} | "
                f"max={float(ev.get('max_return', 0.0)):7.2f}"
            )
            print(msg, flush=True)
            return
        update_step_val = metrics.get(
            "train/update_step",
            metrics.get("train/global_step", metrics.get("step", step)),
        )
        env_step_val = metrics.get("train/env_steps_collected", None)
        ret = metrics.get("train/episode_return_ema", metrics.get("episode_return_ema", 0.0))
        l_wm = metrics.get("loss_wm", metrics.get("loss/loss_wm", 0.0))
        l_actor = metrics.get("loss_actor", metrics.get("loss/loss_actor", 0.0))
        l_critic = metrics.get("loss_critic", metrics.get("loss/loss_critic", 0.0))
        v_mean = metrics.get("critic/values_mean", 0.0)
        r_mean = metrics.get("critic/returns_mean", 0.0)
        a_mean = metrics.get("critic/adv_mean", 0.0)
        buf = getattr(agent, "buffer", None)
        buffer_size = 0
        try:
            if buf is not None:
                if hasattr(buf, "__len__"):
                    buffer_size = int(len(buf))
                elif hasattr(buf, "episodes"):
                    buffer_size = int(len(getattr(buf, "episodes")))
                elif hasattr(buf, "size"):
                    buffer_size = int(getattr(buf, "size"))
        except Exception:
            buffer_size = 0
        source = metrics.get("train/batch_source", "unknown")
        ratio = metrics.get("train/imag_ratio", 0.0)
        step_prefix = f"STEP upd={int(update_step_val):6d}"
        if env_step_val is not None:
            step_prefix += f" | env={int(env_step_val):6d}"
        msg = (
            f"{step_prefix} | [{source} {ratio:.2f}] | "
            f"Return: {float(ret):7.2f} | "
            f"L_wm: {float(l_wm):7.4f} | "
            f"L_act: {float(l_actor):7.4f} | "
            f"L_crit: {float(l_critic):7.4f} | "
            f"Vmu: {float(v_mean):7.3f} | "
            f"Rmu: {float(r_mean):7.3f} | "
            f"Amu: {float(a_mean):7.3f} | "
            f"Buffer: {int(buffer_size):5d}"
        )
        print(msg, flush=True)

    best_eval_mean = [-float("inf")]
    best_eval_step = [0]
    solved_target = float(getattr(train_config, "early_stop_eval_mean", 0.0))
    solved_patience = max(1, int(getattr(train_config, "early_stop_eval_patience", 1)))
    solved_streak = [0]
    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None and os.path.exists(resume_from):
        try:
            resumed_state = TrainingStateManager.load(
                str(resume_from),
                train_config,
                agent.device,
                trusted_source=True,
                restore_policy=TrainingCheckpointRestorePolicy(
                    restore_model=False,
                    restore_optimizers=False,
                    restore_buffer=False,
                    optimizer_restore_mode="skip",
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to pre-load resume metadata from {resume_from}: {exc}"
            ) from exc
        else:
            if resumed_state.best_eval_return > -float("inf"):
                best_eval_mean[0] = float(resumed_state.best_eval_return)
                best_eval_step[0] = int(resumed_state.best_step)
            resume_step = int(getattr(resumed_state, "global_step", 0))
            if eval_history_path is not None and eval_history_path.exists():
                eval_history = trim_history_records_for_resume(
                    eval_history_path,
                    resume_step=resume_step,
                    normalize_record=normalize_eval_record,
                )
            if train_metrics_path is not None and train_metrics_path.exists():
                trim_history_records_for_resume(
                    train_metrics_path,
                    resume_step=resume_step,
                    normalize_record=normalize_train_metric_record,
                )

    def eval_fn(_model, step: int) -> Dict[str, float]:
        del _model
        prev_real_anchor = getattr(agent, "_real_stability_eval_anchor_actor", None)
        had_prev_real_anchor = hasattr(agent, "_real_stability_eval_anchor_actor")
        prev_real_registry = getattr(agent, "_real_stability_eval_anchor_registry_actors", None)
        had_prev_real_registry = hasattr(agent, "_real_stability_eval_anchor_registry_actors")
        setattr(
            agent,
            "_real_stability_eval_anchor_actor",
            getattr(loop, "_real_stability_certified_anchor_actor", None),
        )
        setattr(
            agent,
            "_real_stability_eval_anchor_registry_actors",
            list(getattr(loop, "_real_stability_certified_anchor_registry_actors", [])),
        )
        try:
            results = _evaluate_agent(
                eval_env,
                agent,
                episodes=int(getattr(args, "eval_episodes", 10)),
                max_steps=int(getattr(args, "eval_max_steps", 1000)),
            )
        finally:
            if had_prev_real_anchor:
                setattr(agent, "_real_stability_eval_anchor_actor", prev_real_anchor)
            elif hasattr(agent, "_real_stability_eval_anchor_actor"):
                delattr(agent, "_real_stability_eval_anchor_actor")
            if had_prev_real_registry:
                setattr(agent, "_real_stability_eval_anchor_registry_actors", prev_real_registry)
            elif hasattr(agent, "_real_stability_eval_anchor_registry_actors"):
                delattr(agent, "_real_stability_eval_anchor_registry_actors")
        mean_reward = float(results.get("mean", 0.0))
        telemetry = dict(results.get("telemetry", {}) or {})
        if hasattr(loop, "set_external_eval_feedback"):
            loop.set_external_eval_feedback(
                mean_reward,
                step=int(step),
                telemetry=telemetry,
            )
        if mean_reward > best_eval_mean[0]:
            best_eval_mean[0] = mean_reward
            best_eval_step[0] = int(step)
            best_path = artifact_paths.best_ckpt_path if save_dir is not None else final_ckpt_path
            if best_path is not None:
                agent.save(
                    str(best_path),
                    effective_training_config=effective_training_config,
                    agent_bootstrap_bundle=agent_bootstrap_bundle,
                )
            if best_trainer_path is not None:
                best_trainer_state = TrainingStateManager.create(
                    config=train_config,
                    device=agent.device,
                )
                best_trainer_state.sync_from_loop(
                    loop,
                    best_eval_return=best_eval_mean[0],
                    best_step=best_eval_step[0],
                )
                TrainingStateManager.save(
                    best_trainer_state,
                    str(best_trainer_path),
                    **resume_state_save_args,
                )
            logger.info(
                "New best eval %.2f at step %d%s",
                mean_reward,
                int(step),
                f", saved to {best_path}" if best_path is not None else "",
            )
        record = build_eval_record(
            results=results,
            step=int(step),
            telemetry=telemetry,
            agent_step_count=getattr(agent, "step_count", None),
        )
        eval_history.append(record)
        append_jsonl_record(eval_history_path, record)
        stop_training = False
        if solved_target > 0.0 and mean_reward >= solved_target:
            solved_streak[0] += 1
            stop_training = solved_streak[0] >= solved_patience
        else:
            solved_streak[0] = 0
        return {
            "mean_return": record["mean"],
            "std_return": record["std"],
            "min_return": record["min"],
            "max_return": record["max"],
            "mean_length": record["mean_length"],
            "stop_training": stop_training,
            "solved_streak": int(solved_streak[0]),
        }

    loop_model = build_training_model(
        actor=agent.actor,
        critic=agent.critic,
        world_model=agent.world_model,
        router=agent.router,
        device=agent.device,
    )

    loop = TrainingLoop(
        env=env,
        buffer=agent.buffer,
        model=loop_model,
        will=agent.will,
        config=train_config,
        device=agent.device,
        logger_fn=log_fn,
        save_dir=str(save_dir) if save_dir is not None else None,
        collect_steps_per_cycle=collector_steps,
        train_steps_per_cycle=train_steps_per_cycle,
    )
    resume_state_save_args = {
        "model": loop.model,
        "opt_bundle": loop.opt_bundle,
        "buffer": getattr(loop, "buffer", None),
        "effective_training_config": effective_training_config,
    }

    def _build_resume_restore_policy() -> TrainingCheckpointRestorePolicy:
        raw_layers = str(getattr(args, "resume_restore_layers", "all") or "all").strip().lower()
        valid_layers = {"training_state", "model", "optimizers", "buffer"}
        if raw_layers in {"", "all"}:
            enabled_layers = set(valid_layers)
        elif raw_layers == "none":
            enabled_layers = set()
        else:
            enabled_layers = {
                item.strip()
                for item in raw_layers.split(",")
                if item.strip()
            }
            unknown_layers = sorted(enabled_layers - valid_layers)
            if unknown_layers:
                raise ValueError(
                    "Unsupported resume restore layers: "
                    + ", ".join(unknown_layers)
                )
        return TrainingCheckpointRestorePolicy(
            restore_training_state="training_state" in enabled_layers,
            restore_model="model" in enabled_layers,
            restore_optimizers="optimizers" in enabled_layers,
            restore_buffer="buffer" in enabled_layers,
            model_restore_mode=str(
                getattr(args, "resume_model_restore_mode", "strict") or "strict"
            ).strip().lower(),
            optimizer_restore_mode=str(
                getattr(args, "resume_optimizer_restore_mode", "strict") or "strict"
            ).strip().lower(),
            legacy_steps_since_collect_mode=str(
                getattr(args, "resume_legacy_steps_since_collect_mode", "strict")
                or "strict"
            ).strip().lower(),
        )

    resume_restore_policy = _build_resume_restore_policy()

    def save_fn(_model, step: int) -> None:
        del _model
        if save_dir is None:
            return
        ckpt_path = artifact_paths.periodic_checkpoint_path(int(step))
        trainer_path = artifact_paths.periodic_trainer_state_path(int(step))
        latest_resume_path = artifact_paths.resume_latest_path
        agent.save(
            str(ckpt_path),
            effective_training_config=effective_training_config,
            agent_bootstrap_bundle=agent_bootstrap_bundle,
        )
        trainer_state = TrainingStateManager.create(config=train_config, device=agent.device)
        trainer_state.sync_from_loop(
            loop,
            best_eval_return=best_eval_mean[0],
            best_step=best_eval_step[0],
        )
        TrainingStateManager.save(
            trainer_state,
            str(trainer_path),
            **resume_state_save_args,
        )
        TrainingStateManager.save(
            trainer_state,
            str(latest_resume_path),
            **resume_state_save_args,
        )
        logger.info("Saved periodic checkpoint to %s", str(ckpt_path))
        logger.info("Saved trainer resume state to %s", str(trainer_path))

    collector = create_agent_rollout_collector(
        agent=agent,
        env=env,
        max_steps=int(getattr(args, "eval_max_steps", 1000)),
    )

    try:
        loop.run(
            num_steps=int(total_updates),
            data_collector=collector,
            resume_from=resume_from,
            resume_restore_policy=resume_restore_policy,
            eval_fn=eval_fn if bool(getattr(args, "enable_eval", False)) else None,
            save_fn=save_fn if save_dir is not None else None,
        )
        stats = {
            "total_steps": int(total_updates),
            "episode_count": getattr(loop, "episode_count", 0),
        }
    except KeyboardInterrupt:
        stats = {
            "total_steps": int(total_updates),
            "episode_count": getattr(loop, "episode_count", 0),
        }
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        import traceback
        traceback.print_exc()
        try:
            env.close()
        except Exception:
            pass
        if eval_env is not env:
            try:
                eval_env.close()
            except Exception:
                pass
        return {"status": 1, "error": str(exc)}

    if final_ckpt_path is not None:
        agent.save(
            str(final_ckpt_path),
            effective_training_config=effective_training_config,
            agent_bootstrap_bundle=agent_bootstrap_bundle,
        )
        final_trainer_state = TrainingStateManager.create(
            config=train_config,
            device=agent.device,
        )
        final_trainer_state.sync_from_loop(
            loop,
            best_eval_return=best_eval_mean[0],
            best_step=best_eval_step[0],
        )
        TrainingStateManager.save(
            final_trainer_state,
            str(artifact_paths.trainer_state_final_path),
            **resume_state_save_args,
        )
        TrainingStateManager.save(
            final_trainer_state,
            str(artifact_paths.resume_latest_path),
            **resume_state_save_args,
        )
        if best_eval_mean[0] > -float("inf"):
            logger.info(
                "Best eval during training: %.2f at step %d",
                best_eval_mean[0],
                best_eval_step[0],
            )

    final_current = _evaluate_agent(
        eval_env,
        agent,
        episodes=int(getattr(args, "eval_episodes", 10)),
        max_steps=int(getattr(args, "eval_max_steps", 1000)),
    )
    print(f"final_eval_reward={final_current['mean']:.2f}  {final_current['std']:.2f}", flush=True)

    final_best = None
    best_path = artifact_paths.best_ckpt_path if save_dir is not None else None
    if best_path is not None and best_path.exists():
        try:
            best_agent = load_agent(str(best_path), env=eval_env, device=device)
            final_best = _evaluate_agent(
                eval_env,
                best_agent,
                episodes=int(getattr(args, "eval_episodes", 10)),
                max_steps=int(getattr(args, "eval_max_steps", 1000)),
            )
        except Exception:
            final_best = None

    final_model_source = "current"
    if (
        final_best is not None
        and float(final_best.get("mean", -float("inf"))) > float(final_current.get("mean", -float("inf")))
    ):
        final_current = final_best
        final_model_source = "best"
        if final_ckpt_path is not None and best_path is not None and best_path.exists():
            shutil.copy2(best_path, final_ckpt_path)
            logger.info("Promoted best checkpoint to final artifact: %s", str(final_ckpt_path))

    if summary_path is not None:
        summary = build_summary_payload(
            saved_at_utc=datetime.now(timezone.utc).isoformat(),
            env_id=env_id,
            args=args,
            seed=seed,
            device=device,
            collector_steps=collector_steps,
            train_steps_per_cycle=train_steps_per_cycle,
            total_updates=total_updates,
            expected_env_steps=expected_env_steps,
            effective_training_config=copy.deepcopy(effective_training_config),
            agent_bootstrap_bundle=copy.deepcopy(agent_bootstrap_bundle),
            paths=artifact_paths,
            final_current=final_current,
            final_best=final_best,
            final_model_source=final_model_source,
            best_eval_mean=best_eval_mean[0],
            best_eval_step=best_eval_step[0],
            eval_history=eval_history,
            stats=stats,
            latest_train_metric_record=latest_train_metric_record,
            bootstrap_contract_summary_keys=BOOTSTRAP_CONTRACT_SUMMARY_KEYS,
            loop=loop,
        )
        summary_path.write_text(
            json.dumps(
                _json_safe(summary),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        summary = None

    try:
        env.close()
    except Exception:
        pass
    if eval_env is not env:
        try:
            eval_env.close()
        except Exception:
            pass

    return {
        "status": 0,
        "summary": summary,
        "eval_history": eval_history,
        "final_current": final_current,
        "final_best": final_best,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Lightweight CLI for quick training / evaluation runs."""
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Aletheia v{__version__} - Quick Run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--env", type=str, default=DEFAULT_ENV, help="Environment ID")
    parser.add_argument("--steps", type=int, default=10_000, help="Target environment steps")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--eval-only", action="store_true", help="Evaluation mode only")
    parser.add_argument("--load", type=str, default=None, help="Load checkpoint")
    parser.add_argument("--save", type=str, default=None, help="Save checkpoint directory or file")
    parser.add_argument("--resume-from", dest="resume_from", type=str, default=None, help="Resume from a trainer-state checkpoint")
    parser.add_argument(
        "--resume-model-restore-mode",
        type=str,
        default="strict",
        choices=["strict"],
        help="Model restore mode for trainer-state resume",
    )
    parser.add_argument(
        "--resume-optimizer-restore-mode",
        type=str,
        default="strict",
        choices=["auto", "strict", "skip"],
        help="Optimizer restore mode for trainer-state resume",
    )
    parser.add_argument(
        "--resume-restore-layers",
        type=str,
        default="all",
        help="Comma-separated resume layers: training_state,model,optimizers,buffer; use all or none",
    )
    parser.add_argument(
        "--resume-legacy-steps-since-collect-mode",
        type=str,
        default="strict",
        choices=["strict", "infer"],
        help="How trainer-state resume handles legacy checkpoints missing steps_since_collect metadata",
    )
    parser.add_argument("--enable-eval", action="store_true", help="Enable evaluation during training")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--eval-max-steps", type=int, default=1000, help="Max steps per eval episode")
    parser.add_argument("--collect-steps-per-cycle", type=int, default=32, help="Environment steps collected per train cycle")
    parser.add_argument("--train-steps-per-cycle", type=int, default=4, help="Gradient updates per train cycle")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args(argv)
    args.argv = list(sys.argv if argv is None else [sys.argv[0], *argv])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print(f"Aletheia v{__version__} - {VERSION_NAME}")
    result = run_train(args)
    return int(result.get("status", 1))


if __name__ == "__main__":
    sys.exit(main())


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core entry points
    "create_agent",
    "load_agent",
    "run_train",
    "AgentHandle",
    "AgentRolloutCollector",
    "create_agent_rollout_collector",
    # Configuration
    "ConfigBundle",
    "TrainParams",
    "EnvProfile",
    "AgentProfile",
    "RSSMProfile",
    "PerceptorProfile",
    "CriticProfile",
    "WillProfile",
    "TrainProfile",
    "get_base_profile",
    # Factory and container
    "AgentFactory",
    "AgentContainer",
    "ComponentRegistry",
    # Environment
    "EnvWrapper",
    # Adapters
    "BufferAdapter",
    # Utility functions
    "resolve_device",
    "set_global_seed",
    "extract_env_profile",
    "_prepare_obs_for_buffer",
    # Constants
    "RSSM_DETER_MIN",
    "RSSM_DETER_MAX",
    "RSSM_STOCH_MIN",
    "RSSM_STOCH_MAX",
    "HIDDEN_DIM_MIN",
    "HIDDEN_DIM_MAX",
    "CTRL_DIM_MIN",
    "CTRL_DIM_MAX",
    "ROUTER_Z_EMBED_DIM",
    "ROUTER_PROJ_DIM_MIN",
    "ROUTER_PROJ_DIM_MAX",
    # Version
    "__version__",
    "VERSION_NAME",
    # CLI
    "main",
]
