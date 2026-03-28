from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import warnings
import copy
import numpy as np
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple, Literal, Union, List, Set

from ._wire_schema import (
    ConfigPolicyOverrides,
    FactoryBridgeOverrides,
    export_config_policy_wire_overrides,
    export_factory_bridge_overrides,
    parse_config_policy_overrides,
    parse_factory_bridge_overrides,
)

logger = logging.getLogger("aletheia.config")

# ═══════════════════════════════════════════════════════════
#  全局常量
# ═══════════════════════════════════════════════════════════
DEFAULT_STEPS: int = 100_000
DEFAULT_SEED: int = 42
DEFAULT_ENV: str = "CartPole-v1"
DEFAULT_BATCH_LENGTH: int = 32
DEFAULT_HORIZON: int = 10
PRETRAIN_RATIO: float = 0.1
WARMUP_RATIO: float = 0.1
LOG_INTERVAL: int = 100
EVAL_INTERVAL: int = 1_000
SAVE_INTERVAL: int = 5_000
BUFFER_CAPACITY: int = 1_000
EVAL_SEED_OFFSET: int = 100

CONTRACT_CERTIFICATION_FIELDS: Tuple[str, ...] = (
    "adaptive_imag_task_corridor_enabled",
    "adaptive_imag_task_corridor_high_quantile",
    "adaptive_imag_task_corridor_low_quantile",
    "adaptive_imag_task_corridor_gate_tau",
    "adaptive_imag_task_corridor_analytic_floor",
    "adaptive_imag_task_corridor_confidence_scale",
    "adaptive_imag_task_cert_reward_agreement_quantile",
    "adaptive_imag_task_cert_recovery_rate",
    "adaptive_imag_task_cert_alarm_rate",
    "adaptive_imag_task_cert_state_floor",
)

CONTRACT_AUTHORITY_FIELDS: Tuple[str, ...] = (
    "adaptive_imag_actor_contract_trust_floor",
    "adaptive_imag_actor_unified_contract_enabled",
    "adaptive_imag_actor_unified_contract_blend",
    "adaptive_imag_actor_unified_contract_tail_relief_max",
    "adaptive_imag_actor_unified_contract_tail_relief_adv_threshold",
    "adaptive_imag_actor_unified_contract_tail_relief_adv_tau",
    "adaptive_imag_critic_bootstrap_contract_enabled",
    "adaptive_imag_critic_bootstrap_clean_mix_max",
    "adaptive_imag_critic_bootstrap_external_takeover_floor_ratio",
    "adaptive_imag_critic_bootstrap_precontact_floor_max",
    "adaptive_imag_critic_bootstrap_precontact_frontload",
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
)

CONTRACT_CONSUMER_FIELDS: Tuple[str, ...] = (
    "adaptive_imag_compensation_trigger_persistence_handoff_enabled",
    "adaptive_imag_compensation_trigger_persistence_handoff_min_release_progress",
    "adaptive_imag_compensation_trigger_persistence_handoff_gap_threshold",
    "adaptive_imag_compensation_trigger_persistence_handoff_continue_threshold",
    "adaptive_imag_compensation_trigger_persistence_handoff_eval_threshold",
    "adaptive_imag_compensation_trigger_persistence_handoff_landing_guard_enabled",
    "adaptive_imag_compensation_trigger_persistence_handoff_block_after_late_trigger_base_cap_steps",
    "adaptive_imag_compensation_trigger_persistence_handoff_confirmation_steps_after_late_trigger_base_cap",
)

CONTRACT_COMPENSATION_FIELDS: Tuple[str, ...] = (
    "adaptive_imag_compensation_persistence_eval_threshold",
    "adaptive_imag_compensation_persistence_cap",
    "adaptive_imag_compensation_persistence_actor_scale",
    "adaptive_imag_compensation_persistence_hold_steps",
    "adaptive_imag_compensation_persistence_negative_adv_threshold",
    "adaptive_imag_compensation_persistence_negative_adv_actor_scale",
    "adaptive_imag_compensation_persistence_negative_adv_critic_boost",
    "adaptive_imag_compensation_post_solved_requires_persistence",
    "adaptive_imag_compensation_post_solved_hold_steps",
    "adaptive_imag_compensation_post_solved_negative_adv_threshold",
    "adaptive_imag_compensation_post_solved_negative_adv_actor_scale",
    "adaptive_imag_compensation_post_solved_negative_adv_critic_boost",
)

LEGACY_COMPENSATION_PHASE_FIELD_PREFIXES: Tuple[str, ...] = (
    "adaptive_imag_pretrigger_",
    "adaptive_imag_entry_probe_",
    "adaptive_imag_handoff_",
    "adaptive_imag_compensation_persistence_release_",
    "adaptive_imag_late_trigger_rescue_",
)

LEGACY_COMPENSATION_PHASE_FIELDS: Tuple[str, ...] = (
    "adaptive_imag_" + "".join(("con", "troller")) + "_staging",
)


def is_legacy_compensation_phase_field(name: str) -> bool:
    training_config_cls = globals().get("TrainingConfig")
    if training_config_cls is not None and name in getattr(training_config_cls, "__annotations__", {}):
        return False
    return name in LEGACY_COMPENSATION_PHASE_FIELDS or any(
        name.startswith(prefix) for prefix in LEGACY_COMPENSATION_PHASE_FIELD_PREFIXES
    )

# ═══════════════════════════════════════════════════════════
#  验证 & 同步工具
# ═══════════════════════════════════════════════════════════

def _validate(obj, *, positive=(), non_neg=(), ranges=(), choices=()):
    """批量字段验证，取代分散的 assert / raise 样板。"""
    for n in positive:
        v = getattr(obj, n, 0)
        if v <= 0:
            raise ValueError(f"{n} must be positive, got {v}")
    for n in non_neg:
        v = getattr(obj, n, 0)
        if v < 0:
            raise ValueError(f"{n} must be non-negative, got {v}")
    for n, lo, hi in ranges:
        v = getattr(obj, n, 0)
        if not (lo <= v <= hi):
            raise ValueError(f"{n} must be in [{lo}, {hi}], got {v}")
    for n, opts in choices:
        v = getattr(obj, n, None)
        if v not in opts:
            raise ValueError(f"{n} must be one of {opts}, got {v}")


# [FIX-1] 防止双向调用产生歧义：记录已同步的 frozenset 对
_SYNCED_PAIRS: Set[frozenset] = set()


def _sync_fields(
    obj,
    primary: str,
    secondary: str,
    *,
    force_align: bool = True,
    handler=None,
    _synced: Optional[Set[frozenset]] = None,
):
    """
    同步两个冗余字段——填充缺失值 / 冲突时按策略处理。

    [FIX-1] 增加 _synced 集合防护：同一对字段在同一次 __post_init__
            中只同步一次，避免 (A→B, B→A) 双向调用时第二次覆盖结果。
    """
    pair_key = frozenset((primary, secondary))
    tracker = _synced if _synced is not None else _SYNCED_PAIRS
    if pair_key in tracker:
        return
    tracker.add(pair_key)

    va, vb = getattr(obj, primary), getattr(obj, secondary)
    if va <= 0 and vb > 0:
        setattr(obj, primary, int(vb))
    elif vb <= 0 and va > 0:
        setattr(obj, secondary, int(va))
    elif va > 0 and vb > 0 and va != vb:
        action = f"using {primary}" if force_align else "keep both"
        msg = f"{primary} ({va}) != {secondary} ({vb}), {action}"
        proceed = handler(msg) if handler else True
        if proceed and force_align:
            setattr(obj, secondary, int(va))


# ═══════════════════════════════════════════════════════════
#  枚举
# ═══════════════════════════════════════════════════════════


class EnvType(Enum):
    """环境类型"""
    SIMPLE_VECTOR = "simple_vector"
    COMPLEX_VECTOR = "complex_vector"
    IMAGE = "image"
    MIXED = "mixed"

    @classmethod
    def from_obs_shape(cls, obs_shape: Tuple[int, ...]) -> "EnvType":
        if len(obs_shape) == 1:
            return cls.SIMPLE_VECTOR if obs_shape[0] <= 32 else cls.COMPLEX_VECTOR
        if len(obs_shape) == 3:
            return cls.IMAGE
        return cls.MIXED


# ═══════════════════════════════════════════════════════════
#  基础子配置
# ═══════════════════════════════════════════════════════════

@dataclass
class SymlogConfig:
    """Symlog 变换"""
    use_symlog_obs: bool = False
    use_symlog_reward: bool = True
    use_symlog_value: bool = True
    use_symlog_recon: bool = False
    input_clip: float = 1e6
    symexp_clip: float = 20.0
    output_clip: float = 1e6


@dataclass
class DistributionConfig:
    """
    Categorical 分布配置

    stoch_dim = num_distributions × num_classes（默认 32×32 = 1024）
    """
    dist_type: str = "categorical"
    num_classes: int = 32
    num_distributions: int = 32
    temperature_train: float = 1.0
    temperature_final: float = 0.5
    temperature_eval: float = 0.5
    use_temperature_schedule: bool = True
    temperature_decay_steps: int = 50000
    unimix_ratio: float = 0.01
    use_unimix: bool = True

    @property
    def stoch_dim(self) -> int:
        return self.num_classes * self.num_distributions

    @property
    def effective_dist_type(self) -> str:
        if self.dist_type == "categorical" and self.use_unimix:
            return "mixed_categorical"
        return self.dist_type


@dataclass
class KLConfig:
    """KL 散度"""
    free_nats: float = 1.0
    use_balance: bool = True
    balance_alpha: float = 0.8
    kl_weight: float = 1.0
    use_warmup: bool = False
    warmup_steps: int = 10000
    warmup_start_weight: float = 0.0


@dataclass
class InitConfig:
    """统一初始化"""
    method: Literal["orthogonal", "xavier", "kaiming"] = "orthogonal"
    backbone_gain: float = 1.0
    output_gain: float = 1.0
    bias_init: Literal["zeros", "uniform"] = "zeros"


@dataclass
class TemperatureConfig:
    """RSSM latent / Gumbel-Softmax 温度"""
    initial: float = 1.0
    final: float = 0.5
    steps: int = 100000
    decay_mode: str = "linear"


# ═══════════════════════════════════════════════════════════
#  环境相关
# ═══════════════════════════════════════════════════════════

@dataclass
class EnvConfig:
    """环境配置容器"""
    env_name: str = ""
    obs_shape: Tuple[int, ...] = (4,)
    action_dim: int = 2
    is_discrete: bool = True
    env_type: EnvType = EnvType.SIMPLE_VECTOR
    max_episode_steps: int = 500


@dataclass
class EnvProfile:
    """环境画像，驱动 ConfigPolicy 决策"""
    env_id: str
    obs_shape: Tuple[int, ...]
    obs_modality: str
    obs_dim: int
    action_dim: int
    is_discrete_action: bool
    env_category: str
    temporal_dependency: str
    is_pomdp: bool
    reward_variance_level: str
    max_episode_steps: int

    @property
    def is_image(self) -> bool:
        return self.obs_modality == "image"

    @property
    def is_discrete(self) -> bool:
        return self.is_discrete_action

    @property
    def is_simple(self) -> bool:
        return (not self.is_image) and self.obs_dim <= 64 and self.action_dim <= 8


# ── 环境解析辅助 ──

def _resolve_spec(env: Any) -> Any:
    spec = getattr(env, "spec", None)
    if spec is None:
        base = getattr(env, "env", None)
        if base is not None:
            spec = getattr(base, "spec", None)
    return spec


def _resolve_env_name(env: Any) -> str:
    spec_obj = _resolve_spec(env)
    if spec_obj is not None and hasattr(spec_obj, "id"):
        return spec_obj.id
    return str(getattr(env, "env_name", "unknown"))


def _resolve_spaces(env: Any) -> Tuple[Any, Any]:
    obs_s = getattr(env, "observation_space", None)
    act_s = getattr(env, "action_space", None)
    if obs_s is None or act_s is None:
        base = getattr(env, "env", None)
        if base is not None:
            obs_s = obs_s or getattr(base, "observation_space", None)
            act_s = act_s or getattr(base, "action_space", None)
    if obs_s is None or act_s is None:
        raise AttributeError("环境必须提供 observation_space 和 action_space")
    return obs_s, act_s


def extract_env_config(env_or_name: Union[str, Any]) -> EnvConfig:
    """从环境实例或名称提取 EnvConfig"""
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    if isinstance(env_or_name, str):
        env, env_name, should_close = gym.make(env_or_name), env_or_name, True
    else:
        env, env_name, should_close = env_or_name, _resolve_env_name(env_or_name), False

    try:
        obs_space, act_space = _resolve_spaces(env)
        if hasattr(act_space, "n"):
            action_dim, is_discrete = act_space.n, True
        else:
            action_dim, is_discrete = act_space.shape[0], False
        spec_obj = _resolve_spec(env)
        max_steps = getattr(spec_obj, "max_episode_steps", 500) if spec_obj else 500
        return EnvConfig(
            env_name=env_name, obs_shape=obs_space.shape,
            action_dim=action_dim, is_discrete=is_discrete,
            env_type=EnvType.from_obs_shape(obs_space.shape),
            max_episode_steps=max_steps,
        )
    finally:
        if should_close:
            env.close()


# 环境类型 → 复杂度等级（时间依赖 / 奖励方差共用）
_ENV_COMPLEXITY_MAP = {
    EnvType.SIMPLE_VECTOR: "low",
    EnvType.COMPLEX_VECTOR: "medium",
    EnvType.IMAGE: "high",
    EnvType.MIXED: "medium",
}


def extract_env_profile(env_or_name: Union[str, Any]) -> EnvProfile:
    """从环境提取完整 EnvProfile"""
    cfg = extract_env_config(env_or_name)
    obs_modality = (
        "vector" if len(cfg.obs_shape) == 1
        else "image" if len(cfg.obs_shape) == 3
        else "multi"
    )
    complexity = _ENV_COMPLEXITY_MAP.get(cfg.env_type, "medium")
    return EnvProfile(
        env_id=cfg.env_name, obs_shape=cfg.obs_shape,
        obs_modality=obs_modality,
        obs_dim=int(np.prod(cfg.obs_shape)) if cfg.obs_shape else 0,
        action_dim=cfg.action_dim, is_discrete_action=cfg.is_discrete,
        env_category=cfg.env_type.value.upper(),
        temporal_dependency=complexity, is_pomdp=False,
        reward_variance_level=complexity,
        max_episode_steps=cfg.max_episode_steps,
    )


# ═══════════════════════════════════════════════════════════
#  Actor / ActionCodec
# ═══════════════════════════════════════════════════════════

@dataclass
class ContinuousDistConfig:
    """连续动作分布"""
    init_logstd: float = 0.0
    min_logstd: float = -10.0
    max_logstd: float = 2.0
    logstd_mode: Literal["state_independent", "state_dependent"] = "state_independent"
    tanh_clip: float = 0.999
    entropy_samples: int = 100
    loc_clip: float = 10.0
    scale_min: float = 1e-4
    scale_max: float = 10.0


@dataclass
class DiscreteDistConfig:
    """离散动作分布"""
    min_temperature: float = 0.1
    max_temperature: float = 10.0
    unimix_ratio: float = 0.01
    logits_clip: float = 8.0
    min_prob: float = 0.01


@dataclass
class EntropyProtectionConfig:
    """熵保护"""
    enabled: bool = True
    initial_target: float = 0.5
    floor: float = 0.05
    decay: float = 0.999
    adaptive_alpha: bool = True
    alpha_lr: float = 3e-4
    collapse_threshold: float = 0.02
    recovery_boost: float = 0.3
    initial_log_alpha: float = -5.0


@dataclass
class ActorConfig:
    """Actor 主配置"""
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: Literal["silu", "relu", "gelu", "elu", "tanh"] = "silu"
    layer_norm: bool = True
    dropout: float = 0.0
    init: InitConfig = field(default_factory=InitConfig)
    continuous: ContinuousDistConfig = field(default_factory=ContinuousDistConfig)
    discrete: DiscreteDistConfig = field(default_factory=DiscreteDistConfig)
    intent_dim: int = 0
    entropy_protection: EntropyProtectionConfig = field(
        default_factory=EntropyProtectionConfig
    )


@dataclass
class ActionCodecConfig:
    """动作编解码器（简单控制 + 网络配置合并）"""
    enabled: bool = False
    clip_range: float = 1.0
    embed_dim: int = 64
    hidden_dims: Tuple[int, ...] = (128,)
    use_layer_norm: bool = True


# ═══════════════════════════════════════════════════════════
#  Critic
# ═══════════════════════════════════════════════════════════

CriticMode = Literal["single", "double", "ensemble", "adaptive"]


@dataclass
class GammaSchedulerConfig:
    """γ 退火调度器"""
    base_gammas: Tuple[float, ...] = (0.9, 0.95, 0.99, 0.997)
    warmup_steps: int = 10000
    min_ratio: float = 0.9


@dataclass
class CriticConfig:
    """统一 Critic 配置（Single → Adaptive Ensemble 全模式）"""
    d_feature: int = 640
    intent_dim: int = 0
    mode: CriticMode = "double"
    gammas: Tuple[float, ...] = (0.9, 0.95, 0.99, 0.997)
    n_ensemble: int = 3
    pessimism: float = 0.25
    hidden_dim: int = 512
    hidden_depth: int = 3
    layer_norm: bool = True
    activation: str = "silu"
    use_target_critic: bool = True   # [v5.5] DreamerV3: always use slow EMA target
    target_update_tau: float = 0.005   # [v5.5] 每步软更新，0.005更合适 (0.02等效于~50步; 0.005等效于~200步衰减)
    use_anchor_critic: bool = False
    anchor_hidden_dim: int = 256
    anchor_outputs_symlog: bool = True
    huber_delta: float = 1.0
    symlog_clip: float = 20.0
    use_adaptive_routing: bool = False
    adaptive_hidden_dim: int = 64
    adaptive_temp_range: Tuple[float, float] = (1.0, 0.1)
    adaptive_steps: int = 50000
    lambda_route: float = 1.0
    lambda_load_balance: float = 0.1
    lambda_entropy: float = 1e-3
    route_teacher_beta: float = 1.0
    learning_rate: float = 3e-4
    primary_gamma_index: int = -1
    # --- Twohot Discrete Regression (DreamerV3) ---
    use_twohot: bool = True
    twohot_num_bins: int = 255
    twohot_vmin: float = -20.0
    twohot_vmax: float = 20.0

    def __post_init__(self):
        self.gammas = tuple(sorted(self.gammas))
        if not 0.0 < self.pessimism <= 0.5:
            raise ValueError(f"pessimism 必须在 (0, 0.5]，实际 {self.pessimism}")

    @property
    def input_dim(self) -> int:
        return self.d_feature + self.intent_dim

    @property
    def primary_gamma(self) -> float:
        return self.gammas[self.primary_gamma_index]

    @property
    def effective_ensemble_size(self) -> int:
        return {"single": 1, "double": 2}.get(self.mode, self.n_ensemble)

    @property
    def num_gammas(self) -> int:
        return len(self.gammas)


@dataclass
class CriticTrainConfig:
    """Critic 训练配置"""
    gamma: float = 0.99
    lambda_: float = 0.95
    value_loss_weight: float = 1.0
    use_multihead_critic: bool = False
    multi_gamma_weight: float = 1.0
    anchor_loss_weight: float = 0.1
    uncertainty_weight: float = 0.01
    use_anchor_loss: bool = False
    use_uncertainty_weights: bool = False
    critic_lr: float = 3e-4
    critic_batch_size: int = 64
    critic_n_epochs: int = 4
    critic_grad_clip: float = 100.0
    use_target_critic: bool = True
    target_update_interval: int = 100
    target_update_tau: float = 0.005

    @property
    def effective_value_loss_weight(self) -> float:
        return self.multi_gamma_weight if self.use_multihead_critic else self.value_loss_weight

    def create_multihead_config(
        self, feat_dim: int, intent_dim: int = 0,
        override_gammas: Optional[Tuple[float, ...]] = None,
    ) -> CriticConfig:
        gammas = override_gammas or tuple(sorted([0.9, 0.95, self.gamma, 0.997]))
        return CriticConfig(
            d_feature=feat_dim, intent_dim=intent_dim, gammas=gammas,
            n_ensemble=3, pessimism=0.25,
            use_anchor_critic=self.anchor_loss_weight > 0,
            anchor_outputs_symlog=True, anchor_hidden_dim=256,
            use_target_critic=self.use_target_critic,
            target_update_tau=self.target_update_tau,
            learning_rate=self.critic_lr,
        )


# ═══════════════════════════════════════════════════════════
#  Continue / 多步一致性子配置
# ═══════════════════════════════════════════════════════════

@dataclass
class ContinueConfig:
    """Continue 预测头"""
    temperature: float = 1.0
    loss_type: str = "bce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    optimistic_bias: float = 5.0
    positive_weight: float = 1.0
    negative_weight: float = 1.0
    hidden_dims: Optional[List[int]] = None


@dataclass
class MultiHorizonContinueConfig:
    """多步 Continue 预测"""
    enabled: bool = True
    feat_dim: int = 640
    horizons: Tuple[int, ...] = (1, 4, 16)
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "SiLU"
    loss_scale: float = 1.0
    horizon_weights: Optional[Tuple[float, ...]] = None
    min_seq_len: int = 20
    skip_invalid_horizons: bool = True
    init: InitConfig = field(default_factory=InitConfig)


@dataclass
class NStepTerminationConfig:
    """N-Step Termination Loss"""
    enabled: bool = True
    use_horizon_weights: bool = True
    weight_mode: str = "exponential"
    weight_decay: float = 0.85
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    label_smoothing: float = 0.0
    soft_boundary: bool = True
    boundary_width: float = 1.0
    enforce_monotonicity: bool = True
    loss_scale: float = 1.0


@dataclass
class ShortcutConsistencyConfig:
    """Shortcut Consistency Loss"""
    enabled: bool = True
    shortcut_horizons: Tuple[int, ...] = (1, 2, 4)
    horizon_weights: Optional[Tuple[float, ...]] = None
    weight_mode: str = "exponential"
    weight_decay: float = 0.7
    use_deter_loss: bool = True
    deter_loss_type: str = "mse"
    deter_weight: float = 1.0
    use_feat_loss: bool = True
    feat_weight: float = 0.5
    detach_target: bool = True
    mask_terminated: bool = True
    multi_start: bool = True
    max_starts: int = 8
    batch_starts: bool = True
    loss_scale: float = 1.0


# ═══════════════════════════════════════════════════════════
#  Control / Router / Smoothing / Perceptor / Memory
# ═══════════════════════════════════════════════════════════

@dataclass
class ControlConfig:
    """控制分支 s_ctrl"""
    mode: Literal["static", "gated"] = "static"
    ctrl_dim: int = 128
    hidden_dims: Tuple[int, ...] = (256,)
    align_strength: float = 0.1
    reg_strength: float = 0.01


@dataclass
class RouterConfig:
    """FeatureRouter"""
    mode: Literal["x_only", "ctrl_only", "concat", "concat_full", "attention"] = "concat"
    proj_dim: int = 256
    gate_hidden_dim: int = 128
    gate_temperature_init: float = 1.0
    gate_temperature_final: float = 0.5
    attention_freeze_steps: int = 5000
    lb_weight: float = 1.0
    gate_entropy_weight: float = 0.1
    uncertainty_alpha: float = 0.5
    uncertainty_quantile: float = 0.85
    uncertainty_window: int = 1000
    uncertainty_ema_decay: float = 0.99
    uncertainty_hysteresis: Tuple[float, float] = (0.8, 1.2)
    uncertainty_window_size: int = 2000
    uncertainty_warmup_steps: int = 200
    temp_range: Tuple[float, float] = (1.0, 0.1)
    anneal_steps: int = 50000
    ema_blend_train: float = 0.1
    ema_blend_eval: float = 0.0
    mask_x_factor: float = 0.3
    mask_z_factor: float = 0.1
    hard_fallback_enabled: bool = False
    hard_fallback_weights: Tuple[float, ...] = (0.0, 1.0, 0.0)
    hard_fallback_weights_3branch: Optional[Tuple[float, ...]] = None
    hard_fallback_weights_2branch: Optional[Tuple[float, ...]] = None


@dataclass
class SmoothingConfig:
    """路由平滑"""
    use_mask_smoothing: bool = True
    mask_smoothing_factor: float = 0.9
    use_hysteresis: bool = True
    hysteresis_high_ratio: float = 1.2
    hysteresis_low_ratio: float = 0.8
    use_weight_smoothing: bool = True
    weight_smoothing_factor: float = 0.9
    use_soft_mask: bool = True
    soft_mask_strength: float = 0.9


@dataclass
class PerceptorConfig:
    """感知器"""
    vitals_dim: int = 64
    modality_embed_dim: int = 256
    fusion_type: str = "concat"
    fusion_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    image_depth: int = 32
    image_kernels: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    vector_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    vision_backbone: str = "auto"
    vision_backbone_min_resolution: int = 128
    vision_backbone_patch_size: int = 16
    vision_backbone_min_patches: int = 16
    use_layernorm: bool = True
    activation: str = "silu"
    dropout: float = 0.0


@dataclass
class MemoryConfig:
    """动态记忆增强"""
    enabled: bool = False
    buffer_size: int = 64
    compressed_size: int = 256
    compression_ratio: int = 4
    use_attention: bool = False
    attention_heads: int = 4
    attention_dim: int = 64

    @property
    def total_memory_size(self) -> int:
        return self.buffer_size + self.compressed_size


# ═══════════════════════════════════════════════════════════
#  RSSM
# ═══════════════════════════════════════════════════════════

@dataclass
class RSSMConfig:
    """
    RSSM 完整配置

    feat_dim = deter_dim + z_embed_dim
    z_dim    = num_distributions × num_classes
    """
    vitals_dim: int = 64
    deter_dim: int = 512
    hidden_dim: int = 512
    action_embed_dim: int = 32
    obs_embed_dim: int = 512
    z_embed_dim: int = 128

    use_gru_output_norm: bool = True
    use_feature_norm: bool = True
    use_separate_norm: bool = False
    use_quant_loss: bool = False
    quant_loss_weight: float = 0.01
    grad_clip: float = 100.0

    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    kl: KLConfig = field(default_factory=KLConfig)
    symlog: SymlogConfig = field(default_factory=SymlogConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    continue_config: ContinueConfig = field(default_factory=ContinueConfig)

    mhc: Optional[MultiHorizonContinueConfig] = None
    msc: Optional["MSCConfig"] = None
    gap_monitor: Optional["GapMonitorConfig"] = None
    nst: Optional[NStepTerminationConfig] = None
    shortcut_consistency: Optional[ShortcutConsistencyConfig] = None
    gradient_wall: Optional["GradientWallConfig"] = None
    smoothing: Optional[SmoothingConfig] = None
    memory: Optional[MemoryConfig] = None

    auto_danger_fn: bool = False
    danger_env_type: str = "cartpole"
    adaptive_capacity: bool = True
    min_deter: int = 96
    max_deter: int = 320
    min_stoch: int = 16
    max_stoch: int = 32
    use_persist_gate: bool = True
    danger_kl_multiplier: float = 2.0
    recon_weight: float = 1.0
    reward_weight: float = 1.0
    continue_weight: float = 1.0
    kl_weight: float = 1.0
    will: Optional[Any] = None
    free_bits: float = 1.0
    kl_balance: float = 0.5
    z_type: Literal["gaussian", "categorical"] = "gaussian"
    z_categories: int = 32
    z_classes: int = 32

    def __post_init__(self):
        if self.obs_embed_dim <= 0 < self.hidden_dim:
            self.obs_embed_dim = self.hidden_dim
        if self.z_type != "categorical":
            return
        d = self.distribution
        # 双向填充缺失值
        pairs = [
            (self, "z_categories", d, "num_distributions"),
            (self, "z_classes", d, "num_classes"),
        ]
        for obj_a, attr_a, obj_b, attr_b in pairs:
            va, vb = getattr(obj_a, attr_a), getattr(obj_b, attr_b)
            if va <= 0 and vb > 0:
                setattr(obj_a, attr_a, int(vb))
            elif vb <= 0 and va > 0:
                setattr(obj_b, attr_b, int(va))
        if d.num_distributions != self.z_categories or d.num_classes != self.z_classes:
            raise ValueError(
                "RSSMConfig: distribution dims mismatch: "
                f"z_categories/z_classes=({self.z_categories}, {self.z_classes}) "
                f"!= distribution.num_distributions/num_classes="
                f"({d.num_distributions}, {d.num_classes})"
            )

    @property
    def z_dim(self) -> int:
        return self.distribution.stoch_dim

    @property
    def stoch_dim(self) -> int:
        return self.distribution.stoch_dim

    @property
    def stoch_dim_raw(self) -> int:
        return self.stoch_dim

    @property
    def feat_dim(self) -> int:
        return self.deter_dim + self.z_embed_dim

    @property
    def will_enabled(self) -> bool:
        return self.will is not None

    @property
    def stoch_shape(self) -> Tuple[int, ...]:
        if self.z_type == "categorical":
            return (self.z_categories, self.z_classes)
        return (self.stoch_dim,)


# ═══════════════════════════════════════════════════════════
#  Gradient Wall / Loss Weights
# ═══════════════════════════════════════════════════════════

_WALL_SCHEDULES = {
    "linear":      lambda p: p,
    "cosine":      lambda p: (1 - math.cos(math.pi * p)) * 0.5,
    "step":        lambda p: float(p >= 1.0),
    "exponential": lambda p: (math.exp(p) - 1) / (math.e - 1),
}


@dataclass
class GradientWallConfig:
    """渐进式 Iron Wall"""
    warmup_steps: int = 10000
    wall_schedule: Literal["linear", "cosine", "step", "exponential"] = "cosine"
    initial_wall_strength: float = 0.0
    final_wall_strength: float = 1.0
    bridge_maintenance_floor: float = 0.25
    use_aux_tasks: bool = True
    aux_inverse_weight: float = 1.0
    aux_value_weight: float = 1.0

    def get_progress(self, step: int) -> float:
        return 1.0 if self.warmup_steps <= 0 else min(1.0, step / self.warmup_steps)

    def get_wall_strength(self, step: int) -> float:
        p = self.get_progress(step)
        fn = _WALL_SCHEDULES.get(self.wall_schedule)
        if fn is None:
            raise ValueError(f"未知 wall_schedule: {self.wall_schedule}")
        delta = self.final_wall_strength - self.initial_wall_strength
        return float(self.initial_wall_strength + fn(p) * delta)

    def get_bridge_maintenance_scale(self, step: int) -> float:
        floor = float(min(max(self.bridge_maintenance_floor, 0.0), 1.0))
        wall_strength = min(max(self.get_wall_strength(step), 0.0), 1.0)
        return float(floor + (1.0 - floor) * (1.0 - wall_strength))

    def get_aux_task_scale(self, step: int) -> float:
        return 0.0 if not self.use_aux_tasks else self.get_bridge_maintenance_scale(step)

    def is_fully_isolated(self, step: int) -> bool:
        return self.get_wall_strength(step) >= 1.0

    def get_phase(self, step: int) -> str:
        p = self.get_progress(step)
        return "warmup" if p < 0.1 else ("transition" if p < 1.0 else "isolated")


@dataclass
class LossWeightsConfig:
    """损失权重"""
    recon: float = 1.0
    reward: float = 1.0
    continue_: float = 1.0
    kl: float = 1.0
    quant: float = 0.0
    mhc: float = 0.0
    msc: float = 0.0
    nst: float = 0.0
    sc: float = 0.0
    consistency: float = 0.0
    aux_inverse: float = 0.1
    aux_value: float = 0.1
    projection: float = 0.0
    control_align: float = 0.1
    control_reg: float = 0.01
    abstractor: float = 0.0
    consistency_head: float = 0.0

    def get(self, name: str, default: float = 1.0) -> float:
        return getattr(self, "continue_" if name == "continue" else name, default)

    @classmethod
    def for_env_complexity(cls, is_simple: bool, obs_dim: int = 0,
                           action_dim: int = 0) -> "LossWeightsConfig":
        if is_simple:
            return cls()
        return cls(
            mhc=1.0, msc=1.0, nst=1.0, sc=1.0, consistency=1.0,
            projection=1.0, abstractor=0.5, consistency_head=0.1,
        )

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["continue"] = d.pop("continue_")
        return d


# ═══════════════════════════════════════════════════════════
#  一致性审计器（MHC / MSC / NST / SC）
# ═══════════════════════════════════════════════════════════

@dataclass
class MHCConfig:
    enabled: bool = False
    horizons: Tuple[int, ...] = (1, 4, 8, 16)
    hidden_dim: int = 256
    loss_scale: float = 1.0


@dataclass
class MSCConfig:
    """Multi-Scale Continue（兼容旧 MultiScaleContinueConfig 接口）"""
    enabled: bool = False
    horizons: Tuple[int, ...] = (1, 4, 8, 16)
    hidden_dims: Optional[Tuple[int, ...]] = None
    hidden_dim: int = 256
    predict_danger: bool = False
    danger_hidden_dim: int = 64
    predict_remaining: bool = False
    loss_scale: float = 1.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    temperature: float = 1.0
    enforce_monotonicity: bool = True
    effective_continue_mode: Literal["min", "geometric", "weighted"] = "min"
    bce_weight: float = 1.0
    danger_weight: float = 0.1
    monotonicity_weight: float = 0.01

    @property
    def num_horizons(self) -> int:
        return len(self.horizons)

    def get_hidden_dims(self) -> Tuple[int, ...]:
        return self.hidden_dims if self.hidden_dims is not None else (self.hidden_dim, self.hidden_dim // 2)


MultiScaleContinueConfig = MSCConfig


@dataclass
class NSTConfig:
    enabled: bool = False
    n_steps: Tuple[int, ...] = (1, 4, 8, 16)
    loss_scale: float = 1.0
    use_exponential_weighting: bool = True


@dataclass
class SCConfig:
    enabled: bool = False
    horizons: Tuple[int, ...] = (2, 4, 8)
    loss_scale: float = 1.0
    detach_target: bool = True
    sample_ratio: float = 1.0
    max_starts: int = 0
    feature_loss_scale: float = 0.0


@dataclass
class ConsistencyAuditorConfig:
    """统一管理 MHC / MSC / NST / SC"""
    mhc: MHCConfig = field(default_factory=MHCConfig)
    msc: MSCConfig = field(default_factory=MSCConfig)
    nst: NSTConfig = field(default_factory=NSTConfig)
    sc: SCConfig = field(default_factory=SCConfig)
    strict_auxiliary_losses: bool = True
    remaining_steps_margin: int = 10
    no_done_warning_threshold: float = 0.5

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "ConsistencyAuditorConfig":
        if self.mhc.enabled and self.msc.enabled:
            raise ValueError("MHC and MSC cannot both be enabled in ConsistencyAuditorConfig")
        if self.nst.enabled and not self.msc.enabled:
            raise ValueError("NST requires MSC to be enabled in ConsistencyAuditorConfig")
        return self

    @property
    def any_enabled(self) -> bool:
        return any(c.enabled for c in (self.mhc, self.msc, self.nst, self.sc))

    @property
    def enabled_modules(self) -> List[str]:
        return [n for n, c in [("mhc", self.mhc), ("msc", self.msc),
                                ("nst", self.nst), ("sc", self.sc)] if c.enabled]

    def get_horizons(self) -> Tuple[int, ...]:
        h: set = set()
        if self.mhc.enabled: h.update(self.mhc.horizons)
        if self.msc.enabled: h.update(self.msc.horizons)
        if self.nst.enabled: h.update(self.nst.n_steps)
        if self.sc.enabled:  h.update(self.sc.horizons)
        return tuple(sorted(h)) if h else (1,)

    @property
    def max_horizon(self) -> int:
        return max(self.get_horizons())


# ═══════════════════════════════════════════════════════════
#  WorldModel 子组件 & 引擎
# ═══════════════════════════════════════════════════════════

@dataclass
class ProjectionConfig:
    enabled: bool = True
    hidden_dim: int = 256
    mmd_weight: float = 0.1
    mmd_mode: Literal["linear", "kernel", "none"] = "linear"
    mmd_num_features: int = 256


@dataclass
class AbstractorConfig:
    enabled: bool = False
    z_task_dim: int = 64
    hidden_dim: int = 256
    use_scalar_head: bool = True


@dataclass
class ConsistencyHeadConfig:
    enabled: bool = False
    hidden_dim: int = 256


@dataclass
class AuxTasksConfig:
    enabled: bool = True
    hidden_dim: int = 256
    value_dim: int = 1


@dataclass
class PredictiveEngineConfig:
    """预测引擎"""
    consistency: ConsistencyAuditorConfig = field(default_factory=ConsistencyAuditorConfig)
    use_danger_fn: bool = False
    danger_env_type: str = "cartpole"
    use_loss_budget: bool = False
    budget_recon: float = 0.4
    budget_reward: float = 0.2
    budget_continue: float = 0.2
    budget_kl: float = 0.1
    budget_consistency: float = 0.1
    log_loss_components: bool = True
    log_grad_norms: bool = False

    def get_budget_weights(self) -> Dict[str, float]:
        if not self.use_loss_budget:
            return {}
        return {
            "recon": self.budget_recon, "reward": self.budget_reward,
            "continue": self.budget_continue, "kl": self.budget_kl,
            "consistency": self.budget_consistency,
        }


@dataclass
class ImagineConfig:
    """想象轨迹"""
    horizon: int = 15
    use_lambda_returns: bool = True
    lambda_gae: float = 0.95
    discount_mode: Literal["learned", "fixed"] = "learned"
    fixed_gamma: float = 0.99
    use_checkpoint: bool = False


@dataclass
class GapMonitorConfig:
    """GapMonitor 诊断"""
    enabled: bool = True
    history_size: int = 100
    log_interval: int = 1000
    ema_decay: float = 0.99
    warning_threshold: float = 1.5
    critical_threshold: float = 2.0
    trend_window: int = 50
    window_size: int = 100
    auto_adjust: bool = False
    adjust_factor: float = 0.1
    batch_size: int = 32


# ═══════════════════════════════════════════════════════════
#  WorldModelConfig
# ═══════════════════════════════════════════════════════════

# [FIX-4] 工厂方法预设字典，替代散布在 classmethod 中的硬编码超参数
#         扩展新环境类型时只需加一行
_WM_PRESETS: Dict[str, Dict[str, Any]] = {
    "simple": dict(
        obs_embed_dim_fn=lambda d: max(32, d),
        deter_dim=128,
        stoch_dim=32 * 32,
        control=dict(mode="static", ctrl_dim=64),
        wall=dict(warmup_steps=5000),
        is_simple=True,
        abstractor_enabled=False,
        consistency_modules=dict(msc=False),
    ),
    "complex": dict(
        obs_embed_dim_fn=lambda d: 128,
        deter_dim=256,
        stoch_dim=32 * 32,
        control=dict(mode="gated", ctrl_dim=128),
        wall=dict(warmup_steps=20000),
        is_simple=False,
        abstractor_enabled=True,
        consistency_modules=dict(msc=True, nst=True, sc=True),
    ),
    "minimal": dict(
        obs_embed_dim_fn=lambda d: d,
        deter_dim=128,
        stoch_dim=32 * 32,
        control=dict(mode="static", ctrl_dim=128),
        wall=dict(warmup_steps=10000),
        is_simple=True,
        abstractor_enabled=False,
        projection_enabled=False,
        consistency_head_enabled=False,
        aux_tasks_enabled=False,
        consistency_modules=dict(mhc=False, msc=False, nst=False, sc=False),
    ),
}


def _build_wm_from_preset(
    preset_name: str,
    obs_dim: int,
    action_dim: int,
    is_discrete: bool = False,
) -> "WorldModelConfig":
    """[FIX-4] 根据预设名称构建 WorldModelConfig"""
    p = _WM_PRESETS[preset_name]

    obs_embed_dim = p["obs_embed_dim_fn"](obs_dim)

    # 一致性审计子模块
    cm = p.get("consistency_modules", {})
    consistency = ConsistencyAuditorConfig(
        mhc=MHCConfig(enabled=cm.get("mhc", False)),
        msc=MSCConfig(enabled=cm.get("msc", False)),
        nst=NSTConfig(enabled=cm.get("nst", False)),
        sc=SCConfig(enabled=cm.get("sc", False)),
    )

    return WorldModelConfig(
        obs_embed_dim=obs_embed_dim,
        deter_dim=p["deter_dim"],
        stoch_dim=p["stoch_dim"],
        action_dim=action_dim,
        is_discrete_action=is_discrete,
        control=ControlConfig(**p["control"]),
        wall=GradientWallConfig(**p["wall"]),
        loss_weights=LossWeightsConfig.for_env_complexity(is_simple=p["is_simple"]),
        abstractor=AbstractorConfig(enabled=p.get("abstractor_enabled", False)),
        projection=ProjectionConfig(enabled=p.get("projection_enabled", True)),
        consistency_head=ConsistencyHeadConfig(enabled=p.get("consistency_head_enabled", False)),
        aux_tasks=AuxTasksConfig(enabled=p.get("aux_tasks_enabled", True)),
        engine=PredictiveEngineConfig(consistency=consistency),
    )


@dataclass
class WorldModelConfig:
    """V5.2 世界模型统一配置"""
    obs_embed_dim: int = 128
    deter_dim: int = 256
    stoch_dim: int = 1024
    action_dim: int = 4
    num_classes: int = 32
    is_discrete_action: bool = False

    wall: GradientWallConfig = field(default_factory=GradientWallConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    abstractor: AbstractorConfig = field(default_factory=AbstractorConfig)
    consistency_head: ConsistencyHeadConfig = field(default_factory=ConsistencyHeadConfig)
    aux_tasks: AuxTasksConfig = field(default_factory=AuxTasksConfig)
    engine: PredictiveEngineConfig = field(default_factory=PredictiveEngineConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)

    def __post_init__(self):
        if self.stoch_dim % self.num_classes != 0:
            raise ValueError(
                f"stoch_dim ({self.stoch_dim}) 必须能被 num_classes ({self.num_classes}) 整除"
            )

    @property
    def s_pred_dim(self) -> int:
        return self.deter_dim + self.stoch_dim

    @property
    def feat_dim(self) -> int:
        return self.s_pred_dim

    @property
    def ctrl_dim(self) -> int:
        return self.control.ctrl_dim

    @property
    def num_distributions(self) -> int:
        return self.stoch_dim // self.num_classes

    @property
    def use_projection(self) -> bool:
        return self.projection.enabled

    @property
    def use_abstractor(self) -> bool:
        return self.abstractor.enabled

    @property
    def use_consistency_head(self) -> bool:
        return self.consistency_head.enabled

    @property
    def use_aux_tasks(self) -> bool:
        return self.aux_tasks.enabled and self.wall.use_aux_tasks

    def get_effective_aux_weights(self, step: int) -> Dict[str, float]:
        scale = self.wall.get_aux_task_scale(step)
        return {
            "inverse": scale * self.wall.aux_inverse_weight * self.loss_weights.aux_inverse,
            "value": scale * self.wall.aux_value_weight * self.loss_weights.aux_value,
        }

    def get_bridge_maintenance_scale(self, step: int) -> float:
        return self.wall.get_bridge_maintenance_scale(step)

    # [FIX-4] 工厂方法改为委托给预设字典驱动的 _build_wm_from_preset
    @classmethod
    def for_simple_env(cls, obs_dim: int, action_dim: int,
                       is_discrete: bool = False) -> "WorldModelConfig":
        """为简单环境创建配置"""
        return _build_wm_from_preset("simple", obs_dim, action_dim, is_discrete)

    @classmethod
    def for_complex_env(cls, obs_dim: int, action_dim: int,
                        is_discrete: bool = False) -> "WorldModelConfig":
        """为复杂环境创建配置"""
        return _build_wm_from_preset("complex", obs_dim, action_dim, is_discrete)

    @classmethod
    def minimal(cls, obs_dim: int, action_dim: int) -> "WorldModelConfig":
        """创建最小配置"""
        return _build_wm_from_preset("minimal", obs_dim, action_dim)

    def validate(self):
        for name in ("stoch_dim", "deter_dim", "action_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正数，实际 {getattr(self, name)}")
        if self.control.ctrl_dim <= 0:
            raise ValueError(f"ctrl_dim 必须为正数，实际 {self.control.ctrl_dim}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs_embed_dim": self.obs_embed_dim, "deter_dim": self.deter_dim,
            "stoch_dim": self.stoch_dim, "action_dim": self.action_dim,
            "num_classes": self.num_classes, "is_discrete_action": self.is_discrete_action,
            "s_pred_dim": self.s_pred_dim, "ctrl_dim": self.ctrl_dim,
        }


# ═══════════════════════════════════════════════════════════
#  Emergency / Will
# ═══════════════════════════════════════════════════════════

@dataclass
class WillConfig:
    """Will 系统"""
    enabled: bool = True
    curiosity_enabled: bool = True
    eta_wm: float = 0.5
    eta_v: float = 0.3
    eta_router: float = 0.2
    u0_wm: float = 1.0
    u0_v: float = 0.5
    mastery_enabled: bool = True
    kappa_wm: float = 0.3
    kappa_v: float = 0.2
    autonomy_enabled: bool = True
    kappa_a: float = 0.1
    trust_enabled: bool = True
    trust_ema_beta: float = 0.95
    U0_wm: float = 1.0
    U0_v: float = 0.5
    trust_aggregation: Literal["min", "mean", "weighted"] = "min"
    trust_wm_weight: float = 0.6
    trust_v_weight: float = 0.4
    normalize_uncertainty: bool = True
    ema_alpha: float = 0.01
    use_window_update: bool = False
    window_size: int = 100
    update_freq: int = 10


# ═══════════════════════════════════════════════════════════
#  训练配置（TrainParams / PPO / ConfigBundle）
# ═══════════════════════════════════════════════════════════

@dataclass
class TrainParams:
    """Bootstrap-only train payload carried with agent init / reload metadata.

    Directly active during agent bootstrap:
      - ``buffer_size`` for replay buffer allocation
      - ``batch_size`` for ``AgentHandle.ready_to_train()``
      - ``use_compile`` / ``compile_*`` for inference compilation setup

    Other fields remain as bootstrap metadata defaults that may be useful for
    checkpoint re-save or inspection, but they are not the runtime training
    truth and are not required to build the agent core itself.
    """
    total_steps: int = DEFAULT_STEPS
    buffer_size: int = BUFFER_CAPACITY
    batch_size: int = 32
    sequence_length: int = DEFAULT_BATCH_LENGTH
    imagination_horizon: int = DEFAULT_HORIZON
    wm_pretrain_steps: int = 10_000
    warmup_steps: int = 10_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    log_interval: int = LOG_INTERVAL
    eval_interval: int = EVAL_INTERVAL
    save_interval: int = SAVE_INTERVAL
    use_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    compile_warmup: bool = False
    compile_warmup_batch_size: int = 2
    use_jit: bool = False
    compile_actor: bool = True
    compile_critic: bool = True
    compile_router: bool = True
    compile_wm_encoder: bool = True
    compile_wm_projection: bool = True

    def __post_init__(self):
        _validate(self,
            positive=("total_steps", "buffer_size", "batch_size",
                      "sequence_length", "imagination_horizon",
                      "learning_rate", "log_interval", "eval_interval",
                      "save_interval", "compile_warmup_batch_size"),
            non_neg=("wm_pretrain_steps", "warmup_steps"),
            ranges=(("gamma", 0, 1),),
        )

    def to_training_config(self) -> "TrainingConfig":
        return TrainingConfig(
            optimizer=OptimizerConfig(learning_rate=self.learning_rate),
            rl=RLConfig(gamma=self.gamma),
            total_steps=self.total_steps,
            num_train_steps=self.total_steps,
            total_env_steps=self.total_steps,
            buffer_capacity=self.buffer_size,
            batch_size=self.batch_size,
            seq_len=self.sequence_length,
            wm_seq_len=self.sequence_length,
            wm_batch_size=self.batch_size,
            imagination_horizon=self.imagination_horizon,
            wm_pretrain_steps=self.wm_pretrain_steps,
            warmup_steps=self.warmup_steps,
            log_interval=self.log_interval,
            eval_interval=self.eval_interval,
            save_interval=self.save_interval,
            use_compile=self.use_compile,
            compile_mode=self.compile_mode,
            compile_warmup=self.compile_warmup,
            compile_warmup_batch_size=self.compile_warmup_batch_size,
            use_jit=self.use_jit,
            compile_actor=self.compile_actor,
            compile_critic=self.compile_critic,
            compile_router=self.compile_router,
            compile_wm_encoder=self.compile_wm_encoder,
            compile_wm_projection=self.compile_wm_projection,
        )


@dataclass
class AgentBootstrapTrainParams:
    """Minimal train payload required to bootstrap an agent instance."""

    buffer_size: int = BUFFER_CAPACITY
    batch_size: int = 32
    use_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    compile_warmup: bool = False
    compile_warmup_batch_size: int = 2
    use_jit: bool = False
    compile_actor: bool = True
    compile_critic: bool = True
    compile_router: bool = True
    compile_wm_encoder: bool = True
    compile_wm_projection: bool = True

    def __post_init__(self):
        _validate(
            self,
            positive=("buffer_size", "batch_size", "compile_warmup_batch_size"),
        )


AGENT_BOOTSTRAP_TRAIN_FIELDS: frozenset[str] = frozenset(
    AgentBootstrapTrainParams.__dataclass_fields__
)

TRAIN_PARAMS_BOOTSTRAP_METADATA_FIELDS: frozenset[str] = frozenset(
    TrainParams.__dataclass_fields__
) - AGENT_BOOTSTRAP_TRAIN_FIELDS


def coerce_config_policy_overrides(
    value: Optional[Union[ConfigPolicyOverrides, Mapping[str, Any]]],
) -> ConfigPolicyOverrides:
    """Convert ConfigPolicy input to the canonical typed boundary."""
    if value is None:
        return ConfigPolicyOverrides()
    if isinstance(value, ConfigPolicyOverrides):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "ConfigPolicy overrides must be ConfigPolicyOverrides or a mapping."
        )
    return parse_config_policy_overrides(dict(value))

@dataclass
class ConfigBundle:
    train: AgentBootstrapTrainParams
    profile: Optional[EnvProfile] = None
    custom_overrides: Dict[str, Any] = field(default_factory=dict)
    router_overrides: Dict[str, Any] = field(default_factory=dict)
    perceptor_overrides: Dict[str, Any] = field(default_factory=dict)
    bootstrap_env_profile: Optional[Any] = None

    @staticmethod
    def _serialize_bootstrap_env_profile(value: Optional[Any]) -> Optional[Any]:
        if isinstance(value, EnvProfile):
            return asdict(value)
        return copy.deepcopy(value)

    def to_factory_overrides(self) -> Dict[str, Any]:
        """Encode bootstrap state into the factory bridge payload."""
        return export_factory_bridge_overrides(
            FactoryBridgeOverrides(
                custom_overrides=copy.deepcopy(dict(self.custom_overrides)),
                router_overrides=copy.deepcopy(dict(self.router_overrides)),
                perceptor_overrides=copy.deepcopy(dict(self.perceptor_overrides)),
                bridge_env_profile=self._serialize_bootstrap_env_profile(
                    self.bootstrap_env_profile
                ),
            )
        )

    def to_agent_bootstrap_bundle(self) -> Dict[str, Any]:
        """Encode the checkpoint bootstrap payload for this agent."""
        return {
            "train": asdict(self.train),
            "custom_overrides": copy.deepcopy(dict(self.custom_overrides)),
            "router_overrides": copy.deepcopy(dict(self.router_overrides)),
            "perceptor_overrides": copy.deepcopy(dict(self.perceptor_overrides)),
            "bootstrap_env_profile": self._serialize_bootstrap_env_profile(
                self.bootstrap_env_profile
            ),
        }

    def to_agent_creation_overrides(self) -> Dict[str, Any]:
        """Encode bootstrap state into the create_agent() override surface."""
        payload = asdict(self.train)
        payload.update(self.to_factory_overrides())
        return payload

    @classmethod
    def from_checkpoint_metadata(
        cls,
        checkpoint: Optional[Mapping[str, Any]],
        *,
        profile: Optional[EnvProfile] = None,
    ) -> Optional["ConfigBundle"]:
        """Decode bootstrap metadata from a checkpoint payload when present."""
        if not isinstance(checkpoint, Mapping):
            return None

        payload = checkpoint.get("agent_bootstrap_bundle")
        if not isinstance(payload, Mapping):
            if "config_bundle" in checkpoint:
                raise ValueError(
                    "Legacy checkpoint bootstrap payload 'config_bundle' is no longer "
                    "supported. Re-save the checkpoint with 'agent_bootstrap_bundle'."
                )
            return None

        return cls.from_agent_bootstrap_bundle(payload, profile=profile)

    @classmethod
    def agent_creation_overrides_from_checkpoint_metadata(
        cls,
        checkpoint: Optional[Mapping[str, Any]],
        *,
        profile: Optional[EnvProfile] = None,
    ) -> Optional[Dict[str, Any]]:
        """Decode create_agent() overrides directly from checkpoint metadata."""
        bundle = cls.from_checkpoint_metadata(checkpoint, profile=profile)
        if bundle is None:
            return None
        return bundle.to_agent_creation_overrides() or None

    @classmethod
    def from_factory_bridge_overrides(
        cls,
        overrides: Optional[Mapping[str, Any]],
        *,
        profile: Optional[EnvProfile] = None,
    ) -> "ConfigBundle":
        """Build a bootstrap bundle from the factory bridge payload only."""
        bridge = parse_factory_bridge_overrides(overrides)
        return cls(
            train=AgentBootstrapTrainParams(),
            profile=profile,
            custom_overrides=copy.deepcopy(dict(bridge.custom_overrides or {})),
            router_overrides=copy.deepcopy(dict(bridge.router_overrides or {})),
            perceptor_overrides=copy.deepcopy(dict(bridge.perceptor_overrides or {})),
            bootstrap_env_profile=copy.deepcopy(bridge.bridge_env_profile),
        )

    @classmethod
    def from_agent_bootstrap_bundle(
        cls,
        payload: Mapping[str, Any],
        *,
        profile: Optional[EnvProfile] = None,
    ) -> "ConfigBundle":
        """Build a bootstrap bundle from checkpoint bootstrap metadata."""
        if not isinstance(payload, Mapping):
            raise TypeError("agent_bootstrap_bundle must be a mapping.")
        if "env_profile_override" in payload:
            raise ValueError(
                "Legacy checkpoint bootstrap payload field 'env_profile_override' is no longer "
                "supported. Re-save with 'bootstrap_env_profile'."
            )

        train_payload = payload.get("train")
        if train_payload is None:
            train = AgentBootstrapTrainParams()
        else:
            if not isinstance(train_payload, Mapping):
                raise TypeError(
                    "Checkpoint agent bootstrap train payload must be a mapping."
                )
            unsupported_fields = sorted(
                key for key in train_payload if key not in AGENT_BOOTSTRAP_TRAIN_FIELDS
            )
            if unsupported_fields:
                raise ValueError(
                    "Checkpoint agent bootstrap train payload contains unsupported fields "
                    "that are no longer accepted by load_agent(): "
                    + ", ".join(unsupported_fields)
                    + ". Re-save with AgentBootstrapTrainParams fields only."
                )
            train = AgentBootstrapTrainParams(**copy.deepcopy(dict(train_payload)))

        def _clone_mapping(name: str) -> Dict[str, Any]:
            value = payload.get(name)
            if not isinstance(value, Mapping):
                return {}
            return copy.deepcopy(dict(value))

        return cls(
            train=train,
            profile=profile,
            custom_overrides=_clone_mapping("custom_overrides"),
            router_overrides=_clone_mapping("router_overrides"),
            perceptor_overrides=_clone_mapping("perceptor_overrides"),
            bootstrap_env_profile=copy.deepcopy(payload.get("bootstrap_env_profile")),
        )

    @staticmethod
    def _make_bootstrap_train_params_for_profile(
        profile: EnvProfile,
    ) -> AgentBootstrapTrainParams:
        """Return agent-bootstrap train defaults derived from an environment profile."""
        if profile.is_simple:
            return AgentBootstrapTrainParams(
                buffer_size=BUFFER_CAPACITY,
                batch_size=32,
            )
        return AgentBootstrapTrainParams(
            buffer_size=BUFFER_CAPACITY * 10,
            batch_size=64,
        )

    @staticmethod
    def _merge_config_policy_into_parts(
        policy: ConfigPolicyOverrides,
        *,
        custom_overrides: Dict[str, Any],
        router_overrides: Dict[str, Any],
        perceptor_overrides: Dict[str, Any],
    ) -> Optional[Any]:
        """Apply canonical ConfigPolicy values into bootstrap bundle parts."""
        if policy.curiosity_enabled is not None:
            custom_overrides["curiosity_enabled"] = bool(policy.curiosity_enabled)
        if policy.mastery_enabled is not None:
            custom_overrides["mastery_enabled"] = bool(policy.mastery_enabled)
        custom_overrides.update(policy.custom_overrides)
        router_overrides.update(policy.router_overrides)
        perceptor_overrides.update(policy.perceptor_overrides)
        return policy.policy_env_profile

    @staticmethod
    def _load_profile_config_policy(
        env: Optional[Any],
        *,
        include_env_profile: bool = True,
    ) -> ConfigPolicyOverrides:
        """Resolve typed ConfigPolicy overrides for bootstrap bundle assembly."""
        if env is None:
            return ConfigPolicyOverrides()
        try:
            from .aletheia_foundation import make_config_policy_overrides_for_env
            raw_policy_overrides = make_config_policy_overrides_for_env(
                env,
                include_env_profile=include_env_profile,
            )
        except Exception as e:
            logger.debug(f"ConfigPolicy not available: {e}")
            return ConfigPolicyOverrides()
        return coerce_config_policy_overrides(raw_policy_overrides)

    @staticmethod
    def _merge_explicit_overrides_into_parts(
        train: AgentBootstrapTrainParams,
        overrides: Optional[Dict[str, Any]],
        *,
        custom_overrides: Dict[str, Any],
        router_overrides: Dict[str, Any],
        perceptor_overrides: Dict[str, Any],
    ) -> Optional[Any]:
        """Apply explicit user overrides into bootstrap bundle parts."""
        bridge = parse_factory_bridge_overrides(overrides)
        if bridge.router_overrides:
            router_overrides.clear()
            router_overrides.update(bridge.router_overrides)
        if bridge.perceptor_overrides:
            perceptor_overrides.clear()
            perceptor_overrides.update(bridge.perceptor_overrides)

        bootstrap_env_profile = bridge.bridge_env_profile
        if not bridge.custom_overrides:
            return bootstrap_env_profile
        for key, value in bridge.custom_overrides.items():
            if key in TRAIN_PARAMS_BOOTSTRAP_METADATA_FIELDS:
                raise ValueError(
                    f"Training-only bootstrap metadata override '{key}' is no longer supported. "
                    "Use run_train()/TrainingConfig for runtime training settings."
                )
            if hasattr(train, key):
                setattr(train, key, value)
            else:
                custom_overrides[key] = value
        return bootstrap_env_profile

    @classmethod
    def _assemble(
        cls,
        profile: EnvProfile,
        overrides: Optional[Dict[str, Any]] = None,
        *,
        policy: Optional[ConfigPolicyOverrides] = None,
    ) -> "ConfigBundle":
        """Assemble a bootstrap bundle from profile defaults, optional policy, and explicit overrides."""
        train = cls._make_bootstrap_train_params_for_profile(profile)
        custom_ov: Dict[str, Any] = {}
        router_ov: Dict[str, Any] = {}
        perceptor_ov: Dict[str, Any] = {}
        bootstrap_env_profile: Optional[Any] = None

        if policy is not None:
            bootstrap_env_profile = cls._merge_config_policy_into_parts(
                policy,
                custom_overrides=custom_ov,
                router_overrides=router_ov,
                perceptor_overrides=perceptor_ov,
            )

        explicit_bootstrap_env_profile = cls._merge_explicit_overrides_into_parts(
            train,
            overrides,
            custom_overrides=custom_ov,
            router_overrides=router_ov,
            perceptor_overrides=perceptor_ov,
        )
        if explicit_bootstrap_env_profile is not None:
            bootstrap_env_profile = explicit_bootstrap_env_profile

        return cls(
            train=train,
            profile=profile,
            custom_overrides=custom_ov,
            router_overrides=router_ov,
            perceptor_overrides=perceptor_ov,
            bootstrap_env_profile=bootstrap_env_profile,
        )

    @classmethod
    def for_profile(
        cls,
        profile: EnvProfile,
        overrides: Optional[Dict] = None,
    ) -> "ConfigBundle":
        """Build a bootstrap bundle from profile-only defaults and explicit overrides."""
        return cls._assemble(
            profile,
            overrides,
            policy=None,
        )

    @classmethod
    def for_env(
        cls,
        env: Any,
        overrides: Optional[Dict] = None,
        *,
        include_env_profile: bool = True,
    ) -> "ConfigBundle":
        """Build a bootstrap bundle with env-derived ConfigPolicy injection."""
        profile = extract_env_profile(env)
        return cls._assemble(
            profile,
            overrides,
            policy=cls._load_profile_config_policy(
                env,
                include_env_profile=include_env_profile,
            ),
        )

# ═══════════════════════════════════════════════════════════
#  底层训练配置（ModelConfig / OptimizerConfig / RLConfig）
# ═══════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    vital_dim: int = 4
    action_dim: int = 2
    hidden_dim: int = 256
    num_layers: int = 3
    router_hidden_dim: int = 128
    router_num_experts: int = 4
    router_top_k: int = 2
    use_layer_norm: bool = True
    dropout: float = 0.1
    activation: str = "relu"

    def __post_init__(self):
        _validate(self,
            positive=("vital_dim", "action_dim", "hidden_dim", "num_layers"),
            ranges=(("dropout", 0, 0.999),),
            choices=(("activation", ("relu", "gelu", "tanh", "silu")),),
        )


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    use_grad_clip: bool = True
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000

    def __post_init__(self):
        _validate(self,
            positive=("learning_rate", "max_grad_norm"),
            non_neg=("weight_decay",),
            choices=(("scheduler_type", ("cosine", "linear", "constant")),),
        )


@dataclass
class RLConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    use_vectorized_gae: bool = True

    value_loss_coef: float = 0.5
    value_huber_delta: float = 1.0
    entropy_coef: float = 0.01
    # Fixed entropy scale used by Dreamer-style actor loss.
    actor_entropy_scale: float = 1e-4
    # Dreamer-style actor gradient mode: dynamics | reinforce | both
    imag_gradient: str = "dynamics"
    # Mix factor used when imag_gradient == "both".
    imag_gradient_mix: float = 0.0
    # Enable Dreamer reward EMA normalization (P5/P95).
    use_reward_ema: bool = True
    # Normalize critic value/return scale with running statistics.
    use_value_normalization: bool = True
    # Normalize actor advantages for steadier gradient magnitude.
    use_advantage_normalization: bool = True
    # Weight for analytic return-gradient actor objective.
    actor_analytic_weight: float = 1.0
    # Auxiliary REINFORCE weight for discrete policies.
    actor_reinforce_aux_weight_discrete: float = 0.1
    # 熵控制策略二选一：默认采用自适应 alpha，关闭固定阈值保护项。
    use_entropy_protection: bool = False
    entropy_protection_threshold: float = 0.5
    entropy_protection_strength: float = 0.1
    use_alpha_adaptive: bool = True
    target_entropy: Optional[float] = None
    alpha_lr: float = 3e-4
    use_value_real_anchor: bool = True
    value_real_weight: float = 0.1
    return_clip: Optional[float] = None
    # Decouple imagined critic updates from actor/world-model features.
    detach_critic_features_on_imagination: bool = False
    # Base weight for slow-target critic regularization.
    slow_value_reg_weight: float = 1.0
    # Increase slow-target regularization once online/target critic drift grows.
    slow_value_reg_drift_threshold: float = 2.0
    slow_value_reg_drift_gain: float = 0.0
    # Damp actor updates when critic drift exceeds a threshold.
    use_actor_drift_guard: bool = False
    actor_drift_guard_threshold: float = 2.0
    actor_drift_guard_gain: float = 4.0
    actor_drift_guard_floor: float = 0.1
    actor_drift_guard_imag_only: bool = True

    def __post_init__(self):
        _validate(self,

            non_neg=(
                "value_loss_coef",
                "entropy_coef",
                "actor_entropy_scale",
                "actor_analytic_weight",
                "actor_reinforce_aux_weight_discrete",
                "slow_value_reg_weight",
                "slow_value_reg_drift_threshold",
                "slow_value_reg_drift_gain",
                "actor_drift_guard_threshold",
                "actor_drift_guard_gain",
            ),
            ranges=(
                ("gamma", 0, 1),
                ("gae_lambda", 0, 1),
                ("imag_gradient_mix", 0, 1),
                ("actor_drift_guard_floor", 0, 1),
            ),
            choices=(("imag_gradient", ("dynamics", "reinforce", "both")),),
        )


@dataclass
class DebugConfig:
    enabled: bool = False


# ═══════════════════════════════════════════════════════════
#  TrainingConfig — 全量训练配置 + 字段同步
# ═══════════════════════════════════════════════════════════

_REMOVED_TRAINING_CONFIG_ALIASES = {
    "sequence_length": "seq_len",
    "batch_length": "seq_len",
    "buffer_size": "buffer_capacity",
}

_REMOVED_TRAINING_CONFIG_FIELDS = (
    "adaptive_imag_critic_trust_target_enabled",
    "adaptive_imag_critic_trust_target_takeover_floor",
)


def _complete_training_config_fields(flat: Dict[str, Any]) -> None:
    """Fill canonical sibling fields without accepting removed aliases."""
    if "seq_len" in flat and "wm_seq_len" not in flat:
        flat["wm_seq_len"] = flat["seq_len"]
    if "batch_size" in flat and "wm_batch_size" not in flat:
        flat["wm_batch_size"] = flat["batch_size"]
    if "total_steps" in flat and "num_train_steps" not in flat:
        flat["num_train_steps"] = flat["total_steps"]
    if "num_train_steps" in flat and "total_steps" not in flat:
        flat["total_steps"] = flat["num_train_steps"]
    if (
        "adaptive_imag_persistence_escape_band_release_steps" in flat
        and "adaptive_imag_compensation_persistence_escape_band_release_steps"
        not in flat
    ):
        flat["adaptive_imag_compensation_persistence_escape_band_release_steps"] = (
            flat.pop("adaptive_imag_persistence_escape_band_release_steps")
        )


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    use_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    compile_warmup: bool = False
    compile_warmup_batch_size: int = 2
    use_jit: bool = False
    compile_actor: bool = True
    compile_critic: bool = True
    compile_router: bool = True
    compile_wm_encoder: bool = True
    compile_wm_projection: bool = True

    validation_mode: Literal["warn", "strict", "off"] = "strict"
    config_mode: Literal["compat", "strict"] = "compat"
    config_version: str = "v5.2.5"
    config_hash: Optional[str] = None

    batch_size: int = 32
    seq_len: int = 32
    buffer_capacity: int = 10000
    num_train_steps: int = 100000
    wm_pretrain_steps: int = 0
    eval_interval: int = 5000
    save_interval: int = 10000
    log_interval: int = 100

    env_name: str = "CartPole-v1"
    num_envs: int = 4
    max_episode_steps: int = 500

    warmup_steps: int = 1000

    total_steps: int = 100000
    total_env_steps: int = 100000
    wm_seq_len: int = 32
    wm_batch_size: int = 32
    imagination_horizon: int = 15
    imagination_batch_size: int = 8
    imagination_continue_threshold: float = 0.5
    imagination_hard_truncate: bool = False
    imagination_only: bool = True
    critic_mode: Optional[str] = None
    critic_gammas: Tuple[float, ...] = field(default_factory=tuple)
    critic_n_ensemble: Optional[int] = None
    critic_hidden_dim: Optional[int] = None
    critic_primary_gamma_index: Optional[int] = None
    critic_use_twohot: Optional[bool] = None
    critic_use_adaptive_routing: Optional[bool] = None
    critic_pessimism: Optional[float] = None
    critic_target_update_tau: Optional[float] = None
    enable_mixed_scheduler: bool = True
    wm_real_only: bool = True
    imag_ratio_start: float = 0.0
    imag_ratio_end: float = 1.0
    imag_ratio_ramp_steps: int = 5000
    imag_ratio_max: float = 1.0
    # Dynamic guardrails for mixed real/imag scheduling.
    imag_ratio_wm_guard: float = 6.0
    imag_ratio_critic_guard: float = 250.0
    imag_ratio_guard_floor: float = 0.05
    # Clip imagined return deviation from critic bootstrap baseline.
    imag_return_delta_clip: float = 50.0
    # Optional cap on imagined continue probabilities used for λ-return / actor weights.
    # Set <=0 to disable.
    imag_continue_prob_cap: float = 0.0
    rssm_continue_temperature: float = 1.0
    rssm_continue_loss_type: str = "bce"
    rssm_continue_focal_alpha: float = 0.25
    rssm_continue_focal_gamma: float = 2.0
    rssm_continue_optimistic_bias: float = 5.0
    rssm_continue_positive_weight: float = 1.0
    rssm_continue_negative_weight: float = 1.0
    # Optional adaptive compensation phase for imagined continue capping.
    adaptive_imag_continue_cap: bool = False
    adaptive_imag_continue_cap_min: float = 0.80
    adaptive_imag_continue_cap_max: float = 0.95
    adaptive_imag_continue_cap_target_gap: float = 4.0
    adaptive_imag_continue_cap_target_continue: float = 0.97
    adaptive_imag_continue_cap_target_actor: float = 20.0
    adaptive_imag_continue_cap_gap_gain: float = 0.02
    adaptive_imag_continue_cap_continue_gain: float = 0.50
    adaptive_imag_continue_cap_actor_gain: float = 0.002
    adaptive_imag_continue_cap_ema: float = 0.8
    adaptive_imag_continue_cap_warmup_steps: int = 0
    adaptive_imag_continue_cap_ramp_steps: int = 0
    adaptive_imag_eval_confirmation_count: int = 1
    adaptive_imag_entry_window_steps: int = 0
    adaptive_imag_entry_target_continue: float = 0.0
    adaptive_imag_entry_continue_gain_scale: float = 1.0
    adaptive_imag_compensation_trigger_return_delta_clip: float = 0.0
    adaptive_imag_compensation_trigger_min_pressure: float = 0.0
    adaptive_imag_compensation_trigger_actor_scale_gain: float = 0.0
    adaptive_imag_compensation_trigger_actor_scale_floor: float = 0.1
    adaptive_imag_compensation_trigger_hysteresis: bool = False
    adaptive_imag_compensation_trigger_release_ratio: float = 0.5
    adaptive_imag_compensation_trigger_attack_ema: float = 0.5
    adaptive_imag_compensation_trigger_release_ema: float = 0.9
    adaptive_imag_compensation_trigger_confirmation_steps: int = 1
    adaptive_imag_compensation_trigger_quality_gate: bool = False
    adaptive_imag_compensation_trigger_quality_gap_threshold: float = 2.0
    adaptive_imag_compensation_trigger_quality_actor_threshold: float = 10.0
    adaptive_imag_compensation_trigger_quality_streak: int = 2
    adaptive_imag_compensation_trigger_quality_scale_gain: float = 0.0
    adaptive_imag_compensation_trigger_quality_scale_floor: float = 0.5
    adaptive_imag_compensation_trigger_quality_piecewise: bool = False
    adaptive_imag_compensation_trigger_quality_mild_threshold: float = 2.0
    adaptive_imag_compensation_trigger_quality_severe_threshold: float = 8.0
    adaptive_imag_compensation_trigger_quality_mild_floor: float = 0.85
    adaptive_imag_compensation_trigger_quality_severe_floor: float = 0.55
    adaptive_imag_post_entry_eval_threshold: float = 0.0
    adaptive_imag_post_entry_eval_confirmation_count: int = 1
    adaptive_imag_real_stability_use_pre_eval_registry_support: bool = False
    adaptive_imag_task_cert_real_policy_anchor_gate_enabled: bool = False
    adaptive_imag_task_cert_real_policy_anchor_gate_kl_scale: float = 0.15
    adaptive_imag_post_entry_internal_return_ema_threshold: float = 0.0
    adaptive_imag_post_entry_internal_min_step: int = 0
    adaptive_imag_post_entry_internal_continue_threshold: float = 0.0
    adaptive_imag_post_entry_internal_gap_max: float = 0.0
    adaptive_imag_post_entry_internal_hold_steps: int = 0
    adaptive_imag_post_entry_internal_cap: float = 0.0
    adaptive_imag_post_entry_internal_actor_scale: float = 0.0
    adaptive_imag_handoff_hold_steps: int = 0
    adaptive_imag_handoff_cap: float = 0.0
    adaptive_imag_handoff_actor_scale: float = 0.0
    adaptive_imag_post_entry_cap: float = 0.0
    adaptive_imag_post_entry_actor_scale: float = 0.0
    adaptive_imag_post_entry_soft_cap: float = 0.0
    adaptive_imag_post_entry_soft_cap_floor: float = 0.0
    adaptive_imag_post_entry_soft_actor_scale: float = 0.0
    adaptive_imag_post_entry_soft_actor_scale_floor: float = 0.0
    adaptive_imag_post_entry_soft_trigger_release_steps: int = 200
    adaptive_imag_post_entry_soft_trigger_bad_quality_max_progress: float = 1.0
    adaptive_imag_standard_soft_fallback_trigger_release_steps: int = 200
    adaptive_imag_standard_soft_fallback_trigger_release_quality_clamp: float = 1.0
    adaptive_imag_post_entry_pending_cap: float = 0.0
    adaptive_imag_post_entry_pending_actor_scale: float = 0.0
    adaptive_imag_post_entry_pending_negative_adv_threshold: float = 0.0
    adaptive_imag_post_entry_pending_negative_adv_actor_scale: float = 1.0
    adaptive_imag_post_entry_pending_negative_adv_critic_boost: float = 0.0
    adaptive_imag_post_entry_soft_negative_adv_actor_scale: float = 0.0
    adaptive_imag_post_entry_soft_negative_adv_critic_boost: float = 0.0
    adaptive_imag_post_entry_soft_negative_adv_midwater_eval_threshold: float = 0.0
    adaptive_imag_post_entry_soft_negative_adv_midwater_min_step: int = 0
    adaptive_imag_post_entry_soft_negative_adv_midwater_ramp_steps: int = 0
    adaptive_imag_post_entry_soft_negative_adv_midwater_actor_scale: float = 0.0
    adaptive_imag_post_entry_soft_negative_adv_midwater_critic_boost: float = 0.0
    adaptive_imag_post_entry_soft_negative_adv_highwater_eval_threshold: float = 0.0
    adaptive_imag_post_entry_soft_negative_adv_highwater_actor_scale: float = 0.0
    adaptive_imag_post_entry_soft_negative_adv_highwater_critic_boost: float = 0.0
    adaptive_imag_post_entry_hold_steps: int = 0
    adaptive_imag_post_entry_rearm_delta: float = 0.0
    adaptive_imag_post_entry_preview_min_step: int = 0
    adaptive_imag_post_entry_preview_hold_steps: int = 0
    adaptive_imag_post_entry_preview_improve_margin: float = 0.0
    adaptive_imag_post_entry_allow_persistence_escape: bool = False
    adaptive_imag_post_entry_commit_continue_threshold: float = 0.0
    adaptive_imag_post_entry_commit_gap_max: float = 0.0
    adaptive_imag_post_entry_commit_return_threshold: float = 0.0
    adaptive_imag_post_entry_commit_highwater_eval_threshold: float = 0.0
    adaptive_imag_post_entry_commit_highwater_gap_max: float = 0.0
    adaptive_imag_post_entry_commit_signed_adv_threshold: float = 0.0
    adaptive_imag_post_entry_commit_highwater_hold_steps: int = 0
    adaptive_imag_post_entry_commit_confirmation_steps: int = 0
    adaptive_imag_compensation_persistence_escape_band_release_steps: int = 0
    adaptive_imag_compensation_persistence_eval_threshold: float = 0.0
    adaptive_imag_compensation_persistence_requires_post_entry_commit: bool = False
    adaptive_imag_compensation_persistence_min_step: int = 0
    adaptive_imag_compensation_persistence_eval_drop: float = 0.0
    adaptive_imag_compensation_persistence_eval_confirmation_count: int = 1
    adaptive_imag_compensation_persistence_drop_confirmation_count: int = 1
    adaptive_imag_compensation_persistence_hold_steps: int = 0
    adaptive_imag_compensation_persistence_continue_threshold: float = 0.0
    adaptive_imag_compensation_persistence_adv_threshold: float = 0.0
    adaptive_imag_compensation_persistence_use_relative: bool = False
    adaptive_imag_compensation_persistence_ema: float = 0.9
    adaptive_imag_compensation_persistence_gap_scale: float = 1.25
    adaptive_imag_compensation_persistence_continue_delta: float = 0.08
    adaptive_imag_compensation_persistence_adv_delta: float = 0.75
    adaptive_imag_compensation_persistence_cap: float = 1.0
    adaptive_imag_compensation_persistence_actor_scale: float = 1.0
    adaptive_imag_compensation_persistence_highwater_eval_threshold: float = 0.0
    adaptive_imag_compensation_persistence_highwater_actor_scale: float = 0.0
    adaptive_imag_compensation_persistence_highwater_cap: float = 0.0
    adaptive_imag_compensation_persistence_highwater_tail_target_gap_cap: float = 0.0
    adaptive_imag_compensation_persistence_highwater_tail_weight_floor: float = 0.0
    adaptive_imag_compensation_persistence_highwater_tail_weight_floor_ratio: float = 0.0
    adaptive_imag_compensation_persistence_highwater_tail_critic_boost: float = 0.0
    adaptive_imag_compensation_persistence_highwater_tail_critic_continue_margin: float = 0.0
    adaptive_imag_compensation_persistence_highwater_tail_critic_real_adv_threshold: float = 0.0
    adaptive_imag_compensation_persistence_highwater_tail_critic_real_adv_margin: float = 0.0
    adaptive_imag_compensation_persistence_entry_protect_steps: int = 0
    adaptive_imag_compensation_persistence_entry_protect_eval_ratio: float = 0.0
    adaptive_imag_compensation_persistence_release_eval_ratio: float = 0.0
    adaptive_imag_compensation_persistence_release_steps: int = 0
    adaptive_imag_compensation_persistence_release_cap: float = 0.0
    adaptive_imag_compensation_persistence_release_actor_scale: float = 0.0
    adaptive_imag_compensation_persistence_negative_adv_threshold: float = 0.0
    adaptive_imag_compensation_persistence_negative_adv_actor_scale: float = 0.0
    adaptive_imag_compensation_persistence_negative_adv_critic_boost: float = 0.0
    adaptive_imag_compensation_persistence_real_advantage_threshold: float = 0.0
    adaptive_imag_compensation_persistence_real_advantage_imag_threshold: float = 0.0
    adaptive_imag_compensation_persistence_real_advantage_critic_boost: float = 0.0
    adaptive_imag_compensation_persistence_real_value_anchor_scale: float = 1.0
    adaptive_imag_compensation_persistence_tail_window: int = 0
    adaptive_imag_compensation_persistence_tail_gap_threshold: float = 0.0
    adaptive_imag_compensation_persistence_tail_continue_threshold: float = 0.0
    adaptive_imag_compensation_persistence_tail_weight_threshold: float = 0.0
    adaptive_imag_compensation_persistence_tail_weight_floor: float = 0.0
    adaptive_imag_compensation_persistence_tail_target_gap_cap: float = 0.0
    adaptive_imag_compensation_persistence_tail_weight_floor_ratio: float = 0.0
    adaptive_imag_compensation_persistence_tail_critic_boost: float = 0.0
    adaptive_imag_compensation_post_solved_eval_threshold: float = 0.0
    adaptive_imag_compensation_post_solved_eval_confirmation_count: int = 1
    adaptive_imag_compensation_post_solved_hold_steps: int = 0
    adaptive_imag_compensation_post_solved_min_step: int = 0
    adaptive_imag_compensation_post_solved_requires_persistence: bool = False
    adaptive_imag_compensation_post_solved_cap: float = 1.0
    adaptive_imag_compensation_post_solved_actor_scale: float = 1.0
    adaptive_imag_compensation_post_solved_highwater_eval_threshold: float = 0.0
    adaptive_imag_compensation_post_solved_highwater_cap: float = 0.0
    adaptive_imag_compensation_post_solved_highwater_actor_scale: float = 0.0
    adaptive_imag_compensation_post_solved_actor_anchor_eval_threshold: float = 0.0
    adaptive_imag_compensation_post_solved_actor_anchor_pull: float = 0.0
    adaptive_imag_compensation_post_solved_actor_anchor_hard_pull: float = 0.0
    adaptive_imag_compensation_post_solved_actor_anchor_latched_pull: float = 0.0
    adaptive_imag_compensation_post_solved_actor_anchor_kl: float = 0.0
    adaptive_imag_compensation_post_solved_actor_anchor_latched_kl: float = 0.0
    adaptive_imag_compensation_post_solved_actor_anchor_highwater_eval_threshold: float = 0.0
    adaptive_imag_compensation_post_solved_actor_anchor_highwater_pull: float = 0.0
    adaptive_imag_compensation_post_solved_actor_anchor_highwater_kl: float = 0.0
    adaptive_imag_compensation_post_solved_real_actor_anchor_kl: float = 0.0
    adaptive_imag_compensation_post_solved_real_advantage_threshold: float = 0.0
    adaptive_imag_compensation_post_solved_real_advantage_latch_veto: bool = False
    adaptive_imag_compensation_post_solved_real_advantage_latch_veto_threshold: float = 0.0
    adaptive_imag_compensation_post_solved_real_advantage_actor_scale: float = 1.0
    adaptive_imag_compensation_post_solved_real_advantage_critic_boost: float = 0.0
    adaptive_imag_compensation_post_solved_real_value_anchor_scale: float = 1.0
    adaptive_imag_compensation_post_solved_actor_target_base_blend: float = 0.0
    adaptive_imag_global_target_base_blend: float = 0.0
    adaptive_imag_target_value_consistency_weight: float = 0.0
    adaptive_imag_target_value_consistency_horizon: int = 3
    adaptive_imag_target_value_consistency_delta: float = 1.0
    adaptive_imag_target_value_consistency_high_value_boost: float = 0.0
    adaptive_imag_target_value_consistency_high_value_quantile: float = 0.75
    adaptive_imag_target_value_consistency_high_value_feature_scale: float = 0.0
    adaptive_imag_policy_open_loop_consistency_weight: float = 0.0
    adaptive_imag_policy_open_loop_consistency_horizon: int = 3
    adaptive_imag_policy_open_loop_consistency_delta: float = 1.0
    adaptive_imag_policy_open_loop_consistency_high_value_boost: float = 0.0
    adaptive_imag_policy_open_loop_consistency_high_value_quantile: float = 0.75
    adaptive_imag_policy_open_loop_consistency_value_scale: float = 0.0
    adaptive_imag_policy_open_loop_consistency_late_step_boost: float = 0.0
    adaptive_imag_actor_use_target_value_ruler_enabled: bool = False
    adaptive_imag_actor_use_target_value_ruler_blend: float = 1.0
    adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled: bool = False
    adaptive_imag_actor_use_target_value_ruler_gap_margin: float = 0.0
    adaptive_imag_actor_use_target_value_ruler_critic_distill_weight: float = 0.0
    adaptive_imag_post_entry_actor_base_return_cap_margin: float = -1.0
    adaptive_imag_post_entry_base_return_cap_critic_boost: float = 0.0
    adaptive_imag_post_entry_base_return_cap_value_real_anchor_scale: float = 1.0
    adaptive_imag_compensation_persistence_actor_base_return_cap_margin: float = -1.0
    adaptive_imag_compensation_persistence_base_return_cap_critic_boost: float = 0.0
    adaptive_imag_compensation_persistence_base_return_cap_value_real_anchor_scale: float = 1.0
    adaptive_imag_compensation_persistence_release_actor_base_return_cap_margin: float = -1.0
    adaptive_imag_compensation_persistence_release_base_return_cap_critic_boost: float = 0.0
    adaptive_imag_compensation_persistence_release_base_return_cap_value_real_anchor_scale: float = 1.0
    adaptive_imag_compensation_post_solved_actor_base_return_cap_margin: float = -1.0
    adaptive_imag_compensation_post_solved_base_return_cap_critic_boost: float = 0.0
    adaptive_imag_compensation_post_solved_base_return_cap_value_real_anchor_scale: float = 1.0
    adaptive_imag_idle_corridor_advantage_blend_max: float = 0.0
    adaptive_imag_idle_corridor_negative_adv_threshold: float = 0.0
    adaptive_imag_idle_corridor_negative_adv_tau: float = 1.0
    adaptive_imag_idle_corridor_quantile: float = 0.75
    adaptive_imag_idle_corridor_adv_term_clamp_scale: float = 0.0
    adaptive_imag_idle_corridor_adv_term_clamp_min: float = 0.0
    adaptive_imag_idle_corridor_clean_target_blend_max: float = 0.0
    adaptive_imag_idle_corridor_clean_target_inflation_floor_max: float = 0.0
    adaptive_imag_idle_corridor_inflation_threshold: float = 0.0
    adaptive_imag_idle_corridor_inflation_tau: float = 1.0
    adaptive_imag_task_corridor_enabled: bool = False
    adaptive_imag_task_corridor_high_quantile: float = 0.75
    adaptive_imag_task_corridor_low_quantile: float = 0.25
    adaptive_imag_task_corridor_gate_tau: float = 0.25
    adaptive_imag_task_corridor_analytic_floor: float = 0.35
    adaptive_imag_task_corridor_confidence_scale: float = 0.5
    adaptive_imag_task_cert_reward_agreement_quantile: float = 0.75
    adaptive_imag_task_cert_recovery_rate: float = 0.5
    adaptive_imag_task_cert_alarm_rate: float = 0.7
    adaptive_imag_task_cert_state_floor: float = 0.0
    adaptive_imag_actor_contract_trust_floor: float = 0.0
    adaptive_imag_actor_unified_contract_enabled: bool = False
    adaptive_imag_actor_unified_contract_blend: float = 1.0
    adaptive_imag_actor_unified_contract_tail_relief_max: float = 0.0
    adaptive_imag_actor_unified_contract_tail_relief_adv_threshold: float = 1.0
    adaptive_imag_actor_unified_contract_tail_relief_adv_tau: float = 2.0
    adaptive_imag_critic_bootstrap_contract_enabled: bool = False
    adaptive_imag_critic_bootstrap_clean_mix_max: float = 0.8
    adaptive_imag_critic_bootstrap_external_takeover_floor_ratio: float = 0.5
    adaptive_imag_critic_bootstrap_precontact_floor_max: float = 0.25
    adaptive_imag_critic_bootstrap_precontact_frontload: float = 0.0
    adaptive_imag_critic_bootstrap_semantic_debt_decay: float = 0.9
    adaptive_imag_critic_bootstrap_contact_persistence_decay: float = 0.95
    adaptive_imag_critic_bootstrap_contact_persistence_floor: float = 0.35
    adaptive_imag_critic_bootstrap_anchor_confidence_floor: float = 0.0
    adaptive_imag_critic_bootstrap_anchor_contact_priority_bonus: float = 2.0
    adaptive_imag_critic_bootstrap_dense_contact_priority_floor: float = 0.35
    adaptive_imag_critic_bootstrap_anchor_contact_cap: float = 1.0
    adaptive_imag_critic_bootstrap_min_step: int = 1000
    adaptive_imag_critic_bootstrap_step_ramp: int = 250
    adaptive_imag_critic_bootstrap_eval_threshold: float = 200.0
    adaptive_imag_critic_bootstrap_eval_ramp: float = 50.0
    adaptive_imag_post_entry_negative_online_adv_threshold: float = 0.0
    adaptive_imag_post_entry_negative_online_adv_eval_threshold: float = 0.0
    adaptive_imag_post_entry_negative_online_adv_analytic_scale: float = 1.0
    adaptive_imag_late_trigger_negative_online_adv_threshold: float = 0.0
    adaptive_imag_late_trigger_negative_online_adv_analytic_scale: float = 1.0
    adaptive_imag_late_trigger_negative_online_adv_min_release_progress: float = 0.5
    adaptive_imag_compensation_trigger_persistence_handoff_enabled: bool = False
    adaptive_imag_compensation_trigger_persistence_handoff_min_release_progress: float = 0.75
    adaptive_imag_compensation_trigger_persistence_handoff_gap_threshold: float = 0.0
    adaptive_imag_compensation_trigger_persistence_handoff_continue_threshold: float = 0.0
    adaptive_imag_compensation_trigger_persistence_handoff_eval_threshold: float = 0.0
    adaptive_imag_compensation_trigger_persistence_handoff_landing_guard_enabled: bool = False
    adaptive_imag_compensation_trigger_persistence_handoff_block_after_late_trigger_base_cap_steps: int = 0
    adaptive_imag_compensation_trigger_persistence_handoff_confirmation_steps_after_late_trigger_base_cap: int = 0
    adaptive_imag_compensation_post_solved_negative_online_adv_threshold: float = 0.0
    adaptive_imag_compensation_post_solved_negative_online_adv_analytic_scale: float = 1.0
    adaptive_imag_compensation_post_solved_critic_anchor_weight: float = 0.0
    adaptive_imag_compensation_post_solved_drift_damping_eval_threshold: float = 0.0
    adaptive_imag_compensation_post_solved_drift_damping_wm_scale: float = 1.0
    adaptive_imag_compensation_post_solved_drift_damping_critic_scale: float = 1.0
    adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_actor_scale: float = 0.0
    adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_critic_boost: float = 0.0
    adaptive_imag_compensation_post_solved_negative_adv_threshold: float = 0.0
    adaptive_imag_compensation_post_solved_negative_adv_actor_scale: float = 1.0
    adaptive_imag_compensation_post_solved_negative_adv_critic_boost: float = 0.0
    adaptive_imag_compensation_post_solved_negative_adv_latch_steps: int = 0
    adaptive_imag_compensation_post_solved_negative_adv_latch_actor_scale: float = 1.0
    adaptive_imag_compensation_post_solved_negative_adv_latch_critic_boost: float = 0.0
    adaptive_imag_negative_adv_guard_enabled: bool = False
    adaptive_imag_negative_adv_guard_threshold: float = 0.0
    adaptive_imag_negative_adv_actor_scale: float = 1.0
    adaptive_imag_negative_adv_critic_boost: float = 0.0
    early_stop_eval_mean: float = 0.0
    early_stop_eval_patience: int = 1
    # Do not shrink imagination horizon below this value.
    imagination_horizon_min: int = 3

    rl_batch_size: int = 4
    rl_grad_clip: float = 10.0
    critic_lr: float = 3e-4
    lambda_value_real_anchor: float = 0.0
    value_real_anchor_use_mc_returns: bool = False
    value_real_anchor_corridor_quantile: float = 0.0
    adaptive_imag_critic_semantic_anchor_enabled: bool = True
    adaptive_imag_critic_semantic_anchor_weight: float = 0.1
    adaptive_imag_critic_semantic_anchor_clean_scale: float = 1.0
    adaptive_imag_critic_semantic_anchor_external_scale: float = 1.0
    adaptive_imag_critic_semantic_anchor_authority_floor: float = 0.05
    lambda_intrinsic: float = 0.0
    entropy_target_update_interval: int = 5
    disable_checkpoint: bool = False
    disable_eval: bool = False

    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    exp_name: str = "aletheia_v4.5"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

    @classmethod
    def contract_field_groups(cls) -> Dict[str, Tuple[str, ...]]:
        return {
            "certification": CONTRACT_CERTIFICATION_FIELDS,
            "authority": CONTRACT_AUTHORITY_FIELDS,
            "consumer": CONTRACT_CONSUMER_FIELDS,
            "compensation": CONTRACT_COMPENSATION_FIELDS,
        }

    def contract_group_values(self) -> Dict[str, Dict[str, Any]]:
        return {
            group: {name: getattr(self, name) for name in field_names if hasattr(self, name)}
            for group, field_names in self.contract_field_groups().items()
        }

    def __post_init__(self):
        if self.critic_mode is not None:
            self.critic_mode = str(self.critic_mode).lower()
        if self.critic_gammas:
            self.critic_gammas = tuple(sorted(float(g) for g in self.critic_gammas))
        if self.critic_n_ensemble is not None and int(self.critic_n_ensemble) <= 0:
            self._handle_mismatch(
                "critic_n_ensemble must be positive when provided",
                critical=True,
            )
        if self.critic_hidden_dim is not None and int(self.critic_hidden_dim) <= 0:
            self._handle_mismatch(
                "critic_hidden_dim must be positive when provided",
                critical=True,
            )
        if self.critic_pessimism is not None and not 0.0 < float(self.critic_pessimism) <= 0.5:
            self._handle_mismatch(
                "critic_pessimism must be in (0, 0.5] when provided",
                critical=True,
            )

        # 配置模式
        if str(self.config_mode).lower() == "strict":
            self.validation_mode = "strict"

        # [FIX-1] 每次 __post_init__ 独立的同步追踪集，避免跨实例污染
        synced: Set[frozenset] = set()

        _sync_fields(self, "total_steps", "num_train_steps",
                     handler=self._handle_mismatch, _synced=synced)
        # ``total_steps`` / ``num_train_steps`` describe the update budget,
        # while ``total_env_steps`` is a separate environment-sampling budget.
        # Strict mode should preserve both dimensions instead of treating them
        # as redundant aliases.
        _sync_fields(self, "seq_len", "wm_seq_len",
                     force_align=False, handler=self._handle_mismatch, _synced=synced)

        if self.wm_pretrain_steps + self.warmup_steps > self.num_train_steps:
            self._handle_mismatch(
                "wm_pretrain_steps + warmup_steps exceeds num_train_steps, "
                "training may skip main phase",
                critical=True,
            )

        # 批量验证
        _validate(self,
            positive=("batch_size", "seq_len", "buffer_capacity", "num_envs",
                      "wm_seq_len", "wm_batch_size", "imagination_horizon",
                      "total_steps", "total_env_steps", "num_train_steps",
                      "critic_lr", "rl_batch_size", "rl_grad_clip",
                      "imagination_horizon_min"),
            non_neg=(
                "warmup_steps",
                "imag_ratio_ramp_steps",
                "adaptive_imag_idle_corridor_advantage_blend_max",
                "adaptive_imag_idle_corridor_negative_adv_threshold",
                "adaptive_imag_idle_corridor_negative_adv_tau",
                "adaptive_imag_idle_corridor_adv_term_clamp_scale",
                "adaptive_imag_idle_corridor_adv_term_clamp_min",
                "adaptive_imag_idle_corridor_clean_target_blend_max",
                "adaptive_imag_idle_corridor_clean_target_inflation_floor_max",
                "adaptive_imag_idle_corridor_inflation_threshold",
                "adaptive_imag_idle_corridor_inflation_tau",
                "adaptive_imag_task_corridor_gate_tau",
                "adaptive_imag_task_corridor_confidence_scale",
                "adaptive_imag_task_cert_reward_agreement_quantile",
                "adaptive_imag_task_cert_recovery_rate",
                "adaptive_imag_task_cert_alarm_rate",
                "adaptive_imag_task_cert_state_floor",
                "adaptive_imag_actor_contract_trust_floor",
                "adaptive_imag_actor_unified_contract_enabled",
                "adaptive_imag_actor_unified_contract_blend",
                "adaptive_imag_actor_unified_contract_tail_relief_max",
                "adaptive_imag_actor_unified_contract_tail_relief_adv_threshold",
                "adaptive_imag_actor_unified_contract_tail_relief_adv_tau",
                "adaptive_imag_critic_bootstrap_contract_enabled",
                "adaptive_imag_critic_bootstrap_clean_mix_max",
                "adaptive_imag_critic_bootstrap_external_takeover_floor_ratio",
                "adaptive_imag_critic_bootstrap_precontact_floor_max",
                "adaptive_imag_critic_bootstrap_precontact_frontload",
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
                "adaptive_imag_critic_semantic_anchor_weight",
                "adaptive_imag_critic_semantic_anchor_clean_scale",
                "adaptive_imag_critic_semantic_anchor_external_scale",
                "adaptive_imag_critic_semantic_anchor_authority_floor",
            ),
            ranges=(("imag_ratio_start", 0.0, 1.0),
                    ("imag_ratio_end", 0.0, 1.0),
                    ("imag_ratio_max", 0.0, 1.0),
                    ("imag_ratio_guard_floor", 0.0, 1.0),
                    ("imagination_continue_threshold", 0.0, 1.0),
                    ("adaptive_imag_idle_corridor_advantage_blend_max", 0.0, 1.0),
                    ("adaptive_imag_idle_corridor_clean_target_blend_max", 0.0, 1.0),
                    (
                        "adaptive_imag_idle_corridor_clean_target_inflation_floor_max",
                        0.0,
                        1.0,
                    ),
                    ("adaptive_imag_idle_corridor_quantile", 0.0, 1.0),
                    ("adaptive_imag_task_corridor_high_quantile", 0.0, 1.0),
                    ("adaptive_imag_task_corridor_low_quantile", 0.0, 1.0),
                    ("adaptive_imag_task_corridor_analytic_floor", 0.0, 1.0),
                    ("adaptive_imag_task_cert_reward_agreement_quantile", 0.0, 1.0),
                    ("adaptive_imag_task_cert_recovery_rate", 0.0, 1.0),
                    ("adaptive_imag_task_cert_alarm_rate", 0.0, 1.0),
                    ("adaptive_imag_task_cert_state_floor", 0.0, 1.0),
                    ("adaptive_imag_actor_contract_trust_floor", 0.0, 1.0),
                    ("adaptive_imag_actor_unified_contract_blend", 0.0, 1.0),
                    (
                        "adaptive_imag_actor_unified_contract_tail_relief_max",
                        0.0,
                        1.0,
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_clean_mix_max",
                        0.0,
                        1.0,
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_external_takeover_floor_ratio",
                        0.0,
                        1.0,
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_precontact_floor_max",
                        0.0,
                        1.0,
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_precontact_frontload",
                        0.0,
                        1.0,
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_semantic_debt_decay",
                        0.0,
                        1.0,
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_contact_persistence_decay",
                        0.0,
                        1.0,
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_contact_persistence_floor",
                        0.0,
                        1.0,
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_anchor_confidence_floor",
                        0.0,
                        1.0,
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_anchor_contact_priority_bonus",
                        0.0,
                        float("inf"),
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_dense_contact_priority_floor",
                        0.0,
                        1.0,
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_anchor_contact_cap",
                        0.0,
                        1.0,
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_eval_threshold",
                        0.0,
                        float("inf"),
                    ),
                    (
                        "adaptive_imag_critic_bootstrap_eval_ramp",
                        0.0,
                        float("inf"),
                    ),
                    (
                        "adaptive_imag_critic_semantic_anchor_authority_floor",
                        0.0,
                        1.0,
                    )),
        )

        # 设备校验
        dv = str(self.device).lower()
        if not (dv in ("cuda", "cpu", "mps", "auto") or dv.startswith("cuda:")):
            raise ValueError(f"device must be auto/cuda/cpu/mps or cuda:<id>, got {self.device}")

    def _handle_mismatch(self, message: str, *, critical: bool = False) -> bool:
        mode = str(self.validation_mode).lower()
        if critical or mode == "strict":
            raise ValueError(message)
        if mode == "warn":
            warnings.warn(message)
        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_hash(self) -> str:
        data = self.to_dict()
        data.pop("config_hash", None)
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def certification_contract_config(self) -> Dict[str, Any]:
        return {
            "task_corridor_enabled": bool(self.adaptive_imag_task_corridor_enabled),
            "task_corridor_high_quantile": float(self.adaptive_imag_task_corridor_high_quantile),
            "task_corridor_low_quantile": float(self.adaptive_imag_task_corridor_low_quantile),
            "task_corridor_gate_tau": float(self.adaptive_imag_task_corridor_gate_tau),
            "task_corridor_analytic_floor": float(self.adaptive_imag_task_corridor_analytic_floor),
            "task_corridor_confidence_scale": float(self.adaptive_imag_task_corridor_confidence_scale),
            "task_cert_reward_agreement_quantile": float(
                self.adaptive_imag_task_cert_reward_agreement_quantile
            ),
            "task_cert_recovery_rate": float(self.adaptive_imag_task_cert_recovery_rate),
            "task_cert_alarm_rate": float(self.adaptive_imag_task_cert_alarm_rate),
            "task_cert_state_floor": float(self.adaptive_imag_task_cert_state_floor),
        }

    def authority_contract_config(self) -> Dict[str, Any]:
        return {
            "critic_bootstrap_contract_enabled": bool(
                self.adaptive_imag_critic_bootstrap_contract_enabled
            ),
            "clean_mix_max": float(self.adaptive_imag_critic_bootstrap_clean_mix_max),
            "external_takeover_floor_ratio": float(
                self.adaptive_imag_critic_bootstrap_external_takeover_floor_ratio
            ),
            "precontact_floor_max": float(
                self.adaptive_imag_critic_bootstrap_precontact_floor_max
            ),
            "precontact_frontload": float(
                self.adaptive_imag_critic_bootstrap_precontact_frontload
            ),
            "semantic_debt_decay": float(
                self.adaptive_imag_critic_bootstrap_semantic_debt_decay
            ),
            "contact_persistence_decay": float(
                self.adaptive_imag_critic_bootstrap_contact_persistence_decay
            ),
            "contact_persistence_floor": float(
                self.adaptive_imag_critic_bootstrap_contact_persistence_floor
            ),
            "anchor_confidence_floor": float(
                self.adaptive_imag_critic_bootstrap_anchor_confidence_floor
            ),
            "anchor_contact_priority_bonus": float(
                self.adaptive_imag_critic_bootstrap_anchor_contact_priority_bonus
            ),
            "dense_contact_priority_floor": float(
                self.adaptive_imag_critic_bootstrap_dense_contact_priority_floor
            ),
            "anchor_contact_cap": float(self.adaptive_imag_critic_bootstrap_anchor_contact_cap),
            "min_step": int(self.adaptive_imag_critic_bootstrap_min_step),
            "step_ramp": int(self.adaptive_imag_critic_bootstrap_step_ramp),
            "eval_threshold": float(self.adaptive_imag_critic_bootstrap_eval_threshold),
            "eval_ramp": float(self.adaptive_imag_critic_bootstrap_eval_ramp),
            "actor_contract_trust_floor": float(self.adaptive_imag_actor_contract_trust_floor),
            "actor_unified_contract_enabled": bool(
                self.adaptive_imag_actor_unified_contract_enabled
            ),
            "actor_unified_contract_blend": float(
                self.adaptive_imag_actor_unified_contract_blend
            ),
            "actor_unified_contract_tail_relief_max": float(
                self.adaptive_imag_actor_unified_contract_tail_relief_max
            ),
            "actor_unified_contract_tail_relief_adv_threshold": float(
                self.adaptive_imag_actor_unified_contract_tail_relief_adv_threshold
            ),
            "actor_unified_contract_tail_relief_adv_tau": float(
                self.adaptive_imag_actor_unified_contract_tail_relief_adv_tau
            ),
        }

    def consumer_contract_config(self) -> Dict[str, Any]:
        return {
            "post_solved_negative_online_adv_threshold": float(
                self.adaptive_imag_compensation_post_solved_negative_online_adv_threshold
            ),
            "post_solved_negative_online_adv_analytic_scale": float(
                self.adaptive_imag_compensation_post_solved_negative_online_adv_analytic_scale
            ),
            "post_solved_critic_anchor_weight": float(
                self.adaptive_imag_compensation_post_solved_critic_anchor_weight
            ),
            "post_solved_drift_damping_eval_threshold": float(
                self.adaptive_imag_compensation_post_solved_drift_damping_eval_threshold
            ),
            "post_solved_drift_damping_wm_scale": float(
                self.adaptive_imag_compensation_post_solved_drift_damping_wm_scale
            ),
            "post_solved_drift_damping_critic_scale": float(
                self.adaptive_imag_compensation_post_solved_drift_damping_critic_scale
            ),
            "post_solved_actor_anchor_guard_relax_actor_scale": float(
                self.adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_actor_scale
            ),
            "post_solved_actor_anchor_guard_relax_critic_boost": float(
                self.adaptive_imag_compensation_post_solved_actor_anchor_guard_relax_critic_boost
            ),
        }

    def compensation_config(self) -> Dict[str, Any]:
        return {
            "post_entry_eval_threshold": float(self.adaptive_imag_post_entry_eval_threshold),
            "post_entry_eval_confirmation_count": int(
                self.adaptive_imag_post_entry_eval_confirmation_count
            ),
            "post_entry_hold_steps": int(self.adaptive_imag_post_entry_hold_steps),
            "post_entry_rearm_delta": float(self.adaptive_imag_post_entry_rearm_delta),
            "post_entry_internal_return_ema_threshold": float(
                self.adaptive_imag_post_entry_internal_return_ema_threshold
            ),
            "post_entry_internal_min_step": int(self.adaptive_imag_post_entry_internal_min_step),
            "post_entry_internal_continue_threshold": float(
                self.adaptive_imag_post_entry_internal_continue_threshold
            ),
            "post_entry_internal_gap_max": float(
                self.adaptive_imag_post_entry_internal_gap_max
            ),
            "post_entry_internal_hold_steps": int(
                self.adaptive_imag_post_entry_internal_hold_steps
            ),
            "post_entry_internal_cap": float(self.adaptive_imag_post_entry_internal_cap),
            "post_entry_internal_actor_scale": float(
                self.adaptive_imag_post_entry_internal_actor_scale
            ),
            "handoff_hold_steps": int(self.adaptive_imag_handoff_hold_steps),
            "handoff_cap": float(self.adaptive_imag_handoff_cap),
            "handoff_actor_scale": float(self.adaptive_imag_handoff_actor_scale),
            "post_entry_cap": float(self.adaptive_imag_post_entry_cap),
            "post_entry_actor_scale": float(self.adaptive_imag_post_entry_actor_scale),
            "post_entry_soft_cap": float(self.adaptive_imag_post_entry_soft_cap),
            "post_entry_soft_cap_floor": float(
                self.adaptive_imag_post_entry_soft_cap_floor
            ),
            "post_entry_soft_actor_scale": float(
                self.adaptive_imag_post_entry_soft_actor_scale
            ),
            "post_entry_soft_actor_scale_floor": float(
                self.adaptive_imag_post_entry_soft_actor_scale_floor
            ),
            "post_entry_soft_trigger_release_steps": int(
                self.adaptive_imag_post_entry_soft_trigger_release_steps
            ),
            "post_entry_soft_trigger_bad_quality_max_progress": float(
                self.adaptive_imag_post_entry_soft_trigger_bad_quality_max_progress
            ),
            "standard_soft_fallback_trigger_release_steps": int(
                self.adaptive_imag_standard_soft_fallback_trigger_release_steps
            ),
            "standard_soft_fallback_trigger_release_quality_clamp": float(
                self.adaptive_imag_standard_soft_fallback_trigger_release_quality_clamp
            ),
            "post_entry_pending_cap": float(self.adaptive_imag_post_entry_pending_cap),
            "post_entry_pending_actor_scale": float(
                self.adaptive_imag_post_entry_pending_actor_scale
            ),
            "post_entry_pending_negative_adv_threshold": float(
                self.adaptive_imag_post_entry_pending_negative_adv_threshold
            ),
            "post_entry_pending_negative_adv_actor_scale": float(
                self.adaptive_imag_post_entry_pending_negative_adv_actor_scale
            ),
            "post_entry_pending_negative_adv_critic_boost": float(
                self.adaptive_imag_post_entry_pending_negative_adv_critic_boost
            ),
            "post_entry_soft_negative_adv_actor_scale": float(
                self.adaptive_imag_post_entry_soft_negative_adv_actor_scale
            ),
            "post_entry_soft_negative_adv_critic_boost": float(
                self.adaptive_imag_post_entry_soft_negative_adv_critic_boost
            ),
            "post_entry_soft_negative_adv_midwater_eval_threshold": float(
                self.adaptive_imag_post_entry_soft_negative_adv_midwater_eval_threshold
            ),
            "post_entry_soft_negative_adv_midwater_min_step": int(
                self.adaptive_imag_post_entry_soft_negative_adv_midwater_min_step
            ),
            "post_entry_soft_negative_adv_midwater_ramp_steps": int(
                self.adaptive_imag_post_entry_soft_negative_adv_midwater_ramp_steps
            ),
            "post_entry_soft_negative_adv_midwater_actor_scale": float(
                self.adaptive_imag_post_entry_soft_negative_adv_midwater_actor_scale
            ),
            "post_entry_soft_negative_adv_midwater_critic_boost": float(
                self.adaptive_imag_post_entry_soft_negative_adv_midwater_critic_boost
            ),
            "post_entry_soft_negative_adv_highwater_eval_threshold": float(
                self.adaptive_imag_post_entry_soft_negative_adv_highwater_eval_threshold
            ),
            "post_entry_soft_negative_adv_highwater_actor_scale": float(
                self.adaptive_imag_post_entry_soft_negative_adv_highwater_actor_scale
            ),
            "post_entry_soft_negative_adv_highwater_critic_boost": float(
                self.adaptive_imag_post_entry_soft_negative_adv_highwater_critic_boost
            ),
            "post_entry_preview_min_step": int(self.adaptive_imag_post_entry_preview_min_step),
            "post_entry_preview_hold_steps": int(
                self.adaptive_imag_post_entry_preview_hold_steps
            ),
            "post_entry_preview_improve_margin": float(
                self.adaptive_imag_post_entry_preview_improve_margin
            ),
            "post_entry_allow_persistence_escape": bool(
                self.adaptive_imag_post_entry_allow_persistence_escape
            ),
            "post_entry_commit_continue_threshold": float(
                self.adaptive_imag_post_entry_commit_continue_threshold
            ),
            "post_entry_commit_gap_max": float(self.adaptive_imag_post_entry_commit_gap_max),
            "post_entry_commit_return_threshold": float(
                self.adaptive_imag_post_entry_commit_return_threshold
            ),
            "post_entry_commit_highwater_eval_threshold": float(
                self.adaptive_imag_post_entry_commit_highwater_eval_threshold
            ),
            "post_entry_commit_highwater_gap_max": float(
                self.adaptive_imag_post_entry_commit_highwater_gap_max
            ),
            "post_entry_commit_signed_adv_threshold": float(
                self.adaptive_imag_post_entry_commit_signed_adv_threshold
            ),
            "post_entry_commit_highwater_hold_steps": int(
                self.adaptive_imag_post_entry_commit_highwater_hold_steps
            ),
            "post_entry_commit_confirmation_steps": int(
                self.adaptive_imag_post_entry_commit_confirmation_steps
            ),
            "persistence_escape_band_release_steps": int(
                self.adaptive_imag_compensation_persistence_escape_band_release_steps
            ),
            "persistence_eval_threshold": float(self.adaptive_imag_compensation_persistence_eval_threshold),
            "persistence_min_step": int(self.adaptive_imag_compensation_persistence_min_step),
            "persistence_eval_drop": float(self.adaptive_imag_compensation_persistence_eval_drop),
            "persistence_eval_confirmation_count": int(
                self.adaptive_imag_compensation_persistence_eval_confirmation_count
            ),
            "persistence_drop_confirmation_count": int(
                self.adaptive_imag_compensation_persistence_drop_confirmation_count
            ),
            "persistence_hold_steps": int(self.adaptive_imag_compensation_persistence_hold_steps),
            "persistence_continue_threshold": float(
                self.adaptive_imag_compensation_persistence_continue_threshold
            ),
            "persistence_cap": float(self.adaptive_imag_compensation_persistence_cap),
            "persistence_actor_scale": float(self.adaptive_imag_compensation_persistence_actor_scale),
            "persistence_release_eval_ratio": float(
                self.adaptive_imag_compensation_persistence_release_eval_ratio
            ),
            "persistence_release_steps": int(self.adaptive_imag_compensation_persistence_release_steps),
            "post_solved_eval_threshold": float(self.adaptive_imag_compensation_post_solved_eval_threshold),
            "post_solved_eval_confirmation_count": int(
                self.adaptive_imag_compensation_post_solved_eval_confirmation_count
            ),
            "post_solved_hold_steps": int(self.adaptive_imag_compensation_post_solved_hold_steps),
            "post_solved_min_step": int(self.adaptive_imag_compensation_post_solved_min_step),
            "post_solved_requires_persistence": bool(
                self.adaptive_imag_compensation_post_solved_requires_persistence
            ),
            "post_solved_cap": float(self.adaptive_imag_compensation_post_solved_cap),
            "post_solved_actor_scale": float(self.adaptive_imag_compensation_post_solved_actor_scale),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "TrainingConfig":
        model = ModelConfig(**data.get("model", {}))
        optimizer = OptimizerConfig(**data.get("optimizer", {}))
        rl = RLConfig(**data.get("rl", {}))
        flat = {k: v for k, v in data.items() if k not in ("model", "optimizer", "rl")}
        if "allow_legacy_field_aliases" in flat:
            raise ValueError(
                "allow_legacy_field_aliases is no longer supported. "
                "Use canonical TrainingConfig field names only."
            )
        removed_fields = sorted(
            key for key in _REMOVED_TRAINING_CONFIG_FIELDS if key in flat
        )
        if removed_fields:
            raise ValueError(
                "Deprecated TrainingConfig fields are no longer supported: "
                + ", ".join(removed_fields)
            )

        removed_aliases = {
            old: new for old, new in _REMOVED_TRAINING_CONFIG_ALIASES.items() if old in flat
        }
        if removed_aliases:
            pairs = ", ".join(f"{old}->{new}" for old, new in removed_aliases.items())
            raise ValueError(
                f"Legacy TrainingConfig aliases are no longer supported: {pairs}. "
                "Use canonical field names."
            )

        _complete_training_config_fields(flat)

        return cls(model=model, optimizer=optimizer, rl=rl, **flat)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self.to_dict()
        data["config_hash"] = self.compute_hash()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ═══════════════════════════════════════════════════════════
#  Metrics / State
# ═══════════════════════════════════════════════════════════

@dataclass
class TrainingMetrics:
    loss_total: float = 0.0
    loss_actor: float = 0.0
    loss_critic: float = 0.0
    loss_entropy: float = 0.0
    loss_alpha: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    grad_norm: float = 0.0
    entropy: float = 0.0
    alpha: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss/total": self.loss_total, "loss/actor": self.loss_actor,
            "loss/critic": self.loss_critic, "loss/entropy": self.loss_entropy,
            "loss/alpha": self.loss_alpha,
            "train/approx_kl": self.approx_kl, "train/clip_fraction": self.clip_fraction,
            "train/grad_norm": self.grad_norm,
            "train/entropy": self.entropy, "train/alpha": self.alpha,
        }


@dataclass
class EvalMetrics:
    mean_return: float = 0.0
    std_return: float = 0.0
    mean_length: float = 0.0
    success_rate: float = 0.0
    returns: List[float] = field(default_factory=list)
    lengths: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        return {
            "eval/mean_return": self.mean_return, "eval/std_return": self.std_return,
            "eval/mean_length": self.mean_length, "eval/success_rate": self.success_rate,
        }


@dataclass
class TrainState:
    """训练状态（Checkpoint 用）"""
    step: int = 0
    epoch: int = 0
    total_time: float = 0.0
    best_reward: float = -float("inf")
    last_eval_reward: float = -float("inf")
    model_opt_state: Optional[Dict[str, Any]] = None
    actor_opt_state: Optional[Dict[str, Any]] = None
    critic_opt_state: Optional[Dict[str, Any]] = None
    lr_scheduler_state: Optional[Dict[str, Any]] = None
    will_state: Optional[Dict[str, Any]] = None

    def state_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)

#  __all__
# ═══════════════════════════════════════════════════════════

__all__ = [
    # 枚举
    "EnvType",
    # 基础配置
    "SymlogConfig", "DistributionConfig", "KLConfig", "InitConfig", "TemperatureConfig",
    # 环境
    "EnvConfig", "EnvProfile", "extract_env_config", "extract_env_profile",
    # Actor / Critic
    "ContinuousDistConfig", "DiscreteDistConfig", "EntropyProtectionConfig",
    "ActorConfig", "ActionCodecConfig",
    "CriticMode", "GammaSchedulerConfig", "CriticConfig", "CriticTrainConfig",
    # Continue / 一致性
    "ContinueConfig", "MultiHorizonContinueConfig",
    "NStepTerminationConfig", "ShortcutConsistencyConfig",
    "MHCConfig", "MSCConfig", "MultiScaleContinueConfig", "NSTConfig", "SCConfig",
    "ConsistencyAuditorConfig",
    # 子系统
    "ControlConfig", "RouterConfig", "SmoothingConfig",
    "PerceptorConfig", "MemoryConfig",
    # RSSM / WorldModel
    "RSSMConfig", "GradientWallConfig", "LossWeightsConfig",
    "ProjectionConfig", "AbstractorConfig", "ConsistencyHeadConfig",
    "AuxTasksConfig", "PredictiveEngineConfig",
    "ImagineConfig", "GapMonitorConfig", "WorldModelConfig",
    # Will
    "WillConfig",
    # 训练
    "TrainParams", "AgentBootstrapTrainParams", "ConfigBundle",
    "AGENT_BOOTSTRAP_TRAIN_FIELDS", "TRAIN_PARAMS_BOOTSTRAP_METADATA_FIELDS",
    "ModelConfig", "OptimizerConfig", "RLConfig", "DebugConfig", "TrainingConfig",
    # 指标 / 状态
    "TrainingMetrics", "EvalMetrics", "TrainState",
    # 常量
    "DEFAULT_STEPS", "DEFAULT_SEED", "DEFAULT_ENV", "DEFAULT_BATCH_LENGTH",
    "DEFAULT_HORIZON", "PRETRAIN_RATIO", "WARMUP_RATIO",
    "LOG_INTERVAL", "EVAL_INTERVAL", "SAVE_INTERVAL",
    "BUFFER_CAPACITY", "EVAL_SEED_OFFSET",
]
