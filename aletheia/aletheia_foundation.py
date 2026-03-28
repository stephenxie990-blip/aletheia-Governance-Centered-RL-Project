# Aletheia v5.0.2 优化版
"""
Aletheia v5.0.2 统一核心模块优化版

变更记录（相对 v5.0.1）:
─────────────────────────────────────────────────────────
[FIX-1]  compute_kl: balance 为 float 时正确用作 alpha
[FIX-2]  safe_torch_load: numpy 补丁加版本防护；补充 Path 导入
[FIX-3]  handle_time_dim: 增加维度校验，非 2D/3D 时抛出明确错误
[FIX-4]  symexp: 默认值改用模块常量，避免每次实例化 SymlogConfig
[OPT-1]  build_mlp: 收敛为单一签名
[OPT-2]  消除 _resolve_activation 冗余包装
[OPT-3]  build_normed_mlp 已收敛为 build_mlp 单一路径
[OPT-4]  DistributionFactory 内部方法精简
[OPT-5]  _patch_config_post_init 加安全防护
─────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path  # [FIX-2] 补充导入
from typing import (
    Any, Dict, List, Literal, Optional, Tuple, Union, get_args, get_origin,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

from .aletheia_config import (
    SymlogConfig,
    DistributionConfig,
    KLConfig,
    EnvProfile,
    extract_env_config,
    extract_env_profile,
    InitConfig,
    ContinuousDistConfig,
    DiscreteDistConfig,
    EntropyProtectionConfig,
    ActorConfig,
    ActionCodecConfig,
    CriticMode,
    GammaSchedulerConfig,
    CriticConfig,
    CriticTrainConfig,
    ConfigPolicyOverrides,
    coerce_config_policy_overrides,
    TemperatureConfig,
    ContinueConfig,
    MultiHorizonContinueConfig,
    NStepTerminationConfig,
    ShortcutConsistencyConfig,
    ControlConfig,
    RouterConfig,
    SmoothingConfig,
    PerceptorConfig,
    MemoryConfig,
    RSSMConfig,
    GradientWallConfig,
    LossWeightsConfig,
    MHCConfig,
    MSCConfig,
    NSTConfig,
    SCConfig,
    ConsistencyAuditorConfig,
    ProjectionConfig,
    AbstractorConfig,
    ConsistencyHeadConfig,
    AuxTasksConfig,
    PredictiveEngineConfig,
    ImagineConfig,
    GapMonitorConfig,
    WorldModelConfig,
    WillConfig,
    MultiScaleContinueConfig,
)

logger = logging.getLogger(__name__)

# ======================================================================
# VERSION
# ======================================================================

__version__ = "5.0.2"
VERSION_NAME = f"Aletheia v{__version__}"
ALETHEIA_DEBUG = False

# 类型别名
Tensor = torch.Tensor

# [FIX-4] 模块级常量，避免在函数默认参数中反复实例化 SymlogConfig
_SYMLOG_DEFAULTS = SymlogConfig()
_DEFAULT_SYMEXP_CLIP = _SYMLOG_DEFAULTS.symexp_clip
_DEFAULT_INPUT_CLIP = _SYMLOG_DEFAULTS.input_clip
_DEFAULT_OUTPUT_CLIP = _SYMLOG_DEFAULTS.output_clip


# ======================================================================
# SECTION 1: 梯度工具函数
# ======================================================================

def scale_gradient(x: Tensor, scale: float) -> Tensor:
    """梯度缩放: x * scale + x.detach() * (1 - scale)"""
    return x * scale + x.detach() * (1 - scale)


def _as_float32(x: np.ndarray) -> np.ndarray:
    return x if x.dtype == np.float32 else x.astype(np.float32, copy=False)


def _moving_mean_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return _as_float32(x).copy()
    x = _as_float32(np.asarray(x))
    n = x.shape[0]
    if n < window:
        return np.zeros((0,), dtype=np.float32)
    c = np.cumsum(x, dtype=np.float32)
    sums = c[window - 1:] - np.concatenate(
        [np.zeros((1,), dtype=np.float32), c[:n - window]]
    )
    return sums / float(window)


# ======================================================================
# SECTION 2: 归一化层
# ======================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    相比 LayerNorm 无需计算均值，推理更快、内存更省。
    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = normed.type_as(x)
        if self.weight is not None:
            normed = normed * self.weight
        return normed

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


def create_norm(dim: int, norm_type: str = "layer", eps: float = 1e-6) -> nn.Module:
    """工厂函数：创建归一化层 ("layer" | "rms" | "none")"""
    _NORM_MAP = {"rms": RMSNorm, "layer": nn.LayerNorm, "none": nn.Identity}
    cls = _NORM_MAP.get(norm_type)
    if cls is None:
        raise ValueError(f"未知的 norm_type: {norm_type}")
    if cls is nn.Identity:
        return cls()
    return cls(dim, eps=eps) if cls is RMSNorm else cls(dim, eps=eps)


# ======================================================================
# SECTION 3: 激活函数
# ======================================================================

_ACTIVATION_MAP = {
    "silu": nn.SiLU, "swish": nn.SiLU,
    "elu": nn.ELU, "relu": nn.ReLU,
    "gelu": nn.GELU, "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}

_ACTIVATION_VALIDATION_MODE = "strict"


def set_activation_validation_mode(mode: str) -> None:
    value = str(mode).lower()
    if value not in ("warn", "strict", "off"):
        raise ValueError(f"activation validation mode must be warn/strict/off, got {mode}")
    global _ACTIVATION_VALIDATION_MODE
    _ACTIVATION_VALIDATION_MODE = value


def get_activation_class(activation: Any) -> type:
    if isinstance(activation, type):
        return activation
    if not isinstance(activation, str):
        return type(activation)
    name = activation.lower()
    cls = _ACTIVATION_MAP.get(name)
    if cls is not None:
        return cls
    if hasattr(nn, activation):
        return getattr(nn, activation)
    raise ValueError(f"Unknown activation: {activation}")


def get_activation(activation: Any) -> nn.Module:
    """将字符串/类/实例统一转换为激活函数实例"""
    if isinstance(activation, nn.Module):
        return activation
    return get_activation_class(activation)()


# [OPT-2] 删除冗余的 _resolve_activation，直接使用 get_activation


# ======================================================================
# SECTION 4: MLP 层
# ======================================================================

class SwiGLUMLP(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit MLP
    结构: x → Linear(2h) → chunk → a * SiLU(b) → Linear → out
    """

    def __init__(
        self, in_dim: int, hidden_dim: int,
        out_dim: Optional[int] = None, dropout: float = 0.0, bias: bool = True,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.w1 = nn.Linear(in_dim, hidden_dim * 2, bias=bias)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1.0)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        nn.init.xavier_uniform_(self.w2.weight, gain=0.02)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)

    def forward(self, x: Tensor) -> Tensor:
        x_main, gate = self.w1(x).chunk(2, dim=-1)
        return self.w2(self.dropout(x_main * F.silu(gate)))


class ResidualSwiGLUBlock(nn.Module):
    """Pre-Norm + SwiGLU + Residual"""

    def __init__(self, dim: int, hidden_ratio: float = 2.67,
                 dropout: float = 0.0, norm_type: str = "rms"):
        super().__init__()
        hidden_dim = ((int(dim * hidden_ratio) + 7) // 8) * 8  # 8 对齐
        self.norm = create_norm(dim, norm_type)
        self.mlp = SwiGLUMLP(dim, hidden_dim, dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.norm(x))


def build_swiglu_mlp(
    in_dim: int, out_dim: int,
    hidden_dims: Optional[Tuple[int, ...]] = None,
    num_layers: int = 2, hidden_ratio: float = 2.67,
    dropout: float = 0.0, norm_type: str = "rms",
) -> nn.Module:
    """构建 SwiGLU MLP 堆栈"""
    layers: List[nn.Module] = []
    current_dim = in_dim

    if hidden_dims is not None:
        for h_dim in hidden_dims:
            if current_dim != h_dim:
                layers.append(nn.Linear(current_dim, h_dim))
            layers.append(ResidualSwiGLUBlock(h_dim, hidden_ratio, dropout, norm_type))
            current_dim = h_dim
    else:
        for _ in range(max(num_layers - 1, 0)):
            layers.append(ResidualSwiGLUBlock(in_dim, hidden_ratio, dropout, norm_type))
        current_dim = in_dim

    if current_dim != out_dim:
        layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers) if layers else nn.Identity()


def build_mlp(
    in_dim: Optional[int] = None,
    out_dim: Optional[int] = None,
    hidden_dims: Optional[Tuple[int, ...]] = None,
    *,
    hidden_dim: int = 256,
    depth: int = 2,
    num_layers: Optional[int] = None,
    activation: str = "silu",
    norm: str = "layer",
    dropout: float = 0.0,
    output_activation: bool = False,
    final_gain: float = 0.01,
    use_swiglu: Optional[bool] = None,
    init: str = "orthogonal",
) -> nn.Module:
    """
    通用 MLP 构建器。

    参数:
        in_dim:  输入维度（必需）
        out_dim: 输出维度（必需）
        hidden_dims: 隐藏层维度元组，指定后忽略 hidden_dim/depth
        hidden_dim: 统一隐藏层宽度（hidden_dims 未指定时生效）
        depth: 总层数（含输出层），等价于 num_layers
        activation: 激活函数名
        norm: 归一化类型 ("layer" | "rms" | "none")
        dropout: dropout 率
        output_activation: 输出层后是否加激活
        final_gain: 输出层 orthogonal init 的 gain
        use_swiglu: 是否使用 SwiGLU（None=自动判断）
        init: 初始化方式 ("orthogonal" | "final_gain_only")
    """
    if in_dim is None or out_dim is None:
        raise ValueError("build_mlp 需要 in_dim 和 out_dim")

    if hidden_dims is None:
        n_hidden = max(0, (num_layers or depth) - 1)
        hidden_dims = (hidden_dim,) * n_hidden

    if use_swiglu is None:
        use_swiglu = norm in ("layer", "rms") and activation.lower() in ("silu", "swish")

    if use_swiglu:
        core = build_swiglu_mlp(
            in_dim=in_dim, out_dim=out_dim,
            hidden_dims=hidden_dims or None,
            num_layers=len(hidden_dims) + 1,
            dropout=dropout, norm_type="rms",
        )
        if init in ("orthogonal", "final_gain_only") and final_gain != 1.0:
            _apply_final_gain(core, final_gain)
        if output_activation:
            return nn.Sequential(core, create_norm(out_dim, "rms"), nn.SiLU())
        return core

    # 标准 MLP 路径
    layers: List[nn.Module] = []
    curr_in = in_dim
    for h in hidden_dims:
        linear = nn.Linear(curr_in, h)
        if init == "orthogonal":
            nn.init.orthogonal_(linear.weight, gain=1.0)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        layers.extend([linear, create_norm(h, norm), get_activation(activation)])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        curr_in = h

    out_linear = nn.Linear(curr_in, out_dim)
    if init in ("orthogonal", "final_gain_only"):
        nn.init.orthogonal_(out_linear.weight, gain=final_gain)
        if out_linear.bias is not None:
            nn.init.zeros_(out_linear.bias)
    layers.append(out_linear)

    if output_activation:
        layers.extend([create_norm(out_dim, norm), get_activation(activation)])

    return nn.Sequential(*layers)


def _apply_final_gain(module: nn.Module, gain: float):
    """对 Sequential 最后一个 Linear 施加 orthogonal init"""
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if linears:
        nn.init.orthogonal_(linears[-1].weight, gain=gain)
        if linears[-1].bias is not None:
            nn.init.zeros_(linears[-1].bias)


# ─────────────────────────────────────────────────────────────────
# [FIX-3] 时间维度装饰器
# ─────────────────────────────────────────────────────────────────

def handle_time_dim(func):
    """
    自动折叠/展开时间维度的装饰器。

    仅适用于输入为 (B, D) 或 (B, T, D) 的情况。
    对于图像等高维输入，请使用 handle_nd_time_dim。
    """
    def wrapper(self, x: Tensor, *args, **kwargs) -> Tensor:
        if x.dim() == 2:
            return func(self, x, *args, **kwargs)
        if x.dim() == 3:
            B, T, D = x.shape
            out = func(self, x.reshape(B * T, D), *args, **kwargs)
            return out.reshape(B, T, *out.shape[1:])
        # [FIX-3] 明确拒绝不支持的维度，而非静默产生错误结果
        raise ValueError(
            f"handle_time_dim 仅支持 2D (B,D) 或 3D (B,T,D) 输入，"
            f"收到 {x.dim()}D 张量 shape={tuple(x.shape)}。"
            f"对于高维输入请使用 handle_nd_time_dim。"
        )
    return wrapper


def handle_nd_time_dim(func):
    """通用版本：将 (..., D) 折叠为 (*, D) 再展开回原形状"""
    def wrapper(self, x: Tensor, *args, **kwargs) -> Tensor:
        if x.dim() <= 2:
            return func(self, x, *args, **kwargs)
        leading = x.shape[:-1]
        out = func(self, x.reshape(-1, x.shape[-1]), *args, **kwargs)
        return out.reshape(*leading, *out.shape[1:])
    return wrapper


# ======================================================================
# SECTION 5: 数学与梯度工具
# ======================================================================

def sg(x: Tensor) -> Tensor:
    """停止梯度 (stop gradient)"""
    return x.detach()


def symlog(x: Tensor) -> Tensor:
    """对称对数: sign(x) * log1p(|x|)"""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor, max_val: float = _DEFAULT_SYMEXP_CLIP) -> Tensor:
    """symlog 逆变换: sign(x) * (exp(min(|x|, max_val)) - 1)"""
    return torch.sign(x) * (torch.exp(torch.abs(x).clamp(max=max_val)) - 1.0)


def huber_loss(
    pred: Tensor, target: Tensor, delta: float = 1.0, reduction: str = "none",
) -> Tensor:
    return F.huber_loss(pred, target, reduction=reduction, delta=delta)


def gamma_to_key(g: float) -> str:
    return f"g_{int(round(g * 1_000_000))}"


class SymlogLayer(nn.Module):
    """Symlog / Symexp 变换层（含 clip 保护）"""

    def __init__(
        self,
        input_clip: float = _DEFAULT_INPUT_CLIP,
        symexp_clip: float = _DEFAULT_SYMEXP_CLIP,
        output_clip: float = _DEFAULT_OUTPUT_CLIP,
    ):
        super().__init__()
        self.input_clip = input_clip
        self.symexp_clip = symexp_clip
        self.output_clip = output_clip

    def symlog(self, x: Tensor) -> Tensor:
        x = x.clamp(-self.input_clip, self.input_clip)
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def symexp(self, x: Tensor) -> Tensor:
        x = x.clamp(-self.symexp_clip, self.symexp_clip)
        return torch.sign(x) * torch.expm1(torch.abs(x)).clamp(max=self.output_clip)

    def forward(self, x: Tensor) -> Tensor:
        return self.symlog(x)


# ======================================================================
# SECTION 6: 损失函数
# ======================================================================


# ======================================================================
# SECTION 7: 回报与不确定性
# ======================================================================

def _compute_gae_single_gamma_impl(
    delta: Tensor, values: Tensor, gamma_lambda: float,
) -> Tensor:
    """单 gamma GAE 计算。

    Python 3.14+ 上 ``torch.jit.script`` 会触发 PyTorch 的弃用警告，
    因此这里把脚本化变成可选加速，而不是导入时强制执行的副作用。
    """
    T = delta.shape[0]
    returns = torch.empty_like(values)
    gae = torch.zeros_like(values[0])
    for t in range(T - 1, -1, -1):
        gae = delta[t] + gamma_lambda * gae
        returns[t] = gae + values[t]
    return returns


if sys.version_info < (3, 14):
    try:
        _compute_gae_single_gamma = torch.jit.script(_compute_gae_single_gamma_impl)
    except Exception:
        _compute_gae_single_gamma = _compute_gae_single_gamma_impl
else:
    _compute_gae_single_gamma = _compute_gae_single_gamma_impl


def compute_lambda_returns(
    rewards: Tensor,
    values: Tensor,
    bootstrap: Tensor,
    gamma: Union[float, List[float]],
    lambda_: float,
) -> Tensor:
    """
    计算 λ-returns，支持单/多 gamma。

    参数:
        rewards:   [T, B, ...]
        values:    [T, B, ...]
        bootstrap: [B, ...]  即 V(s_{T+1})
        gamma:     单 float 或 float 列表
        lambda_:   GAE lambda

    返回:
        [T, B, ...] 或 [T, B, ..., G]（多 gamma 时 G 维堆叠）
    """
    gammas = [gamma] if isinstance(gamma, (int, float)) else list(gamma)
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(0)], dim=0)

    results: List[Tensor] = []
    for g in gammas:
        delta = rewards + g * next_values - values
        results.append(_compute_gae_single_gamma(delta, values, g * lambda_))

    return results[0] if len(results) == 1 else torch.stack(results, dim=-1)


# ======================================================================
# SECTION 8: 通用工具函数
# ======================================================================


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak 软更新: θ_t ← τ·θ_s + (1-τ)·θ_t"""
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.lerp_(s, tau)


def normalize_tensor(x: Tensor, eps: float = 1e-5) -> Tensor:
    """标准化到 mean=0, std=1（batch=1 安全）"""
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


# ─────────────────────────────────────────────────────────────────
# [FIX-2] safe_torch_load: 安全的 numpy 补丁 + Path 导入
# ─────────────────────────────────────────────────────────────────

def _patch_numpy_safe_globals() -> None:
    """针对 PyTorch 2.4+ 的 numpy 序列化白名单补丁（版本安全）"""
    try:
        import torch.serialization
        if not hasattr(torch.serialization, "add_safe_globals"):
            return

        dtype_safe_types = []
        for dtype in (
            np.bool_,
            np.float16,
            np.float32,
            np.float64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ):
            try:
                dtype_safe_types.append(type(np.dtype(dtype)))
            except TypeError:
                continue

        safe_types = [np.generic, np.dtype, np.ndarray, EnvProfile, *dtype_safe_types]

        for attr_paths in (
            ("_core.multiarray._reconstruct", "core.multiarray._reconstruct"),
            ("_core.multiarray.scalar", "core.multiarray.scalar"),
        ):
            for attr_path in attr_paths:
                parts = attr_path.split(".")
                obj = np
                for part in parts:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if obj is not None:
                    safe_types.append(obj)
                    break

        torch.serialization.add_safe_globals(safe_types)
    except (AttributeError, ImportError, TypeError):
        pass


def safe_torch_load(
    path: Union[str, Path],
    map_location: str = "cpu",
    weights_only: bool = True,
    allow_unsafe_fallback: bool = False,
    trusted_source: bool = False,
) -> Any:
    """安全地加载 PyTorch checkpoint，兼容新旧版本并处理 numpy 序列化问题。"""
    if allow_unsafe_fallback:
        raise ValueError(
            "allow_unsafe_fallback is no longer supported in safe_torch_load"
        )
    _patch_numpy_safe_globals()

    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError as e:
        # 旧版 PyTorch 不支持 weights_only 参数
        if "unexpected keyword argument 'weights_only'" in str(e):
            return torch.load(path, map_location=map_location)
        raise


# ======================================================================
# SECTION 9: 形状工具（contract 检查）
# ======================================================================

def _extract_shape(x: Any) -> Optional[Tuple[int, ...]]:
    s = getattr(x, "shape", None)
    return tuple(int(i) for i in s) if s is not None else None


def _time_len(s: Optional[Tuple[int, ...]]) -> Optional[int]:
    if s is None or len(s) == 0:
        return None
    return int(s[1]) if len(s) >= 2 else int(s[0])


def _batch_len(s: Optional[Tuple[int, ...]]) -> Optional[int]:
    if s is None or len(s) == 0:
        return None
    return int(s[0])


def check_temporal_contract(batch: Any = None, *, strict: bool = True) -> Dict[str, Any]:
    """检查 batch 的时序契约: vitals_T == actions_T + 1"""
    issues: List[str] = []

    if batch is None:
        issues.append("batch is None")
    elif not isinstance(batch, dict):
        issues.append(f"batch 必须是 dict，实际为 {type(batch).__name__}")
    else:
        keys = ("vitals", "actions", "rewards", "dones")
        shapes = {k: _extract_shape(batch.get(k)) for k in keys}
        for k, s in shapes.items():
            if s is None:
                issues.append(f"缺少 {k}")

        sv, sa, sr, sd = (shapes[k] for k in keys)
        tv, ta, tr, td = map(_time_len, (sv, sa, sr, sd))

        if tv is not None and ta is not None and tv != ta + 1:
            issues.append(f"vitals_T 必须 == actions_T + 1 ({tv} vs {ta})")
        if ta is not None and tr is not None and ta != tr:
            issues.append(f"actions_T 必须 == rewards_T ({ta} vs {tr})")
        if ta is not None and td is not None and ta != td:
            issues.append(f"actions_T 必须 == dones_T ({ta} vs {td})")

        bv = _batch_len(sv)
        for name in ("actions", "rewards", "dones"):
            b = _batch_len(shapes[name])
            if bv is not None and b is not None and b != bv:
                issues.append(f"{name} batch 不匹配 ({b} vs {bv})")

    valid = len(issues) == 0
    if strict and not valid:
        raise ValueError("时序契约违反: " + "; ".join(issues))
    return {"valid": valid, "issues": issues}


def check_imagine_trajectory_contract(
    imagine_out: Any = None, *, strict: bool = True,
) -> Dict[str, Any]:
    """检查想象轨迹的形状一致性"""
    issues: List[str] = []

    if imagine_out is None:
        issues.append("imagine_out is None")
    else:
        field_names = (
            "f_policy", "actions", "log_probs", "old_log_probs",
            "values", "rewards_pred", "continues_pred", "final_value",
        )
        shapes = {
            name: _extract_shape(getattr(imagine_out, name, None))
            for name in field_names
        }
        ref = shapes.get("f_policy")
        if ref is not None and len(ref) >= 2:
            hf, bf = int(ref[0]), int(ref[1])
            for name in ("actions", "log_probs", "values", "rewards_pred", "continues_pred"):
                s = shapes.get(name)
                if s is not None and len(s) >= 2:
                    if int(s[0]) != hf:
                        issues.append(f"{name} horizon 不匹配 ({s[0]} vs {hf})")
                    if int(s[1]) != bf:
                        issues.append(f"{name} batch 不匹配 ({s[1]} vs {bf})")

            lp, op = shapes.get("log_probs"), shapes.get("old_log_probs")
            if lp is not None and op is not None and lp != op:
                issues.append(f"old_log_probs 形状不匹配 ({op} vs {lp})")

            fv = shapes.get("final_value")
            if fv is not None and len(fv) >= 1 and int(fv[0]) != bf:
                issues.append(f"final_value batch 不匹配 ({fv} vs B={bf})")

    valid = len(issues) == 0
    if strict and not valid:
        raise ValueError("想象轨迹契约违反: " + "; ".join(issues))
    return {"valid": valid, "issues": issues}


# ======================================================================
# SECTION 10: Critic 辅助函数
# ======================================================================

def is_multihead_critic(c: Any) -> bool:
    return hasattr(c, "gammas") and len(getattr(c, "gammas", ())) > 1


# ======================================================================
# SECTION 11: ActionCodec
# ======================================================================

class ActionCodec(nn.Module):
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

        self.action_dim = int(action_dim)
        self.is_discrete = bool(is_discrete)
        self.config = config

        if (not self.is_discrete) and action_low is not None and action_high is not None:
            low = torch.as_tensor(action_low, dtype=torch.float32).reshape(-1)
            high = torch.as_tensor(action_high, dtype=torch.float32).reshape(-1)
            if low.numel() != self.action_dim or high.numel() != self.action_dim:
                raise ValueError("action_low/high size mismatch with action_dim")
            self.register_buffer("action_low", low)
            self.register_buffer("action_high", high)
            self.register_buffer("action_scale", (high - low) / 2)
            self.register_buffer("action_bias", (high + low) / 2)
        else:
            self.action_low = self.action_high = None
            self.action_scale = self.action_bias = None

        hidden = list(config.hidden_dims)
        norm = "layer" if config.use_layer_norm else "none"

        self.encoder = build_mlp(
            in_dim=self.action_dim,
            out_dim=int(config.embed_dim),
            hidden_dims=tuple(hidden),
            activation="silu", norm=norm,
            output_activation=True, final_gain=1.0,
            init="final_gain_only",
        )

        self.decoder = build_mlp(
            in_dim=int(config.embed_dim),
            out_dim=self.action_dim,
            hidden_dims=tuple(reversed(hidden)),
            activation="silu", norm=norm,
            output_activation=False, final_gain=1.0,
            init="final_gain_only",
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
        cls, action_space: Any, config: Optional[ActionCodecConfig] = None,
    ) -> "ActionCodec":
        if hasattr(action_space, "n"):
            return cls(action_dim=int(action_space.n), is_discrete=True, config=config)
        shape = getattr(action_space, "shape", None)
        if shape is None:
            raise TypeError(f"Unsupported action_space: {action_space}")
        return cls(
            action_dim=int(np.prod(shape)),
            is_discrete=False, config=config,
            action_low=getattr(action_space, "low", None),
            action_high=getattr(action_space, "high", None),
        )


# ======================================================================
# SECTION 14: 分布工厂
# ======================================================================

class DistributionFactory:
    """
    分布工厂

    功能:
    - Categorical / MixedCategorical 分布创建
    - Gumbel-Softmax STE 采样
    - 平衡 KL 散度计算
    - 量化一致性损失
    """

    def __init__(
        self,
        config: Optional[DistributionConfig] = None,
        num_distributions: Optional[int] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = DistributionConfig(
                num_distributions=num_distributions or 32,
                num_classes=num_classes or 32,
                **{
                    k: v for k, v in kwargs.items()
                    if k in DistributionConfig.__dataclass_fields__
                },
            )
        self.N = self.config.num_distributions
        self.K = self.config.num_classes

    def create_distribution(
        self,
        logits_or_params: Union[Tensor, D.Distribution],
        dist_type: Optional[str] = None,
    ) -> D.Distribution:
        """创建分布：(B, N*K) 或 (B, N, K) → Independent(Categorical, 1)"""
        if isinstance(logits_or_params, D.Distribution):
            return logits_or_params

        dist_type = dist_type or self.config.effective_dist_type
        logits = self._ensure_3d(logits_or_params)

        if dist_type == "categorical":
            return D.Independent(D.Categorical(logits=logits), 1)
        if dist_type == "mixed_categorical":
            return self._create_mixed_categorical(logits)
        raise ValueError(f"未知的 dist_type: {dist_type}")

    def _create_mixed_categorical(self, logits: Tensor) -> D.Distribution:
        """Unimix Categorical: (1-r)*softmax(logits) + r/K"""
        r = self.config.unimix_ratio
        probs = F.softmax(logits, dim=-1)
        mixed = (1 - r) * probs + r / self.K
        return D.Independent(D.Categorical(probs=mixed), 1)

    def _ensure_3d(self, logits: Tensor) -> Tensor:
        return logits.reshape(-1, self.N, self.K) if logits.dim() == 2 else logits

    def sample(
        self,
        dist: D.Distribution,
        temperature: float = 1.0,
        deterministic: bool = False,
        return_soft: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Gumbel-Softmax STE 采样 → (z_hard_flat, z_soft_flat|None)"""
        logits = self._extract_logits(dist)
        B, N, K = logits.shape

        if deterministic:
            z_soft = F.softmax(logits / temperature, dim=-1)
            indices = logits.argmax(dim=-1)
        else:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)))
            z_soft = F.softmax((logits + gumbel) / temperature, dim=-1)
            indices = z_soft.argmax(dim=-1)

        z_hard = F.one_hot(indices, K).float()
        # STE: 前向用 hard，反向用 soft 梯度
        z_hard = z_hard - z_soft.detach() + z_soft
        z_flat = z_hard.reshape(B, N * K)

        if return_soft:
            return z_flat, z_soft.reshape(B, N * K)
        return z_flat, None

    # ─────────────────────────────────────────────────────────────
    # [FIX-1] compute_kl: 正确处理 balance 为 float 的情况
    # ─────────────────────────────────────────────────────────────
    def compute_kl(
        self,
        q: Union[Tensor, D.Distribution],
        p: Union[Tensor, D.Distribution],
        balance: Union[bool, float] = True,
        alpha: float = 0.8,
    ) -> Tensor:
        """
        计算 KL 散度 D_KL[q || p]

        当 balance=True 时使用 Dreamer-style 平衡 KL:
            L = α·KL(sg(q)||p) + (1-α)·KL(q||sg(p))

        参数:
            q: 后验分布/logits
            p: 先验分布/logits
            balance: True/False 或 float∈[0,1]（float 值直接作为 alpha）
            alpha: 平衡系数（仅 balance=True 且未传 float 时使用默认值）
        """
        # [FIX-1] float balance 正确用作 alpha 而非丢弃
        if isinstance(balance, float) and not isinstance(balance, bool):
            if not (0.0 <= balance <= 1.0):
                raise ValueError(
                    f"balance 作为 float 必须在 [0, 1] 范围内，收到 {balance}"
                )
            alpha = balance
            balance = True

        dist_type = self.config.effective_dist_type
        q_dist, q_logits = self._to_dist_and_logits(q, dist_type)
        p_dist, p_logits = self._to_dist_and_logits(p, dist_type)

        if balance and q_logits is not None and p_logits is not None:
            q_sg = self.create_distribution(q_logits.detach(), dist_type)
            p_sg = self.create_distribution(p_logits.detach(), dist_type)
            return (
                alpha * self._kl_categorical(q_sg, p_dist)
                + (1 - alpha) * self._kl_categorical(q_dist, p_sg)
            )
        return self._kl_categorical(q_dist, p_dist)

    def _kl_categorical(self, q: D.Distribution, p: D.Distribution) -> Tensor:
        """D_KL[q || p] = Σ q(x)·(log q(x) - log p(x))"""
        q_logits = self._get_logits(q.base_dist)
        p_logits = self._get_logits(p.base_dist)

        q_logprobs = F.log_softmax(q_logits, dim=-1)
        p_logprobs = F.log_softmax(p_logits, dim=-1)

        kl_per_class = q_logprobs.exp() * (q_logprobs - p_logprobs)
        return kl_per_class.sum(dim=-1).sum(dim=-1)  # (B,)

    @staticmethod
    def _get_logits(base_dist: D.Categorical) -> Tensor:
        if hasattr(base_dist, "logits") and base_dist.logits is not None:
            return base_dist.logits
        return torch.log(base_dist.probs.clamp(min=1e-8))

    def _to_dist_and_logits(
        self, x: Union[Tensor, D.Distribution], dist_type: str,
    ) -> Tuple[D.Distribution, Optional[Tensor]]:
        if isinstance(x, Tensor):
            return self.create_distribution(x, dist_type), x
        return x, self._try_extract_logits(x)

    @staticmethod
    def _try_extract_logits(dist: D.Distribution) -> Optional[Tensor]:
        base = getattr(dist, "base_dist", dist)
        logits = getattr(base, "logits", None)
        if logits is not None:
            return logits
        probs = getattr(base, "probs", None)
        return torch.log(probs.clamp(min=1e-8)) if probs is not None else None

    @staticmethod
    def _extract_logits(dist: D.Distribution) -> Tensor:
        base = getattr(dist, "base_dist", dist)
        if hasattr(base, "logits") and base.logits is not None:
            return base.logits
        return torch.log(base.probs.clamp(min=1e-8))

    def compute_quant_loss(self, z_soft: Optional[Tensor], z_hard: Tensor) -> Tensor:
        if z_soft is None:
            return torch.zeros(1, device=z_hard.device, dtype=z_hard.dtype).squeeze()
        return F.mse_loss(z_soft, z_hard.detach())

    def get_entropy(self, dist: D.Distribution) -> Tensor:
        return dist.entropy()

    def get_mode(self, dist: D.Distribution) -> Tensor:
        logits = self._extract_logits(dist)
        B, N, K = logits.shape
        return F.one_hot(logits.argmax(dim=-1), K).float().reshape(B, N * K)


# ======================================================================
# SECTION 14: 环境预设
# ======================================================================

_ROUTER_OVERRIDE_WHITELIST = frozenset({
    "uncertainty_quantile", "uncertainty_warmup_steps",
    "uncertainty_hysteresis", "uncertainty_window_size",
    "uncertainty_alpha", "uncertainty_ema_decay",
    "mask_x_factor", "mask_z_factor",
    "ema_blend_eval", "ema_blend_train",
    "hard_fallback_enabled",
    "hard_fallback_weights_3branch", "hard_fallback_weights_2branch",
    "mode", "proj_dim", "temp_range", "anneal_steps",
})

_ROUTER_PRESETS: Dict[str, Dict[str, Any]] = {
    "SIMPLE_VECTOR": {
        "uncertainty_quantile": 0.80,
        "uncertainty_warmup_steps": 100,
        "uncertainty_window_size": 1000,
        "uncertainty_hysteresis": (0.85, 1.15),
        "mask_x_factor": 0.4, "mask_z_factor": 0.2,
        "hard_fallback_weights_3branch": (0.4, 0.5, 0.1),
        "hard_fallback_weights_2branch": (0.4, 0.6),
    },
    "COMPLEX_VECTOR": {
        "uncertainty_quantile": 0.85,
        "uncertainty_warmup_steps": 200,
        "uncertainty_window_size": 2000,
        "uncertainty_hysteresis": (0.80, 1.20),
        "mask_x_factor": 0.3, "mask_z_factor": 0.1,
        "hard_fallback_weights_3branch": (0.2, 0.7, 0.1),
        "hard_fallback_weights_2branch": (0.2, 0.8),
    },
    "IMAGE": {
        "uncertainty_quantile": 0.90,
        "uncertainty_warmup_steps": 500,
        "uncertainty_window_size": 3000,
        "uncertainty_hysteresis": (0.75, 1.25),
        "mask_x_factor": 0.2, "mask_z_factor": 0.05,
        "ema_blend_eval": 0.1,
        "hard_fallback_weights_3branch": (0.1, 0.8, 0.1),
        "hard_fallback_weights_2branch": (0.1, 0.9),
    },
    "MIXED": {
        "uncertainty_quantile": 0.85,
        "uncertainty_warmup_steps": 300,
        "uncertainty_window_size": 2500,
        "uncertainty_hysteresis": (0.78, 1.22),
        "mask_x_factor": 0.25, "mask_z_factor": 0.1,
        "mode": "attention",
        "hard_fallback_weights_3branch": (0.2, 0.7, 0.1),
        "hard_fallback_weights_2branch": (0.2, 0.8),
    },
}

_TEMPORAL_ROUTER_ADJUSTMENTS: Dict[str, Dict[str, Any]] = {
    "high": {"uncertainty_alpha": 0.6, "anneal_steps": 80000},
    "static": {"uncertainty_alpha": 0.4, "anneal_steps": 30000},
    "low": {"uncertainty_alpha": 0.4, "anneal_steps": 30000},
}


def make_router_overrides_for_profile(profile: EnvProfile) -> Dict[str, Any]:
    overrides = dict(_ROUTER_PRESETS.get(profile.env_category, _ROUTER_PRESETS["COMPLEX_VECTOR"]))
    temporal_key = "high" if profile.is_pomdp else profile.temporal_dependency
    adj = _TEMPORAL_ROUTER_ADJUSTMENTS.get(temporal_key)
    if adj:
        overrides.update(adj)
    return overrides


_PERCEPTOR_OVERRIDE_WHITELIST = frozenset({
    "vitals_dim", "modality_embed_dim", "fusion_type", "fusion_hidden_dims",
    "vision_backbone", "vision_backbone_min_resolution",
    "vision_backbone_patch_size", "vision_backbone_min_patches",
    "image_depth", "image_kernels", "vector_hidden_dims",
    "use_layernorm", "activation", "dropout",
})


def make_perceptor_overrides_for_profile(profile: EnvProfile) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    cat = profile.env_category

    if cat == "SIMPLE_VECTOR":
        overrides.update(
            vitals_dim=max(64, min(profile.obs_dim * 2, 128)),
            modality_embed_dim=128,
            vector_hidden_dims=[128, 128],
            fusion_type="concat",
        )
    elif cat == "COMPLEX_VECTOR":
        overrides.update(
            vitals_dim=max(128, min(profile.obs_dim, 256)),
            modality_embed_dim=256,
            vector_hidden_dims=[256, 256, 256],
            fusion_type="concat",
        )
    elif cat == "IMAGE":
        overrides.update(vitals_dim=512, modality_embed_dim=512, fusion_type="concat")
        obs_shape = profile.obs_shape
        if len(obs_shape) >= 2:
            h, w = obs_shape[-2], obs_shape[-1]
            if h >= 84 and w >= 84:
                overrides.update(vision_backbone="vit", vision_backbone_patch_size=14)
            else:
                overrides.update(vision_backbone="cnn", image_depth=48)
    else:  # MIXED
        overrides.update(
            vitals_dim=384, modality_embed_dim=384,
            fusion_type="attention", vector_hidden_dims=[256, 256],
        )

    if profile.obs_modality == "image" and cat != "IMAGE":
        overrides.update(vision_backbone="cnn", image_depth=32)

    return overrides


_ENV_CATEGORY_MODEL_DEFAULTS: Dict[str, Dict[str, int]] = {
    "SIMPLE_VECTOR": dict(
        rssm_deter_dim=96,
        rssm_hidden_dim=96,
        rssm_num_distributions=16,
        rssm_num_classes=16,
    ),
    "COMPLEX_VECTOR": dict(
        rssm_deter_dim=1024,
        rssm_hidden_dim=1024,
        rssm_num_distributions=32,
        rssm_num_classes=32,
    ),
    "IMAGE": dict(
        rssm_deter_dim=2048,
        rssm_hidden_dim=1024,
        rssm_num_distributions=32,
        rssm_num_classes=32,
    ),
}


def _make_training_config_policy_defaults_for_profile(
    profile: EnvProfile,
) -> ConfigPolicyOverrides:
    """Return training-policy defaults derived from the environment profile."""
    if profile.env_category == "SIMPLE_VECTOR":
        sequence_length = 32
        imagination_horizon = 15
        curiosity_enabled = True
        mastery_enabled = False
    elif profile.env_category in {"COMPLEX_VECTOR", "IMAGE"}:
        sequence_length = 64
        imagination_horizon = 15
        curiosity_enabled = True
        mastery_enabled = True
    else:
        sequence_length = None
        imagination_horizon = None
        curiosity_enabled = True
        mastery_enabled = True

    if (
        profile.reward_variance_level == "low"
        and profile.temporal_dependency == "high"
    ):
        curiosity_enabled = True

    return ConfigPolicyOverrides(
        sequence_length=sequence_length,
        imagination_horizon=imagination_horizon,
        curiosity_enabled=curiosity_enabled,
        mastery_enabled=mastery_enabled,
    )


def make_config_policy_overrides_for_env(
    env_or_name: Union[str, Any],
    include_env_profile: bool = True,
) -> ConfigPolicyOverrides:
    """Build the canonical ConfigPolicy override object for an environment."""
    profile = extract_env_profile(env_or_name)
    model_defaults = _ENV_CATEGORY_MODEL_DEFAULTS.get(profile.env_category, {})
    training_policy = _make_training_config_policy_defaults_for_profile(profile)
    return coerce_config_policy_overrides(
        ConfigPolicyOverrides(
            sequence_length=training_policy.sequence_length,
            imagination_horizon=training_policy.imagination_horizon,
            curiosity_enabled=training_policy.curiosity_enabled,
            mastery_enabled=training_policy.mastery_enabled,
            router_overrides=make_router_overrides_for_profile(profile),
            perceptor_overrides=make_perceptor_overrides_for_profile(profile),
            policy_env_profile=profile if include_env_profile else None,
            custom_overrides={
                "rssm_deter_dim": int(model_defaults.get("rssm_deter_dim", 96)),
                "rssm_hidden_dim": int(model_defaults.get("rssm_hidden_dim", 96)),
                "rssm_num_distributions": int(
                    model_defaults.get("rssm_num_distributions", 16)
                ),
                "rssm_num_classes": int(model_defaults.get("rssm_num_classes", 16)),
            },
        )
    )


# ─────────────────────────────────────────────────────────────────
# [OPT-5] 配置验证：加安全防护
# ─────────────────────────────────────────────────────────────────

def _is_optional_type(t: Any) -> bool:
    origin = get_origin(t)
    if origin is Union:
        return type(None) in get_args(t)
    return False


_NON_NEGATIVE_KEYWORDS = frozenset({
    "dim", "size", "steps", "length", "depth",
    "layers", "num", "batch", "horizon", "hidden",
    "width", "capacity", "count",
})


def _validate_config_instance(obj: Any) -> None:
    for f in fields(obj):
        val = getattr(obj, f.name)
        if val is None:
            if not _is_optional_type(f.type) and f.default is not None:
                raise ValueError(f"{obj.__class__.__name__}.{f.name} is None")
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if val < 0 and any(k in f.name for k in _NON_NEGATIVE_KEYWORDS):
                raise ValueError(f"{obj.__class__.__name__}.{f.name} must be non-negative")


def _patch_config_post_init() -> None:
    """[OPT-5] 对所有 *Config dataclass 添加字段验证"""
    patched = set()
    for name, obj in list(globals().items()):
        if (
            not isinstance(obj, type)
            or not name.endswith("Config")
            or not is_dataclass(obj)
            or id(obj) in patched  # 防止重复补丁
        ):
            continue
        patched.add(id(obj))
        original = getattr(obj, "__post_init__", None)

        def _make_post_init(orig):
            def __post_init__(self):
                if orig is not None:
                    try:
                        orig(self)
                    except TypeError:
                        pass  # 某些 __post_init__ 签名不兼容
                _validate_config_instance(self)
            return __post_init__

        obj.__post_init__ = _make_post_init(original)


_patch_config_post_init()
