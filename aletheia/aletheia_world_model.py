from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import gc
import math
import inspect
import logging
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    Any, Callable, Dict, List, Literal, Optional,
    NamedTuple, Protocol, Set, Tuple, Type, Union,
    TYPE_CHECKING,
)

from torch import Tensor
from .aletheia_foundation import handle_time_dim, handle_nd_time_dim
from .aletheia_config import LossWeightsConfig


# ======================================================================
# 全局工具函数
# ======================================================================


# ======================================================================
# MERGED FROM: perceptor/types.py
# ======================================================================

class ModalityType(Enum):
    """模态类型枚举"""
    VECTOR = auto()
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()
    TEXT = auto()
    POINTCLOUD = auto()
    TIMESERIES = auto()
    SCALAR = auto()
    DISCRETE = auto()

    _STRING_MAP = None  # type: ignore

    @classmethod
    def _get_string_map(cls) -> Dict[str, "ModalityType"]:
        if cls._STRING_MAP is None:
            cls._STRING_MAP = {m.name.lower(): m for m in cls if m.name != "_STRING_MAP"}
        return cls._STRING_MAP

    @classmethod
    def from_string(cls, s: str) -> "ModalityType":
        return cls._get_string_map().get(s.lower(), cls.VECTOR)

    def to_string(self) -> str:
        return self.name.lower()


@dataclass
class ModalitySpec:
    """
    模态规格描述

    封装单个模态的完整信息包括类型形状嵌入维度等
    """
    name: str
    modality_type: ModalityType
    input_shape: Tuple[int, ...]
    embed_dim: int = 256
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_dim(self) -> int:
        if len(self.input_shape) == 1:
            return self.input_shape[0]
        return int(np.prod(self.input_shape))

    @property
    def is_image(self) -> bool:
        return self.modality_type == ModalityType.IMAGE

    @property
    def is_vector(self) -> bool:
        return self.modality_type == ModalityType.VECTOR

    @property
    def is_sequential(self) -> bool:
        return self.modality_type in (
            ModalityType.VIDEO,
            ModalityType.AUDIO,
            ModalityType.TIMESERIES,
            ModalityType.TEXT,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "modality_type": self.modality_type.to_string(),
            "input_shape": self.input_shape,
            "embed_dim": self.embed_dim,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModalitySpec":
        return cls(
            name=d["name"],
            modality_type=ModalityType.from_string(d["modality_type"]),
            input_shape=tuple(d["input_shape"]),
            embed_dim=d.get("embed_dim", 256),
            extra_params=d.get("extra_params", {}),
        )


# ======================================================================
# MERGED FROM: perceptor/registry.py
# ======================================================================

ENCODER_REGISTRY: Dict[str, Type["BaseEncoder"]] = {}
DECODER_REGISTRY: Dict[str, Type["BaseDecoder"]] = {}
_REGISTRIES_READY: bool = False


def create_encoder(modality_type: str, output_dim: int, **kwargs) -> "BaseEncoder":
    _ensure_registries()
    if modality_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder modality type: {modality_type}. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[modality_type](output_dim=output_dim, **kwargs)


def create_decoder(modality_type: str, input_dim: int, **kwargs) -> "BaseDecoder":
    _ensure_registries()
    if modality_type not in DECODER_REGISTRY:
        raise ValueError(
            f"Unknown decoder modality type: {modality_type}. "
            f"Available: {list(DECODER_REGISTRY.keys())}"
        )
    return DECODER_REGISTRY[modality_type](input_dim=input_dim, **kwargs)


def _init_registries():
    """初始化注册表所有类在同一文件中直接引用"""
    global _REGISTRIES_READY
    if _REGISTRIES_READY:
        return

    ENCODER_REGISTRY.update({
        "vector": VectorEncoder,
        "image": ImageEncoder,
        "video": VideoEncoder,
        "audio": AudioEncoder,
        "text": TextEncoder,
        "pointcloud": PointCloudEncoder,
        "timeseries": TimeSeriesEncoder,
        "scalar": ScalarEncoder,
        "discrete": DiscreteEncoder,
    })

    DECODER_REGISTRY.update({
        "vector": VectorDecoder,
        "image": ImageDecoder,
        "video": VideoDecoder,
        "audio": AudioDecoder,
        "text": TextDecoder,
        "pointcloud": PointCloudDecoder,
        "timeseries": TimeSeriesDecoder,
        "scalar": ScalarDecoder,
        "discrete": DiscreteDecoder,
    })
    _REGISTRIES_READY = True


def _ensure_registries():
    if not _REGISTRIES_READY:
        _init_registries()


# ======================================================================
# MERGED FROM: perceptor/base.py
# ======================================================================

class BaseEncoder(nn.Module):
    """编码器基类"""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "output_dim": self.output_dim,
        }


class BaseDecoder(nn.Module):
    """解码器基类"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "input_dim": self.input_dim,
        }


# ======================================================================
# MERGED FROM: perceptor/codecs/scalar.py
# ======================================================================

class ScalarEncoder(BaseEncoder):
    """
    标量编码器Fourier 特征

    将标量值映射到高维 Fourier 特征空间
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        n_frequencies: int = 16,
    ):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.n_frequencies = n_frequencies

        fourier_dim = input_dim * (1 + 2 * n_frequencies)
        self.net = build_mlp(
            in_dim=fourier_dim,
            out_dim=output_dim,
            hidden_dims=(128,),
            activation="silu",
            final_gain=1.0,
            init="final_gain_only",
        )

        # 预注册频率缓冲区避免每次 forward 重新生成
        freqs = (2.0 ** torch.linspace(0, n_frequencies - 1, n_frequencies)) * math.pi
        self.register_buffer("_freqs", freqs, persistent=False)

    @handle_time_dim
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # x: (B, D), _freqs: (F,)
        x_expanded = x.unsqueeze(-1) * self._freqs  # (B, D, F)

        encoded = torch.cat([
            x,                                      # (B, D)
            torch.sin(x_expanded).flatten(-2),       # (B, D*F)
            torch.cos(x_expanded).flatten(-2),       # (B, D*F)
        ], dim=-1)

        return self.net(encoded)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "input_dim": self.input_dim,
            "n_frequencies": self.n_frequencies,
            "fourier_dim": self.input_dim * (1 + 2 * self.n_frequencies),
        })
        return info


class ScalarDecoder(BaseDecoder):
    """标量解码器"""

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__(input_dim)
        self.output_dim = output_dim
        self.net = build_mlp(
            in_dim=input_dim,
            out_dim=output_dim,
            hidden_dims=(64,),
            activation="silu",
            norm="none",
            final_gain=1.0,
            init="final_gain_only",
        )

    @handle_time_dim
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["output_dim"] = self.output_dim
        return info


# ======================================================================
# MERGED FROM: perceptor/codecs/discrete.py
# ======================================================================

class DiscreteEncoder(BaseEncoder):
    """
    离散类别编码器

    使用 Embedding 层将离散类别映射到连续空间
    """

    def __init__(self, output_dim: int, num_classes: int):
        super().__init__(output_dim)
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 类别索引 (B,) 或 (B, 1) 或 (B, T)
        Returns:
            编码特征 (B, output_dim) 或 (B, T, output_dim)
        """
        x = x.long()

        # (B, 1) -> (B,): squeeze 掉多余的维度
        if x.dim() == 2 and x.shape[1] == 1:
            x = x.squeeze(-1)

        return self.norm(self.embedding(x))

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["num_classes"] = self.num_classes
        return info


class DiscreteDecoder(BaseDecoder):
    """离散类别解码器输出 logits"""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(input_dim)
        self.num_classes = num_classes
        self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["num_classes"] = self.num_classes
        return info


# ======================================================================
# MERGED FROM: perceptor/codecs/vector.py
# ======================================================================

class VectorEncoder(BaseEncoder):
    """向量编码器"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__(output_dim)
        self.input_dim = input_dim
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.net = build_mlp(
            in_dim=input_dim,
            out_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            activation="silu",
            final_gain=1.0,
            init="final_gain_only",
        )

    @handle_nd_time_dim
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["input_dim"] = self.input_dim
        return info


class VectorDecoder(BaseDecoder):
    """向量解码器"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__(input_dim)
        self.output_dim = output_dim
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.net = build_mlp(
            in_dim=input_dim,
            out_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            activation="silu",
            norm="none",
            final_gain=1.0,
            init="final_gain_only",
        )

    @handle_nd_time_dim
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ======================================================================
# MERGED FROM: perceptor/codecs/vit.py
# ======================================================================

def _best_num_heads(embed_dim: int, preferred: int) -> int:
    if preferred <= 0:
        return 1
    if embed_dim % preferred == 0:
        return preferred
    for h in range(min(preferred, embed_dim), 0, -1):
        if embed_dim % h == 0:
            return h
    return 1


def _get_1d_sincos_pos_embed(embed_dim: int, positions: Tensor) -> Tensor:
    if embed_dim <= 0:
        raise ValueError("embed_dim must be positive")

    base_dim = embed_dim if embed_dim % 2 == 0 else embed_dim - 1
    half = base_dim // 2
    device = positions.device
    dtype = positions.dtype

    if half == 0:
        return torch.zeros((positions.numel(), embed_dim), device=device, dtype=dtype)

    omega = torch.arange(half, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / half))
    out = positions.reshape(-1, 1) * omega.reshape(1, -1)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    if base_dim != embed_dim:
        emb = F.pad(emb, (0, 1), mode="constant", value=0.0)
    return emb


def _get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if embed_dim <= 0:
        raise ValueError("embed_dim must be positive")

    gh, gw = int(grid_size[0]), int(grid_size[1])
    if gh <= 0 or gw <= 0:
        return torch.zeros((0, embed_dim), device=device, dtype=dtype)

    base_dim = embed_dim if embed_dim % 2 == 0 else embed_dim - 1
    half = base_dim // 2

    y = torch.arange(gh, device=device, dtype=dtype)
    x = torch.arange(gw, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    emb_y = _get_1d_sincos_pos_embed(half, yy)
    emb_x = _get_1d_sincos_pos_embed(half, xx)
    emb = torch.cat([emb_y, emb_x], dim=1)

    if base_dim != embed_dim:
        emb = F.pad(emb, (0, 1), mode="constant", value=0.0)
    return emb


class _TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        num_heads = _best_num_heads(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = max(1, int(embed_dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBackbone(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int = 3,
        embed_dim: int = 256,
        patch_size: int = 16,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(
            in_ch, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )
        self.blocks = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"ViTBackbone expects 4D input, got shape={tuple(x.shape)}")

        _, _, h, w = x.shape
        ps = self.patch_size
        pad_h = (ps - (h % ps)) % ps
        pad_w = (ps - (w % ps)) % ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        x = self.patch_embed(x)
        gh, gw = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)

        pos = _get_2d_sincos_pos_embed(self.embed_dim, (gh, gw), device=x.device, dtype=x.dtype)
        x = x + pos.unsqueeze(0)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x.mean(dim=1)


# ======================================================================
# MERGED FROM: perceptor/codecs/image.py
# ======================================================================

class ImageEncoder(BaseEncoder):
    """图像编码器支持 CNN 和 ViT backbone"""

    def __init__(
        self,
        output_dim: int,
        channels: int = 3,
        depth: int = 32,
        kernels: Optional[List[int]] = None,
        backbone: str = "cnn",
        vit_patch_size: int = 16,
        vit_depth: int = 4,
        vit_num_heads: int = 4,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.0,
        vit_embed_dim: Optional[int] = None,
    ):
        super().__init__(output_dim)

        if kernels is None:
            kernels = [4, 4, 4, 4]

        self.channels = channels
        self.backbone = backbone

        if backbone == "vit":
            embed_dim = output_dim if vit_embed_dim is None else vit_embed_dim
            self.vit = ViTBackbone(
                in_ch=channels,
                embed_dim=embed_dim,
                patch_size=vit_patch_size,
                depth=vit_depth,
                num_heads=vit_num_heads,
                mlp_ratio=vit_mlp_ratio,
                dropout=vit_dropout,
            )
            self.head = (
                nn.Identity() if embed_dim == output_dim
                else nn.Sequential(
                    nn.Linear(embed_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.SiLU(),
                )
            )
            self.cnn = None
            self.pool = None
            self.flatten = None
            self.mlp = None
        else:
            layers: List[nn.Module] = []
            in_ch = channels
            for i, kernel in enumerate(kernels):
                out_ch = depth * (2 ** i)
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, kernel, stride=2, padding=kernel // 2 - 1),
                    nn.GroupNorm(min(32, out_ch), out_ch),
                    nn.SiLU(),
                ])
                in_ch = out_ch

            self.cnn = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.flatten = nn.Flatten()

            cnn_out_ch = depth * (2 ** (len(kernels) - 1))
            flatten_dim = cnn_out_ch * 4 * 4

            self.mlp = nn.Sequential(
                nn.Linear(flatten_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.SiLU(),
            )
            self.vit = None
            self.head = None

    def forward(self, x: Tensor) -> Tensor:
        has_time_dim = False
        if x.dim() == 5:
            B, T = x.shape[:2]
            has_time_dim = True
            x = x.reshape(B * T, *x.shape[2:])

        if x.dim() == 3:
            x = x.unsqueeze(0)

        # 自动推断 channel-last -> channel-first
        if x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
            x = x.permute(0, 3, 1, 2)

        if self.backbone == "vit":
            x = self.vit(x)
            x = self.head(x)
        else:
            x = self.cnn(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.mlp(x)

        if has_time_dim:
            x = x.reshape(B, T, -1)

        return x


class ImageDecoder(BaseDecoder):
    """图像解码器DreamerV3 风格"""

    def __init__(
        self,
        input_dim: int,
        channels: int = 3,
        output_size: Tuple[int, int] = (64, 64),
        depth: int = 32,
    ):
        super().__init__(input_dim)

        self.channels = channels
        self.output_size = output_size

        self.init_h = max(1, (output_size[0] + 15) // 16)
        self.init_w = max(1, (output_size[1] + 15) // 16)
        init_ch = depth * 8

        self.fc = nn.Sequential(
            nn.Linear(input_dim, init_ch * self.init_h * self.init_w),
            nn.LayerNorm(init_ch * self.init_h * self.init_w),
            nn.SiLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(init_ch, depth * 4, 4, stride=2, padding=1),
            nn.GroupNorm(32, depth * 4),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, stride=2, padding=1),
            nn.GroupNorm(32, depth * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 2, depth, 4, stride=2, padding=1),
            nn.GroupNorm(32, depth),
            nn.SiLU(),
            nn.ConvTranspose2d(depth, channels, 4, stride=2, padding=1),
        )

        self._init_ch = init_ch

    def forward(self, x: Tensor) -> Tensor:
        has_time_dim = False
        if x.dim() == 3:
            B, T = x.shape[:2]
            has_time_dim = True
            x = x.reshape(B * T, -1)

        x = self.fc(x)
        x = x.view(-1, self._init_ch, self.init_h, self.init_w)
        x = self.deconv(x)

        # FIX BUG-1: F.interpolate 现在可以正常使用顶部已导入 F
        if x.shape[-2:] != (self.output_size[0], self.output_size[1]):
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)

        if has_time_dim:
            x = x.reshape(B, T, *x.shape[1:])

        return x


# ======================================================================
# MERGED FROM: perceptor/codecs/video.py
# ======================================================================

class VideoEncoder(BaseEncoder):
    """
    视频编码器

    支持 3D CNNuse_3d_conv=True和 2D CNN + Temporal GRUuse_3d_conv=False
    输入布局(B, T, C, H, W) 或 (B, C, T, H, W)
    """

    def __init__(
        self,
        output_dim: int,
        channels: int = 3,
        num_frames: int = 4,
        use_3d_conv: bool = False,
        depth: int = 32,
        frame_backbone: str = "cnn",
        frame_backbone_kwargs: Optional[dict] = None,
    ):
        super().__init__(output_dim)

        self.channels = channels
        self.num_frames = num_frames
        self.use_3d_conv = use_3d_conv

        if use_3d_conv:
            self.conv = nn.Sequential(
                nn.Conv3d(channels, depth, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
                nn.GroupNorm(8, depth),
                nn.SiLU(),
                nn.Conv3d(depth, depth * 2, (3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
                nn.GroupNorm(16, depth * 2),
                nn.SiLU(),
                nn.Conv3d(depth * 2, depth * 4, (3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
                nn.GroupNorm(32, depth * 4),
                nn.SiLU(),
            )
            self.pool = nn.AdaptiveAvgPool3d((1, 4, 4))
            flatten_dim = depth * 4 * 16
            self.frame_encoder = None
            self.temporal = None
        else:
            if frame_backbone_kwargs is None:
                frame_backbone_kwargs = {}
            self.frame_encoder = ImageEncoder(
                output_dim, channels, depth,
                backbone=frame_backbone,
                **frame_backbone_kwargs,
            )
            self.temporal = nn.GRU(output_dim, output_dim, batch_first=True)
            flatten_dim = output_dim
            self.conv = None
            self.pool = None

        self.out = nn.Sequential(
            nn.Linear(flatten_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def _normalize_layout_3d(self, x: Tensor) -> Tensor:
        """规范为 (B, C, T, H, W) 用于 3D conv"""
        if x.dim() != 5:
            raise ValueError(f"VideoEncoder (3D) expects 5D input, got shape={tuple(x.shape)}")
        if x.shape[1] not in (1, 3, 4) and x.shape[2] in (1, 3, 4):
            x = x.permute(0, 2, 1, 3, 4)
        elif x.shape[1] not in (1, 3, 4):
            raise ValueError(f"Ambiguous video layout for 3D conv: shape={tuple(x.shape)}")
        return x

    def _normalize_layout_2d(self, x: Tensor) -> Tensor:
        """规范为 (B, T, C, H, W) 用于 2D + temporal"""
        if x.dim() != 5:
            raise ValueError(f"VideoEncoder expects 5D input, got shape={tuple(x.shape)}")
        if x.shape[1] in (1, 3, 4) and x.shape[2] != x.shape[1]:
            # (B, C, T, H, W) -> (B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        elif x.shape[1] not in (1, 3, 4) and x.shape[2] in (1, 3, 4):
            pass  # already (B, T, C, H, W)
        else:
            raise ValueError(f"Ambiguous video layout: shape={tuple(x.shape)}")
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.use_3d_conv:
            x = self._normalize_layout_3d(x)
            x = self.conv(x)
            x = self.pool(x)
            x = x.flatten(1)
        else:
            x = self._normalize_layout_2d(x)
            B, T, C, H, W = x.shape
            # FIX PERF: 批量编码所有帧
            x_flat = x.reshape(B * T, C, H, W)
            frames = self.frame_encoder(x_flat)  # (B*T, output_dim)
            x = frames.reshape(B, T, -1)
            x, _ = self.temporal(x)
            x = x[:, -1]

        return self.out(x)


class VideoDecoder(BaseDecoder):
    """视频解码器"""

    def __init__(
        self,
        input_dim: int,
        channels: int = 3,
        num_frames: int = 4,
        output_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__(input_dim)

        self.num_frames = num_frames
        self.channels = channels
        self.output_size = output_size

        self.temporal = nn.GRU(input_dim, input_dim, batch_first=True)
        self.frame_decoder = ImageDecoder(input_dim, channels, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 潜变量 (B, input_dim)
        Returns:
            视频张量 (B, T, C, H, W)
        """
        # FIX PERF-1: 批量解码所有帧避免 Python 循环
        x = x.unsqueeze(1).expand(-1, self.num_frames, -1)
        x, _ = self.temporal(x)

        B, T = x.shape[:2]
        x_flat = x.reshape(B * T, -1)
        frames_flat = self.frame_decoder(x_flat)
        return frames_flat.reshape(B, T, *frames_flat.shape[1:])

# ======================================================================
# MERGED FROM: perceptor/codecs/audio.py
# ======================================================================

class AudioEncoder(BaseEncoder):
    """
    音频编码器

    支持原始波形1D Conv和 Mel 频谱图2D Conv
    """

    def __init__(
        self,
        output_dim: int,
        input_type: str = "waveform",
        sample_rate: int = 16000,
        n_mels: int = 80,
        depth: int = 32,
    ):
        super().__init__(output_dim)

        self.input_type = input_type
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        if input_type == "waveform":
            self.conv = nn.Sequential(
                nn.Conv1d(1, depth, 10, stride=5, padding=2),
                nn.GroupNorm(8, depth),
                nn.SiLU(),
                nn.Conv1d(depth, depth * 2, 8, stride=4, padding=2),
                nn.GroupNorm(16, depth * 2),
                nn.SiLU(),
                nn.Conv1d(depth * 2, depth * 4, 4, stride=2, padding=1),
                nn.GroupNorm(32, depth * 4),
                nn.SiLU(),
                nn.Conv1d(depth * 4, depth * 8, 4, stride=2, padding=1),
                nn.GroupNorm(32, depth * 8),
                nn.SiLU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(8)
            flatten_dim = depth * 8 * 8
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(1, depth, 4, stride=2, padding=1),
                nn.GroupNorm(8, depth),
                nn.SiLU(),
                nn.Conv2d(depth, depth * 2, 4, stride=2, padding=1),
                nn.GroupNorm(16, depth * 2),
                nn.SiLU(),
                nn.Conv2d(depth * 2, depth * 4, 4, stride=2, padding=1),
                nn.GroupNorm(32, depth * 4),
                nn.SiLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            flatten_dim = depth * 4 * 16

        self.out = nn.Sequential(
            nn.Linear(flatten_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.input_type == "waveform":
            if x.dim() == 2:
                x = x.unsqueeze(1)
        else:
            if x.dim() == 3:
                x = x.unsqueeze(1)

        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.out(x)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "input_type": self.input_type,
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
        })
        return info


class AudioDecoder(BaseDecoder):
    """音频解码器输出频谱图"""

    def __init__(
        self,
        input_dim: int,
        output_length: int = 16000,
        n_mels: int = 80,
    ):
        super().__init__(input_dim)

        self.n_mels = n_mels
        self.output_frames = output_length // 160  # 假设 hop_length=160

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256 * 4 * 4),
            nn.LayerNorm(256 * 4 * 4),
            nn.SiLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
        )

        self.resize = nn.AdaptiveAvgPool2d((n_mels, self.output_frames))

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        x = self.deconv(x)
        x = self.resize(x)
        return x.squeeze(1)


# ======================================================================
# MERGED FROM: perceptor/codecs/text.py
# ======================================================================

class TextEncoder(BaseEncoder):
    """文本编码器Embedding + Transformer"""

    def __init__(
        self,
        output_dim: int,
        vocab_size: int = 32000,
        embed_dim: int = 256,
        max_length: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__(output_dim)

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.out = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, L = x.shape

        L = min(L, self.max_length)
        x = x[:, :L]

        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(x) + self.pos_embed(positions)

        src_key_padding_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask[:, :L]
            src_key_padding_mask = ~attention_mask.bool()

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.out(x)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "embed_dim": self.embed_dim,
        })
        return info


class TextDecoder(BaseDecoder):
    """文本解码器"""

    def __init__(
        self,
        input_dim: int,
        vocab_size: int = 32000,
        max_length: int = 128,
        embed_dim: int = 256,
    ):
        super().__init__(input_dim)

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.fc = nn.Linear(input_dim, embed_dim)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_length, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.out = nn.Linear(embed_dim, vocab_size)

        # FIX PERF-2: 预创建最大尺寸的因果 maskforward 时切片
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.ones(max_length, max_length, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        tgt_tokens: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        start_tokens: Optional[Tensor] = None,
        max_length: Optional[int] = None,
    ) -> Tensor:
        B = x.shape[0]
        device = x.device

        latent = self.fc(x).unsqueeze(1)  # (B, 1, E)

        if tgt_tokens is not None:
            L = min(tgt_tokens.shape[1], self.max_length)
            tgt_tokens = tgt_tokens[:, :L]

            positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            tgt_embed = self.token_embed(tgt_tokens) + self.pos_embed(positions)

            tgt_mask = self._causal_mask[:L, :L]

            tgt_key_padding_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask[:, :L]
                tgt_key_padding_mask = ~attention_mask.bool()

            hidden = self.transformer(
                tgt=tgt_embed,
                memory=latent,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            return self.out(hidden)

        return self.generate_logits(x, start_tokens=start_tokens, max_length=max_length)

    def generate_logits(
        self,
        x: Tensor,
        start_tokens: Optional[Tensor] = None,
        max_length: Optional[int] = None,
    ) -> Tensor:
        """自回归生成 logits"""
        B = x.shape[0]
        device = x.device
        latent = self.fc(x).unsqueeze(1)  # (B, 1, E)
        L = max_length or self.max_length

        if start_tokens is None:
            tokens = torch.zeros((B, 1), dtype=torch.long, device=device)
        else:
            tokens = start_tokens.to(device=device, dtype=torch.long)
            if tokens.dim() != 2 or tokens.shape[0] != B:
                raise ValueError(f"start_tokens must be (B, L0), got shape={tuple(tokens.shape)}")

        logits_list: List[Tensor] = []

        for _ in range(L):
            curr_len = tokens.shape[1]
            if curr_len > self.max_length:
                break

            positions = torch.arange(curr_len, device=device).unsqueeze(0).expand(B, -1)
            tgt_embed = self.token_embed(tokens) + self.pos_embed(positions)

            # FIX PERF-2: 从预创建的 mask 切片而非每步重建
            tgt_mask = self._causal_mask[:curr_len, :curr_len]

            hidden = self.transformer(tgt=tgt_embed, memory=latent, tgt_mask=tgt_mask)
            logits = self.out(hidden)
            next_logits = logits[:, -1, :]
            logits_list.append(next_logits.unsqueeze(1))

            if len(logits_list) >= L:
                break

            next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tokens], dim=1)

        return torch.cat(logits_list, dim=1)


# ======================================================================
# MERGED FROM: perceptor/codecs/pointcloud.py
# ======================================================================

class PointCloudEncoder(BaseEncoder):
    """点云编码器PointNet 风格"""

    def __init__(
        self,
        output_dim: int,
        point_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__(output_dim)
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim

        self.point_net = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.global_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.point_net(x)       # (B, N, hidden_dim)
        x = x.max(dim=1)[0]         # (B, hidden_dim)
        return self.global_net(x)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({"point_dim": self.point_dim, "hidden_dim": self.hidden_dim})
        return info


class PointCloudDecoder(BaseDecoder):
    """点云解码器"""

    def __init__(
        self,
        input_dim: int,
        num_points: int = 1024,
        point_dim: int = 3,
    ):
        super().__init__(input_dim)
        self.num_points = num_points
        self.point_dim = point_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, num_points * point_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x.view(-1, self.num_points, self.point_dim)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({"num_points": self.num_points, "point_dim": self.point_dim})
        return info


# ======================================================================
# MERGED FROM: perceptor/codecs/timeseries.py
# ======================================================================

class TimeSeriesEncoder(BaseEncoder):
    """时间序列编码器1D CNN + GRU"""

    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        hidden_dim: int = 128,
    ):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv1d(64, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
        )

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def _to_channel_first(self, x: Tensor) -> Tensor:
        """
        规范化输入为 (B, D, T) 格式

        FIX LOGIC-2: 当两个维度都等于 input_dim 时
        优先假定输入为 (B, T, D) 格式更常见的约定
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T) -> (B, T, 1)
        if x.dim() != 3:
            raise ValueError(f"TimeSeriesEncoder expects 2D/3D input, got shape={tuple(x.shape)}")

        # 如果最后一维是 input_dim假定 (B, T, D)
        if x.shape[-1] == self.input_dim:
            return x.permute(0, 2, 1)
        # 否则如果第二维是 input_dim假定 (B, D, T)
        if x.shape[1] == self.input_dim:
            return x
        raise ValueError(
            f"TimeSeriesEncoder cannot infer layout for shape={tuple(x.shape)} "
            f"with input_dim={self.input_dim}"
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._to_channel_first(x)      # (B, D, T)
        x = self.conv(x)                    # (B, hidden_dim, T')
        x = x.permute(0, 2, 1)             # (B, T', hidden_dim)
        x, _ = self.gru(x)
        x = x[:, -1]                        # (B, hidden_dim)
        return self.out(x)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({"input_dim": self.input_dim, "hidden_dim": self.hidden_dim})
        return info


class TimeSeriesDecoder(BaseDecoder):
    """时间序列解码器"""

    def __init__(
        self,
        input_dim: int,
        output_length: int = 100,
        output_dim: int = 1,
    ):
        super().__init__(input_dim)
        self.output_length = output_length
        self.output_dim = output_dim

        self.gru = nn.GRU(input_dim, 128, batch_first=True)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1).expand(-1, self.output_length, -1)
        x, _ = self.gru(x)
        return self.out(x)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({"output_length": self.output_length, "output_dim": self.output_dim})
        return info


# ======================================================================
# MERGED FROM: perceptor/detector.py
# ======================================================================

class ModalityDetector:
    """根据 obs_space 或 Tensor 自动推断模态类型"""

    @staticmethod
    def _parse_image_shape(input_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
        if len(input_shape) != 3:
            return 1, 0, 0
        if input_shape[0] in (1, 3, 4):
            return int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
        if input_shape[2] in (1, 3, 4):
            return int(input_shape[2]), int(input_shape[0]), int(input_shape[1])
        return 1, int(input_shape[0]), int(input_shape[1])

    @staticmethod
    def suggest_image_backbone(
        input_shape: Tuple[int, ...],
        *,
        min_resolution: int = 128,
        patch_size: int = 16,
        min_patches: int = 16,
    ) -> str:
        c, h, w = ModalityDetector._parse_image_shape(input_shape)
        if h <= 0 or w <= 0 or h < min_resolution or w < min_resolution or c < 3:
            return "cnn"
        gh = (h + patch_size - 1) // patch_size
        gw = (w + patch_size - 1) // patch_size
        if gh * gw < min_patches:
            return "cnn"
        return "vit"

    @staticmethod
    def detect_from_space(space: Any) -> Tuple[ModalityType, Dict[str, Any]]:
        # Discrete space
        if hasattr(space, 'n') and not hasattr(space, 'shape'):
            return ModalityType.DISCRETE, {"num_classes": space.n}

        if hasattr(space, 'shape'):
            shape = space.shape

            if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1):
                return ModalityType.SCALAR, {"input_dim": 1}

            if len(shape) == 1:
                return ModalityType.VECTOR, {"input_dim": int(shape[0])}

            if len(shape) == 2:
                h, w = shape
                if h > 16 and w > 16:
                    return ModalityType.IMAGE, {
                        "channels": 1,
                        "output_size": (int(h), int(w)),
                    }
                return ModalityType.TIMESERIES, {
                    "input_dim": int(shape[1]),
                    "output_length": int(shape[0]),
                }

            if len(shape) == 3:
                if shape[0] in (1, 3, 4):
                    return ModalityType.IMAGE, {
                        "channels": int(shape[0]),
                        "output_size": (int(shape[1]), int(shape[2])),
                    }
                elif shape[2] in (1, 3, 4):
                    return ModalityType.IMAGE, {
                        "channels": int(shape[2]),
                        "output_size": (int(shape[0]), int(shape[1])),
                    }
                else:
                    return ModalityType.VIDEO, {
                        "channels": 1,
                        "num_frames": int(shape[0]),
                        "output_size": (int(shape[1]), int(shape[2])),
                    }

            if len(shape) == 4:
                if shape[1] in (1, 3, 4):
                    return ModalityType.VIDEO, {
                        "channels": int(shape[1]),
                        "num_frames": int(shape[0]),
                        "output_size": (int(shape[2]), int(shape[3])),
                    }
                return ModalityType.VIDEO, {
                    "channels": int(shape[3]),
                    "num_frames": int(shape[0]),
                    "output_size": (int(shape[1]), int(shape[2])),
                }

        return ModalityType.VECTOR, {"input_dim": 1}

    @staticmethod
    def detect_from_tensor(tensor: Tensor) -> Tuple[ModalityType, Dict[str, Any]]:
        shape = tensor.shape

        if len(shape) == 1:
            return ModalityType.VECTOR, {"input_dim": shape[0]}

        if len(shape) == 2:
            h, w = shape
            if h > 16 and w > 16:
                return ModalityType.IMAGE, {"channels": 1, "output_size": (int(h), int(w))}
            return ModalityType.TIMESERIES, {"input_dim": int(shape[1]), "output_length": int(shape[0])}

        if len(shape) == 3:
            if shape[0] in (1, 3, 4) and shape[1] > 16 and shape[2] > 16:
                return ModalityType.IMAGE, {
                    "channels": int(shape[0]),
                    "output_size": (int(shape[1]), int(shape[2])),
                }
            if shape[2] in (1, 3, 4) and shape[0] > 16 and shape[1] > 16:
                return ModalityType.IMAGE, {
                    "channels": int(shape[2]),
                    "output_size": (int(shape[0]), int(shape[1])),
                }
            return ModalityType.TIMESERIES, {"input_dim": int(shape[2])}

        if len(shape) == 4:
            if shape[1] in (1, 3, 4):
                return ModalityType.VIDEO, {
                    "channels": int(shape[1]),
                    "num_frames": int(shape[0]),
                    "output_size": (int(shape[2]), int(shape[3])),
                }
            if shape[3] in (1, 3, 4):
                return ModalityType.VIDEO, {
                    "channels": int(shape[3]),
                    "num_frames": int(shape[0]),
                    "output_size": (int(shape[1]), int(shape[2])),
                }
            return ModalityType.VIDEO, {
                "channels": 1,
                "num_frames": int(shape[0]),
                "output_size": (int(shape[1]), int(shape[2])),
            }

        return ModalityType.VECTOR, {"input_dim": int(np.prod(shape[1:]))}

    @staticmethod
    def suggest_embed_dim(modality_type: ModalityType, input_shape: Tuple[int, ...]) -> int:
        if modality_type in (ModalityType.IMAGE, ModalityType.VIDEO):
            return 512
        elif modality_type == ModalityType.AUDIO:
            return 256
        elif modality_type == ModalityType.TEXT:
            return 256
        elif modality_type == ModalityType.VECTOR:
            input_dim = input_shape[0] if len(input_shape) == 1 else int(np.prod(input_shape))
            return max(64, min(512, input_dim * 2))
        return 256


# ======================================================================
# MERGED FROM: perceptor/builder.py
# ======================================================================

# ======================================================================
# MERGED FROM: perceptor/core.py
# ======================================================================

logger = logging.getLogger(__name__)


class UniversalPerceptor(nn.Module):
    """
    通用多模态感知器

    核心职责
    1. 根据 obs_space 自动选择合适的 Encoder
    2. 多模态融合
    3. 输出统一的 vitals 向量
    """

    def __init__(self, config: Optional[Any] = None):
        super().__init__()

        if config is None:
            from .aletheia_foundation import PerceptorConfig as _PerceptorConfig
            config = _PerceptorConfig()

        self.config = config
        self.vitals_dim = config.vitals_dim

        self.encoders: nn.ModuleDict = nn.ModuleDict()
        self.decoders: nn.ModuleDict = nn.ModuleDict()
        self.modality_specs: Dict[str, ModalitySpec] = {}

        self.fusion: Optional[nn.Module] = None
        self.vitals_head: Optional[nn.Module] = None

        self._finalized = False
        self._total_encoder_dim = 0

    def register_modality(
        self,
        name: str,
        modality_type: Union[str, ModalityType],
        input_dim: Optional[int] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ):
        if self._finalized:
            raise RuntimeError("Cannot register modality after finalize()")

        if isinstance(modality_type, str):
            modality_type_enum = ModalityType.from_string(modality_type)
            modality_type_str = modality_type
        else:
            modality_type_enum = modality_type
            modality_type_str = modality_type.to_string()

        embed_dim = self.config.modality_embed_dim

        if input_shape is None:
            input_shape = (input_dim,) if input_dim is not None else (1,)

        spec = ModalitySpec(
            name=name,
            modality_type=modality_type_enum,
            input_shape=input_shape,
            embed_dim=embed_dim,
            extra_params=kwargs,
        )

        # --- 构建编码器参数 ---
        encoder_kwargs = dict(kwargs)
        if input_dim is not None:
            encoder_kwargs['input_dim'] = input_dim

        encoder_kwargs = self._augment_encoder_kwargs(
            modality_type_enum, modality_type_str,
            encoder_kwargs, input_shape, input_dim,
        )

        self.encoders[name] = create_encoder(
            modality_type=modality_type_str,
            output_dim=embed_dim,
            **encoder_kwargs,
        )

        # --- 构建解码器参数 ---
        decoder_kwargs = self._build_decoder_kwargs(
            modality_type_enum, modality_type_str,
            kwargs, input_shape, input_dim,
        )

        self.decoders[name] = create_decoder(
            modality_type=modality_type_str,
            input_dim=self.vitals_dim,
            **decoder_kwargs,
        )

        self.modality_specs[name] = spec
        self._total_encoder_dim += embed_dim

        logger.debug(f"[UniversalPerceptor] Registered '{name}': type={modality_type_str}")

    def _augment_encoder_kwargs(
        self,
        modality_type_enum: ModalityType,
        modality_type_str: str,
        encoder_kwargs: Dict[str, Any],
        input_shape: Tuple[int, ...],
        input_dim: Optional[int],
    ) -> Dict[str, Any]:
        """根据模态类型补充 encoder 所需的额外参数"""

        if modality_type_enum == ModalityType.IMAGE:
            if "output_size" not in encoder_kwargs and input_shape is not None:
                _, h, w = ModalityDetector._parse_image_shape(input_shape)
                if h > 0 and w > 0:
                    encoder_kwargs["output_size"] = (h, w)
            if "backbone" not in encoder_kwargs:
                vision_backbone = getattr(self.config, "vision_backbone", "auto")
                if vision_backbone in ("cnn", "vit"):
                    encoder_kwargs["backbone"] = vision_backbone
                else:
                    encoder_kwargs["backbone"] = ModalityDetector.suggest_image_backbone(
                        input_shape,
                        min_resolution=getattr(self.config, "vision_backbone_min_resolution", 128),
                        patch_size=getattr(self.config, "vision_backbone_patch_size", 16),
                        min_patches=getattr(self.config, "vision_backbone_min_patches", 16),
                    )

        elif modality_type_enum == ModalityType.VIDEO and input_shape is not None and len(input_shape) == 4:
            t0, t1, t2, t3 = input_shape
            if t1 in (1, 3, 4):
                encoder_kwargs.setdefault("channels", int(t1))
                encoder_kwargs.setdefault("num_frames", int(t0))
                encoder_kwargs.setdefault("output_size", (int(t2), int(t3)))
                frame_shape = (t1, t2, t3)
            elif t3 in (1, 3, 4):
                encoder_kwargs.setdefault("channels", int(t3))
                encoder_kwargs.setdefault("num_frames", int(t0))
                encoder_kwargs.setdefault("output_size", (int(t1), int(t2)))
                frame_shape = (t3, t1, t2)
            else:
                frame_shape = (1, t2, t3)

            if "frame_backbone" not in encoder_kwargs:
                vision_backbone = getattr(self.config, "vision_backbone", "auto")
                if vision_backbone in ("cnn", "vit"):
                    encoder_kwargs["frame_backbone"] = vision_backbone
                else:
                    encoder_kwargs["frame_backbone"] = ModalityDetector.suggest_image_backbone(
                        frame_shape,
                        min_resolution=getattr(self.config, "vision_backbone_min_resolution", 128),
                        patch_size=getattr(self.config, "vision_backbone_patch_size", 16),
                        min_patches=getattr(self.config, "vision_backbone_min_patches", 16),
                    )

        elif modality_type_enum == ModalityType.TIMESERIES and input_shape is not None:
            if "output_length" not in encoder_kwargs and len(input_shape) == 2:
                if input_dim is not None:
                    if input_shape[0] == input_dim:
                        encoder_kwargs["output_length"] = int(input_shape[1])
                    elif input_shape[1] == input_dim:
                        encoder_kwargs["output_length"] = int(input_shape[0])

        return encoder_kwargs

    def _build_decoder_kwargs(
        self,
        modality_type_enum: ModalityType,
        modality_type_str: str,
        kwargs: Dict[str, Any],
        input_shape: Optional[Tuple[int, ...]],
        input_dim: Optional[int],
    ) -> Dict[str, Any]:
        """构建并过滤 decoder kwargs"""
        decoder_kwargs = {k: v for k, v in kwargs.items() if k != 'input_dim'}
        if input_dim is not None:
            decoder_kwargs['output_dim'] = input_dim

        if modality_type_enum == ModalityType.IMAGE and input_shape is not None:
            if "output_size" not in decoder_kwargs:
                _, h, w = ModalityDetector._parse_image_shape(input_shape)
                if h > 0 and w > 0:
                    decoder_kwargs["output_size"] = (h, w)

        elif modality_type_enum == ModalityType.VIDEO and input_shape is not None and len(input_shape) == 4:
            t0, t1, t2, t3 = input_shape
            if "output_size" not in decoder_kwargs:
                if t1 in (1, 3, 4):
                    decoder_kwargs["output_size"] = (int(t2), int(t3))
                else:
                    decoder_kwargs["output_size"] = (int(t1), int(t2))
            decoder_kwargs.setdefault("num_frames", int(t0))

        elif modality_type_enum == ModalityType.TIMESERIES and input_shape is not None:
            if "output_length" not in decoder_kwargs and len(input_shape) == 2:
                if input_dim is not None:
                    if input_shape[0] == input_dim:
                        decoder_kwargs["output_length"] = int(input_shape[1])
                    elif input_shape[1] == input_dim:
                        decoder_kwargs["output_length"] = int(input_shape[0])

        # 过滤掉 decoder 不接受的参数
        _ensure_registries()
        decoder_cls = DECODER_REGISTRY.get(modality_type_str)
        if decoder_cls is not None:
            decoder_sig = inspect.signature(decoder_cls.__init__)
            allowed = set(decoder_sig.parameters.keys()) - {"self", "input_dim"}
            decoder_kwargs = {k: v for k, v in decoder_kwargs.items() if k in allowed}

        return decoder_kwargs

    def finalize(self):
        if self._finalized:
            return

        if len(self.encoders) == 0:
            raise RuntimeError("No modalities registered")

        if self.config.fusion_type == "concat":
            self.fusion = nn.Identity()
            fusion_dim = self._total_encoder_dim
        elif self.config.fusion_type == "gated":
            self.fusion = nn.Sequential(
                nn.Linear(self._total_encoder_dim, self._total_encoder_dim),
                nn.Sigmoid(),
            )
            fusion_dim = self._total_encoder_dim
        elif self.config.fusion_type == "attention":
            self.fusion = nn.MultiheadAttention(
                embed_dim=self.config.modality_embed_dim,
                num_heads=4,
                batch_first=True,
            )
            fusion_dim = self.config.modality_embed_dim
        else:
            raise ValueError(f"Unknown fusion type: {self.config.fusion_type}")

        self.vitals_head = build_mlp(
            in_dim=fusion_dim,
            out_dim=self.vitals_dim,
            hidden_dims=(256, 256),
            activation="silu",
            final_gain=1.0,
            init="final_gain_only",
        )

        self._finalized = True
        logger.info(f"[UniversalPerceptor] Finalized: {len(self.encoders)} modalities")

    def encode(self, obs: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        if not self._finalized:
            raise RuntimeError("Call finalize() before encode()")

        # 处理单张量输入
        if isinstance(obs, Tensor):
            if len(self.encoders) == 1:
                name = list(self.encoders.keys())[0]
                obs = {name: obs}
            else:
                raise ValueError("Multiple modalities registered, but got single Tensor")

        # FIX BUG-3: 安全地获取设备和 batch 信息
        first_tensor: Optional[Tensor] = None
        for v in obs.values():
            if isinstance(v, Tensor):
                first_tensor = v
                break
        if first_tensor is None:
            raise ValueError("obs dict contains no valid Tensors")

        # 检测时间维度
        has_time_dim = False
        B: Optional[int] = None
        T: Optional[int] = None

        for name, tensor in obs.items():
            if name not in self.modality_specs:
                continue
            spec = self.modality_specs[name]
            # 5D: (B, T, C, H, W)
            if tensor.dim() == 5:
                has_time_dim = True
                B, T = tensor.shape[:2]
                break
            # 3D for vector/timeseries: (B, T, D)
            if tensor.dim() == 3 and spec.modality_type not in (
                ModalityType.IMAGE, ModalityType.POINTCLOUD,
            ):
                has_time_dim = True
                B, T = tensor.shape[:2]
                break

        # 编码各模态
        embeddings: List[Tensor] = []
        for name, encoder in self.encoders.items():
            if name in obs:
                emb = encoder(obs[name])
                if has_time_dim and emb.dim() == 2 and B is not None and T is not None:
                    emb = emb.reshape(B, T, -1)
                embeddings.append(emb)
            else:
                # 缺失模态用零填充
                if has_time_dim and B is not None and T is not None:
                    emb = torch.zeros(
                        B, T, self.config.modality_embed_dim,
                        device=first_tensor.device, dtype=first_tensor.dtype,
                    )
                else:
                    emb = torch.zeros(
                        first_tensor.shape[0], self.config.modality_embed_dim,
                        device=first_tensor.device, dtype=first_tensor.dtype,
                    )
                embeddings.append(emb)

        # 融合
        if self.config.fusion_type == "attention":
            if has_time_dim and B is not None and T is not None:
                stacked = torch.stack(embeddings, dim=2)  # (B, T, M, E)
                BT_M_E = stacked.reshape(B * T, stacked.shape[2], stacked.shape[3])
                attn_out, _ = self.fusion(BT_M_E, BT_M_E, BT_M_E)
                fused = attn_out.mean(dim=1).reshape(B, T, -1)
            else:
                stacked = torch.stack(embeddings, dim=1)   # (B, M, E)
                attn_out, _ = self.fusion(stacked, stacked, stacked)
                fused = attn_out.mean(dim=1)
        else:
            fused = torch.cat(embeddings, dim=-1)
            if self.config.fusion_type == "gated":
                gate = self.fusion(fused)
                fused = fused * gate

        return self.vitals_head(fused)

    def decode(
        self,
        vitals: Tensor,
        modality_name: Optional[str] = None,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if modality_name is not None:
            return self.decoders[modality_name](vitals)
        recons = {name: dec(vitals) for name, dec in self.decoders.items()}
        return list(recons.values())[0] if len(recons) == 1 else recons

    def forward(self, obs: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        return self.encode(obs)

    @classmethod
    def from_obs_space(cls, obs_space: Any, config: Optional[Any] = None) -> "UniversalPerceptor":
        perceptor = cls(config)
        if hasattr(obs_space, "spaces"):
            for name, space in obs_space.spaces.items():
                perceptor._register_from_space(name, space)
        else:
            perceptor._register_from_space("obs", obs_space)
        perceptor.finalize()
        return perceptor

    def _register_from_space(self, name: str, space: Any):
        modality_type, params = ModalityDetector.detect_from_space(space)
        if hasattr(space, 'shape'):
            params['input_shape'] = space.shape
        self.register_modality(name=name, modality_type=modality_type, **params)
        logger.info(f"[UniversalPerceptor] Auto-registered '{name}': {modality_type.to_string()}")

    def adapt_to_config(self, rssm_config: Any):
        if rssm_config.vitals_dim != self.vitals_dim:
            logger.info(
                f"[UniversalPerceptor] Adapting vitals_dim: "
                f"{self.vitals_dim}  {rssm_config.vitals_dim}"
            )
            self.vitals_dim = rssm_config.vitals_dim
            self.config.vitals_dim = rssm_config.vitals_dim

            if self._finalized:
                fusion_dim = (
                    self.config.modality_embed_dim
                    if self.config.fusion_type == "attention"
                    else self._total_encoder_dim
                )
                self.vitals_head = build_mlp(
                    in_dim=fusion_dim,
                    out_dim=self.vitals_dim,
                    hidden_dims=(256, 256),
                    activation="silu",
                    final_gain=1.0,
                    init="final_gain_only",
                )

    def get_info(self) -> Dict[str, Any]:
        return {
            "vitals_dim": self.vitals_dim,
            "num_modalities": len(self.encoders),
            "modalities": {n: s.to_dict() for n, s in self.modality_specs.items()},
            "fusion_type": self.config.fusion_type,
            "finalized": self._finalized,
        }

    def __repr__(self) -> str:
        modalities = ", ".join(
            f"{n}:{s.modality_type.to_string()}" for n, s in self.modality_specs.items()
        )
        return f"UniversalPerceptor(vitals_dim={self.vitals_dim}, modalities=[{modalities}])"


# ======================================================================
# MERGED FROM: transition.py
# ======================================================================

# --- 温度模式常量 ---
_TEMP_MODE_TRAIN: int = 0
_TEMP_MODE_EVAL: int = 1
_TEMP_MODE_IMAGINE: int = 2
_TEMP_MODE_MAP: Dict[str, int] = {
    "train": _TEMP_MODE_TRAIN,
    "eval": _TEMP_MODE_EVAL,
    "imagine": _TEMP_MODE_IMAGINE,
}


# =============================================================================
# MemoryState
# =============================================================================

class MemoryState:
    """
    Episode-specific 动态记忆状态

    优化reset / update 的 inplace / non-inplace 版本共享核心实现消除重复代码
    """

    __slots__ = ('buffer', 'compressed', 'buffer_pos', 'total_steps', 'config')

    def __init__(
        self,
        buffer: Tensor,
        compressed: Tensor,
        buffer_pos: Optional[Tensor] = None,
        total_steps: Optional[Tensor] = None,
        config: Optional[Any] = None,
    ):
        self.buffer = buffer
        self.compressed = compressed
        self.buffer_pos = buffer_pos
        self.total_steps = total_steps
        self.config = config

    @classmethod
    def create_initial(
        cls,
        batch_size: int,
        hidden_dim: int,
        buffer_size: int = 64,
        compressed_size: int = 256,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        config: Optional[Any] = None,
    ) -> "MemoryState":
        return cls(
            buffer=torch.zeros(batch_size, buffer_size, hidden_dim, device=device, dtype=dtype),
            compressed=torch.zeros(batch_size, compressed_size, hidden_dim, device=device, dtype=dtype),
            buffer_pos=torch.zeros(batch_size, device=device, dtype=torch.long),
            total_steps=torch.zeros(batch_size, device=device, dtype=torch.long),
            config=config,
        )

    def _clone(self) -> "MemoryState":
        """浅层克隆所有张量"""
        return MemoryState(
            buffer=self.buffer.clone(),
            compressed=self.compressed.clone(),
            buffer_pos=self.buffer_pos.clone() if self.buffer_pos is not None else None,
            total_steps=self.total_steps.clone() if self.total_steps is not None else None,
            config=self.config,
        )

    def reset(self, done_mask: Tensor, inplace: bool = True) -> "MemoryState":
        if inplace:
            return self.reset_inplace(done_mask)
        copy = self._clone()
        return copy.reset_inplace(done_mask)

    def reset_inplace(self, done_mask: Tensor) -> "MemoryState":
        if not done_mask.any():
            return self
        self.buffer[done_mask] = 0
        self.compressed[done_mask] = 0
        if self.buffer_pos is not None:
            self.buffer_pos[done_mask] = 0
        if self.total_steps is not None:
            self.total_steps[done_mask] = 0
        return self

    def update(
        self,
        new_state: Tensor,
        done_mask: Optional[Tensor] = None,
        inplace: bool = True,
    ) -> "MemoryState":
        if inplace:
            return self.update_inplace(new_state, done_mask=done_mask)
        copy = self._clone()
        return copy.update_inplace(new_state, done_mask=done_mask)

    def update_inplace(
        self,
        new_state: Tensor,
        done_mask: Optional[Tensor] = None,
    ) -> "MemoryState":
        if done_mask is not None:
            self.reset_inplace(done_mask)

        B, K, D = self.buffer.shape
        if self.buffer_pos is None:
            self.buffer_pos = torch.zeros(B, device=new_state.device, dtype=torch.long)
        if self.total_steps is None:
            self.total_steps = torch.zeros(B, device=new_state.device, dtype=torch.long)

        pos = self.buffer_pos
        indices = pos.view(B, 1, 1).expand(-1, 1, D)
        self.buffer.scatter_(1, indices, new_state.unsqueeze(1))

        new_pos = (pos + 1) % K
        new_steps = self.total_steps + 1

        # 压缩
        compression_ratio = self._get_compression_ratio()
        if compression_ratio is not None and compression_ratio > 0:
            should_compress = (new_steps % compression_ratio) == 0
            if should_compress.any():
                r = compression_ratio
                M = self.compressed.shape[1]
                ar = torch.arange(r, device=new_state.device, dtype=torch.long).view(1, r)
                gather_idx = (new_pos.view(B, 1) - 1 - ar) % K
                gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, D)
                recent = self.buffer.gather(1, gather_idx)
                summary = recent.mean(dim=1)
                slot = ((new_steps - 1) // r) % M
                comp_idx = slot.view(B, 1, 1).expand(-1, 1, D)
                self.compressed.scatter_(1, comp_idx, summary.unsqueeze(1))

        self.buffer_pos = new_pos
        self.total_steps = new_steps
        return self

    def _get_compression_ratio(self) -> Optional[int]:
        if self.config is None or not hasattr(self.config, "compression_ratio"):
            return None
        try:
            return int(self.config.compression_ratio)
        except Exception:
            return None

    def detach(self) -> "MemoryState":
        return MemoryState(
            self.buffer.detach(),
            self.compressed.detach(),
            self.buffer_pos.detach() if self.buffer_pos is not None else None,
            self.total_steps.detach() if self.total_steps is not None else None,
            self.config,
        )


# =============================================================================
# DynamicMemoryGRU
# =============================================================================

class DynamicMemoryGRU(nn.Module):
    """
    动态记忆增强 GRU

    - Recency Buffer + Compressed Memory
    - 可选的注意力检索或简单池化
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        buffer_size: int = 64,
        compressed_size: int = 256,
        use_attention: bool = False,
        attention_heads: int = 4,
        attention_dim: int = 64,
        compression_ratio: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.compressed_size = compressed_size
        self.use_attention = use_attention
        self.compression_ratio = compression_ratio

        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        if use_attention:
            self.attention_heads = attention_heads
            self.attention_dim = attention_dim
            total_dim = attention_heads * attention_dim
            self.query_attn = nn.Linear(hidden_dim, total_dim)
            self.key_attn = nn.Linear(hidden_dim, total_dim)
            self.value_attn = nn.Linear(hidden_dim, total_dim)
            self.out_attn = nn.Linear(total_dim, hidden_dim)
            self.memory_proj = None
        else:
            self.memory_proj = nn.Linear(hidden_dim, hidden_dim)
            self.query_attn = None
            self.key_attn = None
            self.value_attn = None
            self.out_attn = None

        self.output_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        gate_linear = self.memory_gate[0]
        nn.init.zeros_(gate_linear.weight)
        nn.init.constant_(gate_linear.bias, -1.0)

    def _query_memory(
        self,
        query: Tensor,
        memory: MemoryState,
    ) -> Tensor:
        if self.use_attention:
            return self._query_memory_attention(query, memory)
        return self._query_memory_pooling(query, memory)

    def _query_memory_pooling(self, query: Tensor, memory: MemoryState) -> Tensor:
        B, K, D = memory.buffer.shape

        # 计算有效位置 mask
        indices = torch.arange(K, device=query.device).unsqueeze(0)
        if memory.total_steps is not None:
            filled = memory.total_steps.clamp(max=K)
            valid_mask = indices < filled.unsqueeze(1)
        elif memory.buffer_pos is not None:
            valid_mask = indices < memory.buffer_pos.unsqueeze(1)
        else:
            valid_mask = torch.ones(B, K, device=query.device, dtype=torch.bool)

        similarity = torch.bmm(
            query.unsqueeze(1), memory.buffer.transpose(1, 2)
        ).squeeze(1)  # (B, K)

        similarity = similarity.masked_fill(~valid_mask, float('-inf'))
        attn_weights = F.softmax(similarity / (D ** 0.5), dim=-1)
        attn_weights = attn_weights.masked_fill(~valid_mask, 0.0)

        memory_context = torch.bmm(attn_weights.unsqueeze(1), memory.buffer).squeeze(1)
        return self.memory_proj(memory_context)

    def _query_memory_attention(self, query: Tensor, memory: MemoryState) -> Tensor:
        B, K, D = memory.buffer.shape
        H = self.attention_heads
        A = self.attention_dim
        M = memory.compressed.shape[1]

        kv_source = torch.cat([memory.buffer, memory.compressed], dim=1)

        q = self.query_attn(query).view(B, 1, H, A).transpose(1, 2)
        k = self.key_attn(kv_source).view(B, K + M, H, A).transpose(1, 2)
        v = self.value_attn(kv_source).view(B, K + M, H, A).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (A ** 0.5)
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, H * A)

        return self.out_attn(context)

    def forward(
        self,
        x: Tensor,
        h_prev: Tensor,
        memory: Optional[MemoryState] = None,
    ) -> Tuple[Tensor, Optional[MemoryState]]:
        h_gru = self.gru(x, h_prev)

        if memory is None:
            return self.output_norm(h_gru), None

        query = self.query_proj(h_gru)
        memory_context = self._query_memory(query, memory)

        gate_input = torch.cat([h_gru, memory_context], dim=-1)
        gate = self.memory_gate(gate_input)
        h_new = h_gru + gate * memory_context
        h_new = self.output_norm(h_new)

        updated_memory = memory.update_inplace(h_new)

        return h_new, updated_memory

    def forward_sequence(
        self,
        x_seq: Tensor,
        h_init: Tensor,
        memory: Optional[MemoryState] = None,
        dones: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[MemoryState]]:
        B, T, _ = x_seq.shape

        if memory is None:
            memory = MemoryState.create_initial(
                batch_size=B,
                hidden_dim=self.hidden_dim,
                buffer_size=self.buffer_size,
                compressed_size=self.compressed_size,
                device=x_seq.device,
                dtype=x_seq.dtype,
            )

        h_seq = torch.empty(B, T, self.hidden_dim, device=x_seq.device, dtype=x_seq.dtype)
        h = h_init

        for t in range(T):
            if dones is not None:
                done_t = dones[:, t] > 0.5
                if done_t.any():
                    memory.reset_inplace(done_t)

            h, memory = self.forward(x_seq[:, t], h, memory)
            h_seq[:, t] = h

        return h_seq, h, memory


# =============================================================================
# TransitionOutput
# =============================================================================

class TransitionOutput:
    """StateTransition 单步输出"""

    __slots__ = (
        'h', 'z_post_raw', 'z_prior_raw',
        'features_post', 'features_prior',
        'kl_raw', 'quant_loss', 'state_for_next', '_next_state_cache',
    )

    def __init__(
        self,
        h: Tensor,
        z_post_raw: Tensor,
        z_prior_raw: Tensor,
        features_post: Tensor,
        features_prior: Tensor,
        kl_raw: Tensor,
        quant_loss: Optional[Tensor] = None,
        state_for_next: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        self.h = h
        self.z_post_raw = z_post_raw
        self.z_prior_raw = z_prior_raw
        self.features_post = features_post
        self.features_prior = features_prior
        self.kl_raw = kl_raw
        self.quant_loss = quant_loss
        self.state_for_next = state_for_next
        if state_for_next is not None:
            self._next_state_cache = state_for_next
        elif h is not None and z_post_raw is not None:
            self._next_state_cache = (h, z_post_raw)
        elif h is not None and z_prior_raw is not None:
            self._next_state_cache = (h, z_prior_raw)
        else:
            self._next_state_cache = None

    @property
    def next_state(self) -> Tuple[Tensor, Tensor]:
        cached = self._next_state_cache
        if cached is None:
            if self.state_for_next is not None:
                cached = self.state_for_next
            elif self.h is not None and self.z_post_raw is not None:
                cached = (self.h, self.z_post_raw)
            elif self.h is not None and self.z_prior_raw is not None:
                cached = (self.h, self.z_prior_raw)
            else:
                raise AttributeError(
                    f"Cannot extract next state from {type(self).__name__}. "
                    "Expected attributes: state_for_next or (h, z_*_raw)"
                )
            self._next_state_cache = cached
        return cached

    def detach_all(self) -> "TransitionOutput":
        def _d(t: Optional[Tensor]) -> Optional[Tensor]:
            return t.detach() if t is not None else None

        state_next = None
        if self.state_for_next is not None:
            state_next = (self.state_for_next[0].detach(), self.state_for_next[1].detach())

        return TransitionOutput(
            h=_d(self.h),
            z_post_raw=_d(self.z_post_raw),
            z_prior_raw=_d(self.z_prior_raw),
            features_post=_d(self.features_post),
            features_prior=_d(self.features_prior),
            kl_raw=_d(self.kl_raw),
            quant_loss=_d(self.quant_loss),
            state_for_next=state_next,
        )


# =============================================================================
# SequenceOutput
# =============================================================================

class SequenceOutput:
    """observe_sequence 的输出容器"""

    __slots__ = (
        'feats_post', 'feats_prior', 'h_seq',
        'z_post_raw_seq', 'z_prior_raw_seq', 'kl_raw_seq',
        'quant_loss', 'final_state', 'feat_seq',
    )

    def __init__(
        self,
        feats_post: Tensor,
        feats_prior: Tensor,
        h_seq: Tensor,
        z_post_raw_seq: Tensor,
        z_prior_raw_seq: Tensor,
        kl_raw_seq: Tensor,
        quant_loss: Optional[Tensor],
        final_state: Tuple[Tensor, Tensor],
        feat_seq: Tensor,
    ):
        self.feats_post = feats_post
        self.feats_prior = feats_prior
        self.h_seq = h_seq
        self.z_post_raw_seq = z_post_raw_seq
        self.z_prior_raw_seq = z_prior_raw_seq
        self.kl_raw_seq = kl_raw_seq
        self.quant_loss = quant_loss
        self.final_state = final_state
        self.feat_seq = feat_seq

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feats_post': self.feats_post,
            'feats_prior': self.feats_prior,
            'h_seq': self.h_seq,
            'z_post_raw_seq': self.z_post_raw_seq,
            'z_prior_raw_seq': self.z_prior_raw_seq,
            'z_seq': self.z_post_raw_seq,
            'kl_raw_seq': self.kl_raw_seq,
            'quant_loss': self.quant_loss,
            'final_state': self.final_state,
            'feat_seq': self.feat_seq,
        }


# =============================================================================
# StateTransition
# =============================================================================

class StateTransition(nn.Module):
    """RSSM 状态转移模块"""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        from .aletheia_foundation import DistributionFactory

        d_h = config.deter_dim
        d_z_embed = config.z_embed_dim
        d_obs = config.obs_embed_dim
        d_ae = config.action_embed_dim
        N = config.distribution.num_distributions
        K = config.distribution.num_classes

        # 预计算维度
        self._d_h = d_h
        self._d_z_embed = d_z_embed
        self._d_z_raw = N * K
        self._d_feat = config.feat_dim
        self._d_ae = d_ae
        self._d_obs = d_obs
        self._gru_input_dim = d_z_embed + d_ae
        self._post_input_dim = d_h + d_ae + d_obs
        self._prior_input_dim = d_h + d_ae

        # 预缓存温度
        self._temp_train = config.distribution.temperature_train
        self._temp_imagine = config.distribution.temperature_train
        self._temp_eval = config.distribution.temperature_eval

        # free_nats
        self._free_nats = config.kl.free_nats
        self._use_free_nats = self._free_nats > 0
        self._kl_use_balance = bool(getattr(config.kl, "use_balance", False))
        self._kl_balance_alpha = float(getattr(config.kl, "balance_alpha", 0.8))

        # 特征构建路径标志
        self._use_separate_norm = config.use_separate_norm
        self._use_feature_norm = config.use_feature_norm

        # --- GRU ---
        self.gru = nn.GRUCell(
            input_size=self._gru_input_dim,
            hidden_size=d_h,
        )
        self.gru_output_norm = (
            nn.LayerNorm(d_h) if config.use_feature_norm else nn.Identity()
        )

        # --- 持久性门控 ---
        self._use_persist_gate = getattr(config, "use_persist_gate", False)
        if self._use_persist_gate:
            self.persist_gate = nn.Sequential(
                nn.Linear(d_h + d_ae, 128),
                nn.ELU(),
                nn.Linear(128, 1),
            )
            self.persist_temperature = nn.Parameter(torch.ones(1))
            nn.init.zeros_(self.persist_gate[-1].bias)
            nn.init.normal_(self.persist_gate[-1].weight, std=0.01)
            self._persist_gate_stats = {"mean": 0.0, "std": 0.0, "calls": 0}
        else:
            self.persist_gate = None
            self.persist_temperature = None

        # --- 动态记忆 ---
        memory_cfg = getattr(config, "memory", None)
        self._memory_cfg = memory_cfg
        self._memory_enabled = bool(memory_cfg is not None and getattr(memory_cfg, "enabled", False))
        if self._memory_enabled:
            self._memory_buffer_size = int(getattr(memory_cfg, "buffer_size", 64))
            self._memory_compressed_size = int(getattr(memory_cfg, "compressed_size", 256))
            self._memory_compression_ratio = int(getattr(memory_cfg, "compression_ratio", 4))
            self.memory_scale = nn.Parameter(torch.zeros(1))
            self.memory_to_z = nn.Linear(d_h, d_z_embed, bias=False)
        else:
            self._memory_buffer_size = 0
            self._memory_compressed_size = 0
            self._memory_compression_ratio = 0
            self.memory_scale = None
            self.memory_to_z = None

        # --- 随机路径 ---
        self.z_post_predictor = nn.Sequential(
            nn.Linear(self._post_input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, self._d_z_raw),
        )
        self.z_prior_predictor = nn.Sequential(
            nn.Linear(self._prior_input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, self._d_z_raw),
        )
        self.z_proj = nn.Sequential(
            nn.Linear(self._d_z_raw, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, d_z_embed),
        )

        # --- 归一化层 ---
        if config.use_feature_norm:
            if config.use_separate_norm:
                self.z_embed_norm = nn.LayerNorm(d_z_embed)
                self.h_norm = nn.LayerNorm(d_h)
                self.feature_norm = nn.Identity()
            else:
                self.z_embed_norm = nn.Identity()
                self.h_norm = nn.Identity()
                self.feature_norm = nn.LayerNorm(config.feat_dim)
        else:
            self.z_embed_norm = nn.Identity()
            self.h_norm = nn.Identity()
            self.feature_norm = nn.Identity()

        # --- 辅助组件 ---
        self.dist_factory = DistributionFactory(
            num_distributions=N,
            num_classes=K,
            dtype=torch.float32,
        )

        self.init_deter = nn.Parameter(torch.zeros(d_h))
        self.init_stoch_params = nn.Parameter(torch.zeros(self._d_z_raw))

        self.register_buffer("null_action", torch.zeros(1, d_ae), persistent=False)
        self.register_buffer("_zero_kl_template", torch.zeros(1), persistent=False)

        self._dtype = self.init_deter.dtype

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def feat_dim(self) -> int:
        return self._d_feat

    @property
    def device(self) -> torch.device:
        return self.init_deter.device

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _ensure_dtype(self, t: Tensor) -> Tensor:
        return t if t.dtype == self._dtype else t.to(dtype=self._dtype)

    def _get_temperature(self, mode: str) -> float:
        mode_id = _TEMP_MODE_MAP.get(mode, _TEMP_MODE_TRAIN)
        if mode_id == _TEMP_MODE_EVAL:
            return self._temp_eval
        elif mode_id == _TEMP_MODE_IMAGINE:
            return self._temp_imagine
        return self._temp_train

    def _apply_persist_gate(
        self,
        h_prev: Tensor,
        h_new: Tensor,
        action_embed: Tensor,
        update_stats: bool = True,
    ) -> Tensor:
        if not self._use_persist_gate or self.persist_gate is None:
            return h_new

        gate_input = torch.cat([h_prev, action_embed], dim=-1)
        gate_logit = self.persist_gate(gate_input)
        gate = torch.sigmoid(gate_logit / self.persist_temperature.clamp(min=0.1))
        h_gated = h_prev + gate * (h_new - h_prev)

        if update_stats and self.training:
            with torch.no_grad():
                self._persist_gate_stats["mean"] = gate.mean().item()
                self._persist_gate_stats["std"] = (
                    gate.std().item() if gate.numel() > 1 else 0.0
                )
                self._persist_gate_stats["calls"] += 1

        return h_gated

    def _build_features(self, h: Tensor, z_embed: Tensor) -> Tensor:
        if self._use_separate_norm:
            return torch.cat([self.h_norm(h), self.z_embed_norm(z_embed)], dim=-1)
        return self.feature_norm(torch.cat([h, z_embed], dim=-1))

    # =========================================================================
    # 核心方法
    # =========================================================================

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        add_noise: bool = False,
        noise_scale: float = 0.1,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        h = self.init_deter.unsqueeze(0).expand(batch_size, -1).to(device)
        params = self.init_stoch_params.unsqueeze(0).expand(batch_size, -1).to(device)

        if add_noise:
            h = h.clone().add_(torch.randn_like(h), alpha=noise_scale)
            params = params.clone().add_(torch.randn_like(params), alpha=noise_scale)

        dist_logits = self.dist_factory.create_distribution(params)
        z_raw, _ = self.dist_factory.sample(dist_logits, temperature=1.0, deterministic=deterministic)

        return h.contiguous(), z_raw

    def get_features(self, h: Tensor, z_raw: Tensor) -> Tensor:
        z_embed = self.z_proj(z_raw)
        return self._build_features(h, z_embed)

    def _observe_step_core(
        self,
        h_prev: Tensor,
        z_prev_raw: Tensor,
        action_embed: Tensor,
        obs_embed: Tensor,
        temperature: float,
        use_soft_forward: bool,
        compute_metrics: bool = True,
        deterministic_state: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        # 1. GRU 更新
        z_prev_embed = self.z_proj(z_prev_raw)
        h_new = self.gru(torch.cat([z_prev_embed, action_embed], dim=-1), h_prev)
        h_new = self.gru_output_norm(h_new)
        h_new = self._apply_persist_gate(h_prev, h_new, action_embed, update_stats=True)

        # 2. 预测
        h_action = torch.cat([h_new, action_embed], dim=-1)
        post_params = self.z_post_predictor(torch.cat([h_action, obs_embed], dim=-1))
        prior_params = self.z_prior_predictor(h_action)

        # 3. 采样
        post_dist = self.dist_factory.create_distribution(post_params)
        prior_dist = self.dist_factory.create_distribution(prior_params)

        posterior_deterministic = bool(deterministic_state)
        prior_deterministic = bool(deterministic_state or use_soft_forward)
        z_post_raw, z_post_soft = self.dist_factory.sample(
            post_dist, temperature, deterministic=posterior_deterministic, return_soft=use_soft_forward
        )
        z_prior_raw, _ = self.dist_factory.sample(
            prior_dist, temperature, deterministic=prior_deterministic
        )

        # 4. KL 和量化损失
        kl_raw = None
        quant_loss = None
        if compute_metrics:
            kl_balance = self._kl_balance_alpha if self._kl_use_balance else False
            kl_raw = self.dist_factory.compute_kl(post_dist, prior_dist, balance=kl_balance)
            if use_soft_forward:
                with torch.no_grad():
                    z_post_hard, _ = self.dist_factory.sample(
                        post_dist, temperature, deterministic=True
                    )
                quant_loss = self.dist_factory.compute_quant_loss(z_post_soft, z_post_hard)
        else:
            kl_raw = self._zero_kl_template.expand(h_new.shape[0])

        # 5. 特征
        z_post_embed = self.z_proj(z_post_raw)
        z_prior_embed = self.z_proj(z_prior_raw)
        feat_post = self._build_features(h_new, z_post_embed)
        feat_prior = self._build_features(h_new, z_prior_embed)

        return h_new, z_post_raw, z_prior_raw, feat_post, feat_prior, kl_raw, quant_loss

    def observe_step(
        self,
        prev_state: Tuple[Tensor, Tensor],
        action_embed: Tensor,
        obs_embed: Tensor,
        sampling_mode: str = "posterior",
        temperature: Optional[float] = None,
        use_soft_forward: bool = False,
        temperature_mode: str = "train",
        compute_metrics: bool = True,
        deterministic_state: bool = False,
    ) -> TransitionOutput:
        h_prev, z_prev_raw = prev_state
        dtype = self._dtype

        h_prev = self._ensure_dtype(h_prev)
        z_prev_raw = self._ensure_dtype(z_prev_raw)
        action_embed = self._ensure_dtype(action_embed)
        obs_embed = self._ensure_dtype(obs_embed)

        if temperature is None:
            temperature = self._get_temperature(temperature_mode)

        h_new, z_post_raw, z_prior_raw, feat_post, feat_prior, kl_raw, quant_loss = (
            self._observe_step_core(
                h_prev, z_prev_raw, action_embed, obs_embed,
                temperature, use_soft_forward,
                compute_metrics=compute_metrics,
                deterministic_state=deterministic_state,
            )
        )

        return TransitionOutput(
            h=h_new,
            z_post_raw=z_post_raw,
            z_prior_raw=z_prior_raw,
            features_post=feat_post,
            features_prior=feat_prior,
            kl_raw=kl_raw,
            quant_loss=quant_loss,
            state_for_next=(h_new, z_post_raw),
        )

    def imagine_step(
        self,
        prev_state: Tuple[Tensor, Tensor],
        action_embed: Tensor,
        temperature: Optional[float] = None,
        use_soft_forward: bool = False,
        temperature_mode: str = "imagine",
    ) -> TransitionOutput:
        h_prev, z_prev_raw = prev_state
        dtype = self._dtype

        h_prev = self._ensure_dtype(h_prev)
        z_prev_raw = self._ensure_dtype(z_prev_raw)
        action_embed = self._ensure_dtype(action_embed)

        if temperature is None:
            temperature = self._get_temperature(temperature_mode)

        # 1. GRU 更新
        z_prev_embed = self.z_proj(z_prev_raw)
        h_new = self.gru(torch.cat([z_prev_embed, action_embed], dim=-1), h_prev)
        h_new = self.gru_output_norm(h_new)
        h_new = self._apply_persist_gate(h_prev, h_new, action_embed, update_stats=False)

        # 2. 先验预测
        prior_params = self.z_prior_predictor(torch.cat([h_new, action_embed], dim=-1))
        prior_dist = self.dist_factory.create_distribution(prior_params)
        z_prior_raw, _ = self.dist_factory.sample(
            prior_dist, temperature, deterministic=use_soft_forward
        )

        # 3. 特征
        z_prior_embed = self.z_proj(z_prior_raw)
        feat_prior = self._build_features(h_new, z_prior_embed)

        B = h_new.shape[0]
        zero_kl = self._zero_kl_template.expand(B)

        return TransitionOutput(
            h=h_new,
            z_post_raw=z_prior_raw,
            z_prior_raw=z_prior_raw,
            features_post=feat_prior,
            features_prior=feat_prior,
            kl_raw=zero_kl,
            quant_loss=None,
            state_for_next=(h_new, z_prior_raw),
        )

    def observe_sequence(
        self,
        obs_embeds: Tensor,
        action_embeds: Tensor,
        init_state: Optional[Tuple[Tensor, Tensor]] = None,
        use_soft_forward: bool = False,
        use_checkpoint: bool = False,
        dones: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        B, T, _ = obs_embeds.shape
        device = obs_embeds.device
        dtype = self._dtype

        obs_embeds = self._ensure_dtype(obs_embeds)
        action_embeds = self._ensure_dtype(action_embeds)

        # 处理 dones
        dones_mask: Optional[Tensor] = None
        if dones is not None:
            if dones.shape[0] != B or dones.shape[1] != T:
                raise ValueError(f"dones shape mismatch: expected {(B, T)}, got {tuple(dones.shape[:2])}")
            dones_mask = dones.to(device=device)
            if torch.is_floating_point(dones_mask):
                dones_mask = dones_mask > 0.5
            else:
                dones_mask = dones_mask.to(torch.bool)

        # 初始状态
        if init_state is None:
            h_prev, z_prev_raw = self.initial_state(B, device)
        else:
            h_prev = self._ensure_dtype(init_state[0])
            z_prev_raw = self._ensure_dtype(init_state[1])

        h_init = h_prev
        z_init_raw = z_prev_raw
        temperature = self._temp_train

        # 预分配输出
        h_seq = torch.empty(B, T, self._d_h, device=device, dtype=dtype)
        z_post_raw_seq = torch.empty(B, T, self._d_z_raw, device=device, dtype=dtype)
        z_prior_raw_seq = torch.empty(B, T, self._d_z_raw, device=device, dtype=dtype)
        kl_raw_seq = torch.empty(B, T, device=device, dtype=dtype)
        feats_post_seq = torch.empty(B, T, self._d_feat, device=device, dtype=dtype)
        feats_prior_seq = torch.empty(B, T, self._d_feat, device=device, dtype=dtype)

        quant_loss_sum = torch.zeros(1, device=device, dtype=dtype)
        quant_loss_count = 0

        # 预获取属性引用
        gru = self.gru
        gru_output_norm = self.gru_output_norm
        z_proj = self.z_proj
        z_post_predictor = self.z_post_predictor
        z_prior_predictor = self.z_prior_predictor
        dist_factory = self.dist_factory

        # 重置模板
        reset_h_template: Optional[Tensor] = None
        reset_dist = None
        if init_state is None and dones_mask is not None:
            reset_h_template = self.init_deter.unsqueeze(0).expand(B, -1).to(device)
            reset_params = self.init_stoch_params.unsqueeze(0).expand(B, -1).to(device)
            reset_dist = dist_factory.create_distribution(reset_params)

        # 记忆
        memory_enabled = self._memory_enabled
        memory_state: Optional[MemoryState] = None
        if memory_enabled:
            memory_state = MemoryState.create_initial(
                batch_size=B,
                hidden_dim=self._d_h,
                buffer_size=self._memory_buffer_size,
                compressed_size=self._memory_compressed_size,
                device=device,
                dtype=dtype,
                config=self._memory_cfg,
            )
            # FIX LOGIC-3: checkpoint 与 memory 互斥添加警告
            if use_checkpoint:
                logger.warning(
                    "Gradient checkpointing disabled: incompatible with dynamic memory. "
                    "Set use_checkpoint=False to suppress this warning."
                )
                use_checkpoint = False

        # 序列处理
        for t in range(T):
            # 按需重置
            if dones_mask is not None and t > 0:
                reset_mask = dones_mask[:, t - 1]
                if reset_mask.any():
                    reset_mask_e = reset_mask.unsqueeze(-1)
                    if init_state is None and reset_h_template is not None and reset_dist is not None:
                        z_reset, _ = dist_factory.sample(
                            reset_dist, temperature=1.0, deterministic=False
                        )
                        h_prev = torch.where(reset_mask_e, reset_h_template, h_prev)
                        z_prev_raw = torch.where(reset_mask_e, z_reset, z_prev_raw)
                    else:
                        h_prev = torch.where(reset_mask_e, h_init, h_prev)
                        z_prev_raw = torch.where(reset_mask_e, z_init_raw, z_prev_raw)
                    if memory_state is not None:
                        memory_state.reset_inplace(reset_mask)

            current_action = action_embeds[:, t]
            current_obs = obs_embeds[:, t]

            if use_checkpoint:
                result = torch.utils.checkpoint.checkpoint(
                    self._observe_step_checkpoint_fn,
                    h_prev, z_prev_raw, current_action, current_obs,
                    temperature, use_soft_forward,
                    use_reentrant=False,
                )
                h_new, z_post_raw, z_prior_raw, feat_post, feat_prior, kl_raw, quant_loss = result
            else:
                # 内联核心逻辑
                z_prev_embed = z_proj(z_prev_raw)

                if memory_state is not None and self.memory_to_z is not None and self.memory_scale is not None:
                    pos = memory_state.buffer_pos
                    read_pos = (pos - 1) % self._memory_buffer_size
                    idx = read_pos.view(B, 1, 1).expand(-1, 1, self._d_h)
                    mem_ctx = memory_state.buffer.gather(1, idx).squeeze(1)
                    z_prev_embed = z_prev_embed + torch.tanh(self.memory_scale) * self.memory_to_z(mem_ctx)

                h_new = gru(torch.cat([z_prev_embed, current_action], dim=-1), h_prev)
                h_new = gru_output_norm(h_new)
                h_new = self._apply_persist_gate(h_prev, h_new, current_action, update_stats=False)

                h_action = torch.cat([h_new, current_action], dim=-1)
                post_params = z_post_predictor(torch.cat([h_action, current_obs], dim=-1))
                prior_params = z_prior_predictor(h_action)

                post_dist = dist_factory.create_distribution(post_params)
                prior_dist = dist_factory.create_distribution(prior_params)

                z_post_raw, z_post_soft = dist_factory.sample(
                    post_dist, temperature, deterministic=False, return_soft=use_soft_forward
                )
                z_prior_raw, _ = dist_factory.sample(
                    prior_dist, temperature, deterministic=use_soft_forward
                )

                kl_balance = self._kl_balance_alpha if self._kl_use_balance else False
                kl_raw = dist_factory.compute_kl(post_dist, prior_dist, balance=kl_balance)

                z_post_embed = z_proj(z_post_raw)
                z_prior_embed = z_proj(z_prior_raw)
                feat_post = self._build_features(h_new, z_post_embed)
                feat_prior = self._build_features(h_new, z_prior_embed)

                quant_loss = None
                if use_soft_forward:
                    with torch.no_grad():
                        z_post_hard, _ = dist_factory.sample(
                            post_dist, temperature, deterministic=True
                        )
                    quant_loss = dist_factory.compute_quant_loss(z_post_soft, z_post_hard)

            # 写入预分配张量
            h_seq[:, t] = h_new
            z_post_raw_seq[:, t] = z_post_raw
            z_prior_raw_seq[:, t] = z_prior_raw
            feats_post_seq[:, t] = feat_post
            feats_prior_seq[:, t] = feat_prior
            kl_raw_seq[:, t] = kl_raw

            if use_soft_forward and quant_loss is not None:
                quant_loss_sum.add_(quant_loss)
                quant_loss_count += 1

            h_prev = h_new
            z_prev_raw = z_post_raw
            if memory_state is not None:
                memory_state.update_inplace(h_new)

        # 构建输出
        f_init = self.get_features(h_init, z_init_raw)
        feat_seq = torch.cat([f_init.unsqueeze(1), feats_post_seq], dim=1)
        final_state = (h_prev, z_prev_raw)
        quant_loss_mean = quant_loss_sum / quant_loss_count if quant_loss_count > 0 else None

        return {
            'feats_post': feats_post_seq,
            'feats_prior': feats_prior_seq,
            'h_seq': h_seq,
            'z_post_raw_seq': z_post_raw_seq,
            'z_prior_raw_seq': z_prior_raw_seq,
            'z_seq': z_post_raw_seq,
            'kl_raw_seq': kl_raw_seq,
            'quant_loss': quant_loss_mean,
            'final_state': final_state,
            'feat_seq': feat_seq,
            'memory_state': memory_state,
        }

    def _observe_step_checkpoint_fn(
        self,
        h_prev: Tensor,
        z_prev_raw: Tensor,
        action_embed: Tensor,
        obs_embed: Tensor,
        temperature: float,
        use_soft_forward: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        h_new, z_post_raw, z_prior_raw, feat_post, feat_prior, kl_raw, quant_loss = (
            self._observe_step_core(
                h_prev, z_prev_raw, action_embed, obs_embed,
                temperature, use_soft_forward
            )
        )
        if quant_loss is None:
            quant_loss = self._zero_kl_template
        return h_new, z_post_raw, z_prior_raw, feat_post, feat_prior, kl_raw, quant_loss

    def imagine_sequence(
        self,
        init_state: Tuple[Tensor, Tensor],
        action_embeds: Tensor,
        temperature: Optional[float] = None,
        use_soft_forward: bool = False,
    ) -> Dict[str, Tensor]:
        B, T, _ = action_embeds.shape
        device = action_embeds.device
        dtype = self._dtype

        action_embeds = self._ensure_dtype(action_embeds)
        h_prev = self._ensure_dtype(init_state[0])
        z_prev_raw = self._ensure_dtype(init_state[1])

        if temperature is None:
            temperature = self._temp_imagine

        h_seq = torch.empty(B, T, self._d_h, device=device, dtype=dtype)
        z_seq = torch.empty(B, T, self._d_z_raw, device=device, dtype=dtype)
        feat_seq = torch.empty(B, T, self._d_feat, device=device, dtype=dtype)

        gru = self.gru
        gru_output_norm = self.gru_output_norm
        z_proj = self.z_proj
        z_prior_predictor = self.z_prior_predictor
        dist_factory = self.dist_factory

        for t in range(T):
            action_embed = action_embeds[:, t]

            z_prev_embed = z_proj(z_prev_raw)
            h_new = gru(torch.cat([z_prev_embed, action_embed], dim=-1), h_prev)
            h_new = gru_output_norm(h_new)
            h_new = self._apply_persist_gate(h_prev, h_new, action_embed, update_stats=False)

            prior_params = z_prior_predictor(torch.cat([h_new, action_embed], dim=-1))
            prior_dist = dist_factory.create_distribution(prior_params)
            z_prior_raw, _ = dist_factory.sample(
                prior_dist, temperature, deterministic=use_soft_forward
            )

            z_prior_embed = z_proj(z_prior_raw)
            feat = self._build_features(h_new, z_prior_embed)

            h_seq[:, t] = h_new
            z_seq[:, t] = z_prior_raw
            feat_seq[:, t] = feat

            h_prev = h_new
            z_prev_raw = z_prior_raw

        return {
            'h_seq': h_seq,
            'z_seq': z_seq,
            'feat_seq': feat_seq,
            'final_state': (h_prev, z_prev_raw),
        }

# ======================================================================
# MERGED FROM: heads.py
# ======================================================================

"""
预测头模块

包含:
- SymlogRewardHead, ContinueHead, VitalsDecoder
- MultiHorizonContinueHead, PredictionHeads
"""


# =============================================================================
# 1 配置类
# =============================================================================

@dataclass
class RewardHeadConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: Literal['silu', 'relu', 'gelu', 'elu'] = 'silu'
    layer_norm: bool = True
    symlog: "SymlogConfig" = field(default_factory=lambda: SymlogConfig())
    init: "InitConfig" = field(default_factory=lambda: InitConfig())


@dataclass
class ContinueHeadConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: Literal['silu', 'relu', 'gelu', 'elu'] = 'silu'
    layer_norm: bool = True
    temperature: float = 1.0
    loss_type: Literal['bce', 'focal'] = 'bce'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    optimistic_bias: float = 5.0
    positive_weight: float = 1.0
    negative_weight: float = 1.0
    init: "InitConfig" = field(default_factory=lambda: InitConfig())


@dataclass
class VitalsDecoderConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: Literal['silu', 'relu', 'gelu', 'elu'] = 'silu'
    layer_norm: bool = True
    init: "InitConfig" = field(default_factory=lambda: InitConfig())


@dataclass
class PredictionHeadsConfig:
    feat_dim: int = 640
    vitals_dim: int = 64
    reward: RewardHeadConfig = field(default_factory=RewardHeadConfig)
    continue_: ContinueHeadConfig = field(default_factory=ContinueHeadConfig)
    decoder: VitalsDecoderConfig = field(default_factory=VitalsDecoderConfig)
    mhc: Optional["MultiHorizonContinueConfig"] = None


# 延迟导入的配置类型引用
from .aletheia_foundation import (
    InitConfig,
    SymlogConfig,
    MultiHorizonContinueConfig,
    get_activation_class,
    SymlogLayer,
)


# =============================================================================
# 2 工具函数
# =============================================================================

def _init_linear(
    module: nn.Linear,
    config: "InitConfig",
    is_output: bool = False,
) -> None:
    gain = config.output_gain if is_output else config.backbone_gain

    if config.method == 'orthogonal':
        nn.init.orthogonal_(module.weight, gain=gain)
    elif config.method == 'xavier':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    elif config.method == 'kaiming':
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        module.weight.data *= gain

    if module.bias is not None:
        if config.bias_init == 'zeros':
            nn.init.zeros_(module.bias)
        elif config.bias_init == 'uniform':
            fan_in = module.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(module.bias, -bound, bound)


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: List[int],
    activation: str = 'silu',
    layer_norm: bool = True,
    output_activation: bool = False,
) -> nn.Sequential:
    """构建 MLP优先使用 SwiGLU + RMSNorm 路径"""
    from .aletheia_foundation import build_swiglu_mlp, create_norm

    if layer_norm and activation.lower() == "silu":
        hidden_dims_tuple = tuple(hidden_dims) if hidden_dims else None
        core = build_swiglu_mlp(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims_tuple,
            num_layers=2,
            dropout=0.0,
            norm_type="rms",
        )
        if not output_activation:
            return nn.Sequential(core)
        return nn.Sequential(core, create_norm(out_dim, "rms"), nn.SiLU())

    # 回退路径
    act_cls = get_activation_class(activation)
    layers: List[nn.Module] = []
    prev_dim = in_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_cls())
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, out_dim))

    if output_activation:
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(act_cls())

    return nn.Sequential(*layers)


def _init_last_linear_gain(net: nn.Module, init_cfg: "InitConfig") -> None:
    linear_modules = [m for m in net.modules() if isinstance(m, nn.Linear)]
    if not linear_modules:
        return
    last_linear = linear_modules[-1]
    for m in linear_modules:
        _init_linear(m, init_cfg, is_output=(m is last_linear))


# =============================================================================
# 4 预测头基类
# =============================================================================

class PredictionHead(nn.Module, ABC):
    """预测头抽象基类"""

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        pass

    @abstractmethod
    def compute_loss(
        self,
        features: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        pass

    def compute_symmetric_loss(
        self,
        features_post: Tensor,
        features_prior: Tensor,
        targets: Tensor,
        prior_weight: float = 0.5,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss_post = self.compute_loss(features_post, targets, mask)
        loss_prior = self.compute_loss(features_prior, targets, mask)
        loss = (1 - prior_weight) * loss_post + prior_weight * loss_prior
        return loss, {
            f'{self.name}_loss_post': loss_post.detach(),
            f'{self.name}_loss_prior': loss_prior.detach(),
            f'{self.name}_loss': loss.detach(),
        }

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def _handle_sequence_input(
        self,
        features: Tensor,
    ) -> Tuple[Tensor, bool, Optional[Tuple[int, int]]]:
        if features.dim() == 3:
            B, T, D = features.shape
            return features.reshape(B * T, D), True, (B, T)
        return features, False, None

    def _restore_sequence_output(
        self,
        output: Tensor,
        is_sequence: bool,
        original_shape: Optional[Tuple[int, int]],
    ) -> Tensor:
        if is_sequence and original_shape is not None:
            B, T = original_shape
            return output.reshape(B, T) if output.dim() == 1 else output.reshape(B, T, -1)
        return output


# =============================================================================
# 5 Symlog 奖励头
# =============================================================================

class SymlogRewardHead(PredictionHead):
    """奖励预测头Symlog 空间"""

    def __init__(
        self,
        feat_dim: int,
        config: Optional[RewardHeadConfig] = None,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            config = RewardHeadConfig(
                hidden_dims=kwargs.get('hidden_dims', (256, 256)),
                activation=kwargs.get('activation', 'silu'),
                layer_norm=kwargs.get('layer_norm', True),
                symlog=kwargs.get('symlog', SymlogConfig()),
            )

        self.config = config
        self.feat_dim = feat_dim

        self.net = _build_mlp(
            in_dim=feat_dim,
            out_dim=1,
            hidden_dims=list(config.hidden_dims),
            activation=config.activation,
            layer_norm=config.layer_norm,
        )

        self.symlog = SymlogLayer(
            input_clip=config.symlog.input_clip,
            symexp_clip=config.symlog.symexp_clip,
            output_clip=config.symlog.output_clip,
        )

        _init_last_linear_gain(self.net, self.config.init)

    @property
    def name(self) -> str:
        return 'reward'

    def forward(self, features: Tensor) -> Tensor:
        features, is_seq, shape = self._handle_sequence_input(features)
        pred = self.net(features).squeeze(-1)
        return self._restore_sequence_output(pred, is_seq, shape)

    def predict_real(self, features: Tensor) -> Tensor:
        return self.symlog.symexp(self.forward(features))

    def compute_loss(
        self,
        features: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        pred_symlog = self.forward(features)
        target_symlog = self.symlog.symlog(target)
        loss = (pred_symlog - target_symlog) ** 2
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        return loss.mean()


# =============================================================================
# 6 Continue 预测头
# =============================================================================

class ContinueHead(PredictionHead):
    """
    继续预测头输出 logits

    支持温度缩放和 Focal Loss
    """

    # 乐观初始化偏置值sigmoid(5.0)  0.993
    _DEFAULT_OPTIMISTIC_BIAS: float = 5.0

    def __init__(
        self,
        feat_dim: int,
        config: Optional[ContinueHeadConfig] = None,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            config = ContinueHeadConfig(
                hidden_dims=kwargs.get('hidden_dims', (256, 256)),
                activation=kwargs.get('activation', 'silu'),
                layer_norm=kwargs.get('layer_norm', True),
                temperature=kwargs.get('temperature', 1.0),
                loss_type=kwargs.get('loss_type', 'bce'),
                focal_alpha=kwargs.get('focal_alpha', 0.25),
                focal_gamma=kwargs.get('focal_gamma', 2.0),
                optimistic_bias=kwargs.get('optimistic_bias', self._DEFAULT_OPTIMISTIC_BIAS),
                positive_weight=kwargs.get('positive_weight', 1.0),
                negative_weight=kwargs.get('negative_weight', 1.0),
            )

        self.config = config
        self.feat_dim = feat_dim

        self.net = _build_mlp(
            in_dim=feat_dim,
            out_dim=1,
            hidden_dims=list(config.hidden_dims),
            activation=config.activation,
            layer_norm=config.layer_norm,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        _init_last_linear_gain(self.net, self.config.init)
        # 乐观初始化防止早期"死亡恐慌"
        linear_modules = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
        if linear_modules and linear_modules[-1].bias is not None:
            nn.init.constant_(linear_modules[-1].bias, float(self.config.optimistic_bias))

    @property
    def name(self) -> str:
        return 'continue'

    def forward(self, features: Tensor) -> Tensor:
        features, is_seq, shape = self._handle_sequence_input(features)
        logits = self.net(features).squeeze(-1).clamp(-20, 20)
        return self._restore_sequence_output(logits, is_seq, shape)

    def get_prob(self, features: Tensor) -> Tensor:
        logits = self.forward(features)
        return torch.sigmoid(logits / self.config.temperature)

    def compute_loss(
        self,
        features: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        logits = self.forward(features)
        return self.compute_loss_from_logits(logits, targets, mask=mask)

    def compute_loss_from_logits(
        self,
        logits: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return compute_continue_loss(
            logits,
            targets,
            mask=mask,
            temperature=self.config.temperature,
            loss_type=self.config.loss_type,
            focal_alpha=self.config.focal_alpha,
            focal_gamma=self.config.focal_gamma,
            positive_weight=self.config.positive_weight,
            negative_weight=self.config.negative_weight,
        )

    def compute_symmetric_loss(
        self,
        features_post: Tensor,
        features_prior: Tensor,
        targets: Tensor,
        prior_weight: float = 0.5,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss_post = self.compute_loss(features_post, targets, mask)
        loss_prior = self.compute_loss(features_prior, targets, mask)
        loss = (1 - prior_weight) * loss_post + prior_weight * loss_prior

        with torch.no_grad():
            probs_post = self.get_prob(features_post)
            preds = (probs_post > 0.5).float()

            if mask is not None:
                total = mask.sum() + 1e-8
                acc = ((preds == targets).float() * mask).sum() / total
                prob_mean = (probs_post * mask).sum() / total
                death_pred_ratio = ((probs_post < 0.5).float() * mask).sum() / total
                true_death_ratio = ((targets < 0.5).float() * mask).sum() / total
            else:
                acc = (preds == targets).float().mean()
                prob_mean = probs_post.mean()
                death_pred_ratio = (probs_post < 0.5).float().mean()
                true_death_ratio = (targets < 0.5).float().mean()

        return loss, {
            'continue_loss_post': loss_post.detach(),
            'continue_loss_prior': loss_prior.detach(),
            'continue_loss': loss.detach(),
            'continue_acc': acc,
            'continue_prob_mean': prob_mean,
            'continue_death_pred_ratio': death_pred_ratio,
            'continue_true_death_ratio': true_death_ratio,
            'continue_temperature': torch.tensor(self.config.temperature, device=features_post.device),
        }


# =============================================================================
# 7 Vitals 解码头
# =============================================================================

class VitalsDecoder(PredictionHead):
    """从特征重构 vitals"""

    def __init__(
        self,
        feat_dim: int,
        vitals_dim: int,
        config: Optional[VitalsDecoderConfig] = None,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            config = VitalsDecoderConfig(
                hidden_dims=kwargs.get('hidden_dims', (256, 256)),
                activation=kwargs.get('activation', 'silu'),
                layer_norm=kwargs.get('layer_norm', True),
            )

        self.config = config
        self.feat_dim = feat_dim
        self.vitals_dim = vitals_dim

        self.net = _build_mlp(
            in_dim=feat_dim,
            out_dim=vitals_dim,
            hidden_dims=list(config.hidden_dims),
            activation=config.activation,
            layer_norm=config.layer_norm,
        )
        _init_last_linear_gain(self.net, self.config.init)

    @property
    def name(self) -> str:
        return 'recon'

    def forward(self, features: Tensor) -> Tensor:
        features, is_seq, shape = self._handle_sequence_input(features)
        recon = self.net(features)
        return self._restore_sequence_output(recon, is_seq, shape)

    def compute_loss(
        self,
        features: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        recon = self.forward(features)
        loss = (recon - target) ** 2
        if mask is not None:
            if mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
            loss = loss * mask
            return loss.sum() / (mask.sum() * loss.shape[-1] + 1e-8)
        return loss.mean()


# =============================================================================
# 8 Multi-Horizon Continue Head
# =============================================================================

class MultiHorizonContinueHead(PredictionHead):
    """多视野生存概率预测头"""

    def __init__(self, config: "MultiHorizonContinueConfig"):
        super().__init__()
        from .aletheia_foundation import build_swiglu_mlp, create_norm

        self.config = config
        self.horizons = list(config.horizons)
        self.num_horizons = len(self.horizons)

        # 损失权重
        if config.horizon_weights is None:
            w = torch.ones(self.num_horizons, dtype=torch.float32) / max(self.num_horizons, 1)
        else:
            assert len(config.horizon_weights) == self.num_horizons
            w = torch.tensor(config.horizon_weights, dtype=torch.float32)
            w = w / (w.sum() + 1e-8)
        self.register_buffer("horizon_loss_weights", w)

        # 共享 backbone
        self.backbone = self._build_backbone(config, build_swiglu_mlp, create_norm)

        self.heads = nn.ModuleList([nn.Linear(config.hidden_dim, 1) for _ in self.horizons])

        self._init_weights()

    @staticmethod
    def _build_backbone(cfg, build_swiglu_mlp_fn, create_norm_fn) -> nn.Sequential:
        if str(cfg.activation).lower() == "silu":
            depth = max(1, int(cfg.num_layers))
            hidden_dims = (int(cfg.hidden_dim),) * depth
            return nn.Sequential(build_swiglu_mlp_fn(
                in_dim=int(cfg.feat_dim),
                out_dim=int(cfg.hidden_dim),
                hidden_dims=hidden_dims,
                dropout=0.0,
                norm_type="rms",
            ))

        act_cls = get_activation_class(cfg.activation)
        layers: List[nn.Module] = [nn.Linear(cfg.feat_dim, cfg.hidden_dim), act_cls()]
        for _ in range(max(0, cfg.num_layers - 1)):
            layers.extend([nn.Linear(cfg.hidden_dim, cfg.hidden_dim), act_cls()])
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        init_cfg = self.config.init
        for module in self.backbone.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module, init_cfg, is_output=False)
        for head in self.heads:
            _init_linear(head, init_cfg, is_output=True)

    @property
    def name(self) -> str:
        return 'mhc'

    def forward(self, feats_tm1: Tensor) -> Tensor:
        if feats_tm1.dim() != 3:
            raise ValueError(f"feats_tm1 must be (B, T, D), got shape={tuple(feats_tm1.shape)}")
        shared = self.backbone(feats_tm1)
        return torch.cat([head(shared) for head in self.heads], dim=-1)

    def build_targets(self, continues: Tensor) -> Tuple[Tensor, Tensor]:
        if continues.dim() != 2:
            raise ValueError(f"continues must be (B, T), got shape={tuple(continues.shape)}")
        if not continues.is_floating_point():
            continues = continues.float()

        B, T = continues.shape
        device = continues.device
        dtype = continues.dtype

        targets = torch.ones(B, T, self.num_horizons, device=device, dtype=dtype)
        mask = torch.zeros(B, T, self.num_horizons, device=device, dtype=dtype)

        if T <= 1:
            return targets, mask

        C1 = continues[:, 1:]  # (B, T-1)

        for h_idx, h in enumerate(self.horizons):
            valid_len = T - h
            if valid_len <= 0 or h > (T - 1):
                continue
            win = C1.unfold(dimension=1, size=h, step=1)
            prod = win.prod(dim=-1)
            targets[:, :valid_len, h_idx] = prod
            mask[:, :valid_len, h_idx] = 1.0

        return targets, mask

    def loss(
        self,
        feats_tm1: Tensor,
        continues: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, T, D = feats_tm1.shape

        if continues.dim() != 2:
            raise ValueError(f"continues must be (B, T), got shape={tuple(continues.shape)}")
        if not continues.is_floating_point():
            continues = continues.float()
        continues = continues.to(device=feats_tm1.device, dtype=feats_tm1.dtype)

        if T < self.config.min_seq_len:
            warnings.warn(f"Sequence length {T} < min_seq_len {self.config.min_seq_len}", UserWarning)

        logits = self.forward(feats_tm1)
        targets, mask = self.build_targets(continues)

        per_horizon_losses: List[Tensor] = []
        per_horizon_accs: List[Tensor] = []
        valid_indices: List[int] = []
        metrics: Dict[str, Tensor] = {}

        for h_idx, h in enumerate(self.horizons):
            logit_h = logits[..., h_idx]
            target_h = targets[..., h_idx]
            mask_h = mask[..., h_idx]
            num_valid = mask_h.sum()

            if self.config.skip_invalid_horizons and num_valid == 0:
                continue

            valid_indices.append(h_idx)

            bce = F.binary_cross_entropy_with_logits(logit_h, target_h, reduction='none')
            loss_h = (bce * mask_h).sum() / (num_valid + 1e-8)
            per_horizon_losses.append(loss_h)

            with torch.no_grad():
                pred = (torch.sigmoid(logit_h) > 0.5).float()
                acc_h = ((pred == (target_h > 0.5).float()) * mask_h).sum() / (num_valid + 1e-8)
                per_horizon_accs.append(acc_h)
                metrics[f'mhc_loss_h{h}'] = loss_h.detach()
                metrics[f'mhc_acc_h{h}'] = acc_h

        if not valid_indices:
            total_loss = torch.tensor(0.0, device=feats_tm1.device, requires_grad=True)
        else:
            idxs = torch.tensor(valid_indices, device=feats_tm1.device, dtype=torch.long)
            w = self.horizon_loss_weights[idxs]
            w = w / (w.sum() + 1e-8)
            losses_t = torch.stack(per_horizon_losses)
            total_loss = (w * losses_t).sum()

        metrics['mhc_valid_horizons'] = torch.tensor(float(len(valid_indices)), device=feats_tm1.device)

        return total_loss * self.config.loss_scale, metrics

    def compute_loss(
        self,
        features: Tensor,
        continues: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        loss, _ = self.loss(features, continues)
        return loss

    def predict(self, feats_tm1: Tensor) -> Dict[int, Tensor]:
        with torch.no_grad():
            logits = self.forward(feats_tm1)
            probs = torch.sigmoid(logits)
        return {h: probs[..., h_idx] for h_idx, h in enumerate(self.horizons)}

    def get_survival_estimates(self, feats_tm1: Tensor) -> Tensor:
        probs = self.predict(feats_tm1)
        weights = torch.tensor([1.0 / h for h in self.horizons], device=feats_tm1.device)
        weights = weights / weights.sum()
        stacked = torch.stack([probs[h] for h in self.horizons], dim=-1)
        return (stacked * weights).sum(dim=-1)

    def compute_symmetric_loss(
        self,
        features_post: Tensor,
        features_prior: Tensor,
        continues: Tensor,
        prior_weight: float = 0.5,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss_post, metrics_post = self.loss(features_post, continues)
        loss_prior, _ = self.loss(features_prior, continues)
        loss = (1 - prior_weight) * loss_post + prior_weight * loss_prior
        info = {
            'mhc_loss_post': loss_post.detach(),
            'mhc_loss_prior': loss_prior.detach(),
            'mhc_loss': loss.detach(),
        }
        info.update(metrics_post)
        return loss, info


# =============================================================================
# 9 组合预测头
# =============================================================================

class PredictionHeads(nn.Module):
    """组合预测头reward + continue + vitals_decoder + mhc(可选)"""

    def __init__(self, config: PredictionHeadsConfig):
        super().__init__()

        self.config = config

        self.reward_head = SymlogRewardHead(feat_dim=config.feat_dim, config=config.reward)
        self.continue_head = ContinueHead(feat_dim=config.feat_dim, config=config.continue_)
        self.vitals_decoder = VitalsDecoder(
            feat_dim=config.feat_dim,
            vitals_dim=config.vitals_dim,
            config=config.decoder,
        )

        self.mhc_head: Optional[MultiHorizonContinueHead] = None
        if config.mhc is not None and config.mhc.enabled:
            mhc_config = config.mhc
            if mhc_config.feat_dim != config.feat_dim:
                warnings.warn(
                    f"MHC feat_dim ({mhc_config.feat_dim}) differs from "
                    f"PredictionHeads feat_dim ({config.feat_dim}). Using {config.feat_dim}.",
                    UserWarning,
                )
                from dataclasses import replace
                mhc_config = replace(mhc_config, feat_dim=config.feat_dim)
            self.mhc_head = MultiHorizonContinueHead(mhc_config)

    @property
    def has_mhc(self) -> bool:
        return self.mhc_head is not None

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        # FIX PERF: 避免调用 reward_head 两次
        reward_symlog = self.reward_head(features)
        reward_real = self.reward_head.symlog.symexp(reward_symlog)

        outputs: Dict[str, Tensor] = {
            'reward_symlog': reward_symlog,
            'reward_real': reward_real,
            'continue_logit': self.continue_head(features),
            'continue_prob': self.continue_head.get_prob(features),
            'vitals_recon': self.vitals_decoder(features),
        }

        if self.mhc_head is not None and features.dim() == 3:
            mhc_logits = self.mhc_head(features)
            outputs['mhc_logits'] = mhc_logits
            outputs['mhc_probs'] = torch.sigmoid(mhc_logits)

        return outputs

    def compute_losses(
        self,
        features_post: Tensor,
        features_prior: Tensor,
        targets: Dict[str, Tensor],
        prior_weight: float = 0.5,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        losses: Dict[str, Tensor] = {}
        info: Dict[str, Tensor] = {}

        if 'rewards' in targets:
            reward_loss, reward_info = self.reward_head.compute_symmetric_loss(
                features_post, features_prior, targets['rewards'],
                prior_weight=prior_weight, mask=mask,
            )
            losses['reward'] = reward_loss
            info.update(reward_info)

        if 'continues' in targets:
            continue_loss, continue_info = self.continue_head.compute_symmetric_loss(
                features_post, features_prior, targets['continues'],
                prior_weight=prior_weight, mask=mask,
            )
            losses['continue'] = continue_loss
            info.update(continue_info)

        if 'vitals' in targets:
            recon_loss, recon_info = self.vitals_decoder.compute_symmetric_loss(
                features_post, features_prior, targets['vitals'],
                prior_weight=prior_weight, mask=mask,
            )
            losses['recon'] = recon_loss
            info.update(recon_info)

        if self.mhc_head is not None and 'continues' in targets and features_post.dim() == 3:
            mhc_loss, mhc_info = self.mhc_head.compute_symmetric_loss(
                features_post, features_prior, targets['continues'],
                prior_weight=prior_weight, mask=mask,
            )
            losses['mhc'] = mhc_loss
            info.update(mhc_info)

        return losses, info

    def get_mhc_survival(self, features: Tensor) -> Optional[Tensor]:
        if self.mhc_head is None:
            return None
        return self.mhc_head.get_survival_estimates(features)


# ======================================================================
# MERGED FROM: router.py
# ======================================================================

"""
特征路由系统

包含: PolicyFeatures, UncertaintyEstimator, FeatureRouter, FeatureRouterSystem
"""


# =============================================================================
# PolicyFeatures 数据结构
# =============================================================================

@dataclass(frozen=True)
class PolicyFeatures:
    """
    路由系统输出的不可变封装
    契约: weights 始终为 [B, 3]表示 [x, ctrl, z] 三分支权重
    """
    f_policy: torch.Tensor
    v_x: Optional[torch.Tensor] = None
    v_c: Optional[torch.Tensor] = None
    v_z: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None

    def detach(self) -> "PolicyFeatures":
        def _d(t: Optional[Tensor]) -> Optional[Tensor]:
            return t.detach() if t is not None else None
        return PolicyFeatures(
            f_policy=self.f_policy.detach(),
            v_x=_d(self.v_x), v_c=_d(self.v_c), v_z=_d(self.v_z),
            weights=_d(self.weights), uncertainty=_d(self.uncertainty),
            mask=_d(self.mask),
        )

    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {
            'f_policy': self.f_policy, 'v_x': self.v_x, 'v_c': self.v_c,
            'v_z': self.v_z, 'weights': self.weights,
            'uncertainty': self.uncertainty, 'mask': self.mask,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "f_policy_shape": tuple(self.f_policy.shape),
            "weights_shape": tuple(self.weights.shape) if self.weights is not None else None,
            "uncertainty_mean": (
                float(self.uncertainty.detach().mean().item())
                if self.uncertainty is not None and self.uncertainty.numel() > 0 else None
            ),
            "mask_rate": (
                float(self.mask.detach().float().mean().item())
                if self.mask is not None and self.mask.numel() > 0 else None
            ),
        }


# =============================================================================
# UncertaintyEstimator
# =============================================================================

class UncertaintyEstimator(nn.Module):
    """生理级不确定性度量器"""

    def __init__(
        self,
        alpha: float = 0.5,
        quantile: float = 0.85,
        ema_decay: float = 0.99,
        hysteresis: Tuple[float, float] = (0.8, 1.2),
        window_size: int = 2000,
        min_samples_for_threshold: int = 500,
        warmup_steps: int = 200,
    ):
        super().__init__()
        self.alpha = alpha
        self.quantile = quantile
        self.ema_decay = ema_decay
        self.hyst_low, self.hyst_high = hysteresis
        self.window_size = window_size
        self.min_samples = min_samples_for_threshold
        self.warmup_steps = warmup_steps

        self.register_buffer('threshold', torch.tensor(0.0))
        self.register_buffer('mu_logvar', torch.tensor(0.0))
        self.register_buffer('mu_proj', torch.tensor(0.0))
        self.register_buffer('_history_buffer', torch.zeros(window_size))
        self.register_buffer('_history_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_history_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_global_step', torch.tensor(0, dtype=torch.long))

        self._active_mask: Optional[torch.Tensor] = None

    def _update_history(self, u_score: torch.Tensor) -> None:
        with torch.no_grad():
            ptr = self._history_ptr.item()
            self._history_buffer[ptr] = u_score.mean()
            self._history_ptr.fill_((ptr + 1) % self.window_size)
            if self._history_count < self.window_size:
                self._history_count.add_(1)
            self._global_step.add_(1)

    def _compute_dynamic_threshold(self) -> None:
        count = self._history_count.item()
        if count < self.min_samples:
            return
        self.threshold.fill_(torch.quantile(self._history_buffer[:count], self.quantile))

    def update_statistics(self, logvar: torch.Tensor, proj_err: torch.Tensor) -> None:
        with torch.no_grad():
            self.mu_logvar.lerp_(logvar.mean(), 1.0 - self.ema_decay)
            self.mu_proj.lerp_(proj_err.mean(), 1.0 - self.ema_decay)

    def reset_state(self) -> None:
        self._active_mask = None

    def forward(
        self,
        logvar: Optional[torch.Tensor],
        x_t: Optional[torch.Tensor],
        x_proj: Optional[torch.Tensor],
        reset: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if reset:
            self.reset_state()

        if x_t is not None:
            device, B = x_t.device, x_t.shape[0]
        elif logvar is not None:
            device = logvar.device
            B = logvar.shape[0] if logvar.dim() >= 1 else 1
        else:
            raise ValueError("At least one of logvar or x_t must be provided")

        # 计算 logvar 分数
        if logvar is not None:
            lv = logvar.detach()
            u_logvar = lv.unsqueeze(0) if lv.dim() == 0 else (lv if lv.dim() == 1 else lv.mean(dim=-1))
        else:
            u_logvar = torch.zeros(B, device=device)

        # 计算投影误差分数
        if x_t is not None and x_proj is not None:
            u_proj = F.mse_loss(x_t.detach(), x_proj.detach(), reduction='none').mean(dim=-1)
            if self.training:
                self.update_statistics(u_logvar, u_proj)
        else:
            u_proj = torch.zeros(B, device=device)

        u_score = (
            self.alpha * (u_logvar - self.mu_logvar) +
            (1 - self.alpha) * (u_proj - self.mu_proj)
        )

        if self.training:
            self._update_history(u_score)
            self._compute_dynamic_threshold()

        # warmup 期间不生成 mask
        if self._global_step.item() < self.warmup_steps:
            return u_score, torch.zeros(B, dtype=torch.bool, device=device)

        # 滞回 mask
        if self._active_mask is None or self._active_mask.shape[0] != B:
            self._active_mask = (u_score > self.threshold).clone()
        else:
            new_mask = self._active_mask.clone()
            new_mask[u_score > self.threshold * self.hyst_high] = True
            new_mask[u_score < self.threshold * self.hyst_low] = False
            self._active_mask = new_mask

        return u_score, self._active_mask.clone()

    def get_stats(self) -> Dict[str, float]:
        return {
            'threshold': self.threshold.item(),
            'mu_logvar': self.mu_logvar.item(),
            'mu_proj': self.mu_proj.item(),
            'history_count': self._history_count.item(),
            'global_step': self._global_step.item(),
            'in_warmup': self._global_step.item() < self.warmup_steps,
        }


# =============================================================================
# FeatureRouter
# =============================================================================

class FeatureRouter(nn.Module):
    """
    智能特征路由器
    输出契约: weights 始终为 [B, 3]表示 [x, ctrl, z] 三分支权重
    """

    NUM_BRANCHES: int = 3

    def __init__(
        self,
        d_x: int,
        d_c: int,
        d_z: int = 64,
        proj_dim: int = 256,
        mode: Literal["attention", "concat", "x_only", "ctrl_only"] = "attention",
        temp_range: Tuple[float, float] = (1.0, 0.1),
        anneal_steps: int = 50000,
        ema_decay: float = 0.99,
        ema_blend_train: float = 0.1,
        ema_blend_eval: float = 0.0,
        mask_x_factor: float = 0.3,
        mask_z_factor: float = 0.1,
        hard_fallback_on_high_uncertainty: bool = False,
        hard_fallback_weights: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        hard_fallback_weights_3branch: Optional[Tuple[float, float, float]] = None,
        hard_fallback_weights_2branch: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.mode = mode
        self.proj_dim = proj_dim
        self.d_z = d_z
        self.temp_range = temp_range
        self.anneal_steps = anneal_steps
        self.ema_decay = ema_decay
        self.ema_blend_train = ema_blend_train
        self.ema_blend_eval = ema_blend_eval

        hard_fb_w3 = hard_fallback_weights_3branch or hard_fallback_weights
        hard_fb_w2 = hard_fallback_weights_2branch or (float(hard_fb_w3[0]), float(hard_fb_w3[1]))

        self.hard_fallback_enabled = hard_fallback_on_high_uncertainty
        # FIX: 去掉重复的 _hard_fallback_w统一用 _hard_fallback_w3
        self.register_buffer('_hard_fallback_w3', torch.tensor(hard_fb_w3, dtype=torch.float32))
        self.register_buffer('_hard_fallback_w2', torch.tensor(hard_fb_w2, dtype=torch.float32))

        # FIX PERF: 预注册软干预因子和静态模式权重
        self.register_buffer(
            '_soft_intervention_factors',
            torch.tensor([mask_x_factor, 1.0, mask_z_factor], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer('_w_x_only', torch.tensor([[1.0, 0.0, 0.0]]), persistent=False)
        self.register_buffer('_w_ctrl_only', torch.tensor([[0.0, 1.0, 0.0]]), persistent=False)
        self.register_buffer('_w_concat', torch.tensor([[0.5, 0.5, 0.0]]), persistent=False)

        # 投影层
        self.proj_x = self._make_proj(d_x, proj_dim)
        self.proj_c = self._make_proj(d_c, proj_dim)
        self.proj_z = self._make_proj(d_z, proj_dim) if d_z > 0 else None

        # 门控网络
        if mode == "attention":
            self.gate_net = nn.Sequential(
                nn.Linear(proj_dim * 3, 128),
                nn.SiLU(),
                nn.Linear(128, self.NUM_BRANCHES),
            )
            nn.init.orthogonal_(self.gate_net[-1].weight, gain=0.01)
        else:
            self.gate_net = None

        # EMA
        self.register_buffer('_ema_w', torch.ones(self.NUM_BRANCHES) / self.NUM_BRANCHES)
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

        # 输出维度
        self._output_dim = proj_dim * 2 if mode == "concat" else proj_dim

    @staticmethod
    def _make_proj(in_d: int, out_d: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.SiLU())

    def _ensure_simplex(self, w: torch.Tensor) -> torch.Tensor:
        if w.ndim != 2:
            raise ValueError(f"weights must be rank-2 [B, K], got shape={tuple(w.shape)}")
        # 快速路径
        if torch.isfinite(w).all() and (w >= 0).all():
            if (w.sum(dim=-1) - 1.0).abs().max() <= 1e-6:
                return w
        # 修复路径
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w)).clamp_min(0.0)
        denom = w.sum(dim=-1, keepdim=True)
        uniform = 1.0 / w.shape[-1]
        return torch.where(denom > 0, w / denom, torch.full_like(w, uniform))

    @property
    def feature_dim(self) -> int:
        return self._output_dim

    def get_temp(self) -> float:
        t_start, t_end = self.temp_range
        frac = min(self._step.item() / max(self.anneal_steps, 1), 1.0)
        return t_start + frac * (t_end - t_start)

    def reset_ema_weights(self) -> None:
        self._ema_w.fill_(1.0 / self.NUM_BRANCHES)

    def forward(
        self,
        x_rl: torch.Tensor,
        s_ctrl: torch.Tensor,
        z_task: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        reset_ema: bool = False,
    ) -> PolicyFeatures:
        if reset_ema:
            self.reset_ema_weights()

        B = x_rl.shape[0]
        device = x_rl.device

        v_x = self.proj_x(x_rl)
        v_c = self.proj_c(s_ctrl)
        has_z = z_task is not None and self.proj_z is not None
        v_z = self.proj_z(z_task) if has_z else torch.zeros_like(v_x)

        # 简单模式
        if self.mode == "x_only":
            return PolicyFeatures(
                f_policy=v_x, v_x=v_x, v_c=v_c, v_z=v_z if has_z else None,
                weights=self._w_x_only.expand(B, -1),
            )
        if self.mode == "ctrl_only":
            return PolicyFeatures(
                f_policy=v_c, v_x=v_x, v_c=v_c, v_z=v_z if has_z else None,
                weights=self._w_ctrl_only.expand(B, -1),
            )
        if self.mode == "concat":
            return PolicyFeatures(
                f_policy=torch.cat([v_x, v_c], dim=-1),
                v_x=v_x, v_c=v_c, v_z=v_z if has_z else None,
                weights=self._w_concat.expand(B, -1),
            )

        # 动态注意力模式
        if self.training:
            self._step += 1

        temp = self.get_temp()
        w, w_final = self._compute_3branch_weights(v_x, v_c, v_z, mask, temp, B, device, has_z)

        v_stack = torch.stack([v_x, v_c, v_z], dim=1)
        f_policy = (v_stack * w_final.unsqueeze(-1)).sum(dim=1)

        return PolicyFeatures(
            f_policy=f_policy,
            v_x=v_x, v_c=v_c, v_z=v_z if has_z else None,
            weights=w_final,
            mask=mask,
        )

    def _compute_3branch_weights(
        self,
        v_x: Tensor, v_c: Tensor, v_z: Tensor,
        mask: Optional[Tensor],
        temp: float, B: int, device: torch.device,
        has_z: bool,
    ) -> Tuple[Tensor, Tensor]:
        gate_input = torch.cat([v_x, v_c, v_z], dim=-1)
        w = F.softmax(self.gate_net(gate_input) / temp, dim=-1)

        if not has_z:
            w = self._zero_z_branch(w)

        if mask is not None and mask.any():
            w = self._apply_uncertainty_intervention(w, mask)

        w_final = self._apply_ema_smoothing(w, mask=mask)
        return w, self._ensure_simplex(w_final)

    @staticmethod
    def _zero_z_branch(w: Tensor) -> Tensor:
        w_xy = w[:, :2]
        w_xy = w_xy / (w_xy.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.cat([w_xy, torch.zeros_like(w[:, 2:3])], dim=-1)

    def _apply_uncertainty_intervention(self, w: Tensor, mask: Tensor) -> Tensor:
        if self.hard_fallback_enabled:
            hard_w = self._hard_fallback_w3.unsqueeze(0).expand_as(w)
            return torch.where(mask.unsqueeze(-1), hard_w, w)

        # FIX PERF: 使用预注册的 buffer
        factors = self._soft_intervention_factors.unsqueeze(0).expand_as(w)
        w_adjusted = torch.where(mask.unsqueeze(-1), w * factors, w)
        return w_adjusted / (w_adjusted.sum(dim=-1, keepdim=True) + 1e-8)

    def _apply_ema_smoothing(self, w: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.training:
            self._ema_w.lerp_(w.detach().mean(dim=0), 1.0 - self.ema_decay)
            blend = self.ema_blend_train
        else:
            blend = self.ema_blend_eval

        w_final = (1.0 - blend) * w + blend * self._ema_w.unsqueeze(0) if blend > 0 else w

        if self.hard_fallback_enabled and mask is not None and mask.any():
            w_final = torch.where(mask.unsqueeze(-1), w, w_final)

        return w_final

    def get_stats(self) -> Dict[str, Any]:
        return {
            'temperature': self.get_temp(),
            'step': self._step.item(),
            'ema_w': self._ema_w.tolist(),
            'hard_fallback_enabled': self.hard_fallback_enabled,
            'num_branches': self.NUM_BRANCHES,
        }


# =============================================================================
# FeatureRouterSystem
# =============================================================================

class FeatureRouterSystem(nn.Module):
    """顶层封装整合不确定性估计器与特征路由器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.estimator = UncertaintyEstimator(
            alpha=config.get('uncertainty_alpha', 0.5),
            quantile=config.get('uncertainty_quantile', 0.85),
            ema_decay=config.get('uncertainty_ema_decay', 0.99),
            hysteresis=config.get('uncertainty_hysteresis', (0.8, 1.2)),
            window_size=config.get('uncertainty_window_size', 2000),
            warmup_steps=config.get('uncertainty_warmup_steps', 200),
        )

        self.router = FeatureRouter(
            d_x=config['d_x'],
            d_c=config['d_c'],
            d_z=config.get('d_z', 64),
            proj_dim=config.get('proj_dim', 256),
            mode=config.get('router_mode', 'attention'),
            temp_range=config.get('temp_range', (1.0, 0.1)),
            anneal_steps=config.get('anneal_steps', 50000),
            ema_blend_train=config.get('ema_blend_train', 0.1),
            ema_blend_eval=config.get('ema_blend_eval', 0.0),
            mask_x_factor=config.get('mask_x_factor', 0.3),
            mask_z_factor=config.get('mask_z_factor', 0.1),
            hard_fallback_on_high_uncertainty=config.get('hard_fallback_enabled', False),
            hard_fallback_weights=config.get('hard_fallback_weights', (0.0, 1.0, 0.0)),
            hard_fallback_weights_3branch=config.get('hard_fallback_weights_3branch', None),
            hard_fallback_weights_2branch=config.get('hard_fallback_weights_2branch', None),
        )

        self._config = config

    @property
    def feature_dim(self) -> int:
        return self.router.feature_dim

    def _run_estimator(
        self,
        logvar: Optional[Tensor],
        x_t: Optional[Tensor],
        x_proj: Optional[Tensor],
        reset: bool,
    ) -> Tuple[Tensor, Tensor]:
        return self.estimator(logvar=logvar, x_t=x_t, x_proj=x_proj, reset=reset)

    def _run_router(
        self,
        x_rl: Tensor,
        s_ctrl: Tensor,
        z_task: Optional[Tensor],
        mask: Tensor,
        reset_ema: bool,
    ) -> PolicyFeatures:
        return self.router(x_rl=x_rl, s_ctrl=s_ctrl, z_task=z_task, mask=mask, reset_ema=reset_ema)

    def _finalize_output(
        self,
        output: PolicyFeatures,
        u_score: Tensor,
        mask: Tensor,
        wall_strength: Optional[float],
    ) -> PolicyFeatures:
        if wall_strength is not None and output.weights is not None:
            output = _apply_wall_strength_to_policy_features(output, float(wall_strength))

        return PolicyFeatures(
            f_policy=output.f_policy,
            v_x=output.v_x, v_c=output.v_c, v_z=output.v_z,
            weights=output.weights,
            uncertainty=u_score,
            mask=mask,
        )

    def forward(
        self,
        state: Any,
        logvar: Optional[torch.Tensor] = None,
        reset: bool = False,
        wall_strength: Optional[float] = None,
    ) -> PolicyFeatures:
        u_score, mask = self._run_estimator(
            logvar=logvar,
            x_t=state.x_t,
            x_proj=getattr(state, 'x_proj', None),
            reset=reset,
        )

        x_rl = getattr(state, 'x_proj', None) or state.x_t

        output = self._run_router(
            x_rl=x_rl,
            s_ctrl=state.s_ctrl,
            z_task=getattr(state, 'z_task', None),
            mask=mask,
            reset_ema=reset,
        )

        return self._finalize_output(output, u_score, mask, wall_strength)

    def forward_components(
        self,
        x_rl: torch.Tensor,
        s_ctrl: torch.Tensor,
        z_task: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        x_proj: Optional[torch.Tensor] = None,
        reset: bool = False,
        wall_strength: Optional[float] = None,
    ) -> PolicyFeatures:
        if x_t is None:
            x_t = x_rl

        u_score, mask = self._run_estimator(logvar=logvar, x_t=x_t, x_proj=x_proj, reset=reset)
        output = self._run_router(x_rl=x_rl, s_ctrl=s_ctrl, z_task=z_task, mask=mask, reset_ema=reset)
        return self._finalize_output(output, u_score, mask, wall_strength)

    def get_stats(self) -> Dict[str, Any]:
        return {'estimator': self.estimator.get_stats(), 'router': self.router.get_stats()}

    def reset_state(self) -> None:
        self.estimator.reset_state()
        self.router.reset_ema_weights()


# =============================================================================
# Wall Strength 应用
# =============================================================================

def _apply_wall_strength_to_policy_features(
    features: PolicyFeatures,
    wall_strength: float,
) -> PolicyFeatures:
    ws = max(0.0, min(1.0, wall_strength))

    weights = features.weights
    if weights is None or weights.numel() == 0:
        return features

    if weights.dim() == 1:
        weights = weights.unsqueeze(0)

    # Validate and handle the router output expectations: we expect exactly 3 branches (v_x, v_c, v_z).
    # If the network emitted something else, we log a warning but pad/truncate to maintain functionality
    # rather than crashing out in production. The assertion clarifies the design constraint.
    if weights.shape[-1] != 3:
        warnings.warn(
            f"Expected router to produce 3 branch weights, got {weights.shape[-1]}. "
            f"Padding or truncating to shape 3. This indicates a misconfigured projection layer.", 
            UserWarning
        )
    if weights.shape[-1] < 3:
        weights = F.pad(weights, (0, 3 - weights.shape[-1]), value=0.0)
    elif weights.shape[-1] > 3:
        weights = weights[..., :3]

    B = weights.shape[0]
    device, dtype = weights.device, weights.dtype

    w_open = torch.tensor([0.3, 0.3, 0.4], device=device, dtype=dtype).expand(B, -1)
    w_closed = torch.tensor([0.5, 0.5, 0.0], device=device, dtype=dtype).expand(B, -1)

    if ws <= 0.5:
        alpha = ws / 0.5 if ws > 0.0 else 0.0
        w_eff = (1.0 - alpha) * w_open + alpha * weights
    else:
        alpha = (ws - 0.5) / 0.5
        w_eff = (1.0 - alpha) * weights + alpha * w_closed

    w_eff = w_eff / (w_eff.sum(dim=-1, keepdim=True) + 1e-8)

    # 尝试用新权重重新融合
    v_x, v_c, v_z = features.v_x, features.v_c, features.v_z

    if v_x is None or v_c is None:
        return PolicyFeatures(
            f_policy=features.f_policy, v_x=v_x, v_c=v_c, v_z=v_z,
            weights=w_eff, uncertainty=features.uncertainty, mask=features.mask,
        )

    if features.f_policy is not None and features.f_policy.shape[-1] != v_x.shape[-1]:
        return PolicyFeatures(
            f_policy=features.f_policy, v_x=v_x, v_c=v_c, v_z=v_z,
            weights=w_eff, uncertainty=features.uncertainty, mask=features.mask,
        )

    if v_z is None:
        v_z = torch.zeros_like(v_x)

    f_policy = (torch.stack([v_x, v_c, v_z], dim=1) * w_eff.unsqueeze(-1)).sum(dim=1)

    return PolicyFeatures(
        f_policy=f_policy, v_x=features.v_x, v_c=features.v_c, v_z=features.v_z,
        weights=w_eff, uncertainty=features.uncertainty, mask=features.mask,
    )


# =============================================================================
# 工厂函数
# =============================================================================

def create_router(
    d_x: int,
    d_c: int,
    d_z: int = 64,
    mode: str = "attention",
    proj_dim: int = 256,
    hard_fallback_enabled: bool = False,
    **kwargs,
) -> FeatureRouterSystem:
    config = {
        'd_x': d_x, 'd_c': d_c, 'd_z': d_z,
        'router_mode': mode, 'proj_dim': proj_dim,
        'hard_fallback_enabled': hard_fallback_enabled,
        **kwargs,
    }
    return FeatureRouterSystem(config)


# ======================================================================
# MERGED FROM: world_model/types.py
# ======================================================================

"""
WorldModel 类型定义

核心类型WMState, RSSMOutputs, WMTrajectory, ConsistencyReport,
          PredictiveLossPacket
"""


# =============================================================================
# 枚举类型
# =============================================================================

class WillVersion(Enum):
    NONE = auto()
    V52 = auto()


class RouterMode(Enum):
    STATIC = auto()
    SOFT = auto()
    META = auto()
    ADAPTIVE = auto()


class WallPhase(Enum):
    WARMUP = auto()
    TRANSITION = auto()
    ISOLATED = auto()


# =============================================================================
# WMState
# =============================================================================

class WMState(NamedTuple):
    """
    世界模型统一状态表示 (V5.2)

    梯度隔离约定
    - 策略侧统一走 isolate_gradients(...)
    - 旧 tuple 状态兼容统一收口到 coerce(...)
    """
    x_t: Tensor
    h_shared: Tensor
    h_pred: Tensor
    z: Tensor
    s_ctrl: Tensor
    x_proj: Optional[Tensor] = None
    z_task: Optional[Tensor] = None

    @property
    def s_pred(self) -> Tensor:
        return torch.cat([self.h_pred, self.z], dim=-1)

    @property
    def device(self) -> torch.device:
        return self.x_t.device

    @property
    def batch_size(self) -> int:
        return self.x_t.shape[0]

    def _detach_optional(self, t: Optional[Tensor]) -> Optional[Tensor]:
        return t.detach() if t is not None else None

    def detach_all(self) -> "WMState":
        """Completely detach all tensors, blocking all gradients."""
        return WMState(
            x_t=self.x_t.detach(), h_shared=self.h_shared.detach(),
            h_pred=self.h_pred.detach(), z=self.z.detach(), s_ctrl=self.s_ctrl.detach(),
            x_proj=self._detach_optional(self.x_proj),
            z_task=self._detach_optional(self.z_task),
        )

    def isolate_gradients(self, context: str, wall_strength: float = 1.0) -> "WMState":
        """Unified entry point for gradient isolation strategies.

        Parameters
        ----------
        context : {"rl", "policy", "all"}
            - "rl": Retains gradient on ``s_ctrl`` (for critic/actor training).
            - "policy": Mixes ``x_t`` gradient based on ``wall_strength``.
            - "all": Detaches everything.
        wall_strength : float
            Used only when context="policy". 1.0 means full detach.
        """
        ctx = context.lower()
        if ctx == "all":
            return self.detach_all()
        
        # Base detach for shared latents
        base_kwargs = dict(
            h_shared=self.h_shared.detach(),
            h_pred=self.h_pred.detach(),
            z=self.z.detach(),
            x_proj=self._detach_optional(self.x_proj),
            z_task=self._detach_optional(self.z_task),
        )

        if ctx == "rl":
            return WMState(x_t=self.x_t, s_ctrl=self.s_ctrl, **base_kwargs)
        elif ctx == "policy":
            if wall_strength >= 1.0:
                x_t_out = self.x_t.detach()
            elif wall_strength <= 0.0:
                x_t_out = self.x_t
            else:
                x_t_out = wall_strength * self.x_t.detach() + (1.0 - wall_strength) * self.x_t
            return WMState(x_t=x_t_out, s_ctrl=self.s_ctrl, **base_kwargs)
        else:
            raise ValueError(f"Unknown isolation context: {context}")

    @classmethod
    def from_legacy_state(
        cls,
        h: Tensor,
        z: Tensor,
        ctrl_dim: int,
        obs_embed_dim: Optional[int] = None,
    ) -> "WMState":
        B = h.shape[0]
        device = h.device
        if z.dim() == 3:
            z = z.reshape(B, -1)
        if obs_embed_dim is None:
            obs_embed_dim = h.shape[-1]
        return cls(
            x_t=torch.zeros(B, obs_embed_dim, device=device),
            h_shared=h, h_pred=h, z=z,
            s_ctrl=torch.zeros(B, ctrl_dim, device=device),
        )

    @classmethod
    def coerce(
        cls,
        state: Union["WMState", Tuple[Tensor, Tensor]],
        ctrl_dim: int,
        obs_embed_dim: Optional[int] = None,
    ) -> "WMState":
        """Canonical bridge while legacy tuple state inputs remain supported."""
        if isinstance(state, cls):
            return state
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError(
                "WMState.coerce expects WMState or legacy (h_shared, z) tuple."
            )
        h, z = state
        return cls.from_legacy_state(
            h,
            z,
            ctrl_dim=ctrl_dim,
            obs_embed_dim=obs_embed_dim,
        )

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        obs_embed_dim: int,
        deter_dim: int,
        stoch_dim: int,
        ctrl_dim: int,
        device: torch.device,
    ) -> "WMState":
        return cls(
            x_t=torch.zeros(batch_size, obs_embed_dim, device=device),
            h_shared=torch.zeros(batch_size, deter_dim, device=device),
            h_pred=torch.zeros(batch_size, deter_dim, device=device),
            z=torch.zeros(batch_size, stoch_dim, device=device),
            s_ctrl=torch.zeros(batch_size, ctrl_dim, device=device),
        )


# =============================================================================
# RSSMOutputs
# =============================================================================

@dataclass
class RSSMOutputs:
    """RSSM 分支完整输出 (V5.2)"""
    posterior: Any
    prior: Any
    z: Tensor
    o_pred: Optional[Tensor]
    r_pred: Tensor
    d_pred: Tensor
    decoder_logvar: Optional[Tensor] = None

    # Deprecated
    msc_logits: Optional[Dict[int, Tensor]] = field(default=None, repr=False)
    nst_logits: Optional[Tensor] = field(default=None, repr=False)
    sc_targets: Optional[Any] = field(default=None, repr=False)

    x_proj: Optional[Tensor] = None
    z_task: Optional[Tensor] = None
    s_ctrl_pred: Optional[Tensor] = None
    s_ctrl_prev: Optional[Tensor] = None
    kl_raw: Optional[Tensor] = None
    features_post: Optional[Tensor] = None
    features_prior: Optional[Tensor] = None


# =============================================================================
# WMTrajectory
# =============================================================================

@dataclass
class WMTrajectory:
    """
    世界模型轨迹数据包 (V5.2)

    维度约定
    - vitals: [B, T+1, V]
    - actions/rewards/dones: [B, T, ...]
    """
    vitals: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    h_seq: Tensor
    z_seq: Tensor
    feats_post: Tensor

    obs_embeds: Optional[Tensor] = None
    feats_prior: Optional[Tensor] = None
    kl_seq: Optional[Tensor] = None
    quant_loss: Optional[Tensor] = None
    remaining_steps: Optional[Tensor] = None
    validate_on_init: bool = field(default=True, repr=False)

    def __post_init__(self):
        if self.validate_on_init:
            self.validate()

    @property
    def batch_size(self) -> int:
        return self.vitals.shape[0]

    @property
    def seq_len(self) -> int:
        return self.actions.shape[1]

    @property
    def device(self) -> torch.device:
        return self.vitals.device

    @property
    def continues(self) -> Tensor:
        return 1.0 - self.dones.float()

    @property
    def vitals_aligned(self) -> Tensor:
        return self.vitals[:, 1:, :]

    def validate(self) -> None:
        B = self.vitals.shape[0]
        T = self.actions.shape[1]

        if self.vitals.shape[1] != T + 1:
            raise ValueError(
                f"vitals temporal dim mismatch: expected {T + 1}, got {self.vitals.shape[1]}"
            )

        for name, tensor in [
            ('actions', self.actions), ('rewards', self.rewards),
            ('dones', self.dones), ('h_seq', self.h_seq),
            ('z_seq', self.z_seq), ('feats_post', self.feats_post),
        ]:
            if tensor.shape[0] != B:
                raise ValueError(f"{name} batch size mismatch: expected {B}, got {tensor.shape[0]}")

        for name, tensor in [
            ('rewards', self.rewards), ('dones', self.dones),
            ('h_seq', self.h_seq), ('z_seq', self.z_seq),
            ('feats_post', self.feats_post),
        ]:
            if tensor.shape[1] != T:
                raise ValueError(f"{name} seq_len mismatch: expected {T}, got {tensor.shape[1]}")

    def slice_time(self, start: int, end_exclusive: int) -> "WMTrajectory":
        def _s(t: Optional[Any]) -> Optional[Any]:
            if t is None:
                return None
            if isinstance(t, Tensor) and t.dim() >= 2:
                return t[:, start:end_exclusive]
            return t

        return WMTrajectory(
            vitals=self.vitals[:, start:end_exclusive + 1],
            actions=self.actions[:, start:end_exclusive],
            rewards=self.rewards[:, start:end_exclusive],
            dones=self.dones[:, start:end_exclusive],
            h_seq=self.h_seq[:, start:end_exclusive],
            z_seq=self.z_seq[:, start:end_exclusive],
            feats_post=self.feats_post[:, start:end_exclusive],
            obs_embeds=_s(self.obs_embeds),
            feats_prior=_s(self.feats_prior),
            kl_seq=_s(self.kl_seq),
            quant_loss=self.quant_loss,
            remaining_steps=_s(self.remaining_steps),
            validate_on_init=False,
        )

    def to(self, device: torch.device) -> "WMTrajectory":
        def _to(t: Optional[Any]) -> Optional[Any]:
            return t.to(device) if isinstance(t, Tensor) else t

        return WMTrajectory(
            vitals=self.vitals.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            h_seq=self.h_seq.to(device),
            z_seq=self.z_seq.to(device),
            feats_post=self.feats_post.to(device),
            obs_embeds=_to(self.obs_embeds),
            feats_prior=_to(self.feats_prior),
            kl_seq=_to(self.kl_seq),
            quant_loss=_to(self.quant_loss),
            remaining_steps=_to(self.remaining_steps),
            validate_on_init=False,
        )

    @classmethod
    def from_seq_result(
        cls,
        seq_result: Dict[str, Any],
        vitals: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        remaining_steps: Optional[Tensor] = None,
    ) -> "WMTrajectory":
        feats_post = seq_result.get('feats_post')
        if feats_post is None:
            feats_post = seq_result.get('feats_prior')
        
        if feats_post is None:
            raise ValueError("seq_result must contain 'feats_post' or 'feats_prior'")

        h_seq = seq_result.get('h_seq')
        if h_seq is None:
            deter_dim = seq_result.get('deter_dim')
            if deter_dim is not None:
                h_seq = feats_post[..., :deter_dim]
            else:
                raise ValueError("Cannot infer h_seq: 'h_seq' and 'deter_dim' not in seq_result")

        z_seq = seq_result.get('z_seq')
        if z_seq is None:
            for key in ('z_post_raw_seq', 'z_prior_raw_seq'):
                if key in seq_result and seq_result[key] is not None:
                    z_seq = seq_result[key]
                    break
            if z_seq is None:
                raise ValueError("Cannot infer z_seq from seq_result")

        # 确保 z_seq 扁平化
        if z_seq.dim() == 4:
            z_seq = z_seq.reshape(z_seq.shape[0], z_seq.shape[1], -1)
        elif z_seq.dim() == 2:
            if actions.shape[1] != 1:
                raise ValueError(
                    f"z_seq is 2D but actions has T={actions.shape[1]} > 1."
                )
            z_seq = z_seq.unsqueeze(1)
        elif z_seq.dim() != 3:
            raise ValueError(f"Unexpected z_seq dimensions: {z_seq.dim()}")

        return cls(
            vitals=vitals, actions=actions, rewards=rewards, dones=dones,
            h_seq=h_seq, z_seq=z_seq, feats_post=feats_post,
            feats_prior=seq_result.get('feats_prior'),
            kl_seq=seq_result.get('kl_raw_seq'),
            quant_loss=seq_result.get('quant_loss'),
            remaining_steps=remaining_steps,
            validate_on_init=True,
        )


# =============================================================================
# ConsistencyReport
# =============================================================================

@dataclass
class ConsistencyReport:
    """一致性审计报告 (V5.2)"""
    mhc_loss: Tensor
    msc_loss: Tensor
    nst_loss: Tensor
    sc_loss: Tensor
    metrics: Dict[str, Any] = field(default_factory=dict)
    _device: Optional[torch.device] = field(default=None, repr=False)

    def __post_init__(self):
        if self._device is None:
            self._device = self.mhc_loss.device

    @property
    def total(self) -> Tensor:
        d = self._device
        return self.mhc_loss.to(d) + self.msc_loss.to(d) + self.nst_loss.to(d) + self.sc_loss.to(d)

    @property
    def breakdown(self) -> Dict[str, Tensor]:
        return {'mhc': self.mhc_loss, 'msc': self.msc_loss, 'nst': self.nst_loss, 'sc': self.sc_loss}

    @property
    def is_empty(self) -> bool:
        return self.total.abs().item() < 1e-8

    def to_dict(self) -> Dict[str, Any]:
        d = {f'consistency_{k}': v.detach() for k, v in self.breakdown.items()}
        d['consistency_total'] = self.total.detach()
        d.update(self.metrics)
        return d

    @classmethod
    def zeros(cls, device: torch.device) -> "ConsistencyReport":
        zero = torch.tensor(0.0, device=device)
        return cls(mhc_loss=zero, msc_loss=zero, nst_loss=zero, sc_loss=zero, _device=device)


# =============================================================================
# PredictiveLossPacket
# =============================================================================

@dataclass
class PredictiveLossPacket:
    """预测引擎的统一损失输出 (V5.2)"""
    recon: Tensor
    reward: Tensor
    continue_: Tensor
    kl: Tensor
    quant: Tensor
    projection: Tensor
    control_align: Tensor
    control_reg: Tensor
    abstractor: Tensor
    consistency_head: Tensor
    aux_inverse: Tensor
    aux_value: Tensor
    consistency: ConsistencyReport
    metrics: Dict[str, Any] = field(default_factory=dict)
    weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)

    @property
    def device(self) -> torch.device:
        return self.recon.device

    @property
    def core_loss(self) -> Tensor:
        w = self.weights
        return (
            w.recon * self.recon + w.reward * self.reward + w.continue_ * self.continue_ +
            w.kl * self.kl + w.quant * self.quant +
            w.projection * self.projection +
            w.control_align * self.control_align +
            w.control_reg * self.control_reg +
            w.abstractor * self.abstractor +
            w.consistency_head * self.consistency_head +
            w.aux_inverse * self.aux_inverse +
            w.aux_value * self.aux_value
        )

    @property
    def total(self) -> Tensor:
        return self.core_loss + self.weights.consistency * self.consistency.total.to(self.device)

    def to_dict(self) -> Dict[str, Tensor]:
        d = {
            'loss/recon': self.recon.detach(), 'loss/reward': self.reward.detach(),
            'loss/continue': self.continue_.detach(), 'loss/kl': self.kl.detach(),
            'loss/quant': self.quant.detach(), 'loss/projection': self.projection.detach(),
            'loss/control_align': self.control_align.detach(),
            'loss/control_reg': self.control_reg.detach(),
            'loss/abstractor': self.abstractor.detach(),
            'loss/consistency_head': self.consistency_head.detach(),
            'loss/aux_inverse': self.aux_inverse.detach(),
            'loss/aux_value': self.aux_value.detach(),
            'loss/core': self.core_loss.detach(), 'loss/total': self.total.detach(),
        }
        for name, loss in self.consistency.breakdown.items():
            d[f'loss/consistency_{name}'] = loss.detach()
        d['loss/consistency_total'] = self.consistency.total.detach()
        return d

    def to_metrics(self) -> Dict[str, float]:
        return {k: v.item() for k, v in self.to_dict().items()}

    @classmethod
    def zeros(cls, device: torch.device) -> "PredictiveLossPacket":
        zero = torch.tensor(0.0, device=device)
        return cls(
            recon=zero, reward=zero, continue_=zero, kl=zero, quant=zero,
            projection=zero, control_align=zero, control_reg=zero,
            abstractor=zero, consistency_head=zero,
            aux_inverse=zero, aux_value=zero,
            consistency=ConsistencyReport.zeros(device),
        )


# ======================================================================
# MERGED FROM: world_model/losses.py
# ======================================================================

"""
损失计算模块

核心组件ConsistencyAuditor, RemainingStepsComputer,
          NStepTerminationLoss, ShortcutConsistencyLoss
"""

from .aletheia_foundation import (
    ConsistencyAuditorConfig,
    MSCConfig,
    NSTConfig,
    SCConfig,
)


# =============================================================================
# 协议定义
# =============================================================================

class WorldModelProtocol(Protocol):
    def encode_action(self, action: Tensor) -> Tensor: ...
    @property
    def state_transition(self) -> Any: ...


class MSCHeadProtocol(Protocol):
    @property
    def config(self) -> MSCConfig: ...
    def __call__(self, features: Tensor) -> Dict[str, Any]: ...
    def compute_loss(self, features: Tensor, remaining_steps: Tensor,
                     true_danger: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Any]]: ...


class MHCHeadProtocol(Protocol):
    def loss(self, features: Tensor, continues: Tensor) -> Tuple[Tensor, Dict[str, Any]]: ...


# =============================================================================
# RemainingStepsComputer统一版本合并 losses + monitoring
# =============================================================================

class RemainingStepsComputer:
    """
    精确计算 remaining_steps 的辅助类向量化实现

    remaining_steps[t] = 从状态 t 到最近 done 的步数
    """

    @staticmethod
    def compute_from_dones(
        dones: Tensor,
        max_remaining: int = 100,
        warn_no_done: bool = True,
        no_done_warning_threshold: Optional[float] = None,
    ) -> Tensor:
        B, T = dones.shape
        device = dones.device

        positions = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)
        large_value = T + max_remaining

        done_positions = torch.where(
            dones > 0.5,
            positions,
            torch.full((B, T), large_value, device=device, dtype=torch.long),
        )

        # 从右向左 cummin 找最近 done
        nearest_done = done_positions.flip(1).cummin(dim=1).values.flip(1)
        remaining = (nearest_done - positions).clamp(min=0, max=max_remaining)

        if warn_no_done:
            no_done_ratio = (dones.sum(dim=1) == 0).float().mean().item()
            threshold = 0.5 if no_done_warning_threshold is None else float(no_done_warning_threshold)
            if no_done_ratio > threshold:
                logger.warning(
                    f"RemainingStepsComputer: {no_done_ratio * 100:.1f}% of sequences "
                    f"have no done signal. Using max_remaining={max_remaining}."
                )

        return remaining

    @staticmethod
    def compute_from_continues(
        continues: Tensor,
        max_remaining: int = 100,
    ) -> Tensor:
        return RemainingStepsComputer.compute_from_dones(
            1.0 - continues.float(), max_remaining, warn_no_done=False,
        )

    @staticmethod
    def compute_from_episode_info(
        episode_lengths: Tensor,
        step_indices: Tensor,
    ) -> Tensor:
        return (episode_lengths.unsqueeze(1) - step_indices).clamp(min=0)

    @staticmethod
    def compute_for_msc_training(
        dones: Tensor,
        episode_lengths: Optional[Tensor] = None,
        step_indices: Optional[Tensor] = None,
        max_remaining: int = 100,
    ) -> Tensor:
        if episode_lengths is not None and step_indices is not None:
            return RemainingStepsComputer.compute_from_episode_info(episode_lengths, step_indices)
        return RemainingStepsComputer.compute_from_dones(dones, max_remaining, warn_no_done=True)

    @staticmethod
    def validate_remaining_steps(
        remaining_steps: Tensor,
        dones: Tensor,
    ) -> Dict[str, Any]:
        issues: List[str] = []
        done_positions = dones > 0.5
        if done_positions.any() and (remaining_steps[done_positions] != 0).any():
            issues.append("remaining_steps at done positions should be 0")
        if (remaining_steps < 0).any():
            issues.append("remaining_steps contains negative values")
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "stats": {
                "mean": remaining_steps.float().mean().item(),
                "max": remaining_steps.max().item(),
                "min": remaining_steps.min().item(),
                "zeros_ratio": (remaining_steps == 0).float().mean().item(),
            },
        }


# =============================================================================
# 核心损失计算函数纯函数
# =============================================================================

def compute_mhc_loss(
    mhc_head: MHCHeadProtocol,
    features: Tensor,
    continues: Tensor,
    *,
    strict: bool = True,
) -> Tuple[Tensor, Dict[str, Any]]:
    try:
        loss, metrics = mhc_head.loss(features, continues)
        return loss, {f"mhc_{k}": v for k, v in metrics.items()}
    except Exception as e:
        if strict:
            raise RuntimeError(f"MHC auxiliary loss failed: {e}") from e
        logger.exception(f"MHC loss computation failed: {e}")
        return torch.tensor(0.0, device=features.device), {'mhc_failed': True}


def compute_msc_loss(
    msc_head: MSCHeadProtocol,
    features: Tensor,
    remaining_steps: Tensor,
    true_danger: Optional[Tensor] = None,
    *,
    strict: bool = True,
) -> Tuple[Tensor, Dict[str, Any]]:
    try:
        loss, metrics = msc_head.compute_loss(
            features=features, remaining_steps=remaining_steps, true_danger=true_danger,
        )
        return loss, {f"msc_{k}": v for k, v in metrics.items()}
    except Exception as e:
        if strict:
            raise RuntimeError(f"MSC auxiliary loss failed: {e}") from e
        logger.exception(f"MSC loss computation failed: {e}")
        return torch.tensor(0.0, device=features.device), {'msc_failed': True}


def compute_nst_loss(
    nst_module: "NStepTerminationLoss",
    termination_probs: Dict[int, Tensor],
    remaining_steps: Tensor,
    *,
    strict: bool = True,
) -> Tuple[Tensor, Dict[str, Any]]:
    try:
        loss, metrics = nst_module(termination_probs, remaining_steps)
        return loss, {f"nst_{k}": v for k, v in metrics.items()}
    except Exception as e:
        if strict:
            raise RuntimeError(f"NST auxiliary loss failed: {e}") from e
        logger.exception(f"NST loss computation failed: {e}")
        return torch.tensor(0.0, device=remaining_steps.device), {'nst_failed': True}


def compute_sc_loss(
    sc_module: "ShortcutConsistencyLoss",
    world_model: WorldModelProtocol,
    trajectory: WMTrajectory,
    *,
    strict: bool = True,
) -> Tuple[Tensor, Dict[str, Any]]:
    try:
        loss, metrics = sc_module.compute_loss(
            world_model=world_model,
            vitals=trajectory.vitals,
            actions=trajectory.actions,
            seq_result={'h_seq': trajectory.h_seq, 'z_seq': trajectory.z_seq},
            continues=trajectory.continues,
        )
        return loss, {f"sc_{k}": v for k, v in metrics.items()}
    except Exception as e:
        if strict:
            raise RuntimeError(f"SC auxiliary loss failed: {e}") from e
        logger.exception(f"SC loss computation failed: {e}")
        return torch.tensor(0.0, device=trajectory.device), {'sc_failed': True}


# =============================================================================
# NStepTerminationLoss
# =============================================================================

class NStepTerminationLoss(nn.Module):
    def __init__(self, config: NSTConfig):
        super().__init__()
        self.config = config
        self.n_steps = config.n_steps
        self.loss_scale = config.loss_scale
        self.use_exponential_weighting = config.use_exponential_weighting

    def forward(
        self,
        termination_probs: Dict[int, Tensor],
        remaining_steps: Tensor,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        device = remaining_steps.device
        total_loss = torch.tensor(0.0, device=device)
        metrics: Dict[str, Any] = {}
        count = 0

        for n in self.n_steps:
            if n not in termination_probs:
                continue

            prob_or_logits = termination_probs[n]
            if prob_or_logits.dim() == 3:
                prob_or_logits = prob_or_logits.squeeze(-1)

            target = (remaining_steps < n).float()
            is_logits = prob_or_logits.min() < -0.01 or prob_or_logits.max() > 1.01

            if is_logits:
                loss_n = F.binary_cross_entropy_with_logits(prob_or_logits, target, reduction='mean')
            else:
                loss_n = F.binary_cross_entropy(
                    prob_or_logits.clamp(1e-7, 1 - 1e-7), target, reduction='mean',
                )

            if self.use_exponential_weighting:
                loss_n = loss_n / (n ** 0.5)

            total_loss = total_loss + loss_n
            metrics[f'loss_n{n}'] = loss_n.detach()
            count += 1

        if count > 0:
            total_loss = total_loss / count

        metrics['active_horizons'] = torch.tensor(float(count), device=device)
        total_loss = total_loss * self.loss_scale
        metrics['total_loss'] = total_loss.detach()
        return total_loss, metrics


def _horizon_prob_dict_from_tensor(
    termination_probs: Tensor,
    horizons: Sequence[int],
) -> Dict[int, Tensor]:
    """Convert a packed horizon tensor into the mapping contract NST expects."""
    if termination_probs.dim() < 2:
        raise ValueError(
            "termination_probs tensor must expose horizons on the last dimension."
        )
    horizon_list = [int(h) for h in horizons]
    if termination_probs.shape[-1] != len(horizon_list):
        raise ValueError(
            "termination_probs horizon dimension does not match configured horizons: "
            f"{termination_probs.shape[-1]} vs {len(horizon_list)}"
        )
    return {
        horizon: termination_probs[..., idx]
        for idx, horizon in enumerate(horizon_list)
    }


# =============================================================================
# ShortcutConsistencyLoss
# =============================================================================

class ShortcutConsistencyLoss(nn.Module):
    def __init__(self, config: SCConfig):
        super().__init__()
        self.config = config
        self.horizons = config.horizons
        self.loss_scale = config.loss_scale
        self.detach_target = config.detach_target
        self.sample_ratio = float(getattr(config, 'sample_ratio', 1.0))
        self.max_starts = int(getattr(config, 'max_starts', 0))
        self.feature_loss_scale = float(getattr(config, 'feature_loss_scale', 0.0))

    def compute_loss(
        self,
        world_model: WorldModelProtocol,
        vitals: Tensor,
        actions: Tensor,
        seq_result: Dict[str, Any],
        continues: Tensor,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        device = vitals.device
        B, T = continues.shape
        h_seq = seq_result.get('h_seq')
        z_seq = seq_result.get('z_seq')
        feat_seq = seq_result.get('feats_post')

        if h_seq is None or z_seq is None:
            return torch.tensor(0.0, device=device), {}

        total_loss = torch.tensor(0.0, device=device)
        total_weight = 0.0
        metrics: Dict[str, Any] = {}

        for horizon in self.horizons:
            if horizon >= T:
                continue

            num_candidates = T - horizon
            k = num_candidates
            if self.sample_ratio < 1.0:
                k = max(1, int(num_candidates * self.sample_ratio))
            if self.max_starts > 0:
                k = min(k, self.max_starts)

            if k < num_candidates:
                t_iter = torch.randperm(num_candidates, device=device)[:k].tolist()
            else:
                t_iter = range(num_candidates)

            h_losses: List[Tensor] = []
            h_weights: List[Tensor] = []
            h_feature_losses: List[Tensor] = []

            for t in t_iter:
                t = int(t)
                valid_mask = continues[:, t:t + horizon].prod(dim=1)
                if valid_mask.sum() < 1e-8:
                    continue

                state = (h_seq[:, t], z_seq[:, t])
                for step in range(horizon):
                    action_embed = world_model.encode_action(actions[:, t + step])
                    out = world_model.state_transition.imagine_step(
                        prev_state=state, action_embed=action_embed,
                    )
                    state = self._get_next_state(out)

                h_target = h_seq[:, t + horizon]
                z_target = z_seq[:, t + horizon]
                if self.detach_target:
                    h_target, z_target = h_target.detach(), z_target.detach()

                h_pred, z_pred = state
                latent_loss = (
                    ((h_pred - h_target) ** 2).mean(dim=-1) +
                    ((z_pred.reshape(B, -1) - z_target.reshape(B, -1)) ** 2).mean(dim=-1)
                )
                sample_loss = latent_loss

                if self.feature_loss_scale > 0.0:
                    feat_pred = self._get_features(world_model, h_pred, z_pred)
                    if feat_seq is not None:
                        feat_target = feat_seq[:, t + horizon]
                    else:
                        feat_target = self._get_features(world_model, h_target, z_target)
                    if self.detach_target:
                        feat_target = feat_target.detach()
                    feature_loss = ((feat_pred - feat_target) ** 2).mean(dim=-1)
                    sample_loss = sample_loss + self.feature_loss_scale * feature_loss
                    h_feature_losses.append((feature_loss * valid_mask).sum())

                h_losses.append((sample_loss * valid_mask).sum())
                h_weights.append(valid_mask.sum())

            if h_losses:
                horizon_loss = torch.stack(h_losses).sum()
                horizon_weight = torch.stack(h_weights).sum()
                if horizon_weight > 1e-8:
                    normalized = horizon_loss / horizon_weight
                    total_loss = total_loss + normalized
                    total_weight += 1.0
                    metrics[f'loss_h{horizon}'] = normalized.detach()
                    if h_feature_losses:
                        metrics[f'feature_loss_h{horizon}'] = (
                            torch.stack(h_feature_losses).sum() / horizon_weight
                        ).detach()

        if total_weight > 0:
            total_loss = total_loss / total_weight

        total_loss = total_loss * self.loss_scale
        metrics['total_loss'] = total_loss.detach()
        return total_loss, metrics

    @staticmethod
    def _get_next_state(result: Any) -> Tuple[Tensor, Tensor]:
        state = getattr(result, 'next_state', None)
        if state is not None:
            return state
        for attr in ('state_for_next', 'state_next'):
            state = getattr(result, attr, None)
            if state is not None:
                return state
        h = getattr(result, 'h', None)
        z = getattr(result, 'z_prior_raw', None)
        if h is not None and z is not None:
            return (h, z)
        raise AttributeError(f"Cannot extract next state from {type(result).__name__}")

    @staticmethod
    def _get_features(world_model: WorldModelProtocol, h: Tensor, z_raw: Tensor) -> Tensor:
        get_features = getattr(world_model, 'get_features', None)
        if callable(get_features):
            return get_features(h, z_raw)
        return world_model.state_transition.get_features(h, z_raw)


# =============================================================================
# ConsistencyAuditor
# =============================================================================

class ConsistencyAuditor(nn.Module):
    """
    一致性审计器 (V5.2)

    注意子模块通过 __dict__ 存储不注册为 nn.Module 子模块
    因为它们由 WorldModel 持有
    """

    _EXTERNAL_MODULE_KEYS = frozenset({
        '_mhc_head', '_msc_head', '_nst_module', '_sc_module',
    })

    def __init__(
        self,
        config: ConsistencyAuditorConfig,
        mhc_head: Optional[nn.Module] = None,
        msc_head: Optional[nn.Module] = None,
        nst_module: Optional[NStepTerminationLoss] = None,
        sc_module: Optional[ShortcutConsistencyLoss] = None,
    ):
        super().__init__()
        self.config = config
        self.config.validate()

        self.__dict__['_mhc_head'] = mhc_head
        self.__dict__['_msc_head'] = msc_head
        self.__dict__['_nst_module'] = nst_module
        self.__dict__['_sc_module'] = sc_module
        self.__dict__['_danger_fn'] = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._EXTERNAL_MODULE_KEYS:
            raise AttributeError(
                f"{type(self).__name__}.{name} is an external module reference; "
                "use set_heads(...) to update it."
            )
        if isinstance(value, nn.Module):
            raise TypeError(
                f"{type(self).__name__} does not own submodules; "
                f"refused to set nn.Module attribute '{name}'."
            )
        super().__setattr__(name, value)

    def _validate_module_invariants(self) -> None:
        if self._modules:
            raise RuntimeError(
                f"{type(self).__name__} must not register submodules, "
                f"but found: {list(self._modules.keys())}"
            )

    # --- 属性 ---
    @property
    def mhc_enabled(self) -> bool:
        return self._mhc_head is not None and self.config.mhc.enabled

    @property
    def msc_enabled(self) -> bool:
        return self._msc_head is not None and self.config.msc.enabled

    @property
    def nst_enabled(self) -> bool:
        return self._nst_module is not None and self.config.nst.enabled

    @property
    def sc_enabled(self) -> bool:
        return self._sc_module is not None and self.config.sc.enabled

    @property
    def any_enabled(self) -> bool:
        return self.mhc_enabled or self.msc_enabled or self.nst_enabled or self.sc_enabled

    @property
    def strict_auxiliary_losses(self) -> bool:
        return bool(getattr(self.config, "strict_auxiliary_losses", True))

    # --- 配置 ---
    def set_danger_fn(self, fn: Callable[[Tensor], Tensor]) -> None:
        self.__dict__['_danger_fn'] = fn

    def set_heads(self, **kwargs) -> None:
        for key in ('mhc_head', 'msc_head', 'nst_module', 'sc_module'):
            if key in kwargs and kwargs[key] is not None:
                self.__dict__[f'_{key}'] = kwargs[key]

    # --- 审计接口 ---
    def audit(
        self,
        trajectory: WMTrajectory,
        world_model_ref: Optional[WorldModelProtocol] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> ConsistencyReport:
        self._validate_module_invariants()

        device = trajectory.device
        weights = weights or {}

        if not self.any_enabled:
            return ConsistencyReport.zeros(device)

        remaining_steps = self._ensure_remaining_steps(trajectory) if self._needs_remaining_steps() else None
        metrics: Dict[str, Any] = {}

        mhc_loss = self._compute_mhc(trajectory, metrics, weights)
        msc_loss = self._compute_msc(trajectory, remaining_steps, metrics, weights)
        nst_loss = self._compute_nst(trajectory, remaining_steps, metrics, weights)
        sc_loss = self._compute_sc(trajectory, world_model_ref, metrics, weights)

        return ConsistencyReport(
            mhc_loss=mhc_loss, msc_loss=msc_loss, nst_loss=nst_loss, sc_loss=sc_loss,
            metrics=metrics, _device=device,
        )

    def forward(self, trajectory: WMTrajectory, **kwargs) -> ConsistencyReport:
        return self.audit(trajectory, **kwargs)

    # --- 内部方法 ---
    def _needs_remaining_steps(self) -> bool:
        # remaining_steps 目前只被 MSC / NST 路径消费，而 NST 也依赖 MSC 输出。
        return self.msc_enabled

    def _ensure_remaining_steps(self, trajectory: WMTrajectory) -> Tensor:
        if trajectory.remaining_steps is not None:
            return trajectory.remaining_steps
        return RemainingStepsComputer.compute_from_dones(
            dones=trajectory.dones,
            max_remaining=self.config.max_horizon + self.config.remaining_steps_margin,
            no_done_warning_threshold=self.config.no_done_warning_threshold,
        ).to(trajectory.device)

    def _compute_mhc(self, trajectory, metrics, weights) -> Tensor:
        if not self.mhc_enabled:
            return torch.tensor(0.0, device=trajectory.device)
        loss, m = compute_mhc_loss(
            self._mhc_head,
            trajectory.feats_post,
            trajectory.continues,
            strict=self.strict_auxiliary_losses,
        )
        metrics.update(m)
        return loss * weights.get('mhc', self.config.mhc.loss_scale)

    def _compute_msc(self, trajectory, remaining_steps: Optional[Tensor], metrics, weights) -> Tensor:
        if not self.msc_enabled:
            return torch.tensor(0.0, device=trajectory.device)
        if remaining_steps is None:
            if self.strict_auxiliary_losses:
                raise RuntimeError("MSC auxiliary loss failed: remaining_steps unavailable")
            return torch.tensor(0.0, device=trajectory.device)
        danger = self._compute_danger(trajectory)
        loss, m = compute_msc_loss(
            self._msc_head,
            trajectory.feats_post,
            remaining_steps,
            danger,
            strict=self.strict_auxiliary_losses,
        )
        metrics.update(m)
        return loss * weights.get('msc', self.config.msc.loss_scale)

    def _compute_nst(self, trajectory, remaining_steps: Optional[Tensor], metrics, weights) -> Tensor:
        device = trajectory.device
        if not self.nst_enabled:
            return torch.tensor(0.0, device=device)
        if not self.msc_enabled or self._msc_head is None:
            if self.strict_auxiliary_losses:
                raise RuntimeError("NST auxiliary loss failed: MSC support is required")
            return torch.tensor(0.0, device=device)
        if remaining_steps is None:
            if self.strict_auxiliary_losses:
                raise RuntimeError("NST auxiliary loss failed: remaining_steps unavailable")
            return torch.tensor(0.0, device=device)
        try:
            msc_output = self._msc_head(trajectory.feats_post)
            term_probs = msc_output.get('termination_prob_dict')
            if term_probs is None:
                raw_term_probs = msc_output.get('termination_probs')
                if raw_term_probs is None:
                    if self.strict_auxiliary_losses:
                        raise RuntimeError(
                            "NST auxiliary loss failed: MSC output missing termination probabilities"
                        )
                    return torch.tensor(0.0, device=device)
                term_probs = self._msc_head.build_termination_prob_dict(raw_term_probs)
            if not term_probs:
                if self.strict_auxiliary_losses:
                    raise RuntimeError(
                        "NST auxiliary loss failed: MSC termination probability dict is empty"
                    )
                return torch.tensor(0.0, device=device)
            loss, m = compute_nst_loss(
                self._nst_module,
                term_probs,
                remaining_steps,
                strict=self.strict_auxiliary_losses,
            )
            metrics.update(m)
            return loss * weights.get('nst', self.config.nst.loss_scale)
        except Exception as e:
            if self.strict_auxiliary_losses:
                if isinstance(e, RuntimeError) and "NST auxiliary loss failed" in str(e):
                    raise
                raise RuntimeError(f"NST auxiliary loss failed: {e}") from e
            logger.warning(f"NST computation failed: {e}")
            return torch.tensor(0.0, device=device)

    def _compute_sc(self, trajectory, world_model_ref, metrics, weights) -> Tensor:
        if not self.sc_enabled:
            return torch.tensor(0.0, device=trajectory.device)
        if world_model_ref is None:
            if self.strict_auxiliary_losses:
                raise RuntimeError("SC auxiliary loss failed: world_model_ref is required")
            return torch.tensor(0.0, device=trajectory.device)
        loss, m = compute_sc_loss(
            self._sc_module,
            world_model_ref,
            trajectory,
            strict=self.strict_auxiliary_losses,
        )
        metrics.update(m)
        return loss * weights.get('sc', self.config.sc.loss_scale)

    def _compute_danger(self, trajectory: WMTrajectory) -> Optional[Tensor]:
        fn = self.__dict__.get('_danger_fn')
        if fn is None:
            return None
        if trajectory.vitals is None:
            if self.strict_auxiliary_losses:
                raise RuntimeError("Danger signal computation failed: trajectory.vitals is unavailable")
            logger.warning("Danger signal unavailable: trajectory.vitals is None")
            return None
        try:
            danger = fn(trajectory.vitals[:, 1:])
            return danger.unsqueeze(-1) if danger.dim() == 2 else danger
        except Exception as exc:
            if self.strict_auxiliary_losses:
                raise RuntimeError(f"Danger signal computation failed: {exc}") from exc
            logger.warning("Danger signal computation failed; continuing without danger target: %s", exc)
            return None

    def get_info(self) -> Dict[str, Any]:
        return {
            'mhc_enabled': self.mhc_enabled, 'msc_enabled': self.msc_enabled,
            'nst_enabled': self.nst_enabled, 'sc_enabled': self.sc_enabled,
            'max_horizon': self.config.max_horizon,
            'has_danger_fn': self.__dict__.get('_danger_fn') is not None,
        }


# =============================================================================
# 核心损失函数用于 PredictiveEngine
# =============================================================================

def compute_reward_loss(
    pred_symlog: Tensor, target_symlog: Tensor, mask: Optional[Tensor] = None,
) -> Tensor:
    loss = (pred_symlog - target_symlog) ** 2
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


def compute_continue_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    *,
    temperature: float = 1.0,
    loss_type: str = 'bce',
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
) -> Tensor:
    targets = targets.float()
    scaled_logits = logits / max(float(temperature), 1e-6)
    bce = F.binary_cross_entropy_with_logits(scaled_logits, targets, reduction='none')

    if str(loss_type).lower() == 'focal':
        probs = torch.sigmoid(scaled_logits).clamp(1e-7, 1 - 1e-7)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** float(focal_gamma)
        alpha = float(focal_alpha)
        alpha_weight = (1.0 - alpha) * targets + alpha * (1.0 - targets)
        loss = alpha_weight * focal_weight * bce
    else:
        loss = bce

    class_weight = float(positive_weight) * targets + float(negative_weight) * (1.0 - targets)
    loss = loss * class_weight

    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


def compute_recon_loss(
    pred: Tensor, target: Tensor, mask: Optional[Tensor] = None,
) -> Tensor:
    loss = (pred - target) ** 2
    if mask is not None:
        if mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        loss = loss * mask
        return loss.sum() / (mask.sum() * loss.shape[-1] + 1e-8)
    return loss.mean()



# ======================================================================
# Part 2: World Model Components & Core
# ======================================================================

# 注意此部分依赖于Part 1中定义的基础模块类型和工具函数
# 在完整运行时请确保Part 1的代码在此部分之前加载

pass # from .types import (
#     WMState, RSSMOutputs, WillVersion,
#     WMTrajectory, ConsistencyReport, PredictiveLossPacket,
#     LossWeights,
# )
pass # from .losses import (
#     compute_reward_loss, compute_continue_loss, compute_recon_loss,
#     ConsistencyAuditor,
# )
from .aletheia_foundation import (
    WorldModelConfig, LossWeightsConfig, ConsistencyAuditorConfig,
    ControlConfig, ProjectionConfig, AbstractorConfig,
    ConsistencyHeadConfig, AuxTasksConfig, PredictiveEngineConfig,
    build_swiglu_mlp, create_norm, scale_gradient,
)


# =============================================================================
# 辅助函数
# =============================================================================

def _flatten_if_needed(z: Tensor) -> Tensor:
    """扁平化 z 张量"""
    if not isinstance(z, Tensor):
        raise TypeError(
            f"_flatten_if_needed expects Tensor, got {type(z)}. "
            f"If z is a distribution object, extract .sample() or .mode first."
        )
    if z.dim() == 2:
        return z
    elif z.dim() == 3:
        return z.reshape(z.shape[0], -1)  # [B, N, K] -> [B, N*K]
    elif z.dim() == 4:
        B = z.shape[0]
        return z.reshape(B, -1)  # [B, T, N, K] -> [B, T*N*K]
    else:
        raise ValueError(f"Unexpected z dimensions: {z.dim()}.")


# =============================================================================
# RSSMBranch: RSSM 适配器
# =============================================================================

class RSSMBranch(nn.Module):
    """
    RSSM 适配器包装现有 StateTransition + Heads
    
    职责
    - 封装状态转移逻辑
    - 提供 observe_step / imagine_step 接口
    - 构建 RSSMOutputs
    """

    def __init__(
        self,
        state_transition: nn.Module,
        reward_head: nn.Module,
        continue_head: nn.Module,
        decoder: Optional[nn.Module] = None,
        msc_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.st = state_transition
        self.reward_head = reward_head
        self.continue_head = continue_head
        self.decoder = decoder
        self.msc_head = msc_head
        
        config = self.st.config
        self._deter_dim: int = config.deter_dim
        dist = config.distribution
        self._stoch_dim: int = dist.num_distributions * dist.num_classes
        self._feat_dim: int = getattr(config, "feat_dim", self._deter_dim + self._stoch_dim)

    @property
    def deter_dim(self) -> int:
        return self._deter_dim

    @property
    def stoch_dim(self) -> int:
        return self._stoch_dim

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def _get_next_state(self, result: Any) -> Tuple[Tensor, Tensor]:
        """获取下一状态"""
        state = getattr(result, 'next_state', None)
        if state is not None:
            return state
        for attr in ('state_for_next', 'state_next'):
            state = getattr(result, attr, None)
            if state is not None:
                return state
        
        h = getattr(result, 'h', None)
        z = getattr(result, 'z_post_raw', None)
        if z is None:
            z = getattr(result, 'z_prior_raw', None)
        
        if h is not None and z is not None:
            return (h, z)
        
        raise AttributeError(
            f"Cannot extract next state from {type(result).__name__}. "
            f"Expected attributes: state_for_next, state_next, or (h, z_*_raw)"
        )

    def _build_outputs(
        self,
        result: Any,
        feats: Tensor,
        is_imagine: bool = False,
    ) -> RSSMOutputs:
        """构建 RSSMOutputs"""
        r_pred = self.reward_head(feats)
        d_pred = self.continue_head(feats)
        o_pred = self.decoder(feats) if self.decoder else None
        
        z_post_raw = getattr(result, 'z_post_raw', None)
        if z_post_raw is None:
            z_post_raw = getattr(result, 'z_prior_raw', None)
        z_prior_raw = getattr(result, 'z_prior_raw', None)
        
        if z_post_raw is None:
            raise AttributeError(
                f"TransitionOutput missing z_post_raw/z_prior_raw. "
                f"Available: {[a for a in dir(result) if not a.startswith('_')]}"
            )
        
        z_post = _flatten_if_needed(z_post_raw)
        z_prior = _flatten_if_needed(z_prior_raw) if z_prior_raw is not None else z_post
        
        return RSSMOutputs(
            posterior=z_post_raw,
            prior=z_prior_raw,
            z=z_post if not is_imagine else z_prior,
            o_pred=o_pred,
            decoder_logvar=None,
            r_pred=r_pred,
            d_pred=d_pred,
            x_proj=None,
            z_task=None,
            s_ctrl_pred=None,
            s_ctrl_prev=None,
            kl_raw=getattr(result, 'kl_raw', None) if not is_imagine else None,
            features_post=feats if not is_imagine else None,
            features_prior=feats if is_imagine else None,
        )

    def observe_step(
        self,
        prev_state: Tuple[Tensor, Tensor],
        action_embed: Tensor,
        obs_embed: Tensor,
        **kwargs,
    ) -> Tuple[RSSMOutputs, Tuple[Tensor, Tensor]]:
        """单步观测转移"""
        result = self.st.observe_step(prev_state, action_embed, obs_embed, **kwargs)
        outputs = self._build_outputs(result, result.features_post, is_imagine=False)
        next_state = self._get_next_state(result)
        return outputs, next_state

    def imagine_step(
        self,
        prev_state: Tuple[Tensor, Tensor],
        action_embed: Tensor,
        **kwargs,
    ) -> Tuple[RSSMOutputs, Tuple[Tensor, Tensor]]:
        """单步想象转移"""
        result = self.st.imagine_step(prev_state, action_embed, **kwargs)
        outputs = self._build_outputs(result, result.features_prior, is_imagine=True)
        next_state = self._get_next_state(result)
        return outputs, next_state
    
    def get_features(self, h: Tensor, z_raw: Tensor) -> Tensor:
        """从状态构建特征"""
        return self.st.get_features(h, z_raw)


# =============================================================================
# PredictiveEngine: 预测引擎
# =============================================================================

class PredictiveEngine(nn.Module):
    """
    自纠错预测引擎 (V5.2)
    
    职责
    1. 封装 RSSM 状态转移
    2. 封装所有预测头
    3. 封装一致性审计
    4. 提供统一的 compute_loss() 接口
    """

    def __init__(
        self,
        state_transition: nn.Module,
        reward_head: nn.Module,
        continue_head: nn.Module,
        decoder: nn.Module,
        symlog: nn.Module,
        config: Optional[PredictiveEngineConfig] = None,
        msc_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.rssm_branch = RSSMBranch(
            state_transition=state_transition,
            reward_head=reward_head,
            continue_head=continue_head,
            decoder=decoder,
            msc_head=msc_head,
        )
        
        self.symlog = symlog
        self.config = config or PredictiveEngineConfig()
        
        # 一致性审计器不注册
        self.__dict__['_consistency_auditor'] = None
        self.__dict__['_danger_fn'] = None
        
        st_config = state_transition.config
        self._deter_dim = st_config.deter_dim
        dist = st_config.distribution
        self._stoch_dim = dist.num_distributions * dist.num_classes
        self._feat_dim = getattr(st_config, "feat_dim", self._deter_dim + self._stoch_dim)

    @property
    def deter_dim(self) -> int:
        return self._deter_dim

    @property
    def stoch_dim(self) -> int:
        return self._stoch_dim

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    @property
    def consistency_auditor(self) -> Optional[ConsistencyAuditor]:
        return self.__dict__.get('_consistency_auditor')

    @property
    def has_consistency_auditor(self) -> bool:
        return self.__dict__.get('_consistency_auditor') is not None

    def set_consistency_auditor(self, auditor: ConsistencyAuditor) -> None:
        self.__dict__['_consistency_auditor'] = auditor
        fn = self.__dict__.get('_danger_fn')
        if fn is not None:
            auditor.set_danger_fn(fn)

    def set_danger_fn(self, fn: Callable[[Tensor], Tensor]) -> None:
        self.__dict__['_danger_fn'] = fn
        auditor = self.__dict__.get('_consistency_auditor')
        if auditor is not None:
            auditor.set_danger_fn(fn)

    def observe_step(
        self,
        prev_state: Tuple[Tensor, Tensor],
        action_embed: Tensor,
        obs_embed: Tensor,
        **kwargs,
    ) -> Tuple[RSSMOutputs, Tuple[Tensor, Tensor]]:
        return self.rssm_branch.observe_step(prev_state, action_embed, obs_embed, **kwargs)

    def imagine_step(
        self,
        prev_state: Tuple[Tensor, Tensor],
        action_embed: Tensor,
        **kwargs,
    ) -> Tuple[RSSMOutputs, Tuple[Tensor, Tensor]]:
        return self.rssm_branch.imagine_step(prev_state, action_embed, **kwargs)

    def get_features(self, h: Tensor, z_raw: Tensor) -> Tensor:
        return self.rssm_branch.get_features(h, z_raw)

    def build_trajectory(
        self,
        seq_result: Dict[str, Any],
        vitals: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        remaining_steps: Optional[Tensor] = None,
    ) -> WMTrajectory:
        return WMTrajectory.from_seq_result(
            seq_result=seq_result,
            vitals=vitals,
            actions=actions,
            rewards=rewards,
            dones=dones,
            remaining_steps=remaining_steps,
        )

    def compute_loss(
        self,
        trajectory: WMTrajectory,
        weights: Optional[LossWeightsConfig] = None,
        world_model_ref: Optional[Any] = None,
    ) -> PredictiveLossPacket:
        """
        计算预测引擎的所有损失
        
        权重来源规则
        - 如果传入 LossWeightsConfig 从 config 提取
        - 否则使用默认权重
        """
        device = trajectory.device
        metrics: Dict[str, Any] = {}

        if weights is not None:
            core_weights = weights
            consistency_weights = {
                'mhc': weights.mhc, 'msc': weights.msc,
                'nst': weights.nst, 'sc': weights.sc,
            }
        else:
            core_weights = LossWeightsConfig()
            consistency_weights = {}

        feats = trajectory.feats_prior if trajectory.feats_prior is not None else trajectory.feats_post

        reward_loss = self._compute_reward_loss(trajectory, feats, metrics)
        continue_loss = self._compute_continue_loss(trajectory, feats, metrics)
        recon_loss = self._compute_recon_loss(trajectory, feats, metrics)
        kl_loss = self._compute_kl_loss(trajectory, metrics)
        quant_loss = self._compute_quant_loss(trajectory, device)

        bridge_losses = self._compute_bridge_losses(
            trajectory=trajectory,
            world_model_ref=world_model_ref,
            weights=core_weights,
            metrics=metrics,
        )

        consistency = self._compute_consistency_loss(
            trajectory=trajectory,
            world_model_ref=world_model_ref,
            consistency_weights=consistency_weights,
            metrics=metrics,
        )

        return PredictiveLossPacket(
            recon=recon_loss,
            reward=reward_loss,
            continue_=continue_loss,
            kl=kl_loss,
            quant=quant_loss,
            projection=bridge_losses["projection"],
            control_align=bridge_losses["control_align"],
            control_reg=bridge_losses["control_reg"],
            abstractor=bridge_losses["abstractor"],
            consistency_head=bridge_losses["consistency_head"],
            aux_inverse=bridge_losses["aux_inverse"],
            aux_value=bridge_losses["aux_value"],
            consistency=consistency,
            metrics=metrics,
            weights=core_weights,
        )

    def _loss_mask(self, trajectory: WMTrajectory) -> Tensor:
        continues = trajectory.continues
        B, T = continues.shape
        ones = torch.ones((B, 1), device=continues.device, dtype=continues.dtype)
        shifted = torch.cat([ones, continues[:, :-1]], dim=1)
        return shifted.cumprod(dim=1)

    def _compute_reward_loss(
        self,
        trajectory: WMTrajectory,
        feats: Tensor,
        metrics: Dict[str, Any],
    ) -> Tensor:
        pred_reward = self.rssm_branch.reward_head(feats)
        if pred_reward.dim() > 2:
            pred_reward = pred_reward.squeeze(-1)

        target_reward = self.symlog.symlog(trajectory.rewards)

        mask = self._loss_mask(trajectory)
        loss = compute_reward_loss(pred_reward, target_reward, mask=mask)

        metrics['reward_pred_mean'] = pred_reward.mean().detach()
        metrics['reward_target_mean'] = target_reward.mean().detach()
        return loss

    def _compute_continue_loss(
        self,
        trajectory: WMTrajectory,
        feats: Tensor,
        metrics: Dict[str, Any],
    ) -> Tensor:
        continue_head = self.rssm_branch.continue_head
        continue_logits = continue_head(feats)
        if continue_logits.dim() > 2:
            continue_logits = continue_logits.squeeze(-1)

        mask = self._loss_mask(trajectory)
        loss = continue_head.compute_loss_from_logits(continue_logits, trajectory.continues, mask=mask)

        continue_temp = max(float(getattr(continue_head.config, 'temperature', 1.0)), 1e-6)
        continue_probs = torch.sigmoid(continue_logits / continue_temp)

        metrics['continue_prob_mean'] = continue_probs.mean().detach()
        metrics['continue_target_mean'] = trajectory.continues.mean().detach()
        return loss

    def _compute_recon_loss(
        self,
        trajectory: WMTrajectory,
        feats: Tensor,
        metrics: Dict[str, Any],
    ) -> Tensor:
        if self.rssm_branch.decoder is None:
            return torch.tensor(0.0, device=trajectory.device)
        
        recon_pred = self.rssm_branch.decoder(feats)
        recon_target = trajectory.vitals_aligned
        
        mask = self._loss_mask(trajectory)
        loss = compute_recon_loss(recon_pred, recon_target, mask=mask)
        metrics['recon_error'] = loss.detach()
        return loss

    def _compute_kl_loss(
        self,
        trajectory: WMTrajectory,
        metrics: Dict[str, Any],
    ) -> Tensor:
        kl_seq = trajectory.kl_seq
        device = trajectory.device
        
        if kl_seq is None or not isinstance(kl_seq, Tensor):
            loss = torch.tensor(0.0, device=device)
            raw_loss = torch.tensor(0.0, device=device)
            if not getattr(self, "_warned_missing_kl_seq", False):
                logger.warning(
                    "WMTrajectory.kl_seq is missing; KL loss is forced to 0.0. "
                    "This can destabilize RSSM training."
                )
                self._warned_missing_kl_seq = True
            metrics['kl_missing'] = 1.0
        else:
            mask = self._loss_mask(trajectory).to(dtype=kl_seq.dtype)
            free_nats = float(
                getattr(getattr(self.rssm_branch.st.config, "kl", None), "free_nats", 0.0) or 0.0
            )
            raw_loss = (kl_seq * mask).sum() / (mask.sum() + 1e-8)
            if free_nats > 0.0:
                kl_seq_clamped = kl_seq.clamp(min=free_nats)
            else:
                kl_seq_clamped = kl_seq
            loss = (kl_seq_clamped * mask).sum() / (mask.sum() + 1e-8)
            metrics['kl_missing'] = 0.0
        
        metrics['kl_mean'] = loss.detach()
        metrics['kl_raw_mean'] = raw_loss.detach()
        metrics['kl_free_nats'] = float(
            getattr(getattr(self.rssm_branch.st.config, "kl", None), "free_nats", 0.0) or 0.0
        )
        return loss

    def _compute_quant_loss(
        self,
        trajectory: WMTrajectory,
        device: torch.device,
    ) -> Tensor:
        if trajectory.quant_loss is None:
            return torch.tensor(0.0, device=device)
        
        quant_loss = trajectory.quant_loss
        if isinstance(quant_loss, Tensor):
            return quant_loss
        return torch.tensor(float(quant_loss), device=device)

    def _compute_discounted_returns(
        self,
        rewards: Tensor,
        dones: Tensor,
        gamma: float = 0.99,
    ) -> Tensor:
        returns = torch.zeros_like(rewards)
        bootstrap = torch.zeros_like(rewards[:, 0])
        dones_f = dones.to(dtype=rewards.dtype)
        for idx in reversed(range(rewards.shape[1])):
            bootstrap = rewards[:, idx] + gamma * (1.0 - dones_f[:, idx]) * bootstrap
            returns[:, idx] = bootstrap
        return returns

    def _compute_bridge_losses(
        self,
        trajectory: WMTrajectory,
        world_model_ref: Optional[Any],
        weights: LossWeightsConfig,
        metrics: Dict[str, Any],
    ) -> Dict[str, Tensor]:
        device = trajectory.device
        zero = torch.tensor(0.0, device=device)
        losses = {
            "projection": zero,
            "control_align": zero,
            "control_reg": zero,
            "abstractor": zero,
            "consistency_head": zero,
            "aux_inverse": zero,
            "aux_value": zero,
        }
        metrics["bridge/maintenance_scale"] = 0.0
        metrics["bridge/active"] = 0.0

        if world_model_ref is None:
            return losses

        ensure_components = getattr(world_model_ref, "_ensure_v45_components", None)
        if callable(ensure_components):
            try:
                ensure_components()
            except Exception:
                return losses

        wm_cfg = getattr(world_model_ref, "_wm_config", None)
        if wm_cfg is None:
            return losses

        step_count = getattr(world_model_ref, "_step_count", 0)
        if isinstance(step_count, Tensor):
            step_count = int(step_count.detach().item())
        else:
            step_count = int(step_count)

        bridge_scale = 1.0
        if hasattr(wm_cfg, "get_bridge_maintenance_scale"):
            bridge_scale = float(wm_cfg.get_bridge_maintenance_scale(step_count))
        metrics["bridge/maintenance_scale"] = bridge_scale
        if bridge_scale <= 0.0:
            return losses

        obs_embeds = trajectory.obs_embeds
        if obs_embeds is None:
            encode_obs = getattr(world_model_ref, "encode_obs", None)
            if callable(encode_obs):
                obs_embeds = encode_obs(trajectory.vitals_aligned)
            else:
                obs_embeds = trajectory.vitals_aligned
        if obs_embeds is None:
            return losses

        obs_embeds_detached = obs_embeds.detach()
        action_seq = trajectory.actions.to(dtype=obs_embeds_detached.dtype)
        s_pred = torch.cat([trajectory.h_seq, trajectory.z_seq], dim=-1)
        batch_size, seq_len = action_seq.shape[:2]
        active_terms = 0

        control_head = getattr(world_model_ref, "_control_head", None)
        s_ctrl_seq = None
        if control_head is not None and (
            getattr(weights, "control_align", 0.0) > 0.0
            or getattr(weights, "control_reg", 0.0) > 0.0
            or getattr(weights, "consistency_head", 0.0) > 0.0
        ):
            s_prev = control_head.initial_state(batch_size, device)
            s_ctrl_values: List[Tensor] = []
            s_prev_values: List[Tensor] = []
            for idx in range(seq_len):
                s_prev_values.append(s_prev)
                s_curr = control_head(
                    x_rl=obs_embeds_detached[:, idx],
                    action_prev=action_seq[:, idx],
                    s_ctrl_prev=s_prev,
                )
                s_ctrl_values.append(s_curr)
                s_prev = s_curr
            s_ctrl_seq = torch.stack(s_ctrl_values, dim=1)
            s_ctrl_prev_seq = torch.stack(s_prev_values, dim=1)
            control_losses = control_head.compute_aux_loss(
                s_ctrl=s_ctrl_seq,
                s_ctrl_prev=s_ctrl_prev_seq,
                x_t=obs_embeds_detached,
            )
            if "L_align" in control_losses:
                losses["control_align"] = bridge_scale * control_losses["L_align"]
                metrics["bridge/control_align"] = float(losses["control_align"].detach().item())
                active_terms += int(float(losses["control_align"].detach().item()) > 0.0)
            if "L_reg" in control_losses:
                losses["control_reg"] = bridge_scale * control_losses["L_reg"]
                metrics["bridge/control_reg"] = float(losses["control_reg"].detach().item())
                active_terms += int(float(losses["control_reg"].detach().item()) > 0.0)

        projection = getattr(world_model_ref, "_projection", None)
        if projection is not None and getattr(weights, "projection", 0.0) > 0.0:
            x_proj = projection(s_pred.reshape(batch_size * seq_len, -1)).reshape(
                batch_size, seq_len, -1
            )
            projection_losses = projection.compute_loss(
                x_true=obs_embeds_detached.reshape(batch_size * seq_len, -1),
                x_proj=x_proj.reshape(batch_size * seq_len, -1),
            )
            losses["projection"] = bridge_scale * projection_losses["L_projection"]
            metrics["bridge/projection"] = float(losses["projection"].detach().item())
            active_terms += int(float(losses["projection"].detach().item()) > 0.0)

        abstractor = getattr(world_model_ref, "_abstractor", None)
        if abstractor is not None and getattr(weights, "abstractor", 0.0) > 0.0:
            future_returns = self._compute_discounted_returns(
                rewards=trajectory.rewards,
                dones=trajectory.dones,
            )
            state_returns = torch.cat(
                [future_returns[:, 1:], torch.zeros_like(future_returns[:, :1])],
                dim=1,
            )
            state_advantage = state_returns - state_returns.mean(dim=1, keepdim=True)
            z_task = abstractor(s_pred.reshape(batch_size * seq_len, -1))
            abstractor_losses = abstractor.compute_loss(
                z_task=z_task,
                advantage=state_advantage.reshape(-1),
            )
            losses["abstractor"] = bridge_scale * abstractor_losses["L_abs"]
            metrics["bridge/abstractor"] = float(losses["abstractor"].detach().item())
            active_terms += int(float(losses["abstractor"].detach().item()) > 0.0)

        consistency_head = getattr(world_model_ref, "_consistency_head", None)
        if (
            consistency_head is not None
            and s_ctrl_seq is not None
            and seq_len > 1
            and getattr(weights, "consistency_head", 0.0) > 0.0
        ):
            s_ctrl_pred = consistency_head(
                s_pred=s_pred[:, :-1].reshape(batch_size * (seq_len - 1), -1),
                action=action_seq[:, 1:].reshape(batch_size * (seq_len - 1), -1),
            )
            consistency_losses = consistency_head.compute_loss(
                s_ctrl_true=s_ctrl_seq[:, 1:].reshape(batch_size * (seq_len - 1), -1),
                s_ctrl_pred=s_ctrl_pred,
            )
            losses["consistency_head"] = bridge_scale * consistency_losses["L_cons"]
            metrics["bridge/consistency_head"] = float(
                losses["consistency_head"].detach().item()
            )
            active_terms += int(float(losses["consistency_head"].detach().item()) > 0.0)

        aux_tasks = getattr(world_model_ref, "_aux_tasks", None)
        if aux_tasks is not None and seq_len > 1 and (
            getattr(weights, "aux_inverse", 0.0) > 0.0
            or getattr(weights, "aux_value", 0.0) > 0.0
        ):
            aux_returns = self._compute_discounted_returns(
                rewards=trajectory.rewards[:, 1:],
                dones=trajectory.dones[:, 1:],
            )
            wall_cfg = getattr(wm_cfg, "wall", None)
            inverse_scale = bridge_scale
            value_scale = bridge_scale
            if wall_cfg is not None:
                inverse_scale *= float(getattr(wall_cfg, "aux_inverse_weight", 1.0))
                value_scale *= float(getattr(wall_cfg, "aux_value_weight", 1.0))
            aux_losses = aux_tasks.compute_loss(
                s_pred_t=s_pred[:, :-1].reshape(batch_size * (seq_len - 1), -1),
                s_pred_tp1=s_pred[:, 1:].reshape(batch_size * (seq_len - 1), -1),
                action_t=action_seq[:, 1:].reshape(batch_size * (seq_len - 1), -1),
                returns=aux_returns.reshape(-1, 1),
                inverse_weight=inverse_scale,
                value_weight=value_scale,
            )
            losses["aux_inverse"] = aux_losses["L_inverse"]
            losses["aux_value"] = aux_losses["L_value_aux"]
            metrics["bridge/aux_inverse"] = float(losses["aux_inverse"].detach().item())
            metrics["bridge/aux_value"] = float(losses["aux_value"].detach().item())
            active_terms += int(float(losses["aux_inverse"].detach().item()) > 0.0)
            active_terms += int(float(losses["aux_value"].detach().item()) > 0.0)

        metrics["bridge/active"] = float(active_terms > 0)
        return losses

    def _compute_consistency_loss(
        self,
        trajectory: WMTrajectory,
        world_model_ref: Optional[Any],
        consistency_weights: Dict[str, float],
        metrics: Dict[str, Any],
    ) -> ConsistencyReport:
        device = trajectory.device
        auditor = self.__dict__.get('_consistency_auditor')
        
        if auditor is None:
            return ConsistencyReport.zeros(device)
        
        report = auditor.audit(
            trajectory=trajectory,
            world_model_ref=world_model_ref,
            weights=consistency_weights,
        )
        
        metrics.update(report.metrics)
        return report

    def get_info(self) -> Dict[str, Any]:
        info = {
            'deter_dim': self._deter_dim,
            'stoch_dim': self._stoch_dim,
            'feat_dim': self._feat_dim,
            'has_decoder': self.rssm_branch.decoder is not None,
            'has_msc_head': self.rssm_branch.msc_head is not None,
            'has_consistency_auditor': self.has_consistency_auditor,
        }
        
        auditor = self.__dict__.get('_consistency_auditor')
        if auditor is not None:
            info['consistency_auditor'] = auditor.get_info()
        
        return info


# =============================================================================
# ControlHead: 控制分支
# =============================================================================

class ControlHead(nn.Module):
    """控制分支生成 s_ctrl 供策略使用"""

    MODE_STATIC = "static"
    MODE_GATED = "gated"

    def __init__(
        self,
        input_dim: int,
        ctrl_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256,),
        mode: str = "static",
        align_strength: float = 0.1,
        reg_strength: float = 0.01,
    ):
        super().__init__()
        
        if mode not in (self.MODE_STATIC, self.MODE_GATED):
            raise ValueError(f"mode must be 'static' or 'gated', got '{mode}'")
        
        self.mode = mode
        self.ctrl_dim = ctrl_dim
        self.action_dim = action_dim
        self.align_strength = align_strength
        self.reg_strength = reg_strength
        
        if mode == self.MODE_STATIC:
            self.net = self._build_mlp(input_dim, ctrl_dim, hidden_dims)
            self.gru = None
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(input_dim + action_dim, ctrl_dim)
            self.gru = nn.GRUCell(ctrl_dim, ctrl_dim)
            self.output_norm = create_norm(ctrl_dim, "rms")
            self.net = None
        
        self.register_buffer('_init_state', torch.zeros(1, ctrl_dim), persistent=False)

    @staticmethod
    def _build_mlp(input_dim: int, output_dim: int, hidden_dims: Tuple[int, ...]) -> nn.Sequential:
        hidden_dims_tuple = tuple(int(x) for x in hidden_dims) if hidden_dims else None
        return nn.Sequential(build_swiglu_mlp(
            in_dim=int(input_dim),
            out_dim=int(output_dim),
            hidden_dims=hidden_dims_tuple,
            dropout=0.0,
            norm_type="rms",
        ))

    def initial_state(self, batch_size: int, device: torch.device) -> Tensor:
        return self._init_state.expand(batch_size, -1).to(device)

    def forward(
        self,
        x_rl: Tensor,
        action_prev: Optional[Tensor] = None,
        s_ctrl_prev: Optional[Tensor] = None,
    ) -> Tensor:
        if self.mode == self.MODE_STATIC:
            return self.net(x_rl)
        
        B = x_rl.shape[0]
        device = x_rl.device
        
        if action_prev is None:
            action_prev = torch.zeros(B, self.action_dim, device=device)
        if s_ctrl_prev is None:
            s_ctrl_prev = self.initial_state(B, device)
        
        gru_input = self.input_proj(torch.cat([x_rl, action_prev], dim=-1))
        s_ctrl = self.gru(gru_input, s_ctrl_prev)
        return self.output_norm(s_ctrl)

    def compute_aux_loss(
        self,
        s_ctrl: Tensor,
        s_ctrl_prev: Optional[Tensor],
        x_t: Tensor,
    ) -> Dict[str, Tensor]:
        losses = {}
        
        if self.mode == self.MODE_STATIC and self.align_strength > 0:
            x_t_d = x_t.detach()
            min_dim = min(s_ctrl.shape[-1], x_t_d.shape[-1])
            losses['L_align'] = self.align_strength * F.mse_loss(
                s_ctrl[..., :min_dim], x_t_d[..., :min_dim]
            )
        
        if self.mode == self.MODE_GATED and self.reg_strength > 0 and s_ctrl_prev is not None:
            losses['L_reg'] = self.reg_strength * F.mse_loss(
                s_ctrl, s_ctrl_prev.detach()
            )
        
        return losses


# =============================================================================
# ProjectionLayer: 投影层
# =============================================================================

class ProjectionLayer(nn.Module):
    """将 s_pred 投影回 x 空间"""

    def __init__(
        self, 
        s_pred_dim: int, 
        x_dim: int, 
        hidden_dim: int = 256,
        mmd_weight: float = 0.1,
        mmd_mode: str = "linear",
        mmd_num_features: int = 256,
        gradient_scale: float = 0.1,
    ):
        super().__init__()
        self.mmd_weight = mmd_weight
        self.mmd_mode = mmd_mode
        self.x_dim = x_dim
        self.gradient_scale = gradient_scale
        
        self.net = nn.Sequential(
            nn.Linear(s_pred_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
            nn.LayerNorm(x_dim),
        )
        
        if mmd_mode == "linear":
            self.register_buffer(
                '_rff_weight',
                torch.randn(x_dim, mmd_num_features) / x_dim ** 0.5,
                persistent=False,
            )
            self.register_buffer(
                '_rff_bias',
                torch.rand(mmd_num_features) * 2 * 3.14159,
                persistent=False,
            )

    def forward(self, s_pred: Tensor) -> Tensor:
        s_pred_scaled = scale_gradient(s_pred, self.gradient_scale)
        return self.net(s_pred_scaled)

    def compute_loss(
        self,
        x_true: Tensor,
        x_proj: Tensor,
    ) -> Dict[str, Tensor]:
        x_true_d = x_true.detach()
        
        L_proj_mse = F.mse_loss(x_proj, x_true_d)
        
        if self.mmd_mode == "none" or self.mmd_weight == 0:
            L_proj_mmd = torch.zeros(1, device=x_proj.device)
        elif self.mmd_mode == "linear":
            L_proj_mmd = self._compute_linear_mmd(x_true_d, x_proj)
        else:
            L_proj_mmd = self._compute_kernel_mmd(x_true_d, x_proj)
        
        L_projection = L_proj_mse + self.mmd_weight * L_proj_mmd
        
        return {
            'L_proj_mse': L_proj_mse,
            'L_proj_mmd': L_proj_mmd,
            'L_projection': L_projection,
        }

    def _compute_linear_mmd(self, x: Tensor, y: Tensor) -> Tensor:
        phi_x = torch.cos(x @ self._rff_weight + self._rff_bias)
        phi_y = torch.cos(y @ self._rff_weight + self._rff_bias)
        
        mean_x = phi_x.mean(dim=0)
        mean_y = phi_y.mean(dim=0)
        
        return ((mean_x - mean_y) ** 2).sum()

    def _compute_kernel_mmd(
        self, 
        x: Tensor, 
        y: Tensor, 
        sigma: float = 1.0,
        max_samples: int = 256,
    ) -> Tensor:
        B = x.shape[0]
        
        if B > max_samples:
            indices = torch.randperm(B, device=x.device)[:max_samples]
            x = x[indices]
            y = y[indices]
        
        def gaussian_kernel(a: Tensor, b: Tensor) -> Tensor:
            dist = torch.cdist(a, b, p=2)
            return torch.exp(-dist ** 2 / (2 * sigma ** 2))
        
        xx = gaussian_kernel(x, x).mean()
        yy = gaussian_kernel(y, y).mean()
        xy = gaussian_kernel(x, y).mean()
        
        return xx + yy - 2 * xy


# =============================================================================
# Abstractor: 抽象器
# =============================================================================

class Abstractor(nn.Module):
    """从 s_pred 提取任务相关抽象特征 z_task"""

    def __init__(
        self, 
        s_pred_dim: int, 
        z_task_dim: int, 
        hidden_dim: int = 256,
        use_scalar_head: bool = True,
        gradient_scale: float = 0.1,
    ):
        super().__init__()
        self.z_task_dim = z_task_dim
        self.use_scalar_head = use_scalar_head
        self.gradient_scale = gradient_scale
        
        self.net = nn.Sequential(
            nn.Linear(s_pred_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z_task_dim),
        )
        
        self.scalar_head = nn.Linear(z_task_dim, 1) if use_scalar_head else None

    def forward(self, s_pred: Tensor) -> Tensor:
        s_pred_scaled = scale_gradient(s_pred, self.gradient_scale)
        return self.net(s_pred_scaled)

    def compute_loss(
        self,
        z_task: Tensor,
        advantage: Tensor,
    ) -> Dict[str, Tensor]:
        if self.scalar_head is not None:
            pred = self.scalar_head(z_task).squeeze(-1)
        else:
            pred = z_task.mean(dim=-1)
        
        target = (advantage.detach().view(-1) > 0).float()
        
        L_abs = F.binary_cross_entropy_with_logits(pred.view(-1), target)
        
        return {'L_abs': L_abs}


# =============================================================================
# ConsistencyHead: 一致性预测头
# =============================================================================

class ConsistencyHead(nn.Module):
    """从 (s_pred, action) 预测下一步 s_ctrl"""

    def __init__(
        self, 
        s_pred_dim: int, 
        action_dim: int, 
        ctrl_dim: int, 
        hidden_dim: int = 256,
        gradient_scale: float = 0.1,
    ):
        super().__init__()
        self.gradient_scale = gradient_scale
        self.net = nn.Sequential(
            nn.Linear(s_pred_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, ctrl_dim),
        )

    def forward(self, s_pred: Tensor, action: Tensor) -> Tensor:
        s_pred_scaled = scale_gradient(s_pred, self.gradient_scale)
        return self.net(torch.cat([s_pred_scaled, action], dim=-1))

    def compute_loss(
        self,
        s_ctrl_true: Tensor,
        s_ctrl_pred: Tensor,
    ) -> Dict[str, Tensor]:
        return {'L_cons': F.mse_loss(s_ctrl_pred, s_ctrl_true.detach())}


# =============================================================================
# ControlAuxTasks: 控制辅助任务
# =============================================================================

class ControlAuxTasks(nn.Module):
    """控制相关辅助任务"""

    def __init__(
        self,
        s_pred_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        value_dim: int = 1,
        is_discrete_action: bool = False,
        gradient_scale: float = 0.05,
    ):
        super().__init__()
        
        self.is_discrete_action = is_discrete_action
        self.action_dim = action_dim
        self.s_pred_dim = s_pred_dim
        self.gradient_scale = gradient_scale
        
        # 逆动力学模型
        self.inverse_model = nn.Sequential(
            nn.Linear(s_pred_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # 价值预测模型
        self.value_model = nn.Sequential(
            nn.Linear(s_pred_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, value_dim),
        )

    def forward_inverse(self, s_pred_t: Tensor, s_pred_tp1: Tensor) -> Tensor:
        s_t_scaled = scale_gradient(s_pred_t, self.gradient_scale)
        s_tp1_scaled = scale_gradient(s_pred_tp1, self.gradient_scale)
        return self.inverse_model(torch.cat([s_t_scaled, s_tp1_scaled], dim=-1))

    def forward_value(self, s_pred: Tensor) -> Tensor:
        s_pred_scaled = scale_gradient(s_pred, self.gradient_scale)
        return self.value_model(s_pred_scaled)

    def forward(
        self,
        s_pred_t: Tensor,
        s_pred_tp1: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        action_pred = self.forward_inverse(s_pred_t, s_pred_tp1)
        value_pred = self.forward_value(s_pred_t)
        return action_pred, value_pred

    def compute_loss(
        self,
        s_pred_t: Tensor,
        s_pred_tp1: Tensor,
        action_t: Tensor,
        returns: Tensor,
        inverse_weight: float = 1.0,
        value_weight: float = 1.0,
    ) -> Dict[str, Tensor]:
        action_pred, value_pred = self.forward(s_pred_t, s_pred_tp1)
        
        # 逆动力学损失
        action_target = action_t.detach()
        if self.is_discrete_action:
            if action_target.dim() > 1 and action_target.shape[-1] > 1:
                action_target = action_target.argmax(dim=-1)
            L_inverse = F.cross_entropy(action_pred, action_target.long())
        else:
            L_inverse = F.mse_loss(action_pred, action_target)
        
        # 价值预测损失
        returns_target = returns.detach()
        if returns_target.dim() == 1:
            returns_target = returns_target.unsqueeze(-1)
        L_value_aux = F.mse_loss(value_pred, returns_target)
        
        return {
            'L_inverse': inverse_weight * L_inverse,
            'L_value_aux': value_weight * L_value_aux,
            'L_aux_total': inverse_weight * L_inverse + value_weight * L_value_aux,
        }






# ======================================================================
# Part 3: World Model Core, Builder, MSC & Monitoring
# ======================================================================

# 注意此部分依赖于Part 1和Part 2中的所有模块
# 确保在运行此代码前Part 1和Part 2已经加载

pass # from .types import (
#     WMState, RSSMOutputs, WillVersion, RouterMode, WallPhase,
#     WMTrajectory, ConsistencyReport, PredictiveLossPacket,
#     LossWeights, WorldModelProtocol,
# )
pass # from .components import (
#     PredictiveEngine, ControlHead, ProjectionLayer, Abstractor,
#     ConsistencyHead, ControlAuxTasks,
# )
pass # from .losses import ConsistencyAuditor

from .aletheia_foundation import (
    WorldModelConfig, LossWeightsConfig, ConsistencyAuditorConfig,
    ControlConfig, ProjectionConfig, AbstractorConfig, ConsistencyHeadConfig,
    AuxTasksConfig, PredictiveEngineConfig,
    build_swiglu_mlp, create_norm, SymlogLayer, ActionCodec,
    ActionCodecConfig, safe_torch_load, build_mlp,
)
pass # from .monitoring import GapMonitor, GapMonitorConfig
pass # from .msc import MultiScaleContinue, MultiScaleContinueConfig


# ======================================================================
# MERGED FROM: world_model/monitoring.py
# ======================================================================

class GapMonitor:
    """
    Gap Monitor: 诊断想象与真实回报之间的差距
    
    纯诊断工具不参与训练过程
    """

    def __init__(self, config: Optional["GapMonitorConfig"] = None):
        from .aletheia_foundation import GapMonitorConfig
        self.config = config or GapMonitorConfig()
        
        self._gap_ema: float = 1.5
        self._history: deque = deque(maxlen=self.config.history_size)
        
        self._warning_count: int = 0
        self._critical_count: int = 0
        self._consecutive_critical: int = 0
        self._update_count: int = 0
    
    def _to_scalar(self, value: Union[Tensor, np.ndarray, float]) -> float:
        if isinstance(value, Tensor):
            return value.detach().cpu().mean().item()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        return float(value)
    
    def update(
        self,
        imag_returns: Union[Tensor, float],
        real_returns: Union[Tensor, float],
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            return {'status': 'DISABLED'}
        
        imag_mean = self._to_scalar(imag_returns)
        real_mean = self._to_scalar(real_returns)
        
        real_abs = abs(real_mean) + 1e-8
        gap = imag_mean / real_abs
        
        self._gap_ema = (
            self.config.ema_decay * self._gap_ema +
            (1 - self.config.ema_decay) * gap
        )
        
        self._history.append(gap)
        alert_level = self._check_alert(gap)
        
        if alert_level == 'CRITICAL':
            self._consecutive_critical += 1
        else:
            self._consecutive_critical = 0
        
        self._update_count += 1
        
        result = {
            'gap_instant': gap,
            'gap_ema': self._gap_ema,
            'imag_return': imag_mean,
            'real_return': real_mean,
            'alert_level': alert_level,
            'update_count': self._update_count,
        }
        
        if self._update_count % 100 == 0:
            logger.info(f"[GapMonitor] Update #{self._update_count}: Gap={self._gap_ema:.2f}x ({alert_level})")
        
        return result
    
    def _check_alert(self, gap: float) -> str:
        if gap > self.config.critical_threshold:
            self._critical_count += 1
            return 'CRITICAL'
        elif gap > self.config.warning_threshold:
            self._warning_count += 1
            return 'WARNING'
        return 'OK'
    
    def should_trigger_fallback(self) -> bool:
        if self._consecutive_critical >= 10:
            return True
        if len(self._history) >= 50:
            recent = list(self._history)[-50:]
            first_half = sum(recent[:25]) / 25
            second_half = sum(recent[25:]) / 25
            if second_half > first_half * 1.5:
                return True
        return False
    
    def reset(self):
        self._gap_ema = 1.5
        self._history.clear()
        self._warning_count = 0
        self._critical_count = 0
        self._consecutive_critical = 0
        self._update_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        if len(self._history) == 0:
            return {'status': 'NO_DATA'}
        
        gaps = list(self._history)
        return {
            'gap_ema': self._gap_ema,
            'gap_mean': sum(gaps) / len(gaps),
            'warning_count': self._warning_count,
            'critical_count': self._critical_count,
            'consecutive_critical': self._consecutive_critical,
            'update_count': self._update_count,
        }


# ======================================================================
# MERGED FROM: world_model/msc.py
# ======================================================================

class MultiScaleContinue(nn.Module):
    """
    Multi-Scale Continue (MSC) 模块
    
    预测未来 k 步内终止的累积概率
    输出性质p_1  p_2  ...  p_max (累积概率)
    """

    def __init__(
        self,
        feat_dim: int,
        config: Optional[MultiScaleContinueConfig] = None,
    ):
        super().__init__()
        from .aletheia_foundation import MultiScaleContinueConfig
        self.config = config or MultiScaleContinueConfig()
        self.feat_dim = feat_dim
        
        self.register_buffer(
            'horizons',
            torch.tensor(self.config.horizons, dtype=torch.long),
        )
        
        # 共享编码器
        hidden_dims = [self.config.hidden_dim, self.config.hidden_dim // 2]
        self.shared_encoder = _build_mlp(
            in_dim=feat_dim,
            out_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation='silu',
            layer_norm=True,
            output_activation=True,
        )
        
        # 多尺度终止概率头
        self.termination_head = nn.Linear(hidden_dims[-1], self.config.num_horizons)
        
        # 危险度预测头辅助任务
        if self.config.predict_danger:
            self.danger_head = nn.Sequential(
                nn.Linear(hidden_dims[-1], self.config.danger_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.danger_hidden_dim, 1),
            )
        else:
            self.danger_head = None
        
        self._register_effective_weights()
    
    def _register_effective_weights(self):
        horizons = self.config.horizons
        mode = self.config.effective_continue_mode
        
        if mode == "geometric":
            weights = torch.tensor([1.0 / (h + 1) for h in horizons])
        elif mode == "weighted":
            weights = torch.tensor([0.9 ** i for i in range(len(horizons))])
        else:
            weights = torch.ones(len(horizons))
        
        weights = weights / weights.sum()
        self.register_buffer('_effective_weights', weights)

    def build_termination_prob_dict(
        self,
        termination_probs: Tensor,
    ) -> Dict[int, Tensor]:
        return _horizon_prob_dict_from_tensor(
            termination_probs,
            self.horizons.detach().cpu().tolist(),
        )
    
    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        has_time = features.dim() == 3
        if has_time:
            B, T, D = features.shape
            features = features.reshape(B * T, D)
        
        shared = self.shared_encoder(features)
        term_logits = self.termination_head(shared) / self.config.temperature
        term_probs = torch.sigmoid(term_logits)
        
        if self.config.enforce_monotonicity:
            term_probs, _ = torch.cummax(term_probs, dim=-1)
        
        continue_probs = 1 - term_probs
        effective_continue = (continue_probs * self._effective_weights).sum(dim=-1)
        
        if has_time:
            term_probs = term_probs.reshape(B, T, -1)
            continue_probs = continue_probs.reshape(B, T, -1)
            effective_continue = effective_continue.reshape(B, T)
        
        result = {
            'termination_probs': term_probs,
            'termination_prob_dict': self.build_termination_prob_dict(term_probs),
            'continue_probs': continue_probs,
            'effective_continue': effective_continue,
        }
        
        if self.danger_head is not None:
            danger = torch.sigmoid(self.danger_head(shared))
            if has_time:
                B, T = features.shape[0], features.shape[1] // T
                danger = danger.reshape(B, T, 1)
            result['danger'] = danger
        
        return result
    
    def compute_loss(
        self,
        features: Tensor,
        remaining_steps: Tensor,
        true_danger: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = features.device
        
        if remaining_steps is None:
            max_h = max(self.config.horizons) if self.horizons.numel() > 0 else 10
            remaining_steps = torch.full(
                (features.shape[0],), max_h, device=device, dtype=torch.long
            )
        
        result = self.forward(features)
        term_probs = result['termination_probs']
        
        if term_probs.dim() == 2:
            term_probs = term_probs.unsqueeze(1)
        if remaining_steps.dim() == 1:
            remaining_steps = remaining_steps.unsqueeze(-1)
        
        horizons = self.horizons.float().view(1, 1, -1)
        remaining = remaining_steps.unsqueeze(-1).float()
        labels = (remaining <= horizons).float()
        
        bce = F.binary_cross_entropy(
            term_probs.clamp(1e-7, 1 - 1e-7),
            labels,
            reduction='none',
        )
        bce_loss = bce.mean()
        
        monotonicity_loss = torch.tensor(0.0, device=device)
        if not self.config.enforce_monotonicity:
            diff = term_probs[..., 1:] - term_probs[..., :-1]
            violations = F.relu(-diff)
            monotonicity_loss = violations.mean()
        
        danger_loss = torch.tensor(0.0, device=device)
        if self.config.predict_danger and true_danger is not None:
            pred_danger = result.get('danger')
            if pred_danger is not None:
                danger_loss = F.mse_loss(pred_danger, true_danger)
        
        total_loss = (
            self.config.bce_weight * bce_loss +
            self.config.danger_weight * danger_loss +
            self.config.monotonicity_weight * monotonicity_loss
        )
        
        metrics = {
            'msc_bce': bce_loss.detach(),
            'msc_danger': danger_loss.detach(),
            'msc_monotonicity': monotonicity_loss.detach(),
        }
        
        return total_loss, metrics
    
    def get_prob(self, features: Tensor) -> Tensor:
        return self.forward(features)['effective_continue']
class WorldModel(nn.Module):
    """
    世界模型 V5.2
    
    核心职责
    1. 管理 RSSM 状态转移
    2. 管理预测头
    3. 通过 PredictiveEngine 统一计算损失
    4. 通过 ConsistencyAuditor 计算一致性损失
    """

    def __init__(
        self,
        config: Any,
        wm_config: Optional[WorldModelConfig] = None,
    ):
        super().__init__()
        self.config = config
        
        self._wm_config = wm_config
        self._align_rssm_config(
            validation_mode=getattr(self.config, "validation_mode", "warn")
        )

        # 1. Build Core Components
        components = self._build_core_components()
        
        self.state_transition = components["state_transition"]
        self.encoder = components["encoder"]
        self.decoder = components["decoder"]
        self.reward_head = components["reward_head"]
        self.continue_head = components["continue_head"]
        self.mhc_head = components["mhc_head"]
        self.msc_head = components["msc_head"]
        
        self._nst_loss_module = components["nst_loss"]
        self._sc_loss_module = components["sc_loss"]
        
        self._encoder_out_dim = self._infer_encoder_out_dim()
        self._loss_weights_config = self._parse_loss_weights_config(config)
        
        # 2. PredictiveEngine (Lazy)
        self.__dict__['_predictive_engine'] = None
        self.__dict__['_consistency_auditor'] = None
        self._control_head: Optional[ControlHead] = None
        self._projection: Optional[ProjectionLayer] = None
        self._abstractor: Optional[Abstractor] = None
        self._consistency_head: Optional[ConsistencyHead] = None
        self._aux_tasks: Optional[ControlAuxTasks] = None
        
        self.register_buffer('_step_count', torch.tensor(0, dtype=torch.long))
        
        # 3. Gap Monitor
        self.gap_monitor = GapMonitor(getattr(config, "gap_monitor", None))
        
        # 4. Action Codec
        self.action_codec: Optional[ActionCodec] = None
        self._action_dim: Optional[int] = None
        self._is_discrete: Optional[bool] = None
        
        # 5. Tools
        self._symlog = SymlogLayer(
            input_clip=config.symlog.input_clip,
            symexp_clip=config.symlog.symexp_clip,
            output_clip=config.symlog.output_clip,
        )
        
        self.__dict__['_danger_fn'] = None
        
        # 6. Perceptor Integration
        self.perceptor: Optional[Any] = None
        self._use_perceptor = False
        
        self._cached_device: Optional[torch.device] = None

    def _build_core_components(self) -> Dict[str, Any]:
        config = self.config

        continue_cfg = getattr(config, "continue_config", None)
        continue_hidden_dims = tuple(
            getattr(continue_cfg, "hidden_dims", None) or (256, 256)
        )
        mhc_cfg = getattr(config, "mhc", None)
        msc_cfg = getattr(config, "msc", None) or getattr(config, "msc_config", None)
        nst_cfg = getattr(config, "nst", None)
        sc_cfg = getattr(config, "shortcut_consistency", None)

        return {
            "state_transition": StateTransition(config),
            "encoder": build_mlp(
                in_dim=config.vitals_dim,
                out_dim=config.obs_embed_dim,
                hidden_dim=config.hidden_dim,
                num_layers=3,
                activation="silu",
                final_gain=1.0,
                init="final_gain_only",
            ),
            "decoder": VitalsDecoder(
                feat_dim=config.feat_dim,
                vitals_dim=config.vitals_dim,
                config=VitalsDecoderConfig(hidden_dims=(256, 256)),
            ),
            "reward_head": SymlogRewardHead(
                feat_dim=config.feat_dim,
                config=RewardHeadConfig(hidden_dims=(256, 256), symlog=config.symlog),
            ),
            "continue_head": ContinueHead(
                feat_dim=config.feat_dim,
                config=ContinueHeadConfig(
                    hidden_dims=continue_hidden_dims,
                    temperature=float(getattr(continue_cfg, "temperature", 1.0)),
                    loss_type=str(getattr(continue_cfg, "loss_type", "bce")),
                    focal_alpha=float(getattr(continue_cfg, "focal_alpha", 0.25)),
                    focal_gamma=float(getattr(continue_cfg, "focal_gamma", 2.0)),
                    optimistic_bias=float(getattr(continue_cfg, "optimistic_bias", 5.0)),
                    positive_weight=float(getattr(continue_cfg, "positive_weight", 1.0)),
                    negative_weight=float(getattr(continue_cfg, "negative_weight", 1.0)),
                ),
            ),
            "mhc_head": (
                MultiHorizonContinueHead(mhc_cfg)
                if mhc_cfg is not None and getattr(mhc_cfg, "enabled", False)
                else None
            ),
            "msc_head": (
                MultiScaleContinue(config.feat_dim, msc_cfg)
                if msc_cfg is not None and getattr(msc_cfg, "enabled", False)
                else None
            ),
            "nst_loss": (
                NStepTerminationLoss(nst_cfg)
                if nst_cfg is not None and getattr(nst_cfg, "enabled", False)
                else None
            ),
            "sc_loss": (
                ShortcutConsistencyLoss(sc_cfg)
                if sc_cfg is not None and getattr(sc_cfg, "enabled", False)
                else None
            ),
        }

    def _align_rssm_config(self, validation_mode: str = "warn") -> None:
        config = self.config
        wm_config = self._wm_config
        if wm_config is None or config is None:
            return

        del validation_mode

        dist = getattr(config, "distribution", None)
        if dist is None:
            return

        def _handle(field: str, current: Any, desired: Any) -> None:
            if current == desired:
                return
            raise ValueError(
                f"RSSMConfig.{field} ({current}) != WorldModelConfig.{field} "
                f"({desired}); auto-alignment is no longer supported. "
                "Fix the upstream configuration."
            )

        for name in ("deter_dim", "obs_embed_dim"):
            value = getattr(wm_config, name, None)
            if value is None:
                continue
            _handle(name, getattr(config, name, None), value)

        wm_num_classes = getattr(wm_config, "num_classes", None)
        if wm_num_classes is not None:
            _handle("num_classes", dist.num_classes, int(wm_num_classes))

        wm_stoch_dim = getattr(wm_config, "stoch_dim", None)
        if wm_stoch_dim is None:
            return

        if int(dist.num_classes) <= 0:
            raise ValueError(
                f"RSSMConfig.num_classes must be positive to validate "
                f"WorldModelConfig.stoch_dim ({wm_stoch_dim})"
            )

        if int(wm_stoch_dim) % int(dist.num_classes) != 0:
            raise ValueError(
                f"WorldModelConfig.stoch_dim ({wm_stoch_dim}) not divisible "
                f"by num_classes ({dist.num_classes}); auto-alignment is no longer "
                "supported. Fix the upstream configuration."
            )

        desired_num_distributions = int(wm_stoch_dim) // int(dist.num_classes)
        _handle("num_distributions", dist.num_distributions, desired_num_distributions)

    # =========================================================================
    # 配置解析
    # =========================================================================

    def _parse_loss_weights_config(self, config: Any) -> LossWeightsConfig:
        if self._wm_config is not None:
            weights = copy.deepcopy(self._wm_config.loss_weights)
        else:
            weights = LossWeightsConfig()
        weights.recon = getattr(config, "recon_weight", 1.0)
        weights.reward = getattr(config, "reward_weight", 1.0)
        weights.continue_ = getattr(config, "continue_weight", 1.0)
        weights.kl = getattr(config, "kl_weight", 1.0)
        weights.quant = getattr(config, "quant_loss_weight", 0.0)
        
        for name in ["mhc", "msc", "nst"]:
            cfg = getattr(config, name, None) or getattr(config, f"{name}_config", None)
            val = getattr(cfg, "loss_scale", 1.0) if cfg and getattr(cfg, "enabled", False) else 0.0
            setattr(weights, name, val)
        
        sc_cfg = getattr(config, "shortcut_consistency", None)
        weights.sc = getattr(sc_cfg, "loss_scale", 1.0) if sc_cfg and getattr(sc_cfg, "enabled", False) else 0.0
        return weights

    def _infer_encoder_out_dim(self) -> int:
        if hasattr(self.encoder, 'out_features'):
            return self.encoder.out_features
        if hasattr(self.encoder, 'output_dim'):
            return self.encoder.output_dim
        if hasattr(self.config, 'obs_embed_dim'):
            return self.config.obs_embed_dim
        return self.config.feat_dim

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def predictive_engine(self) -> PredictiveEngine:
        if self.__dict__.get('_predictive_engine') is None:
            self._ensure_predictive_engine()
        return self.__dict__['_predictive_engine']

    @property
    def consistency_auditor(self) -> Optional[ConsistencyAuditor]:
        self._ensure_predictive_engine()
        return self.__dict__['_consistency_auditor']

    @property
    def device(self) -> torch.device:
        if self._cached_device is None:
            self._cached_device = next(self.parameters()).device
        return self._cached_device

    @property
    def feat_dim(self) -> int:
        return self.config.feat_dim

    @property
    def action_dim(self) -> Optional[int]:
        return self._action_dim

    # =========================================================================
    # V5.2 组件初始化
    # =========================================================================

    def _ensure_predictive_engine(self) -> None:
        if self.__dict__.get('_predictive_engine') is not None:
            return
        
        auditor = self._build_consistency_auditor()
        
        engine = PredictiveEngine(
            state_transition=self.state_transition,
            reward_head=self.reward_head,
            continue_head=self.continue_head,
            decoder=self.decoder,
            symlog=self._symlog,
            msc_head=self.msc_head,
        )
        
        if auditor is not None:
            engine.set_consistency_auditor(auditor)
        
        self.__dict__['_predictive_engine'] = engine
        self.__dict__['_consistency_auditor'] = auditor

    def _build_consistency_auditor(self) -> Optional[ConsistencyAuditor]:
        has_any = any([
            self.mhc_head is not None,
            self.msc_head is not None,
            self._nst_loss_module is not None,
            self._sc_loss_module is not None,
        ])
        
        if not has_any:
            return None
        
        auditor_config = ConsistencyAuditorConfig()
        if self.mhc_head:
            auditor_config.mhc.enabled = True
        if self.msc_head:
            auditor_config.msc.enabled = True
        if self._nst_loss_module:
            auditor_config.nst.enabled = True
        if self._sc_loss_module:
            auditor_config.sc.enabled = True
        auditor_config.validate()
        
        auditor = ConsistencyAuditor(
            config=auditor_config,
            mhc_head=self.mhc_head,
            msc_head=self.msc_head,
            nst_module=self._nst_loss_module,
            sc_module=self._sc_loss_module,
        )
        return auditor

    def _ensure_v45_components(self) -> None:
        if self._action_dim is None:
            raise RuntimeError("v4.5 components require action_dim. Call setup_action_space().")
        
        if self._control_head is None:
            self._control_head = ControlHead(
                input_dim=self._encoder_out_dim,
                ctrl_dim=self._wm_config.ctrl_dim,
                action_dim=self._action_dim,
            )
        
        if self._projection is None and self._wm_config.use_projection:
            self._projection = ProjectionLayer(
                s_pred_dim=self._wm_config.s_pred_dim,
                x_dim=self._encoder_out_dim,
            )
        
        if self._abstractor is None and self._wm_config.use_abstractor:
            self._abstractor = Abstractor(
                s_pred_dim=self._wm_config.s_pred_dim,
            )
        
        if self._consistency_head is None and self._wm_config.use_consistency_head:
            self._consistency_head = ConsistencyHead(
                s_pred_dim=self._wm_config.s_pred_dim,
                action_dim=self._action_dim,
                ctrl_dim=self._wm_config.ctrl_dim,
            )

        if self._aux_tasks is None and self._wm_config.use_aux_tasks:
            self._aux_tasks = ControlAuxTasks(
                s_pred_dim=self._wm_config.s_pred_dim,
                action_dim=self._action_dim,
                value_dim=self._wm_config.aux_tasks.value_dim,
                is_discrete_action=bool(self._wm_config.is_discrete_action),
            )

    # =========================================================================
    # 设置方法
    # =========================================================================

    def setup_action_space(self, action_space: Any, codec_config: Optional[Any] = None) -> None:
        if action_space is None:
            raise ValueError("action_space cannot be None")
        
        if codec_config is None:
            embed_dim = getattr(self.config, "action_embed_dim", 32)
            codec_config = ActionCodecConfig(embed_dim=embed_dim)
        
        self.action_codec = ActionCodec.from_action_space(action_space, codec_config)
        self._is_discrete = self.action_codec.is_discrete
        self._action_dim = self.action_codec.action_dim
        # Materialize bridge modules before optimizers are created so their
        # parameters are not silently excluded from world-model training.
        self._ensure_v45_components()

    def set_danger_fn(self, fn: Callable[[Tensor], Tensor]) -> None:
        self.__dict__['_danger_fn'] = fn
        if self.__dict__.get('_predictive_engine') is not None:
            self.predictive_engine.set_danger_fn(fn)

    # =========================================================================
    # 编码/解码
    # =========================================================================

    def encode_obs(self, obs: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        return self.encoder(obs)

    def encode_action(self, action: Tensor) -> Tensor:
        if self.action_codec is None:
            raise RuntimeError("ActionCodec not initialized.")
        return self.action_codec.encode(action)

    def decode_vitals(self, features: Tensor) -> Tensor:
        return self.decoder(features)

    # =========================================================================
    # 状态操作
    # =========================================================================

    def get_initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
        if device is None:
            device = self.device
        return self.state_transition.initial_state(batch_size, device)

    def init_wm_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        deterministic: bool = False,
    ) -> WMState:
        if device is None:
            device = self.device
        
        self._ensure_v45_components()
        h, z_raw = self.state_transition.initial_state(batch_size, device, deterministic=deterministic)
        
        if z_raw.dim() == 3:
            z_raw = z_raw.reshape(batch_size, -1)
        
        s_ctrl = self._control_head.initial_state(batch_size, device)
        x_t = torch.zeros(batch_size, self._encoder_out_dim, device=device)
        
        return WMState(
            x_t=x_t, h_shared=h, h_pred=h, z=z_raw,
            s_ctrl=s_ctrl, x_proj=None, z_task=None,
        )

    def get_features(self, h: Tensor, z_raw: Tensor) -> Tensor:
        return self.state_transition.get_features(h, z_raw)

    # =========================================================================
    # Context & Imagination
    # =========================================================================

    def forward_context(
        self,
        obs: Union[Tensor, Dict[str, Tensor]],
        action_prev: Tensor,
        prev_state: Union[WMState, Tuple[Tensor, Tensor]],
        deterministic_state: bool = False,
        **kwargs,
    ) -> Tuple[WMState, RSSMOutputs]:
        
        self._ensure_v45_components()
        
        obs_embed = self.encode_obs(obs)
        action_embed = self.encode_action(action_prev)
        
        prev_state = WMState.coerce(
            prev_state,
            ctrl_dim=self._wm_config.ctrl_dim,
            obs_embed_dim=self._encoder_out_dim,
        )
        
        # 1. RSSM Transition
        transition_out = self.state_transition.observe_step(
            prev_state=(prev_state.h_shared, prev_state.z),
            action_embed=action_embed,
            obs_embed=obs_embed,
            deterministic_state=deterministic_state,
            **kwargs,
        )
        
        h_new = transition_out.h
        z_raw = transition_out.z_post_raw
        if z_raw.dim() == 3:
            z_raw = z_raw.reshape(z_raw.shape[0], -1)
        
        # 2. Control Head
        s_ctrl = self._control_head(
            x_rl=obs_embed,
            action_prev=action_prev,
            s_ctrl_prev=prev_state.s_ctrl,
        )
        
        # 3. Projection & Abstractor
        s_pred = torch.cat([h_new, z_raw], dim=-1)
        
        x_proj_raw = self._projection(s_pred) if self._projection else None
        z_task_raw = self._abstractor(s_pred) if self._abstractor else None
        
        # FIX: Gradient Isolation for x_proj
        # x_proj 用于 Router 输入应与 WM 解耦避免策略梯度流入 WM
        x_proj = x_proj_raw.detach() if x_proj_raw is not None else None
        z_task = z_task_raw.detach() if z_task_raw is not None else None
        
        # 4. Build Outputs
        r_pred = self.reward_head(transition_out.features_post)
        d_pred = self.continue_head(transition_out.features_post)
        o_pred = self.decoder(transition_out.features_post)
        
        rssm_outputs = RSSMOutputs(
            posterior=None, prior=None, z=z_raw,
            o_pred=o_pred, r_pred=r_pred, d_pred=d_pred,
            x_proj=x_proj, z_task=z_task,
            features_post=transition_out.features_post,
            kl_raw=transition_out.kl_raw,
        )
        
        new_state = WMState(
            x_t=obs_embed, h_shared=h_new, h_pred=h_new,
            z=z_raw, s_ctrl=s_ctrl,
            x_proj=x_proj, z_task=z_task,
        )
        
        return new_state, rssm_outputs

    def forward_imagination(
        self,
        action_prev: Tensor,
        prev_state: WMState,
        **kwargs,
    ) -> Tuple[WMState, RSSMOutputs]:
        
        self._ensure_v45_components()
        action_embed = self.encode_action(action_prev)
        
        transition_out = self.state_transition.imagine_step(
            prev_state=(prev_state.h_shared, prev_state.z),
            action_embed=action_embed,
            **kwargs,
        )
        
        h_new = transition_out.h
        z_raw = transition_out.z_prior_raw
        if z_raw.dim() == 3:
            z_raw = z_raw.reshape(z_raw.shape[0], -1)
        
        s_pred = torch.cat([h_new, z_raw], dim=-1)
        x_proj = self._projection(s_pred).detach() if self._projection else None
        z_task = self._abstractor(s_pred).detach() if self._abstractor else None

        # Imagination must advance the control path with the predicted embedding,
        # not the last real observation embedding carried in ``prev_state.x_t``.
        imag_x_t = x_proj if x_proj is not None else prev_state.x_t
        s_ctrl = self._control_head(
            x_rl=imag_x_t,
            action_prev=action_prev,
            s_ctrl_prev=prev_state.s_ctrl,
        )

        r_pred = self.reward_head(transition_out.features_prior)
        d_pred = self.continue_head(transition_out.features_prior)
        o_pred = self.decoder(transition_out.features_prior)
        
        rssm_outputs = RSSMOutputs(
            posterior=None, prior=None, z=z_raw,
            o_pred=o_pred, r_pred=r_pred, d_pred=d_pred,
            x_proj=x_proj, z_task=z_task,
            features_prior=transition_out.features_prior,
        )
        
        new_state = WMState(
            x_t=imag_x_t, h_shared=h_new, h_pred=h_new,
            z=z_raw, s_ctrl=s_ctrl,
            x_proj=x_proj, z_task=z_task,
        )
        
        return new_state, rssm_outputs

    # =========================================================================
    # 序列处理
    # =========================================================================

    def observe_sequence(
        self,
        vitals: Tensor,
        actions: Tensor,
        dones: Optional[Tensor] = None,
        init_state: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        
        B, T = actions.shape[:2]
        if vitals.shape[1] != T + 1:
            raise ValueError("Vitals must be T+1")
        
        vitals_aligned = vitals[:, 1:]
        obs_embeds = self.encode_obs(vitals_aligned)
        action_embeds = self.encode_action(actions)
        
        result = self.state_transition.observe_sequence(
            obs_embeds=obs_embeds,
            action_embeds=action_embeds,
            init_state=init_state,
            dones=dones,
            **kwargs,
        )
        
        result['deter_dim'] = getattr(self.state_transition, 'deter_dim', getattr(self._wm_config, 'deter_dim', 512))
        return result

    # =========================================================================
    # 统一损失计算
    # =========================================================================

    def build_trajectory(
        self,
        seq_result: Dict[str, Any],
        vitals: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        remaining_steps: Optional[Tensor] = None,
    ) -> WMTrajectory:
        return WMTrajectory.from_seq_result(
            seq_result=seq_result,
            vitals=vitals, actions=actions,
            rewards=rewards, dones=dones,
            remaining_steps=remaining_steps,
        )

    def compute_loss(
        self,
        trajectory: WMTrajectory,
        weights: Optional[LossWeightsConfig] = None,
    ) -> PredictiveLossPacket:
        
        if not isinstance(trajectory, WMTrajectory):
            raise TypeError("WorldModel.compute_loss only accepts WMTrajectory.")
        
        if weights is None:
            weights = self._loss_weights_config
        
        loss_packet = self.predictive_engine.compute_loss(
            trajectory=trajectory,
            weights=weights,
            world_model_ref=self,
        )
        
        if self.training:
            self._step_count += 1
        
        return loss_packet

    # =========================================================================
    # 工具方法
    # =========================================================================

    def clear_all_caches(self, force_gc: bool = True) -> None:
        if hasattr(self.state_transition, 'clear_cache'):
            self.state_transition.clear_cache()
        if force_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_info(self) -> Dict[str, Any]:
        return {
            "version": "5.2",
            "feat_dim": self.feat_dim,
            "device": str(self.device),
            "predictive_engine": self.predictive_engine.get_info(),
        }

    def __repr__(self) -> str:
        return f"WorldModel(v5.2, feat_dim={self.feat_dim}, device={self.device})"


# ======================================================================
# 最终注册表初始化
# ======================================================================

# 确保所有类定义后注册表已初始化
_init_registries()
