from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .aletheia_config import ConfigBundle

logger = logging.getLogger("aletheia.checkpoint_metadata")


def _torch_load_compat(
    path: Path,
    *,
    map_location: Any,
    weights_only: bool,
) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError as exc:
        if "unexpected keyword argument 'weights_only'" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)


def read_checkpoint(
    path: Union[str, Path],
    *,
    map_location: Any = "cpu",
    weights_only: bool = False,
    allow_unsafe_fallback: bool = False,
    trusted_source: bool = False,
    fail_soft: bool = False,
) -> Any:
    """Read a checkpoint payload with the repository's unified load semantics."""
    checkpoint_path = Path(path)
    try:
        from .aletheia_foundation import safe_torch_load

        return safe_torch_load(
            checkpoint_path,
            map_location=map_location,
            weights_only=weights_only,
            allow_unsafe_fallback=allow_unsafe_fallback,
            trusted_source=trusted_source,
        )
    except ImportError:
        return _torch_load_compat(
            checkpoint_path,
            map_location=map_location,
            weights_only=weights_only,
        )
    except Exception as exc:
        if not fail_soft:
            raise
        logger.debug(
            "Failed to read checkpoint metadata from %s: %s",
            checkpoint_path,
            exc,
        )
        return None


def read_checkpoint_metadata(
    path: Union[str, Path],
    *,
    allow_unsafe_fallback: bool = False,
    fail_soft: bool = False,
) -> Any:
    """Read checkpoint payload for metadata inspection."""
    if allow_unsafe_fallback:
        raise ValueError(
            "allow_unsafe_fallback is no longer supported in checkpoint metadata readers"
        )
    return read_checkpoint(
        path,
        map_location="cpu",
        weights_only=True,
        fail_soft=fail_soft,
    )


def read_agent_creation_overrides_from_checkpoint(
    path: Union[str, Path],
    *,
    allow_unsafe_fallback: bool = False,
    fail_soft: bool = False,
) -> Optional[Dict[str, Any]]:
    """Read create_agent() overrides from checkpoint bootstrap metadata."""
    if allow_unsafe_fallback:
        raise ValueError(
            "allow_unsafe_fallback is no longer supported when reading agent checkpoint metadata"
        )
    checkpoint = read_checkpoint_metadata(
        path,
        fail_soft=fail_soft,
    )
    return ConfigBundle.agent_creation_overrides_from_checkpoint_metadata(checkpoint)


def read_training_checkpoint(
    path: Union[str, Path],
    *,
    map_location: Any = "cpu",
    allow_unsafe_fallback: bool = False,
    trusted_source: bool = False,
) -> Any:
    """Read a training checkpoint using the repository's canonical training reader semantics."""
    if allow_unsafe_fallback:
        raise ValueError(
            "allow_unsafe_fallback is no longer supported in training checkpoint readers"
        )
    return read_checkpoint(
        path,
        map_location=map_location,
        weights_only=True,
        trusted_source=trusted_source,
        fail_soft=False,
    )
