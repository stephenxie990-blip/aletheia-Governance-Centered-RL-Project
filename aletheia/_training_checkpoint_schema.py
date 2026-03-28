from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ._checkpoint_schema_common import serialize_effective_training_config
from ._checkpoint_schema_primitives import (
    append_state_dict_sections,
    restore_state_dict_sections,
)

logger = logging.getLogger("aletheia.training_checkpoint_schema")
_VALID_OPTIMIZER_RESTORE_MODES = {"auto", "strict", "compatible", "skip"}
_VALID_MODEL_RESTORE_MODES = {"strict"}


@dataclass(frozen=True)
class TrainingCheckpointRestorePolicy:
    restore_training_state: bool = True
    restore_model: bool = True
    restore_optimizers: bool = True
    restore_buffer: bool = True
    model_restore_mode: str = "strict"
    optimizer_restore_mode: str = "auto"

    def normalized(self) -> "TrainingCheckpointRestorePolicy":
        model_mode = str(self.model_restore_mode or "strict").strip().lower()
        mode = str(self.optimizer_restore_mode or "auto").strip().lower()
        if model_mode not in _VALID_MODEL_RESTORE_MODES:
            raise ValueError(
                "model_restore_mode must be one of "
                f"{sorted(_VALID_MODEL_RESTORE_MODES)}."
            )
        if mode not in _VALID_OPTIMIZER_RESTORE_MODES:
            raise ValueError(
                "optimizer_restore_mode must be one of "
                f"{sorted(_VALID_OPTIMIZER_RESTORE_MODES)}."
            )
        return TrainingCheckpointRestorePolicy(
            restore_training_state=bool(self.restore_training_state),
            restore_model=bool(self.restore_model),
            restore_optimizers=bool(self.restore_optimizers),
            restore_buffer=bool(self.restore_buffer),
            model_restore_mode=model_mode,
            optimizer_restore_mode=mode,
        )


def _normalize_restore_policy(
    restore_policy: Optional[TrainingCheckpointRestorePolicy],
) -> TrainingCheckpointRestorePolicy:
    if restore_policy is None:
        return TrainingCheckpointRestorePolicy()
    if not isinstance(restore_policy, TrainingCheckpointRestorePolicy):
        raise TypeError("restore_policy must be a TrainingCheckpointRestorePolicy.")
    return restore_policy.normalized()


def _resolve_auto_optimizer_restore_mode(
    checkpoint: Mapping[str, Any],
    *,
    current_effective_training_config: Optional[Any],
) -> tuple[str, Optional[str]]:
    checkpoint_effective_config = serialize_effective_training_config(
        checkpoint.get("effective_training_config")
    )
    current_effective_config = serialize_effective_training_config(
        current_effective_training_config
    )
    if checkpoint_effective_config is None:
        return "compatible", "checkpoint lacks effective training config metadata"
    if current_effective_config is None:
        return "compatible", "current effective training config is unavailable"
    if checkpoint_effective_config != current_effective_config:
        return "compatible", "effective training config drift detected"
    return "strict", None


def build_training_checkpoint_payload(
    state: Any,
    *,
    model: Optional[Any] = None,
    opt_bundle: Optional[Any] = None,
    buffer: Optional[Any] = None,
    effective_training_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the canonical training checkpoint payload."""
    serialized_effective_config = serialize_effective_training_config(
        effective_training_config
    )
    if serialized_effective_config is None:
        serialized_effective_config = dict(state.config.to_dict())

    checkpoint: Dict[str, Any] = {
        "training_state": state.state_dict(),
        "global_step": int(state.global_step),
        "effective_training_config": serialized_effective_config,
        "config_hash": state.config.compute_hash(),
    }
    append_state_dict_sections(
        checkpoint,
        (
            ("model", model),
            ("optimizer_bundle", opt_bundle),
            ("replay_buffer", buffer),
        ),
        skip_missing_objects=True,
    )
    return checkpoint


def restore_training_checkpoint_payload(
    checkpoint: Mapping[str, Any],
    *,
    state: Any,
    model: Optional[Any] = None,
    opt_bundle: Optional[Any] = None,
    buffer: Optional[Any] = None,
    checkpoint_path: Optional[str] = None,
    restore_policy: Optional[TrainingCheckpointRestorePolicy] = None,
    current_effective_training_config: Optional[Any] = None,
) -> Any:
    """Restore training objects from the canonical checkpoint payload."""
    policy = _normalize_restore_policy(restore_policy)

    if policy.restore_training_state and "training_state" in checkpoint:
        state.load_state_dict(checkpoint["training_state"])

    optimizer_restore_mode = (
        policy.optimizer_restore_mode
        if policy.restore_optimizers
        else "skip"
    )
    optimizer_restore_reason: Optional[str] = None
    if optimizer_restore_mode == "auto":
        optimizer_restore_mode, optimizer_restore_reason = (
            _resolve_auto_optimizer_restore_mode(
                checkpoint,
                current_effective_training_config=current_effective_training_config,
            )
        )
        if optimizer_restore_reason is not None:
            logger.warning(
                "Training checkpoint restore using optimizer %s mode for %s: %s",
                optimizer_restore_mode,
                checkpoint_path or "<checkpoint>",
                optimizer_restore_reason,
            )

    if policy.restore_model and model is not None and "model" in checkpoint:
        world_model = getattr(model, "world_model", None)
        if world_model is not None and hasattr(world_model, "_ensure_v45_components"):
            try:
                world_model._ensure_v45_components()
            except Exception as exc:
                raise RuntimeError(
                    "Training checkpoint restore failed for "
                    f"{checkpoint_path or '<checkpoint>'}: "
                    f"world_model ensure_v45_components failed: {exc}"
                ) from exc
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Training checkpoint restore failed for "
                f"{checkpoint_path or '<checkpoint>'}: model state is incompatible: {exc}"
            ) from exc

    restore_report = restore_state_dict_sections(
        checkpoint,
        (
            ("optimizer_bundle", opt_bundle),
            ("replay_buffer", buffer),
        ),
        skip_missing_sections=True,
        skip_missing_objects=True,
        section_restore_modes={
            "optimizer_bundle": (
                optimizer_restore_mode if policy.restore_optimizers else "skip"
            ),
            "replay_buffer": "strict" if policy.restore_buffer else "skip",
        },
        section_load_kwargs={
            "optimizer_bundle": {"restore_mode": optimizer_restore_mode},
        },
    )
    for issue in restore_report.issues:
        logger.warning(
            "Training checkpoint restore skipped %s for %s in %s mode: %s (%s)",
            issue.section,
            checkpoint_path or "<checkpoint>",
            optimizer_restore_mode,
            issue.message,
            issue.exception_type,
        )
    optimizer_report = restore_report.section_results.get("optimizer_bundle")
    if isinstance(optimizer_report, dict):
        for issue in optimizer_report.get("issues", []):
            logger.warning(
                "Training checkpoint restore skipped optimizer_bundle/%s for %s in %s mode: %s (%s)",
                issue.get("optimizer", "<unknown>"),
                checkpoint_path or "<checkpoint>",
                optimizer_restore_mode,
                issue.get("message", ""),
                issue.get("exception_type", "Exception"),
            )

    return state
