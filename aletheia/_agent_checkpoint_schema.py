from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from ._checkpoint_schema_common import serialize_effective_training_config
from ._checkpoint_schema_primitives import (
    append_state_dict_sections,
    collect_attr_sections,
    restore_state_dict_sections,
)

_REQUIRED_COMPONENT_FIELDS: Tuple[str, ...] = (
    "world_model",
    "actor",
    "critic",
)

_OPTIONAL_COMPONENT_FIELDS: Tuple[str, ...] = (
    "router",
    "will",
)
logger = logging.getLogger("aletheia.agent_checkpoint_schema")
_VALID_COMPONENT_RESTORE_MODES = {"strict", "compatible", "skip"}


@dataclass(frozen=True)
class AgentCheckpointRestorePolicy:
    restore_required_components: bool = True
    restore_optional_components: bool = True
    required_component_restore_mode: str = "strict"
    optional_component_restore_mode: str = "strict"

    def normalized(self) -> "AgentCheckpointRestorePolicy":
        required_mode = str(self.required_component_restore_mode or "strict").strip().lower()
        optional_mode = str(self.optional_component_restore_mode or "strict").strip().lower()
        if required_mode not in _VALID_COMPONENT_RESTORE_MODES:
            raise ValueError(
                "required_component_restore_mode must be one of "
                f"{sorted(_VALID_COMPONENT_RESTORE_MODES)}."
            )
        if optional_mode not in _VALID_COMPONENT_RESTORE_MODES:
            raise ValueError(
                "optional_component_restore_mode must be one of "
                f"{sorted(_VALID_COMPONENT_RESTORE_MODES)}."
            )
        return AgentCheckpointRestorePolicy(
            restore_required_components=bool(self.restore_required_components),
            restore_optional_components=bool(self.restore_optional_components),
            required_component_restore_mode=required_mode,
            optional_component_restore_mode=optional_mode,
        )


def _normalize_restore_policy(
    *,
    strict: bool,
    restore_policy: Optional[AgentCheckpointRestorePolicy],
) -> AgentCheckpointRestorePolicy:
    if restore_policy is None:
        return AgentCheckpointRestorePolicy(
            required_component_restore_mode="strict",
            optional_component_restore_mode=("strict" if strict else "compatible"),
        )
    if not isinstance(restore_policy, AgentCheckpointRestorePolicy):
        raise TypeError("restore_policy must be an AgentCheckpointRestorePolicy.")
    return restore_policy.normalized()


def build_agent_checkpoint_payload(
    handle: Any,
    *,
    version: str,
    bootstrap_bundle: Mapping[str, Any],
    effective_training_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the canonical agent checkpoint payload from an AgentHandle-like object."""
    checkpoint: Dict[str, Any] = {
        "version": version,
        "agent_bootstrap_bundle": copy.deepcopy(dict(bootstrap_bundle)),
        "step_count": int(getattr(handle, "_step_count", 0)),
    }
    append_state_dict_sections(
        checkpoint,
        collect_attr_sections(handle, _REQUIRED_COMPONENT_FIELDS),
    )
    append_state_dict_sections(
        checkpoint,
        collect_attr_sections(handle, _OPTIONAL_COMPONENT_FIELDS, optional=True),
        skip_missing_objects=True,
    )

    serialized_effective_config = serialize_effective_training_config(
        effective_training_config
    )
    if serialized_effective_config is not None:
        checkpoint["effective_training_config"] = serialized_effective_config

    return checkpoint


def restore_agent_checkpoint_modules(
    handle: Any,
    checkpoint: Mapping[str, Any],
    *,
    strict: bool = True,
    restore_policy: Optional[AgentCheckpointRestorePolicy] = None,
    checkpoint_path: Optional[str] = None,
) -> int:
    """Restore model component states from the canonical agent checkpoint payload."""
    policy = _normalize_restore_policy(strict=strict, restore_policy=restore_policy)
    required_restore_mode = (
        policy.required_component_restore_mode
        if policy.restore_required_components
        else "skip"
    )
    optional_restore_mode = (
        policy.optional_component_restore_mode
        if policy.restore_optional_components
        else "skip"
    )

    required_report = restore_state_dict_sections(
        checkpoint,
        collect_attr_sections(handle, _REQUIRED_COMPONENT_FIELDS),
        strict=None,
        section_restore_modes={
            field: required_restore_mode
            for field in _REQUIRED_COMPONENT_FIELDS
        },
        section_load_kwargs={
            field: {"strict": required_restore_mode == "strict"}
            for field in _REQUIRED_COMPONENT_FIELDS
        },
    )
    optional_report = restore_state_dict_sections(
        checkpoint,
        collect_attr_sections(handle, _OPTIONAL_COMPONENT_FIELDS, optional=True),
        strict=None,
        skip_missing_sections=True,
        skip_missing_objects=True,
        section_restore_modes={
            field: optional_restore_mode
            for field in _OPTIONAL_COMPONENT_FIELDS
        },
        section_load_kwargs={
            field: {"strict": optional_restore_mode == "strict"}
            for field in _OPTIONAL_COMPONENT_FIELDS
        },
    )
    for report, mode in (
        (required_report, policy.required_component_restore_mode),
        (optional_report, policy.optional_component_restore_mode),
    ):
        for issue in report.issues:
            logger.warning(
                "Agent checkpoint restore skipped %s for %s in %s mode: %s (%s)",
                issue.section,
                checkpoint_path or "<checkpoint>",
                mode,
                issue.message,
                issue.exception_type,
            )

    return int(checkpoint.get("step_count", 0))
