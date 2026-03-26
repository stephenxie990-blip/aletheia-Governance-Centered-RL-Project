from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

StatefulSection = Tuple[str, Any]
_VALID_RESTORE_MODES = {"strict", "compatible", "skip"}


@dataclass(frozen=True)
class StateDictRestoreIssue:
    section: str
    message: str
    exception_type: str


@dataclass
class StateDictRestoreReport:
    restored_sections: list[str] = field(default_factory=list)
    skipped_sections: list[str] = field(default_factory=list)
    issues: list[StateDictRestoreIssue] = field(default_factory=list)
    section_results: Dict[str, Any] = field(default_factory=dict)


def collect_attr_sections(
    owner: Any,
    field_names: Sequence[str],
    *,
    optional: bool = False,
) -> Tuple[StatefulSection, ...]:
    """Collect ``(checkpoint_key, object)`` sections from named attributes."""
    sections = []
    for field_name in field_names:
        value = getattr(owner, field_name, None)
        if value is None and optional:
            continue
        sections.append((field_name, value))
    return tuple(sections)


def append_state_dict_sections(
    checkpoint: Dict[str, Any],
    sections: Iterable[StatefulSection],
    *,
    skip_missing_objects: bool = False,
) -> Dict[str, Any]:
    """Append ``state_dict()`` sections into a checkpoint payload."""
    for key, obj in sections:
        if obj is None:
            if skip_missing_objects:
                continue
            raise ValueError(f"Checkpoint section '{key}' is missing its source object.")
        if not hasattr(obj, "state_dict"):
            if skip_missing_objects:
                continue
            raise TypeError(
                f"Checkpoint section '{key}' source does not provide state_dict()."
            )
        checkpoint[key] = obj.state_dict()
    return checkpoint


def restore_state_dict_sections(
    checkpoint: Dict[str, Any],
    sections: Iterable[StatefulSection],
    *,
    strict: Optional[bool] = None,
    skip_missing_sections: bool = False,
    skip_missing_objects: bool = False,
    section_restore_modes: Optional[Mapping[str, str]] = None,
    section_load_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> StateDictRestoreReport:
    """Restore ``load_state_dict()`` sections from a checkpoint payload."""
    report = StateDictRestoreReport()
    for key, obj in sections:
        if obj is None:
            if skip_missing_objects:
                continue
            raise ValueError(f"Checkpoint restore target '{key}' is missing.")
        if not hasattr(obj, "load_state_dict"):
            if skip_missing_objects:
                continue
            raise TypeError(
                f"Checkpoint restore target '{key}' does not provide load_state_dict()."
            )
        if key not in checkpoint:
            if skip_missing_sections:
                continue
            raise KeyError(key)
        restore_mode = "strict"
        if section_restore_modes is not None:
            restore_mode = section_restore_modes.get(key, "strict")
        if restore_mode not in _VALID_RESTORE_MODES:
            raise ValueError(
                f"Unsupported checkpoint restore mode '{restore_mode}' for section '{key}'."
            )
        if restore_mode == "skip":
            report.skipped_sections.append(key)
            continue
        kwargs: Dict[str, Any] = dict((section_load_kwargs or {}).get(key, {}))
        if strict is not None and "strict" not in kwargs:
            kwargs["strict"] = strict
        try:
            result = obj.load_state_dict(checkpoint[key], **kwargs)
        except (RuntimeError, TypeError, ValueError) as exc:
            if restore_mode != "compatible":
                raise
            report.skipped_sections.append(key)
            report.issues.append(
                StateDictRestoreIssue(
                    section=key,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                )
            )
            continue
        report.restored_sections.append(key)
        report.section_results[key] = result
    return report
