from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import torch.nn as nn
from torch import Tensor

# Re-export: canonical definitions live in aletheia.contracts.hold_state
from ..contracts.hold_state import (
    HOLD_CHANNELS,
    active_hold_channel,
    hold_state_summary,
    hold_state_to_float,
    make_empty_hold_state,
    select_hold_channel_value,
    update_hold_state_channels,
)

MINIMAL_COMPENSATION_STATE_VERSION = 2


class CompensationPhase:
    IDLE = "idle"
    TRIGGER = "trigger"
    RELEASE = "release"
    PERSISTENCE = "persistence"
    POST_SOLVED = "post_solved"
    ALL = (IDLE, TRIGGER, RELEASE, PERSISTENCE, POST_SOLVED)


TRIGGER_STAGE_ALIASES = frozenset(
    {
        "pretrigger",
        "entry_probe",
        "internal_post_entry",
        "post_entry_soft",
        "post_entry_pending",
        "post_entry",
        "handoff",
        "trigger",
        "release",
    }
)
PERSISTENCE_STAGE_ALIASES = frozenset({"persistence", "persistence_release"})
logger = logging.getLogger("aletheia.compensation_kernel")

__all__ = [
    "CompensationDecision",
    "CompensationPhase",
    "HOLD_CHANNELS",
    "MINIMAL_COMPENSATION_STATE_VERSION",
    "MinimalCompensationState",
    "active_hold_channel",
    "build_runtime_rl_context",
    "canonicalize_compensation_phase",
    "detect_runtime_compensation_guard_mismatch",
    "export_adaptive_compensation_state",
    "export_minimal_compensation_state",
    "handle_external_eval_feedback",
    "hold_state_summary",
    "hold_state_to_float",
    "initialize_compensation_runtime_state",
    "is_trigger_persistence_handoff_landing_guard_active",
    "make_empty_hold_state",
    "resolve_imag_compensation_phase",
    "resolve_imag_continue_cap",
    "resolve_adaptive_eval_confirmation_count",
    "resolve_persistence_escape_soft_floors",
    "resolve_post_entry_soft_negative_adv_profile",
    "resolve_post_trigger_quality_state",
    "resolve_standard_soft_fallback_trigger_release_progress",
    "restore_adaptive_compensation_state",
    "restore_minimal_compensation_state",
    "select_hold_channel_value",
    "is_persistence_drop_confirmation_pending",
    "summarize_compensation_restore_report",
    "update_adaptive_eval_confirmation",
    "update_hold_state_channels",
]


def canonicalize_compensation_phase(stage: Any) -> str:
    stage_name = str(stage or CompensationPhase.IDLE)
    if stage_name in TRIGGER_STAGE_ALIASES:
        if stage_name == CompensationPhase.RELEASE:
            return CompensationPhase.RELEASE
        return CompensationPhase.TRIGGER
    if stage_name in PERSISTENCE_STAGE_ALIASES:
        return CompensationPhase.PERSISTENCE
    if stage_name == CompensationPhase.POST_SOLVED:
        return CompensationPhase.POST_SOLVED
    return CompensationPhase.IDLE


# hold_state functions (HOLD_CHANNELS, make_empty_hold_state, etc.) are
# re-exported from aletheia.contracts.hold_state at the top of this file.


@dataclass(frozen=True)
class CompensationDecision(Mapping[str, Any]):
    compensation_phase: str = CompensationPhase.IDLE
    trigger_active: bool = False
    trigger_confirmation_ready: bool = False
    release_active: bool = False
    release_progress: float = 0.0
    persistence_active: bool = False
    persistence_seen: bool = False
    post_solved_active: bool = False
    persistence_escape_active: bool = False
    post_entry_source: str = "idle"
    trigger_persistence_handoff_active: bool = False
    trigger_persistence_handoff_release_progress: float = 0.0
    trigger_persistence_handoff_eval_mean: float = 0.0
    episode_return_ema: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compensation_phase",
            canonicalize_compensation_phase(self.compensation_phase),
        )

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    def keys(self) -> Iterator[str]:
        return iter(self)

    def items(self) -> Iterator[tuple[str, Any]]:
        for key in self:
            yield key, getattr(self, key)

    def with_updates(self, **changes: Any) -> "CompensationDecision":
        if "stage" in changes and "compensation_phase" not in changes:
            changes["compensation_phase"] = changes.pop("stage")
        return replace(self, **changes)

    @property
    def stage(self) -> str:
        return self.compensation_phase


@dataclass(frozen=True)
class MinimalCompensationState:
    compensation_phase: str = CompensationPhase.IDLE
    continue_cap_state: float | None = None
    post_trigger_actor_scale_state: float | None = None
    contact_surface_state: float = 0.0
    bonus_hold_state: dict[str, float] = field(default_factory=make_empty_hold_state)
    post_transition_certified_floor_state: float = 0.0
    persistence_hold_until_step: int = -1
    persistence_armed_step: int = -1
    persistence_entry_protect_until_step: int = -1
    persistence_release_hold_until_step: int = -1
    post_solved_hold_until_step: int = -1

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "MinimalCompensationState":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("Minimal compensation payload must be a mapping.")
        schema_version = int(payload.get("schema_version", -1))
        if schema_version != MINIMAL_COMPENSATION_STATE_VERSION:
            raise ValueError(
                f"Unsupported minimal compensation payload schema_version: {schema_version}"
            )
        allowed_keys = {
            "schema_version",
            "compensation_phase",
            "continue_cap_state",
            "post_trigger_actor_scale_state",
            "contact_surface_state",
            "bonus_hold_state",
            "post_transition_certified_floor_state",
            "persistence",
            "post_solved",
        }
        unexpected_keys = sorted(str(key) for key in payload.keys() if key not in allowed_keys)
        if unexpected_keys:
            raise ValueError(
                "Minimal compensation payload contains unsupported keys: "
                + ", ".join(unexpected_keys)
            )
        raw_hold_state = payload.get("bonus_hold_state", make_empty_hold_state())
        if not isinstance(raw_hold_state, Mapping):
            raise ValueError("Minimal compensation payload bonus_hold_state must be a mapping.")
        hold_channels = set(raw_hold_state.keys())
        expected_channels = set(HOLD_CHANNELS)
        if hold_channels != expected_channels:
            raise ValueError(
                "Minimal compensation payload bonus_hold_state must contain exactly "
                f"{sorted(expected_channels)}; got {sorted(hold_channels)}."
            )
        hold_state = {
            channel: float(raw_hold_state[channel])
            for channel in HOLD_CHANNELS
        }
        continue_cap_state = payload.get("continue_cap_state")
        post_trigger_actor_scale_state = payload.get("post_trigger_actor_scale_state")
        persistence = payload.get("persistence", {})
        if not isinstance(persistence, Mapping):
            raise ValueError("Minimal compensation payload persistence must be a mapping.")
        post_solved = payload.get("post_solved", {})
        if not isinstance(post_solved, Mapping):
            raise ValueError("Minimal compensation payload post_solved must be a mapping.")
        return cls(
            compensation_phase=canonicalize_compensation_phase(
                payload.get("compensation_phase", CompensationPhase.IDLE)
            ),
            continue_cap_state=None if continue_cap_state is None else float(continue_cap_state),
            post_trigger_actor_scale_state=(
                None if post_trigger_actor_scale_state is None else float(post_trigger_actor_scale_state)
            ),
            contact_surface_state=float(payload.get("contact_surface_state", 0.0)),
            bonus_hold_state=hold_state,
            post_transition_certified_floor_state=float(
                payload.get("post_transition_certified_floor_state", 0.0)
            ),
            persistence_hold_until_step=int(persistence.get("hold_until_step", -1)),
            persistence_armed_step=int(persistence.get("armed_step", -1)),
            persistence_entry_protect_until_step=int(
                persistence.get("entry_protect_until_step", -1)
            ),
            persistence_release_hold_until_step=int(
                persistence.get("release_hold_until_step", -1)
            ),
            post_solved_hold_until_step=int(post_solved.get("hold_until_step", -1)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": int(MINIMAL_COMPENSATION_STATE_VERSION),
            "compensation_phase": str(self.compensation_phase),
            "continue_cap_state": self.continue_cap_state,
            "post_trigger_actor_scale_state": self.post_trigger_actor_scale_state,
            "contact_surface_state": float(self.contact_surface_state),
            "bonus_hold_state": {
                channel: float(self.bonus_hold_state.get(channel, 0.0))
                for channel in HOLD_CHANNELS
            },
            "post_transition_certified_floor_state": float(
                self.post_transition_certified_floor_state
            ),
            "persistence": {
                "hold_until_step": int(self.persistence_hold_until_step),
                "armed_step": int(self.persistence_armed_step),
                "entry_protect_until_step": int(self.persistence_entry_protect_until_step),
                "release_hold_until_step": int(self.persistence_release_hold_until_step),
            },
            "post_solved": {
                "hold_until_step": int(self.post_solved_hold_until_step),
            },
        }

    @classmethod
    def capture_from(cls, runtime: Any) -> "MinimalCompensationState":
        raw_hold_state = getattr(runtime, "_adaptive_imag_critic_bootstrap_bonus_hold_state", 0.0)
        normalized_hold_state = (
            {
                channel: float(raw_hold_state.get(channel, 0.0))
                for channel in HOLD_CHANNELS
            }
            if isinstance(raw_hold_state, dict)
            else {channel: float(raw_hold_state) for channel in HOLD_CHANNELS}
        )
        return cls(
            compensation_phase=canonicalize_compensation_phase(
                getattr(runtime, "_adaptive_imag_compensation_phase", CompensationPhase.IDLE)
            ),
            continue_cap_state=getattr(runtime, "_adaptive_imag_continue_cap_state", None),
            post_trigger_actor_scale_state=getattr(
                runtime, "_adaptive_imag_compensation_trigger_actor_scale_state", None
            ),
            contact_surface_state=float(
                getattr(runtime, "_adaptive_imag_critic_bootstrap_contact_surface_state", 0.0)
            ),
            bonus_hold_state=normalized_hold_state,
            post_transition_certified_floor_state=float(
                getattr(
                    runtime,
                    "_adaptive_imag_critic_bootstrap_post_transition_certified_floor_state",
                    0.0,
                )
            ),
            persistence_hold_until_step=int(
                getattr(runtime, "_adaptive_imag_compensation_persistence_hold_until_step", -1)
            ),
            persistence_armed_step=int(
                getattr(runtime, "_adaptive_imag_compensation_persistence_armed_step", -1)
            ),
            persistence_entry_protect_until_step=int(
                getattr(
                    runtime,
                    "_adaptive_imag_compensation_persistence_entry_protect_until_step",
                    -1,
                )
            ),
            persistence_release_hold_until_step=int(
                getattr(
                    runtime,
                    "_adaptive_imag_compensation_persistence_release_hold_until_step",
                    -1,
                )
            ),
            post_solved_hold_until_step=int(
                getattr(runtime, "_adaptive_imag_compensation_post_solved_hold_until_step", -1)
            ),
        )

    def apply_to_runtime(self, runtime: Any) -> None:
        setattr(runtime, "_adaptive_imag_compensation_phase", canonicalize_compensation_phase(self.compensation_phase))
        setattr(runtime, "_adaptive_imag_continue_cap_state", self.continue_cap_state)
        setattr(
            runtime,
            "_adaptive_imag_compensation_trigger_actor_scale_state",
            self.post_trigger_actor_scale_state,
        )
        setattr(
            runtime,
            "_adaptive_imag_critic_bootstrap_contact_surface_state",
            float(self.contact_surface_state),
        )
        setattr(
            runtime,
            "_adaptive_imag_critic_bootstrap_bonus_hold_state",
            {
                channel: float(self.bonus_hold_state.get(channel, 0.0))
                for channel in HOLD_CHANNELS
            },
        )
        setattr(
            runtime,
            "_adaptive_imag_critic_bootstrap_post_transition_certified_floor_state",
            float(self.post_transition_certified_floor_state),
        )
        setattr(
            runtime,
            "_adaptive_imag_compensation_persistence_hold_until_step",
            int(self.persistence_hold_until_step),
        )
        setattr(
            runtime,
            "_adaptive_imag_compensation_persistence_armed_step",
            int(self.persistence_armed_step),
        )
        setattr(
            runtime,
            "_adaptive_imag_compensation_persistence_entry_protect_until_step",
            int(self.persistence_entry_protect_until_step),
        )
        setattr(
            runtime,
            "_adaptive_imag_compensation_persistence_release_hold_until_step",
            int(self.persistence_release_hold_until_step),
        )
        setattr(
            runtime,
            "_adaptive_imag_compensation_post_solved_hold_until_step",
            int(self.post_solved_hold_until_step),
        )


def export_minimal_compensation_state(runtime: Any) -> dict[str, Any]:
    return MinimalCompensationState.capture_from(runtime).to_payload()


def restore_minimal_compensation_state(runtime: Any, payload: Any) -> None:
    if payload is None:
        return
    if not isinstance(payload, Mapping):
        raise ValueError("Minimal compensation payload must be a mapping.")
    MinimalCompensationState.from_payload(payload).apply_to_runtime(runtime)


def _compute_reward_health(latest_reward: float, best_reward: float) -> float:
    latest = float(latest_reward)
    best = float(best_reward)
    if not math.isfinite(latest) or not math.isfinite(best) or best <= 1e-6:
        return 1.0
    return min(1.0, max(0.0, latest / max(best, 1e-6)))


def resolve_adaptive_eval_confirmation_count(
    config: Any,
    specific_key: str,
) -> int:
    shared = max(1, int(getattr(config, "adaptive_imag_eval_confirmation_count", 1)))
    specific = int(getattr(config, specific_key, 0))
    if specific > 0:
        return max(1, specific)
    return shared


def update_adaptive_eval_confirmation(
    runtime: Any,
    *,
    mean_return: float,
    threshold: float,
    streak_attr: str,
    count_key: str,
    confirmed_best_attr: Optional[str] = None,
) -> Tuple[int, int, bool]:
    count = resolve_adaptive_eval_confirmation_count(runtime.config, count_key)
    if threshold > 0.0 and mean_return >= threshold:
        streak = int(getattr(runtime, streak_attr, 0)) + 1
    else:
        streak = 0
    setattr(runtime, streak_attr, streak)
    confirmed = threshold > 0.0 and streak >= count
    if confirmed and confirmed_best_attr is not None:
        prev_best = float(getattr(runtime, confirmed_best_attr, -float("inf")))
        if mean_return > prev_best:
            setattr(runtime, confirmed_best_attr, float(mean_return))
    return streak, count, confirmed


def is_persistence_drop_confirmation_pending(runtime: Any) -> bool:
    required = max(
        1,
        int(
            getattr(
                runtime.config,
                "adaptive_imag_compensation_persistence_drop_confirmation_count",
                0,
            )
        ),
    )
    if required <= 1:
        return False
    persistence_threshold = float(
        getattr(runtime.config, "adaptive_imag_compensation_persistence_eval_threshold", 0.0)
    )
    if persistence_threshold <= 0.0:
        return False
    confirmed_best_mean = float(
        getattr(runtime, "_adaptive_imag_compensation_persistence_confirmed_best_mean", -float("inf"))
    )
    if confirmed_best_mean < persistence_threshold:
        return False
    return (
        int(getattr(runtime, "_adaptive_imag_compensation_persistence_drop_confirmation_streak", 0))
        < required
    )


def initialize_compensation_runtime_state(runtime: Any) -> None:
    runtime._adaptive_imag_continue_cap_state = None
    runtime._adaptive_imag_compensation_trigger_actor_scale_state = None
    runtime._adaptive_imag_compensation_trigger_pressure_streak = 0
    runtime._adaptive_imag_compensation_trigger_quality_good_streak = 0
    runtime._adaptive_imag_late_trigger_rescue_until_step = -1
    runtime._adaptive_imag_compensation_phase = CompensationPhase.IDLE
    runtime._adaptive_imag_critic_bootstrap_contact_surface_state = 0.0
    runtime._adaptive_imag_critic_bootstrap_bonus_hold_state = make_empty_hold_state()
    runtime._adaptive_imag_critic_bootstrap_post_transition_certified_floor_state = 0.0
    runtime._adaptive_imag_entry_probe_hold_until_step = -1
    runtime._adaptive_imag_post_entry_hold_until_step = -1
    runtime._adaptive_imag_post_entry_commit_highwater_hold_until_step = -1
    runtime._adaptive_imag_post_entry_commit_candidate_since_step = -1
    runtime._adaptive_imag_handoff_hold_until_step = -1
    runtime._adaptive_imag_post_entry_source = "idle"
    runtime._adaptive_imag_last_post_entry_source = "idle"
    runtime._adaptive_imag_external_post_entry_pending_commit = False
    runtime._adaptive_imag_external_post_entry_soft_fallback_latched = False
    runtime._adaptive_imag_standard_soft_fallback_trigger_release_step = -1
    runtime._adaptive_imag_post_entry_pending_recovery_latched = False
    runtime._adaptive_imag_post_entry_commit_ever_armed = False
    runtime._adaptive_imag_compensation_persistence_hold_until_step = -1
    runtime._adaptive_imag_compensation_persistence_armed_step = -1
    runtime._adaptive_imag_compensation_persistence_entry_protect_until_step = -1
    runtime._adaptive_imag_compensation_persistence_release_hold_until_step = -1
    runtime._adaptive_imag_compensation_persistence_release_recent_until_step = -1
    runtime._adaptive_imag_compensation_persistence_release_tail_bypass_sustain_until_step = -1
    runtime._adaptive_imag_compensation_trigger_persistence_handoff_blocked_until_step = -1
    runtime._adaptive_imag_compensation_trigger_persistence_handoff_confirmation_pending_after_late_trigger_base_cap = False
    runtime._adaptive_imag_compensation_trigger_persistence_handoff_confirmation_streak_after_late_trigger_base_cap = 0
    runtime._adaptive_imag_compensation_persistence_ever_armed = False
    runtime._adaptive_imag_compensation_persistence_ever_active = False
    runtime._adaptive_imag_compensation_post_solved_hold_until_step = -1
    runtime._adaptive_imag_compensation_post_solved_negative_adv_latched_until_step = -1
    runtime._adaptive_imag_compensation_post_solved_actor_anchor_params = {}
    runtime._adaptive_imag_compensation_post_solved_actor_anchor_actor = None
    runtime._adaptive_imag_compensation_post_solved_actor_anchor_step = -1
    runtime._adaptive_imag_compensation_post_solved_actor_anchor_eval = -float("inf")
    runtime._adaptive_imag_compensation_post_solved_critic_anchor_critic = None
    runtime._adaptive_imag_compensation_post_solved_critic_anchor_step = -1
    runtime._adaptive_imag_compensation_post_solved_critic_anchor_eval = -float("inf")
    runtime._external_eval_last_mean = -float("inf")
    runtime._external_eval_best_mean = -float("inf")
    runtime._external_eval_last_step = -1
    runtime._bootstrap_external_eval_feedback_last_step = -1
    runtime._bootstrap_external_eval_feedback_last_mean = -float("inf")
    runtime._bootstrap_external_eval_feedback_best_mean = -float("inf")
    runtime._bootstrap_external_eval_feedback_telemetry = {}
    runtime._adaptive_imag_post_entry_eval_confirmation_streak = 0
    runtime._adaptive_imag_compensation_persistence_eval_confirmation_streak = 0
    runtime._adaptive_imag_compensation_persistence_drop_confirmation_streak = 0
    runtime._adaptive_imag_compensation_post_solved_eval_confirmation_streak = 0
    runtime._adaptive_imag_compensation_persistence_confirmed_best_mean = -float("inf")
    runtime._adaptive_imag_compensation_post_solved_confirmed_best_mean = -float("inf")
    runtime._adaptive_imag_compensation_persistence_adv_ema = 0.0
    runtime._adaptive_imag_compensation_persistence_continue_ema = 0.0
    runtime._adaptive_imag_compensation_persistence_gap_ema = 0.0
    runtime._adaptive_imag_compensation_persistence_ema_initialized = False
    runtime._adaptive_imag_entry_probe_prev_continue = 0.0
    runtime._adaptive_imag_entry_probe_prev_gap = 0.0
    runtime._adaptive_imag_entry_probe_prev_return = 0.0
    runtime._adaptive_imag_entry_probe_prev_initialized = False
    runtime._adaptive_imag_entry_probe_continue_history = deque()
    runtime._adaptive_imag_entry_probe_gap_history = deque()
    runtime._adaptive_imag_entry_probe_return_history = deque()
    runtime._behavior_policy_eval_anchor_actor = None
    runtime._behavior_policy_eval_anchor_step = -1
    runtime._behavior_policy_eval_anchor_eval = -float("inf")
    runtime._real_stability_last_telemetry = {}
    runtime._real_stability_last_step = -1
    runtime._bootstrap_runtime_task_cert_telemetry = {}
    runtime._real_stability_certified_anchor_actor = None
    runtime._real_stability_certified_anchor_step = -1
    runtime._real_stability_certified_anchor_eval = -float("inf")
    runtime._real_stability_certified_telemetry = {}
    runtime._real_stability_certified_anchor_registry_actors = []
    runtime._real_stability_certified_anchor_registry_steps = []
    runtime._real_stability_certified_anchor_registry_evals = []
    runtime._real_stability_certified_anchor_registry_telemetries = []
    runtime._runtime_compensation_guard_mismatch_warned = False
    runtime._adaptive_compensation_restore_report = {
        "status": "not_restored",
        "issues": [],
    }


def _make_compensation_restore_report() -> Dict[str, Any]:
    return {
        "status": "restored",
        "issues": [],
        "post_solved_anchor": {
            "status": "not_attempted",
            "issues": [],
        },
        "behavior_policy_anchor": {
            "status": "not_attempted",
            "issues": [],
        },
        "real_stability_registry": {
            "status": "not_attempted",
            "restored_entries": 0,
            "skipped_entries": 0,
            "issues": [],
        },
    }


def _record_restore_degradation(
    report: Dict[str, Any],
    *,
    section: str,
    section_report: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
) -> None:
    status = str(section_report.get("status", "restored"))
    issues = [str(item) for item in list(section_report.get("issues", []))]
    degraded = status not in {"restored", "not_attempted"}
    if degraded and not issues:
        issues.append(f"{section} restore status={status}")
    normalized_section_report = dict(section_report)
    normalized_section_report["status"] = status
    normalized_section_report["issues"] = issues
    report[section] = normalized_section_report
    report["issues"].extend(issues)
    if degraded:
        report["status"] = "degraded"
    if issues:
        logger.warning(
            "Adaptive compensation restore degraded in %s for %s: %s",
            section,
            checkpoint_path or "<checkpoint>",
            " | ".join(str(item) for item in issues),
        )


def summarize_compensation_restore_report(
    report: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = dict(report or {})
    issues = [str(item) for item in list(payload.get("issues", []))]
    post_solved_anchor = dict(payload.get("post_solved_anchor", {}) or {})
    behavior_policy_anchor = dict(payload.get("behavior_policy_anchor", {}) or {})
    real_stability_registry = dict(payload.get("real_stability_registry", {}) or {})
    status = str(payload.get("status", "not_restored"))
    degraded = status == "degraded"
    return {
        "status": status,
        "degraded": degraded,
        "issue_count": int(len(issues)),
        "issues": issues,
        "post_solved_anchor_status": str(
            post_solved_anchor.get("status", "not_attempted")
        ),
        "post_solved_anchor_issue_count": int(
            len(list(post_solved_anchor.get("issues", [])))
        ),
        "behavior_policy_anchor_status": str(
            behavior_policy_anchor.get("status", "not_attempted")
        ),
        "behavior_policy_anchor_issue_count": int(
            len(list(behavior_policy_anchor.get("issues", [])))
        ),
        "real_stability_registry_status": str(
            real_stability_registry.get("status", "not_attempted")
        ),
        "real_stability_registry_restored_entries": int(
            real_stability_registry.get("restored_entries", 0)
        ),
        "real_stability_registry_skipped_entries": int(
            real_stability_registry.get("skipped_entries", 0)
        ),
        "real_stability_registry_issue_count": int(
            len(list(real_stability_registry.get("issues", [])))
        ),
    }


def export_adaptive_compensation_state(runtime: Any) -> Dict[str, Any]:
    minimal_payload = export_minimal_compensation_state(runtime)
    return {
        "minimal_compensation_state": minimal_payload,
        "adaptive_imag_compensation_trigger_pressure_streak": int(
            runtime._adaptive_imag_compensation_trigger_pressure_streak
        ),
        "adaptive_imag_compensation_trigger_quality_good_streak": int(
            runtime._adaptive_imag_compensation_trigger_quality_good_streak
        ),
        "adaptive_imag_late_trigger_rescue_until_step": int(
            runtime._adaptive_imag_late_trigger_rescue_until_step
        ),
        "adaptive_imag_entry_probe_hold_until_step": int(
            runtime._adaptive_imag_entry_probe_hold_until_step
        ),
        "adaptive_imag_post_entry_hold_until_step": int(
            runtime._adaptive_imag_post_entry_hold_until_step
        ),
        "adaptive_imag_post_entry_commit_highwater_hold_until_step": int(
            runtime._adaptive_imag_post_entry_commit_highwater_hold_until_step
        ),
        "adaptive_imag_post_entry_commit_candidate_since_step": int(
            runtime._adaptive_imag_post_entry_commit_candidate_since_step
        ),
        "adaptive_imag_handoff_hold_until_step": int(
            runtime._adaptive_imag_handoff_hold_until_step
        ),
        "adaptive_imag_post_entry_source": str(runtime._adaptive_imag_post_entry_source),
        "adaptive_imag_last_post_entry_source": str(runtime._adaptive_imag_last_post_entry_source),
        "adaptive_imag_external_post_entry_pending_commit": bool(
            runtime._adaptive_imag_external_post_entry_pending_commit
        ),
        "adaptive_imag_external_post_entry_soft_fallback_latched": bool(
            runtime._adaptive_imag_external_post_entry_soft_fallback_latched
        ),
        "adaptive_imag_standard_soft_fallback_trigger_release_step": int(
            runtime._adaptive_imag_standard_soft_fallback_trigger_release_step
        ),
        "adaptive_imag_post_entry_pending_recovery_latched": bool(
            runtime._adaptive_imag_post_entry_pending_recovery_latched
        ),
        "adaptive_imag_post_entry_commit_ever_armed": bool(
            runtime._adaptive_imag_post_entry_commit_ever_armed
        ),
        "adaptive_imag_compensation_persistence_release_recent_until_step": int(
            runtime._adaptive_imag_compensation_persistence_release_recent_until_step
        ),
        "adaptive_imag_compensation_persistence_release_tail_bypass_sustain_until_step": int(
            runtime._adaptive_imag_compensation_persistence_release_tail_bypass_sustain_until_step
        ),
        "adaptive_imag_compensation_trigger_persistence_handoff_blocked_until_step": int(
            runtime._adaptive_imag_compensation_trigger_persistence_handoff_blocked_until_step
        ),
        "adaptive_imag_compensation_trigger_persistence_handoff_confirmation_pending_after_late_trigger_base_cap": bool(
            runtime._adaptive_imag_compensation_trigger_persistence_handoff_confirmation_pending_after_late_trigger_base_cap
        ),
        "adaptive_imag_compensation_trigger_persistence_handoff_confirmation_streak_after_late_trigger_base_cap": int(
            runtime._adaptive_imag_compensation_trigger_persistence_handoff_confirmation_streak_after_late_trigger_base_cap
        ),
        "adaptive_imag_compensation_persistence_ever_armed": bool(
            runtime._adaptive_imag_compensation_persistence_ever_armed
        ),
        "adaptive_imag_compensation_persistence_ever_active": bool(
            runtime._adaptive_imag_compensation_persistence_ever_active
        ),
        "adaptive_imag_compensation_post_solved_negative_adv_latched_until_step": int(
            runtime._adaptive_imag_compensation_post_solved_negative_adv_latched_until_step
        ),
        "post_solved_anchor_state": {
            "actor_anchor_params": runtime._tensor_dict_to_cpu(
                runtime._adaptive_imag_compensation_post_solved_actor_anchor_params
            ),
            "actor_anchor_actor_state_dict": runtime._module_state_dict_to_cpu(
                runtime._adaptive_imag_compensation_post_solved_actor_anchor_actor,
                context="post-solved actor anchor policy snapshot",
            ),
            "actor_anchor_step": int(
                runtime._adaptive_imag_compensation_post_solved_actor_anchor_step
            ),
            "actor_anchor_eval": float(
                runtime._adaptive_imag_compensation_post_solved_actor_anchor_eval
            ),
            "critic_anchor_critic_state_dict": runtime._module_state_dict_to_cpu(
                runtime._adaptive_imag_compensation_post_solved_critic_anchor_critic,
                context="post-solved critic anchor snapshot",
            ),
            "critic_anchor_step": int(
                runtime._adaptive_imag_compensation_post_solved_critic_anchor_step
            ),
            "critic_anchor_eval": float(
                runtime._adaptive_imag_compensation_post_solved_critic_anchor_eval
            ),
        },
        "behavior_policy_eval_anchor_state": {
            "actor_state_dict": runtime._module_state_dict_to_cpu(
                runtime._behavior_policy_eval_anchor_actor,
                context="behavior-policy eval anchor snapshot",
            ),
            "step": int(runtime._behavior_policy_eval_anchor_step),
            "eval": float(runtime._behavior_policy_eval_anchor_eval),
        },
        "external_eval_last_mean": float(runtime._external_eval_last_mean),
        "external_eval_best_mean": float(runtime._external_eval_best_mean),
        "external_eval_last_step": int(runtime._external_eval_last_step),
        "bootstrap_external_eval_feedback_last_step": int(
            runtime._bootstrap_external_eval_feedback_last_step
        ),
        "bootstrap_external_eval_feedback_last_mean": float(
            runtime._bootstrap_external_eval_feedback_last_mean
        ),
        "bootstrap_external_eval_feedback_best_mean": float(
            runtime._bootstrap_external_eval_feedback_best_mean
        ),
        "bootstrap_external_eval_feedback_telemetry": dict(
            runtime._bootstrap_external_eval_feedback_telemetry
        ),
        "real_stability_last_step": int(runtime._real_stability_last_step),
        "real_stability_last_telemetry": dict(runtime._real_stability_last_telemetry),
        "real_stability_certified_anchor_step": int(
            runtime._real_stability_certified_anchor_step
        ),
        "real_stability_certified_anchor_eval": float(
            runtime._real_stability_certified_anchor_eval
        ),
        "real_stability_certified_telemetry": dict(
            runtime._real_stability_certified_telemetry
        ),
        "real_stability_certified_registry_entries": [
            {
                "step": int(step),
                "eval": float(eval_mean),
                "telemetry": dict(runtime._sanitize_real_stability_telemetry(telemetry)),
                "actor_state_dict": runtime._module_state_dict_to_cpu(
                    actor,
                    context="real-stability certified registry actor snapshot",
                ),
            }
            for actor, step, eval_mean, telemetry in zip(
                runtime._real_stability_certified_anchor_registry_actors,
                runtime._real_stability_certified_anchor_registry_steps,
                runtime._real_stability_certified_anchor_registry_evals,
                runtime._real_stability_certified_anchor_registry_telemetries,
            )
        ],
        "adaptive_imag_post_entry_eval_confirmation_streak": int(
            runtime._adaptive_imag_post_entry_eval_confirmation_streak
        ),
        "adaptive_imag_compensation_persistence_eval_confirmation_streak": int(
            runtime._adaptive_imag_compensation_persistence_eval_confirmation_streak
        ),
        "adaptive_imag_compensation_persistence_drop_confirmation_streak": int(
            runtime._adaptive_imag_compensation_persistence_drop_confirmation_streak
        ),
        "adaptive_imag_compensation_post_solved_eval_confirmation_streak": int(
            runtime._adaptive_imag_compensation_post_solved_eval_confirmation_streak
        ),
        "adaptive_imag_compensation_persistence_confirmed_best_mean": float(
            runtime._adaptive_imag_compensation_persistence_confirmed_best_mean
        ),
        "adaptive_imag_compensation_post_solved_confirmed_best_mean": float(
            runtime._adaptive_imag_compensation_post_solved_confirmed_best_mean
        ),
        "adaptive_imag_compensation_persistence_adv_ema": float(
            runtime._adaptive_imag_compensation_persistence_adv_ema
        ),
        "adaptive_imag_compensation_persistence_continue_ema": float(
            runtime._adaptive_imag_compensation_persistence_continue_ema
        ),
        "adaptive_imag_compensation_persistence_gap_ema": float(
            runtime._adaptive_imag_compensation_persistence_gap_ema
        ),
        "adaptive_imag_compensation_persistence_ema_initialized": bool(
            runtime._adaptive_imag_compensation_persistence_ema_initialized
        ),
        "adaptive_imag_entry_probe_prev_continue": float(
            runtime._adaptive_imag_entry_probe_prev_continue
        ),
        "adaptive_imag_entry_probe_prev_gap": float(
            runtime._adaptive_imag_entry_probe_prev_gap
        ),
        "adaptive_imag_entry_probe_prev_return": float(
            runtime._adaptive_imag_entry_probe_prev_return
        ),
        "adaptive_imag_entry_probe_prev_initialized": bool(
            runtime._adaptive_imag_entry_probe_prev_initialized
        ),
        "adaptive_imag_entry_probe_continue_history": list(
            runtime._adaptive_imag_entry_probe_continue_history
        ),
        "adaptive_imag_entry_probe_gap_history": list(
            runtime._adaptive_imag_entry_probe_gap_history
        ),
        "adaptive_imag_entry_probe_return_history": list(
            runtime._adaptive_imag_entry_probe_return_history
        ),
    }


def restore_adaptive_compensation_state(
    runtime: Any,
    state: Optional[Dict[str, Any]],
    *,
    fallback_best_step: int = -1,
    fallback_best_eval: float = -float("inf"),
    checkpoint_path: Optional[str] = None,
    strict: bool = True,
) -> None:
    if not state:
        runtime._adaptive_compensation_restore_report = {
            "status": "empty_state",
            "issues": [],
        }
        return
    report = _make_compensation_restore_report()
    duplicate_minimal_keys = (
        "adaptive_imag_continue_cap_state",
        "adaptive_imag_compensation_trigger_actor_scale_state",
        "adaptive_imag_compensation_phase",
        "adaptive_imag_critic_bootstrap_contact_surface_state",
        "adaptive_imag_critic_bootstrap_bonus_hold_state",
        "adaptive_imag_critic_bootstrap_post_transition_certified_floor_state",
        "adaptive_imag_compensation_persistence_hold_until_step",
        "adaptive_imag_compensation_persistence_armed_step",
        "adaptive_imag_compensation_persistence_entry_protect_until_step",
        "adaptive_imag_compensation_persistence_release_hold_until_step",
        "adaptive_imag_compensation_post_solved_hold_until_step",
    )
    minimal_compensation_payload = state.get("minimal_compensation_state")
    if minimal_compensation_payload is None:
        legacy_duplicate_keys = [key for key in duplicate_minimal_keys if key in state]
        if legacy_duplicate_keys:
            raise ValueError(
                "Adaptive compensation state is missing minimal_compensation_state "
                f"and still contains removed legacy duplicate keys: {legacy_duplicate_keys}"
            )
    restore_minimal_compensation_state(runtime, minimal_compensation_payload)
    runtime._adaptive_imag_compensation_trigger_pressure_streak = int(
        state.get("adaptive_imag_compensation_trigger_pressure_streak", 0)
    )
    runtime._adaptive_imag_compensation_trigger_quality_good_streak = int(
        state.get("adaptive_imag_compensation_trigger_quality_good_streak", 0)
    )
    runtime._adaptive_imag_late_trigger_rescue_until_step = int(
        state.get("adaptive_imag_late_trigger_rescue_until_step", -1)
    )
    runtime._adaptive_imag_entry_probe_hold_until_step = int(
        state.get("adaptive_imag_entry_probe_hold_until_step", -1)
    )
    runtime._adaptive_imag_post_entry_hold_until_step = int(
        state.get("adaptive_imag_post_entry_hold_until_step", -1)
    )
    runtime._adaptive_imag_post_entry_commit_highwater_hold_until_step = int(
        state.get("adaptive_imag_post_entry_commit_highwater_hold_until_step", -1)
    )
    runtime._adaptive_imag_post_entry_commit_candidate_since_step = int(
        state.get("adaptive_imag_post_entry_commit_candidate_since_step", -1)
    )
    runtime._adaptive_imag_handoff_hold_until_step = int(
        state.get("adaptive_imag_handoff_hold_until_step", -1)
    )
    runtime._adaptive_imag_post_entry_source = str(
        state.get("adaptive_imag_post_entry_source", "idle")
    )
    runtime._adaptive_imag_last_post_entry_source = str(
        state.get("adaptive_imag_last_post_entry_source", "idle")
    )
    runtime._adaptive_imag_external_post_entry_pending_commit = bool(
        state.get("adaptive_imag_external_post_entry_pending_commit", False)
    )
    runtime._adaptive_imag_external_post_entry_soft_fallback_latched = bool(
        state.get("adaptive_imag_external_post_entry_soft_fallback_latched", False)
    )
    runtime._adaptive_imag_standard_soft_fallback_trigger_release_step = int(
        state.get("adaptive_imag_standard_soft_fallback_trigger_release_step", -1)
    )
    runtime._adaptive_imag_post_entry_pending_recovery_latched = bool(
        state.get("adaptive_imag_post_entry_pending_recovery_latched", False)
    )
    runtime._adaptive_imag_post_entry_commit_ever_armed = bool(
        state.get("adaptive_imag_post_entry_commit_ever_armed", False)
    )
    runtime._adaptive_imag_compensation_persistence_release_recent_until_step = int(
        state.get("adaptive_imag_compensation_persistence_release_recent_until_step", -1)
    )
    runtime._adaptive_imag_compensation_persistence_release_tail_bypass_sustain_until_step = int(
        state.get(
            "adaptive_imag_compensation_persistence_release_tail_bypass_sustain_until_step",
            -1,
        )
    )
    runtime._adaptive_imag_compensation_trigger_persistence_handoff_blocked_until_step = int(
        state.get("adaptive_imag_compensation_trigger_persistence_handoff_blocked_until_step", -1)
    )
    runtime._adaptive_imag_compensation_trigger_persistence_handoff_confirmation_pending_after_late_trigger_base_cap = bool(
        state.get(
            "adaptive_imag_compensation_trigger_persistence_handoff_confirmation_pending_after_late_trigger_base_cap",
            False,
        )
    )
    runtime._adaptive_imag_compensation_trigger_persistence_handoff_confirmation_streak_after_late_trigger_base_cap = int(
        state.get(
            "adaptive_imag_compensation_trigger_persistence_handoff_confirmation_streak_after_late_trigger_base_cap",
            0,
        )
    )
    runtime._adaptive_imag_compensation_persistence_ever_armed = bool(
        state.get("adaptive_imag_compensation_persistence_ever_armed", False)
    )
    runtime._adaptive_imag_compensation_persistence_ever_active = bool(
        state.get("adaptive_imag_compensation_persistence_ever_active", False)
    )
    runtime._adaptive_imag_compensation_post_solved_negative_adv_latched_until_step = int(
        state.get("adaptive_imag_compensation_post_solved_negative_adv_latched_until_step", -1)
    )
    runtime._external_eval_last_mean = float(
        state.get("external_eval_last_mean", -float("inf"))
    )
    runtime._external_eval_best_mean = float(
        state.get("external_eval_best_mean", -float("inf"))
    )
    runtime._external_eval_last_step = int(state.get("external_eval_last_step", -1))
    runtime._bootstrap_external_eval_feedback_last_step = int(
        state.get(
            "bootstrap_external_eval_feedback_last_step",
            state.get("real_stability_last_step", -1),
        )
    )
    runtime._bootstrap_external_eval_feedback_last_mean = float(
        state.get(
            "bootstrap_external_eval_feedback_last_mean",
            state.get("external_eval_last_mean", -float("inf")),
        )
    )
    runtime._bootstrap_external_eval_feedback_best_mean = float(
        state.get(
            "bootstrap_external_eval_feedback_best_mean",
            state.get("external_eval_best_mean", -float("inf")),
        )
    )
    runtime._bootstrap_external_eval_feedback_telemetry = runtime._sanitize_real_stability_telemetry(
        state.get(
            "bootstrap_external_eval_feedback_telemetry",
            state.get("real_stability_last_telemetry", {}),
        )
    )
    post_solved_anchor_report = runtime._restore_post_solved_anchor_state(
        state.get("post_solved_anchor_state"),
        fallback_best_step=fallback_best_step,
        fallback_best_eval=fallback_best_eval,
    )
    _record_restore_degradation(
        report,
        section="post_solved_anchor",
        section_report=post_solved_anchor_report,
        checkpoint_path=checkpoint_path,
    )
    behavior_policy_anchor_report = runtime._restore_behavior_policy_eval_anchor_state(
        state.get("behavior_policy_eval_anchor_state"),
        fallback_best_step=fallback_best_step,
        fallback_best_eval=fallback_best_eval,
    )
    _record_restore_degradation(
        report,
        section="behavior_policy_anchor",
        section_report=behavior_policy_anchor_report,
        checkpoint_path=checkpoint_path,
    )
    runtime._real_stability_last_step = int(state.get("real_stability_last_step", -1))
    runtime._real_stability_last_telemetry = runtime._sanitize_real_stability_telemetry(
        state.get("real_stability_last_telemetry", {})
    )
    runtime._bootstrap_runtime_task_cert_telemetry = {}
    runtime._real_stability_certified_anchor_step = int(
        state.get("real_stability_certified_anchor_step", -1)
    )
    runtime._real_stability_certified_anchor_eval = float(
        state.get("real_stability_certified_anchor_eval", -float("inf"))
    )
    runtime._real_stability_certified_telemetry = runtime._sanitize_real_stability_telemetry(
        state.get("real_stability_certified_telemetry", {})
    )
    runtime._real_stability_certified_anchor_actor = None
    runtime._real_stability_certified_anchor_registry_actors = []
    runtime._real_stability_certified_anchor_registry_steps = []
    runtime._real_stability_certified_anchor_registry_evals = []
    runtime._real_stability_certified_anchor_registry_telemetries = []
    registry_report = {
        "status": "not_attempted",
        "restored_entries": 0,
        "skipped_entries": 0,
        "issues": [],
    }
    raw_registry_entries: list[Any] = []
    if "real_stability_certified_registry_entries" not in state:
        registry_report["status"] = "missing"
        registry_report["issues"].append(
            "real_stability_registry entries missing from checkpoint payload"
        )
    else:
        raw_registry_entries = state.get("real_stability_certified_registry_entries", [])
        if not isinstance(raw_registry_entries, (list, tuple)):
            registry_report["status"] = "failed"
            registry_report["issues"].append(
                "real_stability_registry entries must be a list"
            )
            raw_registry_entries = []
    for entry in list(raw_registry_entries):
        if not isinstance(entry, dict):
            registry_report["skipped_entries"] += 1
            registry_report["issues"].append(
                "real_stability_registry entry is not a mapping"
            )
            continue
        anchor_actor = runtime._load_frozen_actor_from_state_dict(entry.get("actor_state_dict"))
        if anchor_actor is None:
            registry_report["skipped_entries"] += 1
            registry_report["issues"].append(
                "real_stability_registry entry skipped because actor anchor could not be restored"
            )
            continue
        telemetry = runtime._sanitize_real_stability_telemetry(entry.get("telemetry", {}))
        runtime._real_stability_certified_anchor_registry_actors.append(anchor_actor)
        runtime._real_stability_certified_anchor_registry_steps.append(
            int(entry.get("step", -1))
        )
        runtime._real_stability_certified_anchor_registry_evals.append(
            float(entry.get("eval", -float("inf")))
        )
        runtime._real_stability_certified_anchor_registry_telemetries.append(dict(telemetry))
        registry_report["restored_entries"] += 1
    if runtime._real_stability_certified_anchor_registry_actors:
        runtime._real_stability_certified_anchor_actor = (
            runtime._real_stability_certified_anchor_registry_actors[-1]
        )
        runtime._real_stability_certified_anchor_step = int(
            runtime._real_stability_certified_anchor_registry_steps[-1]
        )
        runtime._real_stability_certified_anchor_eval = float(
            runtime._real_stability_certified_anchor_registry_evals[-1]
        )
        runtime._real_stability_certified_telemetry = dict(
            runtime._real_stability_certified_anchor_registry_telemetries[-1]
        )
    if registry_report["restored_entries"] > 0 and registry_report["skipped_entries"] > 0:
        registry_report["status"] = "partial_restore"
    elif registry_report["restored_entries"] > 0:
        registry_report["status"] = "restored"
    elif registry_report["skipped_entries"] > 0:
        registry_report["status"] = "failed"
    _record_restore_degradation(
        report,
        section="real_stability_registry",
        section_report=registry_report,
        checkpoint_path=checkpoint_path,
    )
    runtime._adaptive_imag_post_entry_eval_confirmation_streak = int(
        state.get("adaptive_imag_post_entry_eval_confirmation_streak", 0)
    )
    runtime._adaptive_imag_compensation_persistence_eval_confirmation_streak = int(
        state.get("adaptive_imag_compensation_persistence_eval_confirmation_streak", 0)
    )
    runtime._adaptive_imag_compensation_persistence_drop_confirmation_streak = int(
        state.get("adaptive_imag_compensation_persistence_drop_confirmation_streak", 0)
    )
    runtime._adaptive_imag_compensation_post_solved_eval_confirmation_streak = int(
        state.get("adaptive_imag_compensation_post_solved_eval_confirmation_streak", 0)
    )
    runtime._adaptive_imag_compensation_persistence_confirmed_best_mean = float(
        state.get("adaptive_imag_compensation_persistence_confirmed_best_mean", -float("inf"))
    )
    runtime._adaptive_imag_compensation_post_solved_confirmed_best_mean = float(
        state.get("adaptive_imag_compensation_post_solved_confirmed_best_mean", -float("inf"))
    )
    runtime._adaptive_imag_compensation_persistence_adv_ema = float(
        state.get("adaptive_imag_compensation_persistence_adv_ema", 0.0)
    )
    runtime._adaptive_imag_compensation_persistence_continue_ema = float(
        state.get("adaptive_imag_compensation_persistence_continue_ema", 0.0)
    )
    runtime._adaptive_imag_compensation_persistence_gap_ema = float(
        state.get("adaptive_imag_compensation_persistence_gap_ema", 0.0)
    )
    runtime._adaptive_imag_compensation_persistence_ema_initialized = bool(
        state.get("adaptive_imag_compensation_persistence_ema_initialized", False)
    )
    runtime._adaptive_imag_entry_probe_prev_continue = float(
        state.get("adaptive_imag_entry_probe_prev_continue", 0.0)
    )
    runtime._adaptive_imag_entry_probe_prev_gap = float(
        state.get("adaptive_imag_entry_probe_prev_gap", 0.0)
    )
    runtime._adaptive_imag_entry_probe_prev_return = float(
        state.get("adaptive_imag_entry_probe_prev_return", 0.0)
    )
    runtime._adaptive_imag_entry_probe_prev_initialized = bool(
        state.get("adaptive_imag_entry_probe_prev_initialized", False)
    )
    runtime._adaptive_imag_entry_probe_continue_history = deque(
        float(v) for v in state.get("adaptive_imag_entry_probe_continue_history", [])
    )
    runtime._adaptive_imag_entry_probe_gap_history = deque(
        float(v) for v in state.get("adaptive_imag_entry_probe_gap_history", [])
    )
    runtime._adaptive_imag_entry_probe_return_history = deque(
        float(v) for v in state.get("adaptive_imag_entry_probe_return_history", [])
    )
    runtime._adaptive_compensation_restore_report = report
    runtime._backfill_post_solved_resume_state(
        fallback_best_step=fallback_best_step,
        fallback_best_eval=fallback_best_eval,
    )
    if strict and str(report.get("status", "restored")) == "degraded":
        issue_text = " | ".join(str(item) for item in report.get("issues", [])) or "unknown issues"
        raise RuntimeError(
            "Adaptive compensation restore degraded for "
            f"{checkpoint_path or '<checkpoint>'}: {issue_text}"
        )


def resolve_post_trigger_quality_state(
    *,
    config: Any,
    quality_good_streak: int,
    return_baseline: Tensor,
    returns: Tensor,
) -> dict[str, float]:
    quality_gap_abs = float((return_baseline - returns.detach()).abs().mean())
    quality_actor_mean = float(returns.detach().mean())
    quality_gate_enabled = bool(
        getattr(config, "adaptive_imag_compensation_trigger_quality_gate", False)
    )
    quality_gap_threshold = float(
        getattr(config, "adaptive_imag_compensation_trigger_quality_gap_threshold", 2.0)
    )
    quality_actor_threshold = float(
        getattr(config, "adaptive_imag_compensation_trigger_quality_actor_threshold", 10.0)
    )
    quality_required_streak = max(
        1,
        int(getattr(config, "adaptive_imag_compensation_trigger_quality_streak", 2)),
    )
    quality_good = bool(
        quality_gap_abs <= quality_gap_threshold
        and quality_actor_mean <= quality_actor_threshold
    )
    quality_gap_over = max(0.0, quality_gap_abs - quality_gap_threshold)
    quality_actor_over = max(0.0, quality_actor_mean - quality_actor_threshold)
    quality_scale_gain = float(
        getattr(config, "adaptive_imag_compensation_trigger_quality_scale_gain", 0.0)
    )
    quality_scale_floor = float(
        getattr(config, "adaptive_imag_compensation_trigger_quality_scale_floor", 0.5)
    )
    quality_scale_pressure = quality_gap_over + 0.25 * quality_actor_over
    quality_piecewise = bool(
        getattr(config, "adaptive_imag_compensation_trigger_quality_piecewise", False)
    )
    quality_mild_threshold = float(
        getattr(config, "adaptive_imag_compensation_trigger_quality_mild_threshold", 2.0)
    )
    quality_severe_threshold = float(
        getattr(config, "adaptive_imag_compensation_trigger_quality_severe_threshold", 8.0)
    )
    quality_mild_floor = float(
        getattr(config, "adaptive_imag_compensation_trigger_quality_mild_floor", 0.85)
    )
    quality_severe_floor = float(
        getattr(config, "adaptive_imag_compensation_trigger_quality_severe_floor", 0.55)
    )
    quality_scale_multiplier = 1.0
    if quality_piecewise:
        quality_mild_threshold = max(0.0, quality_mild_threshold)
        quality_severe_threshold = max(quality_mild_threshold + 1e-6, quality_severe_threshold)
        quality_mild_floor = min(max(quality_mild_floor, 0.0), 1.0)
        quality_severe_floor = min(max(quality_severe_floor, 0.0), quality_mild_floor)
        if quality_scale_pressure <= quality_mild_threshold:
            quality_scale_multiplier = 1.0
        elif quality_scale_pressure >= quality_severe_threshold:
            quality_scale_multiplier = quality_severe_floor
        else:
            ratio = (quality_scale_pressure - quality_mild_threshold) / max(
                quality_severe_threshold - quality_mild_threshold,
                1e-6,
            )
            quality_scale_multiplier = quality_mild_floor + (
                quality_severe_floor - quality_mild_floor
            ) * ratio
    elif quality_scale_gain > 0.0:
        quality_scale_multiplier = max(
            quality_scale_floor,
            math.exp(-quality_scale_gain * quality_scale_pressure),
        )
    quality_release_ready = (not quality_gate_enabled) or (
        quality_good and int(quality_good_streak) + 1 >= quality_required_streak
    )
    return {
        "gap_abs": float(quality_gap_abs),
        "actor_mean": float(quality_actor_mean),
        "gate_enabled": float(quality_gate_enabled),
        "gap_threshold": float(quality_gap_threshold),
        "actor_threshold": float(quality_actor_threshold),
        "required_streak": float(quality_required_streak),
        "good": float(quality_good),
        "gap_over": float(quality_gap_over),
        "actor_over": float(quality_actor_over),
        "scale_pressure": float(quality_scale_pressure),
        "scale_multiplier": float(quality_scale_multiplier),
        "release_ready": float(quality_release_ready),
    }


def resolve_standard_soft_fallback_trigger_release_progress(
    *,
    current_step: int | None,
    compensation_phase: str,
    post_entry_source: str | None,
    quality_release_ready: bool | None,
    runtime: Any | None = None,
    config: Any | None = None,
) -> float:
    resolved_phase = canonicalize_compensation_phase(compensation_phase)
    source_name = str(post_entry_source or "idle")
    if config is None and runtime is not None:
        config = getattr(runtime, "config", None)
    if resolved_phase not in (CompensationPhase.TRIGGER, CompensationPhase.RELEASE):
        if runtime is not None:
            runtime._adaptive_imag_standard_soft_fallback_trigger_release_step = -1
        return 0.0
    if source_name == "idle":
        if runtime is not None:
            runtime._adaptive_imag_standard_soft_fallback_trigger_release_step = -1
        return 0.0

    step_now = max(0, int(current_step or 0))
    if runtime is not None:
        release_start = int(
            getattr(runtime, "_adaptive_imag_standard_soft_fallback_trigger_release_step", -1)
        )
        if release_start < 0:
            release_start = step_now
            runtime._adaptive_imag_standard_soft_fallback_trigger_release_step = release_start
    else:
        release_start = step_now

    release_steps = max(
        1,
        int(
            getattr(
                config,
                "adaptive_imag_standard_soft_fallback_trigger_release_steps",
                getattr(
                    config,
                    "adaptive_imag_post_entry_soft_trigger_release_steps",
                    max(
                        1,
                        int(getattr(config, "adaptive_imag_post_entry_hold_steps", 1))
                        if config is not None
                        else 1,
                    ),
                ),
            )
        ),
    )
    progress = min(1.0, max(0.0, float(step_now - release_start) / float(release_steps)))
    if quality_release_ready is False:
        progress = min(
            progress,
            min(
                1.0,
                max(
                    0.0,
                    float(
                        getattr(
                            config,
                            "adaptive_imag_standard_soft_fallback_trigger_release_quality_clamp",
                            getattr(
                                config,
                                "adaptive_imag_post_entry_soft_trigger_bad_quality_max_progress",
                                0.5,
                            ),
                        )
                    ),
                ),
            ),
        )
    return float(progress)


def is_trigger_persistence_handoff_landing_guard_active(
    *,
    config: Any,
    gap_abs: float,
    continue_mean: float,
) -> bool:
    if not bool(
        getattr(
            config,
            "adaptive_imag_compensation_trigger_persistence_handoff_landing_guard_enabled",
            False,
        )
    ):
        return False
    tail_gap_threshold = max(
        0.0,
        float(getattr(config, "adaptive_imag_compensation_persistence_tail_gap_threshold", 0.0)),
    )
    tail_continue_threshold = float(
        getattr(config, "adaptive_imag_compensation_persistence_tail_continue_threshold", 0.0)
    )
    handoff_continue_threshold = float(
        getattr(
            config,
            "adaptive_imag_compensation_trigger_persistence_handoff_continue_threshold",
            0.0,
        )
    )
    if tail_gap_threshold <= 0.0 or tail_continue_threshold <= 0.0:
        return False
    landing_continue_threshold = tail_continue_threshold
    if handoff_continue_threshold > 0.0:
        landing_continue_threshold = math.sqrt(
            max(1e-6, tail_continue_threshold)
            * max(1e-6, handoff_continue_threshold)
        )
    return bool(
        float(gap_abs) >= tail_gap_threshold
        and float(continue_mean) <= landing_continue_threshold
    )


def resolve_post_entry_soft_negative_adv_profile(
    *,
    config: Any,
    current_step: int,
    external_eval_best_mean: float,
    base_actor_scale: float,
    base_critic_boost: float,
) -> dict[str, float]:
    resolved_actor_scale = float(base_actor_scale)
    resolved_critic_boost = float(base_critic_boost)
    soft_override_active = False
    midwater_active = 0.0
    midwater_ramp = 0.0
    highwater_active = 0.0

    soft_guard_scale = float(
        getattr(config, "adaptive_imag_post_entry_soft_negative_adv_actor_scale", 0.0)
    )
    soft_guard_boost = float(
        getattr(config, "adaptive_imag_post_entry_soft_negative_adv_critic_boost", 0.0)
    )
    if soft_guard_scale > 0.0:
        resolved_actor_scale = soft_guard_scale
        soft_override_active = True
    if soft_guard_boost > 0.0:
        resolved_critic_boost = soft_guard_boost
        soft_override_active = True

    midwater_guard_eval_threshold = float(
        getattr(config, "adaptive_imag_post_entry_soft_negative_adv_midwater_eval_threshold", 0.0)
    )
    midwater_guard_min_step = int(
        getattr(config, "adaptive_imag_post_entry_soft_negative_adv_midwater_min_step", 0)
    )
    midwater_guard_ramp_steps = max(
        0,
        int(
            getattr(
                config,
                "adaptive_imag_post_entry_soft_negative_adv_midwater_ramp_steps",
                0,
            )
        ),
    )
    midwater_active = float(
        midwater_guard_eval_threshold > 0.0
        and external_eval_best_mean >= midwater_guard_eval_threshold
        and current_step >= midwater_guard_min_step
    )
    if midwater_active > 0.0:
        soft_override_active = True
        if midwater_guard_ramp_steps <= 0:
            midwater_ramp = 1.0
        else:
            midwater_ramp = min(
                1.0,
                float(current_step - midwater_guard_min_step) / float(midwater_guard_ramp_steps),
            )
        midwater_actor_scale = float(
            getattr(config, "adaptive_imag_post_entry_soft_negative_adv_midwater_actor_scale", 0.0)
        )
        midwater_critic_boost = float(
            getattr(
                config,
                "adaptive_imag_post_entry_soft_negative_adv_midwater_critic_boost",
                0.0,
            )
        )
        if midwater_actor_scale > 0.0:
            resolved_actor_scale = resolved_actor_scale + (
                midwater_actor_scale - resolved_actor_scale
            ) * midwater_ramp
        if midwater_critic_boost > 0.0:
            resolved_critic_boost = resolved_critic_boost + (
                midwater_critic_boost - resolved_critic_boost
            ) * midwater_ramp

    highwater_guard_eval_threshold = float(
        getattr(
            config,
            "adaptive_imag_post_entry_soft_negative_adv_highwater_eval_threshold",
            0.0,
        )
    )
    highwater_active = float(
        highwater_guard_eval_threshold > 0.0
        and external_eval_best_mean >= highwater_guard_eval_threshold
    )
    if highwater_active > 0.0:
        soft_override_active = True
        highwater_actor_scale = float(
            getattr(
                config,
                "adaptive_imag_post_entry_soft_negative_adv_highwater_actor_scale",
                0.0,
            )
        )
        highwater_critic_boost = float(
            getattr(
                config,
                "adaptive_imag_post_entry_soft_negative_adv_highwater_critic_boost",
                0.0,
            )
        )
        if highwater_actor_scale > 0.0:
            resolved_actor_scale = highwater_actor_scale
        if highwater_critic_boost > 0.0:
            resolved_critic_boost = highwater_critic_boost

    return {
        "actor_scale": float(resolved_actor_scale),
        "critic_boost": float(resolved_critic_boost),
        "midwater_active": float(midwater_active),
        "midwater_ramp": float(midwater_ramp),
        "highwater_active": float(highwater_active),
        "soft_override_active": float(soft_override_active),
    }


def resolve_persistence_escape_soft_floors(
    *,
    config: Any,
    persistence_armed_step: int,
    current_step: int,
    external_eval_best_mean: float,
) -> dict[str, float]:
    soft_guard_profile = resolve_post_entry_soft_negative_adv_profile(
        config=config,
        current_step=int(current_step),
        external_eval_best_mean=float(external_eval_best_mean),
        base_actor_scale=1.0,
        base_critic_boost=0.0,
    )
    band_specific_active = bool(
        float(soft_guard_profile.get("midwater_active", 0.0)) > 0.0
        or float(soft_guard_profile.get("highwater_active", 0.0)) > 0.0
    )
    if not band_specific_active:
        return {
            "cap_floor": 0.0,
            "actor_scale_floor": 0.0,
            "active": 0.0,
            "band_specific_active": 0.0,
            "release_progress": 0.0,
        }

    soft_cap_floor = max(
        0.0,
        float(getattr(config, "adaptive_imag_post_entry_soft_cap_floor", 0.0)),
    )
    soft_actor_scale_floor = max(
        0.0,
        float(getattr(config, "adaptive_imag_post_entry_soft_actor_scale_floor", 0.0)),
    )
    release_progress = 0.0
    release_steps = max(
        0,
        int(
            getattr(
                config,
                "adaptive_imag_compensation_persistence_escape_band_release_steps",
                getattr(
                    config,
                    "adaptive_imag_persistence_escape_band_release_steps",
                    0,
                ),
            )
        ),
    )
    if release_steps > 0 and int(persistence_armed_step) >= 0:
        release_progress = min(
            1.0,
            max(
                0.0,
                float(int(current_step) - int(persistence_armed_step)) / float(release_steps),
            ),
        )
        target_cap_floor = max(
            0.0,
            float(getattr(config, "adaptive_imag_compensation_persistence_cap", 0.0)),
        )
        target_actor_scale_floor = float(
            getattr(config, "adaptive_imag_compensation_persistence_actor_scale", 0.0)
        )
        if target_actor_scale_floor <= 0.0:
            target_actor_scale_floor = 1.0
        soft_cap_floor = soft_cap_floor + (
            target_cap_floor - soft_cap_floor
        ) * release_progress
        soft_actor_scale_floor = soft_actor_scale_floor + (
            target_actor_scale_floor - soft_actor_scale_floor
        ) * release_progress
    return {
        "cap_floor": float(soft_cap_floor),
        "actor_scale_floor": float(soft_actor_scale_floor),
        "active": 1.0,
        "band_specific_active": 1.0,
        "release_progress": float(release_progress),
    }


def handle_external_eval_feedback(
    runtime: Any,
    mean_return: float,
    step: Optional[int] = None,
    telemetry: Optional[Dict[str, Any]] = None,
) -> None:
    mean_return = float(mean_return)
    resolved_step = max(0, int(runtime.global_step if step is None else step))
    runtime._external_eval_last_mean = mean_return
    runtime._external_eval_last_step = resolved_step
    prev_best = float(runtime._external_eval_best_mean)
    reward_health = _compute_reward_health(mean_return, max(mean_return, prev_best))
    reward_degradation = 1.0 - reward_health
    if telemetry is None:
        telemetry = {}
    prev_real_stability = runtime._sanitize_real_stability_telemetry(
        getattr(runtime, "_real_stability_last_telemetry", {})
    )
    if isinstance(telemetry, dict):
        telemetry["real_reward_health"] = float(reward_health)
        telemetry["real_reward_degradation"] = float(reward_degradation)
        telemetry["real_task_cert_gate"] = float(reward_health)
        if "real_task_cert_state" in prev_real_stability:
            telemetry.setdefault(
                "real_task_cert_state",
                float(prev_real_stability["real_task_cert_state"]),
            )
        if "real_task_cert_alarm" in prev_real_stability:
            telemetry.setdefault(
                "real_task_cert_alarm",
                float(prev_real_stability["real_task_cert_alarm"]),
            )
        if "real_task_cert_recovery" in prev_real_stability:
            telemetry.setdefault(
                "real_task_cert_recovery",
                float(prev_real_stability["real_task_cert_recovery"]),
            )
    sanitized_telemetry = runtime._sanitize_real_stability_telemetry(telemetry)
    runtime._bootstrap_external_eval_feedback_last_step = int(resolved_step)
    runtime._bootstrap_external_eval_feedback_last_mean = float(mean_return)
    runtime._bootstrap_external_eval_feedback_best_mean = float(
        max(
            float(getattr(runtime, "_bootstrap_external_eval_feedback_best_mean", -float("inf"))),
            mean_return,
        )
    )
    runtime._bootstrap_external_eval_feedback_telemetry = dict(sanitized_telemetry)
    runtime._store_real_stability_telemetry(step=resolved_step, telemetry=sanitized_telemetry)
    runtime._capture_behavior_policy_eval_anchor(step=resolved_step, eval_mean=mean_return)
    improved_best = mean_return > runtime._external_eval_best_mean
    if improved_best:
        runtime._external_eval_best_mean = mean_return
        runtime._capture_post_solved_actor_anchor(step=resolved_step, eval_mean=mean_return)
    certified_threshold = runtime._resolve_real_stability_certified_eval_threshold()
    certified_capture_active = improved_best
    if certified_threshold > 0.0 and mean_return >= certified_threshold:
        runtime._capture_real_stability_certified_anchor(step=resolved_step, eval_mean=mean_return)
        certified_capture_active = True
    if certified_capture_active and not (
        certified_threshold > 0.0 and mean_return >= certified_threshold
    ):
        runtime._capture_real_stability_certified_anchor(step=resolved_step, eval_mean=mean_return)

    entry_threshold = float(getattr(runtime.config, "adaptive_imag_post_entry_eval_threshold", 0.0))
    persistence_threshold = float(
        getattr(runtime.config, "adaptive_imag_compensation_persistence_eval_threshold", 0.0)
    )
    post_solved_threshold = float(
        getattr(runtime.config, "adaptive_imag_compensation_post_solved_eval_threshold", 0.0)
    )
    _, _, entry_confirmed = runtime._update_adaptive_eval_confirmation(
        mean_return=mean_return,
        threshold=entry_threshold,
        streak_attr="_adaptive_imag_post_entry_eval_confirmation_streak",
        count_key="adaptive_imag_post_entry_eval_confirmation_count",
    )
    _, _, persistence_confirmed = runtime._update_adaptive_eval_confirmation(
        mean_return=mean_return,
        threshold=persistence_threshold,
        streak_attr="_adaptive_imag_compensation_persistence_eval_confirmation_streak",
        count_key="adaptive_imag_compensation_persistence_eval_confirmation_count",
        confirmed_best_attr="_adaptive_imag_compensation_persistence_confirmed_best_mean",
    )
    _, _, post_solved_confirmed = runtime._update_adaptive_eval_confirmation(
        mean_return=mean_return,
        threshold=post_solved_threshold,
        streak_attr="_adaptive_imag_compensation_post_solved_eval_confirmation_streak",
        count_key="adaptive_imag_compensation_post_solved_eval_confirmation_count",
        confirmed_best_attr="_adaptive_imag_compensation_post_solved_confirmed_best_mean",
    )
    persistence_confirmed_best_mean = float(
        getattr(runtime, "_adaptive_imag_compensation_persistence_confirmed_best_mean", -float("inf"))
    )
    post_solved_confirmed_best_mean = float(
        getattr(runtime, "_adaptive_imag_compensation_post_solved_confirmed_best_mean", -float("inf"))
    )

    entry_hold_steps = max(0, int(getattr(runtime.config, "adaptive_imag_post_entry_hold_steps", 0)))
    entry_rearm_delta = max(0.0, float(getattr(runtime.config, "adaptive_imag_post_entry_rearm_delta", 0.0)))
    explicit_highwater_commit_enabled = bool(
        float(getattr(runtime.config, "adaptive_imag_post_entry_commit_highwater_eval_threshold", 0.0)) > 0.0
    )
    entry_inactive = resolved_step > int(getattr(runtime, "_adaptive_imag_post_entry_hold_until_step", -1))
    improved_eval = (prev_best == -float("inf")) or (mean_return >= prev_best + entry_rearm_delta)
    external_pending_commit = bool(getattr(runtime, "_adaptive_imag_external_post_entry_pending_commit", False))
    soft_fallback_latched = bool(
        getattr(runtime, "_adaptive_imag_external_post_entry_soft_fallback_latched", False)
    )
    allow_rearm = (not external_pending_commit) or improved_eval
    if soft_fallback_latched and not explicit_highwater_commit_enabled:
        allow_rearm = False
    elif soft_fallback_latched and not improved_eval:
        allow_rearm = False
    if entry_confirmed and entry_threshold > 0.0 and entry_hold_steps > 0 and mean_return >= entry_threshold:
        if allow_rearm and (entry_inactive or improved_eval):
            runtime._adaptive_imag_post_entry_hold_until_step = max(
                runtime._adaptive_imag_post_entry_hold_until_step,
                resolved_step + entry_hold_steps,
            )
            runtime._adaptive_imag_post_entry_source = "external_eval"
            runtime._adaptive_imag_last_post_entry_source = "external_eval"
            runtime._adaptive_imag_external_post_entry_pending_commit = True
            if entry_inactive:
                runtime._adaptive_imag_external_post_entry_soft_fallback_latched = False
            runtime._adaptive_imag_post_entry_commit_candidate_since_step = -1

    preview_min_step = max(0, int(getattr(runtime.config, "adaptive_imag_post_entry_preview_min_step", 0)))
    preview_hold_steps = max(0, int(getattr(runtime.config, "adaptive_imag_post_entry_preview_hold_steps", 0)))
    preview_improve_margin = max(0.0, float(getattr(runtime.config, "adaptive_imag_post_entry_preview_improve_margin", 0.0)))
    preview_ready = bool(
        not entry_confirmed
        and entry_threshold > 0.0
        and preview_hold_steps > 0
        and resolved_step >= preview_min_step
        and prev_best != -float("inf")
        and mean_return >= entry_threshold
        and mean_return >= prev_best + preview_improve_margin
    )
    if preview_ready and allow_rearm and (entry_inactive or improved_eval):
        runtime._adaptive_imag_post_entry_hold_until_step = max(
            runtime._adaptive_imag_post_entry_hold_until_step,
            resolved_step + preview_hold_steps,
        )
        runtime._adaptive_imag_post_entry_source = "external_eval"
        runtime._adaptive_imag_last_post_entry_source = "external_eval"
        runtime._adaptive_imag_external_post_entry_pending_commit = True
        if entry_inactive:
            runtime._adaptive_imag_external_post_entry_soft_fallback_latched = False
        runtime._adaptive_imag_post_entry_commit_candidate_since_step = -1

    protect_steps = max(0, int(getattr(runtime.config, "adaptive_imag_compensation_persistence_entry_protect_steps", 0)))
    protect_eval_ratio = max(0.0, float(getattr(runtime.config, "adaptive_imag_compensation_persistence_entry_protect_eval_ratio", 0.0)))
    protect_eval_threshold = entry_threshold if protect_eval_ratio <= 0.0 else max(entry_threshold, runtime._external_eval_best_mean * protect_eval_ratio)
    if improved_best and protect_steps > 0 and mean_return >= protect_eval_threshold:
        runtime._adaptive_imag_compensation_persistence_entry_protect_until_step = max(
            runtime._adaptive_imag_compensation_persistence_entry_protect_until_step,
            resolved_step + protect_steps,
        )

    highwater_eval_threshold = max(persistence_threshold, post_solved_threshold)
    highwater_hold_steps = max(0, int(getattr(runtime.config, "adaptive_imag_post_entry_commit_highwater_hold_steps", 0)))
    highwater_confirmed_best_mean = max(
        persistence_confirmed_best_mean,
        post_solved_confirmed_best_mean,
    )
    if (
        improved_best
        and highwater_hold_steps > 0
        and highwater_eval_threshold > 0.0
        and mean_return >= highwater_eval_threshold
        and highwater_confirmed_best_mean >= highwater_eval_threshold
    ):
        runtime._adaptive_imag_post_entry_commit_highwater_hold_until_step = max(
            runtime._adaptive_imag_post_entry_commit_highwater_hold_until_step,
            resolved_step + highwater_hold_steps,
        )
    persistence_min_step = max(0, int(getattr(runtime.config, "adaptive_imag_compensation_persistence_min_step", 0)))
    persistence_drop = max(0.0, float(getattr(runtime.config, "adaptive_imag_compensation_persistence_eval_drop", 0.0)))
    persistence_hold_steps = max(0, int(getattr(runtime.config, "adaptive_imag_compensation_persistence_hold_steps", 0)))
    persistence_drop_confirmation_count = max(
        1,
        int(getattr(runtime.config, "adaptive_imag_compensation_persistence_drop_confirmation_count", 0)),
    )
    persistence_drop_floor = persistence_threshold
    if persistence_drop_confirmation_count > 1:
        release_eval_ratio = max(
            0.0,
            float(getattr(runtime.config, "adaptive_imag_compensation_persistence_release_eval_ratio", 0.0)),
        )
        if release_eval_ratio > 0.0 and post_solved_threshold > 0.0:
            persistence_drop_floor = max(
                persistence_drop_floor,
                post_solved_threshold * release_eval_ratio,
            )
    persistence_drop_triggered = bool(
        persistence_threshold > 0.0
        and persistence_hold_steps > 0
        and persistence_confirmed_best_mean >= persistence_threshold
        and mean_return <= persistence_confirmed_best_mean - persistence_drop
        and (
            persistence_drop_confirmation_count <= 1
            or mean_return <= persistence_drop_floor
        )
    )
    if persistence_drop_triggered:
        runtime._adaptive_imag_compensation_persistence_drop_confirmation_streak = min(
            persistence_drop_confirmation_count,
            int(getattr(runtime, "_adaptive_imag_compensation_persistence_drop_confirmation_streak", 0)) + 1,
        )
    else:
        runtime._adaptive_imag_compensation_persistence_drop_confirmation_streak = 0
    persistence_would_arm = bool(
        persistence_drop_triggered
        and runtime._adaptive_imag_compensation_persistence_drop_confirmation_streak >= persistence_drop_confirmation_count
    )
    if persistence_would_arm:
        if resolved_step >= persistence_min_step:
            runtime._adaptive_imag_compensation_persistence_hold_until_step = max(
                runtime._adaptive_imag_compensation_persistence_hold_until_step,
                resolved_step + persistence_hold_steps,
            )
            runtime._adaptive_imag_compensation_persistence_armed_step = resolved_step
            runtime._adaptive_imag_compensation_persistence_ever_armed = True
        elif entry_threshold > 0.0 and entry_hold_steps > 0 and persistence_confirmed_best_mean >= entry_threshold:
            runtime._adaptive_imag_post_entry_hold_until_step = max(
                runtime._adaptive_imag_post_entry_hold_until_step,
                persistence_min_step,
            )
            runtime._adaptive_imag_post_entry_source = "external_eval"
            runtime._adaptive_imag_last_post_entry_source = "external_eval"

    release_eval_ratio = max(0.0, float(getattr(runtime.config, "adaptive_imag_compensation_persistence_release_eval_ratio", 0.0)))
    release_steps = max(0, int(getattr(runtime.config, "adaptive_imag_compensation_persistence_release_steps", 0)))
    release_eval_threshold = post_solved_threshold * release_eval_ratio if post_solved_threshold > 0.0 else 0.0
    if (
        bool(getattr(runtime, "_adaptive_imag_compensation_persistence_ever_armed", False))
        and release_steps > 0
        and release_eval_threshold > 0.0
        and mean_return >= release_eval_threshold
        and (
            post_solved_threshold <= 0.0
            or mean_return < post_solved_threshold
            or not post_solved_confirmed
        )
    ):
        runtime._adaptive_imag_compensation_persistence_release_hold_until_step = max(
            runtime._adaptive_imag_compensation_persistence_release_hold_until_step,
            resolved_step + release_steps,
        )

    threshold = post_solved_threshold
    hold_steps = max(0, int(getattr(runtime.config, "adaptive_imag_compensation_post_solved_hold_steps", 0)))
    min_step = max(0, int(getattr(runtime.config, "adaptive_imag_compensation_post_solved_min_step", 0)))
    requires_persistence = bool(getattr(runtime.config, "adaptive_imag_compensation_post_solved_requires_persistence", False))
    persistence_ready = (not requires_persistence) or bool(getattr(runtime, "_adaptive_imag_compensation_persistence_ever_active", False))
    step_ready = resolved_step >= min_step
    if (
        post_solved_confirmed
        and threshold > 0.0
        and hold_steps > 0
        and mean_return >= threshold
        and persistence_ready
        and step_ready
    ):
        runtime._adaptive_imag_compensation_post_solved_hold_until_step = max(
            runtime._adaptive_imag_compensation_post_solved_hold_until_step,
            resolved_step + hold_steps,
        )


def resolve_imag_compensation_phase(
    runtime: Any,
    *,
    gap_abs: float,
    gap_mean: float,
    commit_gap_abs: float,
    commit_gap_mean: float,
    cont_mean: float,
    actor_target_raw: float,
    adaptive_pressure: float,
    adaptive_scale: float,
    trigger_confirmed: bool,
) -> CompensationDecision:
    del gap_mean, commit_gap_abs, commit_gap_mean, cont_mean, actor_target_raw
    current_step = max(0, int(getattr(runtime, "global_step", 0)))
    episode_return_ema = (
        float(runtime._episode_return_ema) if getattr(runtime, "_episode_return_initialized", False) else 0.0
    )
    external_eval_best_mean = float(getattr(runtime, "_external_eval_best_mean", -float("inf")))
    external_eval_last_mean = float(getattr(runtime, "_external_eval_last_mean", -float("inf")))
    persistence_threshold = float(
        getattr(runtime.config, "adaptive_imag_compensation_persistence_eval_threshold", 0.0)
    )
    post_solved_threshold = float(
        getattr(runtime.config, "adaptive_imag_compensation_post_solved_eval_threshold", 0.0)
    )
    post_solved_min_step = max(
        0, int(getattr(runtime.config, "adaptive_imag_compensation_post_solved_min_step", 0))
    )
    persistence_min_step = max(
        0, int(getattr(runtime.config, "adaptive_imag_compensation_persistence_min_step", 0))
    )
    persistence_hold_steps = max(
        0, int(getattr(runtime.config, "adaptive_imag_compensation_persistence_hold_steps", 0))
    )
    post_solved_hold_steps = max(
        0, int(getattr(runtime.config, "adaptive_imag_compensation_post_solved_hold_steps", 0))
    )
    post_solved_requires_persistence = bool(
        getattr(runtime.config, "adaptive_imag_compensation_post_solved_requires_persistence", False)
    )
    trigger_pressure = max(
        0.0,
        float(getattr(runtime.config, "adaptive_imag_compensation_trigger_min_pressure", 0.0)),
    )
    trigger_active = bool(
        trigger_confirmed and adaptive_scale > 0.0 and adaptive_pressure > trigger_pressure
    )
    persistence_seen = bool(getattr(runtime, "_adaptive_imag_compensation_persistence_ever_active", False))
    persistence_active = bool(
        current_step >= persistence_min_step
        and current_step <= int(getattr(runtime, "_adaptive_imag_compensation_persistence_hold_until_step", -1))
    )
    post_solved_active = bool(
        current_step <= int(getattr(runtime, "_adaptive_imag_compensation_post_solved_hold_until_step", -1))
    )

    if (
        not post_solved_active
        and post_solved_threshold > 0.0
        and current_step >= post_solved_min_step
        and external_eval_last_mean >= post_solved_threshold
        and (not post_solved_requires_persistence or persistence_seen)
    ):
        resolved_post_solved_hold_until = (
            current_step + post_solved_hold_steps if post_solved_hold_steps > 0 else current_step
        )
        runtime._adaptive_imag_compensation_post_solved_hold_until_step = max(
            int(getattr(runtime, "_adaptive_imag_compensation_post_solved_hold_until_step", -1)),
            resolved_post_solved_hold_until,
        )
        post_solved_active = True

    if persistence_active and not persistence_seen:
        runtime._adaptive_imag_compensation_persistence_ever_active = True
        persistence_seen = True

    if (
        not post_solved_active
        and not persistence_active
        and persistence_threshold > 0.0
        and current_step >= persistence_min_step
        and external_eval_best_mean >= persistence_threshold
    ):
        resolved_persistence_hold_until = (
            current_step + persistence_hold_steps if persistence_hold_steps > 0 else current_step
        )
        runtime._adaptive_imag_compensation_persistence_hold_until_step = max(
            int(getattr(runtime, "_adaptive_imag_compensation_persistence_hold_until_step", -1)),
            resolved_persistence_hold_until,
        )
        runtime._adaptive_imag_compensation_persistence_armed_step = current_step
        runtime._adaptive_imag_compensation_persistence_ever_armed = True
        runtime._adaptive_imag_compensation_persistence_ever_active = True
        persistence_active = True
        persistence_seen = True

    if post_solved_active:
        persistence_active = False

    post_entry_source = str(getattr(runtime, "_adaptive_imag_post_entry_source", "idle"))
    release_quality_ready = True
    release_active = False
    release_progress = 0.0
    if trigger_active and post_entry_source != "idle":
        release_progress = resolve_standard_soft_fallback_trigger_release_progress(
            current_step=current_step,
            compensation_phase=CompensationPhase.TRIGGER,
            post_entry_source=post_entry_source,
            quality_release_ready=release_quality_ready,
            runtime=runtime,
            config=runtime.config,
        )
        release_active = True
    else:
        runtime._adaptive_imag_standard_soft_fallback_trigger_release_step = -1

    compensation_phase = CompensationPhase.IDLE
    if post_solved_active:
        compensation_phase = CompensationPhase.POST_SOLVED
    elif persistence_active:
        compensation_phase = CompensationPhase.PERSISTENCE
    elif release_active:
        compensation_phase = CompensationPhase.RELEASE
    elif trigger_active:
        compensation_phase = CompensationPhase.TRIGGER

    if compensation_phase not in (CompensationPhase.TRIGGER, CompensationPhase.RELEASE):
        runtime._adaptive_imag_external_post_entry_pending_commit = False
        runtime._adaptive_imag_external_post_entry_soft_fallback_latched = False
        runtime._adaptive_imag_post_entry_commit_candidate_since_step = -1
        if compensation_phase == CompensationPhase.IDLE:
            runtime._adaptive_imag_post_entry_source = "idle"

    runtime._adaptive_imag_compensation_phase = compensation_phase
    return CompensationDecision(
        compensation_phase=compensation_phase,
        trigger_active=trigger_active,
        trigger_confirmation_ready=bool(trigger_confirmed),
        release_active=release_active,
        release_progress=float(release_progress),
        persistence_active=persistence_active,
        persistence_seen=persistence_seen,
        post_solved_active=post_solved_active,
        persistence_escape_active=False,
        post_entry_source=str(getattr(runtime, "_adaptive_imag_post_entry_source", "idle")),
        trigger_persistence_handoff_active=False,
        trigger_persistence_handoff_release_progress=float(release_progress),
        trigger_persistence_handoff_eval_mean=float(external_eval_best_mean),
        episode_return_ema=episode_return_ema,
    )


def resolve_imag_continue_cap(
    runtime: Any,
    continue_probs_im_raw: Optional[Tensor],
    values_im: Tensor,
    returns_raw: Tensor,
) -> Tuple[float, Dict[str, Any]]:
    base_cap = float(getattr(runtime.config, "imag_continue_prob_cap", 0.0))
    adaptive_enabled = bool(getattr(runtime.config, "adaptive_imag_continue_cap", False))
    adaptive_max = float(getattr(runtime.config, "adaptive_imag_continue_cap_max", 0.95))

    if not adaptive_enabled or continue_probs_im_raw is None:
        return base_cap, {
            "stage": CompensationPhase.IDLE,
            "pretrigger_active": False,
            "trigger_active": False,
            "release_active": False,
            "release_progress": 0.0,
            "post_solved_active": False,
            "episode_return_ema": float(runtime._episode_return_ema)
            if getattr(runtime, "_episode_return_initialized", False)
            else 0.0,
        }

    if not (0.0 < adaptive_max < 1.0):
        adaptive_max = base_cap if 0.0 < base_cap < 1.0 else 0.95
    if not (0.0 < adaptive_max < 1.0):
        return base_cap, {
            "stage": CompensationPhase.IDLE,
            "pretrigger_active": False,
            "trigger_active": False,
            "release_active": False,
            "release_progress": 0.0,
            "post_solved_active": False,
            "episode_return_ema": float(runtime._episode_return_ema)
            if getattr(runtime, "_episode_return_initialized", False)
            else 0.0,
        }

    if 0.0 < base_cap < 1.0:
        cap_hi = min(base_cap, adaptive_max)
    else:
        cap_hi = adaptive_max
    cap_lo = float(getattr(runtime.config, "adaptive_imag_continue_cap_min", 0.80))
    cap_lo = max(0.0, min(cap_lo, cap_hi))

    gap_baseline = values_im[:, :-1].detach()
    gap_abs = float((gap_baseline - returns_raw.detach()).abs().mean())
    gap_mean = float((returns_raw.detach() - gap_baseline).mean())
    commit_gap_abs = gap_abs
    commit_gap_mean = gap_mean
    cont_mean = float(continue_probs_im_raw.detach().mean())
    actor_target_raw = float(returns_raw.detach().mean())

    current_step = max(0, int(getattr(runtime, "global_step", 0)))
    target_gap = float(getattr(runtime.config, "adaptive_imag_continue_cap_target_gap", 4.0))
    target_continue = float(getattr(runtime.config, "adaptive_imag_continue_cap_target_continue", 0.97))
    continue_gain = float(getattr(runtime.config, "adaptive_imag_continue_cap_continue_gain", 0.50))
    entry_window_steps = max(0, int(getattr(runtime.config, "adaptive_imag_entry_window_steps", 0)))
    entry_target_continue = float(getattr(runtime.config, "adaptive_imag_entry_target_continue", 0.0))
    entry_continue_gain_scale = max(0.0, float(getattr(runtime.config, "adaptive_imag_entry_continue_gain_scale", 1.0)))
    phase_warmup_end = max(0, int(getattr(runtime.config, "wm_pretrain_steps", 0)) + int(getattr(runtime.config, "warmup_steps", 0)))
    in_entry_window = (
        entry_window_steps > 0
        and current_step >= phase_warmup_end
        and current_step < phase_warmup_end + entry_window_steps
    )
    if in_entry_window and 0.0 < entry_target_continue < 1.0:
        target_continue = min(target_continue, entry_target_continue)
        continue_gain *= entry_continue_gain_scale

    gap_pressure = max(0.0, gap_abs - target_gap)
    cont_pressure = max(0.0, cont_mean - target_continue)
    actor_pressure = max(
        0.0,
        actor_target_raw - float(getattr(runtime.config, "adaptive_imag_continue_cap_target_actor", 20.0)),
    )

    raw_pressure_unscaled = (
        gap_pressure * float(getattr(runtime.config, "adaptive_imag_continue_cap_gap_gain", 0.02))
        + cont_pressure * continue_gain
        + actor_pressure * float(getattr(runtime.config, "adaptive_imag_continue_cap_actor_gain", 0.002))
    )
    raw_pressure = raw_pressure_unscaled
    warmup_steps = max(0, int(getattr(runtime.config, "adaptive_imag_continue_cap_warmup_steps", 0)))
    ramp_steps = max(0, int(getattr(runtime.config, "adaptive_imag_continue_cap_ramp_steps", 0)))
    if current_step < warmup_steps:
        adaptive_scale = 0.0
    elif ramp_steps <= 0:
        adaptive_scale = 1.0
    else:
        adaptive_scale = min(1.0, float(current_step - warmup_steps) / float(ramp_steps))
    raw_pressure *= adaptive_scale
    post_trigger_clip = float(getattr(runtime.config, "adaptive_imag_compensation_trigger_return_delta_clip", 0.0))
    post_trigger_min_pressure = float(getattr(runtime.config, "adaptive_imag_compensation_trigger_min_pressure", 0.0))
    confirmation_steps = max(1, int(getattr(runtime.config, "adaptive_imag_compensation_trigger_confirmation_steps", 1)))
    raw_post_trigger_pressure_active = bool(adaptive_scale > 0.0 and raw_pressure > post_trigger_min_pressure)
    if raw_post_trigger_pressure_active:
        runtime._adaptive_imag_compensation_trigger_pressure_streak = min(
            runtime._adaptive_imag_compensation_trigger_pressure_streak + 1,
            confirmation_steps,
        )
        pressure_confirmation_scale = (
            1.0
            if confirmation_steps <= 1
            else min(
                1.0,
                float(max(runtime._adaptive_imag_compensation_trigger_pressure_streak - 1, 0))
                / float(confirmation_steps - 1),
            )
        )
        pressure = post_trigger_min_pressure + (
            raw_pressure - post_trigger_min_pressure
        ) * pressure_confirmation_scale
    else:
        runtime._adaptive_imag_compensation_trigger_pressure_streak = 0
        pressure_confirmation_scale = 0.0
        pressure = raw_pressure
    trigger_confirmed = bool(raw_post_trigger_pressure_active and pressure_confirmation_scale >= 1.0)
    post_trigger_pressure_active = trigger_confirmed
    if post_trigger_clip > 0.0 and post_trigger_pressure_active:
        commit_delta = (returns_raw.detach() - gap_baseline).clamp(-post_trigger_clip, post_trigger_clip)
        commit_returns = gap_baseline + commit_delta
        commit_gap_abs = float((gap_baseline - commit_returns).abs().mean())
        commit_gap_mean = float((commit_returns - gap_baseline).mean())
    desired_cap = max(cap_lo, min(cap_hi, cap_hi - pressure))
    compensation_decision = resolve_imag_compensation_phase(
        runtime,
        gap_abs=gap_abs,
        gap_mean=gap_mean,
        commit_gap_abs=commit_gap_abs,
        commit_gap_mean=commit_gap_mean,
        cont_mean=cont_mean,
        actor_target_raw=actor_target_raw,
        adaptive_pressure=pressure,
        adaptive_scale=adaptive_scale,
        trigger_confirmed=trigger_confirmed,
    )
    compensation_phase = canonicalize_compensation_phase(compensation_decision.get("stage", CompensationPhase.IDLE))
    persistence_escape_active = bool(compensation_decision.get("post_entry_persistence_escape_active", False))
    persistence_escape_soft_floors = resolve_persistence_escape_soft_floors(
        config=runtime.config,
        persistence_armed_step=int(
            getattr(runtime, "_adaptive_imag_compensation_persistence_armed_step", -1)
        ),
        current_step=int(current_step),
        external_eval_best_mean=float(getattr(runtime, "_external_eval_best_mean", 0.0)),
    )
    trigger_release_quality_state = resolve_post_trigger_quality_state(
        config=runtime.config,
        quality_good_streak=int(getattr(runtime, "_adaptive_imag_compensation_trigger_quality_good_streak", 0)),
        return_baseline=gap_baseline,
        returns=returns_raw,
    )
    standard_soft_fallback_trigger_release_active = bool(
        compensation_phase in (CompensationPhase.TRIGGER, CompensationPhase.RELEASE)
        and str(compensation_decision.post_entry_source) != "idle"
    )
    standard_soft_fallback_trigger_release_progress = resolve_standard_soft_fallback_trigger_release_progress(
        current_step=current_step,
        compensation_phase=compensation_phase,
        post_entry_source=str(compensation_decision.post_entry_source),
        quality_release_ready=bool(trigger_release_quality_state.get("release_ready", 1.0)),
        runtime=runtime,
        config=runtime.config,
    )
    external_eval_best_mean = float(getattr(runtime, "_external_eval_best_mean", -float("inf")))
    post_solved_highwater_active = runtime._is_post_solved_highwater_active(
        compensation_phase=compensation_phase,
        external_eval_best_mean=external_eval_best_mean,
    )
    persistence_highwater_active = runtime._is_persistence_highwater_active(
        compensation_phase=compensation_phase,
        external_eval_best_mean=external_eval_best_mean,
    )
    persistence_highwater_cap_active = 0.0
    persistence_highwater_cap_value = 0.0
    if compensation_phase in (CompensationPhase.TRIGGER, CompensationPhase.RELEASE) and standard_soft_fallback_trigger_release_active:
        soft_cap_floor = float(getattr(runtime.config, "adaptive_imag_post_entry_soft_cap_floor", 0.0))
        if soft_cap_floor <= 0.0:
            soft_cap_floor = float(getattr(runtime.config, "adaptive_imag_post_entry_soft_cap", 0.0))
        if soft_cap_floor > 0.0:
            soft_cap_floor = max(cap_lo, min(cap_hi, soft_cap_floor))
            if desired_cap < soft_cap_floor:
                desired_cap = soft_cap_floor + (
                    desired_cap - soft_cap_floor
                ) * standard_soft_fallback_trigger_release_progress
    elif compensation_phase == CompensationPhase.PERSISTENCE:
        stage_cap = float(getattr(runtime.config, "adaptive_imag_compensation_persistence_cap", cap_hi))
        if persistence_highwater_active:
            highwater_cap = float(getattr(runtime.config, "adaptive_imag_compensation_persistence_highwater_cap", 0.0))
            if highwater_cap > 0.0:
                stage_cap = min(stage_cap, highwater_cap)
                persistence_highwater_cap_active = 1.0
                persistence_highwater_cap_value = highwater_cap
        stage_cap = max(cap_lo, min(cap_hi, stage_cap))
        desired_cap = min(desired_cap, stage_cap)
        if persistence_escape_active:
            soft_cap_floor = float(persistence_escape_soft_floors.get("cap_floor", 0.0))
            if soft_cap_floor > 0.0:
                desired_cap = max(desired_cap, max(cap_lo, min(cap_hi, soft_cap_floor)))
    elif compensation_phase == CompensationPhase.POST_SOLVED:
        stage_cap = float(getattr(runtime.config, "adaptive_imag_compensation_post_solved_cap", cap_lo))
        if post_solved_highwater_active:
            highwater_cap = float(getattr(runtime.config, "adaptive_imag_compensation_post_solved_highwater_cap", 0.0))
            if highwater_cap > 0.0:
                stage_cap = max(stage_cap, highwater_cap)
        stage_cap = max(cap_lo, min(cap_hi, stage_cap))
        desired_cap = min(desired_cap, stage_cap)

    ema = float(getattr(runtime.config, "adaptive_imag_continue_cap_ema", 0.8))
    ema = min(max(ema, 0.0), 0.999)
    prev = runtime._adaptive_imag_continue_cap_state
    if prev is None:
        resolved = desired_cap
    else:
        resolved = ema * float(prev) + (1.0 - ema) * desired_cap
    resolved = max(cap_lo, min(cap_hi, resolved))
    if persistence_highwater_cap_active > 0.0:
        resolved = min(resolved, max(cap_lo, min(cap_hi, persistence_highwater_cap_value)))
    runtime._adaptive_imag_continue_cap_state = resolved

    effective_continue_mean = float(cont_mean)
    if continue_probs_im_raw is not None:
        effective_continue_mean = float(continue_probs_im_raw.detach().clamp(max=resolved).mean())
    trigger_persistence_handoff_landing_guard_active = False
    trigger_persistence_handoff_continue_mean_effective = float(effective_continue_mean)
    if (
        compensation_phase in (CompensationPhase.TRIGGER, CompensationPhase.RELEASE)
        and standard_soft_fallback_trigger_release_active
        and not bool(compensation_decision.trigger_persistence_handoff_active)
        and bool(getattr(runtime.config, "adaptive_imag_compensation_trigger_persistence_handoff_enabled", False))
    ):
        trigger_persistence_handoff_min_release_progress = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        runtime.config,
                        "adaptive_imag_compensation_trigger_persistence_handoff_min_release_progress",
                        0.75,
                    )
                ),
            ),
        )
        trigger_persistence_handoff_gap_threshold = max(
            0.0,
            float(
                getattr(
                    runtime.config,
                    "adaptive_imag_compensation_trigger_persistence_handoff_gap_threshold",
                    0.0,
                )
            ),
        )
        trigger_persistence_handoff_continue_threshold = float(
            getattr(
                runtime.config,
                "adaptive_imag_compensation_trigger_persistence_handoff_continue_threshold",
                0.0,
            )
        )
        trigger_persistence_handoff_eval_threshold = float(
            getattr(
                runtime.config,
                "adaptive_imag_compensation_trigger_persistence_handoff_eval_threshold",
                0.0,
            )
        )
        trigger_persistence_handoff_eval_mean = max(
            float(compensation_decision.trigger_persistence_handoff_eval_mean),
            external_eval_best_mean,
            float(getattr(runtime, "_adaptive_imag_compensation_persistence_confirmed_best_mean", -float("inf"))),
            float(getattr(runtime, "_adaptive_imag_compensation_post_solved_confirmed_best_mean", -float("inf"))),
        )
        trigger_persistence_handoff_gap_ok = (
            trigger_persistence_handoff_gap_threshold <= 0.0 or gap_abs >= trigger_persistence_handoff_gap_threshold
        )
        trigger_persistence_handoff_continue_ok = (
            trigger_persistence_handoff_continue_threshold <= 0.0
            or effective_continue_mean <= trigger_persistence_handoff_continue_threshold
        )
        trigger_persistence_handoff_eval_ok = (
            trigger_persistence_handoff_eval_threshold <= 0.0
            or trigger_persistence_handoff_eval_mean >= trigger_persistence_handoff_eval_threshold
        )
        if (
            standard_soft_fallback_trigger_release_progress >= trigger_persistence_handoff_min_release_progress
            and trigger_persistence_handoff_gap_ok
            and trigger_persistence_handoff_continue_ok
            and trigger_persistence_handoff_eval_ok
        ):
            if runtime._is_persistence_drop_confirmation_pending():
                trigger_persistence_handoff_continue_mean_effective = float(effective_continue_mean)
            else:
                trigger_persistence_handoff_landing_guard_active = bool(
                    is_trigger_persistence_handoff_landing_guard_active(
                        config=runtime.config,
                        gap_abs=gap_abs,
                        continue_mean=effective_continue_mean,
                    )
                )
                if not trigger_persistence_handoff_landing_guard_active:
                    persistence_hold_steps = max(
                        0,
                        int(getattr(runtime.config, "adaptive_imag_compensation_persistence_hold_steps", 0)),
                    )
                    runtime._adaptive_imag_compensation_persistence_hold_until_step = max(
                        int(getattr(runtime, "_adaptive_imag_compensation_persistence_hold_until_step", -1)),
                        current_step + persistence_hold_steps,
                    )
                    runtime._adaptive_imag_compensation_persistence_armed_step = current_step
                    runtime._adaptive_imag_compensation_persistence_ever_armed = True
                    runtime._adaptive_imag_compensation_persistence_ever_active = True
                    compensation_decision = compensation_decision.with_updates(
                        compensation_phase=CompensationPhase.PERSISTENCE,
                        persistence_active=True,
                        persistence_seen=True,
                        trigger_persistence_handoff_active=True,
                        trigger_persistence_handoff_release_progress=float(
                            standard_soft_fallback_trigger_release_progress
                        ),
                        trigger_persistence_handoff_eval_mean=float(
                            trigger_persistence_handoff_eval_mean
                        ),
                    )
                    compensation_phase = compensation_decision.compensation_phase

    if hasattr(runtime, "metrics"):
        runtime.metrics.update(
            {
                "imag/continue_prob_cap_static": float(base_cap) if 0.0 < base_cap < 1.0 else 0.0,
                "imag/continue_prob_cap_dynamic": float(resolved),
                "imag/persistence_escape_band_floor_release_progress": float(
                    persistence_escape_soft_floors.get("release_progress", 0.0)
                ),
                "imag/continue_cap_pressure": float(pressure),
                "imag/continue_cap_pressure_raw": float(raw_pressure),
                "imag/continue_cap_pressure_unscaled": float(raw_pressure_unscaled),
                "imag/continue_cap_pressure_confirmation_scale": float(pressure_confirmation_scale),
                "imag/continue_cap_adaptive_scale": float(adaptive_scale),
                "imag/continue_cap_gap_abs_raw": float(gap_abs),
                "imag/continue_cap_actor_target_raw": float(actor_target_raw),
                "imag/continue_cap_continue_mean_raw": float(cont_mean),
                "imag/entry_window_active": float(in_entry_window),
                "imag/entry_target_continue": float(target_continue),
                "imag/entry_continue_gain": float(continue_gain),
                "imag/compensation_phase": str(compensation_decision.compensation_phase),
                "imag/compensation_post_solved_highwater_active": float(bool(post_solved_highwater_active)),
                "imag/compensation_persistence_highwater_active": float(bool(persistence_highwater_active)),
                "imag/compensation_persistence_highwater_cap_active": float(bool(persistence_highwater_cap_active)),
                "imag/compensation_persistence_highwater_cap_value": float(persistence_highwater_cap_value),
                "imag/compensation_trigger_active": float(bool(compensation_decision.trigger_active)),
                "imag/compensation_trigger_confirmation_ready": float(bool(compensation_decision.trigger_confirmation_ready)),
                "imag/compensation_trigger_raw_active": float(raw_post_trigger_pressure_active),
                "imag/compensation_trigger_pressure_streak": float(runtime._adaptive_imag_compensation_trigger_pressure_streak),
                "imag/compensation_post_entry_source": str(compensation_decision.post_entry_source),
                "imag/compensation_standard_soft_fallback_trigger_release_active": float(
                    standard_soft_fallback_trigger_release_active
                ),
                "imag/compensation_standard_soft_fallback_trigger_release_progress": float(
                    standard_soft_fallback_trigger_release_progress
                ),
                "imag/compensation_standard_soft_fallback_trigger_release_quality_good": float(
                    trigger_release_quality_state.get("good", 0.0)
                ),
                "imag/compensation_standard_soft_fallback_trigger_release_quality_ready": float(
                    trigger_release_quality_state.get("release_ready", 1.0)
                ),
                "imag/compensation_persistence_active": float(bool(compensation_decision.persistence_active)),
                "imag/compensation_persistence_seen": float(bool(compensation_decision.persistence_seen)),
                "imag/compensation_persistence_escape_active": float(bool(compensation_decision.persistence_escape_active)),
                "imag/compensation_trigger_persistence_handoff_active": float(
                    bool(compensation_decision.trigger_persistence_handoff_active)
                ),
                "imag/compensation_trigger_persistence_handoff_landing_guard_active": float(
                    bool(trigger_persistence_handoff_landing_guard_active)
                ),
                "imag/compensation_trigger_persistence_handoff_release_progress": float(
                    compensation_decision.trigger_persistence_handoff_release_progress
                ),
                "imag/compensation_trigger_persistence_handoff_eval_mean": float(
                    compensation_decision.trigger_persistence_handoff_eval_mean
                ),
                "imag/compensation_trigger_persistence_handoff_continue_mean_effective": float(
                    trigger_persistence_handoff_continue_mean_effective
                ),
                "imag/compensation_post_solved_active": float(bool(compensation_decision.post_solved_active)),
                "imag/compensation_release_active": float(bool(compensation_decision.release_active)),
                "imag/compensation_release_progress": float(compensation_decision.release_progress),
                "imag/compensation_episode_return_ema": float(compensation_decision.episode_return_ema),
                "train/external_eval_last_mean": float(runtime._external_eval_last_mean)
                if runtime._external_eval_last_mean > -float("inf")
                else 0.0,
                "train/external_eval_best_mean": float(runtime._external_eval_best_mean)
                if runtime._external_eval_best_mean > -float("inf")
                else 0.0,
                "train/external_eval_post_entry_confirmation_streak": float(
                    runtime._adaptive_imag_post_entry_eval_confirmation_streak
                ),
                "train/external_eval_post_entry_confirmation_required": float(
                    runtime._resolve_adaptive_eval_confirmation_count(
                        "adaptive_imag_post_entry_eval_confirmation_count"
                    )
                ),
                "train/external_eval_persistence_confirmation_streak": float(
                    runtime._adaptive_imag_compensation_persistence_eval_confirmation_streak
                ),
                "train/external_eval_persistence_confirmation_required": float(
                    runtime._resolve_adaptive_eval_confirmation_count(
                        "adaptive_imag_compensation_persistence_eval_confirmation_count"
                    )
                ),
                "train/external_eval_persistence_drop_confirmation_streak": float(
                    runtime._adaptive_imag_compensation_persistence_drop_confirmation_streak
                ),
                "train/external_eval_persistence_drop_confirmation_required": float(
                    max(
                        1,
                        int(
                            getattr(
                                runtime.config,
                                "adaptive_imag_compensation_persistence_drop_confirmation_count",
                                0,
                            )
                        ),
                    )
                ),
            }
        )

    decision_dict = dict(compensation_decision)
    decision_dict["stage"] = compensation_decision.compensation_phase
    decision_dict["post_entry_persistence_escape_active"] = bool(
        compensation_decision.persistence_escape_active
    )
    decision_dict["release_active"] = bool(compensation_decision.release_active)
    decision_dict["release_progress"] = float(compensation_decision.release_progress)
    return resolved, decision_dict


def build_runtime_rl_context(runtime: Any) -> Dict[str, Any]:
    current_step = max(0, int(getattr(runtime, "global_step", 0)))
    compensation_phase = canonicalize_compensation_phase(
        getattr(runtime, "_adaptive_imag_compensation_phase", CompensationPhase.IDLE)
    )
    if current_step <= int(getattr(runtime, "_adaptive_imag_compensation_post_solved_hold_until_step", -1)):
        compensation_phase = CompensationPhase.POST_SOLVED
    elif current_step <= int(getattr(runtime, "_adaptive_imag_compensation_persistence_hold_until_step", -1)):
        compensation_phase = CompensationPhase.PERSISTENCE

    external_eval_last_mean = float(getattr(runtime, "_external_eval_last_mean", -float("inf")))
    external_eval_best_mean = float(getattr(runtime, "_external_eval_best_mean", -float("inf")))
    if not math.isfinite(external_eval_last_mean):
        external_eval_last_mean = 0.0
    if not math.isfinite(external_eval_best_mean):
        external_eval_best_mean = 0.0

    standard_release_active = bool(
        compensation_phase in (CompensationPhase.TRIGGER, CompensationPhase.RELEASE)
        and str(getattr(runtime, "_adaptive_imag_post_entry_source", "idle")) != "idle"
    )
    standard_release_progress = resolve_standard_soft_fallback_trigger_release_progress(
        current_step=current_step,
        compensation_phase=compensation_phase,
        post_entry_source=str(getattr(runtime, "_adaptive_imag_post_entry_source", "idle")),
        quality_release_ready=True,
        runtime=runtime,
        config=getattr(runtime, "config", None),
    )

    return {
        "compensation_phase": compensation_phase,
        "compensation_post_entry_commit_ready": bool(
            not getattr(runtime, "_adaptive_imag_external_post_entry_pending_commit", False)
        ),
        "compensation_post_entry_pending_active": bool(
            getattr(runtime, "_adaptive_imag_post_entry_pending_recovery_latched", False)
        ),
        "compensation_post_entry_pending_recovery_latched": bool(
            getattr(runtime, "_adaptive_imag_post_entry_pending_recovery_latched", False)
        ),
        "compensation_post_entry_persistence_escape_active": False,
        "compensation_persistence_release_active": bool(
            compensation_phase == CompensationPhase.PERSISTENCE
            and current_step <= int(
                getattr(runtime, "_adaptive_imag_compensation_persistence_release_hold_until_step", -1)
            )
        ),
        "compensation_standard_soft_fallback_trigger_release_active": standard_release_active,
        "compensation_standard_soft_fallback_trigger_release_progress": float(
            standard_release_progress
        ),
        "external_eval_last_mean": external_eval_last_mean,
        "external_eval_best_mean": external_eval_best_mean,
    }


def detect_runtime_compensation_guard_mismatch(
    runtime: Any,
    runtime_compensation_phase: str,
) -> tuple[bool, str]:
    if canonicalize_compensation_phase(runtime_compensation_phase) != CompensationPhase.POST_SOLVED:
        return False, "none"

    actor_anchor_pull_enabled = any(
        float(getattr(runtime.config, attr, 0.0)) > 0.0
        for attr in (
            "adaptive_imag_compensation_post_solved_actor_anchor_pull",
            "adaptive_imag_compensation_post_solved_actor_anchor_hard_pull",
            "adaptive_imag_compensation_post_solved_actor_anchor_latched_pull",
            "adaptive_imag_compensation_post_solved_actor_anchor_highwater_pull",
        )
    )
    actor_anchor_kl_enabled = any(
        float(getattr(runtime.config, attr, 0.0)) > 0.0
        for attr in (
            "adaptive_imag_compensation_post_solved_actor_anchor_kl",
            "adaptive_imag_compensation_post_solved_actor_anchor_latched_kl",
            "adaptive_imag_compensation_post_solved_actor_anchor_highwater_kl",
        )
    )
    real_actor_anchor_kl_enabled = (
        float(
            getattr(
                runtime.config,
                "adaptive_imag_compensation_post_solved_real_actor_anchor_kl",
                0.0,
            )
        )
        > 0.0
    )
    critic_anchor_enabled = (
        float(
            getattr(
                runtime.config,
                "adaptive_imag_compensation_post_solved_critic_anchor_weight",
                0.0,
            )
        )
        > 0.0
    )
    drift_damping_enabled = (
        float(
            getattr(
                runtime.config,
                "adaptive_imag_compensation_post_solved_drift_damping_eval_threshold",
                0.0,
            )
        )
        > 0.0
    )

    guards_disabled = not any(
        (
            actor_anchor_pull_enabled,
            actor_anchor_kl_enabled,
            real_actor_anchor_kl_enabled,
            critic_anchor_enabled,
            drift_damping_enabled,
        )
    )
    if not guards_disabled:
        return False, "none"
    return True, "post_solved_guards_disabled"
