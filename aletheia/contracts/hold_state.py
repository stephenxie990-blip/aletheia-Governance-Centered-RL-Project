"""Canonical 3-channel hold state management.

Single source of truth for HOLD_CHANNELS and the hold-state helper functions.
Both ``contracts.consumers`` and ``training.compensation_kernel`` re-export
from this module so that all existing import paths keep working.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "HOLD_CHANNELS",
    "active_hold_channel",
    "hold_state_summary",
    "hold_state_to_float",
    "make_empty_hold_state",
    "select_hold_channel_value",
    "update_hold_state_channels",
]

HOLD_CHANNELS = ("pre_transition", "active_transition", "post_transition")


def make_empty_hold_state() -> dict[str, float]:
    return {channel: 0.0 for channel in HOLD_CHANNELS}


def hold_state_to_float(hold_state: Any) -> float:
    if isinstance(hold_state, dict):
        return float(max(float(hold_state.get(channel, 0.0)) for channel in HOLD_CHANNELS))
    return float(hold_state)


def active_hold_channel(late_gate_mean: float) -> str:
    if late_gate_mean >= 0.80:
        return "post_transition"
    if late_gate_mean >= 0.55:
        return "active_transition"
    return "pre_transition"


def select_hold_channel_value(hold_state: Any, late_gate_mean: float) -> float:
    if isinstance(hold_state, dict):
        return float(hold_state.get(active_hold_channel(late_gate_mean), 0.0))
    return float(hold_state)


def update_hold_state_channels(
    hold_state: Any,
    late_gate_mean: float,
    new_value: float,
    release_pressure_mean: float,
    inactive_decay: float = 0.99,
) -> dict[str, float]:
    del release_pressure_mean  # compat: callers may pass this; currently unused
    if not isinstance(hold_state, dict):
        hold_state = {channel: float(hold_state) for channel in HOLD_CHANNELS}
    current_channel = active_hold_channel(late_gate_mean)
    updated: dict[str, float] = {}
    for channel in HOLD_CHANNELS:
        if channel == current_channel:
            updated[channel] = float(min(1.0, max(0.0, new_value)))
        else:
            updated[channel] = float(
                min(1.0, max(0.0, float(hold_state.get(channel, 0.0)) * inactive_decay))
            )
    if updated["active_transition"] < updated["pre_transition"] * 0.5:
        updated["active_transition"] = max(
            updated["active_transition"],
            updated["pre_transition"] * 0.5,
        )
    if updated["post_transition"] < updated["active_transition"] * 0.5:
        updated["post_transition"] = max(
            updated["post_transition"],
            updated["active_transition"] * 0.5,
        )
    return updated


def hold_state_summary(hold_state: Any) -> dict[str, float]:
    if isinstance(hold_state, dict):
        return {
            f"hold_state_{channel}": float(hold_state.get(channel, 0.0))
            for channel in HOLD_CHANNELS
        }
    return {"hold_state_scalar": float(hold_state)}
