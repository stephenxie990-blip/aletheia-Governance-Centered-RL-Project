from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Union

import torch
from torch import Tensor

from .authority import (
    RetentionCaptureDecision,
    compute_bootstrap_trigger_entry_contract,
    compute_post_transition_retention_capture,
)
from .core import contract_tensor_like
from .hold_state import (  # re-export: canonical definitions live in hold_state.py
    HOLD_CHANNELS,
    active_hold_channel,
    hold_state_summary,
    hold_state_to_float,
    make_empty_hold_state,
    select_hold_channel_value,
    update_hold_state_channels,
)

__all__ = [
    "BootstrapExternalSeedAssemblyResult",
    "BootstrapFinalExternalValueAssemblyResult",
    "BootstrapTargetExecutionResult",
    "HOLD_CHANNELS",
    "active_hold_channel",
    "compute_bootstrap_external_seed_assembly_contract",
    "compute_bootstrap_trigger_entry_contract",
    "compute_bootstrap_bonus_consumer_retention_contract",
    "compute_bootstrap_final_external_value_assembly_contract",
    "compute_bootstrap_final_external_value_quality_contract",
    "compute_bootstrap_midlate_external_value_ratio_cap_contract",
    "compute_bootstrap_prehold_target_cap_contract",
    "compute_bootstrap_target_execution_contract",
    "compute_bootstrap_bonus_source_seed_quality_contract",
    "compute_bootstrap_bonus_source_hold_persistence_contract",
    "compute_bootstrap_bonus_source_transition_bridge_contract",
    "compute_bootstrap_bonus_source_value_replacement_contract",
    "compute_bootstrap_bonus_terminal_truth_source_contract",
    "compute_post_transition_certified_retention_floor_contract",
    "hold_state_summary",
    "hold_state_to_float",
    "make_empty_hold_state",
    "select_hold_channel_value",
    "update_hold_state_channels",
    "RetentionCaptureDecision",
]



@dataclass(frozen=True)
class BootstrapExternalSeedAssemblyResult:
    external_coverage_view: Tensor
    external_confidence_view: Tensor
    anchor_valid_mask: Tensor
    dense_surface_gain: Tensor
    dense_seed_support: Tensor
    dense_anchor_bootstrap_values_next: Tensor
    anchor_bootstrap_values_next: Tensor
    raw_vs_clean_gap: Tensor


@dataclass(frozen=True)
class BootstrapFinalExternalValueAssemblyResult:
    source_consumer_truth: Tensor
    source_consumer_alignment: Tensor
    source_consumer_window_activation: Tensor
    source_consumer_retention_gate: Tensor
    bonus_value_consumer_retained: Tensor
    bonus_value_consumer_delta_abs: Tensor
    regime_quality_gate: Tensor
    final_value_behavior_health: Tensor
    prehold_value_failfast_gate: Tensor
    floor_value_injection_gate: Tensor
    floor_value_clamped: Tensor
    floor_value_delta_abs: Tensor
    final_value_injection_gate: Tensor
    bonus_value_quality_clamped: Tensor
    bonus_value_quality_delta_abs: Tensor


@dataclass(frozen=True)
class BootstrapTargetExecutionResult:
    external_authority_view: Tensor
    floor_authority: Tensor
    bonus_authority: Tensor
    external_value_prefinal: Tensor
    midlate_value_ratio_cap_activation: Tensor
    midlate_external_value_max_ratio: Tensor
    external_value_final: Tensor
    external_value_midlate_ratio_cap_delta_abs: Tensor
    internal_pessimism_gate: Tensor
    internal_value_view: Tensor
    internal_value_relief_delta_abs: Tensor
    mixed_value_prefinal: Tensor
    prehold_target_cap_gate: Tensor
    prehold_safe_cap: Tensor
    mixed_value_final: Tensor
    prehold_target_cap_delta_abs: Tensor




def _value_tensor_like(
    reference: Tensor,
    value: Optional[Union[float, Tensor]],
    default: float,
) -> Tensor:
    """Like contract_tensor_like but WITHOUT [0,1] clamping — for unbounded value caps."""
    if isinstance(value, Tensor):
        if value.shape == reference.shape:
            return value.detach().to(device=reference.device, dtype=reference.dtype)
        if value.numel() == 1:
            return torch.full_like(reference, float(value.detach().item()))
    elif value is not None:
        try:
            return torch.full_like(reference, float(value))
        except (TypeError, ValueError):
            pass
    return torch.full_like(reference, float(default))


# hold_state functions (HOLD_CHANNELS, make_empty_hold_state, etc.) are
# re-exported from .hold_state at the top of this file.


def compute_bootstrap_external_seed_assembly_contract(
    *,
    raw_anchor_bootstrap_values_next: Tensor,
    raw_bootstrap_values_next: Tensor,
    critic_anchor_available_next: Tensor,
    critic_anchor_confidence_next: Tensor,
) -> BootstrapExternalSeedAssemblyResult:
    reference = raw_bootstrap_values_next.detach()
    internal_value = (
        raw_bootstrap_values_next.detach()
        if isinstance(raw_bootstrap_values_next, Tensor)
        and raw_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_value = (
        raw_anchor_bootstrap_values_next.detach()
        if isinstance(raw_anchor_bootstrap_values_next, Tensor)
        and raw_anchor_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_coverage = contract_tensor_like(reference, critic_anchor_available_next, 0.0).clamp(
        0.0, 1.0
    )
    anchor_confidence = contract_tensor_like(
        reference, critic_anchor_confidence_next, 0.0
    ).clamp(0.0, 1.0)
    anchor_valid_mask = (anchor_coverage > 1e-6).to(dtype=reference.dtype).detach()
    available_count = anchor_coverage.sum(dim=1, keepdim=True)
    row_support = (
        available_count / float(max(1, anchor_coverage.shape[1]))
    ).clamp(0.0, 1.0)
    available_gap_sum = ((anchor_value - internal_value) * anchor_coverage).sum(
        dim=1, keepdim=True
    )
    row_gap_mean = torch.where(
        available_count > 0.0,
        available_gap_sum / available_count.clamp(min=1.0),
        torch.zeros_like(available_gap_sum),
    )
    row_confidence = torch.where(
        available_count > 0.0,
        (anchor_confidence * anchor_coverage).sum(dim=1, keepdim=True)
        / available_count.clamp(min=1.0),
        anchor_confidence.mean(dim=1, keepdim=True),
    ).clamp(0.0, 1.0)
    dense_surface_gain = torch.sqrt(
        (row_confidence * row_support).clamp(0.0, 1.0)
    ).detach()
    dense_anchor_bootstrap_values_next = (internal_value + row_gap_mean).detach()
    anchor_bootstrap_values_next = torch.where(
        anchor_coverage > 0.0,
        anchor_value,
        dense_anchor_bootstrap_values_next,
    ).detach()
    raw_vs_clean_gap = (internal_value - anchor_bootstrap_values_next).abs().detach()
    dense_seed_support = dense_surface_gain.expand_as(reference).detach()
    return BootstrapExternalSeedAssemblyResult(
        external_coverage_view=anchor_coverage.detach(),
        external_confidence_view=anchor_confidence.detach(),
        anchor_valid_mask=anchor_valid_mask,
        dense_surface_gain=dense_surface_gain,
        dense_seed_support=dense_seed_support,
        dense_anchor_bootstrap_values_next=dense_anchor_bootstrap_values_next,
        anchor_bootstrap_values_next=anchor_bootstrap_values_next,
        raw_vs_clean_gap=raw_vs_clean_gap,
    )


def compute_bootstrap_bonus_source_transition_bridge_contract(
    *,
    bootstrap_external_value_raw: Tensor,
    raw_anchor_bootstrap_values_next: Tensor,
    raw_bootstrap_values_next: Tensor,
    critic_anchor_available_next: Tensor,
    reference_real_reward_agreement: Union[float, Tensor],
    behavior_policy_task_cert_real_recovery: Tensor,
    behavior_policy_task_cert_real_alarm: Tensor,
    behavior_policy_task_cert_imag_gate: Tensor,
    task_corridor_gate: Tensor,
    critic_contract_task_degradation: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    corridor_semantic_corridor_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    reference = bootstrap_external_value_raw.detach()
    external_value_raw = (
        bootstrap_external_value_raw.detach()
        if isinstance(bootstrap_external_value_raw, Tensor) and bootstrap_external_value_raw.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_value = (
        raw_anchor_bootstrap_values_next.detach()
        if isinstance(raw_anchor_bootstrap_values_next, Tensor) and raw_anchor_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    internal_value = (
        raw_bootstrap_values_next.detach()
        if isinstance(raw_bootstrap_values_next, Tensor) and raw_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_available = contract_tensor_like(reference, critic_anchor_available_next, 0.0).clamp(0.0, 1.0)
    reward_truth = contract_tensor_like(reference, reference_real_reward_agreement, 1.0).clamp(0.0, 1.0)
    real_recovery = contract_tensor_like(reference, behavior_policy_task_cert_real_recovery, 1.0).clamp(0.0, 1.0)
    real_alarm = contract_tensor_like(reference, behavior_policy_task_cert_real_alarm, 0.0).clamp(0.0, 1.0)
    imag_gate = contract_tensor_like(reference, behavior_policy_task_cert_imag_gate, 1.0).clamp(0.0, 1.0)
    task_gate = contract_tensor_like(reference, task_corridor_gate, 1.0).clamp(0.0, 1.0)
    corridor_mask = contract_tensor_like(reference, corridor_semantic_corridor_mask, 0.0).clamp(0.0, 1.0)
    task_degradation = contract_tensor_like(reference, critic_contract_task_degradation, 0.0).clamp(0.0, 1.0)
    late_gate = contract_tensor_like(reference, critic_contract_bootstrap_late_gate, 0.0).clamp(0.0, 1.0)
    reward_semantic_bridge_source = torch.where(anchor_available > 0.0, anchor_value, internal_value)
    source_transition_truth = torch.maximum(reward_truth, real_recovery).clamp(0.0, 1.0)
    source_transition_alignment = torch.maximum(imag_gate, task_gate).clamp(0.0, 1.0)
    source_transition_support = torch.where(
        corridor_mask > 1e-6,
        (1.0 - real_alarm).clamp(0.0, 1.0),
        torch.ones_like(reference),
    ).clamp(0.0, 1.0)
    source_transition_window_activation = (
        ((late_gate - 0.35) / 0.35).clamp(0.0, 1.0)
        * (1.0 - ((late_gate - 0.8) / 0.2).clamp(0.0, 1.0))
    ).clamp(0.0, 1.0)
    source_transition_window_activation = torch.where(late_gate >= 0.8, torch.zeros_like(source_transition_window_activation), source_transition_window_activation)
    source_transition_alarm = torch.maximum(
        (1.0 - source_transition_truth).clamp(0.0, 1.0),
        torch.maximum(
            (1.0 - source_transition_alignment).clamp(0.0, 1.0),
            task_degradation,
        ),
    ).clamp(0.0, 1.0)
    strict_transition_bridge_gate = (
        source_transition_support
        * torch.minimum(source_transition_truth, source_transition_alignment).clamp(0.0, 1.0)
        * (1.0 - source_transition_alarm).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    source_transition_bridge_gate = (
        source_transition_window_activation * strict_transition_bridge_gate
    ).clamp(0.0, 1.0)
    # Stage-3 phase-2: keep transition bridge as a pure observer.
    # It still reports truth/alignment/window/gate diagnostics, but the
    # execution path no longer lets this layer rewrite the carried value.
    _observed_bridged = torch.lerp(
        external_value_raw,
        reward_semantic_bridge_source,
        source_transition_bridge_gate,
    )
    bridged = external_value_raw
    delta_abs = torch.zeros_like(_observed_bridged)
    return (
        source_transition_truth.detach(),
        source_transition_alignment.detach(),
        source_transition_window_activation.detach(),
        source_transition_bridge_gate.detach(),
        bridged.detach(),
        delta_abs.detach(),
    )


def compute_bootstrap_bonus_source_seed_quality_contract(
    *,
    bootstrap_external_value_seed: Tensor,
    raw_anchor_bootstrap_values_next: Tensor,
    raw_bootstrap_values_next: Tensor,
    bootstrap_anchor_valid_mask: Tensor,
    critic_contract_bootstrap_regime_quality_gate: Union[float, Tensor],
    critic_contract_bootstrap_raw_vs_clean_gap: Tensor,
    critic_contract_bootstrap_source_reward_semantic_truth: Tensor,
    critic_contract_bootstrap_source_semantic_alignment: Tensor,
    critic_contract_task_degradation: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    bootstrap_dense_seed_support: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    reference = bootstrap_external_value_seed.detach()
    external_seed = (
        bootstrap_external_value_seed.detach()
        if isinstance(bootstrap_external_value_seed, Tensor)
        and bootstrap_external_value_seed.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_value = (
        raw_anchor_bootstrap_values_next.detach()
        if isinstance(raw_anchor_bootstrap_values_next, Tensor)
        and raw_anchor_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    internal_value = (
        raw_bootstrap_values_next.detach()
        if isinstance(raw_bootstrap_values_next, Tensor)
        and raw_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_valid_mask = contract_tensor_like(reference, bootstrap_anchor_valid_mask, 0.0).clamp(
        0.0, 1.0
    )
    dense_seed_support = contract_tensor_like(reference, bootstrap_dense_seed_support, 0.0).clamp(
        0.0, 1.0
    )
    regime_quality_gate = contract_tensor_like(reference, critic_contract_bootstrap_regime_quality_gate, 1.0).clamp(0.0, 1.0)
    raw_vs_clean_gap = (
        critic_contract_bootstrap_raw_vs_clean_gap.detach().to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if isinstance(critic_contract_bootstrap_raw_vs_clean_gap, Tensor)
        and critic_contract_bootstrap_raw_vs_clean_gap.shape == reference.shape
        else torch.zeros_like(reference)
    ).clamp(min=0.0)
    source_truth = contract_tensor_like(
        reference, critic_contract_bootstrap_source_reward_semantic_truth, 1.0
    ).clamp(0.0, 1.0)
    source_alignment = contract_tensor_like(
        reference, critic_contract_bootstrap_source_semantic_alignment, 1.0
    ).clamp(0.0, 1.0)
    task_degradation = contract_tensor_like(reference, critic_contract_task_degradation, 0.0).clamp(0.0, 1.0)
    late_gate = contract_tensor_like(reference, critic_contract_bootstrap_late_gate, 0.0).clamp(0.0, 1.0)
    source_quality = torch.minimum(source_truth, source_alignment).clamp(0.0, 1.0)
    raw_gap_health = (6.0 / (6.0 + raw_vs_clean_gap)).clamp(0.0, 1.0)
    task_health = (1.0 - task_degradation).clamp(0.0, 1.0)
    dense_seed_quality_gate = (
        regime_quality_gate
        * torch.sqrt(raw_gap_health)
        * torch.sqrt(source_quality)
        * torch.sqrt(task_health)
    ).clamp(0.0, 1.0)
    dense_healthy_override = (
        (anchor_valid_mask <= 1e-6)
        & (source_quality > 0.80)
        & (raw_gap_health > 0.70)
        & (task_health > 0.75)
    ).to(dtype=reference.dtype)
    dense_seed_quality_gate = torch.where(
        dense_healthy_override > 1e-6,
        torch.ones_like(reference),
        dense_seed_quality_gate,
    ).clamp(0.0, 1.0)
    prelate_seed_quality_gate = torch.where(
        anchor_valid_mask > 1e-6,
        torch.ones_like(reference),
        dense_seed_quality_gate,
    ).clamp(0.0, 1.0)
    late_window_activation = ((late_gate - 0.8) / 0.2).clamp(0.0, 1.0)
    source_seed_quality_gate = torch.lerp(
        prelate_seed_quality_gate,
        torch.ones_like(reference),
        late_window_activation,
    ).clamp(0.0, 1.0)
    fallback_seed = torch.where(anchor_valid_mask > 1e-6, anchor_value, internal_value)
    seeded_value = torch.lerp(
        fallback_seed,
        external_seed,
        source_seed_quality_gate,
    ).detach()
    delta_abs = (seeded_value - external_seed).abs().detach()
    return (
        raw_gap_health.detach(),
        source_quality.detach(),
        source_seed_quality_gate.detach(),
        seeded_value.detach(),
        delta_abs.detach(),
    )


def compute_bootstrap_bonus_terminal_truth_source_contract(
    *,
    bootstrap_external_value_raw: Tensor,
    raw_anchor_bootstrap_values_next: Tensor,
    raw_bootstrap_values_next: Tensor,
    anchor_bootstrap_values_next: Tensor,
    critic_anchor_available_next: Tensor,
    reference_real_reward_agreement: Union[float, Tensor],
    behavior_policy_task_cert_real_recovery: Tensor,
    behavior_policy_task_cert_real_alarm: Tensor,
    behavior_policy_task_cert_imag_gate: Tensor,
    task_corridor_gate: Tensor,
    critic_contract_task_degradation: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    critic_contract_bootstrap_raw_vs_clean_gap: Tensor,
    corridor_semantic_corridor_mask: Tensor,
    critic_contract_bootstrap_bonus_hold_state: Union[float, Tensor] = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    reference = bootstrap_external_value_raw.detach()
    external_value_raw = (
        bootstrap_external_value_raw.detach()
        if isinstance(bootstrap_external_value_raw, Tensor) and bootstrap_external_value_raw.shape == reference.shape
        else torch.zeros_like(reference)
    )
    raw_anchor_value = (
        raw_anchor_bootstrap_values_next.detach()
        if isinstance(raw_anchor_bootstrap_values_next, Tensor) and raw_anchor_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    raw_internal_value = (
        raw_bootstrap_values_next.detach()
        if isinstance(raw_bootstrap_values_next, Tensor) and raw_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    clean_anchor_value = (
        anchor_bootstrap_values_next.detach()
        if isinstance(anchor_bootstrap_values_next, Tensor) and anchor_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_available = contract_tensor_like(reference, critic_anchor_available_next, 0.0).clamp(0.0, 1.0)
    reward_truth = contract_tensor_like(reference, reference_real_reward_agreement, 1.0).clamp(0.0, 1.0)
    real_recovery = contract_tensor_like(reference, behavior_policy_task_cert_real_recovery, 1.0).clamp(0.0, 1.0)
    real_alarm = contract_tensor_like(reference, behavior_policy_task_cert_real_alarm, 0.0).clamp(0.0, 1.0)
    imag_gate = contract_tensor_like(reference, behavior_policy_task_cert_imag_gate, 1.0).clamp(0.0, 1.0)
    task_gate = contract_tensor_like(reference, task_corridor_gate, 1.0).clamp(0.0, 1.0)
    corridor_mask = contract_tensor_like(reference, corridor_semantic_corridor_mask, 0.0).clamp(0.0, 1.0)
    raw_vs_clean_gap = (
        critic_contract_bootstrap_raw_vs_clean_gap.detach().to(device=reference.device, dtype=reference.dtype)
        if isinstance(critic_contract_bootstrap_raw_vs_clean_gap, Tensor)
        and critic_contract_bootstrap_raw_vs_clean_gap.shape == reference.shape
        else torch.zeros_like(reference)
    ).clamp(min=0.0)
    task_degradation = contract_tensor_like(reference, critic_contract_task_degradation, 0.0).clamp(0.0, 1.0)
    late_gate = contract_tensor_like(reference, critic_contract_bootstrap_late_gate, 0.0).clamp(0.0, 1.0)
    bonus_hold_state = contract_tensor_like(reference, critic_contract_bootstrap_bonus_hold_state, 0.0).clamp(0.0, 1.0)
    source_terminal_truth = torch.maximum(reward_truth, real_recovery).clamp(0.0, 1.0)
    source_terminal_alignment = torch.maximum(imag_gate, task_gate).clamp(0.0, 1.0)
    source_terminal_support = torch.where(
        corridor_mask > 1e-6,
        (1.0 - real_alarm).clamp(0.0, 1.0),
        torch.ones_like(reference),
    ).clamp(0.0, 1.0)
    base_activation = ((late_gate - 0.8) / 0.2).clamp(0.0, 1.0)
    hold_activation = ((late_gate - 0.55) / 0.25).clamp(0.0, 1.0)
    activation_bridge = (
        ((late_gate - 0.78) / 0.02).clamp(0.0, 1.0)
        * (1.0 - base_activation).clamp(0.0, 1.0)
        * hold_activation
        * bonus_hold_state
    ).clamp(0.0, 1.0)
    window_activation = torch.maximum(base_activation, activation_bridge).clamp(0.0, 1.0)
    relative_gap = (raw_vs_clean_gap / (clean_anchor_value.abs() + raw_internal_value.abs() + 1e-6)).clamp(0.0, 1.0)
    gap_guard = (1.0 - relative_gap).clamp(0.0, 1.0)
    strict_source_health = (
        source_terminal_support
        * torch.minimum(source_terminal_truth, source_terminal_alignment).clamp(0.0, 1.0)
        * (1.0 - task_degradation).clamp(0.0, 1.0)
        * gap_guard
    ).clamp(0.0, 1.0)
    strict_anchor_preference = (
        anchor_available
        * strict_source_health
    ).clamp(0.0, 1.0)
    anchor_preference = (
        (1.0 - window_activation) * anchor_available + window_activation * strict_anchor_preference
    ).clamp(0.0, 1.0)
    strict_source = torch.lerp(raw_internal_value, raw_anchor_value, anchor_preference.clamp(0.0, 1.0))
    strict_source_mix = (window_activation * strict_source_health).clamp(0.0, 1.0)
    terminal_truth = torch.lerp(external_value_raw, strict_source, strict_source_mix)
    delta_abs = (terminal_truth - external_value_raw).abs()
    return (
        source_terminal_truth.detach(),
        source_terminal_alignment.detach(),
        window_activation.detach(),
        activation_bridge.detach(),
        relative_gap.detach(),
        anchor_preference.detach(),
        terminal_truth.detach(),
        delta_abs.detach(),
    )


def compute_bootstrap_final_external_value_quality_contract(
    *,
    bootstrap_external_bonus_value_consumer_retained: Tensor,
    anchor_bootstrap_values_next: Tensor,
    raw_bootstrap_values_next: Tensor,
    bootstrap_anchor_valid_mask: Tensor,
    critic_contract_bootstrap_regime_quality_gate: Union[float, Tensor],
    critic_contract_bootstrap_raw_vs_clean_gap: Tensor,
    critic_contract_bootstrap_source_reward_semantic_truth: Tensor,
    critic_contract_bootstrap_source_semantic_alignment: Tensor,
    actor_contract_task_mismatch: Union[float, Tensor],
    actor_corridor_semantic_inflation_excess: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_source_hold_window_activation: Union[float, Tensor] = 0.0,
    actor_high_value_but_low_task_fraction: Union[float, Tensor] = 0.0,
    actor_task_geom_corridor_disagreement: Union[float, Tensor] = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    reference = bootstrap_external_bonus_value_consumer_retained.detach()
    retained_value = (
        bootstrap_external_bonus_value_consumer_retained.detach()
        if isinstance(bootstrap_external_bonus_value_consumer_retained, Tensor)
        and bootstrap_external_bonus_value_consumer_retained.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_value = (
        anchor_bootstrap_values_next.detach()
        if isinstance(anchor_bootstrap_values_next, Tensor)
        and anchor_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    internal_value = (
        raw_bootstrap_values_next.detach()
        if isinstance(raw_bootstrap_values_next, Tensor)
        and raw_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_valid_mask = contract_tensor_like(reference, bootstrap_anchor_valid_mask, 0.0).clamp(
        0.0, 1.0
    )
    regime_quality_gate = contract_tensor_like(reference, critic_contract_bootstrap_regime_quality_gate, 1.0).clamp(0.0, 1.0)
    raw_vs_clean_gap = (
        critic_contract_bootstrap_raw_vs_clean_gap.detach().to(
            device=reference.device, dtype=reference.dtype
        )
        if isinstance(critic_contract_bootstrap_raw_vs_clean_gap, Tensor)
        and critic_contract_bootstrap_raw_vs_clean_gap.shape == reference.shape
        else torch.zeros_like(reference)
    ).clamp(min=0.0)
    source_truth = contract_tensor_like(
        reference, critic_contract_bootstrap_source_reward_semantic_truth, 1.0
    ).clamp(0.0, 1.0)
    source_alignment = contract_tensor_like(
        reference, critic_contract_bootstrap_source_semantic_alignment, 1.0
    ).clamp(0.0, 1.0)
    task_mismatch = contract_tensor_like(reference, actor_contract_task_mismatch, 0.0).clamp(min=0.0)
    inflation_excess = contract_tensor_like(reference, actor_corridor_semantic_inflation_excess, 0.0).clamp(min=0.0)
    late_gate = contract_tensor_like(reference, critic_contract_bootstrap_late_gate, 0.0).clamp(0.0, 1.0)
    hold_window_activation = contract_tensor_like(reference, critic_contract_bootstrap_source_hold_window_activation, 0.0).clamp(0.0, 1.0)
    high_value_but_low_task_fraction = contract_tensor_like(reference, actor_high_value_but_low_task_fraction, 0.0).clamp(0.0, 1.0)
    task_geom_corridor_disagreement = contract_tensor_like(reference, actor_task_geom_corridor_disagreement, 0.0).clamp(0.0, 1.0)
    source_quality = torch.minimum(source_truth, source_alignment).clamp(0.0, 1.0)
    raw_gap_health = (6.0 / (6.0 + raw_vs_clean_gap)).clamp(0.0, 1.0)
    task_health = (1.0 - task_mismatch / 0.30).clamp(0.0, 1.0)
    inflation_health = (1.0 - inflation_excess / 12.0).clamp(0.0, 1.0)
    behavior_health = torch.minimum(task_health, inflation_health).clamp(0.0, 1.0)
    dense_gate = (
        regime_quality_gate
        * torch.sqrt(raw_gap_health)
        * torch.sqrt(source_quality)
        * behavior_health
    ).clamp(0.0, 1.0)
    dense_floor_gate = (
        regime_quality_gate
        * raw_gap_health
        * source_quality
        * behavior_health
    ).clamp(0.0, 1.0)
    anchor_gate = torch.maximum(
        dense_gate,
        (
            0.25
            + 0.75
            * torch.minimum(
                torch.maximum(source_quality, regime_quality_gate),
                behavior_health,
            )
        ).clamp(0.0, 1.0),
    ).clamp(0.0, 1.0)
    anchor_floor_health = torch.minimum(
        torch.maximum(source_quality, regime_quality_gate),
        torch.minimum(behavior_health, raw_gap_health),
    ).clamp(0.0, 1.0)
    anchor_floor_gate = torch.maximum(
        dense_floor_gate,
        (0.10 + 0.90 * anchor_floor_health).clamp(0.0, 1.0),
    ).clamp(0.0, 1.0)
    healthy_anchor_override = (
        (anchor_valid_mask > 1e-6)
        & (source_alignment > 0.8)
        & (behavior_health > 0.55)
        & (raw_gap_health > 0.5)
    ).to(dtype=reference.dtype)
    external_floor_value_gate = torch.where(
        anchor_valid_mask > 1e-6,
        anchor_floor_gate,
        dense_floor_gate,
    ).clamp(0.0, 1.0)
    external_floor_value_gate = torch.where(
        healthy_anchor_override > 1e-6,
        torch.maximum(external_floor_value_gate, torch.full_like(reference, 0.90)),
        external_floor_value_gate,
    ).clamp(0.0, 1.0)
    final_value_injection_gate = torch.where(
        anchor_valid_mask > 1e-6,
        anchor_gate,
        dense_gate,
    ).clamp(0.0, 1.0)
    final_value_injection_gate = torch.where(
        healthy_anchor_override > 1e-6,
        torch.maximum(final_value_injection_gate, torch.full_like(reference, 0.85)),
        final_value_injection_gate,
    ).clamp(0.0, 1.0)
    prehold_deadzone = (
        (late_gate > 1e-6) & (hold_window_activation < 1e-6)
    ).to(dtype=reference.dtype)
    prehold_high_value_alarm = (
        (high_value_but_low_task_fraction - 0.10) / 0.15
    ).clamp(0.0, 1.0)
    prehold_task_disagreement_alarm = (
        (task_geom_corridor_disagreement - 0.48) / 0.08
    ).clamp(0.0, 1.0)
    prehold_actor_mismatch_alarm = torch.maximum(
        prehold_high_value_alarm,
        prehold_task_disagreement_alarm,
    ).clamp(0.0, 1.0)
    prehold_source_alarm = torch.maximum(
        (1.0 - regime_quality_gate).clamp(0.0, 1.0),
        torch.maximum(
            (1.0 - source_quality).clamp(0.0, 1.0),
            (1.0 - raw_gap_health * behavior_health).clamp(0.0, 1.0),
        ),
    ).clamp(0.0, 1.0)
    external_value_inflation = (
        (
            torch.maximum(retained_value, anchor_value) - internal_value
        ).clamp(min=0.0)
        / (internal_value.abs() + 1.0)
    ).clamp(0.0, 1.0)
    prehold_inflation_alarm = (
        ((external_value_inflation - 0.35) / 0.35).clamp(0.0, 1.0)
        * prehold_source_alarm
    ).clamp(0.0, 1.0)
    prehold_value_failfast_pressure = torch.maximum(
        prehold_actor_mismatch_alarm,
        prehold_inflation_alarm,
    ).clamp(0.0, 1.0)
    prehold_value_failfast_gate = (
        1.0
        - prehold_deadzone
        * torch.sqrt(prehold_source_alarm)
        * prehold_value_failfast_pressure
    ).clamp(0.05, 1.0)
    healthy_prehold_override = (
        (prehold_deadzone > 1e-6)
        & (regime_quality_gate > 0.50)
        & (source_quality > 0.70)
        & (raw_gap_health > 0.50)
        & (behavior_health > 0.55)
        & (prehold_actor_mismatch_alarm < 0.10)
    ).to(dtype=reference.dtype)
    prehold_value_failfast_gate = torch.where(
        healthy_prehold_override > 1e-6,
        torch.maximum(
            prehold_value_failfast_gate,
            torch.full_like(reference, 0.95),
        ),
        prehold_value_failfast_gate,
    ).clamp(0.05, 1.0)
    floor_value_quality_clamped = torch.lerp(
        internal_value,
        anchor_value,
        external_floor_value_gate,
    ).detach()
    # Once hold persistence is active, the carried consumer value is already the
    # best late-stage evidence we have. Do not let the floor path snap all the
    # way back to inflated internal values.
    retained_floor_cap_active = (
        (hold_window_activation > 1e-6)
        & (late_gate > 0.9)
        & (floor_value_quality_clamped > (retained_value + 1e-6))
    ).to(dtype=reference.dtype)
    retained_floor_cap = (
        retained_value
        + 0.25 * (internal_value - retained_value).clamp(min=0.0)
    ).detach()
    floor_value_quality_clamped = torch.where(
        retained_floor_cap_active > 1e-6,
        torch.minimum(floor_value_quality_clamped, retained_floor_cap),
        floor_value_quality_clamped,
    ).detach()
    floor_delta_abs = (floor_value_quality_clamped - anchor_value).abs().detach()
    quality_clamped_value = torch.lerp(
        floor_value_quality_clamped,
        retained_value,
        final_value_injection_gate,
    ).detach()
    delta_abs = (quality_clamped_value - retained_value).abs().detach()
    return (
        regime_quality_gate.detach(),
        behavior_health.detach(),
        prehold_value_failfast_gate.detach(),
        external_floor_value_gate.detach(),
        floor_value_quality_clamped.detach(),
        floor_delta_abs.detach(),
        final_value_injection_gate.detach(),
        quality_clamped_value.detach(),
        delta_abs.detach(),
    )


def compute_bootstrap_final_external_value_assembly_contract(
    *,
    bootstrap_external_bonus_value_hold_persisted: Tensor,
    bootstrap_external_bonus_terminal_truth_source: Tensor,
    bootstrap_external_bonus_value_source_replaced: Tensor,
    reference_real_reward_agreement: Union[float, Tensor],
    behavior_policy_task_cert_real_recovery: Tensor,
    behavior_policy_task_cert_real_alarm: Tensor,
    behavior_policy_task_cert_imag_gate: Tensor,
    task_corridor_gate: Tensor,
    critic_contract_task_degradation: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    critic_contract_bootstrap_bonus_hold_state: Union[float, Tensor],
    critic_contract_bootstrap_source_hold_persistence_gate: Tensor,
    corridor_semantic_corridor_mask: Tensor,
    anchor_bootstrap_values_next: Tensor,
    raw_bootstrap_values_next: Tensor,
    bootstrap_anchor_valid_mask: Tensor,
    critic_contract_bootstrap_regime_quality_gate: Union[float, Tensor],
    critic_contract_bootstrap_raw_vs_clean_gap: Tensor,
    critic_contract_bootstrap_source_reward_semantic_truth: Tensor,
    critic_contract_bootstrap_source_semantic_alignment: Tensor,
    actor_contract_task_mismatch: Union[float, Tensor],
    actor_corridor_semantic_inflation_excess: Union[float, Tensor],
    critic_contract_bootstrap_source_hold_window_activation: Union[float, Tensor] = 0.0,
    actor_high_value_but_low_task_fraction: Union[float, Tensor] = 0.0,
    actor_task_geom_corridor_disagreement: Union[float, Tensor] = 0.0,
) -> BootstrapFinalExternalValueAssemblyResult:
    (
        source_consumer_truth,
        source_consumer_alignment,
        source_consumer_window_activation,
        source_consumer_retention_gate,
        bonus_value_consumer_retained,
        bonus_value_consumer_delta_abs,
    ) = compute_bootstrap_bonus_consumer_retention_contract(
        bootstrap_external_bonus_value_hold_persisted=bootstrap_external_bonus_value_hold_persisted,
        bootstrap_external_bonus_terminal_truth_source=bootstrap_external_bonus_terminal_truth_source,
        bootstrap_external_bonus_value_source_replaced=bootstrap_external_bonus_value_source_replaced,
        reference_real_reward_agreement=reference_real_reward_agreement,
        behavior_policy_task_cert_real_recovery=behavior_policy_task_cert_real_recovery,
        behavior_policy_task_cert_real_alarm=behavior_policy_task_cert_real_alarm,
        behavior_policy_task_cert_imag_gate=behavior_policy_task_cert_imag_gate,
        task_corridor_gate=task_corridor_gate,
        critic_contract_task_degradation=critic_contract_task_degradation,
        critic_contract_bootstrap_late_gate=critic_contract_bootstrap_late_gate,
        critic_contract_bootstrap_bonus_hold_state=critic_contract_bootstrap_bonus_hold_state,
        critic_contract_bootstrap_source_hold_persistence_gate=critic_contract_bootstrap_source_hold_persistence_gate,
        corridor_semantic_corridor_mask=corridor_semantic_corridor_mask,
    )
    # Keep the consumer-retention path observable for diagnostics, but do not
    # let it become a second execution pass inside the final quality assembly.
    source_consumer_window_activation = torch.zeros_like(
        source_consumer_window_activation
    )
    source_consumer_retention_gate = torch.zeros_like(
        source_consumer_retention_gate
    )
    bonus_value_consumer_retained = (
        bootstrap_external_bonus_value_hold_persisted.detach()
    )
    bonus_value_consumer_delta_abs = torch.zeros_like(
        bonus_value_consumer_retained
    )
    (
        regime_quality_gate,
        final_value_behavior_health,
        prehold_value_failfast_gate,
        floor_value_injection_gate,
        floor_value_clamped,
        floor_value_delta_abs,
        final_value_injection_gate,
        bonus_value_quality_clamped,
        bonus_value_quality_delta_abs,
    ) = compute_bootstrap_final_external_value_quality_contract(
        bootstrap_external_bonus_value_consumer_retained=bonus_value_consumer_retained,
        anchor_bootstrap_values_next=anchor_bootstrap_values_next,
        raw_bootstrap_values_next=raw_bootstrap_values_next,
        bootstrap_anchor_valid_mask=bootstrap_anchor_valid_mask,
        critic_contract_bootstrap_regime_quality_gate=critic_contract_bootstrap_regime_quality_gate,
        critic_contract_bootstrap_raw_vs_clean_gap=critic_contract_bootstrap_raw_vs_clean_gap,
        critic_contract_bootstrap_source_reward_semantic_truth=critic_contract_bootstrap_source_reward_semantic_truth,
        critic_contract_bootstrap_source_semantic_alignment=critic_contract_bootstrap_source_semantic_alignment,
        actor_contract_task_mismatch=actor_contract_task_mismatch,
        actor_corridor_semantic_inflation_excess=actor_corridor_semantic_inflation_excess,
        critic_contract_bootstrap_late_gate=critic_contract_bootstrap_late_gate,
        critic_contract_bootstrap_source_hold_window_activation=critic_contract_bootstrap_source_hold_window_activation,
        actor_high_value_but_low_task_fraction=actor_high_value_but_low_task_fraction,
        actor_task_geom_corridor_disagreement=actor_task_geom_corridor_disagreement,
    )
    return BootstrapFinalExternalValueAssemblyResult(
        source_consumer_truth=source_consumer_truth,
        source_consumer_alignment=source_consumer_alignment,
        source_consumer_window_activation=source_consumer_window_activation,
        source_consumer_retention_gate=source_consumer_retention_gate,
        bonus_value_consumer_retained=bonus_value_consumer_retained,
        bonus_value_consumer_delta_abs=bonus_value_consumer_delta_abs,
        regime_quality_gate=regime_quality_gate,
        final_value_behavior_health=final_value_behavior_health,
        prehold_value_failfast_gate=prehold_value_failfast_gate,
        floor_value_injection_gate=floor_value_injection_gate,
        floor_value_clamped=floor_value_clamped,
        floor_value_delta_abs=floor_value_delta_abs,
        final_value_injection_gate=final_value_injection_gate,
        bonus_value_quality_clamped=bonus_value_quality_clamped,
        bonus_value_quality_delta_abs=bonus_value_quality_delta_abs,
    )


def compute_bootstrap_internal_value_relief_contract(
    *,
    raw_bootstrap_values_next: Tensor,
    bootstrap_external_value_final: Tensor,
    critic_contract_bootstrap_requested_floor: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    critic_contract_bootstrap_floor_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_final_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_source_hold_window_activation: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_regime_quality_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_real_value_cap: Optional[Union[float, Tensor]] = None,
    actor_high_value_but_low_task_fraction: Union[float, Tensor] = 0.0,
    actor_task_geom_corridor_disagreement: Union[float, Tensor] = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    reference = raw_bootstrap_values_next.detach()
    internal_value = (
        raw_bootstrap_values_next.detach()
        if isinstance(raw_bootstrap_values_next, Tensor)
        and raw_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    external_value = (
        bootstrap_external_value_final.detach()
        if isinstance(bootstrap_external_value_final, Tensor)
        and bootstrap_external_value_final.shape == reference.shape
        else internal_value
    )
    requested_floor = contract_tensor_like(
        reference,
        critic_contract_bootstrap_requested_floor,
        0.0,
    ).clamp(0.0, 1.0)
    late_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_late_gate,
        0.0,
    ).clamp(0.0, 1.0)
    floor_value_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_floor_value_injection_gate,
        1.0,
    ).clamp(0.0, 1.0)
    final_value_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_final_value_injection_gate,
        1.0,
    ).clamp(0.0, 1.0)
    hold_window_activation = contract_tensor_like(
        reference,
        critic_contract_bootstrap_source_hold_window_activation,
        0.0,
    ).clamp(0.0, 1.0)
    regime_quality_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_regime_quality_gate,
        1.0,
    ).clamp(0.0, 1.0)
    if critic_contract_bootstrap_real_value_cap is None:
        real_value_cap = internal_value
    else:
        real_value_cap = _value_tensor_like(
            reference,
            critic_contract_bootstrap_real_value_cap,
            0.0,
        )
    real_value_cap = torch.maximum(real_value_cap, torch.zeros_like(reference)).detach()
    high_value_but_low_task_fraction = contract_tensor_like(
        reference,
        actor_high_value_but_low_task_fraction,
        0.0,
    ).clamp(0.0, 1.0)
    task_geom_corridor_disagreement = contract_tensor_like(
        reference,
        actor_task_geom_corridor_disagreement,
        0.0,
    ).clamp(0.0, 1.0)

    value_gate = torch.minimum(floor_value_gate, final_value_gate).clamp(0.0, 1.0)
    prehold_deadzone = (
        (late_gate > 1e-6) & (hold_window_activation < 1e-6)
    ).to(dtype=reference.dtype)
    midlate_activation = (
        torch.sqrt(late_gate)
        * (1.0 - ((late_gate - 0.9) / 0.1).clamp(0.0, 1.0))
        * (1.0 - hold_window_activation)
    ).clamp(0.0, 1.0)
    quality_deficit = (
        1.0 - torch.sqrt((value_gate * regime_quality_gate).clamp(0.0, 1.0))
    ).clamp(0.0, 1.0)
    prehold_high_value_alarm = (
        (high_value_but_low_task_fraction - 0.10) / 0.15
    ).clamp(0.0, 1.0)
    prehold_task_disagreement_alarm = (
        (task_geom_corridor_disagreement - 0.48) / 0.08
    ).clamp(0.0, 1.0)
    prehold_actor_mismatch_alarm = torch.maximum(
        prehold_high_value_alarm,
        prehold_task_disagreement_alarm,
    ).clamp(0.0, 1.0)
    relief_target = torch.minimum(external_value, real_value_cap).detach()
    grounding_gap = (
        (internal_value - relief_target).clamp(min=0.0)
        / (relief_target.abs() + 1.0)
    ).clamp(0.0, 1.0)
    relief_pressure = torch.maximum(
        torch.maximum((1.0 - torch.sqrt(value_gate)).clamp(0.0, 1.0), grounding_gap),
        prehold_actor_mismatch_alarm,
    ).clamp(0.0, 1.0)
    requested_floor_support = (0.35 + 0.65 * requested_floor).clamp(0.0, 1.0)
    internal_value_relief_gate = (
        prehold_deadzone
        * midlate_activation
        * requested_floor_support
        * torch.sqrt(quality_deficit)
        * relief_pressure
    ).clamp(0.0, 1.0)
    healthy_override = (
        (midlate_activation > 1e-6)
        & (value_gate > 0.75)
        & (regime_quality_gate > 0.70)
        & (requested_floor > 0.5)
        & (prehold_actor_mismatch_alarm < 0.10)
    ).to(dtype=reference.dtype)
    internal_value_relief_gate = torch.where(
        healthy_override > 1e-6,
        torch.minimum(
            internal_value_relief_gate,
            torch.full_like(reference, 0.05),
        ),
        internal_value_relief_gate,
    ).clamp(0.0, 1.0)
    internal_value_relieved = torch.lerp(
        internal_value,
        relief_target,
        internal_value_relief_gate,
    ).detach()
    delta_abs = (internal_value - internal_value_relieved).abs().detach()
    return (
        internal_value_relief_gate.detach(),
        internal_value_relieved.detach(),
        delta_abs.detach(),
    )


def compute_bootstrap_prehold_target_cap_contract(
    *,
    mixed_bootstrap_values_next: Tensor,
    bootstrap_internal_value_view: Tensor,
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    critic_contract_bootstrap_source_hold_window_activation: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_post_transition_window_activation: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_floor_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_final_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_regime_quality_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_internal_pessimism_gate: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_real_value_cap: Optional[Union[float, Tensor]] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    reference = mixed_bootstrap_values_next.detach()
    mixed_target = (
        mixed_bootstrap_values_next.detach()
        if isinstance(mixed_bootstrap_values_next, Tensor)
        and mixed_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    internal_value = (
        bootstrap_internal_value_view.detach()
        if isinstance(bootstrap_internal_value_view, Tensor)
        and bootstrap_internal_value_view.shape == reference.shape
        else mixed_target
    )
    late_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_late_gate,
        0.0,
    ).clamp(0.0, 1.0)
    hold_window_activation = contract_tensor_like(
        reference,
        critic_contract_bootstrap_source_hold_window_activation,
        0.0,
    ).clamp(0.0, 1.0)
    post_transition_window_activation = contract_tensor_like(
        reference,
        critic_contract_bootstrap_post_transition_window_activation,
        0.0,
    ).clamp(0.0, 1.0)
    floor_value_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_floor_value_injection_gate,
        1.0,
    ).clamp(0.0, 1.0)
    final_value_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_final_value_injection_gate,
        1.0,
    ).clamp(0.0, 1.0)
    regime_quality_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_regime_quality_gate,
        1.0,
    ).clamp(0.0, 1.0)
    internal_pessimism_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_internal_pessimism_gate,
        0.0,
    ).clamp(0.0, 1.0)
    if critic_contract_bootstrap_real_value_cap is None:
        real_value_cap = internal_value
    else:
        real_value_cap = _value_tensor_like(
            reference,
            critic_contract_bootstrap_real_value_cap,
            0.0,
        )
    real_value_cap = torch.maximum(real_value_cap, torch.zeros_like(reference)).detach()

    value_gate = torch.minimum(floor_value_gate, final_value_gate).clamp(0.0, 1.0)
    prehold_window = (
        (late_gate > 1e-6)
        & (hold_window_activation < 1e-6)
        & (post_transition_window_activation < 1e-6)
    ).to(dtype=reference.dtype)
    quality_deficit = (
        1.0 - torch.sqrt((value_gate * regime_quality_gate).clamp(0.0, 1.0))
    ).clamp(0.0, 1.0)
    safe_upper = torch.maximum(internal_value, real_value_cap).detach()
    cap_gate = (
        prehold_window
        * torch.sqrt(internal_pessimism_gate.clamp(0.0, 1.0))
        * torch.sqrt(quality_deficit)
    ).clamp(0.0, 1.0)
    healthy_override = (
        (value_gate > 0.75)
        & (regime_quality_gate > 0.70)
        & (internal_pessimism_gate < 0.05)
    ).to(dtype=reference.dtype)
    cap_gate = torch.where(
        healthy_override > 1e-6,
        torch.zeros_like(cap_gate),
        cap_gate,
    ).clamp(0.0, 1.0)
    above_safe = (mixed_target - safe_upper).clamp(min=0.0)
    capped_target = (mixed_target - cap_gate * above_safe).detach()
    delta_abs = (mixed_target - capped_target).abs().detach()
    return (
        cap_gate.detach(),
        safe_upper.detach(),
        capped_target.detach(),
        delta_abs.detach(),
    )


def compute_bootstrap_midlate_external_value_ratio_cap_contract(
    *,
    bootstrap_external_value_final: Tensor,
    raw_bootstrap_values_next: Tensor,
    critic_contract_bootstrap_floor_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_final_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_midlate_floor_unwind_activation: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_source_hold_window_activation: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_regime_quality_gate: Union[float, Tensor] = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    reference = bootstrap_external_value_final.detach()
    external_value = (
        bootstrap_external_value_final.detach()
        if isinstance(bootstrap_external_value_final, Tensor)
        and bootstrap_external_value_final.shape == reference.shape
        else torch.zeros_like(reference)
    )
    internal_value = (
        raw_bootstrap_values_next.detach()
        if isinstance(raw_bootstrap_values_next, Tensor)
        and raw_bootstrap_values_next.shape == reference.shape
        else external_value
    )
    floor_value_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_floor_value_injection_gate,
        1.0,
    ).clamp(0.0, 1.0)
    final_value_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_final_value_injection_gate,
        1.0,
    ).clamp(0.0, 1.0)
    midlate_floor_unwind_activation = contract_tensor_like(
        reference,
        critic_contract_bootstrap_midlate_floor_unwind_activation,
        0.0,
    ).clamp(0.0, 1.0)
    hold_window_activation = contract_tensor_like(
        reference,
        critic_contract_bootstrap_source_hold_window_activation,
        0.0,
    ).clamp(0.0, 1.0)
    regime_quality_gate = contract_tensor_like(
        reference,
        critic_contract_bootstrap_regime_quality_gate,
        1.0,
    ).clamp(0.0, 1.0)

    value_gate = torch.minimum(floor_value_gate, final_value_gate).clamp(0.0, 1.0)
    onset_activation = (
        midlate_floor_unwind_activation
        * (1.0 - hold_window_activation).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    cap_pressure = (
        onset_activation
        * (1.0 - torch.sqrt(value_gate)).clamp(0.0, 1.0)
        * (1.0 - regime_quality_gate).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    max_ratio = (
        0.45
        + 0.20 * torch.sqrt(value_gate)
        + 0.15 * regime_quality_gate
        + 0.20 * hold_window_activation
    ).clamp(0.45, 1.0)
    allowed_abs = max_ratio * internal_value.abs().clamp(min=1e-6)
    capped_abs = torch.minimum(external_value.abs(), allowed_abs)
    capped_target = torch.sign(external_value) * capped_abs
    capped_value = torch.lerp(
        external_value,
        capped_target,
        cap_pressure,
    ).detach()
    delta_abs = (external_value - capped_value).abs().detach()
    return (
        onset_activation.detach(),
        max_ratio.detach(),
        capped_value.detach(),
        delta_abs.detach(),
    )


def compute_bootstrap_target_execution_contract(
    *,
    bootstrap_external_authority_view: Tensor,
    bootstrap_external_authority_floor: Tensor,
    bootstrap_external_floor_value_clamped: Tensor,
    bootstrap_external_bonus_value_quality_clamped: Tensor,
    raw_bootstrap_values_next: Tensor,
    critic_contract_bootstrap_floor_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_final_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_midlate_floor_unwind_activation: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_source_hold_window_activation: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_regime_quality_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_requested_floor: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_late_gate: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_post_transition_window_activation: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_real_value_cap: Optional[Union[float, Tensor]] = None,
    actor_high_value_but_low_task_fraction: Union[float, Tensor] = 0.0,
    actor_task_geom_corridor_disagreement: Union[float, Tensor] = 0.0,
) -> BootstrapTargetExecutionResult:
    reference = raw_bootstrap_values_next.detach()
    raw_internal_value = (
        raw_bootstrap_values_next.detach()
        if isinstance(raw_bootstrap_values_next, Tensor)
        and raw_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    external_authority_view = _value_tensor_like(
        reference,
        bootstrap_external_authority_view,
        0.0,
    ).clamp(0.0, 1.0)
    floor_authority = _value_tensor_like(
        reference,
        bootstrap_external_authority_floor,
        0.0,
    ).clamp(0.0, 1.0)
    floor_authority = torch.minimum(
        floor_authority,
        external_authority_view,
    ).detach()
    bonus_authority = (
        external_authority_view - floor_authority
    ).clamp(0.0, 1.0).detach()
    floor_value_clamped = (
        bootstrap_external_floor_value_clamped.detach()
        if isinstance(bootstrap_external_floor_value_clamped, Tensor)
        and bootstrap_external_floor_value_clamped.shape == reference.shape
        else raw_internal_value
    )
    bonus_value_quality_clamped = (
        bootstrap_external_bonus_value_quality_clamped.detach()
        if isinstance(bootstrap_external_bonus_value_quality_clamped, Tensor)
        and bootstrap_external_bonus_value_quality_clamped.shape == reference.shape
        else raw_internal_value
    )
    external_value_prefinal = torch.where(
        external_authority_view > 1e-6,
        (
            floor_authority * floor_value_clamped
            + bonus_authority * bonus_value_quality_clamped
        )
        / external_authority_view.clamp(min=1e-6),
        raw_internal_value,
    ).detach()
    (
        midlate_value_ratio_cap_activation,
        midlate_external_value_max_ratio,
        external_value_final,
        external_value_midlate_ratio_cap_delta_abs,
    ) = compute_bootstrap_midlate_external_value_ratio_cap_contract(
        bootstrap_external_value_final=external_value_prefinal,
        raw_bootstrap_values_next=raw_internal_value,
        critic_contract_bootstrap_floor_value_injection_gate=critic_contract_bootstrap_floor_value_injection_gate,
        critic_contract_bootstrap_final_value_injection_gate=critic_contract_bootstrap_final_value_injection_gate,
        critic_contract_bootstrap_midlate_floor_unwind_activation=critic_contract_bootstrap_midlate_floor_unwind_activation,
        critic_contract_bootstrap_source_hold_window_activation=critic_contract_bootstrap_source_hold_window_activation,
        critic_contract_bootstrap_regime_quality_gate=critic_contract_bootstrap_regime_quality_gate,
    )
    (
        internal_pessimism_gate,
        internal_value_view,
        internal_value_relief_delta_abs,
    ) = compute_bootstrap_internal_value_relief_contract(
        raw_bootstrap_values_next=raw_internal_value,
        bootstrap_external_value_final=external_value_final,
        critic_contract_bootstrap_requested_floor=critic_contract_bootstrap_requested_floor,
        critic_contract_bootstrap_late_gate=critic_contract_bootstrap_late_gate,
        critic_contract_bootstrap_floor_value_injection_gate=critic_contract_bootstrap_floor_value_injection_gate,
        critic_contract_bootstrap_final_value_injection_gate=critic_contract_bootstrap_final_value_injection_gate,
        critic_contract_bootstrap_source_hold_window_activation=critic_contract_bootstrap_source_hold_window_activation,
        critic_contract_bootstrap_regime_quality_gate=critic_contract_bootstrap_regime_quality_gate,
        critic_contract_bootstrap_real_value_cap=critic_contract_bootstrap_real_value_cap,
        actor_high_value_but_low_task_fraction=actor_high_value_but_low_task_fraction,
        actor_task_geom_corridor_disagreement=actor_task_geom_corridor_disagreement,
    )
    mixed_value_prefinal = torch.lerp(
        internal_value_view.detach(),
        external_value_final.detach(),
        external_authority_view.detach(),
    ).detach()
    (
        prehold_target_cap_gate,
        prehold_safe_cap,
        mixed_value_final,
        prehold_target_cap_delta_abs,
    ) = compute_bootstrap_prehold_target_cap_contract(
        mixed_bootstrap_values_next=mixed_value_prefinal,
        bootstrap_internal_value_view=internal_value_view.detach(),
        critic_contract_bootstrap_late_gate=critic_contract_bootstrap_late_gate,
        critic_contract_bootstrap_source_hold_window_activation=critic_contract_bootstrap_source_hold_window_activation,
        critic_contract_bootstrap_post_transition_window_activation=critic_contract_bootstrap_post_transition_window_activation,
        critic_contract_bootstrap_floor_value_injection_gate=critic_contract_bootstrap_floor_value_injection_gate,
        critic_contract_bootstrap_final_value_injection_gate=critic_contract_bootstrap_final_value_injection_gate,
        critic_contract_bootstrap_regime_quality_gate=critic_contract_bootstrap_regime_quality_gate,
        critic_contract_bootstrap_internal_pessimism_gate=internal_pessimism_gate.detach(),
        critic_contract_bootstrap_real_value_cap=critic_contract_bootstrap_real_value_cap,
    )
    return BootstrapTargetExecutionResult(
        external_authority_view=external_authority_view.detach(),
        floor_authority=floor_authority.detach(),
        bonus_authority=bonus_authority.detach(),
        external_value_prefinal=external_value_prefinal.detach(),
        midlate_value_ratio_cap_activation=midlate_value_ratio_cap_activation.detach(),
        midlate_external_value_max_ratio=midlate_external_value_max_ratio.detach(),
        external_value_final=external_value_final.detach(),
        external_value_midlate_ratio_cap_delta_abs=external_value_midlate_ratio_cap_delta_abs.detach(),
        internal_pessimism_gate=internal_pessimism_gate.detach(),
        internal_value_view=internal_value_view.detach(),
        internal_value_relief_delta_abs=internal_value_relief_delta_abs.detach(),
        mixed_value_prefinal=mixed_value_prefinal.detach(),
        prehold_target_cap_gate=prehold_target_cap_gate.detach(),
        prehold_safe_cap=prehold_safe_cap.detach(),
        mixed_value_final=mixed_value_final.detach(),
        prehold_target_cap_delta_abs=prehold_target_cap_delta_abs.detach(),
    )


def compute_bootstrap_bonus_source_value_replacement_contract(
    *,
    bootstrap_external_bonus_terminal_truth_source: Optional[Tensor] = None,
    bootstrap_external_value_raw: Optional[Tensor] = None,
    reward_semantic_bonus_source_value: Tensor,
    critic_anchor_available_next: Tensor,
    raw_anchor_bootstrap_values_next: Tensor,
    raw_bootstrap_values_next: Tensor,
    reference_real_reward_agreement: Union[float, Tensor],
    behavior_policy_task_cert_real_recovery: Tensor,
    behavior_policy_task_cert_support_mask: Tensor,
    behavior_policy_task_cert_imag_gate: Tensor,
    task_corridor_gate: Tensor,
    critic_contract_task_degradation: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    corridor_semantic_corridor_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    external_source = bootstrap_external_bonus_terminal_truth_source
    if external_source is None:
        external_source = bootstrap_external_value_raw
    if external_source is None:
        raise ValueError(
            "compute_bootstrap_bonus_source_value_replacement_contract requires "
            "bootstrap_external_bonus_terminal_truth_source or bootstrap_external_value_raw"
        )
    reference = external_source.detach()
    external_value_raw = (
        external_source.detach()
        if isinstance(external_source, Tensor) and external_source.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_value = (
        raw_anchor_bootstrap_values_next.detach()
        if isinstance(raw_anchor_bootstrap_values_next, Tensor) and raw_anchor_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    internal_value = (
        raw_bootstrap_values_next.detach()
        if isinstance(raw_bootstrap_values_next, Tensor) and raw_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_available = contract_tensor_like(reference, critic_anchor_available_next, 0.0) > 0.0
    reward_semantic_value = (
        reward_semantic_bonus_source_value.detach()
        if isinstance(reward_semantic_bonus_source_value, Tensor)
        and reward_semantic_bonus_source_value.shape == reference.shape
        else torch.where(anchor_available, anchor_value, internal_value)
    )
    reward_truth = contract_tensor_like(reference, reference_real_reward_agreement, 1.0).clamp(0.0, 1.0)
    real_recovery = contract_tensor_like(reference, behavior_policy_task_cert_real_recovery, 1.0).clamp(0.0, 1.0)
    support_mask = contract_tensor_like(reference, behavior_policy_task_cert_support_mask, 1.0).clamp(0.0, 1.0)
    imag_gate = contract_tensor_like(reference, behavior_policy_task_cert_imag_gate, 1.0).clamp(0.0, 1.0)
    task_gate = contract_tensor_like(reference, task_corridor_gate, 1.0).clamp(0.0, 1.0)
    corridor_mask = contract_tensor_like(reference, corridor_semantic_corridor_mask, 0.0).clamp(0.0, 1.0)
    task_degradation = contract_tensor_like(reference, critic_contract_task_degradation, 0.0).clamp(0.0, 1.0)
    late_gate = contract_tensor_like(reference, critic_contract_bootstrap_late_gate, 0.0).clamp(0.0, 1.0)
    source_value_truth = torch.maximum(reward_truth, real_recovery).clamp(0.0, 1.0)
    source_value_alignment = torch.maximum(imag_gate, task_gate).clamp(0.0, 1.0)
    source_value_support = torch.where(
        corridor_mask > 1e-6,
        support_mask,
        torch.ones_like(reference),
    ).clamp(0.0, 1.0)
    source_value_late_window_activation = ((late_gate - 0.8) / 0.2).clamp(0.0, 1.0)
    source_value_alarm = torch.maximum(
        (1.0 - source_value_truth).clamp(0.0, 1.0),
        torch.maximum((1.0 - source_value_alignment).clamp(0.0, 1.0), task_degradation),
    ).clamp(0.0, 1.0)
    source_value_bonus_gate = (1.0 - source_value_late_window_activation * source_value_alarm).clamp(0.0, 1.0)
    strict_bonus_source_mix = (
        source_value_support
        * torch.minimum(source_value_truth, source_value_alignment).clamp(0.0, 1.0)
        * source_value_bonus_gate
    ).clamp(0.0, 1.0)
    strict_bonus_value = torch.lerp(internal_value, reward_semantic_value, strict_bonus_source_mix)
    source_value_replacement_mix = (
        source_value_late_window_activation * strict_bonus_source_mix
    ).clamp(0.0, 1.0)
    replaced = torch.lerp(external_value_raw, strict_bonus_value, source_value_replacement_mix)
    delta_abs = (replaced - external_value_raw).abs()
    return (
        source_value_truth.detach(),
        source_value_alignment.detach(),
        source_value_late_window_activation.detach(),
        source_value_bonus_gate.detach(),
        replaced.detach(),
        delta_abs.detach(),
    )


def compute_post_transition_certified_retention_floor_contract(
    *,
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    critic_contract_bootstrap_trigger_gate: Union[float, Tensor],
    critic_contract_bootstrap_hold_state_active_transition: float,
    critic_contract_bootstrap_hold_state_post_transition: float,
    source_hold_seed_mean: float,
    source_hold_health_mean: float,
    source_hold_retention_mean: float,
    source_hold_release_pressure_mean: float,
    critic_contract_bootstrap_source_hold_persistence_gate_mean: float,
    prev_post_transition_certified_floor_state: float,
) -> tuple[float, float, float, float, float, float]:
    if isinstance(critic_contract_bootstrap_late_gate, Tensor):
        late_gate_mean = float(critic_contract_bootstrap_late_gate.detach().float().mean().item())
    else:
        late_gate_mean = float(critic_contract_bootstrap_late_gate)
    if isinstance(critic_contract_bootstrap_trigger_gate, Tensor):
        trigger_gate_mean = float(critic_contract_bootstrap_trigger_gate.detach().float().mean().item())
    else:
        trigger_gate_mean = float(critic_contract_bootstrap_trigger_gate)
    trigger_gate_mean = min(1.0, max(0.0, trigger_gate_mean))
    capture = compute_post_transition_retention_capture(
        late_gate_mean=late_gate_mean,
        active_hold_state=critic_contract_bootstrap_hold_state_active_transition,
        post_hold_state=critic_contract_bootstrap_hold_state_post_transition,
        source_hold_seed_mean=source_hold_seed_mean,
        source_hold_health_mean=source_hold_health_mean,
        source_hold_retention_mean=source_hold_retention_mean,
        source_hold_release_pressure_mean=source_hold_release_pressure_mean,
        prev_post_transition_certified_floor_state=prev_post_transition_certified_floor_state,
    )
    raw_persistence_gate_mean = min(
        1.0,
        max(0.0, float(critic_contract_bootstrap_source_hold_persistence_gate_mean)),
    )
    if capture.window_activation <= 0.0:
        return (
            0.0,
            0.0,
            0.0,
            0.0,
            float(min(1.0, max(0.0, float(critic_contract_bootstrap_hold_state_post_transition)))),
            float(raw_persistence_gate_mean),
        )
    certified_capture_source = float(capture.certified_capture_source)
    if capture.is_historical_winner:
        # Historical carry may recover lost authority, but it must remain
        # bounded by the currently supported seed × health envelope.
        certified_capture_source = min(
            certified_capture_source,
            min(1.0, capture.current_authority + capture.capture_support),
        )
    elif capture.release_blocked and raw_persistence_gate_mean > 0.0:
        # Break exact current-authority ties when persistence evidence exists,
        # so downstream floor/gate comparisons stay strictly ordered.
        certified_capture_source = max(
            certified_capture_source,
            math.nextafter(capture.current_authority, 1.0),
        )
    carry_decay = 0.90 * (1.0 - 0.25 * float(source_hold_release_pressure_mean))
    carry_decay = min(1.0, max(1e-6, carry_decay))
    certified_capture = capture.window_activation * certified_capture_source
    if capture.release_blocked and raw_persistence_gate_mean > 0.0:
        certified_capture = max(
            certified_capture,
            math.nextafter(capture.current_authority, 1.0),
        )
    floor_candidate = max(
        carry_decay * min(1.0, max(0.0, float(prev_post_transition_certified_floor_state))),
        certified_capture,
    )
    if capture.is_historical_winner:
        historical_gap = max(0.0, capture.historical_authority - capture.current_authority)
        historical_gap_support_ratio = (
            min(1.0, capture.capture_support / historical_gap)
            if historical_gap > 1e-6
            else 0.0
        )
        if historical_gap_support_ratio >= 1.0:
            support_bounded_historical_floor = min(
                capture.historical_authority,
                min(1.0, capture.current_authority + capture.capture_support),
            )
            floor_candidate = max(
                floor_candidate,
                support_bounded_historical_floor,
            )
        floor_candidate = max(
            floor_candidate,
            min(1.0, capture.current_authority + capture.dominance_margin),
        )
    floor_state = min(1.0, max(0.0, trigger_gate_mean * floor_candidate))
    hold_state_floored = max(
        min(1.0, max(0.0, float(critic_contract_bootstrap_hold_state_post_transition))),
        floor_state,
    )
    persistence_gate_floored = max(raw_persistence_gate_mean, floor_state)
    if capture.is_historical_winner:
        persistence_gate_floored = max(
            persistence_gate_floored,
            min(1.0, raw_persistence_gate_mean + capture.dominance_margin),
        )
    return (
        float(capture.window_activation),
        float(certified_capture),
        float(floor_candidate),
        float(floor_state),
        float(hold_state_floored),
        float(persistence_gate_floored),
    )


def compute_bootstrap_bonus_source_hold_persistence_contract(
    *,
    bootstrap_external_bonus_value_source_replaced: Tensor,
    bootstrap_external_bonus_value_transition_bridged: Tensor,
    raw_anchor_bootstrap_values_next: Tensor,
    raw_bootstrap_values_next: Tensor,
    reward_semantic_hold_source_value: Tensor,
    critic_anchor_available_next: Tensor,
    reference_real_reward_agreement: Union[float, Tensor],
    behavior_policy_task_cert_real_recovery: Tensor,
    behavior_policy_task_cert_real_alarm: Tensor,
    behavior_policy_task_cert_imag_gate: Tensor,
    task_corridor_gate: Tensor,
    critic_contract_task_degradation: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    critic_contract_bootstrap_source_transition_bridge_gate: Tensor,
    critic_contract_bootstrap_source_value_late_window_activation: Tensor,
    critic_contract_bootstrap_bonus_hold_state: Union[float, Tensor],
    critic_contract_bootstrap_floor_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_final_value_injection_gate: Union[float, Tensor] = 1.0,
    corridor_semantic_corridor_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    reference = bootstrap_external_bonus_value_source_replaced.detach()
    replaced_value = (
        bootstrap_external_bonus_value_source_replaced.detach()
        if isinstance(bootstrap_external_bonus_value_source_replaced, Tensor) and bootstrap_external_bonus_value_source_replaced.shape == reference.shape
        else torch.zeros_like(reference)
    )
    bridged_value = (
        bootstrap_external_bonus_value_transition_bridged.detach()
        if isinstance(bootstrap_external_bonus_value_transition_bridged, Tensor) and bootstrap_external_bonus_value_transition_bridged.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_value = (
        raw_anchor_bootstrap_values_next.detach()
        if isinstance(raw_anchor_bootstrap_values_next, Tensor) and raw_anchor_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    internal_value = (
        raw_bootstrap_values_next.detach()
        if isinstance(raw_bootstrap_values_next, Tensor) and raw_bootstrap_values_next.shape == reference.shape
        else torch.zeros_like(reference)
    )
    anchor_available = contract_tensor_like(reference, critic_anchor_available_next, 0.0).clamp(0.0, 1.0)
    reward_semantic_hold_source = (
        reward_semantic_hold_source_value.detach()
        if isinstance(reward_semantic_hold_source_value, Tensor) and reward_semantic_hold_source_value.shape == reference.shape
        else torch.where(anchor_available > 0.0, anchor_value, internal_value)
    )
    reward_truth = contract_tensor_like(reference, reference_real_reward_agreement, 1.0).clamp(0.0, 1.0)
    real_recovery = contract_tensor_like(reference, behavior_policy_task_cert_real_recovery, 1.0).clamp(0.0, 1.0)
    real_alarm = contract_tensor_like(reference, behavior_policy_task_cert_real_alarm, 0.0).clamp(0.0, 1.0)
    imag_gate = contract_tensor_like(reference, behavior_policy_task_cert_imag_gate, 1.0).clamp(0.0, 1.0)
    task_gate = contract_tensor_like(reference, task_corridor_gate, 1.0).clamp(0.0, 1.0)
    corridor_mask = contract_tensor_like(reference, corridor_semantic_corridor_mask, 0.0).clamp(0.0, 1.0)
    task_degradation = contract_tensor_like(reference, critic_contract_task_degradation, 0.0).clamp(0.0, 1.0)
    late_gate = contract_tensor_like(reference, critic_contract_bootstrap_late_gate, 0.0).clamp(0.0, 1.0)
    transition_bridge_gate = contract_tensor_like(
        reference, critic_contract_bootstrap_source_transition_bridge_gate, 0.0
    ).clamp(0.0, 1.0)
    late_window_activation = contract_tensor_like(
        reference, critic_contract_bootstrap_source_value_late_window_activation, 0.0
    ).clamp(0.0, 1.0)
    bonus_hold_state = contract_tensor_like(reference, critic_contract_bootstrap_bonus_hold_state, 0.0).clamp(0.0, 1.0)
    floor_value_gate = contract_tensor_like(reference, critic_contract_bootstrap_floor_value_injection_gate, 1.0).clamp(0.0, 1.0)
    final_value_gate = contract_tensor_like(reference, critic_contract_bootstrap_final_value_injection_gate, 1.0).clamp(0.0, 1.0)
    source_hold_truth = torch.maximum(reward_truth, real_recovery).clamp(0.0, 1.0)
    source_hold_alignment = torch.maximum(imag_gate, task_gate).clamp(0.0, 1.0)
    source_hold_support = torch.where(
        corridor_mask > 1e-6,
        (1.0 - real_alarm).clamp(0.0, 1.0),
        torch.ones_like(reference),
    ).clamp(0.0, 1.0)
    source_hold_window_activation = ((late_gate - 0.55) / 0.25).clamp(0.0, 1.0)
    source_hold_release_pressure = torch.maximum(real_alarm, task_degradation).clamp(0.0, 1.0)
    source_hold_release_window_activation = ((late_gate - 0.9) / 0.1).clamp(0.0, 1.0)
    source_hold_release_gate = (source_hold_release_window_activation * source_hold_release_pressure).clamp(0.0, 1.0)
    source_hold_release_aware_state = (bonus_hold_state * (1.0 - source_hold_release_gate)).clamp(0.0, 1.0)
    source_hold_handoff_window_activation = ((late_gate - 0.75) / 0.25).clamp(0.0, 1.0)
    handoff_bridge_floor = torch.sqrt((source_hold_handoff_window_activation * transition_bridge_gate).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    handoff_activation = torch.maximum(late_window_activation, handoff_bridge_floor).clamp(0.0, 1.0)
    source_hold_alarm = task_degradation.clamp(0.0, 1.0)
    source_hold_certification = torch.minimum(source_hold_truth, source_hold_alignment).clamp(0.0, 1.0)
    source_hold_state_certified_floor = (
        source_hold_release_aware_state * torch.sqrt(source_hold_certification.clamp(0.0, 1.0))
    ).clamp(0.0, 1.0)
    source_hold_trajectory_floor = torch.maximum(
        source_hold_state_certified_floor,
        (handoff_activation * torch.sqrt(source_hold_certification.clamp(0.0, 1.0))).clamp(0.0, 1.0),
    ).clamp(0.0, 1.0)
    source_hold_live_retention_support = torch.maximum(
        source_hold_support,
        (handoff_activation * source_hold_certification).clamp(0.0, 1.0),
    ).clamp(0.0, 1.0)
    release_balance = (
        1.0
        - (
            source_hold_release_window_activation.pow(4.0)
            * source_hold_release_gate
        )
    ).clamp(0.0, 1.0)
    source_hold_persistence_gate = (
        source_hold_window_activation
        * source_hold_trajectory_floor
        * source_hold_live_retention_support
        * release_balance
    ).clamp(0.0, 1.0)
    value_gate = torch.minimum(floor_value_gate, final_value_gate).clamp(0.0, 1.0)
    source_hold_value_health = torch.minimum(
        source_hold_certification,
        value_gate,
    ).clamp(0.0, 1.0)
    persistence_release_coupling = (
        1.0
        - source_hold_window_activation
        * torch.sqrt(source_hold_release_pressure)
        * (1.0 - source_hold_value_health)
    ).clamp(0.0, 1.0)
    source_hold_persistence_gate = (
        source_hold_persistence_gate * persistence_release_coupling
    ).clamp(0.0, 1.0)
    source_hold_target_pull = (
        source_hold_trajectory_floor * source_hold_certification
    ).clamp(0.0, 1.0)
    strict_hold_value = torch.lerp(
        bridged_value,
        reward_semantic_hold_source,
        source_hold_target_pull,
    )
    hold_internal_fallback_coupling = (
        source_hold_window_activation
        * torch.sqrt(source_hold_release_pressure)
        * (1.0 - source_hold_value_health)
    ).clamp(0.0, 1.0)
    strict_hold_value = torch.lerp(
        strict_hold_value,
        internal_value,
        hold_internal_fallback_coupling,
    )
    persisted = torch.lerp(replaced_value, strict_hold_value, source_hold_persistence_gate)
    delta_abs = (persisted - replaced_value).abs()
    return (
        source_hold_truth.detach(),
        source_hold_alignment.detach(),
        source_hold_window_activation.detach(),
        source_hold_release_window_activation.detach(),
        source_hold_release_pressure.detach(),
        source_hold_release_gate.detach(),
        source_hold_release_aware_state.detach(),
        source_hold_persistence_gate.detach(),
        persisted.detach(),
        delta_abs.detach(),
    )


def compute_bootstrap_bonus_consumer_retention_contract(
    *,
    bootstrap_external_bonus_value_hold_persisted: Tensor,
    bootstrap_external_bonus_terminal_truth_source: Tensor,
    bootstrap_external_bonus_value_source_replaced: Tensor,
    reference_real_reward_agreement: Union[float, Tensor],
    behavior_policy_task_cert_real_recovery: Tensor,
    behavior_policy_task_cert_real_alarm: Tensor,
    behavior_policy_task_cert_imag_gate: Tensor,
    task_corridor_gate: Tensor,
    critic_contract_task_degradation: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    critic_contract_bootstrap_bonus_hold_state: Union[float, Tensor],
    critic_contract_bootstrap_source_hold_persistence_gate: Tensor,
    corridor_semantic_corridor_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    reference = bootstrap_external_bonus_value_hold_persisted.detach()
    hold_persisted_value = (
        bootstrap_external_bonus_value_hold_persisted.detach()
        if isinstance(bootstrap_external_bonus_value_hold_persisted, Tensor) and bootstrap_external_bonus_value_hold_persisted.shape == reference.shape
        else torch.zeros_like(reference)
    )
    terminal_truth_source = (
        bootstrap_external_bonus_terminal_truth_source.detach()
        if isinstance(bootstrap_external_bonus_terminal_truth_source, Tensor) and bootstrap_external_bonus_terminal_truth_source.shape == reference.shape
        else torch.zeros_like(reference)
    )
    reward_truth = contract_tensor_like(reference, reference_real_reward_agreement, 1.0).clamp(0.0, 1.0)
    real_recovery = contract_tensor_like(reference, behavior_policy_task_cert_real_recovery, 1.0).clamp(0.0, 1.0)
    real_alarm = contract_tensor_like(reference, behavior_policy_task_cert_real_alarm, 0.0).clamp(0.0, 1.0)
    imag_gate = contract_tensor_like(reference, behavior_policy_task_cert_imag_gate, 1.0).clamp(0.0, 1.0)
    task_gate = contract_tensor_like(reference, task_corridor_gate, 1.0).clamp(0.0, 1.0)
    corridor_mask = contract_tensor_like(reference, corridor_semantic_corridor_mask, 0.0).clamp(0.0, 1.0)
    task_degradation = contract_tensor_like(reference, critic_contract_task_degradation, 0.0).clamp(0.0, 1.0)
    late_gate = contract_tensor_like(reference, critic_contract_bootstrap_late_gate, 0.0).clamp(0.0, 1.0)
    bonus_hold_state = contract_tensor_like(reference, critic_contract_bootstrap_bonus_hold_state, 0.0).clamp(0.0, 1.0)
    persistence_gate = contract_tensor_like(reference, critic_contract_bootstrap_source_hold_persistence_gate, 0.0).clamp(0.0, 1.0)
    source_consumer_truth = torch.maximum(reward_truth, real_recovery).clamp(0.0, 1.0)
    source_consumer_alignment = torch.maximum(imag_gate, task_gate).clamp(0.0, 1.0)
    source_consumer_support = torch.where(
        corridor_mask > 1e-6,
        (1.0 - real_alarm).clamp(0.0, 1.0),
        torch.ones_like(reference),
    ).clamp(0.0, 1.0)
    window_activation = ((late_gate - 0.9) / 0.1).clamp(0.0, 1.0)
    consumer_health = (
        source_consumer_support
        * torch.minimum(source_consumer_truth, source_consumer_alignment).clamp(0.0, 1.0)
        * (1.0 - torch.maximum(real_alarm, task_degradation)).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    strict_floor = (window_activation * bonus_hold_state * consumer_health).clamp(0.0, 1.0)
    strengthening_margin = (consumer_health - persistence_gate).clamp(0.0, 1.0)
    strengthening = (
        (1.0 - persistence_gate).clamp(0.0, 1.0) * strict_floor * strengthening_margin
    ).clamp(0.0, 1.0)
    retention_gate = (persistence_gate + strengthening).clamp(0.0, 1.0)
    retained = torch.lerp(hold_persisted_value, terminal_truth_source, strengthening)
    delta_abs = (retained - hold_persisted_value).abs()
    return (
        source_consumer_truth.detach(),
        source_consumer_alignment.detach(),
        window_activation.detach(),
        retention_gate.detach(),
        retained.detach(),
        delta_abs.detach(),
    )
