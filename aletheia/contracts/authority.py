from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional, Union

import torch
from torch import Tensor

from .core import contract_tensor_like

__all__ = [
    "AuthorityDecision",
    "compute_bootstrap_task_request_contract",
    "compute_bootstrap_external_authority_floor_contract",
    "compute_bootstrap_external_authority_takeover_floor_contract",
    "compute_bootstrap_external_authority_floor_live_unwind_contract",
    "compute_bootstrap_authority_source_recertification_contract",
    "compute_bootstrap_authority_source_replacement_contract",
    "compute_bootstrap_trigger_entry_contract",
    "compute_post_transition_retention_capture",
    "RetentionCaptureDecision",
]


@dataclass(frozen=True)
class AuthorityDecision:
    """Canonical authority carrier for bootstrap request/floor/replacement outputs."""

    reference: Tensor
    tensors: dict[str, Tensor] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_outputs(
        cls,
        *,
        reference: Tensor,
        tensors: dict[str, Tensor] | None = None,
        scalars: dict[str, float] | None = None,
    ) -> "AuthorityDecision":
        return cls(
            reference=reference.detach(),
            tensors={name: value.detach() for name, value in (tensors or {}).items()},
            scalars={name: float(value) for name, value in (scalars or {}).items()},
        )

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"shape": tuple(self.reference.shape)}
        for name, value in self.tensors.items():
            summary[f"{name}_mean"] = float(value.detach().float().mean().item())
        summary.update({name: float(value) for name, value in self.scalars.items()})
        return summary


@dataclass(frozen=True)
class RetentionCaptureDecision:
    window_activation: float
    current_authority: float
    historical_authority: float
    capture_support: float
    historical_takeover_gate: float
    certified_capture_source: float
    dominant_source: str
    dominance_margin: float
    is_historical_winner: bool
    release_blocked: bool


def _masked_mean_tensor(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.detach().float()
    total_weight = weight.sum().clamp(min=1e-6)
    return (value.detach().float() * weight).sum() / total_weight


def compute_bootstrap_task_request_contract(
    *,
    real_task_alarm: Tensor,
    task_support_mask: Tensor,
    imag_task_gate: Tensor,
    task_gate: Tensor,
    negative_adv_pressure: Tensor,
    release_guard: Tensor,
    task_degradation: Tensor,
    corridor_mask: Tensor,
    semantic_debt: Tensor,
    control_gate: Union[float, Tensor],
    clean_mix_max: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    reference = semantic_debt.detach()
    task_request_alarm = torch.maximum(
        (1.0 - contract_tensor_like(reference, real_task_alarm, 1.0)).clamp(0.0, 1.0),
        (1.0 - contract_tensor_like(reference, task_support_mask, 1.0)).clamp(0.0, 1.0),
    ).clamp(0.0, 1.0)
    task_request_disagreement = torch.maximum(
        (1.0 - contract_tensor_like(reference, imag_task_gate, 1.0)).clamp(0.0, 1.0),
        (1.0 - contract_tensor_like(reference, task_gate, 1.0)).clamp(0.0, 1.0),
    ).clamp(0.0, 1.0)
    task_request_raw = torch.maximum(
        task_request_alarm,
        torch.maximum(
            task_request_disagreement,
            torch.maximum(
                contract_tensor_like(reference, negative_adv_pressure, 0.0).clamp(0.0, 1.0),
                torch.maximum(
                    contract_tensor_like(reference, release_guard, 0.0).clamp(0.0, 1.0),
                    contract_tensor_like(reference, task_degradation, 0.0).clamp(0.0, 1.0),
                ),
            ),
        ),
    ).clamp(0.0, 1.0)
    task_request_scope = torch.maximum(
        contract_tensor_like(reference, corridor_mask, 0.0).clamp(0.0, 1.0),
        torch.maximum(
            contract_tensor_like(reference, semantic_debt, 0.0).clamp(0.0, 1.0),
            task_request_alarm,
        ),
    ).clamp(0.0, 1.0)
    if isinstance(control_gate, Tensor):
        control_gate_tensor = contract_tensor_like(reference, control_gate, 0.0)
    else:
        control_gate_tensor = torch.full_like(
            reference,
            min(1.0, max(0.0, float(control_gate))),
        )
    task_request_floor = torch.clamp(
        control_gate_tensor.clamp(0.0, 1.0)
        * max(0.0, float(clean_mix_max))
        * task_request_scope
        * torch.maximum(
            task_request_raw,
            contract_tensor_like(reference, semantic_debt, 0.0).clamp(0.0, 1.0),
        ),
        min=0.0,
        max=max(0.0, float(clean_mix_max)),
    )
    return (
        task_request_alarm.detach(),
        task_request_disagreement.detach(),
        task_request_scope.detach(),
        task_request_floor.detach(),
    )


def compute_bootstrap_external_authority_floor_contract(
    *,
    critic_contract_bootstrap_requested_floor: Tensor,
    bootstrap_anchor_valid_mask: Tensor,
    critic_contract_bootstrap_semantic_debt: Tensor,
    critic_contract_bootstrap_task_request_alarm: Tensor,
    critic_contract_bootstrap_task_request_disagreement: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    reference = contract_tensor_like(
        critic_contract_bootstrap_requested_floor.detach(),
        critic_contract_bootstrap_requested_floor,
        0.0,
    ).clamp(0.0, 1.0)
    valid_mask = contract_tensor_like(reference, bootstrap_anchor_valid_mask, 0.0).clamp(0.0, 1.0)
    semantic_debt = contract_tensor_like(
        reference,
        critic_contract_bootstrap_semantic_debt,
        0.0,
    ).clamp(0.0, 1.0)
    contract_tensor_like(reference, critic_contract_bootstrap_task_request_alarm, 0.0)
    contract_tensor_like(reference, critic_contract_bootstrap_task_request_disagreement, 0.0)
    external_authority_floor = (valid_mask * reference).clamp(0.0, 1.0)
    external_authority_debt_high_mask = (
        valid_mask * (semantic_debt >= 0.5).to(dtype=valid_mask.dtype)
    ).detach()
    return (
        external_authority_floor.detach(),
        valid_mask.detach(),
        external_authority_debt_high_mask,
    )


def compute_bootstrap_external_authority_takeover_floor_contract(
    *,
    bootstrap_external_authority_floor: Tensor,
    bootstrap_anchor_valid_mask: Tensor,
    critic_contract_bootstrap_requested_floor: Union[float, Tensor],
    external_takeover_floor_ratio: Union[float, Tensor] = 1.0,
    max_external_authority: Union[float, Tensor] = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    reference = contract_tensor_like(
        bootstrap_external_authority_floor.detach(),
        bootstrap_external_authority_floor,
        0.0,
    ).clamp(0.0, 1.0)
    floor_authority = contract_tensor_like(reference, bootstrap_external_authority_floor, 0.0).clamp(
        0.0, 1.0
    )
    valid_mask = contract_tensor_like(reference, bootstrap_anchor_valid_mask, 0.0).clamp(0.0, 1.0)
    requested_floor = contract_tensor_like(
        reference,
        critic_contract_bootstrap_requested_floor,
        0.0,
    ).clamp(0.0, 1.0)
    if isinstance(external_takeover_floor_ratio, Tensor):
        takeover_floor_ratio = contract_tensor_like(
            reference,
            external_takeover_floor_ratio,
            1.0,
        ).clamp(0.0, 1.0)
    else:
        takeover_floor_ratio = torch.full_like(
            reference,
            min(1.0, max(0.0, float(external_takeover_floor_ratio))),
        )
    if isinstance(max_external_authority, Tensor):
        max_external_authority_tensor = contract_tensor_like(
            reference,
            max_external_authority,
            1.0,
        ).clamp(0.0, 1.0)
    else:
        max_external_authority_tensor = torch.full_like(
            reference,
            min(1.0, max(0.0, float(max_external_authority))),
        )
    takeover_floor = (
        valid_mask * requested_floor * takeover_floor_ratio
    ).clamp(0.0, 1.0)
    takeover_floor = torch.minimum(
        takeover_floor,
        max_external_authority_tensor,
    ).detach()
    enforced_floor_authority = torch.maximum(
        floor_authority,
        takeover_floor,
    ).clamp(0.0, 1.0)
    enforcement_delta_abs = (
        enforced_floor_authority - floor_authority
    ).clamp(min=0.0).detach()
    return (
        takeover_floor,
        enforced_floor_authority.detach(),
        enforcement_delta_abs,
    )


def compute_bootstrap_external_authority_floor_live_unwind_contract(
    *,
    bootstrap_external_authority_floor: Tensor,
    critic_contract_bootstrap_requested_floor: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    critic_contract_bootstrap_floor_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_final_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_source_recert_persistence: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_source_hold_window_activation: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_regime_quality_gate: Union[float, Tensor] = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    reference = contract_tensor_like(
        bootstrap_external_authority_floor.detach(),
        bootstrap_external_authority_floor,
        0.0,
    ).clamp(0.0, 1.0)
    floor_authority = contract_tensor_like(reference, bootstrap_external_authority_floor, 0.0).clamp(
        0.0, 1.0
    )
    requested_floor = contract_tensor_like(
        reference,
        critic_contract_bootstrap_requested_floor,
        0.0,
    ).clamp(0.0, 1.0)
    if isinstance(critic_contract_bootstrap_late_gate, Tensor):
        late_gate = contract_tensor_like(
            reference,
            critic_contract_bootstrap_late_gate,
            0.0,
        ).clamp(0.0, 1.0)
    else:
        late_gate = torch.full_like(
            reference,
            min(1.0, max(0.0, float(critic_contract_bootstrap_late_gate))),
        )
    if isinstance(critic_contract_bootstrap_floor_value_injection_gate, Tensor):
        floor_value_gate = contract_tensor_like(
            reference,
            critic_contract_bootstrap_floor_value_injection_gate,
            1.0,
        ).clamp(0.0, 1.0)
    else:
        floor_value_gate = torch.full_like(
            reference,
            min(1.0, max(0.0, float(critic_contract_bootstrap_floor_value_injection_gate))),
        )
    if isinstance(critic_contract_bootstrap_final_value_injection_gate, Tensor):
        final_value_gate = contract_tensor_like(
            reference,
            critic_contract_bootstrap_final_value_injection_gate,
            1.0,
        ).clamp(0.0, 1.0)
    else:
        final_value_gate = torch.full_like(
            reference,
            min(1.0, max(0.0, float(critic_contract_bootstrap_final_value_injection_gate))),
        )
    if isinstance(critic_contract_bootstrap_source_recert_persistence, Tensor):
        source_recert_persistence = contract_tensor_like(
            reference,
            critic_contract_bootstrap_source_recert_persistence,
            1.0,
        ).clamp(0.0, 1.0)
    else:
        source_recert_persistence = torch.full_like(
            reference,
            min(1.0, max(0.0, float(critic_contract_bootstrap_source_recert_persistence))),
        )
    if isinstance(critic_contract_bootstrap_source_hold_window_activation, Tensor):
        hold_window_activation = contract_tensor_like(
            reference,
            critic_contract_bootstrap_source_hold_window_activation,
            0.0,
        ).clamp(0.0, 1.0)
    else:
        hold_window_activation = torch.full_like(
            reference,
            min(1.0, max(0.0, float(critic_contract_bootstrap_source_hold_window_activation))),
        )
    if isinstance(critic_contract_bootstrap_regime_quality_gate, Tensor):
        regime_quality_gate = contract_tensor_like(
            reference,
            critic_contract_bootstrap_regime_quality_gate,
            1.0,
        ).clamp(0.0, 1.0)
    else:
        regime_quality_gate = torch.full_like(
            reference,
            min(1.0, max(0.0, float(critic_contract_bootstrap_regime_quality_gate))),
        )

    midlate_transition_activation = (
        torch.sqrt(((late_gate - 0.30) / 0.25).clamp(0.0, 1.0))
        * (1.0 - ((late_gate - 0.82) / 0.18).clamp(0.0, 1.0))
        * (1.0 - hold_window_activation).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    value_gate = torch.minimum(floor_value_gate, final_value_gate).clamp(0.0, 1.0)
    value_collapse = (1.0 - torch.sqrt(value_gate)).clamp(0.0, 1.0)
    live_health = torch.maximum(source_recert_persistence, regime_quality_gate).clamp(0.0, 1.0)
    live_deficit = (1.0 - live_health).clamp(0.0, 1.0)
    unwind_pressure = (
        torch.sqrt(value_collapse)
        * live_deficit
        * torch.sqrt(requested_floor)
    ).clamp(0.0, 1.0)
    floor_live_unwind_gate = (
        1.0 - midlate_transition_activation * unwind_pressure
    ).clamp(0.20, 1.0)
    healthy_override = (
        (midlate_transition_activation > 1e-6)
        & (value_gate > 0.75)
        & (live_health > 0.70)
    ).to(dtype=reference.dtype)
    floor_live_unwind_gate = torch.where(
        healthy_override > 1e-6,
        torch.maximum(floor_live_unwind_gate, torch.full_like(reference, 0.95)),
        floor_live_unwind_gate,
    ).clamp(0.20, 1.0)
    unwound_floor_authority = (
        floor_authority * floor_live_unwind_gate
    ).clamp(0.0, 1.0)
    floor_live_unwind_delta_abs = (floor_authority - unwound_floor_authority).abs().detach()
    return (
        midlate_transition_activation.detach(),
        floor_live_unwind_gate.detach(),
        unwound_floor_authority.detach(),
        floor_live_unwind_delta_abs,
    )


def compute_bootstrap_authority_source_recertification_contract(
    *,
    bootstrap_external_authority_raw: Tensor,
    bootstrap_external_authority_floor: Tensor,
    behavior_policy_task_cert_real_gate: Tensor,
    behavior_policy_task_cert_real_alarm: Tensor,
    behavior_policy_task_cert_real_recovery: Tensor,
    behavior_policy_task_cert_imag_gate: Tensor,
    behavior_policy_task_cert_support_mask: Tensor,
    task_corridor_gate: Tensor,
    critic_contract_task_degradation: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    critic_contract_bootstrap_floor_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_final_value_injection_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_source_hold_release_pressure: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_source_hold_persistence_gate: Union[float, Tensor] = 1.0,
    critic_contract_bootstrap_source_hold_window_activation: Union[float, Tensor] = 0.0,
    critic_contract_bootstrap_regime_quality_gate: Union[float, Tensor] = 1.0,
    actor_high_value_but_low_task_fraction: Union[float, Tensor] = 0.0,
    actor_task_geom_corridor_disagreement: Union[float, Tensor] = 0.0,
    corridor_semantic_corridor_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    reference = contract_tensor_like(
        bootstrap_external_authority_floor.detach(),
        bootstrap_external_authority_floor,
        0.0,
    ).clamp(0.0, 1.0)
    raw_authority = contract_tensor_like(reference, bootstrap_external_authority_raw, 0.0).clamp(
        0.0, 1.0
    )
    floor_authority = contract_tensor_like(reference, bootstrap_external_authority_floor, 0.0).clamp(
        0.0, 1.0
    )
    real_gate = contract_tensor_like(reference, behavior_policy_task_cert_real_gate, 1.0).clamp(
        0.0, 1.0
    )
    real_alarm = contract_tensor_like(reference, behavior_policy_task_cert_real_alarm, 1.0).clamp(
        0.0, 1.0
    )
    real_recovery = contract_tensor_like(
        reference,
        behavior_policy_task_cert_real_recovery,
        1.0,
    ).clamp(0.0, 1.0)
    imag_gate = contract_tensor_like(reference, behavior_policy_task_cert_imag_gate, 1.0).clamp(
        0.0, 1.0
    )
    support_mask = contract_tensor_like(
        reference,
        behavior_policy_task_cert_support_mask,
        1.0,
    ).clamp(0.0, 1.0)
    task_gate = contract_tensor_like(reference, task_corridor_gate, 1.0).clamp(0.0, 1.0)
    # contract_tensor_like already handles both Tensor and scalar inputs
    task_degradation = contract_tensor_like(reference, critic_contract_task_degradation, 0.0).clamp(0.0, 1.0)
    late_gate = contract_tensor_like(reference, critic_contract_bootstrap_late_gate, 0.0).clamp(0.0, 1.0)
    floor_value_gate = contract_tensor_like(reference, critic_contract_bootstrap_floor_value_injection_gate, 1.0).clamp(0.0, 1.0)
    final_value_gate = contract_tensor_like(reference, critic_contract_bootstrap_final_value_injection_gate, 1.0).clamp(0.0, 1.0)
    hold_release_pressure = contract_tensor_like(reference, critic_contract_bootstrap_source_hold_release_pressure, 0.0).clamp(0.0, 1.0)
    hold_persistence_gate = contract_tensor_like(reference, critic_contract_bootstrap_source_hold_persistence_gate, 1.0).clamp(0.0, 1.0)
    hold_window_activation = contract_tensor_like(reference, critic_contract_bootstrap_source_hold_window_activation, 0.0).clamp(0.0, 1.0)
    regime_quality_gate = contract_tensor_like(reference, critic_contract_bootstrap_regime_quality_gate, 1.0).clamp(0.0, 1.0)
    high_value_but_low_task_fraction = contract_tensor_like(reference, actor_high_value_but_low_task_fraction, 0.0).clamp(0.0, 1.0)
    task_geom_corridor_disagreement = contract_tensor_like(reference, actor_task_geom_corridor_disagreement, 0.0).clamp(0.0, 1.0)
    contract_tensor_like(reference, corridor_semantic_corridor_mask, 0.0)

    bootstrap_external_authority_prefinal = torch.maximum(raw_authority, floor_authority).clamp(
        0.0, 1.0
    )
    bootstrap_external_authority_bonus_prefinal = (
        bootstrap_external_authority_prefinal - floor_authority
    ).clamp(min=0.0, max=1.0)
    source_recert_alarm = torch.maximum(
        (1.0 - real_gate).clamp(0.0, 1.0),
        torch.maximum((1.0 - support_mask).clamp(0.0, 1.0), task_degradation),
    ).clamp(0.0, 1.0)
    source_recert_persistence = torch.minimum(
        real_recovery,
        torch.maximum(imag_gate, task_gate).clamp(0.0, 1.0),
    ).clamp(0.0, 1.0)
    source_recert_gate_core = (
        1.0
        - late_gate
        * torch.maximum(
            source_recert_alarm,
            (1.0 - source_recert_persistence).clamp(0.0, 1.0),
        )
    ).clamp(0.0, 1.0)
    floor_supported_bonus_gate_floor = (
        late_gate
        * torch.sqrt((floor_authority * source_recert_persistence).clamp(0.0, 1.0))
    ).clamp(0.0, 1.0)
    source_recert_gate_pure = torch.maximum(
        source_recert_gate_core,
        floor_supported_bonus_gate_floor,
    ).clamp(0.0, 1.0)
    bonus_headroom_share = (
        bootstrap_external_authority_bonus_prefinal
        / bootstrap_external_authority_prefinal.clamp(min=1e-6)
    ).clamp(0.0, 1.0)
    headroom_peak_recovery_gate_floor = (
        source_recert_gate_pure
        + (1.0 - source_recert_gate_pure)
        * late_gate
        * torch.sqrt(source_recert_persistence)
        * torch.sqrt(bonus_headroom_share)
    ).clamp(0.0, 1.0)
    source_recert_gate_pure = torch.maximum(
        source_recert_gate_pure,
        headroom_peak_recovery_gate_floor,
    ).clamp(0.0, 1.0)
    midlate_transition_activation = (
        torch.sqrt(((late_gate - 0.05) / 0.45).clamp(0.0, 1.0))
        * (1.0 - ((late_gate - 0.85) / 0.15).clamp(0.0, 1.0))
        * (1.0 - hold_window_activation).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    value_gate = torch.minimum(floor_value_gate, final_value_gate).clamp(0.0, 1.0)
    midlate_value_collapse = (1.0 - torch.sqrt(value_gate)).clamp(0.0, 1.0)
    midlate_persistence_deficit = (1.0 - source_recert_persistence).clamp(0.0, 1.0)
    midlate_regime_deficit = (1.0 - regime_quality_gate).clamp(0.0, 1.0)
    midlate_fragility = torch.maximum(
        midlate_persistence_deficit,
        midlate_regime_deficit,
    ).clamp(0.0, 1.0)
    midlate_bonus_alignment_gate = (
        1.0
        - midlate_transition_activation
        * torch.sqrt(midlate_value_collapse)
        * midlate_fragility
    ).clamp(0.0, 1.0)
    healthy_midlate_override = (
        (midlate_transition_activation > 1e-6)
        & (value_gate > 0.75)
        & (source_recert_persistence > 0.70)
        & (regime_quality_gate > 0.70)
    ).to(dtype=reference.dtype)
    midlate_bonus_alignment_gate = torch.where(
        healthy_midlate_override > 1e-6,
        torch.maximum(midlate_bonus_alignment_gate, torch.full_like(reference, 0.95)),
        midlate_bonus_alignment_gate,
    ).clamp(0.0, 1.0)
    source_recert_gate_midlate_aligned = (
        source_recert_gate_pure * midlate_bonus_alignment_gate
    ).clamp(0.0, 1.0)
    prehold_deadzone = (
        (late_gate > 1e-6)
        & (hold_window_activation < 1e-6)
    ).to(dtype=reference.dtype)
    prehold_value_trust = torch.sqrt(
        (value_gate * regime_quality_gate).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    prehold_value_alarm = (1.0 - prehold_value_trust).clamp(0.0, 1.0)
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
    prehold_consistency_gate = (
        1.0
        - prehold_deadzone
        * torch.sqrt(prehold_value_alarm)
        * prehold_actor_mismatch_alarm
    ).clamp(0.20, 1.0)
    healthy_prehold_override = (
        (prehold_deadzone > 1e-6)
        & (prehold_value_trust > 0.70)
        & (prehold_actor_mismatch_alarm < 0.10)
    ).to(dtype=reference.dtype)
    prehold_consistency_gate = torch.where(
        healthy_prehold_override > 1e-6,
        torch.maximum(
            prehold_consistency_gate,
            torch.full_like(reference, 0.95),
        ),
        prehold_consistency_gate,
    ).clamp(0.20, 1.0)
    hold_health = torch.minimum(source_recert_persistence, hold_persistence_gate).clamp(0.0, 1.0)
    release_overhang = (hold_release_pressure - hold_health).clamp(0.0, 1.0)
    value_coupling_health = torch.sqrt((value_gate * hold_health).clamp(0.0, 1.0)).clamp(
        0.0, 1.0
    )
    bonus_value_coupling_gate = (
        1.0
        - late_gate
        * torch.sqrt(release_overhang)
        * (1.0 - value_coupling_health)
    ).clamp(0.0, 1.0)
    healthy_bonus_override = (
        (late_gate > 1e-6)
        & (value_gate > 0.75)
        & (hold_persistence_gate > 0.65)
        & (hold_release_pressure < 0.25)
    ).to(dtype=reference.dtype)
    bonus_value_coupling_gate = torch.where(
        healthy_bonus_override > 1e-6,
        torch.maximum(bonus_value_coupling_gate, torch.full_like(reference, 0.95)),
        bonus_value_coupling_gate,
    ).clamp(0.0, 1.0)
    source_recert_gate_composite = (
        source_recert_gate_midlate_aligned * bonus_value_coupling_gate
    ).clamp(0.0, 1.0)
    bootstrap_external_authority_bonus_midlate_aligned = (
        source_recert_gate_midlate_aligned * bootstrap_external_authority_bonus_prefinal
    ).clamp(0.0, 1.0)
    bootstrap_external_authority_bonus_postrecert = (
        source_recert_gate_composite * bootstrap_external_authority_bonus_prefinal
    ).clamp(0.0, 1.0)
    return (
        source_recert_alarm.detach(),
        source_recert_persistence.detach(),
        source_recert_gate_pure.detach(),
        source_recert_gate_composite.detach(),
        bootstrap_external_authority_prefinal.detach(),
        bootstrap_external_authority_bonus_prefinal.detach(),
        bootstrap_external_authority_bonus_midlate_aligned.detach(),
        bootstrap_external_authority_bonus_postrecert.detach(),
        midlate_transition_activation.detach(),
        midlate_bonus_alignment_gate.detach(),
        bonus_value_coupling_gate.detach(),
        prehold_consistency_gate.detach(),
    )


def compute_bootstrap_authority_source_replacement_contract(
    *,
    bootstrap_external_authority_raw: Tensor,
    bootstrap_external_authority_floor: Tensor,
    bootstrap_anchor_valid_mask: Tensor,
    critic_contract_bootstrap_raw_vs_clean_gap: Tensor,
    critic_contract_bootstrap_regime_quality_gate: Union[float, Tensor] = 1.0,
    reference_real_reward_agreement: Union[float, Tensor],
    behavior_policy_task_cert_real_recovery: Tensor,
    behavior_policy_task_cert_support_mask: Tensor,
    behavior_policy_task_cert_imag_gate: Tensor,
    task_corridor_gate: Tensor,
    critic_contract_task_degradation: Union[float, Tensor],
    critic_contract_bootstrap_late_gate: Union[float, Tensor],
    corridor_semantic_corridor_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    reference = contract_tensor_like(
        bootstrap_external_authority_floor.detach(),
        bootstrap_external_authority_floor,
        0.0,
    ).clamp(0.0, 1.0)
    raw_authority = contract_tensor_like(reference, bootstrap_external_authority_raw, 0.0).clamp(
        0.0, 1.0
    )
    floor_authority = contract_tensor_like(reference, bootstrap_external_authority_floor, 0.0).clamp(
        0.0, 1.0
    )
    anchor_valid_mask = contract_tensor_like(reference, bootstrap_anchor_valid_mask, 0.0).clamp(
        0.0, 1.0
    )
    raw_vs_clean_gap = contract_tensor_like(
        reference,
        critic_contract_bootstrap_raw_vs_clean_gap,
        0.0,
    ).clamp(min=0.0)
    if isinstance(critic_contract_bootstrap_regime_quality_gate, Tensor):
        regime_quality_gate = contract_tensor_like(
            reference,
            critic_contract_bootstrap_regime_quality_gate,
            1.0,
        ).clamp(0.0, 1.0)
    else:
        regime_quality_gate = torch.full_like(
            reference,
            min(1.0, max(0.0, float(critic_contract_bootstrap_regime_quality_gate))),
        )
    if isinstance(reference_real_reward_agreement, Tensor):
        reward_truth = contract_tensor_like(reference, reference_real_reward_agreement, 1.0).clamp(
            0.0, 1.0
        )
    else:
        reward_truth = torch.full_like(
            reference,
            min(1.0, max(0.0, float(reference_real_reward_agreement))),
        )
    real_recovery = contract_tensor_like(
        reference,
        behavior_policy_task_cert_real_recovery,
        1.0,
    ).clamp(0.0, 1.0)
    support_mask = contract_tensor_like(
        reference,
        behavior_policy_task_cert_support_mask,
        1.0,
    ).clamp(0.0, 1.0)
    imag_gate = contract_tensor_like(reference, behavior_policy_task_cert_imag_gate, 1.0).clamp(
        0.0, 1.0
    )
    task_gate = contract_tensor_like(reference, task_corridor_gate, 1.0).clamp(0.0, 1.0)
    corridor_mask = contract_tensor_like(reference, corridor_semantic_corridor_mask, 0.0).clamp(
        0.0, 1.0
    )
    if isinstance(critic_contract_task_degradation, Tensor):
        task_degradation = contract_tensor_like(
            reference,
            critic_contract_task_degradation,
            0.0,
        ).clamp(0.0, 1.0)
    else:
        task_degradation = torch.full_like(
            reference,
            min(1.0, max(0.0, float(critic_contract_task_degradation))),
        )
    if isinstance(critic_contract_bootstrap_late_gate, Tensor):
        late_gate = contract_tensor_like(
            reference,
            critic_contract_bootstrap_late_gate,
            0.0,
        ).clamp(0.0, 1.0)
    else:
        late_gate = torch.full_like(
            reference,
            min(1.0, max(0.0, float(critic_contract_bootstrap_late_gate))),
        )

    bootstrap_external_authority_prefinal = torch.maximum(raw_authority, floor_authority).clamp(
        0.0, 1.0
    )
    bootstrap_external_authority_bonus_prefinal = (
        bootstrap_external_authority_prefinal - floor_authority
    ).clamp(min=0.0, max=1.0)
    source_reward_semantic_truth = torch.maximum(reward_truth, real_recovery).clamp(0.0, 1.0)
    source_semantic_alignment = torch.maximum(imag_gate, task_gate).clamp(0.0, 1.0)
    source_support = torch.where(
        corridor_mask > 1e-6,
        support_mask,
        torch.ones_like(reference),
    ).clamp(0.0, 1.0)
    late_only_alarm = torch.maximum(
        (1.0 - source_reward_semantic_truth).clamp(0.0, 1.0),
        torch.maximum((1.0 - source_semantic_alignment).clamp(0.0, 1.0), task_degradation),
    ).clamp(0.0, 1.0)
    source_late_window_activation = ((late_gate - 0.8) / 0.2).clamp(0.0, 1.0)
    source_bonus_certification_gate = (
        1.0 - source_late_window_activation * late_only_alarm
    ).clamp(0.0, 1.0)
    source_certification = torch.minimum(
        source_reward_semantic_truth,
        source_semantic_alignment,
    ).clamp(0.0, 1.0)
    dense_only_mask = (1.0 - anchor_valid_mask).clamp(0.0, 1.0)
    dense_gap_quality = (2.5 / (2.5 + raw_vs_clean_gap)).clamp(0.0, 1.0)
    anchor_gap_quality = (6.0 / (6.0 + raw_vs_clean_gap)).clamp(0.0, 1.0)
    source_quality = torch.where(
        dense_only_mask > 1e-6,
        torch.minimum(source_support, dense_gap_quality),
        (0.5 + 0.5 * torch.minimum(source_support, anchor_gap_quality)).clamp(0.0, 1.0),
    ).clamp(0.0, 1.0)
    pre_late_truth_shortfall = ((0.35 - source_reward_semantic_truth) / 0.35).clamp(0.0, 1.0)
    pre_late_alignment_shortfall = ((0.8 - source_semantic_alignment) / 0.2).clamp(
        0.0, 1.0
    )
    pre_late_quality_shortfall = (1.0 - source_quality).clamp(0.0, 1.0)
    pre_late_bad_source_activation = torch.pow(
        (
            pre_late_truth_shortfall
            * pre_late_alignment_shortfall
            * pre_late_quality_shortfall
        ).clamp(0.0, 1.0),
        1.0 / 3.0,
    ).clamp(0.0, 1.0)
    pre_late_quality_guard = (
        1.0
        - pre_late_bad_source_activation
    ).clamp(0.0, 1.0)
    healthy_anchor_override = (
        (anchor_valid_mask > 1e-6)
        & (source_semantic_alignment > 0.8)
        & (source_support > 0.7)
    ).to(dtype=reference.dtype)
    anchor_soft_regime_floor = (
        0.25 + 0.75 * source_certification
    ).clamp(0.0, 1.0)
    effective_regime_gate = torch.where(
        healthy_anchor_override > 1e-6,
        torch.maximum(regime_quality_gate, torch.full_like(reference, 0.9)),
        torch.where(
            anchor_valid_mask > 1e-6,
            torch.maximum(regime_quality_gate, anchor_soft_regime_floor),
            regime_quality_gate,
        ),
    ).clamp(0.0, 1.0)
    pre_late_quality_guard = (
        pre_late_quality_guard * effective_regime_gate
    ).clamp(0.0, 1.0)
    strict_bonus_multiplier = (
        source_support
        * source_certification
        * source_bonus_certification_gate
    ).clamp(0.0, 1.0)
    late_bonus_coverage_persistence = (
        source_support
        * torch.sqrt(source_reward_semantic_truth.clamp(0.0, 1.0))
        * torch.sqrt(source_semantic_alignment.clamp(0.0, 1.0))
    ).clamp(0.0, 1.0)
    strict_bonus_multiplier = torch.maximum(
        strict_bonus_multiplier,
        late_bonus_coverage_persistence,
    ).clamp(0.0, 1.0)
    source_bonus_multiplier_prerecert = (
        (1.0 - source_late_window_activation) * pre_late_quality_guard
        + source_late_window_activation * strict_bonus_multiplier
    ).clamp(0.0, 1.0)
    bootstrap_external_authority_bonus_source_certified = (
        source_bonus_multiplier_prerecert * bootstrap_external_authority_bonus_prefinal
    ).clamp(0.0, 1.0)
    bootstrap_external_authority_bonus_source_rejected = (
        bootstrap_external_authority_bonus_prefinal
        - bootstrap_external_authority_bonus_source_certified
    ).clamp(min=0.0, max=1.0)
    bootstrap_external_authority_source_replaced = (
        floor_authority + bootstrap_external_authority_bonus_source_certified
    ).clamp(min=0.0, max=1.0).clamp(0.0, 1.0)
    return (
        source_reward_semantic_truth.detach(),
        source_semantic_alignment.detach(),
        source_late_window_activation.detach(),
        source_bonus_certification_gate.detach(),
        bootstrap_external_authority_source_replaced.detach(),
        source_bonus_multiplier_prerecert.detach(),
        bootstrap_external_authority_bonus_source_certified.detach(),
        bootstrap_external_authority_bonus_source_rejected.detach(),
    )


def compute_post_transition_retention_capture(
    *,
    late_gate_mean: float,
    active_hold_state: float,
    post_hold_state: float,
    source_hold_seed_mean: float,
    source_hold_health_mean: float,
    source_hold_retention_mean: float,
    source_hold_release_pressure_mean: float,
    prev_post_transition_certified_floor_state: float,
) -> RetentionCaptureDecision:
    bounded_late_gate = min(1.0, max(0.0, float(late_gate_mean)))
    window_activation = min(1.0, max(0.0, (bounded_late_gate - 0.80) / 0.20))
    current_authority = min(1.0, max(0.0, max(float(post_hold_state), float(source_hold_retention_mean))))
    carry_decay = 0.90 * (1.0 - 0.25 * float(source_hold_release_pressure_mean))
    carry_decay = min(1.0, max(1e-6, carry_decay))
    historical_authority = min(
        1.0,
        max(0.0, float(prev_post_transition_certified_floor_state) / carry_decay),
    )
    capture_support = min(
        1.0,
        max(0.0, float(source_hold_seed_mean) * float(source_hold_health_mean)),
    )
    historical_gap = max(0.0, historical_authority - current_authority)
    historical_gap_support_ratio = (
        min(1.0, capture_support / historical_gap) if historical_gap > 1e-6 else 0.0
    )
    health_supported_gate = min(
        1.0,
        max(
            capture_support,
            capture_support
            + float(source_hold_health_mean) * window_activation
            + 0.25 * historical_gap,
        ),
    )
    release_gate = min(
        1.0,
        max(0.0, 1.0 - 0.5 * float(source_hold_release_pressure_mean)),
    )
    historical_takeover_gate = min(
        1.0,
        max(0.0, health_supported_gate * release_gate),
    )
    certified_capture_source = current_authority + historical_takeover_gate * historical_gap
    if (
        window_activation > 0.0
        and historical_gap > 1e-6
        and historical_gap_support_ratio > 0.0
    ):
        # Once live support can cover the full historical gap, post-transition
        # capture should acknowledge the carried historical source instead of
        # collapsing it back to current authority.
        supported_historical_source = min(
            historical_authority,
            current_authority + historical_gap_support_ratio * historical_gap,
        )
        certified_capture_source = max(
            certified_capture_source,
            supported_historical_source,
        )
    if float(active_hold_state) > 0.0:
        active_anchor = max(current_authority, 0.5 * float(active_hold_state))
        active_gap = max(0.0, historical_authority - active_anchor)
        certified_capture_source = max(
            certified_capture_source,
            active_anchor + historical_takeover_gate * active_gap,
        )
    certified_capture_source = min(1.0, max(current_authority, certified_capture_source))
    dominance_margin = float(max(0.0, certified_capture_source - current_authority))
    dominance_margin_floor = 1e-2
    if (
        window_activation > 0.0
        and historical_gap > 1e-6
        and historical_gap_support_ratio >= 1.0
    ):
        dominance_margin_floor = min(
            1e-2,
            max(5e-4, 0.35 * historical_gap),
        )
    if dominance_margin < dominance_margin_floor:
        certified_capture_source = float(current_authority)
        dominance_margin = 0.0
    is_historical_winner = dominance_margin > 1e-6 and historical_authority > current_authority
    dominant_source = "historical" if is_historical_winner else "current"
    release_blocked = (
        historical_authority > current_authority
        and float(source_hold_release_pressure_mean) > float(source_hold_health_mean)
        and not is_historical_winner
    )
    return RetentionCaptureDecision(
        window_activation=float(window_activation),
        current_authority=float(current_authority),
        historical_authority=float(historical_authority),
        capture_support=float(capture_support),
        historical_takeover_gate=float(historical_takeover_gate),
        certified_capture_source=float(certified_capture_source),
        dominant_source=dominant_source,
        dominance_margin=float(dominance_margin),
        is_historical_winner=bool(is_historical_winner),
        release_blocked=bool(release_blocked),
    )


def compute_bootstrap_trigger_entry_contract(
    *,
    external_eval_best_mean: float,
    external_eval_last_mean: float,
    real_reward_degradation: Union[float, Tensor],
    real_task_cert_gate: Union[float, Tensor],
    real_behavior_action_switch_rate: float,
    real_behavior_action_oscillation_rate: float,
    behavior_action_switch_rate_preinflation: float,
    behavior_action_oscillation_rate_preinflation: float,
    critic_contract_bootstrap_eval_gate: float,
    bootstrap_step_gate: float,
    bootstrap_precontact_step_gate: float,
    critic_contract_semantic_pressure: Tensor,
    critic_anchor_confidence: Tensor,
    critic_contract_bootstrap_raw_vs_clean_gap: Tensor,
    anchor_bootstrap_values_next: Tensor,
    online_advantages: Tensor,
    returns: Tensor,
    corridor_semantic_corridor_mask: Tensor,
) -> tuple[float, float, float, float, float, Tensor, Tensor, Tensor, Tensor, float, float, float]:
    reference = corridor_semantic_corridor_mask.detach()
    corridor_mask = contract_tensor_like(reference, corridor_semantic_corridor_mask, 0.0).clamp(0.0, 1.0)
    semantic_pressure = contract_tensor_like(reference, critic_contract_semantic_pressure, 0.0).clamp(0.0, 1.0)
    anchor_confidence = contract_tensor_like(reference, critic_anchor_confidence, 0.0).clamp(0.0, 1.0)
    if isinstance(critic_contract_bootstrap_raw_vs_clean_gap, Tensor) and critic_contract_bootstrap_raw_vs_clean_gap.shape == reference.shape:
        raw_vs_clean_gap = critic_contract_bootstrap_raw_vs_clean_gap.detach().to(
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        raw_vs_clean_gap = torch.zeros_like(reference)
    if isinstance(anchor_bootstrap_values_next, Tensor) and anchor_bootstrap_values_next.shape == reference.shape:
        anchor_bootstrap = anchor_bootstrap_values_next.detach().to(
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        anchor_bootstrap = torch.zeros_like(reference)
    if isinstance(online_advantages, Tensor) and online_advantages.shape == reference.shape:
        online_adv = online_advantages.detach().to(device=reference.device, dtype=reference.dtype)
    else:
        online_adv = torch.zeros_like(reference)
    if isinstance(returns, Tensor) and returns.shape == reference.shape:
        returns_value = returns.detach().to(device=reference.device, dtype=reference.dtype)
    else:
        returns_value = torch.zeros_like(reference)
    if isinstance(real_reward_degradation, Tensor):
        reward_degradation_tensor = contract_tensor_like(reference, real_reward_degradation, 0.0).clamp(0.0, 1.0)
        bootstrap_real_reward_degradation = float(_masked_mean_tensor(reward_degradation_tensor, corridor_mask).item())
    else:
        bootstrap_real_reward_degradation = min(1.0, max(0.0, float(real_reward_degradation)))
    if isinstance(real_task_cert_gate, Tensor):
        real_task_gate_tensor = contract_tensor_like(reference, real_task_cert_gate, 1.0).clamp(0.0, 1.0)
        real_task_degradation_tensor = (1.0 - real_task_gate_tensor).clamp(0.0, 1.0)
        bootstrap_real_task_degradation = float(_masked_mean_tensor(real_task_degradation_tensor, corridor_mask).item())
    else:
        bootstrap_real_task_degradation = min(1.0, max(0.0, 1.0 - float(real_task_cert_gate)))

    bootstrap_eval_drawdown = 0.0
    if external_eval_best_mean > 0.0 and math.isfinite(external_eval_best_mean) and math.isfinite(external_eval_last_mean):
        bootstrap_eval_drawdown = min(
            1.0,
            max(
                0.0,
                (external_eval_best_mean - external_eval_last_mean) / max(external_eval_best_mean, 1e-6),
            ),
        )
    bootstrap_kinematic_switch = min(1.0, max(0.0, float(real_behavior_action_switch_rate) - float(behavior_action_switch_rate_preinflation)))
    bootstrap_kinematic_osc = min(1.0, max(0.0, float(real_behavior_action_oscillation_rate) - float(behavior_action_oscillation_rate_preinflation)))
    bootstrap_kinematic_degradation = min(1.0, max(0.0, 0.5 * (bootstrap_kinematic_switch + bootstrap_kinematic_osc)))
    critic_contract_task_degradation = min(
        1.0,
        max(
            0.0,
            bootstrap_eval_drawdown,
            bootstrap_kinematic_degradation,
            bootstrap_real_reward_degradation,
            bootstrap_real_task_degradation,
        ),
    )
    critic_contract_bootstrap_release_guard = (
        corridor_mask
        * anchor_confidence
        * raw_vs_clean_gap
        / (raw_vs_clean_gap + anchor_bootstrap.abs() + 1.0)
    ).clamp(0.0, 1.0).detach()
    critic_contract_bootstrap_release_guard_mean = float(_masked_mean_tensor(critic_contract_bootstrap_release_guard, corridor_mask).item())
    bootstrap_negative_adv_excess = (corridor_mask * (-online_adv).clamp(min=0.0)).detach()
    critic_contract_bootstrap_negative_adv_pressure = (
        bootstrap_negative_adv_excess
        / (bootstrap_negative_adv_excess + returns_value.abs() + 1.0)
    ).clamp(0.0, 1.0).detach()
    critic_contract_bootstrap_negative_adv_pressure_mean = float(
        _masked_mean_tensor(critic_contract_bootstrap_negative_adv_pressure, corridor_mask).item()
    )
    bootstrap_task_degradation_tensor = (corridor_mask * critic_contract_task_degradation).clamp(0.0, 1.0).detach()
    bootstrap_return_anchor_gap = (
        corridor_mask
        * (returns_value - anchor_bootstrap).abs()
        / (returns_value.abs() + anchor_bootstrap.abs() + 1.0)
    ).clamp(0.0, 1.0).detach()
    bootstrap_raw_gap_health = (6.0 / (6.0 + raw_vs_clean_gap)).clamp(0.0, 1.0).detach()
    bootstrap_return_alignment_health = (
        1.0 - bootstrap_return_anchor_gap
    ).clamp(0.0, 1.0).detach()
    bootstrap_trigger_regime_health = (
        1.0
        - torch.maximum(
            critic_contract_bootstrap_release_guard,
            torch.maximum(
                critic_contract_bootstrap_negative_adv_pressure,
                bootstrap_task_degradation_tensor,
            ),
        )
    ).clamp(0.0, 1.0).detach()
    bootstrap_semantic_regime_health = (
        1.0 - semantic_pressure * (1.0 - bootstrap_raw_gap_health)
    ).clamp(0.0, 1.0).detach()
    critic_contract_bootstrap_regime_quality_gate = torch.sqrt(
        (
            bootstrap_raw_gap_health
            * bootstrap_return_alignment_health
            * bootstrap_trigger_regime_health
            * bootstrap_semantic_regime_health
        ).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0).detach()
    critic_contract_bootstrap_trigger_surface = torch.maximum(
        semantic_pressure,
        torch.maximum(
            critic_contract_bootstrap_release_guard,
            torch.maximum(
                critic_contract_bootstrap_negative_adv_pressure,
                bootstrap_task_degradation_tensor,
            ),
        ),
    ).clamp(0.0, 1.0).detach()
    critic_contract_bootstrap_trigger_surface_mean = float(
        _masked_mean_tensor(critic_contract_bootstrap_trigger_surface, corridor_mask).item()
    )
    critic_contract_bootstrap_trigger_gate = min(
        1.0,
        max(
            0.0,
            float(critic_contract_bootstrap_eval_gate),
            critic_contract_task_degradation,
            critic_contract_bootstrap_release_guard_mean,
            critic_contract_bootstrap_negative_adv_pressure_mean,
            critic_contract_bootstrap_trigger_surface_mean,
        ),
    )
    critic_contract_bootstrap_late_gate = min(1.0, max(0.0, float(bootstrap_step_gate) * critic_contract_bootstrap_trigger_gate))
    critic_contract_bootstrap_precontact_gate = min(
        1.0,
        max(0.0, float(bootstrap_precontact_step_gate) * critic_contract_bootstrap_trigger_gate),
    )
    return (
        float(bootstrap_eval_drawdown),
        float(bootstrap_kinematic_degradation),
        float(bootstrap_real_reward_degradation),
        float(bootstrap_real_task_degradation),
        float(critic_contract_task_degradation),
        critic_contract_bootstrap_release_guard.detach(),
        critic_contract_bootstrap_negative_adv_pressure.detach(),
        critic_contract_bootstrap_trigger_surface.detach(),
        critic_contract_bootstrap_regime_quality_gate.detach(),
        float(critic_contract_bootstrap_trigger_gate),
        float(critic_contract_bootstrap_late_gate),
        float(critic_contract_bootstrap_precontact_gate),
    )
