from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

from .core import SemanticContract, bounded_tensor_like, make_semantic_contract

__all__ = [
    "EvidenceBundle",
    "RealFeedbackEvidence",
    "build_behavior_policy_evidence",
    "build_bootstrap_value_evidence",
    "build_real_feedback_evidence",
    "compute_reference_reward_agreement",
    "compute_reward_health",
    "make_semantic_contract",
    "resolve_bootstrap_external_eval_feedback_snapshot",
]


@dataclass(frozen=True)
class RealFeedbackEvidence:
    reward_degradation: float
    task_cert_gate: float
    task_cert_alarm: float
    task_cert_recovery: float
    behavior_action_switch_rate: float
    behavior_action_oscillation_rate: float
    available: float
    age_steps: float


@dataclass(frozen=True)
class EvidenceBundle:
    reference: Tensor
    coverage: Tensor
    confidence: Tensor
    authority: Tensor
    trust: Tensor
    task_agreement: Tensor
    registry_support: Tensor
    semantic_debt: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_shape = self.reference.shape
        for name, tensor in (
            ("coverage", self.coverage),
            ("confidence", self.confidence),
            ("authority", self.authority),
            ("trust", self.trust),
            ("task_agreement", self.task_agreement),
            ("registry_support", self.registry_support),
            ("semantic_debt", self.semantic_debt),
        ):
            if tuple(tensor.shape) != tuple(expected_shape):
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} does not match reference shape {tuple(expected_shape)}."
                )

    def as_semantic_contract(
        self,
        *,
        value: Optional[Tensor] = None,
        source: str,
        freshness: int = 0,
        certified_by: tuple[str, ...] = (),
    ) -> SemanticContract:
        return make_semantic_contract(
            value=self.reference if value is None else value,
            source=source,
            coverage=self.coverage,
            confidence=self.confidence,
            authority=self.authority,
            trust=self.trust,
            task_agreement=self.task_agreement,
            registry_support=self.registry_support,
            semantic_debt=self.semantic_debt,
            freshness=freshness,
            certified_by=certified_by,
        )


def build_behavior_policy_evidence(
    *,
    reference: Tensor,
    coverage: Optional[Tensor] = None,
    confidence: Optional[Tensor] = None,
    authority: Optional[Tensor] = None,
    trust: Optional[Tensor] = None,
    task_agreement: Optional[Tensor] = None,
    registry_support: Optional[Tensor] = None,
    semantic_debt: Optional[Tensor] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> EvidenceBundle:
    reference_value = reference.detach()
    return EvidenceBundle(
        reference=reference_value,
        coverage=bounded_tensor_like(reference_value, coverage, 0.0),
        confidence=bounded_tensor_like(reference_value, confidence, 0.0),
        authority=bounded_tensor_like(reference_value, authority, 0.0),
        trust=bounded_tensor_like(reference_value, trust, 1.0),
        task_agreement=bounded_tensor_like(reference_value, task_agreement, 0.0),
        registry_support=bounded_tensor_like(reference_value, registry_support, 0.0),
        semantic_debt=bounded_tensor_like(reference_value, semantic_debt, 0.0),
        metadata=dict(metadata or {}),
    )


def build_bootstrap_value_evidence(
    *,
    reference: Tensor,
    anchor_available: Optional[Tensor] = None,
    row_support: Optional[Tensor] = None,
    dense_surface_gain: Optional[Tensor] = None,
    semantic_debt: Optional[Tensor] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> EvidenceBundle:
    reference_value = reference.detach()
    return EvidenceBundle(
        reference=reference_value,
        coverage=bounded_tensor_like(reference_value, anchor_available, 0.0),
        confidence=bounded_tensor_like(reference_value, row_support, 0.0),
        authority=bounded_tensor_like(reference_value, dense_surface_gain, 0.0),
        trust=bounded_tensor_like(reference_value, row_support, 1.0),
        task_agreement=bounded_tensor_like(reference_value, row_support, 0.0),
        registry_support=bounded_tensor_like(reference_value, anchor_available, 0.0),
        semantic_debt=bounded_tensor_like(reference_value, semantic_debt, 0.0),
        metadata=dict(metadata or {}),
    )


def compute_reward_health(
    latest_reward: float,
    best_reward: float,
) -> float:
    latest = float(latest_reward)
    best = float(best_reward)
    if not math.isfinite(latest) or not math.isfinite(best) or best <= 1e-6:
        return 1.0
    return min(1.0, max(0.0, latest / max(best, 1e-6)))


def compute_reference_reward_agreement(
    real_targets: Optional[Tensor],
    best_reward: float,
    *,
    quantile: float = 0.75,
) -> float:
    if not isinstance(real_targets, Tensor) or real_targets.numel() <= 0:
        return 1.0
    finite_targets = real_targets.detach().reshape(-1).float()
    finite_targets = finite_targets[torch.isfinite(finite_targets)]
    if finite_targets.numel() <= 0:
        return 1.0
    q = min(1.0, max(0.0, float(quantile)))
    if finite_targets.numel() == 1:
        observed_reward = float(finite_targets.item())
    elif q <= 0.0:
        observed_reward = float(finite_targets.mean().item())
    elif q >= 1.0:
        observed_reward = float(finite_targets.max().item())
    else:
        observed_reward = float(torch.quantile(finite_targets, q).item())
    return compute_reward_health(observed_reward, best_reward)


def build_real_feedback_evidence(
    *,
    telemetry: Optional[dict[str, Any]],
    last_step: int,
    global_step: int,
) -> RealFeedbackEvidence:
    data = telemetry or {}
    available = 1.0 if bool(data) else 0.0
    age_steps = (
        float(max(0, int(global_step) - int(last_step)))
        if int(last_step) >= 0 and available > 0.0
        else -1.0
    )
    return RealFeedbackEvidence(
        reward_degradation=min(1.0, max(0.0, float(data.get("real_reward_degradation", 0.0)))),
        task_cert_gate=min(1.0, max(0.0, float(data.get("real_task_cert_gate", 1.0)))),
        task_cert_alarm=min(
            1.0,
            max(
                0.0,
                float(data.get("real_task_cert_alarm", 1.0 - float(data.get("real_task_cert_gate", 1.0)))),
            ),
        ),
        task_cert_recovery=min(
            1.0,
            max(
                0.0,
                float(data.get("real_task_cert_recovery", float(data.get("real_task_cert_gate", 1.0)))),
            ),
        ),
        behavior_action_switch_rate=min(
            1.0,
            max(0.0, float(data.get("real_behavior_action_switch_rate", 0.0))),
        ),
        behavior_action_oscillation_rate=min(
            1.0,
            max(0.0, float(data.get("real_behavior_action_oscillation_rate", 0.0))),
        ),
        available=available,
        age_steps=age_steps,
    )


def resolve_bootstrap_external_eval_feedback_snapshot(
    *,
    bootstrap_external_eval_feedback_telemetry: Optional[dict[str, Any]],
    bootstrap_external_eval_feedback_last_step: int,
    global_step: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    evidence = build_real_feedback_evidence(
        telemetry=bootstrap_external_eval_feedback_telemetry,
        last_step=bootstrap_external_eval_feedback_last_step,
        global_step=global_step,
    )
    return (
        evidence.reward_degradation,
        evidence.task_cert_gate,
        evidence.task_cert_alarm,
        evidence.task_cert_recovery,
        evidence.behavior_action_switch_rate,
        evidence.behavior_action_oscillation_rate,
        evidence.available,
        evidence.age_steps,
    )
