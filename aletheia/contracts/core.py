from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Optional

import torch
from torch import Tensor

__all__ = [
    "ConsumerContractView",
    "EXTERNAL_CONTRACT_SOURCES",
    "GEOMETRY_CERTIFICATION_ALIASES",
    "TASK_CERTIFICATION_ALIASES",
    "SemanticContract",
    "SemanticArbiterDecision",
    "SemanticArbiter",
    "bounded_tensor_like",
    "contract_tensor_like",
    "contract_certifications",
    "make_semantic_contract",
    "select_consumer_contract_view",
]


EXTERNAL_CONTRACT_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "eval_ground_truth",
        "mc_anchor",
        "replay_mc",
        "replay_suffix",
        "short_return_anchor",
    }
)

GEOMETRY_CERTIFICATION_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "behavior_registry",
        "geometry_cert",
        "geometry_corridor",
        "geometry_registry",
    }
)

TASK_CERTIFICATION_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "mc_anchor",
        "reward_agreement",
        "short_return_consistency",
        "task_cert",
        "task_corridor",
        "task_registry",
    }
)


def _validate_tensor_shape(name: str, value: Tensor, expected_shape: torch.Size) -> None:
    if tuple(value.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} shape {tuple(value.shape)} does not match value shape {tuple(expected_shape)}."
        )


def _validate_bounded_tensor(name: str, value: Tensor) -> None:
    numeric = value.to(dtype=torch.float32)
    if not torch.isfinite(numeric).all():
        raise ValueError(f"{name} must be finite.")
    if not torch.all((numeric >= 0.0) & (numeric <= 1.0)).item():
        raise ValueError(f"{name} must stay within [0, 1].")


def _normalize_certification(name: str) -> str:
    return str(name).strip()


def _certification_channel(name: str) -> str:
    normalized = _normalize_certification(name).lower().replace("-", "_")
    if normalized.startswith("geometry:") or normalized in GEOMETRY_CERTIFICATION_ALIASES:
        return "geometry"
    if normalized.startswith("task:") or normalized in TASK_CERTIFICATION_ALIASES:
        return "task"
    return "other"


def bounded_tensor_like(
    reference: Tensor,
    value: Optional[Any],
    default: float,
) -> Tensor:
    resolved_scalar: Optional[float] = None
    if isinstance(value, Tensor):
        if value.shape == reference.shape:
            resolved = value.detach().to(device=reference.device, dtype=reference.dtype)
            return resolved.clamp(0.0, 1.0)
        if value.numel() == 1:
            resolved_scalar = float(value.detach().item())
    elif value is not None:
        try:
            resolved_scalar = float(value)
        except (TypeError, ValueError):
            resolved_scalar = None
    if resolved_scalar is None:
        resolved = torch.full_like(reference, float(default))
    else:
        resolved = torch.full_like(reference, resolved_scalar)
    return resolved.clamp(0.0, 1.0)


def contract_tensor_like(
    reference: Tensor,
    value: Optional[Tensor],
    default: float,
) -> Tensor:
    return bounded_tensor_like(reference, value, default)


def contract_certifications(
    *,
    geometry_signal: Optional[Tensor] = None,
    task_signal: Optional[Tensor] = None,
    registry_signal: Optional[Tensor] = None,
) -> tuple[str, ...]:
    certs: list[str] = []
    if isinstance(geometry_signal, Tensor) and geometry_signal.numel() > 0:
        if float(geometry_signal.detach().float().mean().item()) > 0.0:
            certs.append("geometry_corridor")
    if isinstance(task_signal, Tensor) and task_signal.numel() > 0:
        if float(task_signal.detach().float().mean().item()) >= 0.5:
            certs.append("task_corridor")
    if isinstance(registry_signal, Tensor) and registry_signal.numel() > 0:
        if float(registry_signal.detach().float().mean().item()) > 0.0:
            certs.append("behavior_registry")
    return tuple(certs)


@dataclass(frozen=True)
class SemanticContract:
    value: Tensor
    source: str
    coverage: Tensor
    confidence: Tensor
    authority: Tensor
    trust: Tensor
    task_agreement: Tensor
    registry_support: Tensor
    semantic_debt: Tensor
    freshness: int = 0
    certified_by: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized_source = str(self.source).strip()
        if not normalized_source:
            raise ValueError("source must be a non-empty string.")
        object.__setattr__(self, "source", normalized_source)

        if self.freshness < 0:
            raise ValueError("freshness must be >= 0.")

        expected_shape = self.value.shape
        for name, tensor in (
            ("coverage", self.coverage),
            ("confidence", self.confidence),
            ("authority", self.authority),
            ("trust", self.trust),
            ("task_agreement", self.task_agreement),
            ("registry_support", self.registry_support),
            ("semantic_debt", self.semantic_debt),
        ):
            _validate_tensor_shape(name, tensor, expected_shape)

        for name, tensor in (
            ("coverage", self.coverage),
            ("confidence", self.confidence),
            ("authority", self.authority),
            ("trust", self.trust),
            ("task_agreement", self.task_agreement),
            ("registry_support", self.registry_support),
            ("semantic_debt", self.semantic_debt),
        ):
            _validate_bounded_tensor(name, tensor)

        normalized_certs = tuple(
            cert for cert in (_normalize_certification(item) for item in self.certified_by) if cert
        )
        object.__setattr__(self, "certified_by", normalized_certs)

    @property
    def is_external_source(self) -> bool:
        normalized = self.source.lower().replace("-", "_")
        return (
            normalized in EXTERNAL_CONTRACT_SOURCES
            or normalized.startswith("replay_")
            or normalized.startswith("eval_")
            or normalized.startswith("external_")
        )

    def has_certification(self, name: str) -> bool:
        normalized = _normalize_certification(name)
        return normalized in self.certified_by

    @property
    def geometry_certifications(self) -> tuple[str, ...]:
        return tuple(
            cert for cert in self.certified_by if _certification_channel(cert) == "geometry"
        )

    @property
    def task_certifications(self) -> tuple[str, ...]:
        return tuple(cert for cert in self.certified_by if _certification_channel(cert) == "task")

    @property
    def is_geometry_certified(self) -> bool:
        return bool(self.geometry_certifications)

    @property
    def is_task_certified(self) -> bool:
        return bool(self.task_certifications)

    def summary(self) -> dict[str, Any]:
        def _mean(value: Tensor) -> float:
            return float(value.to(dtype=torch.float32).mean().item())

        return {
            "source": self.source,
            "shape": tuple(self.value.shape),
            "coverage_mean": _mean(self.coverage),
            "confidence_mean": _mean(self.confidence),
            "authority_mean": _mean(self.authority),
            "trust_mean": _mean(self.trust),
            "task_agreement_mean": _mean(self.task_agreement),
            "registry_support_mean": _mean(self.registry_support),
            "semantic_debt_mean": _mean(self.semantic_debt),
            "freshness": int(self.freshness),
            "certified_by": self.certified_by,
            "geometry_certified": self.is_geometry_certified,
            "task_certified": self.is_task_certified,
        }

    def metric_summary(self, prefix: str) -> dict[str, float]:
        summary = self.summary()
        return {
            f"{prefix}_value_mean": float(self.value.detach().to(dtype=torch.float32).mean().item()),
            f"{prefix}_coverage_mean": float(summary["coverage_mean"]),
            f"{prefix}_confidence_mean": float(summary["confidence_mean"]),
            f"{prefix}_authority_mean": float(summary["authority_mean"]),
            f"{prefix}_trust_mean": float(summary["trust_mean"]),
            f"{prefix}_task_agreement_mean": float(summary["task_agreement_mean"]),
            f"{prefix}_registry_support_mean": float(summary["registry_support_mean"]),
            f"{prefix}_semantic_debt_mean": float(summary["semantic_debt_mean"]),
            f"{prefix}_freshness": float(summary["freshness"]),
            f"{prefix}_geometry_certified": float(summary["geometry_certified"]),
            f"{prefix}_task_certified": float(summary["task_certified"]),
            f"{prefix}_is_external": float(self.is_external_source),
        }


@dataclass(frozen=True)
class ConsumerContractView:
    source: str
    authority_mean: float
    trust_mean: float
    task_certified: bool
    geometry_certified: bool
    registry_support_mean: float
    semantic_debt_mean: float
    is_external: bool
    score: float

    @classmethod
    def from_contract(cls, contract: SemanticContract) -> "ConsumerContractView":
        authority_mean = float(contract.authority.detach().to(dtype=torch.float32).mean().item())
        trust_mean = float(contract.trust.detach().to(dtype=torch.float32).mean().item())
        registry_support_mean = float(
            contract.registry_support.detach().to(dtype=torch.float32).mean().item()
        )
        semantic_debt_mean = float(
            contract.semantic_debt.detach().to(dtype=torch.float32).mean().item()
        )
        task_certified = bool(contract.is_task_certified)
        geometry_certified = bool(contract.is_geometry_certified)
        is_external = bool(contract.is_external_source)
        score = (
            (4.0 if task_certified else 0.0)
            + (1.0 if geometry_certified else 0.0)
            + 2.0 * authority_mean
            + 2.0 * trust_mean
            + registry_support_mean
            - semantic_debt_mean
            + (0.5 if is_external else 0.0)
        )
        return cls(
            source=contract.source,
            authority_mean=authority_mean,
            trust_mean=trust_mean,
            task_certified=task_certified,
            geometry_certified=geometry_certified,
            registry_support_mean=registry_support_mean,
            semantic_debt_mean=semantic_debt_mean,
            is_external=is_external,
            score=float(score),
        )

    def metric_summary(self, prefix: str) -> dict[str, float | str]:
        return {
            f"{prefix}_source": self.source,
            f"{prefix}_authority_mean": float(self.authority_mean),
            f"{prefix}_trust_mean": float(self.trust_mean),
            f"{prefix}_task_certified": float(self.task_certified),
            f"{prefix}_geometry_certified": float(self.geometry_certified),
            f"{prefix}_registry_support_mean": float(self.registry_support_mean),
            f"{prefix}_semantic_debt_mean": float(self.semantic_debt_mean),
            f"{prefix}_is_external": float(self.is_external),
            f"{prefix}_score": float(self.score),
        }


def make_semantic_contract(
    *,
    value: Tensor,
    source: str,
    coverage: Optional[Tensor] = None,
    confidence: Optional[Tensor] = None,
    authority: Optional[Tensor] = None,
    trust: Optional[Tensor] = None,
    task_agreement: Optional[Tensor] = None,
    registry_support: Optional[Tensor] = None,
    semantic_debt: Optional[Tensor] = None,
    freshness: int = 0,
    certified_by: tuple[str, ...] = (),
) -> SemanticContract:
    reference = value.detach()
    return SemanticContract(
        value=reference,
        source=source,
        coverage=bounded_tensor_like(reference, coverage, 0.0),
        confidence=bounded_tensor_like(reference, confidence, 0.0),
        authority=bounded_tensor_like(reference, authority, 0.0),
        trust=bounded_tensor_like(reference, trust, 1.0),
        task_agreement=bounded_tensor_like(reference, task_agreement, 0.0),
        registry_support=bounded_tensor_like(reference, registry_support, 0.0),
        semantic_debt=bounded_tensor_like(reference, semantic_debt, 0.0),
        freshness=max(0, int(freshness)),
        certified_by=tuple(certified_by),
    )


def select_consumer_contract_view(*contracts: SemanticContract) -> ConsumerContractView:
    if not contracts:
        raise ValueError("select_consumer_contract_view requires at least one contract.")
    views = [ConsumerContractView.from_contract(contract) for contract in contracts]
    return max(views, key=lambda view: (view.score, view.authority_mean, view.trust_mean))


def _contract_flag_tensor(reference: Tensor, enabled: bool) -> Tensor:
    return torch.full_like(reference, 1.0 if enabled else 0.0)


def _contract_freshness_scale(reference: Tensor, freshness: int) -> Tensor:
    freshness_value = max(0.0, float(freshness))
    return torch.full_like(reference, 1.0 / (1.0 + 0.1 * freshness_value))


def _contract_semantic_strength(contract: SemanticContract) -> Tensor:
    reference = contract.value
    coverage_gate = (
        contract.coverage
        if contract.is_external_source
        else (0.5 + 0.5 * contract.coverage)
    ).clamp(0.0, 1.0)
    debt_scale = (1.0 - 0.75 * contract.semantic_debt).clamp(0.0, 1.0)
    freshness_scale = _contract_freshness_scale(reference, contract.freshness)
    task_certified = _contract_flag_tensor(reference, contract.is_task_certified)
    geometry_certified = _contract_flag_tensor(reference, contract.is_geometry_certified)

    support = (
        0.25 * contract.confidence
        + 0.25 * contract.authority
        + 0.20 * contract.trust
        + 0.20 * contract.task_agreement
        + 0.10 * contract.registry_support
        + 0.15 * task_certified
        + 0.05 * geometry_certified
    )
    support = support * coverage_gate * debt_scale * freshness_scale
    if contract.is_geometry_certified and not contract.is_task_certified:
        support = support * 0.5
    return support.clamp(0.0, 1.0)


def _external_baseline_authority(contract: SemanticContract) -> Tensor:
    reference = contract.value
    task_certified = _contract_flag_tensor(reference, contract.is_task_certified)
    baseline = 0.05 * contract.coverage * (
        0.5 * contract.confidence
        + 0.3 * contract.task_agreement
        + 0.2 * task_certified
    )
    baseline = baseline * (1.0 - 0.5 * contract.semantic_debt).clamp(0.0, 1.0)
    if contract.is_geometry_certified and not contract.is_task_certified:
        baseline = baseline * 0.5
    return baseline.clamp(0.0, 1.0)


@dataclass(frozen=True)
class SemanticArbiterDecision:
    external_authority: Tensor
    internal_authority: Tensor
    takeover_floor: Tensor
    modulation_bonus: Tensor
    max_external_authority: Tensor

    def __post_init__(self) -> None:
        expected_shape = self.external_authority.shape
        for name, tensor in (
            ("internal_authority", self.internal_authority),
            ("takeover_floor", self.takeover_floor),
            ("modulation_bonus", self.modulation_bonus),
            ("max_external_authority", self.max_external_authority),
        ):
            _validate_tensor_shape(name, tensor, expected_shape)

        for name, tensor in (
            ("external_authority", self.external_authority),
            ("internal_authority", self.internal_authority),
            ("takeover_floor", self.takeover_floor),
            ("modulation_bonus", self.modulation_bonus),
            ("max_external_authority", self.max_external_authority),
        ):
            _validate_bounded_tensor(name, tensor)

    def summary(self) -> dict[str, float]:
        def _mean(value: Tensor) -> float:
            return float(value.to(dtype=torch.float32).mean().item())

        return {
            "external_authority_mean": _mean(self.external_authority),
            "internal_authority_mean": _mean(self.internal_authority),
            "takeover_floor_mean": _mean(self.takeover_floor),
            "modulation_bonus_mean": _mean(self.modulation_bonus),
            "max_external_authority_mean": _mean(self.max_external_authority),
        }


@dataclass(frozen=True)
class SemanticArbiter:
    def arbitrate_bootstrap(
        self,
        *,
        internal_contract: SemanticContract,
        external_contract: SemanticContract,
        takeover_floor: Tensor,
        modulation_bonus: Tensor,
        max_external_authority: float | Tensor = 1.0,
    ) -> SemanticArbiterDecision:
        expected_shape = internal_contract.value.shape
        _validate_tensor_shape("external_contract.value", external_contract.value, expected_shape)
        _validate_tensor_shape("takeover_floor", takeover_floor, expected_shape)
        _validate_tensor_shape("modulation_bonus", modulation_bonus, expected_shape)

        if isinstance(max_external_authority, Tensor):
            cap_tensor = max_external_authority.to(
                device=internal_contract.value.device,
                dtype=internal_contract.value.dtype,
            )
        else:
            cap_tensor = torch.full_like(
                internal_contract.value,
                float(max_external_authority),
            )
        _validate_tensor_shape("max_external_authority", cap_tensor, expected_shape)
        _validate_bounded_tensor("max_external_authority", cap_tensor)

        semantic_floor = _external_baseline_authority(external_contract)
        clamped_floor = torch.maximum(
            takeover_floor.clamp(0.0, 1.0),
            torch.minimum(semantic_floor, cap_tensor),
        )
        remaining_capacity = (cap_tensor - clamped_floor).clamp(0.0, 1.0)
        external_strength = _contract_semantic_strength(external_contract)
        internal_strength = _contract_semantic_strength(internal_contract)
        semantic_margin = (external_strength - internal_strength).clamp(-1.0, 1.0)
        bonus_gate = (0.5 + semantic_margin).clamp(0.0, 1.0)
        clamped_bonus = modulation_bonus.clamp(0.0, 1.0)
        clamped_bonus = torch.minimum(clamped_bonus, remaining_capacity) * bonus_gate
        external_authority = (clamped_floor + clamped_bonus).clamp(0.0, 1.0)
        internal_authority = (1.0 - external_authority).clamp(0.0, 1.0)

        return SemanticArbiterDecision(
            external_authority=external_authority,
            internal_authority=internal_authority,
            takeover_floor=clamped_floor,
            modulation_bonus=clamped_bonus,
            max_external_authority=cap_tensor,
        )
