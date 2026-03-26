from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from torch import Tensor

from .core import bounded_tensor_like, contract_certifications

__all__ = [
    "CertificationState",
    "build_registry_certification",
    "compute_task_certification",
    "compute_task_certification_state",
]


@dataclass(frozen=True)
class CertificationState:
    imag_task_cert: Tensor
    real_task_cert: Tensor
    effective_task_cert: Tensor
    real_task_alarm: Tensor
    real_task_recovery: Tensor
    geometry_support: Tensor
    registry_support: Tensor
    certified_registry_support: Tensor
    labels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def imag_task_gate(self) -> Tensor:
        return self.imag_task_cert

    @property
    def real_task_state(self) -> Tensor:
        return self.real_task_cert

    @property
    def task_cert(self) -> Tensor:
        return self.effective_task_cert


def compute_task_certification(
    *,
    task_gate: Tensor,
    task_confidence: Tensor,
    real_eval_gate: Union[float, Tensor],
    real_reward_agreement: Optional[Union[float, Tensor]] = None,
    prev_real_task_state: Optional[Union[float, Tensor]] = None,
    reward_agreement_available: bool = True,
    recovery_rate: float = 0.5,
    alarm_rate: float = 0.7,
    state_floor: float = 0.0,
    geometry_support: Optional[Tensor] = None,
    registry_support: Optional[Tensor] = None,
) -> CertificationState:
    imag_task_gate = bounded_tensor_like(task_gate, task_gate, 1.0)
    imag_task_confidence = bounded_tensor_like(task_gate, task_confidence, 0.0)
    imag_task_cert = (
        1.0 - imag_task_confidence * (1.0 - imag_task_gate)
    ).clamp(0.0, 1.0)
    if isinstance(real_eval_gate, Tensor):
        real_eval_tensor = bounded_tensor_like(task_gate, real_eval_gate, 1.0)
    else:
        real_eval_tensor = torch.full_like(
            task_gate,
            min(1.0, max(0.0, float(real_eval_gate))),
        )
    if reward_agreement_available and real_reward_agreement is not None:
        if isinstance(real_reward_agreement, Tensor):
            real_reward_tensor = bounded_tensor_like(
                task_gate,
                real_reward_agreement,
                1.0,
            )
        else:
            real_reward_tensor = torch.full_like(
                task_gate,
                min(1.0, max(0.0, float(real_reward_agreement))),
            )
        real_task_alarm = torch.minimum(real_eval_tensor, real_reward_tensor).clamp(
            0.0, 1.0
        )
        real_recovery_basis = torch.sqrt(
            (real_eval_tensor * real_reward_tensor).clamp(0.0, 1.0)
        )
        real_task_recovery = torch.maximum(
            real_task_alarm,
            (imag_task_cert * real_recovery_basis).clamp(0.0, 1.0),
        )
    else:
        real_task_alarm = real_eval_tensor.clamp(0.0, 1.0)
        real_task_recovery = real_task_alarm
    if prev_real_task_state is None:
        prev_real_task_state_tensor = real_task_alarm
    elif isinstance(prev_real_task_state, Tensor):
        prev_real_task_state_tensor = bounded_tensor_like(
            task_gate,
            prev_real_task_state,
            1.0,
        )
    else:
        prev_real_task_state_tensor = torch.full_like(
            task_gate,
            min(1.0, max(0.0, float(prev_real_task_state))),
        )
    rise_rate = min(1.0, max(0.0, float(recovery_rate)))
    fall_rate = min(1.0, max(0.0, float(alarm_rate)))
    state_floor_tensor = torch.full_like(
        task_gate,
        min(1.0, max(0.0, float(state_floor))),
    )
    real_task_state = torch.where(
        real_task_recovery >= prev_real_task_state_tensor,
        prev_real_task_state_tensor
        + rise_rate * (real_task_recovery - prev_real_task_state_tensor),
        prev_real_task_state_tensor
        + fall_rate * (real_task_recovery - prev_real_task_state_tensor),
    )
    real_task_state = torch.maximum(real_task_state, state_floor_tensor)
    real_task_state = torch.minimum(
        real_task_state,
        torch.ones_like(real_task_state),
    )
    task_cert = torch.minimum(imag_task_cert, real_task_state).clamp(0.0, 1.0)
    geometry_support_tensor = bounded_tensor_like(task_gate, geometry_support, 0.0)
    registry_support_tensor = bounded_tensor_like(task_gate, registry_support, 0.0)
    certified_registry_support = torch.minimum(
        geometry_support_tensor,
        task_cert,
    ).clamp(0.0, 1.0)
    return CertificationState(
        imag_task_cert=imag_task_cert.detach(),
        real_task_cert=real_task_state.detach(),
        effective_task_cert=task_cert.detach(),
        real_task_alarm=real_task_alarm.detach(),
        real_task_recovery=real_task_recovery.detach(),
        geometry_support=geometry_support_tensor.detach(),
        registry_support=registry_support_tensor.detach(),
        certified_registry_support=certified_registry_support.detach(),
        labels=contract_certifications(
            geometry_signal=geometry_support_tensor,
            task_signal=task_cert,
            registry_signal=registry_support_tensor,
        ),
    )


def build_registry_certification(
    *,
    geometry_support: Tensor,
    task_cert: Tensor,
    registry_support: Optional[Tensor] = None,
) -> CertificationState:
    geometry_support_tensor = bounded_tensor_like(geometry_support, geometry_support, 0.0)
    task_cert_tensor = bounded_tensor_like(geometry_support_tensor, task_cert, 0.0)
    registry_support_tensor = bounded_tensor_like(
        geometry_support_tensor,
        registry_support,
        0.0,
    )
    certified_registry_support = torch.minimum(
        geometry_support_tensor,
        task_cert_tensor,
    ).clamp(0.0, 1.0)
    return CertificationState(
        imag_task_cert=task_cert_tensor.detach(),
        real_task_cert=task_cert_tensor.detach(),
        effective_task_cert=task_cert_tensor.detach(),
        real_task_alarm=(1.0 - task_cert_tensor).clamp(0.0, 1.0).detach(),
        real_task_recovery=task_cert_tensor.detach(),
        geometry_support=geometry_support_tensor.detach(),
        registry_support=registry_support_tensor.detach(),
        certified_registry_support=certified_registry_support.detach(),
        labels=contract_certifications(
            geometry_signal=geometry_support_tensor,
            task_signal=task_cert_tensor,
            registry_signal=registry_support_tensor,
        ),
    )


def compute_task_certification_state(**kwargs) -> CertificationState:
    return compute_task_certification(**kwargs)
