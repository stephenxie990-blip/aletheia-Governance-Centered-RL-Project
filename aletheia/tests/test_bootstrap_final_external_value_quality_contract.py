import sys
import unittest
from pathlib import Path

import torch


_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.contracts.consumers import (
    compute_bootstrap_bonus_consumer_retention_contract,
    compute_bootstrap_final_external_value_assembly_contract,
    compute_bootstrap_final_external_value_quality_contract,
)


class BootstrapFinalExternalValueQualityContractRegressionTests(unittest.TestCase):
    def test_final_external_value_assembly_keeps_consumer_retention_observer_only(self):
        hold_persisted = torch.tensor([[8.5, 7.5]], dtype=torch.float32)
        terminal_truth_source = torch.tensor([[2.0, 2.5]], dtype=torch.float32)
        replaced_value = torch.tensor([[9.0, 8.0]], dtype=torch.float32)
        anchor_value = torch.tensor([[1.5, 1.0]], dtype=torch.float32)
        internal_value = torch.tensor([[7.0, 6.5]], dtype=torch.float32)
        anchor_valid_mask = torch.ones_like(hold_persisted)
        ones = torch.ones_like(hold_persisted)

        sequential_consumer = compute_bootstrap_bonus_consumer_retention_contract(
            bootstrap_external_bonus_value_hold_persisted=hold_persisted,
            bootstrap_external_bonus_terminal_truth_source=terminal_truth_source,
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            reference_real_reward_agreement=torch.full_like(hold_persisted, 0.97),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                hold_persisted, 0.94
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                hold_persisted, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                hold_persisted, 0.98
            ),
            task_corridor_gate=torch.full_like(hold_persisted, 0.99),
            critic_contract_task_degradation=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_bonus_hold_state=0.8,
            critic_contract_bootstrap_source_hold_persistence_gate=torch.full_like(
                hold_persisted, 0.2
            ),
            corridor_semantic_corridor_mask=ones,
        )
        sequential_quality = compute_bootstrap_final_external_value_quality_contract(
            bootstrap_external_bonus_value_consumer_retained=hold_persisted,
            anchor_bootstrap_values_next=anchor_value,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=anchor_valid_mask,
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                hold_persisted, 0.92
            ),
            critic_contract_bootstrap_raw_vs_clean_gap=torch.full_like(
                hold_persisted, 0.25
            ),
            critic_contract_bootstrap_source_reward_semantic_truth=torch.full_like(
                hold_persisted, 0.95
            ),
            critic_contract_bootstrap_source_semantic_alignment=torch.full_like(
                hold_persisted, 0.96
            ),
            actor_contract_task_mismatch=torch.full_like(hold_persisted, 0.04),
            actor_corridor_semantic_inflation_excess=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_source_hold_window_activation=torch.ones_like(
                hold_persisted
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(
                hold_persisted, 0.02
            ),
            actor_task_geom_corridor_disagreement=torch.full_like(
                hold_persisted, 0.2
            ),
        )
        assembled = compute_bootstrap_final_external_value_assembly_contract(
            bootstrap_external_bonus_value_hold_persisted=hold_persisted,
            bootstrap_external_bonus_terminal_truth_source=terminal_truth_source,
            bootstrap_external_bonus_value_source_replaced=replaced_value,
            reference_real_reward_agreement=torch.full_like(hold_persisted, 0.97),
            behavior_policy_task_cert_real_recovery=torch.full_like(
                hold_persisted, 0.94
            ),
            behavior_policy_task_cert_real_alarm=torch.full_like(
                hold_persisted, 0.02
            ),
            behavior_policy_task_cert_imag_gate=torch.full_like(
                hold_persisted, 0.98
            ),
            task_corridor_gate=torch.full_like(hold_persisted, 0.99),
            critic_contract_task_degradation=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_bonus_hold_state=0.8,
            critic_contract_bootstrap_source_hold_persistence_gate=torch.full_like(
                hold_persisted, 0.2
            ),
            corridor_semantic_corridor_mask=ones,
            anchor_bootstrap_values_next=anchor_value,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=anchor_valid_mask,
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                hold_persisted, 0.92
            ),
            critic_contract_bootstrap_raw_vs_clean_gap=torch.full_like(
                hold_persisted, 0.25
            ),
            critic_contract_bootstrap_source_reward_semantic_truth=torch.full_like(
                hold_persisted, 0.95
            ),
            critic_contract_bootstrap_source_semantic_alignment=torch.full_like(
                hold_persisted, 0.96
            ),
            actor_contract_task_mismatch=torch.full_like(hold_persisted, 0.04),
            actor_corridor_semantic_inflation_excess=torch.zeros_like(hold_persisted),
            critic_contract_bootstrap_source_hold_window_activation=torch.ones_like(
                hold_persisted
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(
                hold_persisted, 0.02
            ),
            actor_task_geom_corridor_disagreement=torch.full_like(
                hold_persisted, 0.2
            ),
        )

        self.assertTrue(torch.allclose(assembled.source_consumer_truth, sequential_consumer[0]))
        self.assertTrue(torch.allclose(assembled.source_consumer_alignment, sequential_consumer[1]))
        self.assertTrue(
            torch.allclose(
                assembled.source_consumer_window_activation,
                torch.zeros_like(sequential_consumer[2]),
            )
        )
        self.assertTrue(
            torch.allclose(
                assembled.source_consumer_retention_gate,
                torch.zeros_like(sequential_consumer[3]),
            )
        )
        self.assertTrue(
            torch.allclose(assembled.bonus_value_consumer_retained, hold_persisted)
        )
        self.assertTrue(
            torch.allclose(
                assembled.bonus_value_consumer_delta_abs,
                torch.zeros_like(hold_persisted),
            )
        )
        self.assertTrue(torch.allclose(assembled.regime_quality_gate, sequential_quality[0]))
        self.assertTrue(torch.allclose(assembled.final_value_behavior_health, sequential_quality[1]))
        self.assertTrue(torch.allclose(assembled.prehold_value_failfast_gate, sequential_quality[2]))
        self.assertTrue(torch.allclose(assembled.floor_value_injection_gate, sequential_quality[3]))
        self.assertTrue(torch.allclose(assembled.floor_value_clamped, sequential_quality[4]))
        self.assertTrue(torch.allclose(assembled.floor_value_delta_abs, sequential_quality[5]))
        self.assertTrue(torch.allclose(assembled.final_value_injection_gate, sequential_quality[6]))
        self.assertTrue(torch.allclose(assembled.bonus_value_quality_clamped, sequential_quality[7]))
        self.assertTrue(torch.allclose(assembled.bonus_value_quality_delta_abs, sequential_quality[8]))

    def test_anchor_valid_low_quality_source_keeps_minimum_fallback_gate(self):
        retained = torch.tensor([[18.0, 16.0]], dtype=torch.float32)
        anchor_value = torch.tensor([[10.0, 9.0]], dtype=torch.float32)
        internal_value = torch.tensor([[5.0, 5.5]], dtype=torch.float32)
        anchor_valid = torch.ones_like(retained)

        (
            _regime_quality_gate,
            _behavior_health,
            prehold_value_failfast_gate,
            floor_gate,
            floor_quality_clamped,
            floor_delta_abs,
            final_gate,
            quality_clamped,
            delta_abs,
        ) = compute_bootstrap_final_external_value_quality_contract(
            bootstrap_external_bonus_value_consumer_retained=retained,
            anchor_bootstrap_values_next=anchor_value,
            raw_bootstrap_values_next=internal_value,
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_regime_quality_gate=torch.full_like(
                retained, 0.24
            ),
            critic_contract_bootstrap_raw_vs_clean_gap=torch.full_like(retained, 11.0),
            critic_contract_bootstrap_source_reward_semantic_truth=torch.full_like(
                retained, 0.22
            ),
            critic_contract_bootstrap_source_semantic_alignment=torch.full_like(
                retained, 0.30
            ),
            actor_contract_task_mismatch=0.26,
            actor_corridor_semantic_inflation_excess=8.0,
            critic_contract_bootstrap_late_gate=torch.full_like(retained, 0.35),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                retained
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(retained, 0.24),
            actor_task_geom_corridor_disagreement=torch.full_like(retained, 0.56),
        )

        self.assertTrue(torch.all(prehold_value_failfast_gate < 0.35).item())
        self.assertTrue(torch.all(floor_gate > 0.2).item())
        self.assertTrue(torch.all(final_gate > 0.3).item())
        self.assertTrue(torch.all(final_gate > floor_gate).item())
        self.assertTrue(torch.all(floor_quality_clamped > internal_value).item())
        self.assertTrue(torch.all(floor_quality_clamped < anchor_value).item())
        self.assertTrue(torch.all(quality_clamped > floor_quality_clamped).item())
        self.assertTrue(torch.all(quality_clamped < retained).item())
        self.assertTrue(torch.all(floor_delta_abs >= 0.0).item())
        self.assertTrue(torch.all(delta_abs >= 0.0).item())

    def test_healthy_anchor_valid_source_still_preserved(self):
        retained = torch.tensor([[12.0, 11.0]], dtype=torch.float32)
        anchor_value = torch.tensor([[7.0, 6.0]], dtype=torch.float32)
        anchor_valid = torch.ones_like(retained)
        regime_quality = torch.full_like(retained, 0.92)
        raw_gap = torch.full_like(retained, 3.0)
        truth = torch.full_like(retained, 0.72)
        alignment = torch.full_like(retained, 0.93)

        (
            _regime_quality_gate,
            behavior_health,
            prehold_value_failfast_gate,
            floor_gate,
            floor_quality_clamped,
            floor_delta_abs,
            final_gate,
            quality_clamped,
            delta_abs,
        ) = compute_bootstrap_final_external_value_quality_contract(
            bootstrap_external_bonus_value_consumer_retained=retained,
            anchor_bootstrap_values_next=anchor_value,
            raw_bootstrap_values_next=torch.tensor([[2.5, 2.0]], dtype=torch.float32),
            bootstrap_anchor_valid_mask=anchor_valid,
            critic_contract_bootstrap_regime_quality_gate=regime_quality,
            critic_contract_bootstrap_raw_vs_clean_gap=raw_gap,
            critic_contract_bootstrap_source_reward_semantic_truth=truth,
            critic_contract_bootstrap_source_semantic_alignment=alignment,
            actor_contract_task_mismatch=0.03,
            actor_corridor_semantic_inflation_excess=1.0,
            critic_contract_bootstrap_late_gate=torch.full_like(retained, 0.45),
            critic_contract_bootstrap_source_hold_window_activation=torch.zeros_like(
                retained
            ),
            actor_high_value_but_low_task_fraction=torch.full_like(retained, 0.02),
            actor_task_geom_corridor_disagreement=torch.full_like(retained, 0.32),
        )

        self.assertTrue(torch.all(behavior_health > 0.85).item())
        self.assertTrue(torch.all(prehold_value_failfast_gate >= 0.95).item())
        self.assertTrue(torch.all(floor_gate >= 0.9).item())
        self.assertTrue(torch.all(final_gate >= 0.85).item())
        self.assertTrue(torch.allclose(floor_quality_clamped, anchor_value, atol=1.0))
        self.assertTrue(torch.all(floor_delta_abs < 1.0).item())
        self.assertTrue(torch.allclose(quality_clamped, retained, atol=1.0))
        self.assertTrue(torch.all(delta_abs < 1.0).item())


if __name__ == "__main__":
    unittest.main()
