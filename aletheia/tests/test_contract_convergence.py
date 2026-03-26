import unittest
from types import SimpleNamespace

import torch

from aletheia.aletheia_config import TrainingConfig
from aletheia.aletheia_train import (
    _compute_bootstrap_trigger_entry_contract,
    _compute_bootstrap_task_request_contract,
)
from aletheia.contracts.consumers import (
    compute_bootstrap_bonus_consumer_retention_contract,
)
from aletheia.contracts.certification import compute_task_certification_state
from aletheia.contracts.core import make_semantic_contract, select_consumer_contract_view
from aletheia.contracts.evidence import build_behavior_policy_evidence
from aletheia.training.compensation import (
    MinimalCompensationState,
    canonicalize_compensation_phase,
    make_empty_hold_state,
    restore_minimal_compensation_state,
)
from aletheia.training.runtime_helpers import (
    is_trigger_persistence_handoff_landing_guard_active,
    resolve_persistence_escape_soft_floors,
    resolve_post_entry_soft_negative_adv_profile,
    resolve_post_trigger_quality_state,
)


class TestContractConvergence(unittest.TestCase):
    def test_evidence_bundle_normalizes_tensors_and_scalars(self):
        reference = torch.ones((2, 2), dtype=torch.float32)
        bundle = build_behavior_policy_evidence(
            reference=reference,
            task_agreement=torch.full_like(reference, 0.4),
            metadata={"late_gate_mean": 0.75},
        )

        self.assertAlmostEqual(float(bundle.task_agreement.mean().item()), 0.4, places=6)
        self.assertAlmostEqual(float(bundle.semantic_debt.mean().item()), 0.0, places=6)
        self.assertAlmostEqual(float(bundle.metadata["late_gate_mean"]), 0.75, places=6)

    def test_task_certification_state_builds_recoverable_gate(self):
        task_gate = torch.tensor([[0.9, 0.2]], dtype=torch.float32)
        task_confidence = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        state = compute_task_certification_state(
            task_gate=task_gate,
            task_confidence=task_confidence,
            real_eval_gate=0.5,
            real_reward_agreement=0.25,
            prev_real_task_state=0.8,
            recovery_rate=0.5,
            alarm_rate=0.7,
            state_floor=0.1,
        )

        self.assertEqual(state.task_cert.shape, task_gate.shape)
        self.assertTrue(torch.all(state.real_task_state >= 0.1).item())
        self.assertTrue(torch.all(state.task_cert <= state.imag_task_gate).item())

    def test_minimal_compensation_state_roundtrip_canonical_payload(self):
        payload = {
            "schema_version": 2,
            "compensation_phase": "persistence",
            "continue_cap_state": 0.4,
            "post_trigger_actor_scale_state": 0.7,
            "contact_surface_state": 0.6,
            "bonus_hold_state": {
                "pre_transition": 0.1,
                "active_transition": 0.2,
                "post_transition": 0.3,
            },
            "post_transition_certified_floor_state": 0.55,
            "persistence": {
                "hold_until_step": 12,
                "armed_step": 10,
                "entry_protect_until_step": 14,
                "release_hold_until_step": 16,
            },
            "post_solved": {"hold_until_step": 20},
        }

        state = MinimalCompensationState.from_payload(payload)
        roundtrip = state.to_payload()

        self.assertEqual(state.compensation_phase, "persistence")
        self.assertAlmostEqual(roundtrip["continue_cap_state"], 0.4, places=6)
        self.assertEqual(roundtrip["schema_version"], 2)
        self.assertEqual(roundtrip["compensation_phase"], "persistence")
        self.assertAlmostEqual(
            roundtrip["bonus_hold_state"]["post_transition"],
            0.3,
            places=6,
        )
        self.assertEqual(roundtrip["post_solved"]["hold_until_step"], 20)

    def test_minimal_compensation_state_rejects_unsupported_payload_keys(self):
        payload = {
            "schema_version": 2,
            "compensation_phase": "persistence",
            "bonus_hold_state": {
                "pre_transition": 0.1,
                "active_transition": 0.2,
                "post_transition": 0.3,
            },
            "legacy_phase_name": "persistence",
        }

        with self.assertRaisesRegex(ValueError, "unsupported keys"):
            MinimalCompensationState.from_payload(payload)

    def test_restore_minimal_compensation_state_rejects_non_mapping_payload(self):
        controller = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "mapping"):
            restore_minimal_compensation_state(controller, 0.3)

    def test_training_config_grouping_helpers_expose_contract_domains(self):
        config = TrainingConfig(
            adaptive_imag_task_corridor_enabled=True,
            adaptive_imag_critic_bootstrap_contract_enabled=True,
            adaptive_imag_compensation_persistence_hold_steps=5,
            adaptive_imag_compensation_post_solved_hold_steps=7,
        )

        self.assertTrue(config.certification_contract_config()["task_corridor_enabled"])
        self.assertTrue(config.authority_contract_config()["critic_bootstrap_contract_enabled"])
        self.assertEqual(config.compensation_config()["persistence_hold_steps"], 5)
        self.assertEqual(config.compensation_config()["post_solved_hold_steps"], 7)

    def test_train_helper_aliases_now_point_to_contract_modules(self):
        self.assertEqual(
            _compute_bootstrap_task_request_contract.__module__,
            "aletheia.contracts.authority",
        )
        self.assertEqual(
            compute_bootstrap_bonus_consumer_retention_contract.__module__,
            "aletheia.contracts.consumers",
        )
        self.assertEqual(
            _compute_bootstrap_trigger_entry_contract.__module__,
            "aletheia.contracts.authority",
        )

    def test_compensation_hold_state_factory_stays_three_channel(self):
        hold_state = make_empty_hold_state()
        self.assertEqual(set(hold_state.keys()), {"pre_transition", "active_transition", "post_transition"})
        self.assertTrue(all(value == 0.0 for value in hold_state.values()))

    def test_consumer_contract_view_prefers_task_certified_contract(self):
        reference = torch.ones((2, 2), dtype=torch.float32)
        geometry_only = make_semantic_contract(
            value=reference,
            source="task_corridor",
            coverage=reference,
            confidence=torch.full_like(reference, 0.8),
            authority=torch.full_like(reference, 0.9),
            trust=torch.full_like(reference, 0.8),
            task_agreement=torch.zeros_like(reference),
            registry_support=torch.full_like(reference, 0.8),
            semantic_debt=torch.full_like(reference, 0.2),
            certified_by=("geometry_corridor",),
        )
        task_certified = make_semantic_contract(
            value=reference,
            source="replay_suffix",
            coverage=reference,
            confidence=torch.full_like(reference, 0.75),
            authority=torch.full_like(reference, 0.7),
            trust=torch.full_like(reference, 0.75),
            task_agreement=torch.full_like(reference, 0.9),
            registry_support=torch.full_like(reference, 0.6),
            semantic_debt=torch.full_like(reference, 0.05),
            certified_by=("task_corridor",),
        )

        view = select_consumer_contract_view(geometry_only, task_certified)

        self.assertEqual(view.source, "replay_suffix")
        self.assertTrue(view.task_certified)
        self.assertFalse(view.geometry_certified)

    def test_compensation_phase_canonicalization_collapses_old_names(self):
        self.assertEqual(canonicalize_compensation_phase("post_entry"), "trigger")
        self.assertEqual(canonicalize_compensation_phase("persistence_release"), "persistence")
        self.assertEqual(canonicalize_compensation_phase("post_solved"), "post_solved")
        self.assertEqual(canonicalize_compensation_phase("unknown"), "idle")

    def test_post_trigger_quality_state_uses_config_only(self):
        config = TrainingConfig(
            adaptive_imag_compensation_trigger_quality_gate=True,
            adaptive_imag_compensation_trigger_quality_gap_threshold=1.0,
            adaptive_imag_compensation_trigger_quality_actor_threshold=5.0,
            adaptive_imag_compensation_trigger_quality_streak=2,
        )
        state = resolve_post_trigger_quality_state(
            config=config,
            quality_good_streak=1,
            return_baseline=torch.tensor([2.0, 2.0], dtype=torch.float32),
            returns=torch.tensor([1.5, 1.5], dtype=torch.float32),
        )

        self.assertEqual(float(state["good"]), 1.0)
        self.assertEqual(float(state["release_ready"]), 1.0)

    def test_handoff_landing_guard_matches_threshold_contract(self):
        config = TrainingConfig(
            adaptive_imag_compensation_trigger_persistence_handoff_landing_guard_enabled=True,
            adaptive_imag_compensation_persistence_tail_gap_threshold=5.0,
            adaptive_imag_compensation_persistence_tail_continue_threshold=0.5,
            adaptive_imag_compensation_trigger_persistence_handoff_continue_threshold=0.25,
        )

        self.assertTrue(
            is_trigger_persistence_handoff_landing_guard_active(
                config=config,
                gap_abs=5.5,
                continue_mean=0.3,
            )
        )

    def test_post_entry_soft_negative_adv_profile_respects_midwater(self):
        config = SimpleNamespace(
            adaptive_imag_post_entry_soft_negative_adv_actor_scale=0.9,
            adaptive_imag_post_entry_soft_negative_adv_critic_boost=0.2,
            adaptive_imag_post_entry_soft_negative_adv_midwater_eval_threshold=120.0,
            adaptive_imag_post_entry_soft_negative_adv_midwater_min_step=1000,
            adaptive_imag_post_entry_soft_negative_adv_midwater_ramp_steps=100,
            adaptive_imag_post_entry_soft_negative_adv_midwater_actor_scale=0.7,
            adaptive_imag_post_entry_soft_negative_adv_midwater_critic_boost=0.8,
        )
        profile = resolve_post_entry_soft_negative_adv_profile(
            config=config,
            current_step=1050,
            external_eval_best_mean=130.0,
            base_actor_scale=1.0,
            base_critic_boost=0.0,
        )
        self.assertEqual(float(profile["midwater_active"]), 1.0)
        self.assertGreater(float(profile["critic_boost"]), 0.2)

    def test_persistence_escape_soft_floors_use_release_progress(self):
        config = SimpleNamespace(
            adaptive_imag_post_entry_soft_negative_adv_midwater_eval_threshold=120.0,
            adaptive_imag_post_entry_soft_negative_adv_midwater_min_step=1000,
            adaptive_imag_post_entry_soft_negative_adv_midwater_ramp_steps=0,
            adaptive_imag_post_entry_soft_cap_floor=0.8,
            adaptive_imag_post_entry_soft_actor_scale_floor=0.9,
            adaptive_imag_compensation_persistence_escape_band_release_steps=100,
            adaptive_imag_compensation_persistence_cap=0.5,
            adaptive_imag_compensation_persistence_actor_scale=0.7,
        )
        floors = resolve_persistence_escape_soft_floors(
            config=config,
            persistence_armed_step=1000,
            current_step=1050,
            external_eval_best_mean=130.0,
        )
        self.assertEqual(float(floors["band_specific_active"]), 1.0)
        self.assertGreater(float(floors["release_progress"]), 0.0)
        self.assertFalse(
            is_trigger_persistence_handoff_landing_guard_active(
                config=config,
                gap_abs=4.0,
                continue_mean=0.3,
            )
        )


if __name__ == "__main__":
    unittest.main()
