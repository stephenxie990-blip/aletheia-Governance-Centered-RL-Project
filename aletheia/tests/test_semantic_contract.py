import sys
import unittest
from pathlib import Path

import torch

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.contracts.core import SemanticArbiter, SemanticContract


class TestSemanticContract(unittest.TestCase):
    def _make_contract(self, **overrides) -> SemanticContract:
        base = {
            "value": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            "source": "replay_suffix",
            "coverage": torch.tensor([[1.0, 0.5], [0.25, 0.0]], dtype=torch.float32),
            "confidence": torch.full((2, 2), 0.7, dtype=torch.float32),
            "authority": torch.full((2, 2), 0.3, dtype=torch.float32),
            "trust": torch.full((2, 2), 0.8, dtype=torch.float32),
            "task_agreement": torch.full((2, 2), 0.6, dtype=torch.float32),
            "registry_support": torch.full((2, 2), 0.5, dtype=torch.float32),
            "semantic_debt": torch.full((2, 2), 0.2, dtype=torch.float32),
            "freshness": 3,
            "certified_by": ("geometry_corridor", "task_corridor"),
        }
        base.update(overrides)
        return SemanticContract(**base)

    def test_semantic_contract_fields_present(self):
        contract = self._make_contract()
        self.assertEqual(contract.source, "replay_suffix")
        self.assertTrue(contract.is_external_source)
        self.assertEqual(contract.coverage.shape, contract.value.shape)
        self.assertEqual(contract.confidence.shape, contract.value.shape)
        self.assertEqual(contract.authority.shape, contract.value.shape)
        self.assertEqual(contract.trust.shape, contract.value.shape)
        self.assertEqual(contract.task_agreement.shape, contract.value.shape)
        self.assertEqual(contract.registry_support.shape, contract.value.shape)
        self.assertEqual(contract.semantic_debt.shape, contract.value.shape)
        self.assertEqual(contract.freshness, 3)
        self.assertEqual(contract.certified_by, ("geometry_corridor", "task_corridor"))
        summary = contract.summary()
        self.assertEqual(summary["source"], "replay_suffix")
        self.assertTrue(summary["geometry_certified"])
        self.assertTrue(summary["task_certified"])

    def test_clean_anchor_has_bounded_coverage(self):
        contract = self._make_contract(
            coverage=torch.tensor([[0.0, 0.1], [0.9, 1.0]], dtype=torch.float32)
        )
        self.assertTrue(torch.equal(contract.coverage, torch.tensor([[0.0, 0.1], [0.9, 1.0]])))
        with self.assertRaisesRegex(ValueError, "coverage must stay within \\[0, 1\\]"):
            self._make_contract(
                coverage=torch.tensor([[0.0, 1.2], [0.5, -0.1]], dtype=torch.float32)
            )

    def test_external_authority_requires_coverage_metadata(self):
        with self.assertRaises(TypeError):
            SemanticContract(
                value=torch.ones((2, 2), dtype=torch.float32),
                source="replay_mc",
                confidence=torch.ones((2, 2), dtype=torch.float32),
                authority=torch.full((2, 2), 0.3, dtype=torch.float32),
                trust=torch.full((2, 2), 0.8, dtype=torch.float32),
                task_agreement=torch.full((2, 2), 0.6, dtype=torch.float32),
                registry_support=torch.full((2, 2), 0.5, dtype=torch.float32),
                semantic_debt=torch.full((2, 2), 0.2, dtype=torch.float32),
            )

    def test_task_cert_and_geometry_cert_are_distinct_channels(self):
        contract = self._make_contract(
            certified_by=("geometry:corridor", "task:reward_agreement", "observer_note")
        )
        self.assertEqual(contract.geometry_certifications, ("geometry:corridor",))
        self.assertEqual(contract.task_certifications, ("task:reward_agreement",))
        self.assertNotEqual(contract.geometry_certifications, contract.task_certifications)
        self.assertTrue(contract.is_geometry_certified)
        self.assertTrue(contract.is_task_certified)
        self.assertFalse(contract.has_certification("task_corridor"))

    def test_metric_summary_emits_numeric_contract_observation_fields(self):
        contract = self._make_contract()
        metrics = contract.metric_summary("contract/test")
        self.assertEqual(metrics["contract/test_is_external"], 1.0)
        self.assertEqual(metrics["contract/test_geometry_certified"], 1.0)
        self.assertEqual(metrics["contract/test_task_certified"], 1.0)
        self.assertAlmostEqual(
            metrics["contract/test_coverage_mean"],
            float(contract.coverage.mean()),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["contract/test_authority_mean"],
            float(contract.authority.mean()),
            places=6,
        )

    def test_semantic_arbiter_enforces_takeover_floor_before_bonus(self):
        arbiter = SemanticArbiter()
        internal = self._make_contract(
            source="critic_bootstrap",
            coverage=torch.full((2, 2), 0.2, dtype=torch.float32),
            confidence=torch.full((2, 2), 0.2, dtype=torch.float32),
            authority=torch.full((2, 2), 0.2, dtype=torch.float32),
            trust=torch.full((2, 2), 0.2, dtype=torch.float32),
            task_agreement=torch.full((2, 2), 0.1, dtype=torch.float32),
            registry_support=torch.full((2, 2), 0.1, dtype=torch.float32),
            semantic_debt=torch.full((2, 2), 0.8, dtype=torch.float32),
            certified_by=(),
        )
        external = self._make_contract(
            source="replay_suffix",
            coverage=torch.full((2, 2), 1.0, dtype=torch.float32),
            confidence=torch.full((2, 2), 0.95, dtype=torch.float32),
            authority=torch.full((2, 2), 0.95, dtype=torch.float32),
            trust=torch.full((2, 2), 0.95, dtype=torch.float32),
            task_agreement=torch.full((2, 2), 0.95, dtype=torch.float32),
            registry_support=torch.full((2, 2), 0.95, dtype=torch.float32),
            semantic_debt=torch.zeros((2, 2), dtype=torch.float32),
            certified_by=("geometry_corridor", "task_corridor"),
        )

        decision = arbiter.arbitrate_bootstrap(
            internal_contract=internal,
            external_contract=external,
            takeover_floor=torch.full((2, 2), 0.4, dtype=torch.float32),
            modulation_bonus=torch.full((2, 2), 0.35, dtype=torch.float32),
            max_external_authority=0.6,
        )

        self.assertTrue(
            torch.allclose(
                decision.takeover_floor,
                torch.full((2, 2), 0.4, dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                decision.modulation_bonus,
                torch.full((2, 2), 0.2, dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                decision.external_authority,
                torch.full((2, 2), 0.6, dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                decision.internal_authority,
                torch.full((2, 2), 0.4, dtype=torch.float32),
            )
        )

    def test_semantic_arbiter_prefers_external_contracts_with_better_semantics(self):
        arbiter = SemanticArbiter()
        internal = self._make_contract(
            source="critic_bootstrap",
            coverage=torch.full((2, 2), 0.8, dtype=torch.float32),
            confidence=torch.full((2, 2), 0.8, dtype=torch.float32),
            authority=torch.full((2, 2), 0.8, dtype=torch.float32),
            trust=torch.full((2, 2), 0.8, dtype=torch.float32),
            task_agreement=torch.full((2, 2), 0.8, dtype=torch.float32),
            registry_support=torch.full((2, 2), 0.8, dtype=torch.float32),
            semantic_debt=torch.full((2, 2), 0.1, dtype=torch.float32),
            certified_by=("task_corridor",),
        )
        weak_external = self._make_contract(
            source="replay_suffix",
            coverage=torch.full((2, 2), 0.2, dtype=torch.float32),
            confidence=torch.full((2, 2), 0.2, dtype=torch.float32),
            authority=torch.full((2, 2), 0.1, dtype=torch.float32),
            trust=torch.full((2, 2), 0.1, dtype=torch.float32),
            task_agreement=torch.full((2, 2), 0.1, dtype=torch.float32),
            registry_support=torch.full((2, 2), 0.1, dtype=torch.float32),
            semantic_debt=torch.full((2, 2), 0.8, dtype=torch.float32),
            certified_by=("geometry_corridor",),
        )
        strong_external = self._make_contract(
            source="replay_suffix",
            coverage=torch.full((2, 2), 1.0, dtype=torch.float32),
            confidence=torch.full((2, 2), 0.95, dtype=torch.float32),
            authority=torch.full((2, 2), 0.9, dtype=torch.float32),
            trust=torch.full((2, 2), 0.95, dtype=torch.float32),
            task_agreement=torch.full((2, 2), 0.9, dtype=torch.float32),
            registry_support=torch.full((2, 2), 0.9, dtype=torch.float32),
            semantic_debt=torch.zeros((2, 2), dtype=torch.float32),
            certified_by=("geometry_corridor", "task_corridor"),
        )

        weak = arbiter.arbitrate_bootstrap(
            internal_contract=internal,
            external_contract=weak_external,
            takeover_floor=torch.full((2, 2), 0.1, dtype=torch.float32),
            modulation_bonus=torch.full((2, 2), 0.5, dtype=torch.float32),
            max_external_authority=0.9,
        )
        strong = arbiter.arbitrate_bootstrap(
            internal_contract=internal,
            external_contract=strong_external,
            takeover_floor=torch.full((2, 2), 0.1, dtype=torch.float32),
            modulation_bonus=torch.full((2, 2), 0.5, dtype=torch.float32),
            max_external_authority=0.9,
        )

        self.assertGreater(
            float(strong.external_authority.mean()),
            float(weak.external_authority.mean()),
        )
        self.assertGreater(
            float(strong.modulation_bonus.mean()),
            float(weak.modulation_bonus.mean()),
        )

    def test_semantic_arbiter_treats_geometry_cert_as_support_not_task_truth(self):
        arbiter = SemanticArbiter()
        internal = self._make_contract(
            source="critic_bootstrap",
            coverage=torch.full((2, 2), 0.4, dtype=torch.float32),
            confidence=torch.full((2, 2), 0.4, dtype=torch.float32),
            authority=torch.full((2, 2), 0.4, dtype=torch.float32),
            trust=torch.full((2, 2), 0.4, dtype=torch.float32),
            task_agreement=torch.full((2, 2), 0.4, dtype=torch.float32),
            registry_support=torch.full((2, 2), 0.4, dtype=torch.float32),
            semantic_debt=torch.full((2, 2), 0.2, dtype=torch.float32),
            certified_by=(),
        )
        geometry_only = self._make_contract(
            source="replay_suffix",
            coverage=torch.full((2, 2), 1.0, dtype=torch.float32),
            confidence=torch.full((2, 2), 0.9, dtype=torch.float32),
            authority=torch.full((2, 2), 0.9, dtype=torch.float32),
            trust=torch.full((2, 2), 0.9, dtype=torch.float32),
            task_agreement=torch.full((2, 2), 0.9, dtype=torch.float32),
            registry_support=torch.full((2, 2), 0.9, dtype=torch.float32),
            semantic_debt=torch.zeros((2, 2), dtype=torch.float32),
            certified_by=("geometry_corridor",),
        )
        task_certified = self._make_contract(
            source="replay_suffix",
            coverage=torch.full((2, 2), 1.0, dtype=torch.float32),
            confidence=torch.full((2, 2), 0.9, dtype=torch.float32),
            authority=torch.full((2, 2), 0.9, dtype=torch.float32),
            trust=torch.full((2, 2), 0.9, dtype=torch.float32),
            task_agreement=torch.full((2, 2), 0.9, dtype=torch.float32),
            registry_support=torch.full((2, 2), 0.9, dtype=torch.float32),
            semantic_debt=torch.zeros((2, 2), dtype=torch.float32),
            certified_by=("geometry_corridor", "task_corridor"),
        )

        geometry_decision = arbiter.arbitrate_bootstrap(
            internal_contract=internal,
            external_contract=geometry_only,
            takeover_floor=torch.full((2, 2), 0.1, dtype=torch.float32),
            modulation_bonus=torch.full((2, 2), 0.6, dtype=torch.float32),
            max_external_authority=0.9,
        )
        task_decision = arbiter.arbitrate_bootstrap(
            internal_contract=internal,
            external_contract=task_certified,
            takeover_floor=torch.full((2, 2), 0.1, dtype=torch.float32),
            modulation_bonus=torch.full((2, 2), 0.6, dtype=torch.float32),
            max_external_authority=0.9,
        )

        self.assertGreater(
            float(task_decision.external_authority.mean()),
            float(geometry_decision.external_authority.mean()),
        )


if __name__ == "__main__":
    unittest.main()
