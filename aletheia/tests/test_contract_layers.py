import sys
import unittest
from pathlib import Path

import torch

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.contracts import (
    build_behavior_policy_evidence,
    build_real_feedback_evidence,
    compute_task_certification,
)


class TestContractLayers(unittest.TestCase):
    def test_behavior_policy_evidence_normalizes_shapes(self):
        reference = torch.ones((2, 2), dtype=torch.float32)
        bundle = build_behavior_policy_evidence(
            reference=reference,
            coverage=torch.full((2, 2), 0.5, dtype=torch.float32),
            confidence=torch.full((2, 2), 0.6, dtype=torch.float32),
            authority=torch.full((2, 2), 0.7, dtype=torch.float32),
            trust=torch.full((2, 2), 0.8, dtype=torch.float32),
            task_agreement=torch.full((2, 2), 0.3, dtype=torch.float32),
            registry_support=torch.full((2, 2), 0.4, dtype=torch.float32),
            semantic_debt=torch.full((2, 2), 0.2, dtype=torch.float32),
        )
        self.assertEqual(bundle.coverage.shape, reference.shape)
        self.assertAlmostEqual(float(bundle.authority.mean().item()), 0.7, places=6)
        contract = bundle.as_semantic_contract(source="replay_suffix")
        self.assertEqual(contract.source, "replay_suffix")

    def test_real_feedback_evidence_reads_snapshot(self):
        evidence = build_real_feedback_evidence(
            telemetry={
                "real_reward_degradation": 0.3,
                "real_task_cert_gate": 0.4,
                "real_behavior_action_switch_rate": 0.2,
            },
            last_step=5,
            global_step=12,
        )
        self.assertAlmostEqual(evidence.reward_degradation, 0.3, places=6)
        self.assertAlmostEqual(evidence.task_cert_gate, 0.4, places=6)
        self.assertAlmostEqual(evidence.available, 1.0, places=6)
        self.assertAlmostEqual(evidence.age_steps, 7.0, places=6)

    def test_task_certification_emits_labels_and_certified_support(self):
        task_gate = torch.full((1, 2), 0.8, dtype=torch.float32)
        result = compute_task_certification(
            task_gate=task_gate,
            task_confidence=torch.full_like(task_gate, 0.9),
            real_eval_gate=0.6,
            real_reward_agreement=0.7,
            geometry_support=torch.full_like(task_gate, 0.9),
            registry_support=torch.full_like(task_gate, 1.0),
        )
        self.assertEqual(result.imag_task_cert.shape, task_gate.shape)
        self.assertTrue("geometry_corridor" in result.labels)
        self.assertTrue("task_corridor" in result.labels)
        self.assertTrue(torch.all(result.certified_registry_support <= 0.9).item())


if __name__ == "__main__":
    unittest.main()
