import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.aletheia_world_model import ContinueHead, ContinueHeadConfig, compute_continue_loss


class TestContinueHeadContracts(unittest.TestCase):
    def test_continue_head_uses_configured_optimistic_bias(self):
        head = ContinueHead(
            feat_dim=4,
            config=ContinueHeadConfig(hidden_dims=(8,), optimistic_bias=1.5),
        )

        linear_modules = [m for m in head.net.modules() if isinstance(m, nn.Linear)]
        self.assertTrue(linear_modules)
        last_bias = linear_modules[-1].bias.detach()
        self.assertTrue(torch.allclose(last_bias, torch.full_like(last_bias, 1.5)))

    def test_compute_continue_loss_upweights_terminal_negatives(self):
        logits = torch.zeros(2)
        targets = torch.tensor([1.0, 0.0])

        base_loss = compute_continue_loss(logits, targets)
        weighted_loss = compute_continue_loss(logits, targets, negative_weight=4.0)

        self.assertAlmostEqual(float(base_loss.item()), 0.693147, places=5)
        self.assertAlmostEqual(float(weighted_loss.item()), 1.732868, places=5)
        self.assertGreater(float(weighted_loss.item()), float(base_loss.item()))


if __name__ == "__main__":
    unittest.main()
