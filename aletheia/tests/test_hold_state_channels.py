"""Tests for the 3-channel hold state management."""

import sys
import unittest
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from aletheia.contracts.consumers import (  # noqa: E402
    HOLD_CHANNELS,
    active_hold_channel,
    compute_post_transition_certified_retention_floor_contract,
    hold_state_summary,
    hold_state_to_float,
    make_empty_hold_state,
    select_hold_channel_value,
    update_hold_state_channels,
)


class TestHoldStateChannels(unittest.TestCase):
    """Core channel management tests."""

    def test_empty_hold_state_has_all_channels_at_zero(self):
        state = make_empty_hold_state()
        self.assertEqual(len(state), 3)
        for channel in HOLD_CHANNELS:
            self.assertEqual(state[channel], 0.0)

    def test_channel_selection_by_late_gate(self):
        self.assertEqual(active_hold_channel(0.0), "pre_transition")
        self.assertEqual(active_hold_channel(0.5), "pre_transition")
        self.assertEqual(active_hold_channel(0.54), "pre_transition")
        self.assertEqual(active_hold_channel(0.55), "active_transition")
        self.assertEqual(active_hold_channel(0.75), "active_transition")
        self.assertEqual(active_hold_channel(0.79), "active_transition")
        self.assertEqual(active_hold_channel(0.80), "post_transition")
        self.assertEqual(active_hold_channel(0.95), "post_transition")
        self.assertEqual(active_hold_channel(1.0), "post_transition")

    def test_select_returns_correct_channel_value(self):
        state = {
            "pre_transition": 0.3,
            "active_transition": 0.6,
            "post_transition": 0.9,
        }
        self.assertAlmostEqual(select_hold_channel_value(state, 0.5), 0.3)
        self.assertAlmostEqual(select_hold_channel_value(state, 0.7), 0.6)
        self.assertAlmostEqual(select_hold_channel_value(state, 0.8), 0.9)
        self.assertAlmostEqual(select_hold_channel_value(state, 1.0), 0.9)

    def test_backward_compat_scalar_input(self):
        self.assertAlmostEqual(select_hold_channel_value(0.42, 0.5), 0.42)
        self.assertAlmostEqual(hold_state_to_float(0.42), 0.42)

    def test_hold_state_to_float_returns_max_channel(self):
        state = {
            "pre_transition": 0.3,
            "active_transition": 0.6,
            "post_transition": 0.1,
        }
        self.assertAlmostEqual(hold_state_to_float(state), 0.6)

    def test_update_only_modifies_active_channel(self):
        state = {
            "pre_transition": 0.5,
            "active_transition": 0.5,
            "post_transition": 0.5,
        }
        updated = update_hold_state_channels(state, 0.6, 0.9, 0.0)
        self.assertAlmostEqual(updated["active_transition"], 0.9)
        self.assertAlmostEqual(updated["pre_transition"], 0.495)
        self.assertAlmostEqual(updated["post_transition"], 0.495)

    def test_inactive_decay_ignores_release_pressure(self):
        """Release pressure should NOT accelerate inactive channel decay."""
        state = {
            "pre_transition": 0.5,
            "active_transition": 0.5,
            "post_transition": 0.5,
        }
        updated_low = update_hold_state_channels(state, 0.6, 0.9, 0.0)
        updated_high = update_hold_state_channels(state, 0.6, 0.9, 1.0)
        self.assertAlmostEqual(updated_low["pre_transition"], updated_high["pre_transition"])
        self.assertAlmostEqual(updated_low["post_transition"], updated_high["post_transition"])

    def test_update_backward_compat_with_scalar(self):
        updated = update_hold_state_channels(0.5, 0.5, 0.8, 0.0)
        self.assertIsInstance(updated, dict)
        self.assertEqual(len(updated), 3)
        self.assertAlmostEqual(updated["pre_transition"], 0.8)
        self.assertAlmostEqual(updated["active_transition"], 0.495)

    def test_cross_channel_floor_inheritance(self):
        """Later stage channels should inherit at least 50% of earlier stage floor."""
        state = {
            "pre_transition": 0.6,
            "active_transition": 0.0,
            "post_transition": 0.0,
        }
        updated = update_hold_state_channels(state, 0.3, 0.7, 0.0)
        self.assertAlmostEqual(updated["pre_transition"], 0.7)
        self.assertGreaterEqual(updated["active_transition"], updated["pre_transition"] * 0.5 - 0.01)

    def test_mid_and_late_hold_independent_with_new_boundaries(self):
        """With boundaries at 0.55/0.80, mid-window and late-window hold diverge."""
        state = make_empty_hold_state()
        state = update_hold_state_channels(state, 0.60, 0.5, 0.0)
        self.assertAlmostEqual(state["active_transition"], 0.5)
        state = update_hold_state_channels(state, 0.85, 0.8, 0.0)
        self.assertAlmostEqual(state["post_transition"], 0.8)
        self.assertGreater(state["active_transition"], 0.4)
        mid_val = select_hold_channel_value(state, 0.70)
        late_val = select_hold_channel_value(state, 0.90)
        self.assertNotAlmostEqual(mid_val, late_val)

    def test_summary_emits_all_channels(self):
        state = {
            "pre_transition": 0.3,
            "active_transition": 0.6,
            "post_transition": 0.9,
        }
        summary = hold_state_summary(state)
        self.assertEqual(len(summary), 3)
        self.assertIn("hold_state_pre_transition", summary)
        self.assertIn("hold_state_active_transition", summary)
        self.assertIn("hold_state_post_transition", summary)

    def test_post_transition_certified_floor_is_neutral_before_window(self):
        (
            activation,
            capture,
            floor_candidate,
            floor_state,
            hold_floored,
            gate_floored,
        ) = compute_post_transition_certified_retention_floor_contract(
            critic_contract_bootstrap_late_gate=0.79,
            critic_contract_bootstrap_trigger_gate=1.0,
            critic_contract_bootstrap_hold_state_active_transition=0.6,
            critic_contract_bootstrap_hold_state_post_transition=0.2,
            source_hold_seed_mean=0.8,
            source_hold_health_mean=0.9,
            source_hold_retention_mean=0.72,
            source_hold_release_pressure_mean=0.2,
            critic_contract_bootstrap_source_hold_persistence_gate_mean=0.05,
            prev_post_transition_certified_floor_state=0.3,
        )

        self.assertEqual(activation, 0.0)
        self.assertAlmostEqual(capture, 0.0)
        self.assertAlmostEqual(floor_candidate, 0.0)
        self.assertEqual(floor_state, 0.0)
        self.assertAlmostEqual(hold_floored, 0.2)
        self.assertAlmostEqual(gate_floored, 0.05)

    def test_post_transition_certified_floor_builds_when_window_is_open(self):
        (
            activation,
            capture,
            _floor_candidate,
            floor_state,
            hold_floored,
            gate_floored,
        ) = compute_post_transition_certified_retention_floor_contract(
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_trigger_gate=1.0,
            critic_contract_bootstrap_hold_state_active_transition=0.3,
            critic_contract_bootstrap_hold_state_post_transition=0.18,
            source_hold_seed_mean=0.5,
            source_hold_health_mean=0.6,
            source_hold_retention_mean=0.3,
            source_hold_release_pressure_mean=0.1,
            critic_contract_bootstrap_source_hold_persistence_gate_mean=0.04,
            prev_post_transition_certified_floor_state=0.1,
        )

        self.assertAlmostEqual(activation, 1.0)
        self.assertGreater(capture, 0.0)
        self.assertAlmostEqual(capture, 0.3)
        self.assertGreater(floor_state, 0.0)
        self.assertGreaterEqual(hold_floored, 0.18)
        self.assertGreaterEqual(gate_floored, 0.04)

    def test_post_transition_certified_capture_uses_history_when_supported(self):
        (
            _activation,
            capture,
            _floor_candidate,
            _floor_state,
            _hold_floored,
            _gate_floored,
        ) = compute_post_transition_certified_retention_floor_contract(
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_trigger_gate=1.0,
            critic_contract_bootstrap_hold_state_active_transition=0.0,
            critic_contract_bootstrap_hold_state_post_transition=0.05,
            source_hold_seed_mean=0.8,
            source_hold_health_mean=0.5,
            source_hold_retention_mean=0.04,
            source_hold_release_pressure_mean=0.2,
            critic_contract_bootstrap_source_hold_persistence_gate_mean=0.0,
            prev_post_transition_certified_floor_state=0.6,
        )

        self.assertAlmostEqual(capture, 0.45)

    def test_post_transition_certified_capture_stays_bounded_by_support(self):
        (
            _activation,
            capture,
            _floor_candidate,
            _floor_state,
            _hold_floored,
            _gate_floored,
        ) = compute_post_transition_certified_retention_floor_contract(
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_trigger_gate=1.0,
            critic_contract_bootstrap_hold_state_active_transition=0.0,
            critic_contract_bootstrap_hold_state_post_transition=0.05,
            source_hold_seed_mean=0.2,
            source_hold_health_mean=0.1,
            source_hold_retention_mean=0.02,
            source_hold_release_pressure_mean=0.8,
            critic_contract_bootstrap_source_hold_persistence_gate_mean=0.0,
            prev_post_transition_certified_floor_state=0.6,
        )

        self.assertAlmostEqual(capture, 0.07)

    def test_post_transition_certified_capture_can_break_v242_current_authority_tie(self):
        (
            activation,
            capture,
            floor_candidate,
            floor_state,
            hold_floored,
            gate_floored,
        ) = compute_post_transition_certified_retention_floor_contract(
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_trigger_gate=1.0,
            critic_contract_bootstrap_hold_state_active_transition=0.0,
            critic_contract_bootstrap_hold_state_post_transition=0.05309565421368734,
            source_hold_seed_mean=1.0,
            source_hold_health_mean=0.05309565421368734,
            source_hold_retention_mean=0.02654782496392727,
            source_hold_release_pressure_mean=0.9054224491119385,
            critic_contract_bootstrap_source_hold_persistence_gate_mean=0.02654782496392727,
            prev_post_transition_certified_floor_state=0.05303715282235631,
        )

        self.assertAlmostEqual(activation, 1.0)
        self.assertGreater(capture, 0.05309565421368734)
        self.assertGreater(floor_candidate, 0.05309565421368734)
        self.assertGreater(floor_state, 0.05309565421368734)
        self.assertGreaterEqual(hold_floored, floor_state)
        self.assertGreater(gate_floored, 0.02654782496392727)
        self.assertAlmostEqual(gate_floored, floor_state)

    def test_post_transition_certified_floor_releases_when_trigger_gate_is_closed(self):
        (
            activation,
            capture,
            floor_candidate,
            floor_state,
            hold_floored,
            gate_floored,
        ) = compute_post_transition_certified_retention_floor_contract(
            critic_contract_bootstrap_late_gate=1.0,
            critic_contract_bootstrap_trigger_gate=0.0,
            critic_contract_bootstrap_hold_state_active_transition=0.4,
            critic_contract_bootstrap_hold_state_post_transition=0.2,
            source_hold_seed_mean=0.7,
            source_hold_health_mean=0.8,
            source_hold_retention_mean=0.56,
            source_hold_release_pressure_mean=0.0,
            critic_contract_bootstrap_source_hold_persistence_gate_mean=0.03,
            prev_post_transition_certified_floor_state=0.5,
        )

        self.assertAlmostEqual(activation, 1.0)
        self.assertAlmostEqual(capture, 0.56)
        self.assertGreater(floor_candidate, 0.0)
        self.assertEqual(floor_state, 0.0)
        self.assertAlmostEqual(hold_floored, 0.2)
        self.assertAlmostEqual(gate_floored, 0.03)


if __name__ == "__main__":
    unittest.main()
