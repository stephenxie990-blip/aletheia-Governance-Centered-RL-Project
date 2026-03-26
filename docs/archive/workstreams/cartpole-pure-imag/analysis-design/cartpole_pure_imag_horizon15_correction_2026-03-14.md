# CartPole Pure-Imag Horizon Correction (2026-03-14)

## Purpose

Correct the interpretation boundary after the short-horizon `v150` experiment and restore the DreamerV3-standard default horizon in the supported pure-imag CartPole entrypoint.

## Correction

- Default `imagination_horizon` for pure-imag CartPole is restored to `15`.
- This correction follows the DreamerV3 standard horizon choice and should be treated as the mainline default unless a later isolated experiment proves otherwise.

## What stays

- The supported API wiring fix for:
  - `rssm_msc`
  - `rssm_nst`
  - `rssm_shortcut_consistency`
- The previously added pure-imag CartPole default critic stabilizers:
  - `rl.detach_critic_features_on_imagination=True`
  - `rl.use_actor_drift_guard=True`
  - `rl.slow_value_reg_drift_gain=2.0`

## How to read `v150`

- `v150` tested a bundled stack built around `imagination_horizon=8`, plus `SC`, `MSC`, and critic stabilizers.
- Therefore `v150` only supports this narrower conclusion:
  - the bundled **short-horizon** package is not a good mainline candidate.
- `v150` does **not** support either of these stronger claims:
  - `horizon=15` is worse;
  - `SC/MSC` are ineffective under DreamerV3-standard horizon.

## Current best judgment

- The mainline default should remain `horizon=15`.
- The supported entrypoint should keep the `SC/MSC/NST` wiring fix.
- Any follow-up test on imagined-geometry constraints should be split by axis under `horizon=15`, not reintroduced as one full bundle.
