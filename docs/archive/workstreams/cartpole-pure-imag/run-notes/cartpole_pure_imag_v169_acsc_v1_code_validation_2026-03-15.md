# CartPole Pure-Imag `v169` ACSC-v1 Code Validation

Date: 2026-03-15

## Scope

This note documents the first-stage implementation validation for:

`Actor Corridor Semantic Calibration (ACSC-v1)`

The goal of this cut is narrow and deliberate:

- do not add new controller rescue logic
- do not reopen the legacy critic path
- do not make stronger replay anchoring the primary answer
- directly modify the actor-use imagined `dynamics` objective so that negative corridor semantics can enter the main analytic update

This is a code-validation note, not yet a new real-run report.

## Executive Conclusion

`ACSC-v1 is correctly integrated into the imagined-dynamics actor path.`

The critical blocker found during validation was not a failed implementation.
It was a test-path mismatch.

The initial integration failure looked like this:

- `corridor_semantic_blend_beta` clearly reached `compute_dreamer_actor_loss()`
- `corridor_semantic_blended_term_mean` changed as expected
- but `weighted_actor_target_mean` and `loss_actor` did not move

After tracing the full `run_step()` path, the real cause was confirmed:

`the failing integration test was exercising the non-precomputed fallback, which forces imag_gradient_mode = reinforce.`

That means the test was not actually running through the analytic `dynamics` branch that ACSC-v1 modifies.

Once the test was corrected to use the same precomputed imagined-policy path used by real imagined training:

- `actor/weighted_actor_target_mean` changed from `3.0` to `-0.25`
- `loss_actor` changed from `-3.0` to `0.25`
- `actor/corridor_semantic_blend_mean` changed from `0.0` to `1.0`

So the first-stage implementation is structurally correct:

`ACSC-v1 now changes the main actor objective on the true imagined-dynamics path, not just side metrics.`

## What Was Implemented

### 1. Actor loss: continuous corridor-semantic blend

In `compute_dreamer_actor_loss()`:

- added `corridor_semantic_blend_beta`
- added negative corridor-advantage clamp controls
- defined:
  - `corridor_adv_term = normed_target - normed_base`
  - optional lower clamp on that term
  - `analytic_term = lerp(normed_target, corridor_adv_term, beta)`

This preserves classic Dreamer-style push when `beta = 0`, and gradually turns the actor objective into an advantage-aware corridor brake when `beta > 0`.

### 2. Imagined batch: corridor-aware beta construction

In `_build_imagined_batch()`:

- compute `advantages = returns - base_actor`
- rank high-value imagined states using:
  - `reference_base_actor` when available
  - otherwise `base_actor`
- construct a high-value corridor mask from the configured quantile
- convert negative-advantage excess into a smooth blend signal
- return `corridor_semantic_blend_beta` in the imagined RL batch

This keeps the intervention local:

- high-value corridor only
- negative-advantage triggered
- continuous rather than controller-switched

### 3. Training config and API exposure

Added first-stage ACSC config controls:

- `adaptive_imag_idle_corridor_advantage_blend_max`
- `adaptive_imag_idle_corridor_negative_adv_threshold`
- `adaptive_imag_idle_corridor_negative_adv_tau`
- `adaptive_imag_idle_corridor_quantile`
- `adaptive_imag_idle_corridor_adv_term_clamp_scale`
- `adaptive_imag_idle_corridor_adv_term_clamp_min`

These are now validated in config and exposed through the training API override path.

### 4. Observability

Added actor-side metrics so the new mechanism is directly auditable:

- `corridor_semantic_blend_mean`
- `corridor_semantic_blend_active_fraction`
- `corridor_semantic_adv_term_mean`
- `corridor_semantic_blended_term_mean`
- `corridor_semantic_adv_term_clamp`
- `corridor_semantic_high_value_fraction`
- `corridor_semantic_neg_adv_excess_mean`
- `corridor_semantic_score_threshold`
- `corridor_semantic_uses_reference_base`
- `corridor_semantic_blend_max`

## Critical Debugging Correction

The most important diagnostic result from this validation pass is:

`ACSC-v1 acts on the precomputed imagined-dynamics path, not on the non-precomputed reinforce fallback.`

In `run_step()`:

- if `use_precomputed_policy_outputs = False`
- then imagined replay is treated as non-differentiable
- and `imag_gradient_mode` is forced to `reinforce`

So a non-precomputed integration test will show:

- corridor semantic metrics changing
- but no change to the analytic actor target

That is not a bug in ACSC-v1.
It is a path-selection mismatch.

This matters for diagnosis because the previous symptom:

`blended term changed, but actor loss did not`

looked like a silent integration failure.

It was actually a test contract issue.

## Verification

### Targeted validation

The following targeted checks passed:

1. `test_actor_loss_dynamics_corridor_semantic_blend_makes_negative_advantage_reduce_main_target`
2. `test_actor_loss_dynamics_corridor_semantic_blend_clamps_negative_advantage_term`
3. `test_run_step_corridor_semantic_blend_reduces_imag_actor_target_and_reports_metrics`
4. `test_explicit_idle_corridor_semantic_calibration_overrides_reach_training_config`

### Broader regression

Passed broader local regression suites:

- `25` tests in `aletheia/tests/test_training_entrypoints.py`
- `53` tests in `aletheia/tests/test_run_train_contracts.py`

### Static verification

`py_compile` passed for:

- `aletheia/aletheia_train.py`
- `aletheia/aletheia_config.py`
- `aletheia/aletheia_api.py`
- `aletheia/tests/test_training_entrypoints.py`
- `aletheia/tests/test_run_train_contracts.py`

## Current Judgment

This stage validates the core structural hypothesis behind the next run:

- the remaining `v168` failure really was inside the actor objective
- the first corrective cut can be made directly in the actor-use imagined branch
- the mechanism can be added without reintroducing controller rescue as the main answer

What is now confirmed:

1. ACSC-v1 is live on the correct branch.
2. It changes the actual actor objective, not only diagnostics.
3. The implementation is minimal and local enough to preserve architectural clarity.

What is not yet claimed:

1. no real-run claim yet
2. no proof yet that late-stage hold is fixed
3. no proof yet that an inflation gate is unnecessary in all cases

## Next Step

The next correct step is now straightforward:

`launch the first real pure-imag run with ACSC-v1 enabled and evaluate whether late-stage corridor hold improves without sacrificing the repaired entry behavior.`

If that real run still shows false-positive braking or weakened early push, the next refinement should remain actor-side and minimal:

- first consider an absolute-inflation gate for the blend
- do not go back to controller rescue
- do not revert to stronger critic-only anchoring as the main fix
