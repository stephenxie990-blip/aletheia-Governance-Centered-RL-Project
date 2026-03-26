# CartPole Pure-Imag `v169` ACSC-v1 Validation Run

Date: 2026-03-15

## Scope

This report validates the first real pure-imag run after the first-stage
`Actor Corridor Semantic Calibration (ACSC-v1)` implementation.

Run directory:

- `outputs/exp_seed42_v169_h15_mc_anchor_policy_corridor_maintenance_acsc_v1_2500_20260315_01`

Base stack kept from `v168`:

- MC replay-grounded absolute anchor
- target-value ruler soft gate
- policy open-loop consistency

New addition in `v169`:

- ACSC-v1 on the imagined `dynamics` actor branch

## Executive Conclusion

`v169 proves the actor-side cut hits a real lever, but ACSC-v1 is still incomplete.`

The strongest positive evidence is:

- `v169` reaches `500.0 @ 1250`

That matters because it confirms:

`actor-side corridor-semantic intervention can produce solved-level behavior on the supported pure-imag mainline.`

But the current checkpoint still does not hold:

- `current checkpoint @2500 = 20.4`

So this is not the final fix.

The right interpretation is:

1. the intervention location is correct
2. the first-stage ACSC mechanism is too weak or too unspecific to maintain the corridor
3. the next cut should stay actor-side and become more selective, not go back to controller rescue

## Important Artifact Clarification

This run auto-promoted the best checkpoint into the final artifact after training.

So:

- `summary.json` shows the promoted final artifact performance
- but the true training-end current-checkpoint result is the last entry in `eval_history.jsonl`

For diagnosis, the number that matters is:

- `step 2500 current checkpoint mean = 20.4`

not the promoted best artifact score.

## Eval Curve

| Step | Eval Mean |
|---:|---:|
| 250 | 154.2 |
| 500 | 245.8 |
| 750 | 104.2 |
| 1000 | 275.4 |
| 1250 | 500.0 |
| 1500 | 310.4 |
| 1750 | 174.4 |
| 2000 | 322.6 |
| 2250 | 196.4 |
| 2500 | 20.4 |

Best during train:

- `500.0 @ 1250`

Current checkpoint at train end:

- `20.4 @ 2500`

## Comparison Against `v168`

`v168` eval curve:

- `250 -> 154.2`
- `500 -> 257.4`
- `750 -> 432.6`
- `1000 -> 423.8`
- `1250 -> 186.0`
- `1500 -> 123.2`
- `1750 -> 258.2`
- `2000 -> 154.8`
- `2250 -> 340.0`
- `2500 -> 58.4`

Key comparison:

### What improved

1. `v169` reaches solved behavior:
   - `v168` never reached `500`
   - `v169` reached `500.0 @ 1250`
2. Some middle-late checkpoints improved:
   - `1500`: `310.4` vs `123.2`
   - `2000`: `322.6` vs `154.8`

### What worsened

1. Early strong build-up weakened:
   - `750`: `104.2` vs `432.6`
   - `1000`: `275.4` vs `423.8`
2. Final current-checkpoint hold got worse:
   - `2500`: `20.4` vs `58.4`

So `v169` is not simply "better" or "worse".
It changed the failure shape.

## Failure Shape

`v169` no longer looks like a simple inability to solve.`

It looks like:

- weaker early build
- intermittent high-quality recovery
- solved spike at mid-run
- repeated loss of corridor hold
- final collapse of the current checkpoint

This means the remaining issue is not:

`actor-side intervention was the wrong place`

The run already disproves that.

The remaining issue is:

`ACSC-v1 does not yet turn negative corridor semantics into a decisive and selective maintenance law.`

## Internal Metrics

Selected checkpoints from `train_metrics.jsonl`:

| Anchor | `blend_mean` | `weighted_actor_target_mean` | `online_adv_mean` | `adv_mean` | `value_target_gap_abs_mean` |
|---:|---:|---:|---:|---:|---:|
| 700 | 0.0261 | 0.6357 | -1.2786 | -1.2098 | 2.0894 |
| 1000 | 0.0747 | 0.5571 | -5.2817 | -5.2208 | 6.0396 |
| 1200 | 0.0919 | 0.2147 | -17.0590 | -16.6807 | 17.6613 |
| 1500 | 0.0926 | 0.2291 | -20.6664 | -19.7557 | 20.6664 |
| 2000 | 0.0795 | 0.3826 | -11.5080 | -10.1623 | 12.1699 |
| 2500 | 0.0751 | 0.2467 | -9.8045 | -9.8003 | 10.1114 |

This table is the most important diagnostic result from `v169`.

It shows:

1. `ACSC-v1` is active
   - `blend_mean` is non-zero after onset
2. but it stays small
   - only around `0.07 ~ 0.09`
3. meanwhile the actor main target remains positive throughout
   - `weighted_actor_target_mean` never turns negative
4. and the value/advantage mismatch still becomes large
   - `online_adv_mean` and `value_target_gap_abs_mean` remain strongly unhealthy

So the current mechanism does not yet create a strong enough corridor brake.

## Updated Diagnosis

`v169` sharpens the diagnosis in a useful way.`

### What `v169` rules out

1. It is no longer credible to say:
   - "actor-side intervention is unnecessary"
2. It is no longer credible to say:
   - "the right answer is to go back to controller rescue"

Why:

- actor-side ACSC changed the run shape materially
- it created a solved checkpoint
- the intervention point is therefore causally real

### What `v169` now suggests

The remaining problem is that ACSC-v1 is:

- corridor-aware
- negative-adv-aware
- but not yet sufficiently selective or sufficiently strong

More concretely:

1. negative advantage alone is not enough as the semantic trigger
   - it can be present both in genuine corridor corruption and in ordinary learning fluctuations
2. because the gate is not anchored to absolute inflation, the blend signal stays small and diffuse
3. the actor main target remains positive even while mismatch becomes large
4. the result is still corridor oscillation rather than stable hold

This matches the earlier review warning:

`the actor-side semantic brake likely needs an absolute-inflation gate, not just negative advantage and corridor rank.`

## Most Likely Next Cut

The next elegant step should remain local and actor-side:

`beta = corridor_mask × negative_adv_gate × inflation_gate`

Where `inflation_gate` is grounded in the already-existing absolute calibration signals, for example:

- MC-anchor critic gap
- common-mode inflation indicator
- or a related absolute value-over-anchor measure

The goal is:

- do not brake just because advantage is negative
- brake specifically when high-value corridor states are both:
  - semantically negative
  - and absolutely over-optimistic

## Bottom Line

`v169` is a meaningful advance in diagnosis even though it is not the final fix.`

It establishes three things:

1. ACSC belongs in the actor branch, not in controller rescue.
2. First-stage ACSC can reach solved behavior.
3. The next refinement should be an actor-side absolute-inflation-gated ACSC, not a return to downstream patchwork.
