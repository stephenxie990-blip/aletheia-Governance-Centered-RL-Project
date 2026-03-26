# CartPole Pure-Imag `v168` Corridor-Internal Actor Objective Diagnosis

Date: 2026-03-15

## Scope

This note pushes the `v168` diagnosis one layer deeper than
`cartpole_pure_imag_v168_next_layer_diagnosis_2026-03-15.md`.

The question here is no longer:

- whether the system can enter the high-value corridor
- whether replay-grounded absolute anchoring works
- whether policy open-loop corridor maintenance improves imagined geometry

Those questions are already answered by `v167` and `v168`.

The narrower question is:

`Why can v168 enter the corridor, recover into the corridor, and still fail to hold the final current checkpoint?`

## Executive Conclusion

The remaining dominant issue is now best described as:

`a corridor-internal actor-objective mismatch, not a primary world-model failure.`

More concretely:

1. `v168` already fixed enough upstream problems that the system can repeatedly reach high-value behavior.
2. But once inside the corridor, the actor still receives a mostly positive `dynamics`-mode push even after `online_adv = returns - online_base_actor` turns negative.
3. That happens because the current actor loss is dominated by the analytic term `normed_target`, while the negative advantage only enters through a weak auxiliary reinforce term.
4. The built-in negative-adv damping path is controller-gated; in `v168` the controller stays `idle`, so that damping never activates.
5. The result is not a clean one-way collapse, but an oscillatory pattern:
   - enter corridor
   - keep pushing after mild over-optimism has already appeared
   - drift out
   - partially recover
   - finally lose the current checkpoint

So the new root cause is not simply:

`critic still a bit wrong`

It is:

`the actor update law still treats a semantically over-optimistic high-value corridor as if it were safe to keep exploiting.`

## Why This Is A New-Layer Diagnosis

`v167` showed:

- absolute anchor matters
- but anchor alone cannot hold the corridor

`v168` then showed:

- corridor entry is much stronger
- corridor re-entry / recovery is much stronger
- late-stage value mismatch is much smaller than `v167`

Yet final hold still fails.

That means the old explanations are no longer sufficient:

- not mainly "world model cannot imagine the corridor"
- not mainly "critic family has no absolute anchor"
- not mainly "the system cannot re-enter once it falls out"

The new failure must be sought inside the actor/critic improvement loop that operates after the corridor is already reachable.

## Run-Level Evidence

### 1. `v168` starts pushing the wrong way before eval visibly collapses

The first `online_adv < 0` in `v168` appears at `step 800`.

Selected training metrics around the onset:

| Step | `actor/online_adv_mean` | `actor/weighted_actor_target_mean` | `imag/reference_value_ruler_alpha_mean` | `critic/real_mc_value_gap_abs_mean` | `imag/open_loop_audit_imag_value_abs_to_short_return_mean` |
|---:|---:|---:|---:|---:|---:|
| 700 | 2.14 | 0.33 | 0.30 | 4.21 | 8.07 |
| 800 | -2.74 | 0.70 | 0.09 | 6.96 | 15.04 |
| 900 | -2.87 | 0.63 | 0.21 | 8.64 | 20.51 |
| 1000 | -4.16 | 0.30 | 0.37 | 9.72 | 20.85 |

But the nearest evaluation points are still high:

- `eval@750 = 432.6`
- `eval@1000 = 423.8`

This is the key timing fact:

`the training signal has already turned semantically unsafe while the checkpoint is still high-performing.`

So the late failure is not a sudden cliff.
It begins as a corridor-internal mis-update while the model is still good.

### 2. `v168` reduced the mismatch magnitude, but did not convert it into a brake

Comparison at aligned checkpoints:

| Run | First `online_adv < 0` step | `@2000 online_adv` | `@2000 value_target_gap_abs` | `@2000 eval` |
|---|---:|---:|---:|---:|
| `v167` | 1100 | -29.95 | 30.03 | 177.8 |
| `v168` | 800 | -4.67 | 4.69 | 154.8 |

Interpretation:

- `v168` clearly made the system much less wrong internally than `v167`
- but even a moderate residual mismatch is enough to lose the final hold

So the remaining issue is no longer gross semantic explosion.
It is that the actor still keeps moving when it should start damping.

### 3. The actor keeps receiving a positive main objective even when `online_adv` is negative

Selected `v168` points:

| Step | `loss_actor` | `actor/online_adv_mean` | `actor/weighted_actor_target_mean` | `actor/analytic_weight_scale_online_adv_active` | `actor/negative_adv_guard_active` |
|---:|---:|---:|---:|---:|---:|
| 800 | -0.70 | -2.74 | 0.70 | 0.00 | 0.00 |
| 900 | -0.63 | -2.87 | 0.63 | 0.00 | 0.00 |
| 1000 | -0.30 | -4.16 | 0.30 | 0.00 | 0.00 |
| 1500 | -0.58 | -3.92 | 0.58 | 0.00 | 0.00 |
| 2000 | -0.60 | -4.67 | 0.60 | 0.00 | 0.00 |
| 2500 | -0.52 | -9.09 | 0.52 | 0.00 | 0.00 |

This is the most important evidence table in the report.

It shows:

1. `online_adv` is already negative
2. the actor's effective weighted target is still positive
3. the actor-side negative-adv scale mask is not active
4. the negative-adv guard is not active either

So the actor is not being told:

`stop, you are already overshooting`

It is still being told:

`there is positive target mass here, keep optimizing`

### 4. The actor objective is dominated by the analytic term, not by the negative advantage

Selected `v168` points:

| Step | `actor_analytic_coef` | `actor_reinforce_coef` | `actor/weighted_analytic_term_mean` | `actor/weighted_reinforce_term_mean` |
|---:|---:|---:|---:|---:|
| 800 | 1.0 | 0.1 | 0.7041 | -0.0035 |
| 1000 | 1.0 | 0.1 | 0.3031 | -0.0012 |
| 1500 | 1.0 | 0.1 | 0.5842 | -0.0043 |
| 2000 | 1.0 | 0.1 | 0.5939 | 0.0066 |
| 2500 | 1.0 | 0.1 | 0.5237 | -0.0035 |

So even when the corridor semantics have already turned unfavorable:

- the analytic term remains the main signal
- the reinforce term is tiny
- the negative advantage does not dominate the gradient

This makes the remaining failure mode structurally understandable.

## Code-Path Diagnosis

The run evidence above matches the training code.

### 1. Returns are still computed through the imagined rollout

In `aletheia/aletheia_train.py`:

- `_compute_lambda_returns()` computes imagined lambda returns
- `_build_imagined_batch()` uses those returns to define the actor target stream

So the actor is still trained from the imagined branch, which is expected for this project.

### 2. The ruler logic only changes the relative baseline; it does not create a true always-on actor brake

In the imagined batch builder:

- `online_base_actor = values_im[:, :-1]`
- target-critic / ruler logic produces `reference_returns`
- soft gating builds `reference_value_ruler_alpha`
- `returns` and `base_actor` are blended toward the reference branch

This helps with relative ruler mismatch.
It is one reason `v168` is much healthier than `v167`.

But it still does not solve the deeper issue:

`after blending, the actor objective is still mainly a target-maximization objective, not a calibrated negative-adv-aware brake.`

### 3. The actor loss is structurally target-dominant in `dynamics` mode

In `compute_dreamer_actor_loss()`:

```python
raw_adv = target - base
analytic_term = normed_target
reinforce_term = log_probs * reinforce_adv
actor_target = analytic_coef * analytic_term_scaled + reinforce_coef * reinforce_term
actor_loss = -(weights * actor_target).mean()
```

This matters because:

- `raw_adv` can already be negative
- but `analytic_term` is `normed_target`, not `raw_adv`
- with current settings the actor runs with:
  - `analytic_coef = 1.0`
  - `reinforce_coef = 0.1`

Therefore the main actor gradient is still:

`maximize positive normalized target`

not:

`back off when target has fallen below the current value baseline`

This is the central structural mismatch now exposed by `v168`.

### 4. The existing negative-adv damping path is stage-gated, and `v168` stays in `idle`

The code path that can reduce analytic weight based on negative online advantage only activates through controller-stage-specific branches.

But in the `v168` run, the logged controller stage at the key late checkpoints remains:

`idle`

And the metrics show:

- `actor/analytic_weight_scale_online_adv_active = 0.0`
- `actor/negative_adv_guard_active = 0.0`

So the negative-adv signal never becomes an always-on corridor-maintenance brake.

It remains a rescue pathway that only exists in special controller stages.

That is exactly why the user-facing intuition:

`I do not want a model that needs rescue after it already starts falling`

is correct here.

## Pathology, Cause, Root Cause

### Pathology

Observed symptoms:

- current checkpoint eventually falls off
- `online_adv` turns negative
- `value > return` remains present, though much smaller than before
- curve shape becomes oscillatory instead of purely collapsing

### Proximate Cause

The actor keeps updating in the exploit direction even after corridor semantics have turned mildly unsafe.

### Structural Root Cause

`The actor's primary analytic objective is not semantically coupled to corridor-safe advantage in the idle path.`

Said differently:

- the world model can now represent the corridor
- the critic family is less inflated than before
- but the actor still optimizes a target-dominant objective that does not automatically translate mild negative `online_adv` into reduced forward pressure

### Why This Is Not Just "Critic Still Wrong"

If the remaining problem were only:

`critic is still slightly inaccurate`

then the system should become gradually more stable as the mismatch shrinks.

But `v168` shows something more specific:

- mismatch shrinks a lot relative to `v167`
- recovery ability improves a lot
- yet final hold still fails

That pattern points to a structural update-law issue, not merely a smaller version of the old semantic drift.

## Why Phase 1 And Phase 2 Still Matter

This diagnosis does not invalidate the previous two phases.

On the contrary:

- Phase 1 (`MC absolute anchor`) removed the missing-absolute-ruler problem
- Phase 2 (`policy corridor maintenance`) removed the dominant actor-use manifold decay problem

Those repairs are exactly what made the remaining problem legible.

Without them, the actor-objective mismatch would still be hidden behind larger upstream failures.

## Engineering Implication

The next elegant repair should still avoid:

- controller-first rescue
- reverting to old critic paths
- simply stacking more replay anchoring
- brute-force downstream clamps

The next correct target is the actor-use branch itself.

### What must be repaired

The actor's idle-path analytic objective must stop behaving like:

`maximize target whenever target is numerically high`

and start behaving like:

`only keep pushing when corridor-calibrated target is still semantically supportive`

### The precise missing contract

The current system is missing:

`an always-on corridor-semantic damping rule inside the actor analytic path`

not merely a late rescue rule.

## Recommended Next Direction

The next cut should be framed as:

`policy-space actor objective calibration`

not as controller work.

Minimal target behavior for the next repair:

1. when `online_adv < 0` inside the high-value corridor, the actor analytic term should weaken continuously
2. that weakening should work in the normal `idle` path, not only in trigger/post-entry/post-solved controller stages
3. the correction should act on the actor-use branch directly, instead of trying to patch the failure later

In short:

`Phase 1 fixed the missing absolute ruler.`

`Phase 2 fixed corridor geometry and corridor sustain.`

`The next phase must fix the actor objective itself.`

## Final Diagnosis

`v168` did not fail because the model can no longer find the corridor.`

`v168` failed because, once inside the corridor, the actor objective still keeps applying forward pressure after the corridor has already started sending semantically negative feedback.`

That is why the current best description of the remaining root cause is:

`corridor-internal actor-objective mismatch under dynamics-mode target dominance.`
