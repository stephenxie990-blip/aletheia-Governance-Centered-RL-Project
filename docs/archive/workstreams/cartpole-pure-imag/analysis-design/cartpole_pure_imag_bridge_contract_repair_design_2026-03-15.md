# CartPole Pure-Imag Bridge Contract Repair Design

Date: 2026-03-15

## Core Judgment

The most elegant repair is not:

- stronger controller rescue,
- stronger replay-only anchor,
- or simply opening more RL gradient into the world model.

The most elegant repair is:

`complete the post-wall bridge contract`

In plain language:

- let the `iron wall` keep doing its job,
- but stop orphaning the bridge behind it.

## The Deeper Design Error

The current system mixes up two different responsibilities:

1. `truth protection`
   - block actor / critic reward-seeking gradients from corrupting predictive truth
2. `bridge maintenance`
   - continuously train the modules that convert predictive truth into actor-use policy features

These two responsibilities are currently entangled.

As a result:

- once the wall is effectively closed,
- the bridge also loses its training contract,
- so policy space slowly drifts away from predictive truth even if the predictive core remains healthy.

This is why the real repair should not be "break the wall".

It should be:

`separate truth isolation from bridge training`

## Proposed Solution

## Name

`Post-Wall Bridge Contract`

## Principle

The predictive core remains protected.

The bridge is trained by replay-side teacher/open-loop supervision, not by direct policy reward gradients.

## Architecture

### A. Predictive Core

Modules:

- RSSM `state_transition`
- reward head
- continue head
- decoder

Responsibility:

- model the world truthfully

Gradient source:

- replay teacher-forced predictive losses only

### B. Bridge Layer

Modules:

- `control_head` producing `s_ctrl`
- `projection` producing `x_proj`
- `abstractor` producing `z_task`
- `consistency_head`
- `aux_tasks`
- `router`

Responsibility:

- translate predictive truth into actor-use policy features
- preserve control semantics
- preserve short-horizon value semantics

Gradient source:

- replay-side bridge supervision
- replay open-loop policy-space consistency
- optional weak target-ruler distill

### C. RL Head

Modules:

- actor
- critic

Responsibility:

- optimize policy/value on top of the bridge output

Gradient source:

- imagined RL as usual

But:

- no direct RL gradient should be required to keep the bridge semantically alive

## Loss Design

The repair should be built as a bridge-loss pack, mostly reusing modules that already exist in the repo.

## 1. Projection Loss

Use existing `ProjectionLayer.compute_loss()`.

Goal:

- make `x_proj(s_pred)` stay close to the real encoded observation embedding `x_t`

Why it matters:

- actor-use policy path currently consumes `x_proj/x_t`
- if `x_proj` is not trained, the policy bridge starts from a semantically drifting input

Status today:

- module exists
- loss implementation exists
- main WM loss currently sets projection loss to zero

## 2. Control Loss

Use existing `ControlHead.compute_aux_loss()`.

For CartPole simple preset, `control.mode=static`, so `L_align` is especially relevant.

Goal:

- make `s_ctrl` remain an action-relevant control summary, not a free latent that drifts arbitrarily

Status today:

- loss exists
- not wired

## 3. Consistency Loss For Control Transition

Use existing `ConsistencyHead.compute_loss()`.

Goal:

- from `(s_pred_t, action_t)` predict `s_ctrl_{t+1}`
- tie control semantics to predictive dynamics

Why this is elegant:

- still uses replay/local sequence supervision
- does not leak future truth from unrealized policy branches
- directly closes the missing contract between predictive dynamics and control state

Status today:

- module/loss exist
- not instantiated in current world-model build
- not wired into training

## 4. Auxiliary Bridge Tasks

Use existing `ControlAuxTasks.compute_loss()`.

Two parts:

- inverse dynamics: `(s_pred_t, s_pred_{t+1}) -> action_t`
- short-return/value auxiliary prediction from `s_pred_t`

Goal:

- force bridge-relevant latent structure to remain action-sensitive and return-sensitive

Why this is elegant:

- it trains semantics without opening policy gradients into the predictive core

Status today:

- module/loss exist
- not instantiated
- not wired

## 5. Policy-Space Open-Loop Consistency

This is the one new loss I would explicitly add.

But it can reuse existing open-loop audit code almost directly.

Current audit already computes:

- teacher state under replay future actions
- imagined state under replay future actions
- teacher policy features
- imagined policy features

The elegant move is:

turn this from audit into training signal.

### Loss form

For replay suffix actions only:

- `L_policy_feat = huber(f_policy_imag, stopgrad(f_policy_teacher))`
- `L_policy_value = huber(V_target(f_policy_imag), stopgrad(V_target(f_policy_teacher)))`

Recommended weighting:

- feature loss: small
- value-semantic loss: medium

Why both:

- feature-only is too geometric and can overconstrain
- value-only is too weak and may permit uncontrolled local distortion

Together they say:

- keep actor-use policy space locally aligned,
- but align it mainly on task semantics

This is the cleanest answer to the user-space problem:

it directly supervises the space the actor actually consumes,
without cheating with unrealized future labels.

## The Critical Schedule Fix

This is, in my view, the deepest part.

The current conceptual schedule is roughly:

- wall closes
- aux tasks fade out with it

That is wrong for this architecture.

Because:

- predictive truth still needs protection,
- but the bridge still needs maintenance forever.

So the repair must introduce two separate knobs:

## 1. `truth_wall_strength(step)`

Controls:

- RL gradient isolation into predictive core

This can still go to `1.0`.

## 2. `bridge_maintenance_scale(step)`

Controls:

- projection
- control
- consistency head
- aux tasks
- policy open-loop bridge consistency

This must not decay to zero.

It should decay to a positive floor.

### Recommended shape

`bridge_scale(step) = bridge_floor + (1 - bridge_floor) * curriculum(step)`

Where:

- early stage: strong bridge training
- late stage: weaker but still alive

Recommended intuition:

- the bridge should be most strongly shaped early,
- but never fully abandoned late

That one change is more important than any controller patch.

## Router Strategy

The router should stop depending primarily on imagined RL gradient for survival.

That is the current structural trap.

Recommended strategy:

### Router primary trainer

- replay-side policy-space bridge consistency
- bridge auxiliary semantics

### Router optional secondary trainer

- weak target-ruler or critic-side utility term

### Router should not rely on

- raw actor policy gradient
- raw online imagined critic drift

Why:

- otherwise router becomes an RL-only bridge
- which is exactly how semantic drift sneaks in behind the wall

## What Not To Do

## 1. Do not simply disable `detach_critic_features_on_imagination`

That does restore router gradient.

But it restores the wrong gradient:

- router starts following online imagined critic pressure directly
- bridge semantics become downstream-RL-shaped instead of teacher-anchored

This is brute force, not elegance.

## 2. Do not continue strengthening replay-only value anchors as the main fix

Those help the critic stay sane.

They do not by themselves train the bridge.

## 3. Do not go back to controller-first rescue

That is late-stage damage control.

It does not heal the upstream contract.

## The Most Elegant Repair Sequence

## Phase 1. Restore Existing Dead Design

Wire in:

- projection loss
- control aux loss
- consistency head loss
- aux inverse/value losses

Add:

- `bridge_maintenance_scale`
- positive late-stage maintenance floor

Do not touch controller.

Do not relax wall.

## Phase 2. Promote Open-Loop Audit To Actor-Use Bridge Loss

Reuse current replay open-loop audit path.

Turn:

- teacher policy feature vs imagined policy feature
- teacher target-value vs imagined target-value

into actual training losses.

This is the first point where the actor-use space is explicitly and cleanly aligned.

## Phase 3. Re-evaluate Need For RL-Side Router Gradient

Only after bridge contract is alive again:

- test whether router still needs any imagined critic gradient

My expectation:

- if bridge training is correctly wired,
- router RL dependence can remain minimal or even zero for CartPole

## Phase 4. Keep Unified Value Ruler As Secondary Stabilizer

The current `target value ruler` work is not wasted.

It should remain as:

- secondary critic geometry stabilizer
- not primary bridge trainer

## Why This Is The Best Fit For `iron wall`

Because it respects the philosophy:

- predictive truth should not be reward-corrupted
- policy/control semantics should still be trainable

So the final system becomes:

- truth stays upstream
- bridge stays supervised
- policy stays downstream

That is architecturally clean.

## One-Sentence Summary

The elegant fix is:

`do not open the wall; finish the bridge`

More concretely:

- revive the already-designed but currently dead bridge losses,
- decouple bridge maintenance from wall closure,
- and align actor-use policy space directly with replay open-loop teacher policy space.

That is the repair most consistent with both the codebase and the design philosophy.
