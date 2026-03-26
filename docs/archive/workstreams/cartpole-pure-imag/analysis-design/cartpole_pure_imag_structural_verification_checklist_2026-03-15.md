# CartPole Pure-Imag Structural Verification Checklist

Date: 2026-03-15

## Goal

Pin the current root cause with structural evidence instead of continuing patch-first iteration.

This checklist treats the `iron wall` design as intentional and valid. The question is not whether the wall should exist. The question is whether the bridge behind the wall is actually trained, anchored, and observable strongly enough.

## Executive Conclusion

The current main problem is not simply "imagined rollout is inaccurate".

The deeper verified root cause is:

1. Predictive truth space and policy/control space are structurally split by design.
2. The bridge modules that should carry predictive truth into actor-use policy features are underconstrained.
3. Under the current pure-imag CartPole mainline, most of that bridge is not receiving sustained training signal at all.
4. As a result, the system can remain internally self-consistent while drifting semantically wrong in the actor-use imagined corridor.

In short:

`iron wall` itself is not the bug.

The bug is that the post-wall bridge contract is incomplete.

## Structural Verification Checklist

| ID | Claim | Status | Evidence | Implication |
|---|---|---|---|---|
| SV-01 | WM predictive loss is teacher-forced and primarily trains the predictive core. | Verified | `TrainingStep` WM update calls `world_model.observe_sequence()` -> `build_trajectory()` -> `compute_loss()` in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L5220) and [aletheia_world_model.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_world_model.py#L7797). Predictive loss inside `PredictiveEngine.compute_loss()` is reward/continue/recon/kl/quant plus consistency terms in [aletheia_world_model.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_world_model.py#L6141). | The core RSSM and predictive heads are trained on replay teacher-forced data, not on actor-induced imagined branches. |
| SV-02 | Actor/critic main training uses self-generated imagined batches, not the teacher-forced WM path. | Verified | Imagined batch comes from `_build_imagined_batch()` in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L9971), which calls `imagine_rollout_differentiable()` in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L5587). | The actor-use corridor lives on a different path from the WM predictive supervision path. |
| SV-03 | Replay-side semantic alignment tools do not directly constrain the actor's current-policy imagined manifold. | Verified | `_compute_target_value_consistency_loss()` uses replay suffix actions in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L3322). Open-loop audit also uses replay future actions in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L9191). | Existing semantic repairs are informative and helpful, but they supervise replay continuation geometry, not the actor-induced imagined branch directly. |
| SV-04 | Target critic is not an external truth source. It is only a slow EMA ruler. | Verified | `critic.update_target()` performs soft update in [aletheia_actor_critic.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_actor_critic.py#L1742) and is called after RL updates in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L5180). | Target critic can stabilize internal ruler mismatch, but cannot guarantee semantic truth if both critics drift together. |
| SV-05 | `router` is optimized only through the RL-side extra-parameter bucket, not through WM optimizer. | Verified | `TrainingLoop` puts non-WM non-actor non-critic params into `extra_rl_params` in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L6805). Live optimizer introspection confirmed `rl_opt_params = 115907`, all of them router params, with zero WM/actor/critic overlap. | The router is structurally a post-wall RL-side bridge module. If RL path does not give it gradient, it has no backup trainer. |
| SV-06 | `control_head`, `projection`, and `abstractor` belong to `world_model`, but WM loss does not actually train them. | Verified | `PredictiveEngine.compute_loss()` hardcodes `projection_loss = 0` and `abstractor_loss = 0` in [aletheia_world_model.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_world_model.py#L6175). Live backward test on current agent gave nonzero `state_transition` gradient, but zero gradients for `_control_head`, `_projection`, and `_abstractor`. | These bridge components exist inside WM, but current WM training does not supervise them. |
| SV-07 | `ControlHead` auxiliary losses are defined but not wired into the training loss. | Verified | `ControlHead.compute_aux_loss()` defines `L_align` and `L_reg` in [aletheia_world_model.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_world_model.py#L6430). Repo-wide search found no call site for `compute_aux_loss()`. `ConsistencyHead.compute_loss()` defines `L_cons` in [aletheia_world_model.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_world_model.py#L6636), but no training call site was found. | The bridge has declared self-supervision hooks, but they are currently dead definitions rather than active contract. |
| SV-08 | World-model imagination explicitly freezes WM params and starts from detached state. | Verified | `initial_wm_state.detach_all()` is used in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L5611). WM and critic params are frozen during imagination in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L5634). | The imagination path intentionally blocks RL gradients from flowing back into WM internals. This is consistent with `iron wall`, but it also means bridge modules inside WM need their own explicit training path. |
| SV-09 | `x_proj` and `z_task` are detached before entering policy-space routing. | Verified | In `forward_context()`, `x_proj_raw` and `z_task_raw` are detached in [aletheia_world_model.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_world_model.py#L7716). In `forward_imagination()`, `x_proj` and `z_task` are detached in [aletheia_world_model.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_world_model.py#L7762). | The policy bridge consumes detached projections by design, so predictive truth does not automatically keep the bridge aligned. |
| SV-10 | Actor-use policy features are routed through a separate bridge path: `x_proj/x_t + s_ctrl + z_task -> router -> f_policy`. | Verified | `policy_from_wm_state()` builds policy features through router in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L6574). Router entry is `forward_components()` in [aletheia_world_model.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_world_model.py#L4533). | Actor/critic do not directly consume predictive RSSM loss heads. They consume a bridged policy space. |
| SV-11 | Under current pure-imag CartPole defaults, imagined critic features are detached before critic update. | Verified | `run_train()` injects `detach_critic_features_on_imagination=true` for CartPole pure-imag in [aletheia_api.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py#L3136) and again in the train-config construction path in [aletheia_api.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py#L3224). Current `v163` resolved config confirms this in [config_resolved.json](/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v163_h15_multih_sc_highvalue_semantic_anchor_medium_target_ruler_soft_gate_2500_20260315_01/config_resolved.json#L184). | The mainline intentionally protects critic from imagined feature drift, but also cuts the last remaining direct gradient path from imagined critic loss into router features. |
| SV-12 | With `detach_critic_features_on_imagination=true`, router gets zero gradient from imagined RL. | Verified | In RL update, actor uses `feats_flat.detach()` in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L3825), and critic uses `critic_inputs = feats.detach()` when detach flag is on in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L3820). Live gradient test produced router grad `(0.0, 0)` with detach on and nonzero router grad with detach off. | In the current mainline, router is placed in an optimizer bucket but effectively starved of gradient. |
| SV-13 | `control_head` and `projection` get zero RL gradient even when router is allowed to train. | Verified | Live gradient test produced zero grads for `control_head` and `projection` under both detach modes, while router recovered gradient only when detach was disabled. This matches the frozen-WM and detached-state design in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L5611). | The bridge inside WM is not merely weakly trained. In current structure it is effectively untrained by RL. |
| SV-14 | Value normalization can preserve internal consistency while external semantic quality worsens. | Verified | Critic target normalization updates on imagined targets in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L3891), while replay anchor uses `update=False` in [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L4267). | The system can internally rationalize scale drift even while semantic truth is drifting. This amplifies the bridge-undertraining problem by hiding it behind stable internal metrics. |

## Root-Cause Tree

### Symptom Layer

- The model can enter a high-score corridor but cannot keep it.
- Internal rulers can look more aligned while real control performance still degrades.
- Controller may stay `idle`, so the failure is not explained by stage rescue logic.

### First-Layer Pathology

- Actor-use imagined corridor drifts semantically away from true short-horizon value semantics.
- Online actor/critic geometry can become self-consistent but externally wrong.

### Second-Layer Cause

- Replay-side semantic corrections supervise replay suffix geometry.
- Actor/critic main training uses current-policy imagined geometry.
- The two spaces are not guaranteed to share one trained bridge.

### Deeper Structural Cause

- `iron wall` isolates predictive truth from policy pollution.
- But the project does not currently give the post-wall bridge a complete training contract.
- `router` is RL-only.
- `control_head/projection/abstractor` are in WM but receive zero WM-loss gradient.
- Current pure-imag defaults additionally detach imagined critic features, which removes router's main remaining gradient path.

### Final Diagnosis

The present root cause is:

`policy/control bridge undertraining behind a valid iron wall`

More explicitly:

- predictive core is trained,
- replay-side semantic audits exist,
- target critic stabilizers exist,
- but the modules that turn predictive truth into actor-use policy features are not continuously trained and anchored.

That is why the system can look locally repaired while still losing the long corridor.

## What This Diagnosis Rules Out

- It is not best explained as a controller-stage bug.
- It is not best explained as a single formula error in lambda return.
- It is not best explained as "world model predictive heads are globally broken".
- It is not best explained as "target critic alone can solve truth alignment".

## What Must Be True For A Real Fix

Any real fix must satisfy all of the following:

1. Preserve `iron wall` as a truth-protection principle.
2. Give the post-wall bridge an actual training contract.
3. Constrain actor-use imagined manifold directly, not only replay suffixes.
4. Avoid replacing upstream alignment with downstream rescue scaffolding.
5. Make bridge drift observable with metrics, not only inferable after score collapse.

## Current Repair Implication

If we continue from this diagnosis, the next repair target should not be "controller stronger".

It should be:

`how to train and anchor the post-wall bridge without breaking iron wall`

That is the real engineering question now.
