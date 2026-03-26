# CartPole Pure-Imag H15 Light-SC-Only Real Run (2026-03-14)

## Run

- Artifact directory:
  - `outputs/exp_seed42_v151_h15_light_sc_only_2500_20260314_01`
- Command:
  - `./.venv/bin/python scripts/cartpole_train.py --env CartPole-v1 --update-steps 2500 --collect-steps-per-cycle 32 --train-steps-per-cycle 4 --seed 42 --device auto --save outputs/exp_seed42_v151_h15_light_sc_only_2500_20260314_01 --imagination-only --imagination-horizon 15 --enable-eval --eval-interval 250 --eval-episodes 5 --eval-max-steps 500 --log-interval 50 --save-interval 250 --pretrain-ratio 0.05 --warmup-ratio 0.0 --overrides '{"rssm_msc":{"enabled":false},"rssm_shortcut_consistency":{"enabled":true,"horizons":[2],"loss_scale":0.1,"sample_ratio":0.25,"max_starts":2}}'`
- Supported entrypoint base under test:
  - pure-imag CartPole default `imagination_horizon = 15`
  - default critic stabilizers remain on
  - `MSC` explicitly disabled
  - `SC` kept on in a lighter single-axis form

## Observed Outcome

- Full eval trajectory:
  - `250 -> 40.4`
  - `500 -> 137.8`
  - `750 -> 161.8`
  - `1000 -> 43.6`
  - `1250 -> 500.0`
  - `1500 -> 190.0`
  - `1750 -> 96.0`
  - `2000 -> 16.0`
  - `2250 -> 62.6`
  - `2500 -> 18.4`
- Best observed eval:
  - `500.0 @ step 1250`
- Final eval:
  - `18.4 @ step 2500`

## What Changed Mechanistically

- This run did not behave like `v150`.
- It also did not stay stable like a solved mainline.
- Instead it produced a wider but still fragile learning corridor:
  - early stage was modest rather than explosive
  - mid stage achieved real balance and briefly reached solved-level eval
  - late stage still drifted out of corridor and failed to hold performance
- Training-internal snapshots support that reading:
  - `step 3616`: `Return=31.55`, `Amu=0.294`
  - `step 4000`: `Return=26.83`, `Amu=0.916`
  - `step 19616`: `Return=30.92`, `Amu=-5.070`
  - `step 20000`: `Return=35.13`, `Amu=-4.418`
- In plain terms:
  - light `SC` under `horizon=15` helped the policy enter a workable corridor;
  - but critic calibration on imagined states still degraded later, and actor advantage eventually turned negative again.

## Judgment

- `SC` should no longer be treated as "probably useless" under DreamerV3-standard `horizon=15`.
- `v151` is positive evidence that a light imagined-geometry constraint can help open the corridor.
- But `SC` alone is not enough to stabilize late-stage pure-imag CartPole.
- The remaining bottleneck still looks closer to:
  - imagined rollout drift reappearing later,
  - critic / target-critic calibration lag on imagined states,
  - actor updates becoming pessimistic after the mid-stage peak.

## Decision

- Keep:
  - default `horizon=15`
  - the supported entrypoint wiring for `SC/MSC/NST`
  - the earlier default critic stabilizers
- Update the next repair direction:
  - retain light `SC`
  - strengthen late-stage critic anchoring earlier and more explicitly
  - avoid reintroducing `MSC` until this lighter critic-calibration axis is isolated
