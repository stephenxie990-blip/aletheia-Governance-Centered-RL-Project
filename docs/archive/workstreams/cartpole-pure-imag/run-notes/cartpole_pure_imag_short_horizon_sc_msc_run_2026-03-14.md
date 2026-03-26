# CartPole Pure-Imag Short-Horizon + SC/MSC Real Run (2026-03-14)

## Run

- Artifact directory:
  - `outputs/exp_seed42_v150_short_horizon_sc_msc_2500_20260314_01`
- Command:
  - `./.venv/bin/python scripts/cartpole_train.py --env CartPole-v1 --update-steps 2500 --collect-steps-per-cycle 32 --train-steps-per-cycle 4 --seed 42 --device auto --save outputs/exp_seed42_v150_short_horizon_sc_msc_2500_20260314_01 --imagination-only --enable-eval --eval-interval 250 --eval-episodes 5 --eval-max-steps 500 --log-interval 50 --save-interval 250 --pretrain-ratio 0.05 --warmup-ratio 0.0`
- Supported entrypoint defaults under test:
  - pure-imag CartPole default `imagination_horizon -> 8`
  - default `rssm_msc = {enabled=True, horizons=(1,4,8), loss_scale=0.25}`
  - default `rssm_shortcut_consistency = {enabled=True, horizons=(2,4), loss_scale=0.25, sample_ratio=0.5, max_starts=4}`
  - plus existing default critic stabilizers and `imag_continue_prob_cap=0.95`

## Observed Outcome

- This run was stopped early after the failure pattern became unambiguous.
- Eval trajectory written before stop:
  - `250 -> 120.8`
  - `500 -> 9.2`
  - `750 -> 9.6`
- Best observed eval:
  - `120.8 @ step 250`
- No evidence of a later recovery corridor appeared after the early collapse.

## What Changed Mechanistically

- The new stack did not produce the previous `v149` shape of "late-stage oscillation after partial progress".
- Instead it produced a much more conservative but much less learnable regime:
  - training-internal episode return stayed around low `20s-30s`
  - `train/active_horizon` stayed at `8.0`
  - many consistency batches reported high `no done` ratios
  - imagined actor advantage turned clearly negative again
- Late sampled rows from `train_metrics.jsonl` showed:
  - `step 950`: `actor/online_adv_mean = -3.30`
  - `step 950`: `imag/value_target_gap_abs_mean = 3.93`
  - `step 950`: `imag/signal_model_freshness = 0.60`
  - controller stage still `idle`

## Judgment

- This combination is not a good next-mainline candidate.
- The failure is not "still late drift but milder".
- The failure is "too much upstream regularization plus shorter horizon removed the narrow learning corridor entirely".
- In plain terms:
  - it reduced hallucinated long-rollout freedom,
  - but it also removed the useful gradient corridor before the policy had really learned balance,
  - so performance peaked briefly and then collapsed to near-random behavior.

## Decision

- Keep the API wiring fix that makes `rssm_msc / rssm_shortcut_consistency` actually reachable from supported entrypoints.
- Do **not** keep the pure-imag CartPole default package of:
  - `horizon=8`
  - `MSC on`
  - `SC on`
- Next repair should separate these axes instead of stacking them all at once:
  - test shorter horizon alone,
  - then test lighter SC alone,
  - only then revisit MSC.
