# CartPole Pure-Imag `v166` Single-Ruler Mainline Validation Run

Date: 2026-03-15
Status: complete

## Run

- Output:
  - `outputs/exp_seed42_v166_h15_light_sc_bridge_phase1_policy_open_loop_single_ruler_mainline_2500_20260315_01`
- Core setup:
  - `horizon=15`
  - pure imagination
  - `v165` light baseline:
    - `rssm_msc.enabled = false`
    - `rssm_shortcut_consistency = {enabled=true, horizons=[2], loss_scale=0.1, sample_ratio=0.25, max_starts=2}`
    - `adaptive_imag_policy_open_loop_consistency_* = v165 same`
  - mainline default entry additionally injected:
    - `adaptive_imag_actor_use_target_value_ruler_enabled = true`
    - `adaptive_imag_actor_use_target_value_ruler_blend = 1.0`
    - `adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled = true`
    - `adaptive_imag_actor_use_target_value_ruler_gap_margin = 1.0`
    - `adaptive_imag_actor_use_target_value_ruler_critic_distill_weight = 0.15`

## Eval Curve

- `250 -> 9.2`
- `500 -> 89.2`
- `750 -> 110.8`
- `1000 -> 124.0`
- `1250 -> 187.6`
- `1500 -> 140.8`
- `1750 -> 140.8`
- `2000 -> 96.0`
- `2250 -> 82.0`
- `2500 -> 17.4`

关键口径：

- best eval during training:
  - `187.6 @ 1250`
- final current-checkpoint eval:
  - `17.4 @ 2500`
- `summary.json` 末尾的 `final_current/final_best=187.6` 对应的是训练结束后重新载入 `best.pt` 的最终评估；
- 判断“当前 checkpoint 是否守住 corridor”仍应以 `eval_history.jsonl` 的 `2500 -> 17.4` 为准。

## Comparison vs `v165`

- `v165`:
  - `9.6, 103.8, 65.4, 166.0, 30.0, 202.0, 293.8, 243.0, 500.0, 19.8`
- `v166`:
  - `9.2, 89.2, 110.8, 124.0, 187.6, 140.8, 140.8, 96.0, 82.0, 17.4`

直接差异：

1. `v166` 明显修好了 `1000-1500` 这一段的 corridor continuity
   - `v165 @1250 = 30.0`
   - `v166 @1250 = 187.6`
2. `v166` 没有复现 `v165` 那种后段脉冲式冲顶
   - `v165 @2250 = 500.0`
   - `v166 @2250 = 82.0`
3. `v166` 也没有完成“长期保持 solved”
   - `v166 @2500 = 17.4`
   - 与 `v165 @2500 = 19.8` 一样，最终 current checkpoint 仍然掉回低分

## Internal Mechanics

关键内部点：

- `@1000`
  - `actor/online_adv_mean = 1.36`
  - `critic/value_target_gap_abs_mean = 2.60`
  - `imag/reference_value_ruler_alpha_mean = 0.076`
  - `imag/reference_value_ruler_alpha_active_fraction = 0.292`
  - `imag/open_loop_audit_value_gap_mean = 1.35`
- `@1250`:
  - 对应最近训练点 `1200`
  - `actor/online_adv_mean = -1.50`
  - `critic/value_target_gap_abs_mean = 3.28`
  - `alpha_mean = 0.0`
  - `open_loop_gap = 0.35`
- `@1500`
  - `actor/online_adv_mean = -7.11`
  - `critic/value_target_gap_abs_mean = 7.34`
  - `alpha_mean = 0.335`
  - `alpha_active_fraction = 0.60`
  - `open_loop_gap = 4.81`
- `@1750`
  - 对应最近训练点 `1700`
  - `actor/online_adv_mean = -39.65`
  - `critic/value_target_gap_abs_mean = 39.73`
  - `alpha_mean = 0.713`
  - `alpha_active_fraction = 0.967`
  - `open_loop_gap = 21.05`
- `@2000`
  - `actor/online_adv_mean = -48.92`
  - `critic/value_target_gap_abs_mean = 48.92`
  - `alpha_mean = 0.028`
  - `alpha_active_fraction = 0.058`
  - `open_loop_gap = 24.27`
- `@2250`
  - 对应最近训练点 `2200`
  - `actor/online_adv_mean = -49.10`
  - `critic/value_target_gap_abs_mean = 49.10`
  - `alpha_mean = 0.054`
  - `alpha_active_fraction = 0.10`
  - `open_loop_gap = 43.20`
- `@2500`
  - `actor/online_adv_mean = -56.51`
  - `critic/value_target_gap_abs_mean = 56.83`
  - `alpha_mean = 0.774`
  - `alpha_active_fraction = 0.883`
  - `open_loop_gap = 70.94`

## Highest-Confidence Diagnosis After `v166`

`v166` 把当前判断再往前推进了一层：

1. single-ruler mainline default 是有效修复，不是伪信号
   - 它显著改善了 `500-1500` 的 corridor continuity
   - 并且介入方式是 state-selective：
     - `1100` 时会强拉回
     - `1200` 又能退回接近 `0`
   - 说明它不是简单把整条 imagined corridor 永久压钝

2. 但它仍然只是“延后并缓冲”晚期失稳，不足以单独根治
   - 到 `1600+` 后，`online_adv_mean` 仍会重新深负
   - `critic/value_target_gap_abs_mean` 仍会重新放大到 `40-56`
   - `open_loop_gap` 也会在晚期重新拉大到 `20-70`

3. 因而剩余问题已经不再是：
   - single-ruler 没接进主线
   - 或 early imagined branch 没被对齐

4. 剩余问题进一步收敛为：
   - local, reactive 的 single-ruler correction 仍然太晚
   - 它可以在冲击发生时回拉 actor-use imagined states
   - 但还不能把 late-stage 整体 online critic inflation 预先钉住

## Conclusion

`v166` 的结论不是失败，而是收敛：

- 它证明了：
  - `single-ruler` 主线化是正确方向
  - 并且已经实质改善前中段 actor-use corridor
- 它也同时证明了：
  - 当前 local soft-gated ruler 仍不足以单独守住 late-stage
  - 因而下一刀不能回到 controller
  - 也不能回到 legacy critic
  - 而应该继续做更前置、更持续的 actor-use corridor maintenance

一句话总结：

`v166` 已把问题从“早期 imagined drift + 双标尺并存”进一步压缩成“晚期整体 online ruler 仍会重新抬高”。  
下一步修的应该是“late-stage proactive corridor maintenance”，而不是“再多加一次事后拉回”。
