# CartPole Pure-Imag `v168` Policy Corridor Maintenance Validation Run

Date: 2026-03-15
Status: complete

## Run

- Output:
  - `outputs/exp_seed42_v168_h15_mc_anchor_policy_corridor_maintenance_2500_20260315_01`
- Core setup:
  - `horizon=15`
  - pure imagination
  - Phase 1 `MC absolute anchor` retained:
    - `lambda_value_real_anchor = 0.15`
    - `value_real_anchor_use_mc_returns = true`
    - `value_real_anchor_corridor_quantile = 0.75`
  - `v167` mainline retained:
    - light `SC`
    - `policy open-loop consistency`
    - `single-ruler`
  - new Phase 2 addition:
    - `adaptive_imag_policy_open_loop_consistency_value_scale = 0.35`
    - `adaptive_imag_policy_open_loop_consistency_late_step_boost = 1.0`

## Validation Intent

这条 run 的目标不是重新证明 `MC anchor` 有没有用，
而是验证第二刀是否真的补上了：

`actor-use policy-space corridor maintenance`

更具体地说，就是看它能不能同时做到：

1. 保留 `v167` 已经证明有效的上限提升
2. 让中后段 current checkpoint 不再像 `v167` 那样持续掉穿

## Eval Curve

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

关键口径：

- best eval during training:
  - `432.6 @ 750`
- final current-checkpoint eval:
  - `58.4 @ 2500`

补充说明：

- `summary.json` 末尾的 `final_current/final_best=432.6` 对应的是训练结束后重新载入 `best.pt` 的最终评估
- 判断“当前 checkpoint 是否守住 corridor”仍必须以 `eval_history.jsonl` 的 `2500 -> 58.4` 为准

## Direct Comparison vs `v167`

对比 run：

- `v167`:
  - `outputs/exp_seed42_v167_h15_light_sc_mainline_mc_anchor_abs_calib_2500_20260315_02`
- `v168`:
  - `outputs/exp_seed42_v168_h15_mc_anchor_policy_corridor_maintenance_2500_20260315_01`

两条曲线：

- `v167`
  - `9.2, 10.0, 60.4, 112.0, 467.8, 73.6, 329.8, 177.8, 108.4, 43.8`
- `v168`
  - `154.2, 257.4, 432.6, 423.8, 186.0, 123.2, 258.2, 154.8, 340.0, 58.4`

这组对比非常重要：

1. `v168` 的前中段推进力明显更强
   - `v167 @250 = 9.2`
   - `v168 @250 = 154.2`
   - `v167 @750 = 60.4`
   - `v168 @750 = 432.6`
2. `v168` 的后段恢复能力也明显更强
   - `v167 @2250 = 108.4`
   - `v168 @2250 = 340.0`
3. 但它仍然没守住 final current checkpoint
   - `v167 @2500 = 43.8`
   - `v168 @2500 = 58.4`

所以这不是“没用”，而是：

`第二刀明显有效，但还没有把 current-checkpoint corridor hold 做到完成。`

## Internal Metrics

### Early phase: second cut does not kill push

`step 500`

- `critic/real_mc_value_gap_abs_mean = 4.10`
- `critic/value_target_gap_abs_mean = 3.20`
- `teacher_value_abs_to_short_return = 6.16`
- `imag_value_abs_to_short_return = 4.02`
- `actor/online_adv_mean = +3.17`
- `imag/value_mean = 4.25`
- `imag/return_mean = 7.42`

这说明：

- 第二刀没有像“更强 anchor”那种方案一样直接压死前段推进力
- 早期仍然是 `value < return`、`adv > 0`

### Mid phase: corridor maintenance really changes behavior shape

`step 1000`

- `critic/real_mc_value_gap_abs_mean = 9.72`
- `critic/value_target_gap_abs_mean = 4.87`
- `teacher_value_abs_to_short_return = 21.78`
- `imag_value_abs_to_short_return = 20.85`
- `actor/online_adv_mean = -4.16`
- `imag/value_mean = 16.79`
- `imag/return_mean = 12.64`
- `wm/policy_open_loop_consistency_loss = 3.83`
- `wm/policy_open_loop_consistency_value_loss_mean = 7.18`
- `wm/policy_open_loop_consistency_step_weight_mean = 1.5`

这个点和 `v167 @1000` 的差异很关键：

- `v167` 当时内部更“干净”，但 eval 只有 `112`
- `v168` 这里已经出现轻度 `value > return`
- 但 eval 却达到 `423.8`

这说明第二刀带来的不是“内部误差完全消失”，
而是：

`即使进入轻度错位区，corridor 仍能保持高性能更久。`

### Late phase: still not a solved hold

`step 1500`

- `critic/real_mc_value_gap_abs_mean = 18.11`
- `critic/value_target_gap_abs_mean = 4.36`
- `teacher_value_abs_to_short_return = 29.03`
- `imag_value_abs_to_short_return = 20.74`
- `actor/online_adv_mean = -3.92`
- `imag/value_mean = 24.01`
- `imag/return_mean = 20.09`

`step 2000`

- `critic/real_mc_value_gap_abs_mean = 14.89`
- `critic/value_target_gap_abs_mean = 4.69`
- `teacher_value_abs_to_short_return = 27.50`
- `imag_value_abs_to_short_return = 21.76`
- `actor/online_adv_mean = -4.67`
- `imag/value_mean = 23.61`
- `imag/return_mean = 18.94`

`step 2500`

- `critic/real_mc_value_gap_abs_mean = 16.17`
- `critic/value_target_gap_abs_mean = 9.18`
- `teacher_value_abs_to_short_return = 28.18`
- `imag_value_abs_to_short_return = 28.54`
- `actor/online_adv_mean = -9.09`
- `imag/value_mean = 29.64`
- `imag/return_mean = 20.55`

这些点说明：

- 第二刀明显压小了 late-stage value mismatch 的恶化幅度
  - 对比 `v167 @2000`
    - `value_target_gap_abs`: `30.03 -> 4.69`
    - `online_adv_mean`: `-29.95 -> -4.67`
- 但最终仍没有彻底消失
  - `2500` 时仍是 `value > return`
  - `online_adv_mean` 仍为负

## What This Run Proves

### Proven

1. 第二刀是有效修复，不是表面正则
   - 它显著提高了前中段高分进入能力
   - 也显著提高了后段恢复能力
2. `policy-space corridor maintenance` 确实打中了 `v167` 暴露的缺口
   - 当前系统不再表现为单调下坠式 late-stage collapse
   - 而变成可回升、可再进入高分 corridor 的形态
3. 它还改变了内部病理形状
   - `v167` 的问题更像深度后期再通胀
   - `v168` 更像中度持续错位 + 振荡恢复

### Not Yet Solved

1. final current checkpoint 仍没守住
   - `58.4 @ 2500`
2. 晚期仍保留：
   - `value > return`
   - `online_adv < 0`
3. 因而第二刀虽然修了：
   - corridor entry
   - corridor re-entry
   但还没修完：
   - corridor final hold

## Engineering Judgment

这条 run 最准确的评价是：

`第二刀把 failure mode 从“后段单向塌陷”改成了“高位振荡但仍会失守”。`

更具体地说：

- 它明显比 `v167` 更强
- 也明显比 `v167` 更能恢复
- 但还没有让 current checkpoint 在 late stage 变成真正稳定

所以这不是失败，而是进一步收敛：

- `MC anchor` 修了“绝对锚缺失”
- `policy corridor maintenance` 修了“corridor entry / re-entry”
- 剩余问题继续收敛为：
  - RL-side actor/critic value ruler 在高分 corridor 上仍然偏欠阻尼
  - 最终 hold 仍不足

## Final Conclusion

`v168` 的一句话结论是：

`第二刀已经明显打中病灶，但它把系统从“暴跌型失败”推进成了“振荡型失败”，离最终稳定只差最后一层。`

下一步不该回退，
而应该继续顺着这条主线往更深一层诊断：

- 为什么 corridor 已经能进、也能重新进，
- 但最后还是守不住 current checkpoint。
