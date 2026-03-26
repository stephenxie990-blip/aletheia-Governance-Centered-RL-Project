# CartPole Pure-Imag `v167` MC Anchor Absolute Calibration Validation Run

Date: 2026-03-15
Status: complete

## Run

- Effective validation run:
  - `outputs/exp_seed42_v167_h15_light_sc_mainline_mc_anchor_abs_calib_2500_20260315_02`
- Crash artifact from first launch:
  - `outputs/exp_seed42_v167_h15_light_sc_mainline_mc_anchor_abs_calib_2500_20260315_01`
- Core setup:
  - `horizon=15`
  - pure imagination
  - `v166` mainline retained:
    - light `SC`
    - `policy open-loop consistency`
    - `single-ruler`
  - new Phase 1 addition:
    - `lambda_value_real_anchor = 0.15`
    - `value_real_anchor_use_mc_returns = true`
    - `value_real_anchor_corridor_quantile = 0.75`

## Launch Note

第一次启动的 `_01` 不是训练结论，而是实现 bug：

- `sample_sequences_with_remaining()` 在 valid episode 子集上采样
- 但把子集局部 `ep_idx` 直接写进了 `episode_ids`
- `_build_real_batch()` 回查 replay 完整 episode 做 full-MC reconstruction 时，需要的是 buffer 全局 episode id
- 当前面存在较短、被 valid 过滤掉的 episode 时，会直接对错轨迹并越界

该问题已在正式验证前修复，因此本报告只以 `_02` 为有效 run。

## Validation Intent

这条 run 的目的不是继续堆新机制，而是回答一个更窄也更关键的问题：

`只在 critic side 加 replay full-MC absolute anchor，能不能把 high-value corridor 上的 common-mode inflation 真正压住？`

因此本轮刻意保持：

- 不改 actor calibrated branch
- 不改 controller / rescue
- 不引入新的 critic tail bootstrap
- 不恢复 legacy critic path

## Eval Curve

- `250 -> 9.2`
- `500 -> 10.0`
- `750 -> 60.4`
- `1000 -> 112.0`
- `1250 -> 467.8`
- `1500 -> 73.6`
- `1750 -> 329.8`
- `2000 -> 177.8`
- `2250 -> 108.4`
- `2500 -> 43.8`

关键口径：

- best eval during training:
  - `467.8 @ 1250`
- final current-checkpoint eval:
  - `43.8 @ 2500`

补充说明：

- `summary.json` 末尾的 `final_current/final_best=467.8` 对应的是训练结束后重新载入 `best.pt` 的最终评估
- 判断“当前 checkpoint 是否守住 corridor”必须以 `eval_history.jsonl` 的 `2500 -> 43.8` 为准

## Direct Comparison vs `v166`

对比 run：

- `v166`:
  - `outputs/exp_seed42_v166_h15_light_sc_bridge_phase1_policy_open_loop_single_ruler_mainline_2500_20260315_01`
- `v167`:
  - `outputs/exp_seed42_v167_h15_light_sc_mainline_mc_anchor_abs_calib_2500_20260315_02`

两条曲线：

- `v166`
  - `9.2, 89.2, 110.8, 124.0, 187.6, 140.8, 140.8, 96.0, 82.0, 17.4`
- `v167`
  - `9.2, 10.0, 60.4, 112.0, 467.8, 73.6, 329.8, 177.8, 108.4, 43.8`

结论非常直接：

1. `v167` 明显提高了可达上限
   - `v166 best = 187.6 @ 1250`
   - `v167 best = 467.8 @ 1250`
2. `v167` 没有根治后期 current-checkpoint 回撤
   - `v166 final current = 17.4 @ 2500`
   - `v167 final current = 43.8 @ 2500`
3. 这意味着 MC anchor 不是无效修复
   - 它显著提升了前中段价值语义质量与上限
   - 但它还不足以单独建立长期 corridor 保持

## Internal Metrics

### Early phase: Phase 1 really activates

`step 100`

- `critic/value_real_anchor_is_mc = 1.0`
- `critic/real_mc_value_gap_abs_mean = 5.22`
- `critic/value_target_gap_abs_mean = 4.66`
- `imag/open_loop_audit_teacher_value_abs_to_short_return_mean = 4.41`
- `imag/open_loop_audit_imag_value_abs_to_short_return_mean = 0.58`
- `actor/online_adv_mean = +4.22`

`step 200`

- `critic/value_real_anchor_is_mc = 1.0`
- `critic/real_mc_value_gap_abs_mean = 5.83`
- `critic/value_target_gap_abs_mean = 5.39`
- `actor/online_adv_mean = +5.39`

这些数据证明：

- full-MC anchor 的确进了主链
- 而且早期 value semantics 没有像历史坏 run 那样快速爆炸

### Mid phase: model can reach high score corridor

`step 1000`

- `critic/real_mc_value_gap_abs_mean = 4.36`
- `critic/value_target_gap_abs_mean = 1.15`
- `teacher_value_abs_to_short_return = 12.70`
- `imag_value_abs_to_short_return = 14.49`
- `actor/online_adv_mean = +0.82`

对应：

- `eval@1000 = 112.0`
- `eval@1250 = 467.8`

说明这条 run 不是“太保守学不动”。

更准确地说：

- 它先把 critic 早期绝对语义拉回真实世界
- 之后策略确实获得了接近 solved 的推进能力

### Late phase: corridor still cannot be held

`step 1500`

- `critic/real_mc_value_gap_abs_mean = 19.11`
- `critic/value_target_gap_abs_mean = 22.07`
- `teacher_value_abs_to_short_return = 30.89`
- `imag_value_abs_to_short_return = 36.72`
- `actor/online_adv_mean = -21.86`

`step 2000`

- `critic/real_mc_value_gap_abs_mean = 22.31`
- `critic/value_target_gap_abs_mean = 30.03`
- `teacher_value_abs_to_short_return = 43.78`
- `imag_value_abs_to_short_return = 45.92`
- `actor/online_adv_mean = -29.95`

`step 2500`

- `critic/real_mc_value_gap_abs_mean = 24.41`
- `critic/value_target_gap_abs_mean = 29.61`
- `teacher_value_abs_to_short_return = 33.80`
- `imag_value_abs_to_short_return = 46.67`
- `actor/online_adv_mean = -29.33`

这三组点把病理顺序钉得很清楚：

- absolute anchor 在前半程有效
- 但到中后段，critic 又重新抬到 return 上方
- actor 继续吃到深负 advantage
- current checkpoint 随之持续回撤

## What This Run Proves

### Proven

1. replay full-MC anchor 是有效的上游修复，不是空补丁
   - 它真实改善了前中段绝对价值语义
   - 并且显著提高了可达性能上限
2. common-mode inflation 诊断是对的
   - 因为加上真正外生锚之后，早期确实不再立刻共漂
3. 但 current failure mode 也被进一步收紧了
   - 现在的问题不是“早期就学不起来”
   - 而是“冲到高分 corridor 之后，长期维持机制仍不足”

### Not Yet Solved

1. 这条 run 没有守住 solved
   - `467.8 @ 1250`
   - `43.8 @ 2500`
2. replay MC anchor 没能单独阻止后期再通胀
   - `real_mc_gap_abs`
   - `value_target_gap_abs`
   - `teacher/imag value semantics`
   在 `1500+` 后一起重新恶化
3. 因此 Phase 1 不是终局修复
   - 它只修到了“绝对语义锚缺失”
   - 没修到“actor-use corridor 的长期保持”

## Engineering Judgment

这条 run 的最准确评价是：

`Phase 1 有效，但只解决了前半个问题。`

更具体地说：

- 它修到了：
  - early critic family common-mode inflation
  - early absolute semantic grounding
- 它还没修到：
  - late-stage actor-use corridor maintenance
  - lambda return -> actor loss -> critic update 这一闭环在高分后的再抬高

因此下一刀不应继续停留在：

- 更强的 critic anchor
- controller rescue
- 旧 critic 路径回退

下一刀应直接进入：

- actor-use policy-space corridor maintenance
- 也就是让 actor 真正吃到的 imagined corridor 本身具备长期 open-loop consistency / sustain contract

## Final Conclusion

`v167` 给出的最终结论可以压缩成一句话：

`MC absolute calibration 证明“绝对锚缺失”确实是病灶的一部分，但它单独不足以防止高分后的 actor-use corridor 再次失稳。`

因此现在最合理的工程顺序是：

1. 保留 Phase 1 的 MC anchor，因为它是有效底座
2. 下一步不要再加 controller
3. 直接把第二刀打到：
   - policy-space open-loop corridor maintenance
   - 让 actor/critic 在高分 corridor 上不仅能进，还能守住
