# CartPole Pure-Imag `v165` Policy Open-Loop Consistency Validation Run

Date: 2026-03-15
Status: complete

## Run

- Output:
  - `outputs/exp_seed42_v165_h15_light_sc_bridge_phase1_policy_open_loop_2500_20260315_01`
- Core setup:
  - `horizon=15`
  - pure imagination
  - light `SC`
  - `MSC=false`
  - Phase 1 bridge contract restoration already active
  - 新增 `policy-space open-loop consistency loss`
  - controller 不作为主修复位点，实际本 run 全程 `idle`

## Validation Configuration

主要 override：

- `rssm_msc.enabled = false`
- `rssm_shortcut_consistency.enabled = true`
- `rssm_shortcut_consistency.horizons = [2]`
- `rssm_shortcut_consistency.loss_scale = 0.1`
- `adaptive_imag_policy_open_loop_consistency_weight = 0.15`
- `adaptive_imag_policy_open_loop_consistency_horizon = 3`
- `adaptive_imag_policy_open_loop_consistency_delta = 0.5`
- `adaptive_imag_policy_open_loop_consistency_high_value_boost = 1.0`
- `adaptive_imag_policy_open_loop_consistency_high_value_quantile = 0.75`

## Eval Curve

- `250 -> 9.6`
- `500 -> 103.8`
- `750 -> 65.4`
- `1000 -> 166.0`
- `1250 -> 30.0`
- `1500 -> 202.0`
- `1750 -> 293.8`
- `2000 -> 243.0`
- `2250 -> 500.0`
- `2500 -> 19.8`

关键结果：

- best eval during training:
  - `500.0 @ 2250`
- final current-checkpoint eval:
  - `19.8 @ 2500`

说明：

- `summary.json` 末尾的 `final_current/final_best=500.0` 对应的是训练结束后重新载入 `best.pt` 的最终评估；
- 判断“当前 checkpoint 是否守住 solved”应以 `eval_history.jsonl` 的 `2500 -> 19.8` 为准。

## Direct Comparison vs `v164`

对比 run：

- `v164`: `outputs/exp_seed42_v164_h15_light_sc_bridge_phase1_validation_2500_20260315_01`
- `v165`: `outputs/exp_seed42_v165_h15_light_sc_bridge_phase1_policy_open_loop_2500_20260315_01`

### 1. 能力上限

- `v164 best = 117.6 @ 2500`
- `v165 best = 500.0 @ 2250`

这说明 `policy-space open-loop consistency` 明确拓宽了可达高分 corridor。

### 2. 早中期 imagined drift

在相同观察口径下：

- `v164 @1000`
  - `actor/online_adv_mean = -15.55`
  - `critic/value_target_gap_abs_mean = 17.00`
  - `imag/open_loop_audit_value_gap_mean = 17.37`
- `v165 @1000`
  - `actor/online_adv_mean = 1.76`
  - `critic/value_target_gap_abs_mean = 2.78`
  - `imag/open_loop_audit_value_gap_mean = 0.96`

这说明新的 policy-space loss 明确压住了“早期 actor-use imagined manifold 脱锚”。

### 3. 后期主病灶是否消失

到了后段：

- `v164 @2500`
  - `actor/online_adv_mean = -44.84`
  - `critic/value_target_gap_abs_mean = 49.08`
  - `imag/open_loop_audit_value_gap_mean = 94.94`
- `v165 @2500`
  - `actor/online_adv_mean = -37.88`
  - `critic/value_target_gap_abs_mean = 37.93`
  - `imag/open_loop_audit_value_gap_mean = 4.80`

结论非常明确：

- `imagined open-loop semantic drift` 被大幅压住了；
- 但 `online critic / actor-use value ruler mismatch` 仍在后段持续放大。

也就是说：

- `v164` 是 imagined branch 先坏；
- `v165` 则变成 imagined branch 更稳，但 online critic 尺子后续继续漂高。

## What This Run Proves

### Proven

1. `policy-space open-loop consistency loss` 不是表面正则：
   - 它真实改变了纯 imagined training 的性能上限。
2. 它确实修到了此前锁定的主前因：
   - 早期 self-generated imagined rollout semantic drift。
3. controller 不是这条 run 的承载物：
   - `post_entry_audit` / `controller_stage` 全程 `idle`。

### Not Yet Solved

1. 模型仍然不能稳定守住 solved：
   - `2250 -> 500.0`
   - `2500 -> 19.8`
2. 后期仍出现明显双标尺现象：
   - `Vmu` 持续显著高于 `Rmu`
   - `actor/online_adv_mean` 深度转负
   - 但外部 eval 还能一段时间维持高位

这意味着：

- 问题已经不再是“world model imagined rollout 很早就不准”；
- 而是“online critic actor-use 空间与 target / teacher semantic space 的长期对账机制仍不足”。

## Highest-Confidence Diagnosis After `v165`

`v165` 使当前根因判断进一步收紧为两层：

### 已经被证明显著改善的层

- self-generated imagined branch 的早期语义漂移
- actor-use policy feature manifold 的 replay open-loop 对齐

### 剩余未修复的层

- late-stage online critic 尺子抬高
- actor 使用的 online advantage 被这把漂高的尺子拖负
- solved corridor 缺少长期保持机制

因此当前主病灶不再是：

- `world model imagined rollout globally inaccurate`

而是：

- `target critic semantic space` 与 `online critic actor-use space` 在高分后仍会重新错位

## Engineering Judgment

这一刀是有效且必要的。

但它把问题从：

- “早期 imagined manifold 脱锚”

推进成了更深一层的：

- “高分后的 online critic / actor-use ruler sustain failure”

所以它不是终局修复，而是把真正剩余病灶从噪声里分离出来。

## Recommended Next Step

下一刀不应再回头做：

- controller rescue
- legacy critic path
- 更强的纯 feature regularization

下一刀应直接做：

- `target critic semantic space` vs `online critic actor-use space` 的 single-ruler unification

更具体地说，应优先研究：

1. 如何让 online critic 在 actor 真正使用的 imagined states 上长期对齐 target / teacher 尺子；
2. 如何让 lambda return -> actor loss -> critic update 这一闭环在高分后不再重新抬高 `online value ruler`；
3. 如何把 solved corridor 的“进入能力”升级成“保持能力”。
