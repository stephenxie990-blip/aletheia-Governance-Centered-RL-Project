# CartPole Pure-Imag Phase 1 Implementation: MC Anchor Absolute Calibration

Date: 2026-03-15
Status: implemented and test-verified

## Goal

第一阶段目标是以最小变量修复并验证当前最主要的结构病灶：

- `critic family common-mode inflation`

本阶段坚持：

- 只动 critic-side
- 不动 actor calibrated branch
- 不动 controller
- 不引入 critic tail bootstrap 作为新锚

## What Was Implemented

### 1. Replay full-MC anchor helper

新增：

- `compute_discounted_returns_to_go()`

作用：

- 对完整 episode 的真实 rewards / dones 计算 full Monte Carlo discounted return
- 零 critic 参与
- 零 bootstrap 污染

### 2. Replay sampling metadata for MC reconstruction

`ReplayBuffer.sample_sequences_with_remaining()` 现在额外返回：

- `episode_ids`

作用：

- 让 `TrainingLoop._build_real_batch()` 可以从 sampled sequence 反查完整 episode
- 基于真实 episode rewards/dones 重建该 sequence 对齐的 full-MC target

### 3. Real batch now supports MC-based `value_real`

`TrainingLoop._build_real_batch()` 新增支持：

- `value_real_anchor_use_mc_returns`

开启后：

- real batch 走 `sample_sequences_with_remaining()`
- 用完整 episode 重建 sampled timesteps 对应的 full-MC return
- 写入 `value_real`
- 保留现有 `returns / target_actor / base_actor` 主训练路径不变

这意味着：

- 第一阶段只改变 real anchor
- 不改变 real RL batch 本身的 actor/critic 训练目标

### 4. Corridor gating for real anchor

`TrainingLoop._build_real_batch()` 新增支持：

- `value_real_anchor_corridor_quantile`

开启后：

- 基于 `value_real` 计算高价值 corridor 阈值
- 生成 `value_real_corridor_mask`
- 同步写入：
  - `value_real_corridor_threshold`
  - `value_real_corridor_fraction`
  - `value_real_corridor_active`

### 5. Existing `value_real_anchor` contract now respects corridor mask

`TrainingStep.run_step()` 中现有 `value_real_anchor` 路径已增强：

- 如果 `value_real_batch` 提供 `value_real_corridor_mask`
- 则 critic absolute calibration loss 只在 mask 覆盖区域上生效

同时新增指标：

- `critic/value_real_anchor_is_mc`
- `critic/value_real_anchor_corridor_active`
- `critic/value_real_anchor_corridor_fraction`
- `critic/value_real_anchor_corridor_threshold`
- `critic/real_mc_value_gap_mean`
- `critic/real_mc_value_gap_abs_mean`

## What Was Intentionally Not Changed

本阶段明确保持不动：

- imagined branch actor target/base/advantage 主路径
- single-ruler 主逻辑
- controller / rescue 机制
- target critic 逻辑

因此这次实现是一次非常克制的“外生锚补丁”，不是新一轮大范围结构改造。

## Verification

### New tests

新增并通过：

1. `test_compute_discounted_returns_to_go_matches_full_mc_return`
2. `test_value_real_anchor_corridor_mask_limits_penalty_to_high_value_states`
3. `test_build_real_batch_uses_full_mc_returns_and_corridor_mask_when_enabled`
4. `test_explicit_mc_real_anchor_overrides_propagate_to_training_config`

### Regression verification

执行：

```bash
./.venv/bin/python -m py_compile \
  aletheia/aletheia_train.py \
  aletheia/aletheia_config.py \
  aletheia/aletheia_api.py \
  aletheia/tests/test_replay_buffer_contracts.py \
  aletheia/tests/test_training_entrypoints.py \
  aletheia/tests/test_training_loop_integration.py \
  aletheia/tests/test_run_train_contracts.py
```

通过。

执行：

```bash
./.venv/bin/python -m unittest \
  aletheia.tests.test_replay_buffer_contracts \
  aletheia.tests.test_training_entrypoints \
  aletheia.tests.test_run_train_contracts \
  aletheia.tests.test_pure_imagination_path \
  aletheia.tests.test_training_loop_integration -v
```

结果：

- `266` tests passed

## Current Interpretation

这次实现完成后，第一阶段已经具备了以下能力：

- 对 replay 真实轨迹提供真正外生的 MC anchor
- 在 actor-use real feature space 上校准 online critic
- 且只在 high-value corridor 上施加绝对校准

这正好对应当前诊断中缺失的第三层防线：

- 几何一致性：已有
- 相对双尺对齐：已有
- 绝对语义标定：现在已有第一版

## Next Step

下一步不应再做新的结构扩张，而应做真实验证：

1. 开启一条真实训练 run
2. 观察：
   - `critic/real_mc_value_gap_abs_mean`
   - `critic/value_target_gap_abs_mean`
   - `imag/open_loop_audit_teacher_value_abs_to_short_return_mean`
   - `imag/open_loop_audit_imag_value_abs_to_short_return_mean`
3. 判断：
   - late-stage common-mode inflation 是否明显被压住
   - eval current checkpoint 是否不再 solved 后整体坍塌

若第一阶段真实验证有效，再进入第二阶段：

- actor calibrated branch
- absolute inflation detector 正式入主逻辑
