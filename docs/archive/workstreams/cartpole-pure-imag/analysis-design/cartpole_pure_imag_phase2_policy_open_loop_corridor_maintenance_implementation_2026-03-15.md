# CartPole Pure-Imag Phase 2 Implementation: Policy Open-Loop Corridor Maintenance

Date: 2026-03-15
Status: implemented and test-verified

## Goal

在 `v167` 之后，问题已经收紧为：

- replay full-MC absolute anchor 可以修早期绝对语义
- 但高分之后的 actor-use corridor 仍然守不住

因此第二刀不再继续堆：

- controller rescue
- 更强 replay anchor
- legacy critic path

而是直接补上：

`policy-space open-loop corridor maintenance`

## Core Design

现有 `adaptive_imag_policy_open_loop_consistency` 已经能做：

- replay suffix 动作下
- imagined policy features 对 teacher policy features 的短程 open-loop 对齐

但它本质上仍然偏“几何对齐”，缺两样东西：

1. `value corridor sustain`
   - 即便 feature 很接近，imagined actor-use value semantics 仍可能偏离 replay short-return corridor
2. `late-step emphasis`
   - 当前每个 open-loop step 权重基本均匀，不足以把约束压到更深的 open-loop 段

这次实现就在现有主链上补这两项，而不新开旁路。

## What Was Implemented

### 1. Policy open-loop loss now supports value-semantic sustain

新增配置：

- `adaptive_imag_policy_open_loop_consistency_value_scale`

作用：

- 在原有 feature Huber loss 之外，
- 对 imagined policy features 的 critic value 增加 open-loop corridor 对齐项

具体对齐对象：

- replay suffix short-return target
- teacher policy feature value

也就是说，这次不是只要求：

- `f_imag ~= f_teacher`

还要求：

- `V(f_imag)` 不能脱离 replay suffix value corridor
- 并且不能脱离 teacher actor-use semantic ruler

这正是“从几何对齐升级到 corridor sustain”的核心。

### 2. Policy open-loop loss now supports late-step boost

新增配置：

- `adaptive_imag_policy_open_loop_consistency_late_step_boost`

作用：

- 对 open-loop horizon 中更深的 step 施加更大权重

当前实现：

- 线性 step weighting
- 越靠近 audit horizon 尾端，weight 越大

这让 loss 不再只关心“刚离开 teacher 的头两步对不对”，
而是更关注：

- deeper open-loop segment 能不能继续留在 corridor 里

### 3. Metrics were extended for diagnosis

新增指标：

- `wm/policy_open_loop_consistency_value_scale`
- `wm/policy_open_loop_consistency_value_loss_mean`
- `wm/policy_open_loop_consistency_teacher_value_gap_mean`
- `wm/policy_open_loop_consistency_imag_target_gap_mean`
- `wm/policy_open_loop_consistency_late_step_boost`
- `wm/policy_open_loop_consistency_step_weight_mean`
- per-step:
  - `...value_loss_stepN`
  - `...teacher_value_gap_stepN`
  - `...imag_target_gap_stepN`
  - `...step_weight_stepN`

这使后续 run 可以明确区分：

- geometry 是否对齐
- corridor value semantics 是否对齐
- deeper open-loop steps 是否真的被更强约束

### 4. Override plumbing was updated

新配置已支持从 run entry overrides 透传：

- `adaptive_imag_policy_open_loop_consistency_value_scale`
- `adaptive_imag_policy_open_loop_consistency_late_step_boost`

## Why This Is Elegant

这次实现仍然保持了上游修复原则：

- 不改 actor 主 loss
- 不加 controller 补救
- 不把更多真实锚直接塞进 actor
- 不回退旧 critic path

而是直接在：

- actor 真正使用的 policy-space world-model contract

上补“几何 + 价值 + deeper-step sustain”。

所以它不是下游止血，
而是把 corridor maintenance 放回了它最应该存在的位置。

## Verification

### New targeted tests

新增并通过：

1. `test_policy_open_loop_consistency_value_scale_penalizes_semantic_drift_even_when_features_match`
   - 证明：
     - 即使 teacher / imag features 已经匹配
     - value corridor sustain term 仍会因为 replay short-return mismatch 而发力
2. `test_policy_open_loop_consistency_late_step_boost_emphasizes_deeper_open_loop_steps`
   - 证明：
     - late-step boost 确实会把更深 step 权重抬高
3. `test_explicit_policy_open_loop_corridor_maintenance_overrides_reach_training_config`
   - 证明：
     - 两个新配置可从 run entry 正常透传

### Regression verification

执行：

```bash
./.venv/bin/python -m py_compile \
  aletheia/aletheia_config.py \
  aletheia/aletheia_train.py \
  aletheia/aletheia_api.py \
  aletheia/tests/test_training_entrypoints.py \
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

- `270` tests passed

## Recommended Validation Run

建议下一条真实验证 run 保持最小变量：

- 保留 `v167` 的 Phase 1 MC anchor 底座
- 保留现有 `policy_open_loop_consistency` 与 `single-ruler`
- 仅新增：
  - `adaptive_imag_policy_open_loop_consistency_value_scale`
  - `adaptive_imag_policy_open_loop_consistency_late_step_boost`

建议首轮参数：

- `adaptive_imag_policy_open_loop_consistency_value_scale = 0.35`
- `adaptive_imag_policy_open_loop_consistency_late_step_boost = 1.0`

理由：

- 不过强，不至于一上来把 actor 推进力压死
- 但足够明确地区分：
  - “单纯 feature open-loop”
  - 和“带 corridor sustain 的 policy open-loop”

## Current Interpretation

第二刀当前还只是代码级验证，不是 run-level 结论。

但从结构上，它已经满足三条要求：

1. 修复位置准确
   - 直接打在 actor-use policy-space maintenance
2. 机制优雅
   - 沿现有 mainline loss 升级，而非叠更多旁路
3. 可验证
   - 可以通过 run-level 指标直接判断是不是守住了 high-value corridor

下一步就是：

- 起新的真实训练 run
- 看它能否把 `v167` 的
  - `467.8 -> 43.8`
  这种 late-stage collapse 进一步压住
