# CSS-v1.6 最小接管律重写与验证

时间：2026-03-16 19:31

## 本轮目标

- 不改 actor 主合同。
- 不加新 loss family。
- 不提高现有 anchor 权重。
- 只重写 critic trust-aware target 的 takeover law。

## 触发病灶

- `CSS-v1.5` 的问题不是方向错，而是 `behavior_certainty` 被用成前置硬上限：
  - `authority_cap = min(behavior_certainty, task_degradation)`
  - 导致 `behavior_certainty = 0` 时：
    - `critic_contract_authority = 0`
    - `critic_contract_trust = 1`
    - `target_critic_delta = 0`
- 真实 run 已证明：
  - 语义膨胀会在 `behavior_certainty` 拉起之前持续扩大；
  - clean critic target 接管太晚、太弱。

## 本轮改动

### 1. critic 接管律改成 floor + bonus

- 新结构：
  - `authority_floor = semantic_pressure * anchor_confidence * takeover_floor * real_behavior_available`
  - `authority_bonus = semantic_pressure * anchor_confidence * (1 - takeover_floor) * behavior_certainty * task_degradation`
  - `authority = clamp(authority_floor + authority_bonus, 0, 1)`
- 结构含义：
  - `semantic_pressure` 负责更早启动最小自纠偏；
  - `behavior_certainty` 不再决定 0/1，只负责在 solved-phase 认证更强时继续加码；
  - `task_degradation` 仍保留在 bonus 支路里；
  - anchor 仍只通过 `clean target` 分支介入，没有新增 critic loss。

### 2. 新增最小配置项

- `adaptive_imag_critic_trust_target_takeover_floor`
  - 默认 pure-imag override：`0.2`

### 3. 新增可观测量

- `critic_contract_authority_floor_mean`
- `critic_contract_authority_bonus_mean`

这两个指标用于区分：
- 是底座没起来；
- 还是 bonus 没加上。

## 验证

### 新增测试

1. `test_build_imagined_batch_critic_target_keeps_floor_takeover_when_behavior_certainty_is_zero`
   - 钉住：即使 `behavior_certainty = 0`，只要语义压力高，`authority_floor` 与 `target_critic_delta` 也必须非零。

2. `test_build_imagined_batch_critic_target_behavior_certainty_adds_bonus_on_top_of_floor_takeover`
   - 钉住：高 `behavior_certainty` 时，`authority_bonus` 与总 authority 必须高于低 certainty 分支。

### 本轮验证命令

```bash
./.venv/bin/python -m py_compile \
  /Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_config.py \
  /Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py \
  /Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py \
  /Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_training_loop_integration.py
```

```bash
./.venv/bin/python -m unittest \
  aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_build_imagined_batch_emits_certified_critic_target_branch_without_changing_actor_target \
  aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_build_imagined_batch_critic_target_keeps_floor_takeover_when_behavior_certainty_is_zero \
  aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_build_imagined_batch_critic_target_behavior_certainty_adds_bonus_on_top_of_floor_takeover \
  aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_train_step_prefers_separate_critic_target_branch_when_available \
  aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_run_train_console_log_disambiguates_update_and_env_steps
```

结果：全部通过。

## 当前阶段判断

- 这一步修的是：
  - `critic clean target` 为什么在 pre-collapse 窗口完全不接管。
- 这一步还没修的是：
  - 真实 run 中 `floor=0.2` 是否足以在 `gap_abs 7 -> 13 -> 24 -> 35` 这段窗口提前压住膨胀。
- 下一步正确动作：
  - 起新真实 pure-imag run；
  - 重点盯：
    - `critic/critic_contract_authority_floor_mean`
    - `critic/critic_contract_authority_bonus_mean`
    - `critic/target_critic_delta_mean`
    - `critic/real_mc_value_gap_abs_mean`
    - `imag/open_loop_audit_teacher_value_abs_to_short_return_mean`
    - `imag/open_loop_audit_imag_value_abs_to_short_return_mean`
