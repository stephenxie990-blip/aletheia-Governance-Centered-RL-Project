# CSS-v1.5 认证信任自举实现与验证

- 时间：2026-03-16 18:59
- 阶段：critic target 单独分支最小实现
- 原则：
  - actor 主合同不动
  - critic 使用单独 target 分支
  - 不新增 loss family
  - 不提高现有 anchor 权重

## 这刀实际改了什么

这次没有做“外部 critic 清洗器”，而是把 critic 的学习目标改成一条独立的、受认证信任度调制的 target 分支：

1. 在 `_build_imagined_batch()` 中新增 critic 专用量：
   - `critic_contract_behavior_certainty`
   - `critic_contract_task_degradation`
   - `critic_contract_authority`
   - `critic_contract_trust`
   - `target_critic`

2. `target_critic` 的构造逻辑：
   - 以原始 imagined `returns` 为基线
   - 以 `corridor_semantic_clean_target` 为认证锚
   - 以 `semantic pressure + behavior certainty + task degradation + anchor confidence` 的保守交集决定接管强度
   - actor 继续使用原有 `target_actor`

3. 在 `train_step()` 中：
   - actor 仍使用 `target_actor`
   - critic 优先使用 `target_critic`
   - 若 batch 中不存在 `target_critic`，则自动回退到旧路径

## 关键实现判断

这版实现不是“提高 critic anchor loss 权重”，而是把“该信自己还是该信认证锚”直接写进 critic target 构造。

为了避免再走回多层乘法门控，这次 authority 采用了保守的 `min(...)` 聚合，而不是继续把一串 gate 相乘。

## 代码落点

- `aletheia/aletheia_train.py`
  - `_build_imagined_batch()`
  - `train_step()`
- `aletheia/aletheia_config.py`
  - 新增 `adaptive_imag_critic_trust_target_enabled`
- `aletheia/aletheia_api.py`
  - pure-imag CartPole 默认 override 开启 `adaptive_imag_critic_trust_target_enabled`
- `aletheia/tests/test_training_loop_integration.py`
  - 新增 2 条集成测试

## 新增验证点

1. `test_build_imagined_batch_emits_certified_critic_target_branch_without_changing_actor_target`
   - 钉住 actor target 不变
   - 钉住 critic target 分支会在高 pressure / 高 certainty / 有 degradation 时真实生成

2. `test_train_step_prefers_separate_critic_target_branch_when_available`
   - 钉住训练时 critic 确实优先吃 `target_critic`
   - 钉住 actor 仍吃 `target_actor`

## 本地验证结果

已通过：

```bash
./.venv/bin/python -m py_compile aletheia/aletheia_train.py aletheia/aletheia_config.py aletheia/aletheia_api.py aletheia/tests/test_training_loop_integration.py
```

```bash
./.venv/bin/python -m unittest \
  aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_build_imagined_batch_emits_certified_critic_target_branch_without_changing_actor_target \
  aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_train_step_prefers_separate_critic_target_branch_when_available
```

```bash
./.venv/bin/python -m unittest \
  aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_actor_contract_trust_drives_corridor_beta_and_reduces_weighted_actor_target_when_task_and_registry_mismatch_align \
  aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_actor_contract_trust_activates_on_task_semantic_mismatch_even_when_registry_support_remains_high \
  aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_build_imagined_batch_emits_certified_critic_target_branch_without_changing_actor_target \
  aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_train_step_prefers_separate_critic_target_branch_when_available
```

结果：
- `py_compile` 通过
- 2 条新测试通过
- 2 条相邻旧测试 + 2 条新测试，共 4 条测试通过

## 当前最准确的阶段结论

- 这次已经把“critic 自己知道该信谁”接进了真实代码主链
- 但目前还只是最小实现验证版
- 还没有跑新的真实 pure-imag 长程实验

所以下一步不再是继续加机制，而是直接开新 run，看三个问题：

1. `real_mc_value_gap_abs` 是否开始放缓
2. `teacher/imag value abs to short return` 是否明显下降
3. `5000-step` 后段是否不再出现 corridor 内语义性腐烂
