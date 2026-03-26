# CartPole Pure-Imag Policy Open-Loop Consistency Implementation

Date: 2026-03-15
Status: complete

## Goal

把现有 `replay suffix open-loop audit` 从“纯观测指标”升级为真正参与 world-model 更新的 `policy-space open-loop consistency loss`，直接约束 actor / critic 真正使用的 imagined `f_policy` 空间，而不是继续只看 teacher-forced reconstruction 或只靠 controller 下游补救。

## Design Decision

本次实现遵循三个明确边界：

1. 不恢复 legacy handwritten critic loss。
2. 不先动 controller rescue 路径。
3. 不重新打开 imagined critic -> world model 的 brute-force 梯度。

因此实现位点只放在 `TrainingStep` 的 world-model loss 聚合链路中。

## Implemented Mechanism

### 1. 新增 `policy-space open-loop consistency loss`

在 `TrainingStep` 中新增 `_compute_policy_open_loop_consistency_loss()`：

- 复用现有 replay batch；
- 将序列切成：
  - `context prefix`
  - `future suffix`
- 用同一条 replay suffix action 序列同时推进两条状态分支：
  - `teacher branch`: `forward_context(obs_next, action_t, state)`
  - `imagined branch`: `forward_imagination(action_t, state)`
- 从两条分支抽出 actor-use `policy features`
- 对 imagined feature 施加：
  - `Huber(imag_feat, teacher_feat.detach())`

这意味着：

- teacher 分支是 replay-grounded anchor；
- imagined 分支是真正被训练去贴近 anchor 的对象；
- 损失直接命中 actor / critic 消费的 feature manifold，而不是绕道 value head 或 controller。

### 2. 引入 value-aware high-value weighting

虽然核心损失是 feature 对齐，但选择性加权不是盲做的，而是使用：

- `replay suffix short-return target`
- 和可用时的 `teacher target-critic value`

构造：

- `score = max(replay_target, teacher_value)`

再按 quantile 选出高价值样本，对 imagined branch loss 做加权：

- `weight = 1 + boost * high_value_mask`

这样做的目的不是把 value loss 再做一遍，而是让 `policy-space` 对齐优先发生在更关键的 corridor 上。

### 3. 训练主链接线

新的 world-model 总损失现在变成：

- `base_loss_wm`
- `+ semantic_consistency_penalty`
- `+ policy_open_loop_consistency_penalty`

其中：

- `semantic consistency` 仍保留为独立路径；
- 新的 `policy open-loop consistency` 与之并列，而不是替代 bridge / core loss。

### 4. 配置与 override 支持

新增配置字段：

- `adaptive_imag_policy_open_loop_consistency_weight`
- `adaptive_imag_policy_open_loop_consistency_horizon`
- `adaptive_imag_policy_open_loop_consistency_delta`
- `adaptive_imag_policy_open_loop_consistency_high_value_boost`
- `adaptive_imag_policy_open_loop_consistency_high_value_quantile`

并完成：

- `TrainingConfig` 字段接入
- `TrainingStep._rl_cfg()` 暴露
- `aletheia_api.py` overrides 透传
- `aletheia_train.py` config normalization known-field 注册

## Metrics Added

新增 world-model metrics：

- `wm/policy_open_loop_consistency_active`
- `wm/policy_open_loop_consistency_loss`
- `wm/policy_open_loop_consistency_penalty`
- `wm/policy_open_loop_consistency_feature_l1_mean`
- `wm/policy_open_loop_consistency_feature_l2_mean`
- `wm/policy_open_loop_consistency_cosine_gap_mean`
- `wm/policy_open_loop_consistency_teacher_value_mean`
- `wm/policy_open_loop_consistency_target_mean`
- `wm/policy_open_loop_consistency_high_value_*`

同时保留逐 step 指标，便于后续分析 horizon 内部哪个位置开始脱钩。

## Tests and Verification

本轮先红后绿补上了以下新增测试：

- imagined / teacher 一致时，policy open-loop consistency loss 归零
- imagined / teacher 不一致时，loss 能推动 world-model 梯度
- high-value boost 确实改变 loss weighting
- overrides 能把新配置字段送到 `TrainingConfig`

完整回归：

- `./.venv/bin/python -m unittest aletheia.tests.test_training_entrypoints -v`
- `./.venv/bin/python -m unittest aletheia.tests.test_core_components -v`
- `./.venv/bin/python -m unittest aletheia.tests.test_run_train_contracts -v`
- `./.venv/bin/python -m unittest aletheia.tests.test_pure_imagination_path -v`
- `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration -v`
- `./.venv/bin/python -m py_compile aletheia/aletheia_train.py aletheia/aletheia_config.py aletheia/aletheia_api.py aletheia/tests/test_training_entrypoints.py aletheia/tests/test_run_train_contracts.py`

结果：

- `286` 条测试全部通过
- 语法编译检查通过

## Implementation Judgment

这一阶段实现可以判定为：

- 优雅：
  - 直接使用现有 open-loop audit 数据流，不新造并行训练管线
- 准确：
  - 约束对象就是 actor-use imagined `policy features`
- 正确：
  - 不依赖 controller
  - 不恢复旧 critic 路径
  - 不打破当前 bridge / wall 结构

但它是否足以解决最终 solved 后回撤，需要看真实 run 验证；该验证见：

- `../run-notes/cartpole_pure_imag_v165_policy_open_loop_consistency_validation_run_2026-03-15.md`
