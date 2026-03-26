# CartPole Pure-Imag Single-Ruler Mainline Default Entry

Date: 2026-03-15
Status: complete

## Goal

把已经验证过的 `single-ruler` 路径，从“需要手写 override 才会打开的实验能力”提升为 pure-imag CartPole 主入口的正式默认能力。

这一步的目标不是再发明新机制，而是把当前最高置信度的上游修复链真正接进 supported `run_train()` 主线：

- early actor-use manifold:
  - `policy-space open-loop consistency`
- late actor-use value ruler:
  - `target-ruler soft gate + critic distill`

## Implementation

修改文件：

- `aletheia/aletheia_api.py`
- `aletheia/tests/test_run_train_contracts.py`

主入口默认注入新增为：

- `adaptive_imag_policy_open_loop_consistency_weight = 0.15`
- `adaptive_imag_policy_open_loop_consistency_horizon = 3`
- `adaptive_imag_policy_open_loop_consistency_delta = 0.5`
- `adaptive_imag_policy_open_loop_consistency_high_value_boost = 1.0`
- `adaptive_imag_policy_open_loop_consistency_high_value_quantile = 0.75`
- `adaptive_imag_actor_use_target_value_ruler_enabled = true`
- `adaptive_imag_actor_use_target_value_ruler_blend = 1.0`
- `adaptive_imag_actor_use_target_value_ruler_soft_gate_enabled = true`
- `adaptive_imag_actor_use_target_value_ruler_gap_margin = 1.0`
- `adaptive_imag_actor_use_target_value_ruler_critic_distill_weight = 0.15`

设计约束保持不变：

- 只对 `CartPole + imagination_only` 自动注入
- 继续使用 `setdefault()`，所以显式 override 仍然优先
- 不引入 controller 依赖
- 不恢复 legacy critic 路径

## Why This Is The Minimal Elegant Step

这一步没有新增新的 stage、gate、controller scaffold，也没有把更多 downstream rescue 混进训练。

它只是把已经存在且已被局部验证过的两条上游对齐机制，正式并入主训练入口：

1. `policy-space open-loop consistency`
   - 负责尽早压住 actor-use imagined manifold 脱锚
2. `single-ruler soft gate`
   - 负责在 actor 真正使用的 imagined states 上，把 `online critic` 往同一 reference ruler 拉回去

因此这一步的本质是“主线收口”，不是“实验拼装”。

## Verification

定向契约测试：

- `./.venv/bin/python -m unittest aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_run_train_pure_imag_cartpole_defaults_enable_consistency_modules -v`
- `./.venv/bin/python -m unittest aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_explicit_cartpole_consistency_overrides_win_over_defaults -v`

全量相关回归：

- `./.venv/bin/python -m unittest aletheia.tests.test_training_entrypoints aletheia.tests.test_core_components aletheia.tests.test_run_train_contracts aletheia.tests.test_pure_imagination_path aletheia.tests.test_training_loop_integration -v`
- `./.venv/bin/python -m py_compile aletheia/aletheia_api.py aletheia/tests/test_run_train_contracts.py`

结果：

- `286` 条测试通过
- `py_compile` 通过

## Outcome

当前 pure-imag CartPole 主入口已经不再只是：

- 继续 cap
- 继续 detach
- 继续 MSC/shortcut

而是正式升级为：

- early manifold alignment
- late single-ruler unification

都在 supported `run_train()` 路径中默认开启。
