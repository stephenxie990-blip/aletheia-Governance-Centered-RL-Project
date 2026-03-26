# 方案一 Reward-Grounded Trust 与 Critic Bootstrap 重写 - 2026-03-16 20:21

## 本轮目标

在不回退 actor 主合同、不新造 loss family、不提高现有 anchor 权重的前提下，验证如下最小方案：

1. 把 `task_degradation` 接入共享 `trust` 信号源；
2. 把 `trust` 正式接入 critic bootstrap，而不只是在后验 `target_critic` 分支做 clean-target takeover。

## 已实现改动

### 1. reward-grounded trust

在 imagined batch 构造中新增共享退化量：

- `eval_drawdown`
- `switch_degradation`
- `oscillation_degradation`
- `shared_contract_task_degradation = max(eval_drawdown, task_kinematic_degradation)`

并将其接入：

- `actor_contract_task_degradation_pressure_mean`
- `actor_contract_semantic_pressure = max(semantic_certified_mismatch, inflation_pressure, task_degradation_pressure)`

### 2. critic bootstrap rewrite

在 critic target 主链中新增 trust-aware bootstrap rewrite：

- 先用 `actor_contract_semantic_trust` 构造 critic bootstrap trust
- 用 replay-grounded clean target 作为 anchor value
- 用 trust 在 target critic value 和 anchor value 之间混合
- 用混合后的 values 重新计算 critic raw lambda-return

保留现有 `CSS-v1.6` 的第二层：

- `target_critic = lerp(target_critic_raw, clean_target, authority)`

因此当前 critic 路径已经变成：

1. reward-grounded trust 改 bootstrap raw target
2. authority 再对 clean target 做后验接管

## 新增观测量

- `actor_contract_task_degradation_pressure_mean`
- `critic_contract_bootstrap_trust_mean`
- `critic_contract_bootstrap_effective_trust_mean`
- `critic_contract_bootstrap_active_fraction`
- `critic_contract_bootstrap_delta_mean`

## 已完成验证

- `./.venv/bin/python -m py_compile ...` 通过
- `./.venv/bin/python -m unittest aletheia.tests.test_run_train_contracts aletheia.tests.test_training_loop_integration`
- 结果：`246 tests OK`

## 本地探针结论

### 探针 1：bootstrap rewrite 真实发生

- 未开启 critic trust target：
  - `target_critic_raw_mean = 2.97`
  - `critic_contract_bootstrap_delta_mean = 0.0`
- 开启 critic trust target：
  - `target_critic_raw_mean = 0.0`
  - `critic_contract_bootstrap_delta_mean = 2.97`

结论：

- critic raw target 已经先于 `target_critic` takeover 发生变化

### 探针 2：task degradation 可单独拉低 trust

在人工构造的：

- `inflation_pressure = 0`
- `semantic_certified_mismatch = 0`
- `corridor_high_value_fraction = 1`

场景下，观测到：

- `actor_contract_task_degradation_pressure_mean = 0.7`
- `actor_contract_semantic_pressure_mean = 0.7`
- `actor_contract_semantic_trust_mean = 0.3`
- `critic_contract_bootstrap_delta_mean > 0`

结论：

- 假 corridor 里即使 inflation 不响，只要任务退化，trust 也会下降

## 当前真实 run

- 输出目录：
  - `outputs/exp_seed42_v183_h15_reward_grounded_trust_bootstrap_2500_20260316_01`

## 回退点

若真实 run 失败，回退基线位于：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/备份/方案一实施前状态-2026-03-16-2005`
