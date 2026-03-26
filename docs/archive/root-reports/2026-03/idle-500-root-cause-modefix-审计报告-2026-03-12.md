# idle/500 主线回退根因审计与 mode-fix 修复报告（2026-03-12）

## 1. 结论摘要

当前代码之所以从旧的 `v87 tailfix final` solved 主线回退，不是 `post_entry` 保护链导致，也不是 `persistence tail mismatch` 修复本身导致。

一级根因在真实采样链：

- `AgentHandle.act()` 在 **随机训练采样**（`deterministic=False`）时，仍然强制把 `world_model / router / actor / critic` 切到 `eval()`。
- 这会静音 `router` 与 `world_model` 中依赖 `self.training` 的 live state update 行为。
- 结果不是单纯“数值略变”，而是 **真实采样分布被改写**，导致 replay 数据链在 `idle -> 500` 阶段整体偏离旧 solved 主线。

最小修复：

- 只在 **deterministic evaluation rollout** 路径强制 `eval()`。
- 对随机训练采样路径，保持模块原始 mode，不再静音 live routing / uncertainty / EMA 行为。

修复后 fresh pure-imag run 已重新恢复旧 solved 轨迹：

- `eval@250 = 15.90`
- `eval@500 = 107.60`
- `eval@750 = 355.80`
- `eval@1000 = 125.30`
- `eval@1250 = 500.00`

与旧有效 run 完全对账。

## 2. 症状复盘

修复前的 fresh current-control run：

- `17.1 / 34.6 / 87.7 / 135.4 / 199.1 / 155.4`
- 对应输出目录：`outputs/exp_seed42_v87tailfix_control_1500_full_20260312`

旧有效 solved run：

- `15.9 / 107.6 / 355.8 / 125.3 / 500.0`
- 对应输出目录：`outputs/exp_seed42_v87tailfix_pureimag_1500_20260311`

差异不是 post-entry 之后才出现，而是在 `idle/500` 阶段就已经展开。

## 3. 为什么排除 post_entry / persistence

已确认在 `step <= 500` 的 current-control 中：

- `imag/controller_stage = idle`
- `post_entry / internal_post_entry / persistence` 均未介入
- 所有 post-entry 专用 guard 均未触发

因此：

- `post_entry` 只能算二级问题
- `persistence tail mismatch` 不是导致 early 回退的原因

## 4. early idle 阶段的关键异常

old vs current step500 指标对账表明：

- current `imag/value_mean` 更高
- current `actor/online_adv_mean` 更低
- current `imag/continue_prob_mean_raw` 更高
- current `critic/slow_value_gap_abs_mean` 显著更大

同时，`train/episode_count` 与 `Buffer` 也对不上旧 solved run：

- old `Buffer = 193`
- current `Buffer = 216`

这说明问题不只是“同样数据被学坏”，而是 **真实采样进来的数据链本身已经变了**。

## 5. 被证伪的修复支线

为了避免误判，先做了两条最小筛选：

### 5.1 detach imagined critic features

override：

- `rl.detach_critic_features_on_imagination = true`

结果：

- `eval@250 = 9.6`
- `eval@500 = 44.6`
- `eval@750 = 108.0`

判定：

- 这是一个真实的结构风险，但不是当前一级根因。
- 切断 critic->actor imagined feature 梯度不能恢复 old solved 主线。

### 5.2 detach + drift guard

override：

- `rl.detach_critic_features_on_imagination = true`
- `rl.use_actor_drift_guard = true`
- `rl.slow_value_reg_drift_gain = 2.0`

结果：

- `eval@250 = 17.9`
- `eval@500 = 79.8`
- `eval@750 = 46.7`

判定：

- 过度纠偏，仍不能恢复 solved 主线。

## 6. 根因定位

代码审计发现 `AgentHandle.act()` 存在如下行为：

- 进入 `act()` 后，无论是不是 deterministic evaluation，都先把
  - `world_model`
  - `router`
  - `actor`
  - `critic`
  - compiled variants
  强制切到 `eval()`
- 结束后再恢复原 mode

这个设计对纯前馈无状态网络问题不大，但当前工程不是这种情况。

### 6.1 router 受影响的原因

`router` 存在显式 `self.training` 分支：

- uncertainty statistics update
- history update
- dynamic threshold update
- EMA smoothing update
- internal step progression

一旦训练采样时被强行切到 `eval()`：

- uncertainty history 不再按真实轨迹推进
- EMA smoothing 行为切到 eval branch
- live routing policy 不再沿真实采样持续更新

### 6.2 world_model 受影响的原因

`world_model` 内也存在训练态统计更新与 step 计数逻辑：

- 部分统计/历史只在 `self.training` 下推进

因此训练采样链被“伪评估化”后：

- real rollout 看到的并不是训练态 live model
- replay buffer 收到的是被静音后的 policy/data distribution

## 7. 修复内容

修改文件：

- `aletheia/aletheia_api.py`

修复逻辑：

- 只在 `deterministic=True` 的评估 rollout 上强制 `eval()`
- 在 `deterministic=False` 的训练采样路径中，不再改写模块 mode

修复后语义变为：

- evaluation：冻结训练态行为，保证评估可重复
- collection：保持 live policy / router / world_model 的训练态动态

## 8. 回归测试

新增并通过的核心测试：

- `test_agent_act_keeps_train_mode_for_stochastic_collection`
- `test_agent_act_forces_eval_mode_for_deterministic_rollouts`

此外保留并再次通过：

- `test_training_state_load_materializes_lazy_world_model_components`
- 全量 `test_run_train_contracts`

本次实际运行：

- `python -m py_compile aletheia/aletheia_api.py aletheia/tests/test_run_train_contracts.py`
- `python -m unittest aletheia.tests.test_run_train_contracts aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_training_state_load_materializes_lazy_world_model_components`

结果：

- `37 tests OK`

## 9. 修复后的 fresh 验证

### 9.1 750-step smoke run

输出目录：

- `outputs/exp_seed42_v87tailfix_modefix_750_20260312`

结果：

- `eval@250 = 15.90`
- `eval@500 = 107.60`
- `eval@750 = 355.80`

并且中间训练日志与旧 solved run 逐点一致：

- `STEP 1216: Vmu=4.496, Rmu=4.347, Amu=-0.149, Buffer=73`
- `STEP 1600: Vmu=2.994, Rmu=5.300, Amu=2.306, Buffer=91`
- `STEP 4000: Vmu=2.726, Rmu=5.387, Amu=2.660, Buffer=193`

这些点与旧 run 完全对账。

### 9.2 1500-step fresh validation

输出目录：

- `outputs/exp_seed42_v87tailfix_modefix_1500_20260312`

结果：

- `eval@250 = 15.90`
- `eval@500 = 107.60`
- `eval@750 = 355.80`
- `eval@1000 = 125.30`
- `eval@1250 = 500.00`

训练在 `step 1250` 触发 early stop，恢复 solved。

## 10. 这次为什么能确认是根因

因为它同时解释了之前所有关键矛盾：

1. 为什么 post-entry 修补始终救不回来
   - 因为 replay 数据在进入 post-entry 之前就已经被采坏。

2. 为什么 old/current 的 `Buffer`、`episode_count`、`continue/value/adv` 会整体错位
   - 因为真实采样时 live routing/world-model 状态没有按训练态推进。

3. 为什么修复后能“一步回到 old solved 主线”
   - 因为这不是调参，而是把真实采样链恢复到原本应有的运行语义。

## 11. 当前判断

当前工程恢复 pure-imag 优势的主矛盾已经被修复。

后续工作重点不应该再围绕 `idle/500` 主线排查“莫须有”的 critic/actor 几何问题，而应转为：

- 把这个 mode-fix 固化为正式基线
- 基于该基线做新的稳健性验证
- 再决定是否继续优化 post_entry / persistence 的中后段策略稳定性


## 12. fresh mode-fix 基线的分层鲁棒性审计

由于当前 CPU 上直接执行 `500 episodes × 20 seeds × 3 scenarios` 预计需要小时级时间，本轮先执行了同口径缩短版审计：

- checkpoint: `outputs/exp_seed42_v87tailfix_modefix_1500_20260312/final.pt`
- episodes: `100`
- num_seeds: `10`
- scenarios: `standard / init_expand / obs_noise`
- output: `outputs/exp_seed42_v87tailfix_modefix_1500_20260312/final_audit_100ep_10seeds_perturbed.json`

结果：

- `standard`: `mean_of_seed_means = 500.0`, `perfect_seed_count = 10/10`
- `init_expand`: `mean_of_seed_means = 500.0`, `perfect_seed_count = 10/10`
- `obs_noise`: `mean_of_seed_means = 500.0`, `perfect_seed_count = 10/10`

与旧 full audit 的方向完全一致：

- 旧 full audit: `500 episodes × 20 seeds` 下三场景 `20/20` 完美
- 新 fresh mode-fix audit: `100 episodes × 10 seeds` 下三场景 `10/10` 完美

这说明修复后的 fresh 基线不仅重新回到旧 solved 训练轨道，而且在标准条件、初始状态扩展、轻观测噪声三个场景下都表现出与旧 solved 基线一致的稳定性。

## 13. fresh mode-fix 基线的复合扰动审计

在第 12 节的三类单因素扰动全部通过后，进一步执行了复合扰动审计：

- checkpoint: `outputs/exp_seed42_v87tailfix_modefix_1500_20260312/final.pt`
- episodes: `100`
- num_seeds: `10`
- scenario: `combined`
- 含义: `init_expand + obs_noise` 同时启用
- output: `outputs/exp_seed42_v87tailfix_modefix_1500_20260312/final_audit_100ep_10seeds_combined.json`

结果：

- `mean_of_seed_means = 500.0`
- `std_of_seed_means = 0.0`
- `perfect_seed_count = 10/10`
- `solved_seed_count_475 = 10/10`
- 每个 seed 在 `100 episodes` 上均为 `500.0`

这说明修复后的 fresh mode-fix 基线不只是能在单因素轻扰动下保持 solved，也能在“初始状态扩展 + 观测噪声”复合存在时维持完全稳定。

当前结论升级为：

- `idle/500` 主线已经恢复
- fresh pure-imag 基线已重新达到 solved
- 在 `standard / init_expand / obs_noise / combined` 四种评估场景下，当前缩短版 audit 全部为完美通过

下一步应继续加严验证，而不是重新回到已证伪的 `post_entry` 猜测分支上反复打补丁。

## 14. 加严版复合扰动审计

在 `100 episodes × 10 seeds × combined` 全绿之后，继续执行了一档更强的复合扰动验证：

- checkpoint: `outputs/exp_seed42_v87tailfix_modefix_1500_20260312/final.pt`
- episodes: `100`
- num_seeds: `20`
- scenario: `combined`
- output: `outputs/exp_seed42_v87tailfix_modefix_1500_20260312/final_audit_100ep_20seeds_combined.json`

结果：

- `mean_of_seed_means = 500.0`
- `std_of_seed_means = 0.0`
- `min_seed_mean = 500.0`
- `max_seed_mean = 500.0`
- `perfect_seed_count = 20/20`
- `solved_seed_count_475 = 20/20`

这比第 13 节进一步说明：当前修复后的 fresh pure-imag 基线，不仅在小样本 seed 覆盖下稳定，而且在扩展到 `20` 个随机种子后，面对复合扰动依旧没有出现任何性能塌陷、seed 漂移或 solved 不稳定现象。

到这一阶段，可以把当前结论从“主线恢复”提升为“高置信恢复”：

- fresh pure-imag 训练重新在 `step 1250` 达到 `500.0 solved`
- `standard / init_expand / obs_noise / combined` 四场景已完成缩短版审计
- 其中最强的 `combined` 场景在 `100 episodes × 20 seeds` 下为 `20/20 perfect`

因此，当前工程的主问题已经不再是“pure imag 是否还能站起来”，而是“在已恢复 solved 基线之上，是否还要继续打磨 post_entry / persistence / post_solved 的后段稳定器和泛化余量”。
