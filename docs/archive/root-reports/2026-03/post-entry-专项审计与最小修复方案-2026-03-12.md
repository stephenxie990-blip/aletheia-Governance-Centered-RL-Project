# post_entry / post_entry_soft 专项审计与最小修复方案

## 1. 审计目标

本轮目标不是再改 persistence tail repair，而是围绕 `post_entry / post_entry_soft` 阶段单独拆出 imagined critic 偏悲观与负 advantage 的成因，并给出不污染 persistence 主线的最小修复集。

## 2. 关键观测

基于 `outputs/exp_seed42_v87tailfix_pureimag_1500_20260311/train_metrics.jsonl`：

- `step=900`
  - `stage=post_entry`
  - `actor/base_source=online`
  - `target_mean=9.11`
  - `base_mean=9.63`
  - `online_adv_mean=-0.52`
  - `analytic_scale_online_adv=1.0`
  - `negative_adv_guard_active=0.0`
- `step=1000`
  - `stage=post_entry`
  - `target_mean=11.89`
  - `base_mean=14.97`
  - `online_adv_mean=-3.09`
  - `analytic_scale_online_adv=1.0`
  - `negative_adv_guard_active=0.0`
- `step=1050/1100/1150/1200`
  - `stage=post_entry_soft`
  - `online_adv_mean` 仍持续为负
  - `analytic_scale_online_adv=1.0`
  - `negative_adv_guard_active=1.0`
  - 但 guard 只是 0.75 actor scale + critic boost，并没有直接约束 `base_actor`
- `step=1250`
  - 真实 eval 已经 `500.0`
  - 但 imagined `base_mean=16.18 > target_mean=9.48`
  - `online_adv_mean=-6.70`

结论：

- `post_entry` 段的主矛盾不是 persistence tail mismatch
- imagined rollout 的 `continue` 并没有先崩，真正先崩的是 `online critic base > actor target`
- actor objective 在这段时间里是被负 online advantage 持续压缩的
- 当前实现里，专门针对 `negative online advantage` 的两个钩子虽然存在，但默认完全关闭：
  - `adaptive_imag_post_entry_negative_online_adv_threshold`
  - `adaptive_imag_post_entry_negative_online_adv_analytic_scale`
- 同时，限制 `base_actor` 悲观偏移的钩子也存在但关闭：
  - `adaptive_imag_post_entry_actor_base_return_cap_margin`

## 3. 根因判断

这不是 replay / evaluator / persistence tail 的问题，而是 `post_entry` 段 imagined actor-critic geometry 的局部缺口：

1. `base_actor` 直接取 `online critic`
2. `target_actor` 仍来自 imagined return
3. 当 `online critic` 在 post_entry 段局部抬得过高时，`online_advantage = target - base` 变成显著负值
4. actor loss 仍然全量吃 analytic gradient
5. `post_entry_soft` 的 negative-adv guard 只是在结果层缩 actor / 放大 critic，没有直接约束 `base_actor` 或 analytic term

因此会出现：

- 真实策略已经变强，但 imagined critic 还在滞后或局部漂高
- actor 在 post_entry 段被错误的负 advantage 压扁
- 曲线表现为 `750 -> 1000` 的中途掉点，随后又恢复

## 4. 发现的实现问题

项目里已有两个名字上属于 `post_entry` 的保护钩子：

- `post_entry negative online advantage analytic scale`
- `post_entry actor base return cap`

但原实现实际上把它们也作用到了：

- `persistence`
- `persistence_release`

这会造成两个问题：

1. 命名和语义不一致
2. 一旦启用这些钩子，就会污染已经验证有效的 persistence tail repair 主线

因此第一步修复不是“直接开参数”，而是先把作用域收干净。

## 5. 已实施的最小代码修复

### 5.1 收紧 post_entry negative online advantage analytic scale 的作用域

位置：`aletheia/aletheia_train.py:3378`

现在只在这些阶段可用：

- `post_entry`
- `post_entry_soft`
- `post_entry_pending`
- `handoff`

不再影响：

- `internal_post_entry`
- `persistence`
- `persistence_release`

### 5.2 收紧 post_entry actor base return cap 的作用域

位置：`aletheia/aletheia_train.py:7942`

现在只在这些阶段可用：

- `post_entry`
- `post_entry_soft`
- `post_entry_pending`
- `handoff`

不再影响：

- `internal_post_entry`
- `persistence`
- `persistence_release`

## 6. 新增回归测试

位置：`aletheia/tests/test_training_loop_integration.py`

新增 5 条：

- `test_post_entry_negative_online_adv_analytic_scale_applies_only_in_post_entry_soft`
- `test_post_entry_negative_online_adv_analytic_scale_does_not_leak_into_persistence`
- `test_post_entry_base_return_cap_applies_only_in_post_entry_soft`
- `test_post_entry_base_return_cap_does_not_leak_into_persistence`
- `test_post_entry_only_guards_do_not_apply_in_internal_post_entry`

这些测试锁定了两个不变量：

1. post_entry 钩子在 `post_entry_soft` 必须生效
2. post_entry 钩子绝不能污染 `internal_post_entry` 和 `persistence`

## 7. 已验证结果

已通过：

- `./.venv/bin/python -m unittest` 针对上述 5 条新增/受影响测试
- `./.venv/bin/python -m py_compile scripts/cartpole_checkpoint_audit.py aletheia/aletheia_train.py aletheia/tests/test_training_loop_integration.py`

## 8. 错误分支记录

我曾经做过一版过宽的作用域修复，把 `internal_post_entry` 也纳入了 post_entry 保护范围。结果在新试训里：

- `eval@250 = 17.1`，略高于基线 `15.9`
- 但 `eval@500 = 34.6`，明显低于基线 `107.6`

这说明：

- `internal_post_entry` 不能直接吃同一套 post_entry 保护
- 早期内部 handoff 阶段对学习速度更敏感
- 因此当前修复必须保持 `internal_post_entry` 隔离

## 9. 当前最小参数方案

已固化到：`tmp/v87_postentry_minfix_overrides.json`

当前只尝试 3 个 post_entry 参数：

- `adaptive_imag_post_entry_negative_online_adv_threshold = 0.75`
- `adaptive_imag_post_entry_negative_online_adv_analytic_scale = 0.4`
- `adaptive_imag_post_entry_actor_base_return_cap_margin = 2.0`

设计逻辑：

- `0.75`：不去干扰 `step=900` 这种轻微负 advantage，只拦截 `1000` 后的明显负值
- `0.4`：不是完全关闭 analytic term，而是先降到 40%
- `2.0`：只限制 `base_actor` 比 `target_actor` 高出过多的情况，避免把正常小偏差也截断

## 10. 下一步执行顺序

1. 先完成 `final.pt` 的 `500ep × 20 seeds` 标准与轻扰动审计
2. 若标准与轻扰动都稳定，则用当前最小参数方案重跑一轮 `v87 tailfix postentry-minfix`
3. 对比基线最关键的三个点：
   - `eval@500`
   - `eval@750`
   - `eval@1000`
4. 若 `@1000` 掉点明显收窄且 `@1250` 不回归，再把这组参数固化为新的 post_entry 基线

## 11. 新增链路发现：highwater commit 会过早推进到 hard post_entry

补充审计 `outputs/exp_seed42_v87tailfix_pureimag_1500_20260311/train_metrics.jsonl` 后，确认 `post_entry` 掉点不仅是“进入了 post_entry 之后没有保护”，还包含一个更前置的控制器链路问题：

- `step=800/900/1000`
  - `imag/controller_stage = post_entry`
  - `imag/controller_post_entry_commit_gap_ok = 0`
  - `imag/controller_post_entry_commit_highwater_ok = 1`
  - `imag/controller_post_entry_commit_gap_abs = 4.69 / 4.61 / 5.86`
- 当前配置下：
  - `adaptive_imag_post_entry_commit_gap_max = 2.2`
  - `adaptive_imag_post_entry_commit_highwater_gap_max = 6.0`

这意味着：

1. 标准 commit gap 条件其实并没有满足；
2. 但 highwater commit 仍然因为“历史最好 eval 足够高 + 高水位 gap 上限更宽”而成立；
3. 控制器因此把系统推进到了 `hard post_entry`；
4. 而 `hard post_entry` 恰好没有 `post_entry_soft/post_entry_pending` 那套负 advantage 保护。

所以完整链路应修正为：

- 外部 eval 在 `step=750` 达到高值后，控制器进入 `external post_entry`；
- highwater commit 在 `gap_abs` 仍明显偏大的情况下，把阶段从更软的保护区间推进到 `hard post_entry`；
- `_build_imagined_batch()` 仍用 `online critic` 作为 `base_actor`，`imagined returns` 作为 `target_actor`；
- `base > target` 导致 `online_adv_mean < 0`；
- `train_step()` 中 `hard post_entry` 既没有 soft/pending negative-adv guard，也没有默认启用的 analytic scaling / base cap；
- 于是 actor 被错误负 advantage 全量压缩，直到后续阶段重新回落或恢复。

这条新增结论不推翻前面的“imagined critic / actor geometry 缺口”判断，而是把它进一步定位为：

- 上游：`post_entry controller highwater commit` 过早；
- 中游：`base_actor = online critic` 与 `target_actor = imagined return` 的几何关系失真；
- 下游：`hard post_entry` 缺少针对负 online advantage 的专用保护。

因此，当前最小修复集仍保持两层思路：

- 第一层：先用“作用域收紧后的 post_entry analytic scaling + base return cap”去修复 `hard post_entry` 的 actor geometry；
- 第二层：如果这层仍不足，再单独收紧 `adaptive_imag_post_entry_commit_highwater_gap_max`，避免控制器过早把系统推进到 `hard post_entry`。

## 12. 阶段级聚合证据

对 `train_metrics.jsonl` 按 `imag/controller_stage` 做聚合后，可以看到问题并不是单点抖动，而是阶段性稳定出现：

- `idle` 阶段（`step=50..750`）
  - `online_adv_mean_avg = +1.75`
  - `base_minus_target_avg = -1.75`
  - `negative advantage ratio = 13.3%`
- `post_entry` 阶段（`step=800..1000`）
  - `online_adv_mean_avg = -1.16`
  - `base_minus_target_avg = +1.16`
  - `negative advantage ratio = 100%`
  - `analytic_scaled_ratio = 0%`
  - `base_cap_ratio = 0%`
  - `negative_adv_guard_ratio = 0%`
- `post_entry_soft` 阶段（`step=1050..1200`）
  - `online_adv_mean_avg = -2.10`
  - `base_minus_target_avg = +2.10`
  - `negative advantage ratio = 100%`
  - `analytic_scaled_ratio = 0%`
  - `base_cap_ratio = 0%`
  - `negative_adv_guard_ratio = 100%`
- `trigger` 阶段（`step=1250`）
  - `online_adv_mean = -6.70`
  - `base_minus_target = +6.70`
  - `analytic_scaled_ratio = 0%`
  - `base_cap_ratio = 0%`
  - `negative_adv_guard_ratio = 0%`

额外观察：

- `post_entry` 阶段的 `imag/continue_prob_mean_raw` 平均约 `0.963`
- 同期 `imag/value_target_gap_abs_mean` 只有约 `2.05`
- 也就是说，world model / imagined rollout 的继续概率并没有先崩
- 真正首先系统性翻负的是 `online critic base - imagined target` 这组 actor geometry

这进一步支持前述判断：

- 这里的主矛盾不是 imagined rollout 先失效
- 而是进入 `post_entry` 后，actor 先持续看到“错误负 advantage”
- `post_entry_soft` 虽然有 guard，但它作用在后段；对更早的 `hard post_entry` 仍缺乏直接约束

## 13. 恢复链路审计：trainer_state resume 之前存在 lazy component 漏加载

在尝试做 `step750 -> step1250` 的 phase replay 时，发现旧的 `trainer_state_step750.pt` 恢复到当前代码会触发：

- `TrainingStateManager.load falling back to non-strict model restore`
- 未消费的 key 主要来自：
  - `world_model._control_head.*`
  - `world_model._projection.*`

根因是：

- `AgentHandle.load()` 在普通 checkpoint 加载前会调用 `world_model._ensure_v45_components()`；
- 但 `TrainingStateManager.load()` 之前没有做同样的 materialize；
- 对于带懒初始化 v4.5 组件的旧 trainer-state，这会把 checkpoint 里的参数判成 `unexpected keys`；
- 随后进入 non-strict load，等价于把这部分权重静默丢掉。

本轮已修复：

- `TrainingStateManager.load()` 在恢复 `_ModelWrapper` 前，先对 `model.world_model` 调用 `_ensure_v45_components()`（若存在）；
- 已新增回归测试：
  - `test_training_state_load_materializes_lazy_world_model_components`

注意：

- 这个修复解决了“resume 路径会静默漏加载 v4.5 lazy modules”的实现 bug；
- 但它并没有让 `phase replay` 变成原始训练轨迹的高保真重放。

## 14. 恢复 replay 的可信度边界

修复上述 lazy component 加载 bug 后，继续做了两个对照：

1. 原始 `checkpoint_step750.pt` 的快速评估
   - 当前代码下 `10ep` 仍可达 `425.9`
2. 修复后的 `trainer_state_step750.pt` resume replay（原始 tailfix 配置）
   - `eval@800 = 99.8`
   - `eval@850 = 81.4`

这说明：

- 修复后的 resume 路径不再静默漏加载模型参数；
- 但 `step750` 之后的 replay 仍然不能忠实复现原始 run 的 `355.8 -> 355.8` 轨迹；
- 因此它只能用作“相对筛选工具”，不能再用作“绝对对账工具”。

当前最合理的解释是：

- trainer-state 只保存了模型 / 优化器 / replay buffer / adaptive controller state；
- 没有保存 collector 侧环境内部状态、in-flight rollout state、环境 RNG 链路等；
- 所以恢复后虽然模型和 buffer 接近，但后续新采样的数据分布并不等于原始训练历史的延续。

因此后续方法论调整为：

- `resume replay` 只做候选修复的相对筛选；
- 最终是否采纳，必须以 fresh run 的 `1500-step` 真实训练结果为准。

## 15. 相对 replay 筛选结论

基于修复后的 resume replay，对比了三类候选：

1. `control`（原始 tailfix 配置）
   - `eval@800 = 99.8`
   - `eval@850 = 81.4`
2. `geometry min-fix`（analytic scaling + base return cap）
   - `eval@800 = 51.4`
   - `eval@850 = 74.9`
   - `eval@900 = 64.5`
   - `eval@950 = 59.1`
   - `eval@1000 = 51.1`
3. `commit-gate` 候选
   - `highwater_gap_max = 3.5`
     - `eval@800 = 68.3`
     - `eval@850 = 85.7`
     - `eval@900 = 113.2`
   - `highwater_gap_max = 3.0`
     - `eval@800 = 68.1`
     - `eval@850 = 331.9`
     - `eval@900 = 50.7`

结论：

- 仅修 imagined actor geometry 的 `min-fix` 应判定为失败分支；
- 它虽然改善了 batch 内 `online_adv_mean / base-target` 几何关系，但 replay 下的真实 eval 更差；
- 相比之下，直接收紧 `highwater commit` 更符合主矛盾定位；
- 尤其 `3.5` 和 `3.0` 都成功改变了 `controller_stage` 的切换模式，证明上游 gate 的确是强影响因子；
- 但由于 replay 本身不高保真，最终仍需 fresh run 做裁决。

## 16. fresh run 基线复核：当前代码的主回退点已经前移到 idle / 500

为了排除 replay 偏差，我使用当前代码、旧 `v87 tailfix` 的同一套 resolved config（唯一差异仅为关闭 early-stop 以跑完整曲线）重跑了 fresh 1500-step control：

输出目录：
- `outputs/exp_seed42_v87tailfix_control_1500_full_20260312`

结果：
- `eval@250 = 17.1`
- `eval@500 = 34.6`
- `eval@750 = 87.7`
- `eval@1000 = 135.4`
- `eval@1250 = 199.1`
- `eval@1500 = 155.4`

对比旧基线 `outputs/exp_seed42_v87tailfix_pureimag_1500_20260311`：
- 旧：`15.9 / 107.6 / 355.8 / 125.3 / 500.0`
- 新：`17.1 / 34.6 / 87.7 / 135.4 / 199.1 / 155.4`

更关键的是，`step <= 500` 的训练指标显示：

- `imag/controller_stage` 全程都是 `idle`
- `post_entry` / `internal_post_entry` / `persistence` 都还没有介入
- 所有 post-entry 专用保护都是关闭的：
  - `actor/negative_adv_guard_active = 0`
  - `actor/base_return_cap_active = 0`
  - `actor/analytic_weight_scale_online_adv_active = 0`

这意味着：

- 当前代码与旧 solved run 的主要差异，已经不是 `post_entry` 阶段造成的；
- 真正的主回退点已经前移到 `idle -> 500`；
- 因此，post-entry 修复现在只能视为二级问题；
- 不先修复 `idle/500` 主线，继续围绕 post-entry 打补丁不会让项目恢复旧的 pure-imag 优势。

## 17. 对 post-entry 分支的重新判定

基于上述 fresh control 结果：

- `commitgate35` fresh run 与 current-control fresh run 在 `250 / 500 / 750 / 1000 / 1250 / 1500` 上完全同轨；
- 这说明在当前代码基线上，`adaptive_imag_post_entry_commit_highwater_gap_max = 3.5` 实际上没有改变主训练轨迹；
- `commitgate30` fresh run 则在 `500` 即显著更差，属于明确失败分支；
- `geometry min-fix` 在 replay 相对筛选中也明显更差，应继续视为失败分支。

因此当前路线应调整为：

1. 暂停继续围绕 post-entry 做参数和局部逻辑修补；
2. 把审计焦点前移到 `idle/500` 主线；
3. 优先排查以下链路的实现差异与数据差异：
   - rollout collector / actor `act()` 真实采样链
   - replay buffer 写入与 sample_sequences 输出
   - world model 早期训练目标与 reward / continue / value 预测
   - imagined batch 早期 `returns / values / weights_actor` 构建
   - actor / critic 在 idle 阶段的损失几何
