# post_entry / post_entry_soft 专项审计（初步结论）

项目：`/Users/zhangsan/Desktop/缸中之脑v5.6`
日期：`2026-03-12`
基线：`outputs/exp_seed42_v87tailfix_modefix_1500_20260312`

## 1. 审计目的

在 `idle/500` 主线根因已经定位并修复后，重新审视 `post_entry / post_entry_soft / post_entry_pending / persistence escape` 这一整套中后段状态机，明确：

- 它是不是当前 pure-imag 主线恢复的一级因
- 训练链路和数据链路是否按设计运行
- 当前实现有没有明显的行为错误或测试缺口
- 后续应当继续“修 bug”，还是转向“简化架构、增强可观测性”

## 2. 先给结论

当前结论很明确：

1. `post_entry` 不是这次 `idle/500` 回退的一级根因。
2. mode-fix 之后，fresh pure-imag 已重新在 `step 1250` 达到 `500.0 solved`，并在复合扰动审计中 `100ep x 20 seeds` 全部 `500.0`。
3. 当前 solved 基线中，`post_entry` 确实被激活，但它表现为“后段保护器”，不是“主训练链修复器”。
4. 现有 `post_entry` 行为有较多集成测试约束，关键行为与当前实现一致；当前更大的问题是状态机复杂、参数面过宽、可维护性差，而不是立即可见的功能错误。

## 3. 模块结构梳理

### 3.1 上游输入

`post_entry` 状态机同时消费两类信号：

- 外部评估反馈：`set_external_eval_feedback()`
  - 位置：`aletheia/aletheia_train.py:6350`
  - 作用：更新 `external_eval_best_mean`、entry/persistence/post_solved confirmation streak、设置 `post_entry_hold_until_step`、`post_entry_source`、`external_post_entry_pending_commit`

- 训练内生信号：`_resolve_imag_controller_stage()`
  - 位置：`aletheia/aletheia_train.py:6674`
  - 作用：根据 imagined `continue mean`、`gap_abs`、`actor_target_raw`、`episode_return_ema`、外部评估确认状态，决定 stage

### 3.2 中间状态

状态机核心 stage：

- `idle`
- `pretrigger`
- `entry_probe`
- `trigger`
- `internal_post_entry`
- `post_entry`
- `post_entry_soft`
- `post_entry_pending`
- `handoff`
- `persistence`
- `persistence_release`
- `post_solved`

对 `post_entry` 相关阶段最关键的内部状态：

- `_adaptive_imag_post_entry_hold_until_step`
- `_adaptive_imag_post_entry_source`
- `_adaptive_imag_last_post_entry_source`
- `_adaptive_imag_external_post_entry_pending_commit`
- `_adaptive_imag_post_entry_commit_candidate_since_step`
- `_adaptive_imag_post_entry_pending_recovery_latched`
- `_adaptive_imag_post_entry_commit_highwater_hold_until_step`

### 3.3 下游影响

一旦 stage 被解析出来，会影响两条训练链：

- imagined critic/continue cap 链
  - 位置：`aletheia/aletheia_train.py:7616`
  - 作用：为 `post_entry / post_entry_soft / post_entry_pending / handoff / persistence` 施加不同的 continue cap / floor

- imagined actor / negative advantage guard 链
  - 位置：`aletheia/aletheia_train.py:3550`、`aletheia/aletheia_train.py:8134`
  - 作用：按 stage 施加 actor scale、critic multiplier、pending latch、soft/midwater/highwater guard

换句话说，`post_entry` 不是一个孤立开关，而是直接参与 imagined actor 与 critic 的权重调制。

## 4. 当前 solved 基线中的真实运行轨迹

使用 `outputs/exp_seed42_v87tailfix_modefix_1500_20260312/train_metrics.jsonl` 对 fresh solved run 进行复盘后，得到：

- stage 分布：`idle=15, post_entry=5, post_entry_soft=4, trigger=1`
- 在这次 solved run 中，没有进入 `post_entry_pending / persistence / post_solved`
- `post_entry` 大致活跃在 `step 800-1000`
- `post_entry_soft` 大致活跃在 `step 1050-1200`
- `step 1250` 时训练直接 solved early-stop

这说明两件事：

1. `post_entry` 模块在当前 solved 基线中不是“死代码”，它确实参与了训练。
2. 但它也不是这次 solved 的主要贡献来源，因为 persistence/post_solved 等更后段机制尚未真正介入，训练已经 solved。

## 5. post_entry_soft 负 advantage 观测

在 solved 基线的 `post_entry_soft` 段，观测到：

- `step 1050`: `raw_adv_mean=-0.9223`, `guard_scale=0.75`, `critic_multiplier=2.0`
- `step 1100`: `raw_adv_mean=-1.3698`, `guard_scale=0.75`, `critic_multiplier=2.0`
- `step 1150`: `raw_adv_mean=-2.9723`, `guard_scale=0.75`, `critic_multiplier=2.0`
- `step 1200`: `raw_adv_mean=-3.1378`, `guard_scale=0.75`, `critic_multiplier=2.0`

同时：

- `controller_stage = post_entry_soft`
- `controller_post_entry_commit_ready = 1.0`
- `controller_post_entry_commit_gap_ok = 0.0`
- `controller_post_entry_commit_return_ok = 1.0`
- `controller_post_entry_commit_active = 0.0`

这表示当前 solved run 在后段确实出现了 imagined negative advantage，但系统按设计进入 `post_entry_soft` 保护态：

- actor 被下调到 `0.75`
- critic loss 被放大到 `2.0`
- 由于 `gap_ok` 不成立，没有直接 commit 到更硬的 stage

随后训练仍在 `step 1250` solved。

所以从证据上看：

- `post_entry_soft` 里的负 imagined advantage 不是“当前基线无法 solved”的证据
- 更像是后段保护逻辑成功接管、避免了更激进的错误更新

## 6. 代码实现与设计一致性检查

针对 `post_entry` 关键行为，已执行 13 个高相关集成测试，全部通过：

- `test_post_entry_soft_stage_holds_until_internal_health_commits`
- `test_post_entry_soft_highwater_signed_adv_can_commit_despite_large_abs_gap`
- `test_external_post_entry_without_commit_blocks_late_post_solved_activation`
- `test_post_entry_soft_blocks_post_solved_until_commit`
- `test_post_entry_soft_blocks_persistence_release_until_commit`
- `test_post_entry_soft_can_escape_to_persistence_after_confirmed_drawdown`
- `test_post_entry_soft_unresolved_negative_adv_guard_downscales_actor_and_boosts_critic`
- `test_post_entry_pending_negative_adv_guard_downscales_actor_and_boosts_critic`
- `test_post_entry_pending_negative_adv_guard_latches_until_external_commit`
- `test_post_entry_pending_recovery_latch_releases_after_non_negative_advantage`
- `test_post_entry_soft_negative_adv_guard_uses_soft_specific_strength`
- `test_post_entry_soft_negative_adv_guard_uses_highwater_strength_when_best_eval_is_high`
- `test_post_entry_soft_negative_adv_guard_uses_midwater_strength_before_highwater`

执行结果：

- `Ran 13 tests in 0.524s`
- `OK`

这说明至少在“作者当前设计意图”的语境里，关键 `post_entry` 行为是有行为性回归保护的。

## 7. 当前风险判断

### 7.1 不是当前 bug 的点

- `post_entry` 状态机没有证据显示当前实现已经偏离设计
- 负 imagined advantage 在 `post_entry_soft` 被观察到，但被 guard 正常接管
- solved 基线和扩展 seed 审计均已证明，这套机制目前没有破坏 pure-imag solved 能力

### 7.2 真正的风险点

当前更大的风险不是单个 if/else 写错，而是：

- stage 数量过多
- 参数面过宽
- 同一阶段同时叠加 cap / actor_scale / negative_adv_guard / pending latch / persistence escape
- 从外部 eval 到 internal metrics 再到 actor/critic 权重调制的链路过长

这会带来两个工程问题：

1. 可解释性差
   - 一次训练波动很难迅速判断是哪个 stage/guard 在主导

2. 维护成本高
   - 新补丁极易和已有 guard 形成隐性交互，重新制造“错误修复分支”

## 8. 下一步建议

当前不建议直接修改 `post_entry` 主逻辑。更稳妥的路线是：

1. 保持 `mode-fix` solved 基线不动
2. 补一个轻量的 `post_entry` 审计脚本
   - 输入：`train_metrics.jsonl`
   - 输出：每个 stage 的活跃步段、负 advantage 区间、guard scale、critic multiplier、commit 条件命中率
3. 如果未来要继续优化 `post_entry`
   - 优先做结构降复杂而不是继续叠加新 guard
   - 把 `post_entry_soft / post_entry_pending / persistence escape` 的责任边界进一步收紧
4. 真要做功能改动时
   - 先补 behavioral regression test
   - 再做最小变更
   - 坚决不回到“大量同时调 guard 参数”的旧路径

## 9. 当前结论

截至当前审计节点，可以把 `post_entry` 模块定性为：

- 复杂，但当前没有证据显示它已实现错误
- 会在 solved run 的后段介入，但不是本次主线恢复的一级因
- 已有较强测试约束，当前优先级低于基线稳定性维护
- 后续应以“增强可观测性、减少复杂度”为主，而不是继续把它当作 pure-imag 主线失败的解释中心

## 10. 新增审计工具与产物

为避免后续继续手工翻 `train_metrics.jsonl`，本轮新增了一个轻量审计脚本：

- 脚本：`scripts/post_entry_audit.py`

已对当前 solved 基线生成产物：

- Markdown：`outputs/exp_seed42_v87tailfix_modefix_1500_20260312/post_entry_audit.md`
- JSON：`outputs/exp_seed42_v87tailfix_modefix_1500_20260312/post_entry_audit.json`

脚本当前会自动汇总：

- stage counter
- transition points
- 各 stage 的连续活跃区间
- negative advantage 行数
- commit_ready / commit_active / gap_ok / return_ok 行数
- actor scale / critic multiplier / continue cap / target gap 等关键指标区间

这为后续继续审计 `post_entry / persistence / post_solved` 提供了可复用的基线工具。
