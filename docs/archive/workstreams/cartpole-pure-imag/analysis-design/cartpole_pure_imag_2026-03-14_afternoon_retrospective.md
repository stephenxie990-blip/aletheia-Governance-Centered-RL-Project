# CartPole Pure-Imag 2026-03-14 下午修复线复盘

## 结论先行

这次“先回退，再分析”的结论很明确：

1. 今天下午真正走偏的是 `v144-v148` 这条修复线，不是公式实现错误，也不是训练链路断了。
2. 这些改动的共同副作用是同一件事：
   - 它们让系统更早、更久地停在 `persistence`
   - 却没有把 `persistence` 本身的质量修好
   - 结果是 `trigger` 失去重新接管和纠偏的机会
3. 审计层对这件事给出了非常一致的证据：
   - 健康版本仍保留较多 `trigger` 行数，并且通常还有 `persistence_release`
   - 走偏版本在 `1400` 左右就提前掉进 `persistence`，之后长期出不来
   - 信号层面同时出现 `post_peak_trigger_effectiveness_low` 和 `trigger_release_window_not_effective`
4. 当前代码先回退是对的。
   - 已撤掉今天后半段确认走偏的 `fullrelease handoff bypass` 和 `soft-release light-tail bridge`
   - 保留了此前被真实 run 证明局部有效的两点：
     - `persistence_release` 在未完成 `post_solved` confirm 前也能承接恢复
     - negative online advantage analytic candidate 合并修复
5. 下一轮主线不应该继续从 `v145-v148` 这条线往前推。
   - 应先从当前回退后的干净基线重新跑
   - 再与 `v130 / v143` 做对照
   - 如果还要修，只能走“轻量的质量退出规则”，不能再堆新的 handoff/persistence patch 家族

## 审计层意见

这次复盘主要依赖两层审计：

1. `post_entry_audit.py`
   - 看每个 run 在 `idle / post_entry / trigger / persistence / persistence_release` 各阶段实际停了多久
   - 这层用来回答“系统有没有卡死在某个阶段”
2. `controller_signal_audit.py`
   - 看 `imagination_trust / trigger_effectiveness / handoff_necessity / model_freshness`
   - 这层用来回答“为什么会卡死”

两层审计对今天下午的结论高度一致。

### 阶段层结论

健康版本的共同特征不是“完全不进 persistence”，而是：

- 还能在 `trigger <-> persistence` 之间流动
- `persistence_release` 还能出现
- `trigger` 还有重新接管机会

走偏版本的共同特征是：

- `trigger` 行数骤减
- `persistence` 行数激增
- `persistence_release` 直接消失

最能说明问题的是这组对比：

| 版本 | late_avg | 1500 | 1750 | 2000 | 2250 | 2500 | trigger | persistence | persistence_release |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v125` | 159.7 | 167.1 | 175.5 | 147.8 | 145.7 | 162.2 | 20 | 2 | 4 |
| `v130` | 211.5 | 167.1 | 389.2 | 117.9 | 63.5 | 320.0 | 16 | 6 | 4 |
| `v136` | 124.9 | 75.2 | 164.7 | 198.1 | 129.8 | 56.9 | 18 | 5 | 4 |
| `v143` | 142.0 | 74.7 | 103.4 | 313.3 | 139.4 | 79.1 | 13 | 10 | 4 |
| `v144` | 50.5 | 69.4 | 16.1 | 124.4 | 20.9 | 21.6 | 12 | 15 | 0 |
| `v145` | 45.7 | 99.9 | 17.8 | 24.2 | 65.3 | 21.5 | 4 | 23 | 0 |
| `v146` | 45.7 | 99.9 | 17.8 | 24.2 | 65.3 | 21.5 | 4 | 23 | 0 |
| `v147` | 67.4 | 32.9 | 101.1 | 92.3 | 87.6 | 23.3 | 4 | 23 | 0 |
| `v148` | 56.3 | 106.5 | 29.7 | 86.2 | 22.5 | 36.7 | 4 | 23 | 0 |

可以直观看到：

- `v145-v148` 的坏，不是单点分数不巧，而是阶段结构整体坏了
- 它们几乎都变成了“`trigger=4, persistence=23, persistence_release=0`”的锁死形态

### 信号层结论

`controller_signal_audit.py` 给出的关键信号判断是：

| 版本 | peak_eval | final_eval | eval_drop | post_peak_trust | post_peak_trigger_eff | post_peak_handoff_need | post_peak_freshness | likely_causes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `v130` | 389.2 | 320.0 | -69.2 | 0.264 | 0.590 | 0.835 | 0.469 | `post_peak_imagination_trust_low`, `trigger_release_window_carries_high_handoff_pressure`, `persistence_release_inherits_low_trust`, `persistence_phase_low_model_freshness` |
| `v136` | 355.8 | 56.9 | -298.9 | 0.292 | 0.411 | 0.701 | 0.483 | `post_peak_imagination_trust_low`, `trigger_release_window_carries_high_handoff_pressure`, `persistence_release_inherits_low_trust`, `persistence_phase_low_model_freshness` |
| `v143` | 355.8 | 79.1 | -276.7 | 0.361 | 0.351 | 0.659 | 0.527 | `post_peak_imagination_trust_low`, `post_peak_trigger_effectiveness_low`, `trigger_release_window_carries_high_handoff_pressure`, `persistence_release_inherits_low_trust`, `persistence_phase_low_model_freshness` |
| `v145` | 355.8 | 21.5 | -334.3 | 0.356 | 0.171 | 0.682 | 0.511 | `post_peak_imagination_trust_low`, `post_peak_trigger_effectiveness_low`, `trigger_release_window_not_effective`, `trigger_release_window_carries_high_handoff_pressure`, `persistence_phase_low_model_freshness` |
| `v147` | 355.8 | 23.3 | -332.5 | 0.377 | 0.178 | 0.682 | 0.537 | `post_peak_imagination_trust_low`, `post_peak_trigger_effectiveness_low`, `trigger_release_window_not_effective`, `trigger_release_window_carries_high_handoff_pressure` |
| `v148` | 355.8 | 36.7 | -319.1 | 0.368 | 0.175 | 0.682 | 0.525 | `post_peak_imagination_trust_low`, `post_peak_trigger_effectiveness_low`, `trigger_release_window_not_effective`, `trigger_release_window_carries_high_handoff_pressure` |

这里最重要的一点是：

- `post_peak_imagination_trust_low` 几乎所有版本都有，所以它解释不了“为什么只有后半段这条线特别烂”
- 真正把 `v145-v148` 区分出来的是：
  - `post_peak_trigger_effectiveness_low`
  - `trigger_release_window_not_effective`

换句话说，今天后半段的问题不是“系统太敢冒险”，而是相反：

- 系统已经掉进低质量状态
- 但 `trigger` 的再接管窗口被做得越来越没用
- 保护在，但纠偏没有了

## 今天下午改动的阶段性复盘

### 一段有效但没有彻底解决的线

这条线的代表版本是：

- `v123`
- `v125`
- `v130`
- `v136`
- `v143`

它们的共同特点是：

- 没有把系统永久钉在 `persistence`
- 即使后段起伏大，也还保留某种“回到 trigger 再试一次”的流动性

其中最值得保留的结论：

1. `v125`
   - 说明固定 `handoff block 100` 比更早的 release-tail 变体更稳
2. `v130`
   - 说明 `highwater floor` 真正接管时，确实能把 late-stage retention 抬起来
   - 这是今天下午最强、最干净的一条正向证据
3. `v143`
   - 说明 `persistence_release` 在未完成 `post_solved` confirm 前也能承接恢复，这一条是有价值的
   - 但这条修复本身还不足以解决后段稳定性

### 被证伪的线

这条线的代表版本是：

- `v131`
- `v132`
- `v144`
- `v145`
- `v146`
- `v147`
- `v148`

它们虽然改动点不同，但本质上都在做同一类事情：

- 更强地留住 `persistence`
- 更早地让 `persistence` 接管
- 更晚地允许 `trigger` 回来

真实结果证明，这类方向总体是错的。

其中最明确的死胡同是：

1. `v145 / v146`
   - `fullrelease handoff bypass`
   - 它直接把系统更早推进到长期 `persistence`
   - 审计层首次明确给出 `trigger_release_window_not_effective`
2. `v144`
   - `soft-release light-tail bridge`
   - 本质不是修好问题，而是把崩点往后搬
3. `v147 / v148`
   - 在已经锁死的 `persistence` 上再叠 negative-adv 保护
   - 并没有恢复 `trigger` 的接管能力
   - 只是把低分 `persistence` 再包装一层

## 为什么测试全绿，却没拦住这次退化

这次最值得吸取的教训，不是“测试没用”，而是：

- 测试验证的是局部规则是否按设计生效
- 这次真正退化的是长链动态是否还能自我纠偏

两者不是一回事。

### 1. 今天的测试主要锁的是局部门控，不是 run 级稳定性

今天涉及的测试大多属于这种类型：

- handoff 是否在指定条件下触发
- release progress gate 是否生效
- negative online advantage analytic scale 是否在指定阈值下激活
- persistence release 是否能承接单次恢复

这些测试非常重要，因为它们能防止“代码写错”。

但它们无法回答下面这个更大的问题：

- 当这些局部规则组合到真实 2500-step 训练里时，系统会不会逐渐失去 `trigger` 的再接管能力？

这需要 run 级审计才能看出来。

### 2. 单元/集成测试用了大量 mocked controller state

很多测试为了稳定，会直接构造：

- `controller_info`
- 固定 `continue_probs`
- 固定 `returns_raw`
- 固定 `external_eval_best_mean`

这样做的好处是：

- 可以精准测一条规则

坏处是：

- 它天然不会暴露“阶段长期占用比例变化”
- 更不会暴露“从 1400 开始卡死到 2500”这种慢性退化

### 3. 只看 best checkpoint 很容易误判 run 健康度

今天很多坏 run 仍然保留：

- `best_eval_mean_during_train = 355.8 @ 750`

但这并不代表后段健康。

最典型的反例：

- `v145`
  - `best = 355.8 @ 750`
  - 但 `2500 = 21.5`
  - `late_avg = 45.7`
  - `trigger = 4`, `persistence = 23`, `persistence_release = 0`

如果只看 `best.pt` 或 `best_eval_mean_during_train`，会以为这条 run 还不错。
但一看 `eval_history + stage_counter + controller_signal_audit`，就知道它其实已经坏掉了。

### 4. 当前测试体系缺少“晋升门槛”

今天真正缺的不是更多局部测试，而是一个更高层的 run promotion gate。

至少应该补上这样的条件：

1. 新分支不能只看 `best_eval`
2. 还必须看 `last_eval` 和 `late_avg`
3. 还必须看 `stage_counter`
4. 还必须看 `controller_signal_audit` 是否出现：
   - `trigger_release_window_not_effective`
   - 极低的 `post_peak_trigger_effectiveness`

否则就会出现：

- 代码逻辑全部正确
- 测试也全绿
- 但真实动态越来越差

## 当前代码状态判断

当前代码已经从今天后半段的死胡同改动里回退出来，处于一个更干净的状态。

已确认撤掉：

1. `fullrelease handoff bypass`
2. `soft-release light-tail bridge`
3. 只为这两条错误分支服务的额外测试

已确认保留：

1. `persistence_release` 的有效恢复承接
2. analytic scale candidate 合并修复

回退后的最小回归验证已通过：

- `3/3` targeted integration tests
- `56/56` controller / contract / audit 相关测试

但必须强调：

- 这只能说明当前代码重新回到了“局部语义正确”的状态
- 不能说明“模型已经恢复稳定”
- 当前回退后的干净基线还没有重新做新的真实训练 run

## 下一轮主线

下一轮不应继续扩展 `v145-v148` 这条线。

正确主线应该是：

1. 先用当前回退后的代码重新跑一次干净基线
   - 目标是确认回退本身有没有把系统带回 `v130/v143` 一侧的流动性
2. 再和这几条健康基线做对照
   - `v125`
   - `v130`
   - `v136`
   - `v143`
3. 如果还需要修，只能做最小增量：
   - 优先研究“低质量 persistence 何时应该主动退出”
   - 而不是继续研究“怎样让 persistence 更早、更稳、更久地接管”

下一轮设计约束应明确写死：

1. 不再新增新的 handoff bypass 家族
2. 不再新增新的 persistence freeze 家族
3. 不再把“延长保护窗口”误当成“提升稳定性”
4. 所有新修复必须先过 run 级 promotion gate

## 审计后的研究判断

今天下午的失败，不是在证明“这个模型必须靠很窄的参数窗口才能成功”。

更准确的判断是：

1. 这个模型当前确实存在一个危险倾向：
   - 一旦 late-stage 退化，就很容易滑进“被保护但无法自救”的状态
2. 今天后半段的错误改动，不是在缩小成功窗口，而是在主动把系统往这个陷阱里推
3. 所以它们表现出来像是：
   - 越修越复杂
   - 越修越炸
   - 越修越只能靠碰运气

这不是因为“世界模型一定要复杂到这样”。
更像是我们在没有把主矛盾压缩清楚前，过早对局部门控做了过多干预。

这次复盘之后，问题其实已经更清楚了：

- 不是公式错
- 不是训练链路断
- 不是某一个局部门槛错一个数字
- 而是 `trigger -> persistence` 之后，系统失去了高质量再接管能力

这就是下一轮真正要修的主问题。

## 附录：2026-03-14 实验账本

下面是今天全部 `20260314` 训练实验的统一摘要，指标口径为：

- `best@step`：训练期间出现的最佳 eval
- `last_eval`：最后一个 eval 点的分数
- `late_avg`：`1500/1750/2000/2250/2500` 可用点的均值

```text
v	best@step	last_eval	late_avg	1000	1250	1500	1750	2000	2250	2500	trigger	persist	release	name
v120	355.8@750	179.8@1500	179.8	183.5	177.4	179.8	None	None	None	None	2	0	4	exp_seed42_v120_release_only_persistcap_20260314_01
v120	355.8@750	26.1@2500	64.6	183.5	177.4	179.8	43.7	40.6	32.6	26.1	16	2	8	exp_seed42_v120_release_only_persistcap_2500_20260314_01
v121	355.8@750	32.0@2000	124.9	183.5	177.4	167.1	175.5	32.0	None	None	10	2	4	exp_seed42_v121_release_tail_bypass300_2000_20260314_01
v122	355.8@750	86.5@2000	95.5	183.5	177.4	167.1	32.8	86.5	None	None	6	6	4	exp_seed42_v122_release_tail_sustain_persistcap_2000_20260314_01
v123	355.8@750	125.6@2000	156.1	183.5	177.4	167.1	175.5	125.6	None	None	11	1	4	exp_seed42_v123_release_tail_sustain_only_2000_20260314_01
v123	355.8@750	191.6@2500	158.4	183.5	177.4	167.1	175.5	125.6	132.1	191.6	18	4	4	exp_seed42_v123_release_tail_sustain_only_2500_20260314_01
v124	355.8@750	23.8@2500	80.5	183.5	177.4	167.1	175.5	21.1	15.1	23.8	17	5	4	exp_seed42_v124_release_tail_sustain_latch100_2500_20260314_01
v125	355.8@750	162.2@2500	159.7	183.5	177.4	167.1	175.5	147.8	145.7	162.2	20	2	4	exp_seed42_v125_release_tail_sustain_handoffblock100_2500_20260314_01
v126	355.8@750	162.2@2500	159.7	183.5	177.4	167.1	175.5	147.8	145.7	162.2	20	2	4	exp_seed42_v126_release_tail_sustain_handoffblock100_confirm2_2500_20260314_01
v127	500.0@1500	182.4@2500	267.2	218.4	329.7	500.0	500.0	23.1	130.3	182.4	29	0	0	exp_seed42_v127_landing_guard_on_v125line_2500_20260314_01
v128	355.8@750	23.8@2500	90.9	183.5	177.4	167.1	175.5	70.6	17.5	23.8	20	2	4	exp_seed42_v128_landing_guard_true_exact_v125line_2500_20260314_01
v129	355.8@750	250.7@2500	105.2	183.5	177.4	167.1	64.9	18.3	25.1	250.7	18	4	4	exp_seed42_v129_landing_guard_geom_exact_v125line_2500_20260314_01
v130	389.2@1750	320.0@2500	211.5	183.5	177.4	167.1	389.2	117.9	63.5	320.0	16	6	4	exp_seed42_v130_highwaterfloor_exact_v125line_2500_20260314_01
v131	355.8@750	24.1@2500	37.3	183.5	177.4	15.9	118.5	14.5	13.5	24.1	1	21	4	exp_seed42_v131_persistencehold_exact_v125line_2500_20260314_01
v132	355.8@750	24.1@2500	37.3	183.5	177.4	15.9	118.5	14.5	13.5	24.1	1	21	4	exp_seed42_v132_confirmationbridge_exact_v125line_2500_20260314_01
v134	355.8@750	23.8@2500	60.1	183.5	212.8	120.1	23.7	81.6	51.2	23.8	2	21	4	exp_seed42_v134_persistdropconfirm2_exact_v125line_2500_20260314_01
v135	355.8@750	29.2@2500	76.3	183.5	212.8	62.3	20.3	208.2	61.7	29.2	6	17	4	exp_seed42_v135_persistdropconfirm2floor_exact_v125line_2500_20260314_01
v136	355.8@750	56.9@2500	124.9	183.5	212.8	75.2	164.7	198.1	129.8	56.9	18	5	4	exp_seed42_v136_dropconfirmguard_exact_v125line_2500_20260314_01
v137	355.8@750	19.9@2500	51.0	183.5	215.2	25.3	146.7	41.4	21.8	19.9	12	15	0	exp_seed42_v137_dropconfirmguard_basecap_exact_v125line_2500_20260314_01
v138	355.8@750	42.7@2500	97.5	183.5	212.8	176.7	91.0	106.3	70.9	42.7	17	10	0	exp_seed42_v138_lateadvgate_th65_scale06_min075_2500_20260314_01
v139	355.8@750	23.3@2500	77.0	183.5	212.8	16.5	240.0	86.9	18.4	23.3	18	5	4	exp_seed42_v139_releasefallback_exact_v136line_2500_20260314_01
v140	355.8@750	84.7@2500	106.9	183.5	177.4	165.1	14.0	218.4	52.3	84.7	17	5	4	exp_seed42_v140_postentryhold450_exact_v136line_2500_20260314_01
v141	355.8@750	232.4@2500	95.3	183.5	189.8	79.6	117.0	29.5	18.0	232.4	12	15	0	exp_seed42_v141_softrelease_negadv_bridge_exact_v136line_2500_20260314_01
v142	355.8@750	20.3@2500	123.7	183.5	75.8	74.7	103.4	313.3	106.8	20.3	17	10	0	exp_seed42_v142_softrelease_negadv_bridge_analyticfix_exact_v136line_2500_20260314_01
v143	355.8@750	79.1@2500	142.0	183.5	75.8	74.7	103.4	313.3	139.4	79.1	13	10	4	exp_seed42_v143_softrelease_negadv_bridge_analyticfix_persistrelease_exact_v136line_2500_20260314_01
v144	355.8@750	21.6@2500	50.5	183.5	189.8	69.4	16.1	124.4	20.9	21.6	12	15	0	exp_seed42_v144_softrelease_lighttail_bridge_persistrelease_exact_v136line_2500_20260314_01
v145	355.8@750	21.5@2500	45.7	183.5	189.8	99.9	17.8	24.2	65.3	21.5	4	23	0	exp_seed42_v145_fullrelease_handoff_bypass_exact_v136line_2500_20260314_01
v146	355.8@750	21.5@2500	45.7	183.5	189.8	99.9	17.8	24.2	65.3	21.5	4	23	0	exp_seed42_v146_fullrelease_handoff_bypass_landingguard_exact_v136line_2500_20260314_01
v147	355.8@750	23.3@2500	67.4	183.5	189.8	32.9	101.1	92.3	87.6	23.3	4	23	0	exp_seed42_v147_persistence_negadv_guard_exact_v136line_2500_20260314_01
v148	355.8@750	36.7@2500	56.3	183.5	189.8	106.5	29.7	86.2	22.5	36.7	4	23	0	exp_seed42_v148_persistence_negadv_critic_only_exact_v136line_2500_20260314_01
```
