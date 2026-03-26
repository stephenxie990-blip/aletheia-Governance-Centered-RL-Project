# CartPole Pure-Imag Landing Guard 实验结论

日期：2026-03-14

相关工件：

- [signal 对比报告（含 v127）](./cartpole_pure_imag_signal_run_audit_2026-03-14_v127.md)
- [signal 对比 JSON（含 v127）](./cartpole_pure_imag_signal_run_audit_2026-03-14_v127.json)
- [v125 run](../outputs/exp_seed42_v125_release_tail_sustain_handoffblock100_2500_20260314_01)
- [v127 run](../outputs/exp_seed42_v127_landing_guard_on_v125line_2500_20260314_01)

## 1. 这轮实验想验证什么

目标很简单：

> 在 `v125` 同一条实验线上，只加入这次新的 `trigger -> persistence landing guard`，看它能不能避免 `2000` 那种低质量 persistence 落点。

也就是说，这轮不是要直接证明“模型已经修好”，而是要回答一个更窄、更关键的问题：

> `v125` 里那个坏的 persistence 落地，是否真能被新的 guard 拦住？

## 2. 实际结果

### 2.1 `v127` 并没有复现 `v125` 的那条后半段轨迹

`v127` 的 eval 轨迹：

- `250`: `15.9`
- `500`: `107.6`
- `750`: `355.8`
- `1000`: `218.4`
- `1250`: `329.7`
- `1500`: `500.0`
- `1750`: `500.0`
- `2000`: `23.1`
- `2250`: `130.3`
- `2500`: `182.4`

`v125` 的对应轨迹：

- `250`: `15.9`
- `500`: `107.6`
- `750`: `355.8`
- `1000`: `183.5`
- `1250`: `177.4`
- `1500`: `167.1`
- `1750`: `175.5`
- `2000`: `147.8`
- `2250`: `145.7`
- `2500`: `162.2`

这说明：

- 两者在 `750` 之前完全一致
- `v127` 在 `1000~1750` 反而显著更强，甚至达到 `500 solved`
- 但 `v127` 在 `2000` 出现了更剧烈的塌点

所以这次实验不能被解读为：

> landing guard 成功拦住了坏 persistence，所以系统更稳了

相反，更准确的说法是：

> 当前代码下，系统走到了另一条轨迹；这条轨迹中段更强，但后段发生了更剧烈的 trigger 失稳。

### 2.2 `landing guard` 在真实 run 里实际上没有触发

这是本轮最关键的发现。

在 `v127` 的真实训练指标里：

- `imag/controller_trigger_persistence_handoff_landing_guard_active` 全程最大值是 `0.0`
- `imag/controller_trigger_persistence_handoff_active` 全程最大值是 `0.0`
- `imag/controller_standard_soft_fallback_trigger_release_active` 全程最大值也是 `0.0`

也就是说：

> `v127` 并不是“guard 挡住了坏 handoff 之后引发副作用”。

真实发生的是：

- 系统根本没有进入我们预想中的 late handoff / persistence 链路
- 它在 `1100` 后基本一直停留在 `trigger`
- 然后 `trigger` 自己在后半段失稳了

## 3. 控制阶段对照

### 3.1 `v125` 的阶段迁移

`v125`：

- `50`: `idle`
- `800`: `post_entry`
- `950`: `post_entry_soft`
- `1050`: `post_entry`
- `1100`: `post_entry_soft`
- `1250`: `trigger`
- `1300`: `persistence_release`
- `1500`: `trigger`
- `2000`: `persistence`
- `2050`: `trigger`
- `2200`: `persistence`
- `2250`: `trigger`

### 3.2 `v127` 的阶段迁移

`v127`：

- `50`: `idle`
- `1100`: `trigger`

这点非常重要。

它说明：

> 当前这轮真实 run 的主问题，已经不是 “坏 persistence 落点” 本身，而是系统压根没有走到原先那条 post-entry / persistence-release / persistence 过渡链路。

## 4. 关键窗口信号

### 4.1 `v125` 在 `2000`

- stage: `persistence`
- trust: `0.147`
- trigger_effectiveness: `0.066`
- handoff_necessity: `1.000`
- freshness: `0.327`
- `trigger_persistence_handoff_active = 1`

这正是我们之前识别出来的“低质量 persistence 落点”。

### 4.2 `v127` 在 `2000`

- stage: `trigger`
- trust: `0.118`
- trigger_effectiveness: `0.000`
- handoff_necessity: `0.528`
- freshness: `0.295`
- `trigger_persistence_handoff_active = 0`
- `trigger_persistence_handoff_landing_guard_active = 0`

这说明 `v127` 的 `2000` 坏状态和 `v125` 不是同一种坏法。

`v125` 更像：

- 进入了一个低质量 persistence 截面

`v127` 更像：

- 根本没 handoff
- 但 trigger 已经完全失去有效性
- imagination trust 和 freshness 也一起塌掉了

### 4.3 `v127` 的 trigger 为什么会“失效”

把 `trigger_effectiveness` 和 `freshness` 的组成项拆开后，可以看到：

`v127 @ 1500`：

- `trigger_effectiveness = 0.475`
- `trigger_effectiveness_adv_component = 0.730`
- `model_freshness_stability_component = 0.640`

`v127 @ 1750`：

- `trigger_effectiveness = 0.205`
- `trigger_effectiveness_adv_component = 0.315`
- `model_freshness_stability_component = 0.087`

`v127 @ 2000`：

- `trigger_effectiveness = 0.000`
- `trigger_effectiveness_adv_component = 0.000`
- `model_freshness_stability_component = 0.000`

这说明 `v127` 的 trigger 并不是被某个 handoff guard 或 release gate 人工关掉了。

更像是：

- online advantage 先持续恶化
- trigger 的有效性分量被耗尽
- freshness 的稳定性分量也同时掉空
- 系统虽然还名义上处于 `trigger`，但实际上已经失去“继续靠 trigger 保稳”的能力

换句话说：

> `trigger` 在 `v127` 的后半段是“空壳还在，控制能力没了”。

## 5. 这轮实验说明了什么

### 5.1 `persistence landing guard` 不是错方向，但它不是当前主矛盾

这次实验至少说明两件事：

- 它没有在真实 run 中被触发，所以不能把 `v127` 的崩塌直接怪到它头上
- 它也没有回答当前系统的核心问题，因为当前崩塌发生在 trigger 链路自身

所以更准确的判断应该是：

> `landing guard` 是一个有效的局部保护件，但它不是当前控制不稳定的主修复线。

### 5.2 当前更大的问题是：`trigger-only` 轨迹本身也不稳

`v127` 最值得重视的不是 `2000 = 23.1` 这个数字本身，而是：

- 它曾经在 `1500/1750` 达到并保持 `500`
- 但随后在没有 handoff、没有 persistence、没有 landing guard 介入的情况下崩掉

这意味着：

> 系统并不是只在“切错 persistence”时会失败。

它还会在另一条轨迹上失败：

> 一直停留在 `trigger`，但 `trigger_effectiveness`、`trust`、`freshness` 一起衰竭，最后高水位保不住。

## 6. 对下一轮修复主线的影响

基于这轮结果，下一轮主线不应该继续只围绕：

- `trigger -> persistence handoff 是否太早`
- `landing guard 是否足够硬`

更值得推进的新主线是：

### 6.1 `trigger exhaustion / freshness collapse`

要回答的问题变成：

> 当系统还停在 `trigger` 时，怎样判断它已经不再有效，且继续硬扛只会把 trust / freshness 一起耗光？

### 6.2 可能更优雅的修复方向

比继续叠 handoff patch 更值得做的是：

1. `trigger effectiveness decay guard`
   目标：当 `trigger_effectiveness` 接近 `0` 且 `trust/freshness` 同时快速下滑时，不允许系统继续把自己锁在 `trigger`。

2. `freshness-aware soft release`
   目标：不是立刻切进 hard persistence，而是给 trigger 一个“软退场”或“软释放”的过渡机制。

3. `highwater retention` 从“阶段切换问题”升级为“控制健康度保持问题”
   目标：让系统在拿到 `500` 之后，不是依赖某个阶段刚好生效，而是依赖一套更稳定的健康度约束。

## 7. 当前最稳妥的结论

这轮实验最重要的结论不是“landing guard 修好了系统”，而是：

> 我们证明了当前主问题已经不再只是 persistence 落点问题。

现在系统至少有两种失败模式：

1. `v125/v126` 型：
   `trigger -> persistence` 落到了低 trust / 低 freshness 的坏截面。

2. `v127` 型：
   系统根本没进 persistence，但 `trigger` 自己在 highwater 之后耗竭并失效。

所以后续修复主线应该从：

> “怎么把 handoff 修得更细”

切换为：

> “怎么让 highwater 之后的控制健康度不崩”
