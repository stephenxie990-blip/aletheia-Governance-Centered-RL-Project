# CartPole Pure-Imag 信号窗口研究结论

日期：2026-03-14  
依据：

- [signal 对比报告](./cartpole_pure_imag_signal_run_audit_2026-03-14.md)
- [signal 对比 JSON](./cartpole_pure_imag_signal_run_audit_2026-03-14.json)

目标：

> 用统一的 `imagination_trust / trigger_effectiveness / handoff_necessity / model_freshness` 语言，重新解释 `v123 / v125 / v126` 的差异，明确下一轮修复主线。

## 1. 这轮最重要的两个结论

### 1.1 `v125` 和 `v126` 在真实内部轨迹上几乎完全相同

从 signal 审计结果看：

- `v125` 与 `v126` 的 post-peak 平均信号完全重合
- 两者在 `2000 / 2250 / 2500` 这些关键 checkpoint 上的 stage 和 signal 值也完全相同
- 最终 eval 也完全相同：`162.2`

这说明：

> `confirm=2` 基本没有改变系统的真实控制轨迹。

所以后续不应该继续把主要精力放在“多加一步确认”这类补丁上。

### 1.2 `handoff block` 不是无效，但它把系统推到了一个更差的 persistence 落点

`v123` 与 `v125/126` 的关键差异，不是全局均值突然天差地别，而是：

> 在 `2000` 这个关键点，系统落入了不同的状态截面。

## 2. 关键 checkpoint 对照

### 2.1 Step `1750`

三者完全一致：

- stage: `trigger`
- trust: `0.281`
- trigger_effectiveness: `0.738`
- handoff_necessity: `0.787`
- freshness: `0.486`

这说明在 `1750` 之前，`v123 / v125 / v126` 还没有真正分叉。

### 2.2 Step `2000`

这里开始出现决定性的分叉。

`v123`：

- stage: `trigger`
- trust: `0.251`
- trigger_effectiveness: `0.756`
- handoff_necessity: `0.846`
- freshness: `0.492`
- eval: `125.6`

`v125 / v126`：

- stage: `persistence`
- trust: `0.147`
- trigger_effectiveness: `0.066`
- handoff_necessity: `1.000`
- freshness: `0.327`
- eval: `147.8`

这个截面对我们很重要。

它说明：

- `handoff block` 并不是简单地“让 handoff 更晚发生”
- 它实际上改变了系统在 `2000` 时刻的着陆位置
- `v125 / v126` 在 `2000` 直接落进了一个更低 trust、更低 freshness、更高 handoff necessity 的 persistence 截面

虽然这个 persistence 截面在 `2000` 当下的 eval 比 `v123` 更高，但它并不健康。

换句话说：

> `v125 / v126` 在中段拿到的是一个“局部更好、但内部状态更脆”的结果。

### 2.3 Step `2250`

`v123`：

- stage: `trigger`
- trust: `0.415`
- trigger_effectiveness: `0.488`
- handoff_necessity: `0.655`
- freshness: `0.568`
- eval: `132.1`

`v125 / v126`：

- stage: `trigger`
- trust: `0.282`
- trigger_effectiveness: `0.789`
- handoff_necessity: `0.819`
- freshness: `0.530`
- eval: `145.7`

这里可以看到另一个很关键的点：

- `v125 / v126` 虽然重新回到了 `trigger`
- 但 handoff pressure 仍然更高
- trust 仍然更低
- freshness 也没有明显恢复

所以它不像是“稳定修复完成后自然回到 trigger”，更像是：

> 从一个低质量 persistence 截面里弹回 trigger，但底层健康度没有真正恢复。

### 2.4 Step `2500`

`v123`：

- stage: `trigger`
- trust: `0.333`
- trigger_effectiveness: `0.799`
- handoff_necessity: `0.752`
- freshness: `0.558`
- eval: `191.6`

`v125 / v126`：

- stage: `trigger`
- trust: `0.236`
- trigger_effectiveness: `0.741`
- handoff_necessity: `0.858`
- freshness: `0.478`
- eval: `162.2`

这一步已经把差异说得很清楚了：

- `v123` 虽然也不健康，但它的 trust / freshness 仍然明显高于 `v125 / v126`
- `v125 / v126` 到最终点的 handoff pressure 更高，freshness 更低
- 这和它们的最终分数更差是一致的

## 3. 从 signal 角度看，这三次实验到底说明了什么

### 3.1 `confirm=2` 不是主线

`v125 == v126` 这一点几乎可以定论：

> 当前问题不是单次坏信号误触发，而是持续性的坏状态。

所以继续堆：

- confirmation steps
- debounce
- 局部 latch

这些修复大概率都只会继续变复杂，不会真正改变轨迹。

### 3.2 当前更像是“persistence 接管时的落地质量”有问题

从 `2000` 这个分叉点看，真正需要追问的问题不是：

> “要不要 handoff”

而更像是：

> “当系统决定进入 persistence 时，它是不是正在一个足够健康、足够可恢复的截面上落地？”

现在 `v125 / v126` 的问题，不像是 handoff 本身发生了就错，而像是：

- 落地时 trust 太低
- freshness 太差
- handoff pressure 已经过高

这会让 persistence 更像“接管一个已经坏透的尾段”，而不是“接管一个仍可修复的过渡段”。

### 3.3 这也解释了为什么平均值层面差异不大，但最终结果差很多

signal 平均值只会告诉我们总体倾向。  
但这次真正决定结果的，是少数几个关键窗口：

- `1750`
- `2000`
- `2250`
- `2500`

也就是说：

> 当前系统更像是被关键落点质量主导，而不是被全局均值主导。

## 4. 下一轮修复主线

基于这轮 signal 审计，下一轮不建议继续优先做：

- 更长的 handoff block
- 更多 confirm steps
- 更多触发后的局部补丁

更值得做的，是下面这条主线。

### 4.1 新主线：`persistence landing quality`

要显式回答一个问题：

> 当前这个时刻，进入 persistence 以后，是落到“可修复状态”，还是落到“低 trust / 低 freshness 的硬崩状态”？

这条主线优先考虑的不是“更晚 handoff”，而是：

- handoff 发生前的 landing quality 评估
- persistence entry 的软着陆版本
- 当 landing quality 太差时，优先保守 trigger，而不是硬切 persistence

### 4.2 可能的修复方向

下一轮更值得尝试的机制不是新阶段，而是新条件：

1. `persistence landing guard`
   目标：当 `trust / freshness` 低到一定程度时，不让系统直接以当前强度落进 persistence。

2. `soft persistence entry`
   目标：不是直接切成当前 persistence 全强度，而是先进入一个更保守的过渡态。

3. `pre-handoff trust shaping`
   目标：在 handoff 之前先把 imagination horizon / actor 权重 / target blend 往保守方向压一小步，改善落地质量。

## 5. 本文结论

这一轮 signal 审计把一个关键判断坐实了：

> 现在的问题不只是“handoff 该不该发生”，更是“handoff 发生时，系统是不是落到了一个足够健康的 persistence 截面”。

更具体地说：

- `v125 / v126` 证明 `confirm=2` 几乎没用
- `v123` 证明“留在 trigger 并不一定更差”
- 真正的差异发生在 `2000` 这个关键落点

所以下一轮修复主线应该从：

- “是否 handoff”

转向：

- “handoff 的 landing quality 如何评估和控制”
