# CartPole Pure-Imag `v168` Next-Layer Diagnosis

Date: 2026-03-15
Status: complete

## Core Question

`v168` 已经把系统从 `v167` 的“后段一路塌”修成了“中后段还能恢复并重新回到高分区”，
为什么最后还是守不住 current checkpoint？

## High-Confidence Observation

`v168` 不是简单地“还是一样坏”。

它已经出现了三个结构性改善：

1. 早期进入 corridor 更强
2. 后段恢复能力更强
3. late-stage 内部错位幅度明显小于 `v167`

关键对比：

### `v167 @2000`

- `critic/value_target_gap_abs_mean = 30.03`
- `actor/online_adv_mean = -29.95`
- `eval = 177.8`

### `v168 @2000`

- `critic/value_target_gap_abs_mean = 4.69`
- `actor/online_adv_mean = -4.67`
- `eval = 154.8`

这说明：

- 第二刀显著降低了内部错位幅度
- 但外部评估并没有同比例变成稳定

因此下一层诊断不能再简单说：

`critic gap 还在，所以它就掉了`

真正的问题已经更细：

`为什么中度错位就足以让 current checkpoint 最终失守？`

## Diagnosis

当前最合理的解释是：

`系统的剩余主病灶已经不是“几何不对”或“绝对锚缺失”，而是 RL-side policy improvement loop 在高分 corridor 上仍然欠阻尼。`

更直白地说：

- world-model side 现在已经能把 actor-use manifold 维护得更久
- critic 也不再像 `v167` 那样迅速深度通胀
- 但 actor/critic 这套 self-bootstrapping 改进闭环仍然会在高分 corridor 上做过冲

所以：

- 现在的问题不是“路找不到”
- 而是“车会在弯道里来回摆，最后还是出线”

## Why `v168` Looks Like Oscillation Instead Of Collapse

这次最值得注意的形状是：

- `750 -> 432.6`
- `1250 -> 186.0`
- `1750 -> 258.2`
- `2250 -> 340.0`
- `2500 -> 58.4`

这不是单调坠落，而是：

`进入 corridor -> 偏出 -> 拉回 -> 再偏出 -> 最终失守`

这类曲线更像：

- corridor 本身还在
- 进入能力也还在
- 但 policy/value 更新在 corridor 内的阻尼不足

## Pathology vs Cause

### 病理表现

- `value > return`
- `online_adv < 0`
- current checkpoint 回撤

### 更深病因

- actor/critic 在高分 corridor 上的改进步长仍然偏大
- 但这次不是通过 world model feature drift 放大
- 而是通过 RL-side ruler mismatch 累积出来

换句话说：

`第二刀已经把上游 manifold 维护得更好了，剩余问题更像下游 actor/critic 自举闭环的欠阻尼。`

## Why This Matters

这意味着下一步已经不该继续优先修改：

- replay anchor
- world model 几何对齐
- controller rescue

因为这些层面都已经带来了明显改善。

剩余问题更像：

`high-value corridor 内的 policy/value update law 还不够稳。`

## Practical Interpretation

如果把系统想成三层：

1. `truth grounding`
   - `MC anchor`
2. `corridor geometry / semantics`
   - `policy open-loop corridor maintenance`
3. `corridor-internal improvement dynamics`
   - actor/critic 在高分 corridor 内如何继续更新而不出线

那么 `v168` 之后，前两层已经明显改善，
第三层成了新的主战场。

## Next-Step Hypothesis

当前最值得验证的新假设是：

`模型最后守不住，不是因为它又找不到高分 corridor，而是因为它在 corridor 内的 actor/critic 更新依然过冲。`

如果这个判断成立，那么下一刀应优先研究：

- 如何降低 high-value corridor 内的 value-ruler overshoot
- 如何让 actor 在 `online_adv` 轻度转负时不要继续把当前 checkpoint 推离 corridor
- 如何让“已经进入的高分 corridor”变成真正的稳定吸引子，而不是可往返的临时轨道

## Final Diagnosis

`v168` 把系统推进到了一个更深但更清晰的阶段：`

`从“找不到 corridor”推进成了“能进 corridor，但 corridor 内更新仍欠阻尼，最终会振荡失守”。`
