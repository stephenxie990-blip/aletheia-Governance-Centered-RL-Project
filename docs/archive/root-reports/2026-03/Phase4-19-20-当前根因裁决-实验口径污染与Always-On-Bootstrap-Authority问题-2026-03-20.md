# Phase 4.19 / 4.20 当前根因裁决：实验口径污染 + Always-On Bootstrap Authority 问题

> 日期：2026-03-20
>
> 本文主裁决对象：
> - `v237_phase418_hold_state_channel_expansion`
> - `v238_phase419_hold_channel_dynamics_calibration`
> - `v239_phase419_fixed`
> - `v240_phase420_trigger_optimization`
>
> 本文主目的：
> - 把 `Phase 4.19 / 4.20` 的失败来源从“代码本体”与“实验口径污染”中拆开
> - 确认当前唯一允许继续修改的层
> - 明确禁止继续横跳回 corridor / entry threshold / actor / controller / floor / authority

## 1. 结论摘要

本轮复核后的正式结论固定为：

1. `v238` 不能作为否定 `Phase 4.19` 动力学校准本体的证据。
   真实问题不是 “3 通道 hold channel 方案天然错误”，而是该 run 的 `resolved config` 已经漂移，导致它并非“只改了 `Phase 4.19` 动力学”的干净对照。

2. `v239 / v240` 证明了两件不同的事：
   - `Phase 4.19` 的 `0.55 / 0.80` 通道边界和 3 通道 hold state 设计本身不是错刀。
   - `Phase 4.20` 通过降低 `eval_threshold` 和放缓 `step_ramp`，修掉了 “bootstrap entry 太脆、一次跌破 200 就掉电” 的一部分问题。

3. 但 `v240` 同时证明：
   - 当前真正主问题已经不再是 “entry 进不去”
   - 也不再是 “corridor 全零”
   - 而是 `bootstrap authority / retention` 在高退化窗口里变成了 **always-on availability**，却没有足够强的 **certified authority floor**
   - 结果是：gate 长期开着，但真正的 late-window hold / persistence 会被持续高 degradation 慢慢抽空

4. 因而当前真正根因应裁决为：

> `Phase 4.19 / 4.20` 之后，Phase 4 的主故障已经从 “entry 缺失” 上移为：
> **实验口径污染掩盖下的 Always-On Bootstrap Authority 问题。**
>
> 更具体地说，是：
> `bootstrap availability` 被修成了更常开，
> 但 `post-transition late-window` 中真正能被认证保留的 authority / retention floor 仍然不够强。

5. 后续唯一允许继续动的层必须收缩为：

> **late-window post-transition certified retention floor / authority strength**

不是 corridor，不是 entry threshold，不是 actor/controller，不是 floor / authority 主执行链，也不是再回去改 `Phase 4.19` 的 channel 划分。

---

## 2. 需要先纠正的三条误判

### 2.1 误判一：`v238` 的失败等于 `Phase 4.19` 代码本体失败

这条结论不成立。

对账 `resolved config` 后可以确认：

- `v237` 中：
  - `adaptive_imag_task_corridor_enabled = True`
  - `adaptive_imag_idle_corridor_advantage_blend_max = 0.5`
  - `adaptive_imag_actor_use_target_value_ruler_enabled = True`
  - `value_real_anchor_use_mc_returns = True`
  - `imag_continue_prob_cap = 0.95`
- `v238` 中：
  - `adaptive_imag_task_corridor_enabled = False`
  - `adaptive_imag_idle_corridor_advantage_blend_max = 0.0`
  - `adaptive_imag_actor_use_target_value_ruler_enabled = False`
  - `value_real_anchor_use_mc_returns = False`
  - `imag_continue_prob_cap = 0.0`

这说明 `v238` 已经不是 “旧配置 + 新动力学” 的干净对照，而是连任务走廊、critic corridor、target ruler、value anchor 与 imagined continue cap 一起发生了口径漂移。

因此，`v238` 只能证明：

> 实验口径污染会直接把 Phase 4 结论打坏。

它不能证明：

> `Phase 4.19` 的 3 通道 hold channel 动力学本体一定错误。

### 2.2 误判二：`v239 / v240` 的早期轨迹差异主要来自“新代码额外消耗 RNG”

这条结论目前证据不足。

本轮检查到的 `Phase 4.19` 新增逻辑：

- [`_active_hold_channel(...)`](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L1396)
- [`_update_hold_state_channels(...)`](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L1414)
- hold update call-site [`aletheia_train.py#L17550`](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L17550)

这些代码路径都是确定性算子，没有直接引入新的采样调用。

这不代表它绝对不会改变训练轨迹，但目前更强的解释是：

- `v239 / v240` 与 `v237` 不是同一份 `resolved config`
- 它们从第 `0` 步就启用了完整 SAI bundle
- 因此早期行为变化首先应归因于 **配置链变化**，而不是先归因于 “新增 helper 改变 RNG 消耗序列”

### 2.3 误判三：过去多次 `500` 已经足够说明旧线稳定 solved

这条结论也必须收紧。

当前真实 run 的评估协议并不统一：

- `v237` 的 `eval_episodes = 5`
- `v239 / v240` 的 `eval_episodes = 15`

而 CartPole-v1 官方 solved 标准是：

> 连续 `100` 局平均回报 `>= 475`

因此，当前仓库里很多历史 `500` 更准确的含义只是：

> 在一个短评估 checkpoint 上打满了当次 episode 样本

它并不自动等价于：

> 已达到官方 solved 的稳定收敛

这也解释了为什么旧线虽然多次出现 `500`，但最终分数经常重新崩回低位。

---

## 3. 行为总对账

### 3.1 `v237 / v239 / v240` 的主行为轨迹

| step | `v237` | `v239` | `v240` |
| --- | ---: | ---: | ---: |
| `250` | `15.4` | `46.7` | `46.7` |
| `500` | `57.0` | `218.1` | `218.1` |
| `750` | `144.8` | `24.0` | `62.1` |
| `1000` | `146.2` | `89.7` | `31.1` |
| `1250` | `194.2` | `251.9` | `203.9` |
| `1500` | `115.6` | `63.8` | `92.2` |
| `1750` | `188.4` | `48.0` | `51.9` |
| `2000` | `155.0` | `-` | `264.3` |
| `2250` | `176.4` | `-` | `25.0` |
| `2500` | `191.8` | `-` | `22.5` |

这个对账说明：

- `Phase 4.19` 并没有天然压死上限，`v239@1250 = 251.9` 已经高于 `v237@1250 = 194.2`
- `Phase 4.20` 也不是完全失败，`v240@2000 = 264.3` 证明更低阈值能让 contract 在更差 checkpoint 后继续工作
- 但 `v240` 的 `2250/2500` 暴跌，又证明 “让 gate 更常开” 本身并不等于 “把晚窗保住”

### 3.2 历史大盘的冷事实

按当前仓库 `outputs` 中可直接对账的 `exp_seed42_*_2500_*` 工件统计：

- 旧组 `v207-v238`：`37` 个 run
- 新组 `v239-v240`：`2` 个 run

统计结果如下：

| 指标 | 旧组 `v207-v238` | 新组 `v239-v240` |
| --- | ---: | ---: |
| run 数 | `37` | `2` |
| `best >= 475` | `16` | `0` |
| `best >= 300` | `26` | `0` |
| `best < 200` | `5` | `0` |
| `max(best)` | `500.0` | `264.3` |
| `avg(best)` | `382.9` | `258.1` |
| `avg(final)` | `183.0` | `35.3` |
| `best >= 475 且 final >= 475` | `5` | `0` |

这张表的正式含义是：

- 新线当前上限确实明显下降
- 但旧线本身也远谈不上稳定 solved
- 旧线真正“既打到高分又最终保住”的 run 很少

因此，当前不能得出：

> “旧线稳定，新线只是代码写坏了”

更准确的说法应该是：

> 旧线本来就是高方差、概率性命中的不稳定系统；新线目前又在一个被实验口径污染的阶段，尚未完成公平 A/B。

---

## 4. 结构证据：`v239 / v240` 实际修到了哪里

### 4.1 `v239` 证明了 `Phase 4.19` 不是错刀

`v239` 的关键行为点：

- `1000`: `89.7`
- `1250`: `251.9`
- `1500`: `63.8`
- `1750`: `48.0`

`1250` 的突破说明：

- `0.55 / 0.80` 的通道边界没有提前把 early path 搅坏
- `3` 通道 hold state 架构并没有天然压制 bootstrap 进入后的行为上限
- `Phase 4.19` 至少具备 “把中段和晚窗拆到不同 channel” 的结构合理性

### 4.2 `v240` 证明了 `Phase 4.20` 修的是 availability，不是 authority

`v240` 的关键 train-side 指标：

| step | late_gate | trigger_gate | entry_deg | post_hold | hold_persistence_gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1500` | `0.998` | `1.0` | `0.4904` | `0.1798` | `0.2263` |
| `2000` | `1.0` | `1.0` | `0.7622` | `0.0762` | `0.0633` |
| `2500` | `1.0` | `1.0` | `0.9054` | `0.0531` | `0.0212` |

对应行为：

- `2000`: `264.3`
- `2250`: `25.0`
- `2500`: `22.5`

这组数据的正式含义非常明确：

1. `late_gate` 与 `trigger_gate` 已经不是主问题。
   它们在后段基本是常开状态。

2. `entry_deg` 也不是主问题。
   它在 `2000-2500` 反而越来越高。

3. 真正塌的是：
   - `post_transition` hold state
   - `source_hold_persistence_gate`

也就是说：

> `Phase 4.20` 把 bootstrap 的可用性修成了更常开，
> 但没有同时修出一个足够强、足够抗退化的 certified retention floor。

因此，系统进入了一个新的失败形态：

> gate 还开着，contract 也还活着，
> 但真正能被保住的 authority / persistence 已经在持续高 degradation 中衰空。

这就是本文所说的：

> **Always-On Bootstrap Authority 问题**

更精确地说，它其实是：

> availability always-on，authority not certified enough

---

## 5. 当前真正根因裁决

本轮正式根因裁决固定写成：

> `Phase 4.19 / 4.20` 当前不再卡在：
> - corridor 是否存在
> - entry 是否能打开
> - trigger gate 是否会归零
> - floor / authority 主执行链是否短路
>
> 当前真正的问题是：
> **实验口径污染掩盖下，bootstrap availability 与 bootstrap certified authority 没有被正确拆分。**

更具体一点：

1. `v238` 暴露的是实验口径污染。
   该 run 的 resolved config 已经偏离基线，不具备判刀资格。

2. `v239` 暴露的是：
   3 通道 hold state 与新通道边界并没有天然错误，至少中段是能抬起来的。

3. `v240` 暴露的是：
   当 `eval_threshold` 放低、`step_ramp` 放缓后，bootstrap 进入与持续激活被恢复了；
   但系统只是更长时间地保持 “可介入”，并没有在持续坏状态下提供足够强的 late-window certified retention floor。

因此，当前 Phase 4 的本质问题不是：

- actor 路线失败
- controller 路线失败
- SAI 路线整体失败
- `Phase 4.19` channel expansion 理论失败

而是：

> 我们已经把 “能进来” 修得比以前更好，
> 但还没有把 “进来以后在高 degradation 晚窗里到底凭什么继续保留 authority” 修正确。

---

## 6. 当前底座是否稳定

### 6.1 可以继续视为稳定的层

当前仍可视为冻结底座的层：

- actor unified contract
- controller stage
- `_compute_task_cert_gate`
- `Phase 4.2` floor overlay
- `Phase 4.3` recert
- `Phase 4.5` first-drop safe activation
- `Phase 4.7` transition bridge
- `Phase 4.9 / 4.10` entry recoupling 主体
- `Phase 4.11 / 4.12 / 4.13` terminal source / activation edge / consumer 层
- floor / contact / authority 主执行链

### 6.2 仍不稳定的层

当前唯一仍不稳定的核心层是：

> `post-transition late-window` 中，
> `bootstrap authority / hold persistence` 的 certified retention floor

换句话说：

- 底座执行链没有重新坏掉
- 问题也没有退回到 corridor 或 entry 缺失
- 当前坏的是 “always-on availability 之后的 authority 强度”

---

## 7. 后续唯一允许动的层

从现在开始，后续只允许动一层：

> **late-window post-transition certified retention floor / authority strength**

### 7.1 允许修改的对象

只允许改这类语义：

- `post_transition` channel 在持续高 degradation 下的最小 certified floor
- `source_hold_persistence_gate` 在后段 fully-active late-window 中的 authority 强度维持
- availability 已经打开之后，真正允许 bonus value 被保留的 certified authority 条件
- `post_transition` hold state 与 persistence gate 之间的 authority-strength coupling

### 7.2 明确禁止继续修改的层

以下层禁止再动：

- corridor 基础开关与 corridor 公式本体
- `adaptive_imag_critic_bootstrap_eval_threshold`
- `adaptive_imag_critic_bootstrap_step_ramp`
- trigger entry 公式本体
- actor unified contract
- controller stage
- floor / contact / authority 主执行链
- terminal source family
- public CLI / run naming / training entry params

正式禁止的错误方向包括：

- 再次把问题写成 “entry 还不够强”
- 再次把问题写成 “corridor 没打开”
- 再次把问题写成 “继续把 gate 放得更开”
- 再次横跳回 actor / controller / task corridor 总线

---

## 8. 唯一允许下一刀的正式表述

下一刀如果继续留在 `Phase 4`，正式目标必须写成：

> **在不再修改 entry / corridor / threshold 的前提下，给 `post-transition late-window` 建立一个更强的 certified retention floor，使 bootstrap authority 在高 degradation 条件下不再只剩 availability，而能保留真正有效的 persistence strength。**

这刀的判定标准也应固定成：

1. 不再接受只看 `late_gate / trigger_gate` 的通过标准。
2. 必须同时看：
   - `post_transition hold state`
   - `source_hold_persistence_gate`
   - 晚窗 `2250 / 2500` 行为是否继续塌陷
3. 若 gate 常开而 `post_transition hold` 与 `persistence_gate` 仍持续下滑，则直接判为 authority-strength 失败。

---

## 9. 最终裁决

最终裁决固定为：

> `Phase 4.19 / 4.20` 当前还不能判通过。
>
> 但当前失败也不能再被简单归因为 “新代码写坏了” 或 “3 通道 hold state 理论失败”。
>
> 已确认存在两层问题：
>
> 1. `v238` 级别的实验口径污染
> 2. `v240` 级别的 Always-On Bootstrap Authority 问题
>
> 在剥离实验污染之后，当前唯一允许的下一刀已经收敛为：
>
> **late-window post-transition certified retention floor / authority strength**

