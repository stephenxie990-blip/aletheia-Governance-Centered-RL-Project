# Phase 4.22：Post-Transition Certified Capture Source Replacement / Durable Authority Capture

> 日期：2026-03-20
>
> 审查基线：
> - 裁决文档：
>   [Phase4-19-20-当前根因裁决-实验口径污染与Always-On-Bootstrap-Authority问题-2026-03-20.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-19-20-当前根因裁决-实验口径污染与Always-On-Bootstrap-Authority问题-2026-03-20.md)
> - 上一刀计划：
>   [Phase4-21-PostTransition-Certified-Retention-Floor-执行计划-2026-03-20.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-21-PostTransition-Certified-Retention-Floor-执行计划-2026-03-20.md)
> - 关键 run：
>   - `v240_phase420_trigger_optimization`
>   - `v241_phase421_post_transition_certified_retention_floor`

## 0. 结论摘要

`v241` 已经把 `Phase 4.21` 的成败边界说明得很清楚：

- 这刀没有修歪。
- `post-transition window` 按合同正常打开，没有 early leakage。
- 但它也没有真正修出 `late-window authority durability`。

`v241` 的关键对账如下：

| step | eval | late_gate | post_hold | raw persistence gate | certified_capture | certified_floor_state | floored persistence gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1500` | `92.2` | `0.998` | `0.1798` | `0.2278` | `0.1780` | `0.1780` | `0.2263` |
| `2000` | `264.3` | `1.0` | `0.0762` | `0.0639` | `0.0762` | `0.0762` | `0.0633` |
| `2500` | `22.5` | `1.0` | `0.0531` | `0.0265` | `0.0531` | `0.0531` | `0.0265` |

这张表说明：

1. `certified_capture` 基本只是贴着当前 `post_hold` 走。
2. `certified_floor_state` 没有高于正在衰减中的 `post_hold`。
3. `floored persistence gate` 几乎没有高过原始 `source_hold_persistence_gate_mean`。
4. 因此 `Phase 4.21` 实际上没有形成真正的 `durable authority floor`，只是把当前退化中的 late-window authority 又复述了一遍。

正式根因固定收敛为：

> 当前主故障不再是 window 边界、不再是 entry、不再是 hold path 是否存在，
> 而是 `post-transition certified floor` 的 **capture 来源选错了**。
>
> 它现在主要消费“当前时刻已经在退化的 `post_hold / retention`”，
> 而不是消费“晚窗里最近一次真正被建立过的高质量 certified authority”。

因此，下一刀唯一允许修改的层固定为：

> **只改 `post-transition certified capture source`，不改窗口边界，不改 entry，不改 hold path 本体。**

---

## 1. 当前真正问题的因果链

`v241` 之后，当前因果链固定写成：

1. `late_gate` 不是问题。
   - `1500/2000/2500` 时已经是 `0.998 / 1.0 / 1.0`
2. `trigger_gate` 不是问题。
   - 对应同步点均为 `1.0`
3. `post-transition window activation` 不是问题。
   - `0.80` 以下保持严格中性
   - `1500` 后正常打开
4. `hold path` 本体不是问题。
   - `post_hold` 明确为正，`persistence gate` 明确为正
5. 真正的问题在于：
   - `certified_capture` 当前取源过于 contemporaneous
   - 它跟着当前退化中的 `post_hold` 一起掉
   - 于是 `certified_floor_state` 也同步衰减
   - floor 并没有承担 “把已观测过的高质量 authority 留住” 的职责

因此，当前最精确的根因裁决是：

> `Phase 4.21` 失败不是因为 floor 不存在，
> 而是因为 floor 的 capture source 仍然来自 “当前退化中的 authority snapshot”，
> 而不是 “最近被认证过的 late-window durable authority snapshot”。

---

## 2. 冻结边界

本阶段默认冻结，不允许再动：

- `post_transition_window_activation` 的边界与公式
- `late_gate`
- `trigger_gate`
- `entry` 及其任何 resolver
- `Phase 4.8 hold persistence` 本体公式
- `Phase 4.19` hold channel 边界与动力学
- `Phase 4.20` threshold / step ramp
- actor unified contract
- controller stage
- `_compute_task_cert_gate`
- `Phase 4.2 / 4.3 / 4.5 / 4.7 / 4.9 / 4.10 / 4.11 / 4.12 / 4.13`
- corridor 公式与 corridor 开关
- public CLI
- 训练入口参数
- run naming 约定

本阶段唯一允许动的层：

> `_compute_post_transition_certified_retention_floor_contract(...)`
> 内部的 `certified_capture` 来源构造

---

## 3. Phase 4.22 的唯一目标

`Phase 4.22` 只修一件事：

> 把 `post-transition certified floor` 的 capture source，
> 从 “当前时刻的 `post_hold / retention` 镜像”
> 升级成 “最近一次可被认证的 late-window authority snapshot”。

这一刀不追求：

- 让 gate 更早打开
- 让 entry 更强
- 让 hold path 更复杂
- 让 floor 无条件更大

这一刀只追求：

- 当 `post-transition` 已 fully active 时
- 如果系统曾经建立过更高质量的 late-window authority
- 后续即使当前 `post_hold` 因短时退化下滑，capture 也不应立刻同步塌缩

换句话说，这刀修的是：

> `capture source` 要对 “最近的高质量 authority” 有记忆，
> 而不是只盯着 “当前这一拍已经衰减后的 post_hold”。

---

## 4. 实现方案

### 4.1 不新增新层，只替换 `capture` 来源

保留现有 helper 名称与整体职责：

- [`_compute_post_transition_certified_retention_floor_contract(...)`](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py)

固定不改：

- 输入接口
- 输出接口
- `window_activation` 公式
- `trigger_gate` 约束
- `carry_decay` 语义
- `hold_state_floored / persistence_gate_floored` 的下游接线

只替换这一项：

- `post_transition_certified_capture`

### 4.2 新的 capture 语义

当前错误语义近似于：

- `capture ~ max(current post_hold, current retention, tiny active handoff)`

这会导致：

- 当前 authority 一旦下降，capture 立刻下降

`Phase 4.22` 改成：

> `capture` 必须优先消费 “晚窗里最近被认证过的 durable authority candidate”，
> 再与当前 `post_hold` 做保守合成。

### 4.3 新增一个内部 candidate，但不新增独立层

在 helper 内部新增一个中间量，命名固定为：

- `post_transition_certified_capture_source`

这个量不出 helper，不成为新的系统层，只服务于 `capture`。

其语义固定为：

- 如果 `post-transition window` 未开：完全为 `0`
- 如果 window 已开：
  - 优先使用 `prev_post_transition_certified_floor_state` 的慢衰减版本
  - 再对照当前的 `post_hold`
  - 再对照当前的 `source_hold_retention_mean`
  - 只在当前观测仍满足最基本真实支撑时，才允许把历史 authority 延续进 capture

### 4.4 capture 合同固定为“历史高质量 authority 优先，当前退化值只作下界”

合同固定改成：

- `post_transition_window_activation = clamp((critic_contract_bootstrap_late_gate - 0.80) / 0.20, 0, 1)`
- `post_transition_current_authority = max(critic_contract_bootstrap_hold_state_post_transition, source_hold_retention_mean)`
- `post_transition_historical_authority = 0.95 * prev_post_transition_certified_floor_state`
- `post_transition_capture_support = clamp(source_hold_seed_mean * source_hold_health_mean, 0, 1)`
- `post_transition_certified_capture_source = max(post_transition_current_authority, min(post_transition_historical_authority, post_transition_capture_support))`
- `post_transition_certified_capture = post_transition_window_activation * post_transition_certified_capture_source`
- `post_transition_certified_floor_candidate = max(carry_decay * prev_post_transition_certified_floor_state, post_transition_certified_capture)`

固定语义解释：

1. 历史 authority 只允许以慢衰减形式继续存在，不允许无上界累积。
2. 历史 authority 不能脱离当前最基本的 `seed * health` 支撑。
3. 当前 `post_hold` 仍然是保守下界，但不再是 capture 的唯一主来源。
4. 一旦 `seed * health` 很低，历史 authority 不能被无条件硬灌进 capture。

### 4.5 为什么这刀仍然是“只改 capture 来源”

本方案虽然引入了 `post_transition_certified_capture_source` 这个内部量，但它：

- 不改变 window activation
- 不改变 hold state update
- 不改变 persistence gate update
- 不改变 entry
- 不改变 trigger / late gate
- 不新增新的 trainer state

它只是把原本“capture 取什么”的逻辑改掉。

因此这仍然严格属于：

> 只改 `capture 来源`

---

## 5. 观测指标

新增并导出到 [`aletheia_train.py`](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py) 与 [`aletheia_api.py`](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py)：

- `critic/critic_contract_bootstrap_post_transition_current_authority`
- `critic/critic_contract_bootstrap_post_transition_historical_authority`
- `critic/critic_contract_bootstrap_post_transition_capture_support`
- `critic/critic_contract_bootstrap_post_transition_certified_capture_source`

保留并继续主判读的现有指标：

- `critic/critic_contract_bootstrap_post_transition_window_activation`
- `critic/critic_contract_bootstrap_post_transition_certified_capture`
- `critic/critic_contract_bootstrap_post_transition_certified_floor_state`
- `critic/critic_contract_bootstrap_post_transition_hold_state_floored`
- `critic/critic_contract_bootstrap_post_transition_persistence_gate_floored`
- `critic/critic_contract_bootstrap_hold_state_post_transition`
- `critic/critic_contract_bootstrap_source_hold_persistence_gate_mean`
- `critic/critic_contract_bootstrap_late_gate`
- `critic/critic_contract_bootstrap_trigger_gate`

---

## 6. 测试计划

### 6.1 代码层测试

只改：

- [`aletheia/tests/test_hold_state_channels.py`](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_hold_state_channels.py)
- [`aletheia/tests/test_training_loop_integration.py`](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_training_loop_integration.py)

新增或扩展这些断言：

1. `late_gate < 0.80` 时：
   - `post_transition_certified_capture_source == 0`
   - `post_transition_certified_capture == 0`
   - floor 行为保持完全中性

2. `late_gate == 1.0`，当前 `post_hold` 较低，但 `prev_floor_state` 较高、`seed * health` 仍为正时：
   - `certified_capture_source > current post_hold`
   - `certified_capture` 不再简单贴着当前 `post_hold`

3. `late_gate == 1.0`，`prev_floor_state` 高，但 `seed * health` 很低时：
   - `capture_source` 必须被 `capture_support` 压住
   - 不允许出现与当前真实支撑严重脱钩的虚高 capture

4. `trigger_gate = 0` 时：
   - floor 仍必须释放
   - 不允许把历史 authority 变成新的 always-on 幻觉

5. 现有护栏不回归：
   - `effective_contact_on_valid >= requested_floor_on_valid`
   - `floor_shortfall_on_valid == 0.0`
   - `authority_shortfall_on_valid == 0.0`

### 6.2 严格真实 run

新 run 名称固定为：

- `v242_phase422_post_transition_certified_capture_source_replacement`

参数完全对齐 `v241`：

- `seed = 42`
- `steps = 80000`
- `eval_interval = 250`
- `eval_episodes = 15`
- 其余入口参数与 `v241` 保持一致
- 除 `run_name / save path` 外，不允许再引入新的配置漂移

---

## 7. 通过标准

### 7.1 结构通过标准

必须同时满足：

- `1250-1400` 区间：
  - `post_transition_window_activation == 0.0`
  - 不允许 early leakage
- `1500-2500` 区间：
  - `post_transition_window_activation > 0.0`
  - `post_transition_certified_capture_source >= post_transition_current_authority`
- 至少一个 `2000/2250/2500`：
  - `post_transition_certified_capture_source > post_transition_current_authority`
  - 证明 capture 不再只是当前 `post_hold` 镜像
- 至少一个 `2000/2250/2500`：
  - `post_transition_persistence_gate_floored > source_hold_persistence_gate_mean`
  - 证明 floor 对 persistence gate 形成了实际托举
- 任何同步点都不允许：
  - `capture_source` 明显高于 `capture_support`
  - 避免虚高 authority 幻觉

### 7.2 行为通过标准

这一步的目标不是直接 solved，而是证明：

> capture source 改正后，`late-window authority durability` 不再只是同步复述退化中的 `post_hold`

最低行为目标固定为：

- `eval@1250 >= 203.9`
- `eval@1500 > 92.2`
- `eval@1750 > 51.9`
- `eval@2000 >= 264.3`
- `eval@2250 > 25.0`
- `eval@2500 > 22.5`

更重要的行为判读不是单点峰值，而是：

- `2000` 之后不再从高点评估直接塌回 `20-25`

---

## 8. 失败分流

如果失败，必须按以下方式分流，不允许重新摊大问题：

### 8.1 如果 `capture_source` 仍然几乎等于 `current_authority`

说明 `Phase 4.22` 仍未摆脱 contemporaneous mirror 形态。

下一步继续留在同一层：

> 只增强 `historical authority` 的 capture 权重，
> 不动窗口边界，不动 hold path 本体。

### 8.2 如果 `capture_source` 已明显高于 `current_authority`，但行为仍然崩

说明问题已经不在 capture 来源，
而在：

> floor downstream coupling 太弱，
> `hold_state_floored / persistence_gate_floored` 没有把 capture 真正转成可消费 authority

那时才允许进入下一层：

> 只改 floor-to-gate coupling，不改 capture 来源本体。

### 8.3 如果 `1250/1500` 回归

说明 capture 改法虽然 nominally 只在 post-transition 工作，但实际仍通过状态链影响了早段。

那时优先收回 `capture_support` 影响范围，
不允许回头改 entry / window 边界。

---

## 9. 最终裁决句式

本阶段唯一合法裁决句式固定为：

> `Phase 4.21` 失败不是因为 floor 没有被接上，
> 而是因为 floor 的 capture 仍然主要来自当前退化中的 late-window authority snapshot。
>
> `Phase 4.22` 的唯一正确刀口，
> 就是把 `post-transition certified capture` 改成“历史高质量 authority 优先、当前 authority 只作保守下界”的 capture source replacement。

