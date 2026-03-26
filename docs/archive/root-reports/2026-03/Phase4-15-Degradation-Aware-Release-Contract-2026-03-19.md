# Phase 4.15：Late-Window Hold Persistence Selective Retention / Degradation-Aware Release Contract

> 日期：2026-03-19
>
> 作用范围：
> - 只允许继续留在 `Phase 4.8 hold persistence`
> - 不动 actor / controller / `_compute_task_cert_gate`
> - 不动 `Phase 4.2 / 4.3 / 4.5 / 4.7 / 4.9 / 4.10 / 4.11 / 4.12 / 4.13`
> - 不改 public CLI / run naming / training entry params

---

## 1. 这一步到底在修什么

`degradation-aware release` 修的不是“让 hold 更强”，也不是“让 semantic source 更早进来”。

它修的是：

> 当 `late window` 已 fully active，且 `Phase 4.8` 已经拿到一份曾经被 semantic health 证明有效的 bonus value hold 以后，
> 如果后续 `real-side degradation` 持续恶化，系统必须能及时撤销这份历史保留，而不是继续把它留在最终 bootstrap value 里。

一句话说：

> 这一步修的是 “stale retained trust 的撤销语义”。

---

## 2. `release` 到底 release 什么

这一步必须把“release 对象”说死，避免再出现 `v231` 那种“数值动了，但语义动错了”的情况。

### 2.1 要 release 的对象

只允许 release 下面三类量：

- `critic_contract_bootstrap_bonus_hold_state`
- `source_hold_late_certified_floor` 里来自历史 hold 的那一部分
- `bootstrap_external_bonus_value_hold_persisted` 对旧 hold source 的继续保留

### 2.2 明确不 release 的对象

这一步不允许 release：

- `bootstrap_external_authority_view`
- `bootstrap_external_authority_floor`
- `effective_contact`
- `Phase 4.11 terminal-truth source resolver`
- `Phase 4.6 source value replacement` 的 source-truth 定义
- `late_gate` / `entry trigger`

也就是说：

> `release` 不是把系统重新打回 internal bootstrap；
> `release` 只是撤销“历史 hold 继续主导 late-window bonus value”的资格。

---

## 3. 为什么 `v231` 证明我们修错了

`v231` 的关键事实：

- `2250` 时：
  - `hold_gate = 0.0291`，高于 `v229 = 0.00373`
  - `hold_delta_abs = 0.4748`，高于 `v229 = 0.0676`
- 但行为更差：
  - `2250 = 138.8`，低于 `v229 = 160.8`
  - `2500 = 143.8`，低于 `v229 = 257.4`
  - `2500` 时 `real_mc_value_gap_abs_mean = 20.3506`，高于 `v229 = 14.6976`

这说明：

- `Phase 4.14` 不是 inert
- 它确实增强了 `Phase 4.8`
- 但增强的是 “retention strength”
- 不是 “retained value 的 terminal truth”

所以 `v231` 的正式 verdict 是：

> 当前错误不是 “hold 太弱”，而是 “在高真实退化窗口里，系统会更稳定地保留一个已经不够 terminal-true 的 bonus value”。

---

## 4. 为什么这类错误不能只靠静态看公式提前断言

这一步要修的是闭环问题，不是单个张量公式问题。

原因有三条：

- `Phase 4.8` 消费的是带记忆的 `bonus_hold_state`，不是单次 batch 的纯函数输入。
- `hold persistence` 的输出会反过来影响后续训练轨迹，所以局部 gate 合理，不等于全局动力学合理。
- 当前单测主要验证“窗口前中性、truth 高时可移动、alarm 高时可收缩”，但没有覆盖“历史 hold 已点亮后，真实退化持续恶化时，系统是否会及时撤销旧 hold”这个闭环场景。

因此：

> `degradation-aware release` 必须被定义成一份动态合同，而不是只靠一条 helper 公式的局部正确性来判断。

---

## 5. Phase 4.15 的正式语义

这一刀只允许引入一类新语义：

> `historical hold permission` 必须和 `current degradation pressure` 重新耦合。

也就是说：

- 当前 semantic health 仍成立时，可以 retain
- 当前 semantic health 已经恶化时，历史 hold 必须 release
- release 必须优先作用在 “历史保留权” 上，而不是重新破坏晚窗 activation / authority / floor

---

## 6. 允许修改的最小层

只允许修改 `Phase 4.8` 相关的两段：

### 6.1 helper 内

只允许改：

- `_compute_bootstrap_bonus_source_hold_persistence_contract(...)`

### 6.2 call-site 的 hold state 更新

只允许改：

- `critic_contract_bootstrap_bonus_hold_state` 的更新规则

原因很简单：

- 如果只改 helper，不改 state，旧 trust 仍会持续灌入
- 如果只改 state，不改 helper，release 只会慢慢体现在未来，不会在当前退化窗口及时生效

---

## 7. Phase 4.15 的正式 release 合同

下面是建议采用的最小合同。

### 7.1 新增语义量

在 `Phase 4.8` helper 内新增：

- `source_hold_release_pressure`
- `source_hold_release_window_activation`
- `source_hold_release_gate`
- `source_hold_release_aware_state`

### 7.2 release pressure 的语义

`release pressure` 不应该重新重复惩罚所有 truth/alignment 项。

它只应该代表：

> 当前窗口里，“继续相信历史 hold” 这件事是否已经被真实退化证伪。

因此建议固定成：

- `source_hold_release_pressure = max(real_alarm, critic_contract_task_degradation)`

固定解释：

- `real_alarm` 表示 real-side 语义告警
- `task_degradation` 表示 bootstrap entry 主链已经感知到的真实退化
- 不再把 `1 - truth` / `1 - alignment` 额外并入 release pressure

原因：

- `truth/alignment` 已经在 `strict_hold_gate` 里管“当前是否还能保留”
- release pressure 只负责管“历史保留权是否该被撤销”
- 如果再次把 truth/alignment 也并进来，就会重新落回 `Phase 4.14` 那种混合语义不清的问题

### 7.3 release window 的语义

release 不允许侵入 early-safe / mid-window。

因此建议固定成：

- `source_hold_release_window_activation = clamp((critic_contract_bootstrap_late_gate - 0.9) / 0.1, 0, 1)`

固定解释：

- `late_gate <= 0.9` 时，不做 release
- 只在 fully-active late-window 里，对历史 hold 执行撤销语义

### 7.4 release gate 的语义

- `source_hold_release_gate = source_hold_release_window_activation * source_hold_release_pressure`

固定解释：

- late window 没 fully active 时，release gate 必须是 `0`
- fully-active 之后，release 强度直接跟真实退化挂钩

### 7.5 release 应先作用在 state，而不是先砍当前 semantic candidate

建议固定为：

- `source_hold_release_aware_state = critic_contract_bootstrap_bonus_hold_state * (1 - source_hold_release_gate)`

然后：

- `source_hold_late_certified_floor = max(source_hold_release_aware_state, source_value_late_window_activation * min(source_hold_truth, source_hold_alignment))`

固定解释：

- release 的第一目标，是撤销“历史记忆的继续主导权”
- 不是直接撤销当前 late-window semantic source
- 当前 semantic source 如果依然健康，仍可通过 `source_value_late_window_activation * min(truth, alignment)` 留在链上

### 7.6 persistence gate 的语义保持单层职责

建议保持为：

- `source_hold_persistence_gate = source_hold_window_activation * source_hold_late_certified_floor * strict_hold_gate`

但这里的 `source_hold_late_certified_floor` 已经换成 release-aware 版本。

固定解释：

- `strict_hold_gate` 继续代表“当前保留资格”
- `release_aware_state` 代表“历史保留权是否还有效”
- 两者职责分开，不再把 release 压力硬塞进所有项里

### 7.7 hold value 仍然只在当前两端之间插值

保持：

- `strict_hold_value = lerp(transition_bridged, reward_semantic_hold_source_value, strict_hold_gate)`
- `hold_persisted = lerp(replaced_value, strict_hold_value, source_hold_persistence_gate)`

固定解释：

- `Phase 4.15` 不引入新的 value family
- 只修 “保留与释放的权限语义”

---

## 8. hold state 更新合同也必须同步改成 release-aware

如果 helper 只对当前 batch release，但 state 继续按旧方式滞留，系统会继续把 stale trust 养回来。

因此建议把 state 更新改成：

- `source_hold_seed = max(source_transition_bridge_gate, source_value_late_window_activation)`
- `source_hold_health = min(source_value_truth, source_value_alignment) * clamp(1 - max(real_alarm, task_degradation), 0, 1)`
- `source_hold_release_pressure_mean = masked_mean(max(real_alarm, task_degradation), corridor_mask)`
- `source_hold_seed_mean = masked_mean(source_hold_seed, corridor_mask)`
- `source_hold_health_mean = masked_mean(source_hold_health, corridor_mask)`
- `source_hold_retention_mean = source_hold_seed_mean * source_hold_health_mean`
- `critic_contract_bootstrap_bonus_hold_state = min(1.0, max((1 - source_hold_release_pressure_mean) * 0.9 * prev_bonus_hold_state, source_hold_retention_mean))`

固定解释：

- 没有新 seed 时，state 只能衰减
- degradation 越高，历史 state 衰减越快
- 当前窗口里如果 semantic health 重新变好，仍允许新的 seed 把 state 再抬起来

---

## 9. 这一步成功的真正判据

`Phase 4.15` 不再以“hold 更强”为通过标准。

真正要看的是：

- `1250/1500` 不回归
- `2250/2500` 时：
  - `late_gate` 仍 fully active
  - `terminal_truth_delta_abs_mean > 0`
  - `hold` 相关指标不再像 `v231` 那样出现：
    - `hold_delta_abs` 更高
    - 但 `eval` 更低
    - 且 `real_mc_value_gap_abs_mean` 更高

也就是说，正式判据应该是：

> `hold` 的变化必须和行为收益、终局误差同向，而不是只和自身 delta 同向。

---

## 10. 新测试必须补什么

单测必须从“静态 gate 对不对”升级到“历史 hold 在真实退化下会不会及时 release”。

最少要新增这几类断言：

- `bonus_hold_state > 0`、`late_gate = 1`、`real_alarm/task_degradation` 高时：
  - `source_hold_release_gate > 0`
  - `source_hold_release_aware_state < bonus_hold_state`
- 同样场景下：
  - `source_hold_persistence_gate` 必须显著低于无 release 时的结果
  - `hold_persisted` 必须更接近 `replaced_value`
- `late_gate <= 0.9` 时：
  - `source_hold_release_window_activation == 0`
  - 不允许侵入 `1250/1500/1750`
- state 更新回归：
  - 无新 seed 且 degradation 高时，`bonus_hold_state` 必须只衰减不抬升
  - 有新 seed 且 degradation 低时，state 才允许再被抬起

---

## 11. 最终裁决

`Phase 4.15` 的目标可以固定写成一句话：

> 我们不是继续把 `hold persistence` 做得更强，而是让它在 late-window fully-active 之后，学会只保留仍被当前真实语义支持的 bonus value，并在真实退化恶化时及时释放历史上已经过时的 retained trust。

这就是下一刀的最小正确目标。
