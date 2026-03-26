# Phase 4.21：Post-Transition Certified Retention Floor / Authority Strengthening

> 日期：2026-03-20
>
> 审查基线：
> - 裁决文档：
>   [Phase4-19-20-当前根因裁决-实验口径污染与Always-On-Bootstrap-Authority问题-2026-03-20.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-19-20-当前根因裁决-实验口径污染与Always-On-Bootstrap-Authority问题-2026-03-20.md)
> - 关键 run：
>   - `v237_phase418_hold_state_channel_expansion`
>   - `v238_phase419_hold_channel_dynamics_calibration`
>   - `v239_phase419_fixed`
>   - `v240_phase420_trigger_optimization`

## 0. 审查结论

对上一份裁决文档复核后，主结论成立，可以作为后续唯一合法刀口的依据。

确认成立的部分：

- `v238` 的确属于实验口径污染，不能拿来判定 `Phase 4.19` 本体失败。
- `v240` 的确把 bootstrap availability 修成了更常开。
- `v240` 的真正塌陷位置也的确不是 `trigger_gate / late_gate`，而是 `post_transition hold` 与 `hold persistence gate`。
- 当前主问题已经不再是 corridor 是否存在，也不再是 entry 是否能进。

需要收紧但不推翻主裁决的一句话是：

> `v239` 只能证明 `Phase 4.19` 的 3 通道 hold channel 与 `0.55 / 0.80` 边界具有结构可行性，
> 不能单凭一次 run 就写成 “已经证明该方向正确”。

因此，本计划正式采用上一份裁决文档的主因果链，并把下一刀严格收缩到：

> `post-transition late-window` 中，bootstrap certified retention floor / authority strength

---

## 1. 当前真正问题的因果链

当前已经确认的因果链固定写成：

1. `v238` 的异常主要来自 `resolved config` 漂移，而不是 `Phase 4.19` 本体公式天然错误。
2. `v239` 说明 `Phase 4.19` 的 channel split 并未天然压死中段能力，`1250 = 251.9` 已经足以说明 late-window 之前的 channel 划分至少具备合理性。
3. `v240` 说明把 `eval_threshold` 降到 `75.0`、把 `step_ramp` 提到 `500` 后，`late_gate / trigger_gate` 已经基本常开。
4. 但 `v240` 同时显示：
   - `1500`: `post_hold ≈ 0.1798`, `persistence_gate ≈ 0.2263`
   - `2000`: `post_hold ≈ 0.0762`, `persistence_gate ≈ 0.0633`
   - `2500`: `post_hold ≈ 0.0531`, `persistence_gate ≈ 0.0212`
5. 也就是说，在 `late_gate = 1.0`、`trigger_gate = 1.0`、`entry_deg` 持续较高的情况下，真正负责保留晚窗 bonus authority 的 `post_transition` 路径还在被慢慢抽空。

正式根因固定为：

> 当前主故障不是 “进不去”，而是 “进去以后没有足够强的 post-transition certified retention floor”。
>
> 现在的 bootstrap 更像是 availability 被修成了 always-on，
> 但 authority strength 还没有被修成 certified and durable。

---

## 2. Phase 4.21 的唯一目标

`Phase 4.21` 只修一件事：

> 在不再修改 corridor / entry / threshold / actor / controller / floor / authority 主执行链的前提下，
> 为 `post-transition late-window` 建立一个更稳定的 certified retention floor，
> 防止 `post_transition hold state` 与 `source_hold_persistence_gate` 在 `2000-2500` 的高 degradation 区间被抽到接近 `0`。

这一刀的正确方向不是：

- 再把 gate 放得更开
- 再降低 eval threshold
- 再增强 corridor
- 再 sharpen trigger entry

而是：

- 在 gate 已经打开之后
- 给 post-transition retention 一个独立的、可持续的 authority floor

---

## 3. 冻结边界

本阶段默认冻结，不允许再动：

- actor unified contract
- controller stage
- `_compute_task_cert_gate`
- `Phase 4.2` floor overlay
- `Phase 4.3` recert
- `Phase 4.5` first-drop safe activation
- `Phase 4.7` transition bridge
- `Phase 4.9 / 4.10` entry source / external feedback resolver
- `Phase 4.11 / 4.12 / 4.13` terminal source / activation / consumer family
- `adaptive_imag_critic_bootstrap_eval_threshold`
- `adaptive_imag_critic_bootstrap_step_ramp`
- corridor 开关与 corridor 公式本体
- public CLI
- 训练入口参数
- run naming 约定

本阶段唯一允许动的层：

> `Phase 4.8 hold persistence` 内部的
> `post-transition certified retention floor / authority-strength coupling`

---

## 4. 实现方案

### 4.1 新增一个只负责 post-transition authority floor 的 helper

在 [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py) 新增一个纯标量 / 纯张量 helper，命名固定为：

`_compute_post_transition_certified_retention_floor_contract(...)`

它的职责固定为：

- 不改 entry
- 不改 corridor
- 不改 late_gate
- 不改 trigger_gate
- 只在 `post-transition` 区间内，给 hold-state 与 persistence-gate 提供一个最小 certified floor

### 4.2 helper 输入固定为

- `critic_contract_bootstrap_late_gate`
- `critic_contract_bootstrap_trigger_gate`
- `critic_contract_bootstrap_hold_state_active_transition`
- `critic_contract_bootstrap_hold_state_post_transition`
- `source_hold_seed_mean`
- `source_hold_health_mean`
- `source_hold_retention_mean`
- `source_hold_release_pressure_mean`
- `critic_contract_bootstrap_source_hold_persistence_gate_mean`

### 4.3 helper 输出固定为

- `post_transition_window_activation`
- `post_transition_certified_capture`
- `post_transition_certified_floor_candidate`
- `post_transition_certified_floor_state`
- `post_transition_hold_state_floored`
- `post_transition_persistence_gate_floored`

### 4.4 helper 公式语义固定为

合同固定为：

- `post_transition_window_activation = clamp((critic_contract_bootstrap_late_gate - 0.80) / 0.20, 0, 1)`
- `post_transition_certified_capture = post_transition_window_activation * max(critic_contract_bootstrap_hold_state_post_transition, 0.5 * critic_contract_bootstrap_hold_state_active_transition, source_hold_retention_mean)`
- `post_transition_certified_floor_candidate = max(0.90 * prev_post_transition_certified_floor_state, post_transition_certified_capture)`
- `post_transition_certified_floor_state = clamp(critic_contract_bootstrap_trigger_gate * post_transition_certified_floor_candidate, 0, 1)`
- `post_transition_hold_state_floored = max(critic_contract_bootstrap_hold_state_post_transition, post_transition_certified_floor_state)`
- `post_transition_persistence_gate_floored = max(critic_contract_bootstrap_source_hold_persistence_gate_mean, 0.5 * post_transition_certified_floor_state)`

固定语义解释：

- `0.80` 以下完全中性，不侵入 `pre / active_transition`
- floor 只能在 `post-transition` 区间工作
- floor 的来源只能是已经被建立过的 late-window authority，不允许平地无中生有
- floor 可以慢衰减，但不允许像 `v240` 那样在 gate 常开时一路掉到接近 `0`
- floor 仍然受 `trigger_gate` 约束；若 bootstrap availability 真正关闭，floor 也必须释放

### 4.5 trainer 持久状态新增一项

新增 trainer 内部持久状态，命名固定为：

- `self._adaptive_imag_critic_bootstrap_post_transition_certified_floor_state: float = 0.0`

固定语义：

- 这是 `post-transition` authority floor 的独立状态
- 它不是新的 entry gate
- 它不是新的 corridor state
- 它只服务于 `Phase 4.8 hold persistence`

### 4.6 接线顺序固定改成

当前 hold 路径基础顺序保留，只加一层 floor：

- hold seed / health / release pressure
- `post_transition_certified_retention_floor_contract`
- hold state update
- floored post-transition hold state
- floored persistence gate
- `Phase 4.8 hold persistence` value path

更具体地说：

1. 先按现有逻辑算出原始 `post_transition` hold state 与 `source_hold_persistence_gate`
2. 再通过 `post_transition_certified_floor_state` 对它们做 floor
3. 后续 value-side hold persistence 继续消费 floored 结果

固定禁止：

- 不允许把 floor 回写到 `trigger_gate`
- 不允许把 floor 重新解释成新的 entry
- 不允许把 floor 影响 corridor / task-cert 公式

---

## 5. 观测指标

新增并导出到 [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py) 与 [aletheia_api.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py)：

- `critic/critic_contract_bootstrap_post_transition_window_activation`
- `critic/critic_contract_bootstrap_post_transition_certified_capture`
- `critic/critic_contract_bootstrap_post_transition_certified_floor_state`
- `critic/critic_contract_bootstrap_post_transition_hold_state_floored`
- `critic/critic_contract_bootstrap_post_transition_persistence_gate_floored`

保留现有主判读指标：

- `critic/critic_contract_bootstrap_late_gate`
- `critic/critic_contract_bootstrap_trigger_gate`
- `critic/critic_contract_bootstrap_trigger_entry_degradation`
- `critic/critic_contract_bootstrap_hold_state_active_transition`
- `critic/critic_contract_bootstrap_hold_state_post_transition`
- `critic/critic_contract_bootstrap_source_hold_persistence_gate_mean`

---

## 6. 测试计划

### 6.1 代码层测试

只改：

- [aletheia/tests/test_hold_state_channels.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_hold_state_channels.py)
- [aletheia/tests/test_training_loop_integration.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_training_loop_integration.py)

新增或扩展这些断言：

1. `late_gate < 0.80` 时：
   - `post_transition_window_activation == 0`
   - `post_transition_certified_floor_state == 0`
   - floor 不得改写 `post_transition hold` 与 `persistence gate`

2. `late_gate == 1.0`、已有 post hold 且 trigger gate 打开时：
   - `post_transition_certified_floor_state > 0`
   - `post_transition_hold_state_floored >= raw post_transition hold`
   - `post_transition_persistence_gate_floored >= raw persistence gate`

3. 在高 degradation 下，如果 gate 仍开着：
   - `post_transition` floored 值允许缓慢下降
   - 但不允许像当前 `v240` 那样迅速掉到接近 `0`

4. `trigger_gate = 0` 时：
   - `post_transition_certified_floor_state` 必须释放
   - 不允许产生新的 authority 幻觉

5. guardrail 不回归：
   - `effective_contact_on_valid >= requested_floor_on_valid`
   - `floor_shortfall_on_valid == 0.0`
   - `authority_shortfall_on_valid == 0.0`

### 6.2 运行前实验卫生检查

这一项不属于新算法层，但属于本阶段的必做前置检查：

- 新 run 必须使用与 `v240` 同一份 resolved config 口径
- 除 `run_name / save path` 外，不允许再出现配置漂移
- 若 `resolved config` 与 `v240` 有任何额外差异，则该 run 自动判为无效

---

## 7. 严格真实 run

新 run 命名固定为：

`v241_phase421_post_transition_certified_retention_floor`

参数完全对齐 `v240`：

- `seed=42`
- `steps=80000`
- `update_steps=2500`
- `eval_interval=250`
- `save_interval=250`
- `log_interval=50`
- `eval_episodes=15`

另外固定新增一个后验评估要求：

- 对 `best checkpoint` 与 `final checkpoint` 各做一次 `100-episode audit`

这一步不是改训练协议，而是防止再次把短评估高分误判成 solved。

---

## 8. 通过标准

### 8.1 结构通过标准

必须同时满足：

- `critic/critic_contract_bootstrap_late_gate > 0.95` 于至少一个 `2000/2250/2500`
- `critic/critic_contract_bootstrap_trigger_gate == 1.0` 或接近 `1.0` 于至少一个 `2000/2250/2500`
- `critic/critic_contract_bootstrap_post_transition_certified_floor_state > 0.0` 于至少一个 `2000/2250/2500`
- `critic/critic_contract_bootstrap_post_transition_hold_state_floored >= critic/critic_contract_bootstrap_hold_state_post_transition`
- `critic/critic_contract_bootstrap_post_transition_persistence_gate_floored >= critic/critic_contract_bootstrap_source_hold_persistence_gate_mean`
- 不允许再次出现 `v240` 这种形态：
  - `late_gate = 1.0`
  - `trigger_gate = 1.0`
  - 但 `post_transition hold` 与 `persistence_gate` 一路掉到接近 `0`

### 8.2 行为通过标准

这一刀的主目标不是刷新全局最高分，而是修复晚窗 authority 崩塌。

主行为标准固定为：

- `eval@1250 >= 200.0`
- `eval@2000 >= 200.0`
- `eval@2250 > v240@2250 = 25.0`
- `eval@2500 > v240@2500 = 22.5`

更严格的理想目标：

- `eval@2250 >= 100.0`
- `eval@2500 >= 100.0`

后验 `100-episode audit` 目标：

- 若 `best checkpoint` 的 `15-episode eval` 很高，但 `100-episode audit` 崩塌，则该 run 不得被判 solved

---

## 9. 失败分流

若失败，固定按下面口径分流：

1. 若 `post_transition_certified_floor_state` 始终接近 `0`：
   - 说明 capture 条件过严或接线错误
   - 下一步仍留在同一层，只修 capture / floor 建立条件

2. 若 floor state 已建立，但 `persistence_gate` 仍迅速掉空：
   - 说明 gate-floor coupling 过弱
   - 下一步仍留在同一层，只修 authority-strength coupling

3. 若 floor 与 gate 都更稳，但行为仍然塌：
   - 说明真正问题已经不是 authority floor，而是 retained value base 本身不对
   - 到那时才允许重新审视 terminal-truth candidate family

4. 若 guardrail 回归：
   - 优先回滚并修执行链
   - 不允许直接跳去下一层

---

## 10. 最终执行口令

本计划的正式执行约束固定为：

> 只动 `post-transition certified retention floor / authority strength`，
> 不再把问题重新打成 entry、corridor、threshold 或 actor/controller 的局部 tradeoff。

