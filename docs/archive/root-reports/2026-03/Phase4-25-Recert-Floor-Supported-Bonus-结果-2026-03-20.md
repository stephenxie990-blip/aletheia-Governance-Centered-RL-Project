# Phase 4.25 Recert Floor-Supported Bonus 结果

日期：2026-03-20

## 0. 目标

本轮只回答一个问题：

> 在 `capture source / floor / gate / consumer` 都已被证明“结构可改、但行为不动”之后，
> 如果只继续推进 `postrecert authority consumption`，
> 给 `source_recert_gate` 一个由 `floor_authority` 支撑的 bonus floor，
> 能否第一次把 `v242` 的晚期坍塌轨道打穿。

本轮 run：

- `v248_phase425_recert_floor_supported_bonus`

对照基线：

- `v242_phase422_post_transition_certified_capture_source_replacement`
- `v247_phase424_consumer_strengthening_lift`

---

## 1. 行为对账

### 1.1 全量 eval 对比

| step | `v242` | `v247` | `v248` | `v248-v242` |
| --- | ---: | ---: | ---: | ---: |
| `250` | `46.7` | `46.7` | `46.7` | `0.0` |
| `500` | `218.1` | `218.1` | `218.1` | `0.0` |
| `750` | `62.1` | `62.1` | `62.1` | `0.0` |
| `1000` | `31.1` | `31.1` | `31.1` | `0.0` |
| `1250` | `203.9` | `203.9` | `203.9` | `0.0` |
| `1500` | `92.2` | `92.2` | `97.1` | `+4.9` |
| `1750` | `51.9` | `51.9` | `67.1` | `+15.3` |
| `2000` | `264.3` | `264.3` | `166.5` | `-97.8` |
| `2250` | `25.0` | `25.0` | `60.3` | `+35.3` |
| `2500` | `22.5` | `22.5` | `107.6` | `+85.1` |

### 1.2 行为结论

正式结论：

1. `v248` 在 `1250` 之前与 `v242/v247` 完全同型，说明上游训练轨道没有变化。
2. `v248` 从 `1500` 开始首次系统性偏离旧轨。
3. 这种偏离不是“整体抬升”，而是：
   - 放弃了 `2000` 附近的尖峰爆发；
   - 换来了 `2250/2500` 的晚期存活与终点保持。

更直白地说：

> `v248` 把系统从“中后段冲高后快速坍塌”
> 推向了“峰值降低，但 authority 保留更久、尾部不再崩穿”。

---

## 2. 晚期指标汇总

| 指标 | `v242` | `v248` | 变化 |
| --- | ---: | ---: | ---: |
| `1500-2500 avg` | `91.19` | `99.72` | `+8.53` |
| `2250-2500 tail avg` | `23.77` | `83.93` | `+60.17` |
| `final@2500` | `22.53` | `107.60` | `+85.07` |

这组数值说明：

> `Phase 4.25` 的真实贡献不在“更高峰值”，
> 而在“显著削弱 terminal collapse”。

---

## 3. 结构对账：瓶颈是否真的在 recert

### 3.1 `v247` vs `v248` 关键结构点

| step | run | hold gate | recert gate | `recert/hold` | bonus prefinal | bonus postrecert | `postrecert/prefinal` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1500` | `v247` | `0.2476` | `0.1913` | `0.7724` | `0.02659` | `0.00520` | `0.1956` |
| `1500` | `v248` | `0.2513` | `0.2331` | `0.9279` | `0.02689` | `0.00542` | `0.2014` |
| `2000` | `v247` | `0.1524` | `0.0817` | `0.5362` | `0.00249` | `0.000213` | `0.0856` |
| `2000` | `v248` | `0.2018` | `0.1704` | `0.8448` | `0.00423` | `0.000554` | `0.1311` |
| `2500` | `v247` | `0.1062` | `0.0562` | `0.5295` | `0.00129` | `0.000073` | `0.0566` |
| `2500` | `v248` | `0.1854` | `0.1286` | `0.6935` | `0.00462` | `0.000487` | `0.1055` |

### 3.2 这张表说明了什么

1. `hold gate` 和 `consumer gate` 之前已经被打上去，但行为不动。
2. 真正继续塌的是 `recert gate` 与 `postrecert bonus`。
3. `Phase 4.25` 之后：
   - `recert gate` 明显更贴近 `hold gate`
   - `prefinal bonus` 在晚期不再掉成几乎为零
   - `postrecert retention ratio` 在 `2000/2500` 都明显抬升

因此当前最准确的结构解释是：

> 旧主线真正卡住的，不是 `WM core`，
> 也不是 `capture source / floor / consumer` 本体，
> 而是 `authority bonus` 在 `postrecert` 处被进一步压扁。

---

## 4. 为什么这不支持“世界模型精度问题”

如果根因是 `WM core` 精度不够，理论上更应该看到：

1. 从前中期开始整体轨道漂移；
2. 上游结构修改难以稳定复现；
3. 单改下游 `authority consumption` 不应显著改变晚期行为。

但真实结果恰好相反：

1. `250-1250` 完全同型，说明上游学习态与任务理解没有发生变化。
2. `capture_source > current_authority`、`floor -> gate` 等结构改动都已被连续 run 复现。
3. 只有当我们继续推进到 `postrecert` 这一层时，行为才第一次真正脱离旧轨。

正式裁决：

> 截至 `v248`，证据不支持“当前主问题是世界模型精度不足”。
>
> 当前证据支持：
> 主瓶颈位于 `downstream authority consumption`，
> 且更具体地位于 `postrecert suppression`。

---

## 5. 当前根因裁决

截至 `Phase 4.25`，最稳的根因描述更新为：

> `WM core` 不是主瓶颈；
> `capture source / floor / consumer` 是必要但不充分条件；
> 真正导致 `v242` 晚期坍塌的关键层，是 `authority bonus` 在 `recertification` 之后仍被过度抑制。

`v248` 已经第一次证明：

> 只动这一个层，就足以把晚期坍塌轨迹打穿。

但也要保留一个代价判断：

> 当前是“耐久性增强、尖峰下降”而不是“全段统治性提升”。

---

## 6. 下一步约束

基于这轮结果，下一步仍然只能留在同一主线：

1. 只允许继续做 `postrecert` 及其相邻的 authority consumption 精修。
2. 不允许回头重开：
   - `WM core`
   - imagined geometry
   - corridor
   - trigger / threshold / ramp / entry
   - hold path
   - actor
   - controller
3. 判断标准不再是“能否把结构打上去”，而是：
   - 能否在保住 `2250/2500` 的前提下，回收 `2000` 的峰值损失。

---

## 7. 配套引用

- [`v207-v242全量实验对账表与根因分层矩阵-2026-03-20.md`](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/v207-v242全量实验对账表与根因分层矩阵-2026-03-20.md)
- [`Phase4-最小主线与唯一允许A-B-2026-03-20.md`](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-最小主线与唯一允许A-B-2026-03-20.md)
- [`Phase4-决策版摘要-2026-03-20.md`](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-决策版摘要-2026-03-20.md)
- [`Phase4-22B-Historical-Authority-Dominance-A-B-结果-2026-03-20.md`](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-22B-Historical-Authority-Dominance-A-B-结果-2026-03-20.md)
