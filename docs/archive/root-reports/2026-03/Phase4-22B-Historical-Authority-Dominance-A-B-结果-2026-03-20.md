# Phase 4.22B Historical Authority Dominance A/B 结果

日期：2026-03-20

## 0. 目标

本轮只回答一个问题：

> 在不改 `window / trigger / entry / hold path / actor / controller` 的前提下，
> 只改 `post_transition_certified_capture_source`，
> 能否先在结构上打出 `capture_source > current_authority`，
> 并判断这是不是可复现结果。

本轮 run：

- `v243_phase422b_post_transition_historical_authority_dominance`
- `v244_phase422b_historical_authority_dominance_rerun`

对照基线：

- `v242_phase422_post_transition_certified_capture_source_replacement`

---

## 1. 实验口径

三条 run 保持同一口径：

- `seed = 42`
- `steps = 80000`
- `eval_interval = 250`
- `eval_episodes = 15`
- 相同训练入口
- 相同 overrides bundle

唯一变量：

- `Phase 4.22B` 将 `capture source` 从“当前 authority 镜像 + support 硬上限”
  改为“release-debiased historical authority + current authority 作为下界 + support 作为允许 lift 的上限”

---

## 2. 行为对账

| step | `v242` | `v243` | `v244` |
| --- | ---: | ---: | ---: |
| `250` | `46.7` | `46.7` | `46.7` |
| `500` | `218.1` | `218.1` | `218.1` |
| `750` | `62.1` | `62.1` | `62.1` |
| `1000` | `31.1` | `31.1` | `31.1` |
| `1250` | `203.9` | `203.9` | `203.9` |
| `1500` | `92.2` | `92.2` | `92.2` |
| `1750` | `51.9` | `51.9` | `51.9` |
| `2000` | `264.3` | `264.3` | `264.3` |
| `2250` | `25.0` | `25.0` | `25.0` |
| `2500` | `22.5` | `22.5` | `22.5` |

正式结论：

> `v243` 与 `v244` 在行为上都与 `v242` 逐点同型；
> 本轮改动没有带来可见的行为增益，也没有带来行为回归。

---

## 3. 结构对账

### 3.1 `v243`

| step | current authority | historical authority | capture support | capture source | raw gate | floored gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1500` | `0.1798` | `0.2501` | `0.1798` | `0.2501` | `0.2278` | `0.2263` |
| `2000` | `0.0762` | `0.1775` | `0.0762` | `0.1524` | `0.0762020424` | `0.0762020528` |
| `2500` | `0.0531` | `0.1179` | `0.0531` | `0.1062` | `0.0530956499` | `0.0530956542` |

### 3.2 `v244`

| step | current authority | historical authority | capture support | capture source | raw gate | floored gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1500` | `0.1798` | `0.2501` | `0.1798` | `0.2501` | `0.2278` | `0.2263` |
| `2000` | `0.0762` | `0.1775` | `0.0762` | `0.1524` | `0.0762020424` | `0.0762020528` |
| `2500` | `0.0531` | `0.1179` | `0.0531` | `0.1062` | `0.0530956499` | `0.0530956542` |

正式结论：

1. `capture_source > current_authority` 已被连续两次复现。
2. `persistence_gate_floored > raw gate` 已被连续两次复现。
3. 因此 `capture source` 这一层已经不再只是单次样本。

---

## 4. 结构结论与行为结论如何同时成立

当前最准确的解释是：

> `capture source` 本体已经被打穿，
> 但它向下游 `hold_state_floored / persistence_gate_floored` 的耦合过弱，
> 以至于这份结构增益没有转成行为增益。

换句话说：

- 本轮不再支持“capture source 还没改对”
- 本轮支持“capture source 已改对，但 floor downstream coupling 太弱”

---

## 5. 正式裁决

本轮 A/B 的正式裁决固定为：

> `Phase 4.22B` 已经用 `v243 + v244` 两次同口径 run 证明：
> `post_transition certified capture source` 可以稳定高于 `current authority`，
> 且 `floored persistence gate` 也可以稳定高于原始 gate。
>
> 但行为轨迹仍与 `v242` 完全同型，
> 因此当前主问题已经不再位于 `capture source` 本体，
> 而是允许进入下一层：
> **`floor-to-gate coupling`**

---

## 6. 下一步唯一允许动作

下一步唯一允许推进的层固定为：

> 只改 `hold_state_floored / persistence_gate_floored` 的 downstream coupling，
> 不再回头改 `capture source` 本体，
> 也不允许回到 `WM core / imagined geometry / corridor / entry / actor / controller`。

当前结果文档配套引用：

- [`v207-v242全量实验对账表与根因分层矩阵-2026-03-20.md`](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/v207-v242全量实验对账表与根因分层矩阵-2026-03-20.md)
- [`Phase4-最小主线与唯一允许A-B-2026-03-20.md`](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-最小主线与唯一允许A-B-2026-03-20.md)
- [`Phase4-决策版摘要-2026-03-20.md`](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-决策版摘要-2026-03-20.md)
