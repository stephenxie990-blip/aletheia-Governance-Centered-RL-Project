# CartPole Pure-Imag `v160` 高价值语义锚 sweep 真实训练结论

## 运行对象

- 主 run: `outputs/exp_seed42_v160_h15_multih_sc_highvalue_semantic_anchor_strong_2500_20260315_01`
- 类型: 真实训练 run
- 目标: 验证“只直接打 self-generated imagined branch 的高价值语义漂移”后，pure-imag CartPole 能否减少后段回撤
- 固定基线:
  - `horizon=15`
  - 不新增 controller rescue
  - 保留 replay-value anchor
  - 保留多时域 feature-level `SC`
- 本轮 sweep 上下文:
  - `v158` 首版高价值 gating 因为直接用 replay short-return 选高价值，mask 在 CartPole 上几乎全饱和，提前中止
  - `v159` 改成 teacher-aligned score 后恢复选择性，但强度偏弱，跑到 `1000` 基本仍沿着 `v157` 轨迹前进，因此中止
  - `v160` 是这组思路的强版本验证
- `v160` 关键改动:
  - `adaptive_imag_target_value_consistency_weight=0.15`
  - `adaptive_imag_target_value_consistency_high_value_boost=4.0`
  - `adaptive_imag_target_value_consistency_high_value_quantile=0.5`
  - `adaptive_imag_target_value_consistency_high_value_feature_scale=0.75`

## 真实评估轨迹

| eval step | mean |
| --- | ---: |
| 250 | 9.2 |
| 500 | 10.4 |
| 750 | 9.2 |
| 1000 | 24.6 |
| 1250 | 37.6 |
| 1500 | 140.2 |
| 1750 | 72.6 |
| 2000 | 65.2 |
| 2250 | 50.0 |
| 2500 | 14.4 |

补充说明:

- 本轮最好成绩是 `140.2 @ 1500`。
- 训练末尾实时成绩是 `14.4 @ 2500`，因此这条 run 没有修复后段回撤。
- `summary.json` 中的 best/final 仍会受到 best checkpoint promotion 影响，run 是否稳定仍要以 `eval_history.jsonl` 末尾为准。

## 直接结论

这轮 sweep 给出的结论很明确:

1. 高价值 imagined state 语义约束这条路打中了主问题，不是空补丁。
2. `v159` 过弱时几乎改不动轨迹，`v160` 加强后立刻出现明显不同的训练相位，说明它确实碰到了 self-generated imagined branch 本体。
3. 但 `v160` 不是稳定修复，它表现为“中段能抬升，后段又振荡回落”，最终掉到 `14.4`。
4. 更关键的新发现是: 到尾段时不只是 imagined 分支没完全修好，连 teacher 分支自己的 replay 对齐也被拖松了。

一句话压缩:

**这一刀打中了 imagined drift，但当前力道过硬、覆盖过宽，最后把局部修复做成了振荡。**

## 关键证据

### 1. 它确实先压住了 imagined drift 的前期爆发

`v160` 前中段的 imagined gap 很明显不是 `v157` 那种早早失控:

| train step | teacher_to_real_gap | imag_to_real_gap | teacher_imag_gap | semantic_loss | online_adv |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.723 | 0.798 | 0.639 | 0.492 | 3.255 |
| 250 | 0.626 | 1.138 | 1.151 | 0.581 | 1.978 |
| 500 | 0.540 | 1.099 | 1.065 | 0.496 | 3.478 |
| 750 | 0.566 | 8.700 | 8.599 | 4.195 | -0.040 |

这里最重要的不是分数，而是形态:

- 到 `500` 为止，`imag_to_real_gap` 仍只在 `~1.1` 左右，说明强选择性语义锚的确先把 imagined branch 压在了真实语义附近。
- 这和 `v159` 那种“选择性已经对了，但整体轨迹几乎没改”是不同的。
- 因此可以确认: **高价值定向约束不是错方向，它能真实改变 imagined branch。**

### 2. 但它不是平滑收敛，而是进入了“局部修复窗口反复开关”的振荡形态

后半段内部信号非常像“有时贴近，有时失锁”:

| train step | teacher_to_real_gap | imag_to_real_gap | semantic_loss | online_adv | open_loop_value_gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.570 | 17.693 | 8.754 | -1.956 | 6.774 |
| 1250 | 0.644 | 20.606 | 10.289 | -0.417 | 11.291 |
| 1500 | 0.644 | 18.971 | 9.461 | -3.859 | 12.084 |
| 1750 | 0.616 | 16.702 | 8.310 | -2.934 | 12.549 |
| 2000 | 0.988 | 14.518 | 7.430 | -1.486 | 4.574 |
| 2250 | 1.915 | 13.911 | 7.561 | -1.015 | 1.499 |
| 2500 | 2.344 | 16.232 | 8.954 | -0.373 | 4.380 |

这张表说明了三件事:

1. `imag_to_real_gap` 在 `1000` 以后仍然很大，但从 `20+` 慢慢降到 `14~16`，说明它不是完全没拉动 imagined branch，而是在慢速回拉。
2. 分数抬升不是因为 imagined manifold 已经整体恢复，而更像是系统间歇性进入了几个 `online_adv` 更接近 0 的“局部可工作窗口”。
3. 到尾段 `teacher_to_real_gap` 自己也从 `~0.6` 上升到 `2.344`，说明当前强度已经开始反向拖坏 teacher 端的 replay 语义锚。

第三点最重要，因为它把 `v160` 的失败性质说清了:

- 不是“高价值语义约束没打中 imagined drift”
- 而是“它打中了，但打得过重，最后连 teacher 端都被带偏”

### 3. high-value mask 虽然不再全饱和，但仍然偏宽

`v160` 的 `high_value_fraction_mean` 大多数时间在 `0.667`，有时到 `0.833`；
对应的 `high_value_weight_mean` 大多在 `3.667`，有时到 `4.333`。

这意味着:

- `v158` 那种 `1.0` 全饱和 bug 已经修掉了
- 但 `v160` 仍不是一个“很窄的高价值 corridor 修复器”
- 它更接近“半数以上 imagined 位置都被加重拉拽”

这正好解释了为什么它会出现下面这种形态:

- 前期比 `v159` 有力得多，能明显改轨
- 中段能冲高
- 后段又把 teacher 端一起拖松，最终回撤

## 与 `v159` / `v157` 的对比

| run | best eval | final realtime eval | 结论 |
| --- | ---: | ---: | --- |
| `v157` | 290.8 | 75.2 | imagined drift 被延后，但高价值阶段仍会扩张 |
| `v159` | 25.6 | 16.2 at 1000 stop | 选择性恢复了，但力度太弱，几乎改不动轨迹 |
| `v160` | 140.2 | 14.4 | 方向打中了 imagined drift，但强度过大，后段振荡并拖坏 teacher 锚 |

因此这轮 sweep 的工程判断不是“这条路失败了”，而是:

1. `v159` 告诉我们不能太轻，不然只是看起来更优雅，实际上不生效。
2. `v160` 告诉我们也不能太重，不然 imagined branch 会被硬拉成振荡，甚至把 teacher 端也拖偏。
3. 下一条主线应该是这两者之间的中等强度版本。

## 下一步工程顺序

下一刀不该回 controller，也不该继续只加 teacher/critic 真实锚。

最合理的顺序是:

1. 保留 teacher-score 驱动的高价值 gating，因为它已经证明能真正碰到 imagined branch。
2. 把高价值约束从 `v160` 的强版本往回收一档，重点是“缩窄覆盖、降低硬拉强度”。
3. 下一条优先验证的参数方向:
   - `adaptive_imag_target_value_consistency_weight=0.12`
   - `adaptive_imag_target_value_consistency_high_value_boost=2.5`
   - `adaptive_imag_target_value_consistency_high_value_quantile=0.67`
   - `adaptive_imag_target_value_consistency_high_value_feature_scale=0.4`

这组参数的目标不是再去赌一次中段冲高，而是:

- 保留 `v160` 早期对 imagined drift 的压制
- 避免 `high_value_fraction` 过宽
- 尽量把 `teacher_to_real_gap` 稳在 `~0.6` 附近，不要在尾段被一起拖坏

## 本轮最终结论

`v160` 最有价值的地方不是分数本身，而是把“下一刀该怎么砍”压得更实了:

1. 高价值 imagined 语义约束是有效方向，不需要退回下游 controller。
2. 当前失败不是“约束无效”，而是“约束太硬、覆盖太宽”。
3. 今后必须同时监控 `imag_to_real_gap` 和 `teacher_to_real_gap`；如果前者下降但后者被拖高，说明修复已经开始伤到 teacher 锚。
4. 下一条应做中强度、窄覆盖版本，而不是再做更强版本。
