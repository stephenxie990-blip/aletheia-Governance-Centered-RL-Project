# CartPole Pure-Imag `v156` 回放锚定 value calibration 真实训练结论

## 运行对象

- run: `outputs/exp_seed42_v156_h15_light_sc_replay_value_anchor_2500_20260315_01`
- 类型: 真实训练 run
- 目标: 验证“去掉全局 target blend，只保留 replay-anchored short-return calibration”后，pure-imag CartPole 能否在不依赖 controller rescue 的前提下避免后段回撤
- 固定基线:
  - `horizon=15`
  - light `SC`
  - 不打开新的 controller rescue
- 本轮关键改动:
  - 保留 `lambda_value_real_anchor=0.2`
  - 把 `adaptive_imag_global_target_base_blend` 降为 `0.0`
  - 把 `target value consistency` 改成“用 replay 短回报同时约束 teacher 分支和 imagined 分支”
  - `adaptive_imag_target_value_consistency_weight=0.1`

## 真实评估轨迹

| eval step | mean |
| --- | ---: |
| 250 | 25.8 |
| 500 | 35.0 |
| 750 | 17.4 |
| 1000 | 23.4 |
| 1250 | 33.8 |
| 1500 | 39.2 |
| 1750 | 81.0 |
| 2000 | 28.8 |
| 2250 | 15.6 |
| 2500 | 17.0 |

补充说明:

- 这条 run 没有达到 `eval mean = 500.0`，因此没有完成“solved 后是否还会从 500 掉回去”的正向验证。
- 本轮最好成绩是 `81.0 @ 1750`，之后一路回撤到 `17.0 @ 2500`。
- 训练结束时框架继续把 `best.pt` 提升成最终产物，因此 `summary.json` 里的 `final_current=81.0` 对应的是最佳检查点，不是训练末尾实时成绩。

## 直接结论

这次修复没有通过主目标验证，而且 run-level 表现比 `v155` 和 `v154` 都更差。

但它带来了一条更尖锐、更靠前的根因结论：

1. replay anchor 的确把 **teacher-forced 分支** 校准得更接近 replay 短回报。
2. 但真正给 actor 用的 **self-generated imagined 分支** 仍然长期停留在远离 replay 语义锚的位置。
3. 所以当前最本质的问题已经可以收敛成一句话：

**不是 critic 完全看不见真实值，而是 self-generated imagined rollout 本体仍然会逃出 replay-aligned 的 value manifold。**

## 关键证据

### 关键训练步位

| train step | online_adv | teacher_to_real_gap | imag_to_real_gap | teacher_imag_gap | trust | freshness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1150 | -2.88 | 0.60 | 22.20 | 22.14 | 0.712 | 0.733 |
| 1450 | -1.43 | 0.56 | 19.46 | 19.44 | 0.786 | 0.832 |
| 1750 | 0.43 | 0.52 | 16.36 | 16.36 | 0.899 | 0.946 |
| 2300 | -3.50 | 0.54 | 16.49 | 16.42 | 0.649 | 0.672 |
| 2450 | -1.14 | 0.53 | 16.18 | 16.09 | 0.837 | 0.886 |
| 2500 | 0.13 | 0.58 | 15.97 | 15.81 | 0.909 | 0.947 |

这张表直接说明了三件事：

1. `teacher_to_real_gap` 全程大致稳定在 `0.5~0.6`，说明 replay anchor 对 teacher 分支是有效的。
2. `imag_to_real_gap` 却长期卡在 `15~22` 的高位，没有被一起拉回去。
3. 即使在 `1750` 这类相对较好的区间，imagined 分支也没有真正对齐 replay 语义，只是 actor 暂时在一个可工作的窄窗口里拿到了还算能用的梯度。

### 一个非常关键的新现象

这轮 run 还暴露出一个比 `v155` 更清楚的新事实：

- 到 `2400/2500` 时，局部的 imagined target gap、短期开环 audit，甚至 `online_adv` 都可以看起来不算差；
- 但 `imag_to_real_gap` 仍然维持在 `16` 左右；
- 同时真实 eval 已经掉到 `17.0`。

这说明：

1. “当前 imagined batch 上的局部目标看起来还行”，不等于“self-generated rollout 已经回到真实可控的语义区域”。
2. 现在的短期 audit 更像是在看局部平滑性；
3. 但真正把策略带偏的，是 imagined 分支在 replay 锚定语义坐标系里的长期偏移。

换句话说：

**局部顺滑，不代表全局对齐。**

## 为什么 `v156` 比 `v155` 分数更差，但诊断反而更有价值

`v155` 的结果是：

- 最好 `350.2 @ 1750`
- 末尾 `30.8 @ 2500`

`v156` 的结果是：

- 最好 `81.0 @ 1750`
- 末尾 `17.0 @ 2500`

从分数上看，`v156` 更差，这点不能粉饰。

但从诊断价值看，`v156` 比 `v155` 更干净，因为它把两个问题分开了：

1. replay short-return anchor 能不能把 teacher 分支拉回真实尺度。
2. 这件事本身能不能自动让 imagined 分支也一起回到正确流形。

答案现在已经很清楚：

- 第一个问题，答案是“能一部分”；
- 第二个问题，答案是“不能”。

所以 `v156` 虽然不是候选主线，却帮我们排除了一个常见误判：

**“只要 critic 重新看到真实值，actor 用的 imagined rollout 就会自然变好。”**

这条假设现在基本可以判定为不成立。

## 与 `v154 / v155` 的对比

| run | best eval | final eval | 主要结论 |
| --- | ---: | ---: | --- |
| `v154` | 500.0 | 231.6 | solved 后仍会回撤，第一失稳点更像 value semantics |
| `v155` | 350.2 | 30.8 | 上游语义对齐方向正确，但持续锚定不够 |
| `v156` | 81.0 | 17.0 | teacher 分支被 replay 锚住了，但 imagined 分支没有被一起锚住 |

因此：

1. `v156` 不是新的主线候选。
2. 但它让根因又往前收敛了一步。
3. 当前真正需要修的，不再是“怎么让 teacher critic 更清醒”，而是“怎么让 self-generated imagined branch 本身不要脱离 replay-aligned 语义流形”。

## 本轮最终结论

`v156` 的结论可以压缩成四句话：

1. replay-anchored value calibration 没有修好 late-stage collapse。
2. 它把 teacher 分支校准得明显比 imagined 分支好。
3. 当前主问题不再是“critic 完全失准”，而是“imagined rollout 分支本体仍然漂移”。
4. 因此下一步修复必须直接作用在 imagined 分支本身，而不是只给 teacher / critic 补更强的真实锚。
