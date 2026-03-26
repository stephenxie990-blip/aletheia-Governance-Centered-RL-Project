# CartPole Pure-Imag `v155` 语义对齐修复运行结论

## 运行对象

- run: `outputs/exp_seed42_v155_h15_light_sc_early_slowcritic_semalign_2500_20260315_01`
- 类型: 真实训练 run
- 目标: 验证“只做上游对齐、不加新的 controller rescue”后，pure-imag CartPole 是否还能在后段从高分重新坠落
- 固定基线:
  - `horizon=15`
  - light `SC`
  - early slow critic
  - 不打开新的 late-trigger / persistence rescue controller
- 本轮新增上游修复:
  - `lambda_value_real_anchor=0.2`
  - `adaptive_imag_global_target_base_blend=0.35`
  - `adaptive_imag_target_value_consistency_weight=0.2`
  - `adaptive_imag_target_value_consistency_horizon=3`
  - `adaptive_imag_target_value_consistency_delta=1.0`

## 这条 run 回答了什么

这条 run 主要回答三个问题：

1. 只靠 critic 语义对齐，能不能不用 controller 就把后段坠落修掉。
2. 这组上游修复到底是“把学习压死了”，还是“方向对但强度和时机还不够”。
3. `summary.json` 里显示的最终成绩，和真实训练末尾的成绩是不是同一个概念。

## 真实评估轨迹

| eval step | mean |
| --- | ---: |
| 250 | 11.2 |
| 500 | 13.2 |
| 750 | 16.8 |
| 1000 | 33.8 |
| 1250 | 145.0 |
| 1500 | 43.8 |
| 1750 | 350.2 |
| 2000 | 244.6 |
| 2250 | 24.6 |
| 2500 | 30.8 |

补充说明：

- 这条 run 从来没有达到 `eval mean = 500.0`，因此它**没有完成“solved 后是否还会回撤”的正向验证**。
- 它在 `1750` 达到本轮最好成绩 `350.2`，随后在 `2250` 和 `2500` 出现明显 late-stage collapse。
- 训练结束后框架把 `best.pt` 提升成最终产物，因此 `summary.json` 中的 `final_current=350.2` 实际上对应**最佳检查点**，不是训练末尾的实时成绩。

## 直接结论

这次修复**没有通过**主目标验证。

更准确地说：

1. 它不是“完全没效果”。
2. 但它也没有把问题修好。
3. 它证明了“上游 value semantics 对齐”这个方向是对的。
4. 但当前这组实现与强度，仍然挡不住后段 imagined value semantics 再次失稳。

如果只用一句话概括：

**`v155` 说明我们找对了矛盾方向，但这版修复还不足以把 pure-imag CartPole 稳定留在高分 corridor 内。**

## 为什么说它不是单纯“把学习压死”

如果这组 patch 只是把模型整体压死，那么这条 run 不应该在中后段出现明显恢复。

但 `v155` 实际上发生了下面这件事：

- `1500 -> 43.8`
- `1750 -> 350.2`

这说明：

1. 语义对齐 patch 确实在某个阶段把 imagined distribution 拉回过较好的区域。
2. 它不是“从头到尾只会保守、不会学”。
3. 所以当前失败更像“恢复能力有了，但持续锚定不够”。

## 关键内部信号

| step | online_adv | value_gap_abs | trust | freshness | semantic_loss | real_anchor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1250 | -3.52 | 3.74 | 0.676 | 0.688 | 2.74 | 0.311 |
| 1500 | -3.63 | 3.87 | 0.677 | 0.690 | 5.88 | 0.213 |
| 1750 | -1.41 | 2.02 | 0.790 | 0.827 | 5.53 | 0.116 |
| 2000 | -2.16 | 2.45 | 0.766 | 0.770 | 14.18 | 0.072 |
| 2250 | -7.17 | 7.22 | 0.436 | 0.456 | 8.42 | 0.205 |
| 2500 | -7.16 | 7.47 | 0.400 | 0.445 | 16.03 | 0.032 |

这张表很关键，因为它把 run 的形态讲清楚了：

### `1750` 为什么能冲到 `350.2`

在这一段：

- `online_adv` 从更负的区间收回到 `-1.41`
- `value_gap_abs` 压到 `2.02`
- `trust / freshness` 一起抬升到 `0.79 / 0.83`

也就是说，真实评估抬起来时，内部信号确实在往“critic 对 imagined latent 重新更校准”这个方向走。

### `2250 -> 2500` 为什么会再次塌掉

到后段：

- `online_adv` 重新掉到 `-7` 左右
- `value_gap_abs` 重新拉大到 `7+`
- `trust / freshness` 一起跌破 `0.46`
- `semantic_consistency_loss` 再次变大

这说明：

1. world model + critic 的语义对齐不是一直稳定维持住的；
2. actor 后面又重新吃到了明显偏负的 imagined advantage；
3. 最终真实 eval 再次离开高分 corridor。

## 一个非常重要的事实：controller 全程没接管

本轮还有一个很干净的实验性质：

- `actor/controller_stage` 全程为 `idle`
- post-entry / persistence / post-solved 等阶段都没有接管

这意味着：

1. 这条 run 不是靠新 controller 在后面“托住”；
2. 它测到的就是上游修复本身的能力边界；
3. 因此可以更干净地判断：
   - 上游方向正确；
   - 但这版上游修复本身还不够。

## 与 `v154` 的对比

`v154` 的关键轨迹是：

- `1250 -> 500.0`
- `1500 -> 500.0`
- `1750 -> 500.0`
- `2000 -> 500.0`
- `2250 -> 88.2`
- `2500 -> 231.6`

`v155` 的关键轨迹是：

- `1250 -> 145.0`
- `1500 -> 43.8`
- `1750 -> 350.2`
- `2000 -> 244.6`
- `2250 -> 24.6`
- `2500 -> 30.8`

这组对比说明：

1. `v155` 没有把 `v154` 的“先 solved，再 late-stage 回撤”问题修掉；
2. 更严格地说，`v155` 连 solved 区都没进去；
3. 它的优点是：
   - 在 `1500 -> 1750` 出现了一次明显的语义修复回升；
4. 它的缺点是：
   - 后段回撤更深；
   - 整体 run-level 表现比 `v154` 更差。

因此当前不能把 `v155` 视为 mainline 候选。

## 对“优雅上游修复”这条思路的最新判断

这条 run 没有否定“上游修复优先”的原则，反而把它讲得更清楚了：

1. 方向是对的。
2. 但“软语义惩罚 + 全局 target blend + real anchor”这一版组合，还不够形成持续稳定的上游锚。

从工程上看，当前更像是：

- **恢复能力有了**
- **保持能力不够**

也就是说，它能把模型短时拉回较好的 imagined value geometry，但还不能长期把它锁在那附近。

## 本轮最终结论

`v155` 的最终结论可以压缩成四句话：

1. 这条 run 没有 solved，最好只到 `350.2 @ 1750`。
2. 它在 `2250 -> 24.6`、`2500 -> 30.8` 发生了明显 late-stage collapse。
3. 所以上游语义对齐 patch **还不是最终修复方案**。
4. 但 run 内部信号强烈支持我们之前的根因判断：
   - 关键矛盾仍然是 imagined latent 上的 value semantics 稳定性，
   - 而不是 reward / continue 头先坏。
