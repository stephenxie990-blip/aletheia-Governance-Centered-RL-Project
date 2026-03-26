# CartPole Pure-Imag `v157` 多时域 feature-level SC 真实训练结论

## 运行对象

- run: `outputs/exp_seed42_v157_h15_multih_sc_feat_replay_value_anchor_2500_20260315_01`
- 类型: 真实训练 run
- 目标: 验证“直接增强 self-generated imagined branch 的多时域 feature-level 对齐”后，pure-imag CartPole 能否在不依赖 controller rescue 的前提下，减少后段回撤
- 固定基线:
  - `horizon=15`
  - 不新增 controller rescue
  - 保留 replay-value anchor
- 本轮关键改动:
  - `rssm_shortcut_consistency.horizons=[2,4,8]`
  - `rssm_shortcut_consistency.loss_scale=0.15`
  - `rssm_shortcut_consistency.feature_loss_scale=0.5`
  - `rssm_shortcut_consistency.sample_ratio=0.5`
  - `rssm_shortcut_consistency.max_starts=4`
  - 修正 `ConsistencyAuditor` 只在 `MSC` 启用时才计算 `remaining_steps`，因此本轮不再出现 `RemainingStepsComputer` 的 warning spam

## 真实评估轨迹

| eval step | mean |
| --- | ---: |
| 250 | 9.6 |
| 500 | 9.8 |
| 750 | 25.6 |
| 1000 | 16.2 |
| 1250 | 20.8 |
| 1500 | 25.0 |
| 1750 | 290.8 |
| 2000 | 98.2 |
| 2250 | 210.2 |
| 2500 | 75.2 |

补充说明:

- 本轮最好成绩是 `290.8 @ 1750`。
- 训练末尾实时成绩是 `75.2 @ 2500`，因此仍然存在明显回撤。
- `summary.json` 中的 `final_current=290.8`、`final_best=290.8` 来自 best checkpoint promotion，`final_model_source=best`；它不代表训练末尾实时轨迹。
- `post_entry_audit` 显示 controller stage 全程都是 `idle`，说明这条 run 没有靠 controller 救火。

## 直接结论

`v157` 明显优于 `v156`，但仍然没有修复主问题。

可以把结论压成四句话:

1. 多时域 feature-level `SC` 确实有用，它把 `v156` 那种过早、过猛的 imagined branch 漂移明显推迟了。
2. 这条 run 已经能多次进入高分区，最好冲到 `290.8`，说明它不再只是“低位保守学不会”。
3. 但一旦进入更高价值区域，`imag_to_real_gap` 还是会重新扩张到 `15+`，而 `teacher_to_real_gap` 仍维持在 `0.5~0.6` 左右。
4. 所以当前根因仍然是：**self-generated imagined branch 在高价值阶段会重新逃离 replay-aligned value manifold。**

换句话说：

`v157` 修掉的是“太早炸开”，没有修掉的是“冲高以后还能长期说真话”。

## 关键证据

### imagined gap 的时间线

| train step | teacher_to_real_gap | imag_to_real_gap | teacher_imag_gap | online_adv | episode_return_ema |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.72 | 0.80 | 0.63 | 2.44 | 13.50 |
| 350 | 0.60 | 0.73 | 0.36 | 3.59 | 23.44 |
| 600 | 0.59 | 3.98 | 4.02 | 2.22 | 20.25 |
| 750 | 0.58 | 4.83 | 4.83 | 0.87 | 25.76 |
| 1000 | 0.56 | 7.04 | 6.86 | 0.75 | 26.31 |
| 1250 | 0.49 | 6.10 | 6.01 | -1.40 | 27.43 |
| 1500 | 0.55 | 9.84 | 9.85 | -1.64 | 28.91 |
| 1750 | 0.53 | 16.73 | 16.74 | 0.59 | 58.68 |
| 2000 | 0.52 | 15.60 | 15.51 | -3.40 | 25.51 |
| 2250 | 0.46 | 15.16 | 15.00 | 0.40 | 30.36 |
| 2500 | 0.63 | 15.33 | 15.14 | 0.02 | 18.97 |

这张表有三层含义:

1. 到 `350` 为止，teacher 和 imagined 两个分支都还靠近 replay 语义锚，说明 feature-level `SC` 在前中段确实起作用。
2. 从 `600` 开始，imagined 分支重新先脱钩，而 teacher 分支仍然稳定，这说明剩余问题仍然只发生在 self-generated imagined branch 本体。
3. 到 `1750` 时虽然真实 eval 能冲到 `290.8`，但 imagined 分支的语义偏移已经重新扩大到 `16.7`，说明高分只是落在一个暂时可工作的窄窗口里，不是因为 imagined manifold 已被真正修正。

### run-level 轨迹说明了什么

`v157` 的真实表现不是单调崩，也不是单调好，而是：

- 前段很弱，`250/500` 还只有 `9.6 / 9.8`
- 中段缓慢抬升，但一直没进 solved 区
- 到 `1750` 突然冲到 `290.8`
- 随后又掉到 `98.2 @ 2000`
- 然后再回到 `210.2 @ 2250`
- 最终掉到 `75.2 @ 2500`

这说明它已经具备“进入高分区”的能力，但没有“在高分区维持闭环稳定”的能力。

也就是说，系统现在的真实状态是：

- 不是完全学不会
- 不是 controller 才能救
- 而是 imagined value geometry 在高分阶段仍然会松动，导致 actor 和 critic 的局部闭环反复失稳

### warning 修复是有效的，但它只是观测层修复

本轮还有一个工程侧结论值得单独记住：

- 之前日志里大量的 `RemainingStepsComputer` warning 并不是当前主失稳本体；
- 它来自 `SC-only audit` 也在无条件计算 `remaining_steps`；
- 本轮修正后 warning spam 消失，run 监控恢复清晰。

这证明观测层确实有一个边界错误被修掉了。

但训练动态没有因此自动变好到 solved，这再次说明：

**warning 修复是必要的工程清理，不是主机制修复。**

## 与 `v156` 的对比

| run | best eval | final realtime eval | gap 形态 |
| --- | ---: | ---: | --- |
| `v156` | 81.0 | 17.0 | teacher 长期对齐，imagined 从较早阶段就长期停在 `15~22` |
| `v157` | 290.8 | 75.2 | 前中段 imagined 明显更贴近真实，但高价值阶段又重新扩张到 `15+` |

因此：

1. 多时域 feature-level `SC` 不是无效补丁，它明显提升了 ceiling，也明显延后了失稳时点。
2. 但它还不能单独作为主线终解，因为高价值 imagined states 仍然会重新漂移。
3. 下一刀如果继续修，应该保留这次 feature-level imagined alignment 的收益，再把约束进一步聚焦到“高价值 imagined 区域”的语义稳定，而不是退回 controller 路线，也不是只继续加 teacher/critic 真实锚。

## 本轮最终结论

`v157` 最关键的收获不是“已经修好”，而是把问题又往前压实了一步：

1. 观测噪声层的问题已经清掉，warning spam 不再干扰判断。
2. imagined branch 的前中段漂移可以被多时域 feature `SC` 明显延后。
3. 但真正导致 run 仍然回撤的，是 imagined branch 在高价值阶段重新偏离 replay-aligned value manifold。
4. 因此下一刀必须继续直接打在 self-generated imagined branch 本身，而且要专门针对高价值阶段，而不是回到下游救火。
