# CartPole Pure-Imag `v154` 纯观测运行结论

## 运行对象

- run: `outputs/exp_seed42_v154_h15_light_sc_early_slowcritic_obsaudit_2500_20260314_01`
- 类型: 纯观测 run
- 目标: 不改训练行为，只给 pure-imag imagined batch 增加短开环审计指标，直接观察 imagined rollout 的第一失稳点
- 基线配置:
  - `horizon=15`
  - light `SC`
  - early slow critic
  - 不额外打开新的 controller 保护

## 这条 run 回答了什么

这条 run 不是再看“某个 patch 有没有把分数抬高”，而是直接回答三个问题：

1. solved 前后，最先失真的到底是哪一层。
2. world model 的 imagined rollout 到底是不是“整体不准”。
3. solved 后回撤时，真正先坏的是 reward / continue / feature 还是 value semantics。

## 真实评估轨迹

| eval step | mean |
| --- | ---: |
| 250 | 9.6 |
| 500 | 10.8 |
| 750 | 183.6 |
| 1000 | 144.0 |
| 1250 | 500.0 |
| 1500 | 500.0 |
| 1750 | 500.0 |
| 2000 | 500.0 |
| 2250 | 88.2 |
| 2500 | 231.6 |

补充结论：

- 这条 run 不是“从头到尾不行”，而是明确进入过一段长 solved 区。
- 它也不是“已经彻底稳定 solved”，因为 `2250` 出现了明显 solved 后回撤。
- 因此它非常适合拿来分析 solved 后为什么会重新掉出 corridor。

## 新增观测指标真正告诉了什么

`v154` 新增的是短开环 open-loop 审计：

- 用同一批 imagination seed
- 取短 context prefix + held-out suffix
- 在同一组 held-out 动作下，对比：
  - teacher-forced rollout
  - free-running imagined rollout
- 记录：
  - feature 漂移
  - reward gap
  - continue gap
  - value gap
  - teacher / imagined 对真实短回报的误差

最关键的是，这些指标不参与 loss，不改变 actor/critic/world model/controller 行为，所以它们更接近“系统自己正在发生什么”。

## 第一层结论：不是 reward / continue 头先坏

在整条 run 的 solved 前后，以下现象非常稳定：

- `imag/open_loop_audit_continue_gap_mean` 基本始终为 `0`
- `imag/open_loop_audit_reward_gap_mean` 一直很小，通常在 `0.005 ~ 0.016`
- `imag/open_loop_audit_feature_l1_mean` 不高，通常在 `0.16 ~ 0.25`

这意味着：

1. world model 并没有表现出“短期开环下一步就把 reward / continue 预测打坏”。
2. 也没有证据说明 short open-loop feature geometry 会直接爆炸。
3. 所以不能把当前主问题简单表述成“imagined rollout 完全不准”。

更准确的说法应该是：

**短期开环动力学仍然大体可用，但 value semantics 比 reward / continue 更早失稳。**

## 第二层结论：真正先敏感的是 critic 的 value 语义

### solved 前的早期信号

在 `700 ~ 1000` 区间，虽然 reward/continue 指标一直健康，但：

- `imag/open_loop_audit_value_gap_mean` 已经在 `1.7 ~ 3.3` 波动
- `actor/online_adv_mean` 经常明显为负
- `imag/signal_model_freshness` 和 `imag/signal_imagination_trust` 会同步走弱

其中最典型的是：

| step | feature_l1 | reward_gap | continue_gap | value_gap | online_adv | freshness | trust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 950 | 0.163 | 0.0107 | 0.0 | 2.334 | -5.977 | 0.506 | 0.475 |
| 1000 | 0.191 | 0.0125 | 0.0 | 1.713 | 0.506 | 0.959 | 0.903 |

这里最重要的不是某一列绝对值，而是顺序：

- reward/continue 仍然没坏
- feature 漂移也不算大
- 但 value 语义和 actor advantage 可以突然明显恶化，再突然恢复

这说明系统会周期性进入“critic 对 imagined latent 解释不稳”的局部区域。

## 第三层结论：solved 后回撤，第一前兆仍然不是 reward / continue

### solved 区

`1250 -> 2000` 这段真实评估连续为 `500.0`，说明系统确实进入过稳定 solved 区。

而且这个 solved 区不是靠 controller 在台面上硬救出来的：

- `imag/controller_stage` 全程为 `idle`
- `actor/negative_adv_guard_active` 全程为 `0`
- post-solved anchor 相关激活也没有真正接管

也就是说，这次 solved 主要不是“防护补丁短时托住”，而是主系统自己回到了相对正确的几何区。

### solved 后回撤窗口

`2250` 评估掉到 `88.2` 时，审计指标是：

| step | feature_l1 | reward_gap | continue_gap | value_gap | online_adv | freshness | trust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2200 | 0.221 | 0.0050 | 0.0 | 0.941 | -1.055 | 0.844 | 0.758 |
| 2250 | 0.215 | 0.0122 | 0.0 | 1.277 | 0.757 | 0.945 | 0.879 |
| 2350 | 0.205 | 0.0108 | 0.0 | 0.970 | -1.312 | 0.787 | 0.715 |
| 2400 | 0.217 | 0.0160 | 0.0 | 1.395 | -0.849 | 0.862 | 0.776 |
| 2450 | 0.211 | 0.0089 | 0.0 | 1.417 | -3.534 | 0.679 | 0.611 |
| 2500 | 0.234 | 0.0097 | 0.0 | 1.199 | -1.045 | 0.860 | 0.802 |

这一段的含义非常关键：

1. solved 后回撤时，`continue_gap_mean` 仍然没有坏。
2. `reward_gap_mean` 仍然很小。
3. `feature_l1_mean` 仍然没有出现灾难性跳升。
4. 最敏感的量仍然是：
   - `value_gap_mean`
   - `online_adv_mean`
   - `model_freshness`
   - `imagination_trust`

因此 solved 后回撤的第一前兆，依然不是 reward/continue prediction failure，而是：

**critic 对 imagined latent 的 value 解释先变差，系统对 imagined distribution 的信任度先下降，然后真实 eval 才掉。**

## 对“world model imagined rollout 不准”这句话的更准确表述

旧说法：

- world model imagined rollout 不准

这句话方向不算错，但太粗。

`v154` 之后更准确的版本应该是：

1. short open-loop 下，imagined reward / continue prediction 没有先坏。
2. imagined latent 也没有表现出立刻脱流形的灾难性 feature 爆炸。
3. 真正更早失稳的是：
   - critic 对 imagined latent 的 value semantics
   - 系统由此产生的 `online_adv`、`freshness`、`trust`
4. 所以“imagined rollout 不准”要更具体地理解为：
   - **不是动力学一步预测全面失真**
   - **而是 imagined latent 一旦走到某些局部区域，critic 的值语义不再稳定**

这比“world model 彻底不准”更贴近实际。

## 对模型的最新判断

目前最可信的结构性判断已经更新为：

1. 主要矛盾不是公式 bug。
2. 主要矛盾也不是 teacher-forced world model loss 爆炸。
3. 主要矛盾是：
   - actor 训练依赖的 imagined state distribution 会进入局部失配带；
   - reward/continue 看起来仍正常；
   - 但 critic 在这些 imagined states 上的值语义不稳；
   - actor 随后吃到错误或不稳定的 advantage 几何；
   - 系统可以在一段时间里 solved，但仍可能在后段重新掉出 corridor。

如果只用一句话概括：

**模型现在最本质的问题，不是“想象完全错了”，而是“想象状态一旦进入某些局部区域，value 语义比动力学语义更早坏掉”。**

## 工程上的最新思考

这条 `v154` 给出的工程启示很明确：

1. 下一轮主线不应该优先去改 reward/continue 头。
2. 也不应该先把 controller 再堆复杂。
3. 真正更该优先的方向是：
   - 让 critic 对 imagined latent 的 value semantics 更稳定
   - 让 solved 后的 imagined distribution 不那么容易滑进局部失配带
   - 用观测信号而不是训练内 return 作为早期失稳指标

具体来说，后续判断优先级应改成：

1. 先看 `imag/open_loop_audit_value_gap_*`
2. 再看 `actor/online_adv_mean`
3. 再看 `imag/signal_model_freshness` 与 `imag/signal_imagination_trust`
4. 最后才看训练内 `episode_return_ema`

因为 `v154` 已经证明：

- 训练内 return 可以很难看，但真实 eval 仍然是 `500`
- 反过来，训练内某些均值也可能没立刻爆，但真实 eval 已经回撤

## 本轮最终结论

`v154` 这条纯观测 run 让根因判断明显前进了一步：

- solved 前后都没有证据表明 reward / continue 头先坏；
- solved 后回撤也不是 feature geometry 先明显爆炸；
- 第一前兆更稳定地落在 value semantics / advantage geometry / trust-freshness 上；
- 因此当前模型的真正主问题，应更准确地定义为：

**imagined latent 的局部分布一旦偏离 critic 已校准的区域，value 语义先失稳，actor 随后被带偏，系统于是从 solved corridor 中掉出。**
