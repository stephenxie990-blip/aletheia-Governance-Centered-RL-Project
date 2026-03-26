# CartPole Pure-Imag 机理级分析

## 目标

这份分析不再回答“哪个 patch 更好”，而是回答更底层的问题：

1. 当前 pure-imag 系统为什么会在 late stage 变得不稳定。
2. 不稳定是先从世界模型、价值基线、策略梯度，还是从 controller 开始的。
3. 为什么有些版本只能在很窄的窗口里成功。

结论先说：

- 当前主问题不是公式实现错误。
- 当前主问题也不是 world model 的全局训练 loss 爆炸。
- 当前最像根因的，是一种“teacher-forced world model 看起来正常，但 actor 真正使用的 imagined rollout 几何已经失真”的结构性失配。
- controller 的复杂化，更多是在补偿这类失配；它能决定问题以什么方式表现出来，但它不是第一因。

## 训练链的真实工作方式

在当前 CartPole pure-imag 主链上，真正起作用的是这条链：

1. `_build_imagined_batch()`
   - 用 world model rollout imagined trajectory
   - 基于 imagined reward / continue / critic value 计算 return
   - 生成 actor target / base / weights / online_advantage
2. `train_step()`
   - actor 用 `target - base` 做梯度
   - critic 用 imagined target 训练 value
   - world model 继续按 reward / continue / recon / KL 做 teacher-forced 监督
3. controller / stage 逻辑
   - 通过 `trigger / persistence / persistence_release / post_solved` 改写 actor scale、base return cap、tail repair、critic multiplier 等

这里最重要的结构事实是：

- world model 的训练损失主要在真实轨迹上算
- actor / critic 的决策更新主要在 imagined trajectory 上算

这两条链不是同一件事。

也正因为如此，world model 即使“训练 loss 看起来正常”，也不代表 imagined rollout 的长链几何一定正常。

## 四个必须满足的不变量

一个真正稳定的 pure-imag CartPole 系统，至少要同时满足下面四条：

1. imagined tail 必须自洽
   - `continue / return / value` 不能长期互相冲突
2. actor 的有效样本不能塌缩
   - imagined branches 里不能只剩极少数分支还有梯度价值
3. 保护态不能切断纠偏通道
   - 系统可以保守，但不能只会“被保护地继续错”
4. teacher-forced world model 与 imagined rollout 不能长期背离
   - 如果真实轨迹上看起来一直很好，但 imagined trajectory 上已经明显失真，这套系统就会天然脆

当前项目最先破掉的，是第 1 条和第 4 条。

## 关键证据一：公式正确，但这不等于几何稳定

前面的公式级审计已经说明：

- lambda return
- reward / continue / recon / KL loss
- actor / critic / twohot / normalizer
- negative online advantage gate
- persistence tail repair

这些公式在当前主链上基本都实现正确。

所以问题不在“算错了”，而在“这些算对了的东西组合起来，是否形成了稳定的几何关系”。

## 关键证据二：world model 全局 loss 没坏，但 imagined rollout 已经坏了

对比 `v130`、`v143`、`v145` 的训练指标，可以看到一个非常重要的现象：

- `loss_wm` 一直比较平稳
- `wm/continue_prob_mean` 一直维持在 `0.97-0.99`
- `wm/reward_pred_mean` 也一直稳定

也就是说，teacher-forced 的世界模型训练并没有显示出“系统性崩坏”。

但是一看 actor 真正使用的 imagined batch，就不是这么回事了。

### 典型对比：`wm continue` 和 `imag continue` 的分裂

| step | run | stage | `wm/continue_prob_mean` | `imag/continue_prob_mean` | gap |
| --- | --- | --- | ---: | ---: | ---: |
| `1250` | `v130` | `trigger` | `0.988` | `0.845` | `0.143` |
| `1250` | `v145` | `trigger` | `0.994` | `0.831` | `0.163` |
| `1350` | `v130` | `persistence_release` | `0.979` | `0.757` | `0.222` |
| `1350` | `v145` | `trigger` | `0.981` | `0.679` | `0.302` |
| `1400` | `v130` | `persistence_release` | `0.982` | `0.794` | `0.188` |
| `1400` | `v145` | `persistence` | `0.982` | `0.615` | `0.366` |
| `1500` | `v130` | `trigger` | `0.989` | `0.881` | `0.108` |
| `1500` | `v145` | `persistence` | `0.984` | `0.705` | `0.278` |

这说明：

- world model 在真实/teacher-forced 轨迹上的 continue 预测一直很正常
- 但一旦 rollout 进入 actor 真正训练用的 imagined trajectory，continue 就显著下降
- 而且坏版本下降得更严重

这不是“world model 完全没学会”，而是：

- 它在真实数据流形附近还能工作
- 但在自生成轨迹上开始累积失真

这正是 pure-imag 训练里最危险的一种失败形态。

## 关键证据三：在 CartPole 里，真正塌的是 horizon，不是 reward

CartPole 的 reward 基本是每步 `1`。

这意味着 imagined return 的高低，主要不是由 reward head 决定，而是由“还能活多久”决定。

实际数据也支持这一点：

### `v130 / v145 / v143` 对比

| step | run | stage | `imag/reward_mean` | `imag/return_mean` | `imag/value_mean` | `imag/continue_prob_mean` | `imag/effective_horizon` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1400` | `v130` | `persistence_release` | `0.950` | `8.398` | `13.309` | `0.794` | `7.217` |
| `1400` | `v145` | `persistence` | `0.952` | `5.237` | `15.863` | `0.615` | `5.790` |
| `1400` | `v143` | `trigger` | `1.002` | `8.550` | `16.090` | `0.735` | `5.190` |
| `1500` | `v130` | `trigger` | `0.965` | `13.241` | `16.007` | `0.881` | `8.425` |
| `1500` | `v145` | `persistence` | `0.993` | `6.685` | `13.767` | `0.705` | `6.111` |
| `1500` | `v143` | `trigger` | `1.011` | `7.047` | `13.916` | `0.697` | `5.668` |

核心现象非常清楚：

- `imag/reward_mean` 在这些版本里其实差异不大，基本都在 `0.95-1.01`
- 真正拉开差距的是：
  - `imag/continue_prob_mean`
  - `imag/effective_horizon`
  - `imag/return_mean`

所以对于 CartPole，这个系统的主矛盾不是 reward 学坏了，而是：

- imagined horizon 被提前截断
- return 主要被 premature termination 压低

这就是为什么 controller 老在围绕 `continue cap / persistence / release` 打转，因为它碰到的是 horizon 问题，不是 reward 问题。

## 关键证据四：critic 仍然乐观，于是 online advantage 变成系统性负值

一旦 imagined horizon 塌缩，就会出现第二层问题：

- imagined return 掉下去
- 但 critic value 没有同步掉下去

结果就是：

- `online_adv = returns - online_base_actor`

长期为负。

### 典型例子

| step | run | stage | `imag/return_mean` | `imag/value_mean` | `actor/online_adv_mean` |
| --- | --- | --- | ---: | ---: | ---: |
| `1400` | `v130` | `persistence_release` | `8.398` | `13.309` | `-4.911` |
| `1400` | `v145` | `persistence` | `5.237` | `15.863` | `-10.626` |
| `1500` | `v130` | `trigger` | `13.241` | `16.007` | `-2.766` |
| `1500` | `v145` | `persistence` | `6.685` | `13.767` | `-7.082` |

这说明：

- bad run 的问题不是“value 和 return 一起降了”
- 而是 value 明显比 return 更慢、更不愿意下调

这会直接造成 actor 看到一大片负优势样本。

## 关键证据五：真正决定能不能恢复的，不是“有没有保护”，而是 actor 的 `adv` 能不能被修回正值

这是今天这轮分析里最关键的新结论。

我们之前一直盯着：

- 是否进入 `persistence`
- 是否触发 `negative_adv_guard`
- 是否打开 `highwater`

但更根本的分界线其实是：

- actor 真正用来更新的 `adv = target_actor - base_actor`
- 能不能在坏阶段里被修回到接近 0 或轻微正值

### `v130` 的关键现象

在 `1300-1450`，`v130` 已经进入 `persistence_release`，而且：

- `actor/online_adv_mean` 仍然是明显负值
- 但 `actor/adv_mean` 已经被修回正值

例如：

| step | run | stage | `online_adv` | `adv` | target repair | weight floor | mismatch multiplier | base return cap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1300` | `v130` | `persistence_release` | `-7.468` | `2.014` | `1.0` | `1.0` | `1.500` | `1.0` |
| `1350` | `v130` | `persistence_release` | `-7.482` | `1.817` | `1.0` | `1.0` | `1.500` | `1.0` |
| `1400` | `v130` | `persistence_release` | `-4.911` | `1.089` | `1.0` | `1.0` | `1.500` | `1.0` |
| `1450` | `v130` | `persistence_release` | `-4.112` | `0.177` | `0.0` | `0.0` | `1.000` | `1.0` |

这意味着：

- world model / critic 的原始关系仍然是失配的
- 但 `persistence_release` 这段“修复走廊”把 actor 的有效学习信号保住了

### `v145` 的关键现象

`v145` 则完全不同：

- 它更早掉进 `persistence`
- 虽然 tail repair 也会激活
- 但 `adv` 仍然保持明显负值

例如：

| step | run | stage | `online_adv` | `adv` | target repair | weight floor | mismatch multiplier | base return cap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1400` | `v145` | `persistence` | `-10.626` | `-7.274` | `1.0` | `1.0` | `1.656` | `0.0` |
| `1450` | `v145` | `persistence` | `-8.271` | `-5.599` | `1.0` | `1.0` | `1.500` | `0.0` |
| `1500` | `v145` | `persistence` | `-7.082` | `-5.198` | `1.0` | `1.0` | `1.500` | `0.0` |

这说明：

- `tail repair` 本身不是万能修复
- 如果它发生在一个不再保留正向 actor corridor 的阶段里，它只会把系统变成“被保护，但仍在负梯度里”

## 为什么 `v130` 和 `v145` 会走出完全不同的命运

从代码上看，最关键的区别是：

### 1. base return cap 的阶段作用域不同

在 [_build_imagined_batch()](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L9498) 里，`base_return_cap_margin` 会按 stage 决定来源：

- `post_solved`
- `persistence_release`
- `persistence`
- `post_entry*`
- 或者满足 release 条件的 `trigger`

对应逻辑在 [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L9498) 到 [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L9525)。

`v130` 的强点是：

- 它在 `persistence_release` 里还能拿到一个学习友好的 `base_return_cap`

`v145` 的问题是：

- 它更早绕过这段走廊，直接落进 `persistence`
- 结果虽然也有保护，但 `base_return_cap_active` 不再作为主要修复力量出现

### 2. tail mismatch repair 只负责“修尾巴”，不负责“恢复再接管”

`persistence tail mismatch` 的核心逻辑在 [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L9656) 到 [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py#L9869)。

它做的是：

1. 检测 tail gap / continue / weight 是否同时变坏
2. 如果变坏：
   - 把 target 往上修一截
   - 给 tail weights 设地板
   - 给 critic 增 multiplier

这套逻辑能修“尾巴”，但它本身不保证 actor 重新获得正向更新。

所以：

- 在 `v130` 里，它和 `persistence_release + base_return_cap` 配合，形成了修复走廊
- 在 `v145` 里，它只剩“保守保护”，没有形成学习走廊

### 3. signal 层其实是在描述这个结果

`trigger_effectiveness` 的定义在 [signals.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/controller/signals.py#L174) 到 [signals.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/controller/signals.py#L196)。

它由这几项组成：

- `adv_component`
- `guard_component`
- `release_component`
- `stage_component`

坏版本的致命特征正好是：

1. `adv_component` 太低
2. `guard_component` 失效
3. `stage_component` 一旦落到 persistence，只剩 `0.25`

所以 signal 层看到的是：

- `trigger_effectiveness` 越来越低
- `handoff_necessity` 越来越高

但这不是 signal 算错了，而是 signal 正在忠实地报告：

- 这个系统已经不在一个“还能靠 trigger 恢复”的区域里了

## 为什么 world model loss 没报警

这点非常关键。

如果只看这些量：

- `loss_wm`
- `wm/continue_prob_mean`
- `wm/reward_pred_mean`

很多坏 run 看上去根本不坏。

原因是：

1. world model 训练目标是 teacher-forced 的局部监督
2. actor / critic 用的是 imagined rollout 的长链几何
3. 这两者之间缺少一个强约束，确保“在自生成轨迹上也不能漂”

因此系统会出现一种典型错觉：

- world model loss 很正常
- 但 imagined horizon 已经开始缩
- critic 还在按旧几何给高 value
- actor 看到大面积负优势

也就是说：

- 当前世界模型更像是“真实数据上的局部预测器”
- 还没有变成“想象闭环里的稳定动力学器”

## 为什么 controller 会越修越复杂

这不是因为问题本身一定需要复杂 controller。

更可能是因为：

- 上游的 imagined rollout 几何已经失配
- 下游又必须继续训练
- 于是 controller 只能不断承担更多“补偿上游错误”的职责

最终 controller 就会同时做四件本不该都由它做的事：

1. 决定何时保守
2. 决定何时切 stage
3. 决定如何给 actor 留一点正向梯度
4. 决定如何让 critic 不至于把系统带偏

这也是为什么最近几轮修复会表现成：

- 越修越复杂
- 越修越像 patch 叠 patch
- 一旦 patch 方向错了，就会整条线一起退化

## 当前最可信的机理判断

到这一步，我对当前系统的判断可以压缩成一句话：

> 当前 pure-imag 的主问题，是 imagined rollout 的 horizon 几何先坏，critic baseline 跟着失配，actor 有效梯度随之塌缩；controller 只能决定系统是“带着一点修复通道地坏”，还是“被保护地锁死”。

拆开说就是：

1. 第一因：imagined rollout 在自生成轨迹上发生 continue / horizon 失真
2. 第二因：critic value 没有同步校准，导致 value > return
3. 第三因：actor 看到大量负优势，更新空间迅速收窄
4. 第四因：controller 若过早把系统推进 `persistence`，就会切断恢复所需的正向 actor corridor

这四件事加在一起，就是我们现在看到的 late-stage 不稳定。

## 对下一轮修复的约束

如果这个判断是对的，那么下一轮修复就必须遵守这几个约束：

1. 不再把 controller 当主战场
   - controller 只能做轻量调度
   - 不能继续承担越来越多上游补偿任务
2. 优先修 imagined rollout 的闭环几何
   - 重点是 continue / horizon / tail consistency
   - 不是继续加 release / bypass / freeze
3. 优先修 actor 有没有有效梯度
   - 关注 `adv`、`weights`、tail effective sample
   - 不是只看 stage 名字
4. 只要一个修复不能恢复“正向 actor corridor”，就不算真正有效
   - 即使它能暂时把分数崩点往后搬，也不能晋升主线

## 机理层面的下一步

下一轮研究不应该直接开始 patch，而应该继续验证下面三个问题：

1. imagined continue 为什么会在 self-generated rollout 上明显低于 teacher-forced continue
   - 这是单步 continue 头的问题
   - 还是 latent state 漂移问题
2. critic 为什么会对这些 imagined states 保持偏乐观的 value
   - 是更新太慢
   - 是 target 设计本身偏稳
   - 还是 value 学到的是另一套 latent 几何
3. actor 的有效梯度空间从哪一步开始塌
   - 是由 horizon 先塌导致
   - 还是先由 value-return gap 放大导致

只有把这三件事继续往前拆清楚，后面的修复才可能重新回到“优雅解法”。
