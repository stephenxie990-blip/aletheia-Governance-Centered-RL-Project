# CartPole Pure-Imag 下一轮主线分析

## 结论先行

当前最可信的判断已经比“controller 哪个 patch 更好”更往前走了一层：

1. 问题的第一起点不在公式实现错误。
2. 问题的第一起点也不在 world model 的 teacher-forced loss 爆炸。
3. 第一性问题更像是：
   - world model 在真实轨迹附近看起来正常；
   - 但 actor/critic 真正训练所依赖的 self-generated imagined rollout 更早进入失真区；
   - imagined continue 先塌，effective horizon 跟着缩；
   - critic 对这些 imagined states 的 value 下调不够快，导致 `value - return` gap 拉大；
   - `online_advantage` 长期转负；
   - actor 只有在“正向 corridor 仍被保住”的极窄窗口里才能继续学到东西。
4. 这就是为什么当前系统会出现“只有很窄的参数区间或阶段时机才能成功”的现象。
   - 这不是一种真正健康的 solved；
   - 它更像是在多个误差相互抵消时，系统短暂踩中了一个还能工作的小走廊。

## 这轮分析回答了什么

这次不再问“哪个补丁分数更高”，而是问三件更底层的事：

1. 真实坏链路是从哪里开始破的。
2. controller 复杂化到底是在修根因，还是只是在改变坏掉的表现形式。
3. 为什么 `v130` 这类 run 还能活，而 `v145-v148` 会锁死。

## 真实坏链路

在当前 CartPole pure-imag 主链里，真正影响训练的是两条不同路径：

1. world model 监督路径
   - 主要在真实 replay / teacher-forced 条件下计算 reward、continue、recon、KL
2. actor/critic 更新路径
   - 主要在 imagined rollout 上计算 return、base、advantage、weights

这两条路径不是同一回事。

所以“world model loss 正常”并不能推出“actor 真正看到的 imagined geometry 也正常”。

当前最像真实故障序列的是：

1. imagined continue 先开始比 teacher-forced continue 更差
2. effective horizon 随之缩短
3. return 主要因为 horizon 而不是 reward 下滑
4. critic value 对 imagined states 仍偏高
5. `online_advantage = return - online_base` 变成系统性负值
6. 如果 actor 的 `adv = target - base` 还能被阶段修复拉回到零附近或轻微正值，run 还能续命
7. 如果这条 corridor 断掉，系统就会进入“被保护地继续负梯度”的锁死状态

## 关键数据证据

### 1. teacher-forced continue 正常，不代表 imagined continue 正常

`v130` 与 `v145` 在 `1250-1500` 的对比最能说明问题：

| step | run | stage | `wm/continue_prob_mean` | `imag/continue_prob_mean` | gap |
| --- | --- | --- | ---: | ---: | ---: |
| 1300 | v130 | `persistence_release` | 0.984 | 0.774 | 0.210 |
| 1300 | v145 | `trigger` | 0.985 | 0.786 | 0.199 |
| 1350 | v130 | `persistence_release` | 0.979 | 0.757 | 0.222 |
| 1350 | v145 | `trigger` | 0.981 | 0.679 | 0.302 |
| 1400 | v130 | `persistence_release` | 0.982 | 0.794 | 0.188 |
| 1400 | v145 | `persistence` | 0.982 | 0.615 | 0.366 |
| 1500 | v130 | `trigger` | 0.989 | 0.881 | 0.108 |
| 1500 | v145 | `persistence` | 0.984 | 0.705 | 0.278 |

结论很直接：

- world model 在 teacher-forced 轨迹上的 continue 预测基本一直正常；
- actor 真正吃到的 imagined continue 才是先坏掉的那一层；
- 坏 run 不是 reward 先崩，而是 horizon 先崩。

### 2. CartPole 的核心矛盾是 horizon，不是 reward

CartPole 每步 reward 基本接近 `1`，所以 `return` 的主要变化来自“还能活多久”。

`v130` 与 `v145` 在 `1400-1500` 的对比：

| step | run | `imag/reward_mean` | `imag/return_mean` | `imag/value_mean` | `imag/effective_horizon` |
| --- | --- | ---: | ---: | ---: | ---: |
| 1400 | v130 | 0.950 | 8.398 | 13.309 | 7.217 |
| 1400 | v145 | 0.952 | 5.237 | 15.863 | 5.790 |
| 1500 | v130 | 0.965 | 13.241 | 16.007 | 8.425 |
| 1500 | v145 | 0.993 | 6.685 | 13.767 | 6.111 |

reward 差异很小，真正拉开差距的是：

- imagined continue
- effective horizon
- return

### 3. critic 对 imagined states 下调太慢，导致 online advantage 长期转负

| step | run | `imag/return_mean` | `imag/value_mean` | `actor/online_adv_mean` |
| --- | --- | ---: | ---: | ---: |
| 1400 | v130 | 8.398 | 13.309 | -4.911 |
| 1400 | v145 | 5.237 | 15.863 | -10.626 |
| 1500 | v130 | 13.241 | 16.007 | -2.766 |
| 1500 | v145 | 6.685 | 13.767 | -7.082 |

这说明：

- imagined return 已经掉下去了；
- critic value 还没有同步掉下来；
- actor 看到的是一大片负 online advantage。

## 为什么 `v130` 还能活，而 `v145` 会锁死

真正的分水岭不是“有没有保护”，而是：

**坏阶段里，actor 的 `adv = target - base` 能不能被修回正值。**

### `v130`

`v130` 在 `1300-1450` 虽然 `online_adv` 已明显为负，但 actor 真正用的 `adv` 被阶段修复拉回了正值或接近零：

| step | stage | `online_adv` | `adv` | `base_return_cap_active` | tail repair | mismatch multiplier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1300 | `persistence_release` | -7.468 | 2.014 | 1.0 | 1.0 | 1.5 |
| 1350 | `persistence_release` | -7.482 | 1.817 | 1.0 | 1.0 | 1.5 |
| 1400 | `persistence_release` | -4.911 | 1.089 | 1.0 | 1.0 | 1.5 |
| 1450 | `persistence_release` | -4.112 | 0.177 | 1.0 | 0.0 | 1.0 |

这说明 `v130` 真正活下来的原因不是“它没坏”，而是：

- 它虽然也发生了 imagined geometry 失配；
- 但 `persistence_release` 这条修复走廊暂时保住了 actor 的正向学习信号。

### `v145`

`v145` 更早进入坏区，而且进入 `persistence` 之后虽然也触发了 tail repair，但 `adv` 仍持续为负：

| step | stage | `online_adv` | `adv` | `base_return_cap_active` | tail repair | mismatch multiplier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1350 | `trigger` | -8.026 | -8.026 | 0.0 | 0.0 | 1.0 |
| 1400 | `persistence` | -10.626 | -7.274 | 0.0 | 1.0 | 1.656 |
| 1450 | `persistence` | -8.271 | -5.599 | 0.0 | 1.0 | 1.5 |
| 1500 | `persistence` | -7.082 | -5.198 | 0.0 | 1.0 | 1.5 |

也就是说：

- `v145` 并不是没保护；
- 它的问题是保护发生在一个已经失去正向 actor corridor 的阶段里；
- 于是保护不再是“帮系统恢复”，而变成“让系统在负梯度里更稳定地待着”。

## 为什么会出现“窄窗成功”

这件事现在可以用很直白的话讲清楚：

1. 上游 imagined rollout 本身已经不够稳
2. critic 对 imagined states 的校准也不够稳
3. controller 在试图补这个失配
4. 当 controller 的修复强度、接管时机、release 时机刚好对上时，actor 还能保住一条很窄的正向 corridor
5. 一旦稍微偏一点，就会掉进两种坏结果之一：
   - 修复太弱：负 online advantage 直接吞掉 actor 梯度
   - 修复太强：系统过早进入长期 `persistence`，trigger 无法再接管

所以现在的“能成功”更像是偶然踩中一个小走廊，而不是找到了一套宽容、稳定、可推广的机制。

## 为什么今天下午的方向会越修越坏

`v144-v148` 这条线的共同特征不是“公式错了”，而是：

1. 更早让 `persistence` 接管
2. 更久把系统留在 `persistence`
3. 却没有修好 `persistence` 内部的 imagined continue、value 校准、actor corridor

结果就是：

- `trigger` 重新接管和纠偏的机会变少；
- `persistence_release` 消失；
- 系统从“带波动地坏”变成“被保护地锁死”。

## 对当前系统复杂度的判断

当前 controller 之所以变复杂，不是因为任务本身真的需要这么多花样，而更像是：

- 上游 imagined rollout 漂移
- 中游 critic 校准滞后
- 下游 actor 目标几何脆弱

这三个问题没有被从根上修掉，所以代码里才会堆出很多阶段门控、floor、release、repair。

如果后续真的把根因修对，controller 反而应该有机会简化。

## 下一轮修复主线

下一轮不应该继续把主要精力花在 controller patch 上，而应该按优先级收敛到下面三件事：

### 1. 优先审 imagined rollout 为什么比 teacher-forced 更早出流形

要回答的问题：

- imagined latent/state 是否在 `1250-1400` 开始明显偏离 replay posterior 附近的分布；
- continue head 是因为 rollout state 漂了才给出更低 continue，还是 head 本身在 imagined 分布上外推很差；
- 这种漂移是否可以用更短 horizon、anchored bootstrap、replay-posterior short rollout 做约束。

### 2. 优先审 critic 为什么对 imagined bad states 仍偏乐观

要回答的问题：

- 是 value normalizer 响应太慢；
- 还是 critic target 本身在 imagined tail 上已经偏高；
- 还是 critic 缺少来自真实 replay state 的校准锚点。

### 3. 重新定义 controller 的职责

controller 不应继续承担“替上游几何错误兜底”的主要责任。

更合理的目标应该是：

- 只负责把系统从危险区平稳带回；
- 而不是靠越来越多的阶段 patch 去制造一个临时可用的小走廊。

## 当前工作结论

如果只用一句话概括当前状态：

**模型现在不是“算错了”，而是“上游 imagined 几何已经开始漂，但下游仍在按一个过于乐观的价值几何去更新 actor”，controller 只能偶尔把这件事补成一个窄窗成功。**

真正的下一轮主线应该是把这层结构性失配拆开，而不是再继续往 controller 上堆补丁。
