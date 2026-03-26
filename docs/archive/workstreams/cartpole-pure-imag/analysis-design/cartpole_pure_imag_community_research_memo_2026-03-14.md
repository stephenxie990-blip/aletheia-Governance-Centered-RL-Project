# CartPole Pure-Imag 社区研究备忘

日期：2026-03-14  
目的：在继续修改本项目之前，先看社区和一手资料里别人是否遇到过类似问题，以及主流解法到底长什么样。

## 1. 研究问题

我们当前面对的不是“公式写错”，而是这种现象：

- pure-imag 主链可以在一段时间内学起来
- 但后段会出现控制权切换抖动、性能回落、长程 retention 不稳
- 外围 controller patch 能修一段，但会伤另一段

需要回答的问题是：

1. 社区有没有遇到类似的 world-model / imagination instability？
2. 主流解法是继续堆 controller，还是回到底层训练设计？
3. 对我们这个项目，哪些结论值得借鉴，哪些不值得机械照搬？

## 2. 检索范围

本轮优先看了信噪比最高的一手材料：

- DreamerV3 官方论文与项目页
- DreamerV3 官方仓库公开讨论
- MBPO 论文
- TD-MPC / TD-MPC2 论文与官方仓库
- 直接讨论 MBRL world-model failure mode 的论文

说明：

- 我也扫了部分公开 issue / 讨论页，但仓库层讨论很多集中在实现、环境支持、复现细节。
- 对“为什么 pure imagination 后段会崩”这类问题，真正有解释力的材料主要还是论文和官方项目页，而不是 issue 贴子。

## 3. 一手资料结论

### 3.1 DreamerV3 的共识不是“外面包很多阶段机”，而是“核心训练机制要自稳”

DreamerV3 论文摘要直接把稳定性归因到三类技术：

- normalization
- balancing
- transformations

来源：

- 论文：[Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
- 官方项目页：[DreamerV3](https://danijar.com/project/dreamerv3/)

对我们的启发：

- 成熟 world-model 系统的主流思路，是把稳定性尽量做进核心训练路径里。
- 如果我们最后必须靠越来越复杂的外围 controller 才能跑稳，通常说明还没有抓到真正的内核稳定变量。

### 3.2 社区默认就知道 raw actor target 不够稳

DreamerV3 官方仓库里，有复现者专门追问 actor loss 为什么不是直接按原始 return 做，而是先做 normalize，再减 base，再乘 trajectory weight。

来源：

- 官方仓库 issue：[danijar/dreamerv3 issue #20](https://github.com/danijar/dreamerv3/issues/20)

对我们的启发：

- “actor 直接吃 imagined return” 这件事，社区本来就不认为天然稳定。
- 我们当前围绕 `base return cap / target blend / weight / online advantage` 搭脚手架，并不是完全离谱；它对应的是一个真实存在的稳定性问题。
- 但这也反过来说明：如果这些保护越来越碎，应该回头去看核心 actor target 设计，而不是只在外层补窗口。

### 3.3 MBPO 的答案非常明确：模型有偏差时，不要让它做长自由 rollout

MBPO 的核心结论是：

- model bias 会积累
- 所以要用 short model rollouts branched from real states
- 不要让模型长距离自由展开后再把整段都喂给策略

来源：

- 论文：[When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)

对我们的启发：

- 我们现在的 pure-imag 问题，和 MBPO 警告的风险高度同型。
- 如果 imagination 自己变成主要训练场，而世界模型又不是完全可信，越到后段越容易被模型偏差放大。
- 这意味着“继续用 controller 保底”不是唯一方向；另一条更本质的方向，是减少让长自由 imagination 直接主导梯度的程度。

### 3.4 TD-MPC / TD-MPC2 的方向是“短 horizon + terminal value”，而不是长自由 imagination

TD-MPC 和 TD-MPC2 都强调：

- short-horizon planning
- latent consistency
- terminal value 接长程价值

来源：

- 论文：[TD-MPC](https://proceedings.mlr.press/v162/hansen22a.html)
- 项目页：[TD-MPC](https://td-mpc.github.io/)
- 论文：[TD-MPC2](https://arxiv.org/abs/2310.16828)
- 官方仓库：[tdmpc2 README](https://github.com/nicklashansen/tdmpc2)

额外观察：

- TD-MPC2 官方仓库直到 2025 年 4 月才把 episodic tasks 作为正式支持项加入。
- 这说明 termination / continue / episodic handling 在 world-model 系统里不是小问题。

对我们的启发：

- 对像 CartPole 这样终止很明确的 episodic 任务，continue / tail / terminal value 处理非常关键。
- 如果我们的 imagined controller 大量围绕 tail、release、continue、handoff 打补丁，这不奇怪，但也提醒我们要优先回到 horizon 设计本身。

### 3.5 “Mind the Model, Not the Agent” 给了一个最贴近我们现象的提醒

这篇论文研究的是 MBRL 里的 primacy bias。核心发现是：

- 先学歪的 world model 会长期影响后续学习
- agent 侧很难自己把它纠正回来
- reset / refresh world model 往往比 reset agent 更有效

来源：

- 论文：[Mind the Model, Not the Agent: Correcting Primacy Bias in Model-Based RL](https://arxiv.org/abs/2310.15017)

对我们的启发：

- 如果我们后面越来越像是在“保护 actor 不要被 imagined bias 继续带坏”，那可能只是症状。
- 真正该怀疑的，是 world model 在某个阶段之后已经形成偏置，而 controller 只是在替它兜底。

### 3.6 另一条社区共识：当模型不完全可信时，要把“不可信”显式量化

这条线在 uncertainty-aware / pessimistic MBRL 里非常明确。

代表工作：

- PETS：[Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114)
- MOPO：[Model-based Offline Policy Optimization](https://arxiv.org/abs/2005.13239)
- MOReL：[Model-Based Offline Reinforcement Learning](https://arxiv.org/abs/2005.05951)

这些工作的共识不是：

- “发现模型不稳，就多加几层 controller”

而是：

- 用 ensemble / uncertainty / pessimism 去显式表示“这里的模型不可靠”
- 然后在策略优化时主动降权、惩罚、或拒绝利用这些不可靠区域

对我们的启发：

- 我们现在反复出现的 `late-trigger base cap / sustain / handoff block`，本质上也在表达“这里的 imagination 不完全可信”。
- 区别在于，当前项目里这种“不可信”是通过很多局部阈值间接表达的，还不是一个统一、可测量的信号。
- 这说明一个更优雅的方向，不一定是再造更复杂的阶段机，而是把 `imagination_trust` 直接定义成一种 uncertainty / disagreement / pessimism 指标，再让 actor 与 handoff 逻辑依赖它。

## 4. 和我们项目的映射

### 4.1 当前实验事实

本地关键 run：

| Run | 1750 | 2000 | 2250 | 2500 | 结论 |
| --- | --- | --- | --- | --- | --- |
| `v123` | 175.5 | 125.6 | 132.1 | 191.6 | release-tail sustain 有帮助，但后段仍不稳 |
| `v125` | 175.5 | 147.8 | 145.7 | 162.2 | handoff block 修好中后段，但末段变差 |
| `v126` | 175.5 | 147.8 | 145.7 | 162.2 | handoff confirm=2 基本不改变真实轨迹 |

对应输出目录：

- [v123](../outputs/exp_seed42_v123_release_tail_sustain_only_2500_20260314_01)
- [v125](../outputs/exp_seed42_v125_release_tail_sustain_handoffblock100_2500_20260314_01)
- [v126](../outputs/exp_seed42_v126_release_tail_sustain_handoffblock100_confirm2_2500_20260314_01)

### 4.2 这些实验和社区结论是对得上的

对照上面的外部资料，本地现象有三个强匹配点。

第一，问题不像是“某个公式错了”，更像是 model trust 问题。

- 我们自己的 `formula_chain_audit.py` 已经证明主链公式在当前范围内对得上。
- 所以更像是 imagined rollout 变得没那么可信，而不是代码在算错。

第二，问题不像是“单点噪声”，更像是持续性偏差信号。

- `v126` 加了两次确认，轨迹和 `v125` 一样。
- 这说明坏 handoff 不是因为偶发抖一下，而是坏信号会持续几个窗口。
- 这更符合“底层模型信任下降”而不是“controller 少了一个 debounce”。

第三，当前 controller patch 更像在修代理变量，不是在修根变量。

- `base cap`
- `sustain`
- `handoff block`
- `confirmation`

这些都在围着同一个问题转：

> trigger 现在到底还有没有真实修复能力？

但系统没有显式建模这个量，所以只能靠很多局部阈值去猜。

## 5. 当前最像真的根因

基于社区资料和本地实验，我现在最倾向的解释是：

### 5.1 纯 imagination 扛了太多长程训练职责

长自由 rollout 天然更容易被 world model bias 放大。  
这和 MBPO、TD-MPC 的结论一致。

### 5.2 controller 缺一个统一的“model trust / trigger effectiveness”语义

目前 controller 大量依赖：

- release progress
- gap
- continue
- eval threshold
- online advantage
- base return cap

这些都是旁证，不是主证。  
真正缺的是：

> 当前 trigger 是否仍在真实地把局面拉回，而不是只是在表面维持。

### 5.3 world model 可能存在阶段性固化偏差

如果 world model 在某一阶段先学歪，后面的 actor / critic 会一起围着它打转。  
这和 primacy bias 论文的结论是相符的。

## 6. 当前不建议继续走的路

以下方向现在都不该成为主线：

- 继续叠更多阶段门控
- 继续补更多局部阈值
- 继续加更多短 latch / debounce / block
- 继续把一次实验有效的 patch 永久化

原因：

- 这些动作可以短期定位问题
- 但越来越像“症状补丁”
- 不会自然收敛为优雅结构

## 7. 更值得走的主线

### 7.1 先把 controller 收敛到更少的核心量

优先定义三个更本质的内部语义：

- `imagination_trust`
- `trigger_effectiveness`
- `handoff_necessity`

让 controller 不再只看“阈值是否满足”，而是看“当前这条 imagined 修复链还有没有效”。

### 7.2 考虑让长自由 imagination 退居二线

社区更成熟的路线，不是让长自由 imagination 一直主导训练，而是：

- 短 rollout
- 从更可信的状态起跳
- 用 terminal value 接长程目标

### 7.3 把 trust 直接变成训练信号，而不是只做 controller patch

结合 uncertainty-aware MBRL 的经验，更值得尝试的不是“再加一层 handoff 条件”，而是让下列东西显式化：

- imagined rollout 的 disagreement
- posterior 起点附近和长自由 rollout 末端的误差差异
- imagined target 与 online / real support 的持续偏离

这样做的好处是：

- controller 不必再猜太多
- actor / critic / handoff 都可以共享同一套 trust 信号
- 复杂度有机会从“十几个 patch”收敛成“少数高层语义”

### 7.4 把 world model freshness 纳入调试主线

后续不只盯 actor / handoff，也要验证：

- world model 是否在某一段后开始固化偏差
- 是否需要 refresh / reweight / retrain 策略

## 8. 下一轮建议研究题

在继续改代码前，最值得验证的不是“再加一个阈值”，而是下面这些问题：

1. imagination trust 能否被直接观测，而不是通过一堆 proxy 间接猜？
2. 当前 tail collapse 更像 horizon 设计问题，还是 world model 固化问题？
3. 如果缩短 pure-imag 主优化 horizon，会不会比继续加 handoff patch 更有效？
4. 是否应该增加一类“world model refresh”实验，而不是继续只改 actor/controller？

## 9. 本轮研究的结论

一句话总结：

> 社区主流解法并不支持“越修越复杂的外围 controller”，更支持把稳定性收回到更核心的训练设计里。

对我们这个项目来说，这意味着：

- 现有脚手架先别急着拆
- 但也不要继续把它当长期架构
- 下一步应该从“统一信任语义”和“减少长自由 imagination 的训练主导地位”这两条线继续深挖

## 10. 参考资料

- DreamerV3 论文：[https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
- DreamerV3 项目页：[https://danijar.com/project/dreamerv3/](https://danijar.com/project/dreamerv3/)
- DreamerV3 仓库 actor loss 讨论：[https://github.com/danijar/dreamerv3/issues/20](https://github.com/danijar/dreamerv3/issues/20)
- PETS 论文：[https://arxiv.org/abs/1805.12114](https://arxiv.org/abs/1805.12114)
- MBPO 论文：[https://arxiv.org/abs/1906.08253](https://arxiv.org/abs/1906.08253)
- MOPO 论文：[https://arxiv.org/abs/2005.13239](https://arxiv.org/abs/2005.13239)
- MOReL 论文：[https://arxiv.org/abs/2005.05951](https://arxiv.org/abs/2005.05951)
- TD-MPC 论文：[https://proceedings.mlr.press/v162/hansen22a.html](https://proceedings.mlr.press/v162/hansen22a.html)
- TD-MPC 项目页：[https://td-mpc.github.io/](https://td-mpc.github.io/)
- TD-MPC2 论文：[https://arxiv.org/abs/2310.16828](https://arxiv.org/abs/2310.16828)
- TD-MPC2 仓库：[https://github.com/nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2)
- Mind the Model, Not the Agent：[https://arxiv.org/abs/2310.15017](https://arxiv.org/abs/2310.15017)
