# CartPole Pure-Imag 系统级审查与判断校验

## 审查范围

这份报告不复述既有长报告，而是做一次独立校验：

1. 回到磁盘代码，确认 world model、imagined rollout、actor/critic、controller 的真实实现链路。
2. 回到原始 run 产物，核对 `v130` 与 `v145` 在 `1200-1700` 区间的关键信号。
3. 判断当前“问题画像”里哪些已经被代码与数据双重支撑，哪些还应保留为高可信推断。

## 核验结果

### 1. “teacher-forced world model 指标正常，不代表 imagined rollout 正常”成立

这个判断被代码直接支持。

- world model 的 `loss_wm` 与 `wm/continue_prob_mean` 来自 `world_model.compute_loss(trajectory=traj)`，监督目标是 `trajectory.rewards` / `trajectory.continues`，即 teacher-forced 轨迹上的预测损失与统计，而不是 actor 真正吃到的 self-generated imagined rollout。
- 对应实现位置：
  - `loss_wm` 与 `wm/*` 记录：`aletheia/aletheia_train.py:4686-4732`
  - continue loss 的监督目标：`aletheia/aletheia_world_model.py:6189-6207`

所以把 `loss_wm` 平稳理解成“actor 看到的 imagined geometry 也稳定”是不成立的；这两条链路在实现上就是分开的。

### 2. “actor/critic 真正训练依赖 imagined rollout 的 return/base/weight 几何”成立

这个判断也被代码直接支持。

- imagined batch 是由 `_build_imagined_batch()` 通过 `imagination_engine.imagine_rollout_differentiable(...)` 构造的。
- 这里直接从 imagined rollout 取出：
  - `rewards_im`
  - `values_im`
  - `returns`
  - `continue_probs_im_raw`
- 然后继续在 imagined batch 内做：
  - continue cap
  - lambda return 重算
  - `base_actor` 构造
  - `base_return_cap`
  - tail target repair
  - `weights_actor`
  - `advantages` / `online_advantages`
- 对应实现位置：
  - imagined rollout 与 continue cap：`aletheia/aletheia_train.py:9246-9278`
  - base/return cap/online adv：`aletheia/aletheia_train.py:9285-9566`
  - tail repair / weights / metrics：`aletheia/aletheia_train.py:9622-10310`
- actor loss 侧明确取 `target_actor`、`base_actor`、`weights_actor`、`online_advantages` 做更新，而不是直接拿 teacher-forced world model loss 决定 actor 行为。
  - 对应实现位置：`aletheia/aletheia_train.py:3432-3619`

因此“真正先坏的是 actor 训练吃到的 imagined geometry，而不是 teacher-forced world model 监督项”这条主判断，和代码结构是吻合的。

### 3. “`v145` 的核心恶化早于 `persistence`，`v130` 则靠 `persistence_release` 暂时保住正向 corridor”成立

这个判断被原始 `train_metrics.jsonl` 支持，而且支撑力度很强。

#### `v145`

- `1350 @ trigger`
  - `wm_cont=0.981`
  - `imag_cont=0.679`
  - gap=`0.302`
  - `value-return=8.026`
  - `online_adv=-8.026`
  - `adv=-8.026`
  - `base_return_cap_active=0`
- `1400 @ persistence`
  - `imag_cont=0.615`
  - `value-return=10.626`
  - `online_adv=-10.626`
  - `adv=-7.274`
  - 虽然 tail repair 已开启，但 actor corridor 仍未被修回正值

#### `v130`

- `1300 @ persistence_release`
  - `wm_cont=0.984`
  - `imag_cont=0.774`
  - gap=`0.210`
  - `value-return=7.468`
  - `online_adv=-7.468`
  - `adv=+2.014`
  - `base_return_cap_active=1`
- `1350 @ persistence_release`
  - `online_adv=-7.482`
  - `adv=+1.817`
- `1400 @ persistence_release`
  - `online_adv=-4.911`
  - `adv=+1.089`
- 真正进入“同等级别坏区”要到 `1650-1700`
  - `1650`: `imag_cont=0.729`, `adv=-5.663`
  - `1700`: `value-return=8.527`, `online_adv=-8.527`

所以当前报告里最重要的句子基本成立：

- `v145` 的恶化先于 `persistence`
- `v130` 不是没坏，而是 `persistence_release + base_return_cap` 暂时保住了 actor 的正向更新走廊

### 4. “问题不在明显公式算错”目前成立，但表述需要收敛

在这次审查范围内，我没有看到足以把问题归因为“公式实现错误”的直接证据。

- lambda return 的实现是标准递推形式：`aletheia/aletheia_train.py:4985-4999`
- imagined batch 内的度量、cap、repair 与 actor loss 的连接，在实现上是连通且可解释的
- 当前坏象更像是：
  - imagined rollout state distribution 漂移
  - continue/horizon 先塌
  - critic 对 imagined bad states 下调滞后
  - controller 再去补一个已经开始失真的目标几何

但这里仍应保留边界：

- “没有发现明显公式错误”不等于“所有局部变换都最优”
- 特别是 `base_return_cap`、tail repair、analytic scale、target/base blend 这类机制，仍可能在组合上制造额外脆弱性
- 只是从当前证据看，它们更像补偿系统的一部分，而不是第一起点

### 5. “窄窗成功”判断是高可信推断，不是单条代码可直接证明的事实

这条判断和现有实验史高度一致，但它的证据类型应归为“高可信系统推断”。

成立依据：

1. `v145` 这类 run 会更早出现 imagined continue 脱锚、`value-return` 扩大、`online_adv` 深负值。
2. `v130` 这类 run 虽然也坏，但能在特定阶段里借助 `persistence_release` 把 `adv` 暂时修回正值。
3. 代码实现上，controller 的很多 patch 都是在 imagined batch 构造后对 target/base/weight/scale 做阶段性补偿，而不是修正 imagined rollout 本体。

因此“当前成功更像多个误差在小窗口里暂时抵消”是合理判断。

但更稳妥的表述应是：

- 这是当前最可信的系统级解释；
- 不是单个 metrics 点就能完全证明的数学定理。

## 需要补上的两个边界条件

### A. 失稳起点必须限定在后段主线区间

如果不加时间条件，`wm_cont - imag_cont > 0.20` 或 `imag_cont < 0.75` 在更早的预热区间也可能短暂出现。

所以当前更准确的说法应该是：

- **在 `1000+` 的主线区间内**，`v145` 首先在 `1350` 进入持续性坏区；
- `v130` 对应级别的持续性恶化要到 `1650-1700` 才出现。

这能避免把前期零散波动误写成真正的 failure onset。

### B. 当前主矛盾是“上游 imagined geometry + critic calibration”，而不是单独 controller

代码结构表明 controller 现在承担了过多补偿职责：

- 它既要调 continue cap
- 又要改 actor scale
- 又要裁 base
- 又要做 tail repair / handoff / release

这本身就是一个信号：

- 当前系统把很多上游失配压到了 controller 层解决
- 所以 patch-first 会越来越复杂，也越来越脆

## 我的最终判断

### 已被充分支撑的判断

1. 当前问题的第一起点不在 teacher-forced world model loss 爆炸。
2. actor/critic 真正更新依赖的是 imagined rollout 构造出的 `return/base/weight/advantage` 几何。
3. `v145` 的核心恶化早于 `persistence`，`1350` 就进入持续性坏区。
4. `v130` 能活下来，不是因为 imagined geometry 没坏，而是因为 `persistence_release` 暂时保住了正向 actor corridor。
5. 因此继续把主线放在 controller patch-first 上，风险很高。

### 高可信但应保留为推断的判断

1. 当前“能成功”主要是一个窄窗口对齐现象，而不是稳定宽容的机制。
2. imagined rollout 漂移是第一根因层，critic calibration 滞后是第二根因层，controller 只是第三层补偿。

## 建议的下一轮主线

1. 先做 imagined rollout 漂移的更直接观测。
   - 目标不是再看更多 stage，而是看 self-generated latent / feature 是否在 `1250-1400` 开始偏离 replay posterior 邻域。
2. 再做 critic imagined-state calibration 审计。
   - 重点确认是 value normalizer 响应慢、target 偏高，还是 critic 本体对 imagined tail 外推过乐观。
3. 最后才决定 controller 应保留哪些补偿逻辑。
   - 当前 controller 更像症状管理层，不应该继续承担根因修复主线。

## 一句话结论

你这轮新报告的主判断和我的系统级审查是基本一致的，而且核心部分已经被代码与 run 数据双重支撑：

**当前坏链路更像是 self-generated imagined rollout 先出流形，critic 又对这些 imagined bad states 下调不够快，controller 只能偶尔把它补成一个很窄的可学习走廊。**
