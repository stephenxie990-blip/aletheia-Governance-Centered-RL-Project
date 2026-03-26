# CSS-v1.5阶段性复盘与步数口径修复-2026-03-16-1922

## 一、步数统计口径不一致问题

### 现象

- 命令行传入的是 `--update-steps 5000`
- 启动日志显示 `Starting training for 5000 steps`
- 但终端训练行显示 `STEP 12000+`
- eval 行却显示 `EVAL step=1000 / 1250 / 1500 / 1750`

这会让人误以为训练器没有按 `5000` 停下，或者内部计步器已经失控。

### 根因

问题不在训练主循环，而在控制台日志口径混用：

- `EVAL step=...` 使用的是 `global_step / update_step`
- `train_metrics.jsonl` 顶层 `step` 记录的也是 `global_step / update_step`
- 但训练 `STEP ...` 这行，优先打印的是 `train/env_steps_collected`

也就是说：

- 终端 `STEP 12000`
- 实际上表示的是 `env_steps_collected = 12000`
- 并不是 `update_step = 12000`

在当前配置下：

- 每轮 collect `32` 个环境步
- 每轮做 `4` 次 train step
- 所以环境步增长速度天然会高于 update step

### 修复

已修改控制台日志格式：

- 不再把环境步伪装成统一的 `STEP`
- 改为同时打印：
  - `upd=`: update/global step
  - `env=`: collected env steps

修复位置：

- `aletheia/aletheia_api.py`

新增回归测试：

- `aletheia/tests/test_run_train_contracts.py`

回归验证：

- `py_compile` 通过
- 新增日志口径测试通过
- 相邻 `run_train` 旧测试通过

---

## 二、这轮 CSS-v1.5 到底修到了什么

### 结构上修对的部分

这轮真正实现的是：

- actor 主合同完全不动
- critic 获得独立 `target_critic`
- critic target 不再被迫与 actor 共享 `target_actor`
- 不新增新的 loss family
- 不直接放大现有 anchor 权重

这条设计边界是对的，而且已经被代码和测试钉死。

### 真实 run 表现

输出目录：

- `outputs/exp_seed42_v181_h15_css_v1_5_critic_trust_target_5000_20260316_01`

关键 eval：

- `250 = 75.4`
- `500 = 342.6`
- `750 = 23.4`
- `1000 = 47.2`
- `1250 = 252.6`
- `1500 = 26.0`
- `1750 = 96.4`

行为侧结论：

- 它不是一条纯坏线
- 模型能重新回到高 corridor
- 但 hold 不住，会在 corridor 内外反复掉摆

### 为什么说它“修到了一半”

因为这轮最关键的内部量给出的是：

前半段：

- `critic_contract_behavior_certainty = 0`
- `critic_contract_authority_mean = 0`
- `critic_contract_trust_mean = 1`
- `target_critic_delta_mean = 0`

直到后段：

- `step 1100`: `behavior_certainty = 0.436`
- `step 1100`: `authority_mean = 0.0399`
- `step 1100`: `trust_mean = 0.9601`
- `step 1100`: `target_critic_delta_mean = 1.49`

- `step 1400`: `authority_mean = 0.0913`
- `step 1400`: `trust_mean = 0.9087`
- `step 1400`: `target_critic_delta_mean = 5.32`

这说明：

1. critic 自纠偏支路最终确实接上了  
2. 但它接上得太晚  
3. 而且接上以后力度仍然太弱

---

## 三、为什么修复实效只有部分成立

### 1. 有效的地方

它证明了一件重要的事：

> “让 critic 自己在 target 构造时判断该信自己还是该信 clean anchor” 这条主线是可以接进真实训练闭环的。

这是这轮最大的正面结果。

因为此前我们只知道结构正确，不知道真实 run 里它会不会永远停留在审计层。

现在知道了：

- 它不是死路
- 后段能接入
- 接入后也确实能带来某种程度的行为恢复

`1250 = 252.6` 就是这个信号。

### 2. 失效的地方

这轮之所以没守住，不是因为“方向错”，而是因为控制律有两个问题：

#### 问题 A：触发过晚

在 critic 语义已经明显膨胀时，它仍未触发：

- `step 700`: `real_mc_gap_abs = 7.23`
- `step 800`: `13.25`
- `step 900`: `24.17`
- `step 1000`: `35.59`

但这整个窗口里：

- `behavior_certainty = 0`
- `authority = 0`
- `target_critic_delta = 0`

也就是说：

> 语义已经坏得很明显了，但 critic trust branch 还在完全信自己。

#### 问题 B：接管太弱

即使后面触发了：

- `authority` 仍只在 `0.04 ~ 0.09`
- `trust` 仍在 `0.91 ~ 0.96`
- `target_critic_delta` 只有 `1.5 ~ 5.3`

而同时：

- `values_mean` 已经在 `67 -> 94`
- `returns_mean` 只有 `42 -> 54`
- `real_mc_gap_abs` 上升到 `47 -> 70`
- `imag value abs to short return` 上升到 `66 -> 90`

所以它的净效果是：

> clean critic target 确实开始说话了，但音量远小于膨胀 critic 的自举音量。

---

## 四、为什么这轮会出现“行为恢复了，但语义还是脏的”

这是本轮最值得保留的诊断结论。

在 `step 1250` 附近：

- `eval = 252.6`
- `occupancy = 0.917`
- `persistence = 0.914`
- `registry KL = 0.024`

说明行为层重新回到了高回报 corridor。

但同时：

- `real_mc_gap_abs = 59.97`
- `teacher gap = 68.43`
- `imag gap = 79.84`

说明 critic 语义并没有被真正洗干净。

这意味着：

> 当前系统可以靠 actor-side 已有约束和局部 critic target 修正，把行为重新拉回 corridor；但 critic 语义标尺本身仍在持续膨胀。

这就是典型的：

- 行为暂时恢复
- 语义未真正守恒
- 所以后面仍会反复掉摆

---

## 五、当前最准确的阶段性诊断

### 不是

- 方向全错
- actor 又变成主病灶
- bridge / Iron Wall 又回到第一病位

### 而是

当前问题已经收敛成：

> `CSS-v1.5` 的 critic 自举纠偏方向是对的，但它的触发合同把 `behavior_certainty` 卡得太死，导致介入窗口太晚；而介入后 authority 量级又太小，无法追上 critic inflation 的增长速度。  

更短一点说：

> 这刀不是打偏了，是打到了，但切口太晚、太浅。

---

## 六、下一步该怎么调

### 原则

- actor 继续冻结
- 不回去做“外部硬清洗 critic”
- 继续保留 “critic 自己知道该信谁” 这个优雅主线

### 调整方向

#### 1. 把 `behavior_certainty` 从生死门，降成软调制项

现在最大问题是：

- corridor 认证没抬起来之前
- critic trust target 完全不能接管

这会把介入时间拖到语义已经严重膨胀以后。

应改成：

- `semantic pressure` / `inflation pressure` 先主导触发
- `behavior_certainty` 只负责增强或减弱，而不是决定 0/1 生死

#### 2. 给 critic target takeover 一个非零底座

当前 `authority` 从 `0` 到 `0.04` 到 `0.09` 太慢。

应改成：

- 只要 semantic pressure 明显存在
- critic clean target 就应获得一个基础接管份额
- 再由 behavior certainty 和 degradation 决定是否进一步增强

#### 3. 触发窗口前移到 pre-collapse 区间

本轮已经知道真正该介入的窗口不是 `gap_abs 46+` 以后，而是：

- `7 -> 13 -> 24 -> 35`

这才是能救尾段的时间带。

#### 4. 保持 actor 不动，继续只修 critic target law

这条工程纪律仍然成立。

因为这轮失败不是 actor 主合同再次失控，而是 critic trust bootstrap 接管太晚太弱。

---

## 七、一句话结论

> 这轮 CSS-v1.5 证明了“critic 内部 trust-aware target”这条路线是可行的；但当前版本把 `behavior_certainty` 设成了过强的启动前提，导致自纠偏支路在 critic 语义已经严重膨胀后才微弱介入。下一刀不该推翻方向，而应该把它改成“更早触发 + 持续接管 + behavior 只做软调制”的 critic target law。  
