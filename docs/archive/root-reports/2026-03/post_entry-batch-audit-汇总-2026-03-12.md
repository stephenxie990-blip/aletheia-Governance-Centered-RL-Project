# post_entry batch audit 汇总（2026-03-12）

项目路径：`/Users/zhangsan/Desktop/缸中之脑v5.6`

## 1. 目标

本轮完成两个目标：

1. 用新的 `post_entry` 审计脚本批量审计更多 fresh run，确认 solved run 中 `post_entry` 的活跃窗口和 guard 模式是否稳定复现。
2. 在不改变训练决策逻辑的前提下，把 `post_entry` 审计摘要自动并入训练产物，形成自动报告。

## 2. 已完成的工程改动

### 2.1 审计 helper

新增可复用模块：

- `aletheia/post_entry_audit.py`

提供能力：

- 读取 `train_metrics.jsonl`
- 提取 stage counter / transition points / active segments
- 汇总 `post_entry / post_entry_soft / post_entry_pending / persistence / post_solved`
- 生成 JSON 摘要、Markdown 报告、compact digest

### 2.2 单 run 审计脚本

- `scripts/post_entry_audit.py`

### 2.3 batch 审计脚本

- `scripts/post_entry_batch_audit.py`

### 2.4 训练产物自动报告集成

已在训练收尾阶段自动生成：

- `post_entry_audit.json`
- `post_entry_audit.md`
- `summary.json` 中的 `artifacts.post_entry_audit_json`
- `summary.json` 中的 `artifacts.post_entry_audit_md`
- `summary.json` 中的 `post_entry_audit.generated`
- `summary.json` 中的 `post_entry_audit.summary`

关键点：

- 该集成只发生在训练结束后的产物整理阶段
- 不改变训练采样、world model、actor、critic、controller stage 的任何决策逻辑

## 3. fresh run 结果

### 3.1 solved 正样本组

批量审计对象：

- `outputs/exp_seed42_v87tailfix_modefix_1500_20260312`
- `outputs/exp_seed42_v87tailfix_modefix_autoaudit_r1_1500_20260312`
- `outputs/exp_seed42_v87tailfix_modefix_autoaudit_r2_1500_20260312`

batch 结果文件：

- `outputs/post_entry_batch_audit_solved_20260312.json`
- `outputs/post_entry_batch_audit_solved_20260312.md`

核心结论：

- `num_runs = 3`
- `runs_with_post_entry = 3`
- `runs_with_post_entry_soft = 3`
- `stable_transition_sequence = True`
- `stable_post_entry_window = True`
- `stable_post_entry_soft_window = True`
- `stable_post_entry_soft_guard_scale = True`
- `stable_post_entry_soft_critic_multiplier = True`

稳定复现的 solved 模式：

- transition sequence: `idle -> post_entry -> post_entry_soft -> trigger`
- `post_entry`: `800 -> 1000 (5 rows)`
- `post_entry_soft`: `1050 -> 1200 (4 rows)`
- `post_entry_soft guard_scale`: `0.75`
- `post_entry_soft critic_multiplier`: `2.0`
- `best_eval_step`: `1250`
- `best_eval_mean`: `500.0`

结论：

- 在 solved 主线中，`post_entry` 活跃窗口和 `post_entry_soft` guard 模式是稳定复现的，不是一次性偶然现象。

### 3.2 非 solved 对照组

批量审计对象：

- `outputs/exp_seed43_v87tailfix_modefix_autoaudit_1500_20260312`
- `outputs/exp_seed44_v87tailfix_modefix_autoaudit_1500_20260312`
- `outputs/exp_seed45_v87tailfix_modefix_autoaudit_1500_20260312`

batch 结果文件：

- `outputs/post_entry_batch_audit_unsolved_20260312.json`
- `outputs/post_entry_batch_audit_unsolved_20260312.md`

核心结论：

- `num_runs = 3`
- `runs_with_post_entry = 1`
- `runs_with_post_entry_soft = 3`
- `stable_transition_sequence = False`
- `stable_post_entry_soft_window = False`
- `stable_post_entry_soft_guard_scale = False`
- `stable_post_entry_soft_critic_multiplier = False`

典型差异：

- `seed43`: `idle -> post_entry_soft -> post_entry_pending`
- `seed44`: `idle -> trigger -> idle -> trigger -> post_entry -> post_entry_soft -> persistence`
- `seed45`: `idle -> trigger -> post_entry_soft -> post_entry_pending`

说明：

- 非 solved run 并没有稳定进入 solved 正样本组那种 `post_entry(800-1000) -> post_entry_soft(1050-1200) -> trigger` 模式。
- 它们更容易提前进入 `post_entry_soft/pending/persistence` 等分支，且 guard 强度和时间窗口不稳定。

## 4. 自动报告集成验证

以新的 fresh solved run 为例：

- `outputs/exp_seed42_v87tailfix_modefix_autoaudit_r1_1500_20260312/summary.json`
- `outputs/exp_seed42_v87tailfix_modefix_autoaudit_r2_1500_20260312/summary.json`

都已自动包含：

- `post_entry_audit.generated = true`
- `artifacts.post_entry_audit_json`
- `artifacts.post_entry_audit_md`
- compact digest，例如：
  - transition: `50 idle -> 800 post_entry -> 1050 post_entry_soft -> 1250 trigger`
  - `post_entry_soft num_rows = 4`
  - `negative_adv_rows = 4`
  - `guard_scale_range = 0.75`
  - `critic_multiplier_range = 2.0`

这证明自动报告链已经打通。

## 5. 当前结论

到本轮为止，可以确认两件事：

1. solved run 的 `post_entry` 窗口和 guard 模式可以稳定复现。
2. 当前基线跨 seed 仍然存在明显学习方差；`43/44/45` 没有 solved，而且其 `post_entry` 状态机轨迹与 solved 正样本组并不同构。

因此，`post_entry` 现在不再是“不可观测的黑盒”。
它已经具备：

- 自动产物
- 批量对账能力
- solved / unsolved 的结构化对照

接下来如果继续做优化，应该优先研究：

- 为什么一些 seed 会偏离 solved 正样本组的阶段序列
- 是 external eval 触发时机不同，还是 imagined advantage/gap 统计在中段开始分叉
- `post_entry_pending / persistence` 是否是结果，而不是原因

## 6. solved template 对账：最早分叉点

已新增 solved-template 对账脚本：

- `scripts/post_entry_template_compare.py`

对应结果文件：

- `outputs/post_entry_template_compare_800_1250_20260312.json`
- `outputs/post_entry_template_compare_800_1250_20260312.md`

template 使用：

- `exp_seed42_v87tailfix_modefix_1500_20260312`
- `exp_seed42_v87tailfix_modefix_autoaudit_r1_1500_20260312`
- `exp_seed42_v87tailfix_modefix_autoaudit_r2_1500_20260312`

candidate 使用：

- `exp_seed43_v87tailfix_modefix_autoaudit_1500_20260312`
- `exp_seed44_v87tailfix_modefix_autoaudit_1500_20260312`
- `exp_seed45_v87tailfix_modefix_autoaudit_1500_20260312`

template 的标准 solved 序列是：

- `800-1000`: `post_entry`
- `1050-1200`: `post_entry_soft`
- `1250`: `trigger`

自动对账结论：三个非 solved candidate 的最早分叉点都出现在 `step 800`。

### seed43

- first stage mismatch: `800`
- expected: `post_entry`
- actual: `post_entry_soft`
- 同步出现的指标偏离：
  - `raw_adv_mean`: `-0.777 -> +1.312`
  - `continue_cap_dynamic`: `0.900 -> 0.930`
  - `post_trigger_scale`: `0.920 -> 0.970`

### seed44

- first stage mismatch: `800`
- expected: `post_entry`
- actual: `idle`
- 同步出现的指标偏离：
  - `raw_adv_mean`: `-0.777 -> +0.241`
  - `continue_cap_dynamic`: `0.900 -> 0.950`
  - `post_trigger_scale`: `0.920 -> 1.000`
  - `post_entry_commit_return_ok`: `1.0 -> 0.0`

### seed45

- first stage mismatch: `800`
- expected: `post_entry`
- actual: `post_entry_soft`
- 同步出现的指标偏离：
  - `raw_adv_mean`: `-0.777 -> +0.249`
  - `continue_cap_dynamic`: `0.900 -> 0.930`
  - `post_trigger_scale`: `0.920 -> 0.970`
  - `value_target_gap_abs_mean`: `1.979 -> 0.900`

### 结论升级

这说明非 solved run 的问题不是到了 `post_entry_soft` 后才出现，而是在 `step 800` 左右就已经偏离 solved 模板：

- stage 先分叉
- 同时伴随 `continue_cap_dynamic / post_trigger_scale / raw_adv_mean` 的同步偏离
- 后续进入 `post_entry_pending / persistence` 更像是这个早期偏离的结果，而不是起因

## 7. 前兆窗口对账（500-800）

已执行更早窗口的 solved-template 对账：

- `outputs/post_entry_template_compare_500_800_20260312.json`
- `outputs/post_entry_template_compare_500_800_20260312.md`

template 仍使用 solved 正样本组 3 个 run。

结果表明：

- `seed43`
  - `last_fully_aligned_step = 750`
  - `first_metric_divergence = 600`
  - `first_stage_mismatch = 800`
  - 最早偏离的指标：
    - `raw_adv_mean: 1.974 -> 3.366`
    - `value_target_gap_abs_mean: 2.183 -> 3.447`

- `seed44`
  - `last_fully_aligned_step = 750`
  - `first_metric_divergence = 550`
  - `first_stage_mismatch = 800`
  - 最早偏离的指标：
    - `raw_adv_mean: 3.017 -> 1.237`
    - `value_target_gap_abs_mean: 3.146 -> 1.836`

- `seed45`
  - `last_fully_aligned_step = 700`
  - `first_metric_divergence = 550`
  - `first_stage_mismatch = 750`
  - 最早偏离的指标：
    - `raw_adv_mean: 3.017 -> 0.972`
    - `value_target_gap_abs_mean: 3.146 -> 2.078`

这把问题进一步收窄了：

1. stage 分叉不是最早事件。
2. 最早的可观测偏离通常先出现在 `550-600` 的 imagined 优势 / target-gap 统计量上。
3. `750-800` 才出现 stage 级别的显性分叉：
   - 提前进入 `trigger`
   - 或直接进入 `post_entry_soft`
   - 或在该进该退时仍停留 `idle`

因此，后续如果继续做根因定位，优先级应当从 `post_entry` 本体再往前提一段，聚焦：

- 为什么在 `550-600` 区间，不同 seed 的 imagined `raw_adv_mean` 和 `value_target_gap_abs_mean` 会开始分化
- 这种分化如何在后续传导到 `continue_cap_dynamic / post_trigger_scale / controller_stage`

## 8. 前兆链升级审计（400-800）

已新增前兆链对账工具：

- `aletheia/post_entry_precursor_audit.py`
- `scripts/post_entry_precursor_audit.py`
- `aletheia/tests/test_post_entry_precursor_audit.py`

对应产物：

- `outputs/post_entry_precursor_audit_400_800_20260312.json`
- `outputs/post_entry_precursor_audit_400_800_20260312.md`

这轮审计相比前面的 template compare，多拆了一层：

- `continue_chain`: `continue_prob_mean_raw / continue_prob_mean / effective_horizon`
- `value_chain`: `reward / return / value / target_value / gap`
- `actor_bridge`: `target / base / online_adv / raw_adv`
- `controller_chain`: `continue_cap_dynamic / post_trigger_scale / commit_ready`

同时为每个 step 计算了：

- `return/value` 的相对模板偏移
- `target/base` 对 `raw_adv` 的分解驱动
- `same_direction_masked / split_direction_reinforcing` 等模式标签

### 8.1 统一结论：最早的公共前兆不是 stage，也不是 raw_adv，而是 continue/horizon

三个非 solved seed 的最早 group divergence 全部落在：

- `continue_chain`
- `step = 400`
- first metric: `imag/continue_prob_mean_raw`

这说明：

1. `post_entry` 不是最早事件。
2. 仅看 `raw_adv_mean / value_target_gap_abs_mean` 仍然太晚。
3. 真正更上游的统一前兆是 imagined `continue` 偏高、`effective_horizon` 偏长。

也就是说，非 solved seed 在 `400` 左右就已经开始“想得太长”。
只是这个偏差在不同 seed 上，后续传导方式并不一样。

### 8.2 seed43 / seed44：split-direction reinforcing

`seed43` 和 `seed44` 在 `450` 附近进入同一种模式：

- `return_mean` 相对 solved 模板上移
- `value_mean` 相对 solved 模板下移
- 二者对 `raw_adv` 形成同向放大
- `raw_adv` 因此很早穿阈值

典型表现：

- `seed43 @ 450`
  - `d_cont_raw = +0.178`
  - `d_horizon = +0.490`
  - `d_return = +0.842`
  - `d_value = -0.208`
  - `d_raw_adv = +1.050`
  - pattern: `split_direction_reinforcing`
  - adv_driver: `return_dominant_reinforcing`

- `seed44 @ 450`
  - `d_cont_raw = +0.137`
  - `d_horizon = +0.618`
  - `d_return = +0.730`
  - `d_value = -0.879`
  - `d_raw_adv = +1.609`
  - pattern: `split_direction_reinforcing`
  - adv_driver: `mixed_reinforcing`

这类失败形态的特征是：

- imagined horizon 先变长
- 之后 `return` 与 `value` 发生反向偏移
- 导致 advantage 被很快放大
- 最终在 `800` 才表现为 `post_entry/post_entry_soft/idle` 的 stage 分叉

### 8.3 seed45：same-direction masked -> later unbalanced

`seed45` 是另一种更隐蔽的失败形态：

- 从 `400` 开始，`return_mean` 和 `value_mean` 都明显高于 solved 模板
- 但两者方向相同，前期会互相抵消一部分 `raw_adv` 偏差
- 所以前面的 `raw_adv/gap_abs` 审计并没有第一时间把它显出来

典型表现：

- `seed45 @ 400`
  - `d_cont_raw = +0.155`
  - `d_horizon = +0.758`
  - `d_return = +2.447`
  - `d_value = +2.297`
  - `d_raw_adv = +0.149`
  - pattern: `same_direction_masked`

- `seed45 @ 450`
  - `d_return = +2.253`
  - `d_value = +1.789`
  - `d_raw_adv = +0.464`
  - 仍然是 `same_direction_masked`

- 到 `550` 才转成：
  - `d_return = +2.630`
  - `d_value = +4.675`
  - `d_raw_adv = -2.045`
  - pattern: `same_direction_unbalanced`
  - adv_driver: `value_dominant_cancelling`

这解释了为什么 `seed45` 会出现：

- 前期看起来不像是 raw advantage 爆掉
- 但后面会突然在 `750` 进入 `trigger`
- 然后在 `800` 落到 `post_entry_soft`

因此，`seed45` 的问题不是“没有前兆”，而是前兆被 `return/value` 的同向抬升掩盖了。

### 8.4 对项目链路的含义

这轮前兆链审计把因果方向进一步固定了：

1. 一级前兆在 imagined continue / horizon 校准层。
2. `value/return` 的分叉是第二层传导。
3. `raw_adv / negative_adv_guard` 是第三层显性信号。
4. `post_entry/post_entry_soft/persistence` 是更后面的状态机后果。

换句话说：

- `post_entry` 不是当前主线的一级根因
- 继续只在 `post_entry` 上加 guard，会继续治标不治本
- 真正要修的，是 pure-imag 早期 imagined horizon 偏长、以及它如何把 `return/value` 推向两种不同失稳模式

### 8.5 下一步修复重点

基于当前证据，后续优先级应调整为：

1. 审计 `continue head` / lambda-return 上游，确认为什么 unsolved seed 会在 `400` 左右统一出现 `continue_prob_mean_raw` 偏高。
2. 审计 early pure-imag rollout 的 horizon 约束是否过松，为什么 `continue_cap_dynamic` 仍停在高位，直到 `750-800` 才开始补救。
3. 把 `seed43/44` 的 `split-direction reinforcing` 和 `seed45` 的 `same-direction masked` 分成两个 failure mode，不再用单一 `raw_adv` 指标概括所有失败。
4. 后续若做训练修复，优先落在 imagined continue / effective horizon 的前段校准，而不是继续加厚 `post_entry` 下游 guard。

## 9. continue/horizon 上游专项审计

已新增专项脚本：

- `scripts/imag_continue_chain_audit.py`

对应产物：

- `outputs/imag_continue_chain_audit_20260312.json`
- `outputs/imag_continue_chain_audit_20260312.md`

这个专项不是再看 `post_entry` 状态机结果，而是直接审计：

- `continue_prob_mean_raw`
- `effective_horizon`
- `continue_cap_pressure`
- `continue_prob_cap_dynamic`
- `trigger_raw_active`
- checkpoint 中 `continue_head` 最后一层 bias
- trainer_state 中 replay buffer 的真实 `continue_target_mean`

### 9.1 配置层一级问题：controller 存在硬盲区

当前 solved 基线真实配置：

- `imag_continue_prob_cap = 0.95`
- `adaptive_imag_continue_cap = true`
- `adaptive_imag_continue_cap_warmup_steps = 700`
- `adaptive_imag_continue_cap_ramp_steps = 300`
- `adaptive_imag_continue_cap_target_continue = 0.97`
- `adaptive_imag_post_trigger_min_pressure = 0.01`

这意味着：

1. `step < 700` 时，adaptive scale 固定为 `0`。
2. 即使 theoretical pressure 已经超过 trigger 阈值，实际 `scaled_pressure` 也会被硬压成 `0`。
3. 最早公共前兆在 `step 400`，但 continue controller 到 `step 750` 才第一次真正“看见”问题。

这是明确的时间错位。

### 9.2 真实对账：多个 run 在 controller 可见之前就已经“理论可触发”

专项审计结果：

- `solved`
  - `first_triggerable = 550`
  - `first_adaptive_visible = 750`
  - `first_dynamic_cap = 750`

- `seed43`
  - `first_triggerable = 600`
  - `first_adaptive_visible = 750`
  - `first_dynamic_cap = 750`
  - `first_trigger_raw = 800`

- `seed44`
  - `first_triggerable = None`
  - `first_adaptive_visible = 750`
  - `first_dynamic_cap = 750`

- `seed45`
  - `first_triggerable = 400`
  - `first_adaptive_visible = 750`
  - `first_dynamic_cap = 750`
  - `first_trigger_raw = 750`

最关键的是 `seed45`：

- 在 `step 400`，理论 pressure 已经 `0.0128 > 0.01`
- 但由于 warmup，`adaptive_scale = 0`
- 所以实际 `scaled_pressure = 0`
- `dynamic_cap` 仍保持 `0.95`
- controller 完全没有动作

这说明对于 `seed45` 这种 failure mode，不是 controller 力度不够，而是 controller 启动时机晚了 `350` step 左右。

### 9.3 model 层二级问题：continue head 保持强乐观初始化，几乎未被学掉

checkpoint 审计结果显示，所有 run 的 continue head 最后一层 bias 几乎始终维持在初始化值附近：

- solved: `4.9994 -> 4.9999`
- seed43: `4.9991 -> 4.9988`
- seed44: `4.9985 -> 4.9965`
- seed45: `4.9981 -> 4.9980`

而 `ContinueHead` 的初始化明确是：

- optimistic bias = `5.0`
- `sigmoid(5.0) ≈ 0.993`

这说明：

1. continue head 的全局乐观先验非常强。
2. 训练过程中这个先验几乎没有被真正校正掉。
3. 早期 imagined continue 偏高，不只是 controller 的问题，head 本身也在持续提供高 continuation prior。

### 9.4 loss 层三级问题：continue loss 没有类不平衡修正

代码审计结果：

- `compute_continue_loss` 使用普通 BCE
- 没有 `pos_weight` / class weighting
- 也没有针对 terminal 稀疏样本的专门校正

而 replay buffer 真实统计显示：

- solved `trainer_step500`: `continue_target_mean = 0.9180`
- seed43 `trainer_step500`: `0.9188`
- seed44 `trainer_step500`: `0.9221`
- seed45 `trainer_step500`: `0.9320`

换句话说：

- 真实 continue target 本来就是正类占优
- done 比例只有 `~6.8% - 8.2%`
- 在这种类不平衡下，再叠加 optimistic bias=5.0
- continue head 极容易长期保持“过于乐观”

因此，continue head 当前更像是：

- 强正先验
- 弱负样本约束
- 迟启动下游 controller

三者叠加后，早期 pure-imag horizon 校准天然偏松。

### 9.5 分层结论

当前可以把 continue/horizon 根因拆成三层：

1. 配置/调度问题：controller warmup/ramp 与真实前兆窗口严重错位。
2. 模型问题：continue head 的 optimistic bias 过强且几乎未被学掉。
3. 损失问题：continue BCE 对稀疏 terminal 样本没有做任何平衡处理。

其中：

- `seed45` 更像“早期理论可触发，但被 warmup 硬压住”的失败。
- `seed43` 更像“到 600 已经可触发，但等到 800 才进入显性 trigger/post_entry_soft”的失败。
- `seed44` 说明不是所有失败都来自同一种 continue 过高模式，它更多是后续 `value/return` 分叉，但这不否定 continue blind window 的系统性问题。

### 9.6 修复方向已经收敛

基于当前证据，后续修复不应再优先放在 `post_entry` 下游，而应优先落在以下三条主线：

1. 缩短或分段重设 continue controller warmup，让 `400-600` 的真实前兆窗口进入可控范围。
2. 削弱 continue head 的 optimistic bias，或让 bias 在早期更快退火。
3. 给 continue loss 增加 terminal class balancing / hard-negative 强化，提升对 done 稀疏样本的校准能力。

这是目前最接近一级根因的纯 imag 修复方向。
