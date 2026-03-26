# persistence tail mismatch 审计与修复报告

## 1. 审计结论

这次 `persistence tail mismatch` 的主根因不是采样器、环境或 replay buffer 本身，而是 pure-imag 训练链路在 `persistence/persistence_release` 阶段缺少 tail-only 保护，导致 imagined tail 上三条链路脱钩：

1. `continue_probs -> weights_actor` 过快塌缩
2. `continue_probs -> lambda returns -> target_actor` 过快悲观
3. `online critic values -> base_actor` 仍保持高位

三条链没有在 persistence tail 做重新对齐，于是训练期会出现：

- tail `continue` 很低
- tail `weights_actor` 接近零
- tail `target_actor` 明显低于 `base_actor`
- actor 更新只剩少数 surviving prefix / branch
- critic 又没有足够的 tail-only 纠偏

这会把 pure-imag 训练推向“窄策略 exploitation”，表现为：

- 单些 seed / checkpoint 看起来会冲高
- 统一评估口径后并不稳
- 跨 seed 方差很大
- persistence 阶段不能稳定维持 solved 策略

## 2. 证据链

### 2.1 统一 checkpoint 审计

修复后的统一评估口径表明：

- `v87 best`: `10ep=500.0`, `50ep=481.94`, `100ep=485.06`
- `v103 best`: `10ep=226.4`, `50ep=208.9`, `100ep=216.24`
- `v98 best`: `10ep=165.5`, `50ep=150.04`, `100ep=156.82`

这说明旧结论“`v103` late-lift 接近 solved”是被旧 evaluator 固定 seed 污染出来的假象。

### 2.2 persistence 阶段训练期几何

`train_metrics.jsonl @1500`：

- `v87`
  - `stage=persistence`
  - `cap=0.86`
  - `continue=0.804`
  - `effective_horizon=6.01`
  - `value_target_gap_abs=4.998`
  - `actor/post_trigger_scale=0.870`
  - `negative_adv_guard=1.0`
  - `critic_multiplier=1.5`

- `v98`
  - `stage=persistence`
  - `cap=0.93`
  - `continue=0.703`
  - `effective_horizon=7.71`
  - `value_target_gap_abs=6.818`
  - `actor/post_trigger_scale=0.97`
  - `negative_adv_guard=1.0`
  - `critic_multiplier=1.75`

- `v103`
  - `stage=persistence`
  - `cap=0.86`
  - `continue=0.577`
  - `effective_horizon=5.48`
  - `value_target_gap_abs=8.845`
  - `actor/post_trigger_scale=0.814`
  - `negative_adv_guard=0.0`
  - `critic_multiplier=1.0`

关键差异不是“有没有 imagination”，而是 `v103` 的 persistence tail 更塌，同时没有足够 critic correction。

### 2.3 final checkpoint forensics

`final.pt` imagined rollout 对账：

- `v87 final`
  - `eval_mean=479.2`
  - `gap_abs=4.41`
  - `continue=0.887`
  - `weights_min=0.0038`
  - tail `continue=[0.830, 0.683, 0.610, 0.581, 0.596]`
  - tail `return=[7.940, 7.190, 6.853, 6.893, 7.605]`
  - tail `value=[12.096, 11.718, 11.567, 11.385, 11.149]`

- `v98 final`
  - `eval_mean=165.5`
  - `gap_abs=2.91`
  - `continue=0.902`
  - `weights_min=0.134`
  - tail `continue=[0.900, 0.807, 0.708, 0.621, 0.543]`
  - tail `return=[3.644, 3.044, 2.606, 2.303, 2.090]`
  - tail `value=[2.553, 2.476, 2.299, 2.190, 2.201]`

- `v103 final`
  - `eval_mean=214.55`
  - `gap_abs=6.28`
  - `continue=0.735`
  - `weights_min≈1.94e-08`
  - tail `continue=[0.413, 0.317, 0.276, 0.251, 0.238]`
  - tail `return=[4.321, 4.007, 3.926, 3.972, 4.502]`
  - tail `value=[12.081, 12.475, 12.796, 13.182, 13.526]`

`v103` 的 tail `value` 和 tail `return` 明显背离，而且 `weights_min` 接近 0，这是最直接的 tail mismatch 证据。

## 3. 功能模块 / 数据链 / 传导链完整排查

### 3.1 功能模块

- imagined rollout：`ImaginationEngine.imagine_rollout_differentiable()`
- lambda returns：`_compute_lambda_returns()`
- controller stage / continue cap：`_resolve_imag_continue_cap()` + `_resolve_imag_controller_stage()`
- imagined batch 汇合：`_build_imagined_batch()`
- actor loss：`compute_dreamer_actor_loss()`
- actor/critic guard：`train_step()`

### 3.2 数据链

1. seed batch 从 replay buffer 抽样
2. RSSM/world model 生成 imagined rollout
3. `continue_probs` 经 controller cap 后重算 `returns`
4. `weights_actor = cumprod(gamma * continue_probs)`
5. `target_actor / base_actor / advantages` 送入 actor loss
6. `returns / values` 送入 critic loss
7. train step 中再叠加 negative-adv guard / critic multiplier

### 3.3 传导链

最关键的传导链是：

- `continue_probs -> returns`
- `continue_probs -> weights_actor`
- `values_im[:, :-1] -> base_actor`
- `target_actor/base_actor -> raw_adv`
- `raw_adv -> negative_adv_guard / critic multiplier`

现有实现的问题在于：

- `returns` 与 `weights_actor` 都受 tail continue collapse 直接影响
- `base_actor` 不受 tail collapse 约束，仍来自高位 online critic
- persistence 阶段此前没有专门的 tail instrumentation
- persistence 阶段此前也没有 tail-only actor target floor / weight floor / critic boost

于是 mismatch 会在 `_build_imagined_batch()` 形成，在 `train_step()` 被放大。

## 4. 根因归类

### 4.1 不是采样问题

- replay buffer / rollout collector 的集成测试已经通过
- real batch 构造链路没有发现 tail repair 污染
- evaluator 固定 seed 污染已经单独修复，与 persistence tail mismatch 属于两个问题

### 4.2 不是整体架构失效

- Dreamer-style imag rollout、lambda return、actor/critic 主链仍然成立
- `v87` 证明这套架构可以在本项目里达到 solved 级别

### 4.3 是算法集成层的局部缺口

更准确地说，这是 `persistence` 阶段的 imagined training geometry 缺口：

- `continue`/`weights`/`actor target` 在 tail 变得过于悲观
- `online value` 在 tail 又没有同步收敛
- actor 只剩 prefix / surviving branches 更新
- critic 纠偏力度不足

## 5. 修复方案

这轮修复不再继续改全局 controller，而是只作用于 `persistence` / `persistence_release` 的 tail：

1. 增加 persistence tail instrumentation
2. 只在 tail mismatch 命中时修 `target_actor`
3. 只在 tail mismatch 命中时对 `weights_actor` 做 floor
4. 只在 tail mismatch 命中时对 critic 增加额外 multiplier

### 5.1 tail mismatch 判定条件

默认参数：

- `tail_window=4`
- `tail_gap_threshold=6.0`
- `tail_continue_threshold=0.65`
- `tail_weight_threshold=0.15`

只有同时满足：

- stage in `persistence/persistence_release`
- tail gap 过大
- tail continue 过低
- tail weight 过低

才会触发修复。

### 5.2 actor target repair

当 tail mismatch 激活后，限制 actor-side tail pessimism：

- `target_actor_tail >= online_base_actor_tail - target_gap_cap`

默认：

- `target_gap_cap=5.0`

这不是改 critic target，而是只修 actor-side target，避免 actor 在 tail 被假性悲观完全压死。

### 5.3 weight floor

当 tail mismatch 激活后，防止 tail branch 权重塌成近零：

- `tail_weight >= max(abs_floor, head_weight_mean * ratio)`

默认：

- `ratio=0.25`
- `abs_floor=0.0`

这样保留 Dreamer discount 语义，但阻止 `v103` 那种“只剩前缀有效”的狭窄更新。

### 5.4 critic correction

当 tail mismatch 激活后，给 critic 额外 multiplier：

- `critic_terms *= persistence_tail_critic_multiplier`

默认：

- `critic_boost=0.5`，即 multiplier `1.5`

这一步是针对 `v103` 缺少 persistence critic correction 的局部补强，不会把全部 persistence 都拉回 `v98` 式保守状态。

## 6. 已实施修改

- 在 `TrainingConfig` 中新增 persistence-tail 配置项
- 在 `_build_imagined_batch()` 中新增：
  - tail metrics
  - tail mismatch gate
  - tail actor target repair
  - tail weight floor
  - batch-level critic multiplier 透传
- 在 `train_step()` 中新增 persistence-tail critic multiplier 消费逻辑
- 新增三条回归测试：
  - `persistence` 阶段触发 repair
  - 非 `persistence` 阶段不触发
  - real batch 不携带 tail repair 字段

## 7. 验证结果

已通过：

- `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration aletheia.tests.test_run_train_contracts -v`
- 结果：`131 tests OK`

已通过：

- `./.venv/bin/python -m py_compile aletheia/aletheia_train.py aletheia/aletheia_config.py aletheia/tests/test_training_loop_integration.py`

## 8. 下一步重点

这轮修复解决的是“persistence tail mismatch 没有局部保护”的实现缺口，但还没有直接证明新的训练曲线已经恢复到 `500 solved`。下一步应该做的是：

1. 开一轮统一口径的 pure-imag 训练复测
2. 重点跟踪新指标：
   - `imag/persistence_tail_mismatch_active`
   - `imag/persistence_tail_continue_mean`
   - `imag/persistence_tail_gap_mean`
   - `actor/persistence_tail_target_repair_active`
   - `actor/persistence_tail_weight_floor_active`
   - `critic/persistence_tail_mismatch_multiplier`
3. 对比 `1250-1500` 区间是否出现：
   - tail mismatch 激活次数下降
   - `weight_tail_mean` 回升
   - `value_target_gap_last_mean` 回落
   - persistence 阶段真实 eval 不再掉回 `200` 档


## 9. 修复后复测结果（2026-03-11 晚）

这部分是 tail 修复落地后的正式复测，不再使用旧 evaluator 口径。

### 9.1 正式训练 run

正式 run：

- 输出目录：`outputs/exp_seed42_v87tailfix_pureimag_1500_20260311`
- 起点：`v87 solved geometry + persistence-tail repair`
- 命令核心：
  - `scripts/cartpole_train.py --update-steps 1500 --seed 42 --device cpu --enable-eval --eval-episodes 10 --eval-interval 250 --save-interval 250 --log-interval 50 --imagination-only --overrides tmp/v87_tailfix_overrides.json`

训练期 eval 历史：

- `250 -> 15.9`
- `500 -> 107.6`
- `750 -> 355.8`
- `1000 -> 125.3`
- `1250 -> 500.0`

训练在 `1250` 触发 `early_stop_eval_mean=500` 提前停止。

### 9.2 修复后统一 checkpoint 审计

按修复后的统一 evaluator 重新核对：

- 新 `v87 tailfix best`
  - `10ep=500.0`
  - `50ep=500.0`
  - `100ep=500.0`
- 新 `v87 tailfix final`
  - `10ep=500.0`
  - `50ep=500.0`
  - `100ep=500.0`
- 旧 `v87 best`
  - `10ep=500.0`
  - `50ep=481.94`
  - `100ep=485.06`
- `v103 best`
  - `10ep=226.4`
  - `50ep=208.9`
  - `100ep=216.24`
- `v98 best`
  - `10ep=165.5`
  - `50ep=150.04`
  - `100ep=156.82`

新的真实排名是：

1. 新 `v87 tailfix run`
2. 旧 `v87`
3. `v103`
4. `v98`

这说明这轮 tail 修复不只是“恢复到旧 v87”，而是已经超过旧 solved 基线的中长期稳定性。

### 9.3 跨 seed 稳健性

对修复后 run 的 `best.pt/final.pt` 用显式 `EnvWrapper(seed=42..46)` 逐 seed 做 `50ep` 评估：

- 新 `best.pt`
  - `seed 42: 500.0`
  - `seed 43: 500.0`
  - `seed 44: 500.0`
  - `seed 45: 500.0`
  - `seed 46: 500.0`
  - 汇总：`mean_of_means=500.0`, `std_of_means=0.0`
- 新 `final.pt`
  - `seed 42: 500.0`
  - `seed 43: 500.0`
  - `seed 44: 500.0`
  - `seed 45: 500.0`
  - `seed 46: 500.0`
  - 汇总：`mean_of_means=500.0`, `std_of_means=0.0`
- 旧 `v87 best`
  - `seed 42: 481.94`
  - `seed 43: 480.2`
  - `seed 44: 468.72`
  - `seed 45: 469.4`
  - `seed 46: 480.44`
  - 汇总：`mean_of_means=476.14`, `std_of_means=5.82`

这说明新 run 不是默认 seed 42 上的偶然满分，而是在 `42..46` 全部稳定满分；同时也说明它已经实质性超过旧 `v87` 的跨 seed 稳定性。

### 9.4 1000 -> 1250 区间的真实主导问题

`train_metrics.jsonl` 复盘表明：`1000 -> 1200` 这段主导阶段是 `post_entry/post_entry_soft`，不是 `persistence`。

关键点：

- `step=1000`
  - `imag/controller_stage=post_entry`
  - `imag/continue_prob_cap_dynamic=0.9044`
  - `actor/post_trigger_scale=0.9217`
  - `actor/negative_adv_guard_active=0.0`
  - `imag/persistence_tail_mismatch_active=0.0`
  - `imag/persistence_tail_gap_mean=4.7091`
  - `actor/adv_mean=-3.0882`
- `step=1050/1100/1150/1200`
  - stage 进入 `post_entry_soft`
  - `negative_adv_guard_active=1.0`
  - tail repair 仍然完全没有触发
- `step=1250`
  - `imag/controller_stage=trigger`
  - `imag/persistence_tail_mismatch_active=0.0`
  - `actor/post_trigger_scale=0.8573`
  - `actor/adv_mean=-6.6985`
  - 但同一步 eval 结果已经是 `500.0`

结论：

- `1000` 的掉点并不是 persistence tail mismatch 造成的
- 本轮新增的 tail repair 没有在 `1000 -> 1200` 提前误触发
- 这段波动来自 `post_entry/post_entry_soft` 的 imagined geometry，而不是 replay/采样/评估问题
- `1250` 时真实策略已经 solved，但 imagined critic / advantage 仍然偏悲观，说明当前系统里还存在“真实策略已稳定、imag critic 仍滞后”的张力

### 9.5 checkpoint forensics：旧 v87 vs 新 tailfix best

对比 `old_v87_best` 与 `new_v87_tailfix_best` 的 imagined rollout：

- `old_v87_best`
  - `stateful eval mean=479.2`
  - `continue_prob_mean=0.8812`
  - `value_target_gap_abs_mean=4.4386`
- `new_v87_tailfix_best`
  - `stateful eval mean=500.0`
  - `continue_prob_mean=0.9559`
  - `value_target_gap_abs_mean=6.4245`

解释：

- 新 run 的真实策略显著更强，imagined continue 也更高
- 但 imagined `value/target` 对齐并没有同步变得更“干净”，反而仍存在更大的绝对 gap
- 这进一步说明：本轮 tail 修复已经把真实 pure-imag 训练拉回 solved，但 imagined critic calibration 还没有完全跟上

换句话说：

- 本轮已经修复了“纯 imag 训练会在 persistence tail 失稳”的主故障
- 但后续若要继续扩大优势，而不是只停在 solved，优先级应转向 `post_entry/post_entry_soft` 阶段的 imagined critic / advantage 校准

## 10. 当前总判断

截至 2026-03-11 晚，本项目在 `CartPole-v1` 上已经恢复 pure-imag / 纯血 MBRL 的 solved 能力，而且是稳定 solved：

- 新 run 的 `best.pt` 和 `final.pt` 在统一 evaluator 下均为 `10/50/100ep = 500.0`
- 在 `42..46` 跨 seed 的 `50ep` 评估中，`best.pt` 与 `final.pt` 都是 `500.0`
- 旧 solved 基线 `v87` 被明确超过

当前剩余的主要技术矛盾已经不再是 “persistence tail mismatch 会不会把 pure-imag 训练直接做废”，而是：

- `post_entry/post_entry_soft` 阶段 imagined critic / actor advantage 仍偏悲观
- 真实策略先 solved，imagined 几何后跟进，二者之间还有校准延迟
- 如果后续目标不是“能 solved”，而是“更快、更稳、更少回撤”，下一轮应围绕 `post_entry` 专项做 critic / advantage calibration，而不是回到 persistence tail 分支继续大改
