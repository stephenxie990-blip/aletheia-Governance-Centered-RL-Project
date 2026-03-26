# 纯血 MBRL 修复总结与下一步方案

## 1. 目标与当前结论

本项目的目标不是“让 CartPole 偶尔跑高分”，而是恢复纯 imagined trajectory 训练链路，使其重新具备 Dreamer 类纯血 MBRL 的核心优势，并最终稳定达到 CartPole solved。

截至 2026-03-08，本轮工作已经完成两件关键事情：

1. 修复了会直接污染纯 imag 结论的训练链路缺陷。
2. 在此基础上加入了一个低侵入的 late-stage 稳定化方案，使交付 artifact 的性能显著好于修复后的 baseline。

但结论也必须说清楚：

- 现在可以确认，项目不是“纯 imag 根本没实现”。
- 现在也可以确认，之前纯 imag 失败并不全是算法本身失败，其中包含严重的实现缺陷。
- 当前已经恢复了纯 imag artifact 的明显优势。
- 当前还没有恢复到“训练过程稳定 solved”的状态，late-stage oscillation 仍然明显存在。

一句话总结：

> 项目已经从“纯 imag 结论不可信”修复到“纯 imag 路径正确且交付性能显著提升”，但还没有到“稳定 solved CartPole”。

## 2. 已固化的修复

### 2.1 imag rollout 状态推进修复

问题性质：架构/模块级缺陷。

在 `WorldModel.forward_imagination()` 中，后续 imagined step 原先复用了 `prev_state.x_t`，导致控制分支 `s_ctrl` 没有沿 imagined embedding 正常前进，而是持续带着上一个真实观测 embedding 的残留语义。

这会直接破坏“想象轨迹训练”的语义完整性。表面上看在做 imag rollout，实际上控制路径没有真正跟着 imagined latent 演化。

修复方式：

- 使用当前 step 的 `x_proj` 作为 imagined step 的 `imag_x_t`
- 同时令 `new_state.x_t = imag_x_t`

代码位置：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_world_model.py`

相关测试：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_pure_imagination_path.py`

### 2.2 eval 污染训练 state 修复

问题性质：训练链路级缺陷。

`_evaluate_agent()` 之前直接使用并改写了与训练 collector 共用的 stateful `AgentHandle` runtime state，包括：

- `_wm_state`
- `_prev_action`

结果是：训练中途一旦触发 eval，后续 collector 的 rollout 起点就被评估阶段污染。对于纯 imag 训练，这种污染足以扭曲后续实验结论。

修复方式：

- 在 eval 前 clone runtime state
- eval 完成后 restore runtime state

代码位置：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py`

相关测试：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_run_train_contracts.py`

### 2.3 imagined continue probability cap 稳定化

问题性质：算法稳定性修复，不再是 plumbing 修复。

在前两项链路修复完成后，pure imag 100k 仍然存在明显的 late-stage oscillation。法证结果显示，后半程的主要问题是 imagined continue prediction 过于自信，进而带来：

- effective horizon 膨胀
- actor target inflation
- value-target gap 扩大
- 训练后半段剧烈振荡

为此新增配置：

- `imag_continue_prob_cap: float = 0.0`

启用后行为：

- 对 imagined `continue_probs` 做 cap
- 基于 capped continue 重新计算 imagined λ-return
- 基于 capped continue 重新计算 actor weights
- 输出 raw vs capped continue 指标，便于法证和监控

代码位置：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_config.py`
- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py`
- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py`

相关测试：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_training_loop_integration.py`

### 2.4 CartPole pure-imag 默认入口接线

为了让修复后的稳定化方案成为支持路径的一部分，而不是“实验时手工加 override 才生效”，现在已经对支持入口做了默认接线：

条件：

- `env` 为 CartPole
- `imagination_only=True`

默认行为：

- 自动注入 `imag_continue_prob_cap=0.95`
- 仍然保留显式 override 优先级

这一步的意义是把修复从“研究状态”推进到“实际交付入口状态”。

## 3. 验证结果

### 3.1 测试验证

当前验证结果：

- 新增入口契约测试：通过
- imag 路径推进测试：通过
- eval runtime state 恢复测试：通过
- imag continue cap 重算 returns/weights 测试：通过
- 全量测试：`76` 个测试全部通过

关键测试文件：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_pure_imagination_path.py`
- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_run_train_contracts.py`
- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_training_loop_integration.py`

### 3.2 默认入口 smoke 验证

已运行一个不显式传 cap override 的 CLI smoke run，验证默认入口确实消费到了新配置，而不只是测试里 mock 出来的结果。

产物位置：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/smoke_cartpole_defaultcap_1024/config_resolved.json`
- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/smoke_cartpole_defaultcap_1024/train_metrics.jsonl`
- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/smoke_cartpole_defaultcap_1024/summary.json`

验证到的事实：

- `train_config.imag_continue_prob_cap = 0.95`
- `overrides_effective.imag_continue_prob_cap = 0.95`
- 训练日志里出现 `imag/continue_prob_cap`
- 同时出现 raw continue 和 capped continue 指标

说明默认入口接线已经真实生效。

## 4. 关键实验结论

### 4.1 repaired baseline 100k

产物：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_pureimag_baseline_100k_imagxtfix_sched32_evalrestore`

结论：

- 10-episode eval 历史已经明显好于早期错误实现
- fixed-seed 100-episode 复评：
  - `best.pt mean=158.85 std=49.65`
  - `final.pt mean=144.05 std=44.58`

意义：

- 证明纯 imag 并非完全失效
- 但仍远未达到稳定 solved

### 4.2 repaired 20k + continue cap 0.95

产物：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_pureimag_contcap095_20k`

结论：

- 10-episode summary 很强：`final_current = 475.1`
- `best_eval = 483.6 @ step 600`
- 但 fixed-seed 100-episode 复评只有：
  - `mean=169.56`
  - `median=91.0`
  - `std=158.17`

意义：

- 有真实提升
- 但不能把短程 10-episode 高分误判为稳定 solved

### 4.3 repaired 100k + continue cap 0.95

产物：

- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_pureimag_contcap095_100k`

结论：

- 训练过程 eval 仍然剧烈振荡
- 但最终交付 artifact 显著强于 repaired baseline
- fixed-seed 100-episode 复评：
  - `mean=299.36`
  - `std=140.97`
  - `median=192.5`
  - `p90=500.0`

与 repaired baseline 对比：

- repaired baseline best：`158.85`
- continue-cap 100k artifact：`299.36`

意义：

- 交付 artifact 的 pure imag 优势已经明显恢复
- 但训练时间序列仍然明显不稳定

## 5. 这次修复后，问题到底还剩什么

当前剩余主问题不是“系统有没有按纯 imag 设计实现”，而是：

- late-stage imagined overconfidence
- target inflation
- effective horizon 偏长
- actor/value 在后半程相互放大
- 最终表现为训练振荡大，重复性不足

这说明当前阶段的主问题归类应当是：

- 不是入口问题
- 不是 collector 是否真实接线的问题
- 不是 replay buffer / rollout collector / training loop 的基本数据链路缺失问题
- 主体是算法稳定性问题

更直接一点说：

> 现在再继续修 plumbing，收益会很低。下一阶段应该集中攻 late-stage stabilization，而不是再怀疑入口和基本链路。

## 6. 下一步工作重点

下一步的核心目标只有一个：

> 把“artifact 变强但过程振荡”推进到“训练过程也更稳，fixed-seed 复评持续接近 solved”。

### 6.1 工作重点一：做自适应稳定化，不再依赖静态 cap

当前 `imag_continue_prob_cap=0.95` 已经证明一个事实：

- imagined continue 的过高自信是有效打击点

但静态 `0.95` 也暴露出局限：

- 早期可能过松
- 后期可能仍然不够
- 不能根据 target inflation 动态响应

下一步建议：

- 将静态 cap 升级为自适应 cap 或自适应 effective horizon 控制
- 触发信号直接使用：
  - `imag/value_target_gap_abs_mean`
  - `imag/continue_prob_mean_raw`
  - `actor/target_mean`
  - `imag/effective_horizon`

建议方案：

- 当 `continue_prob_mean_raw` 和 `value_target_gap_abs_mean` 同时抬升时，自动收紧 continue cap
- 当指标恢复正常时，缓慢释放 continue cap
- 避免硬开关，采用平滑更新

### 6.2 工作重点二：给 imagined target 增加二级防爆机制

continue cap 已经证明 horizon inflation 是问题之一，但它未必是唯一问题。下一步应增加“二级保护”，避免 actor target 和 value target 在后半段继续失控。

建议方案：

- 对 imagined returns 相对 bootstrap baseline 的增量做自适应 clip，而不是固定 clip
- clip 阈值与近期 `value_target_gap_abs_mean` 或 return scale 挂钩
- 只在 imag batch 上生效，不污染 real batch

目标不是把学习压扁，而是避免 late-stage target runaway。

### 6.3 工作重点三：建立固定评估协议，拒绝被 10-episode 噪声误导

后续所有迭代必须固定两类评估：

- 训练中 10-episode eval：只用于在线监控
- 固定种子 100-episode re-eval：作为是否真的改进的准入标准

准入标准建议：

- 不接受仅凭 10-episode 高分宣布 solved
- 至少要求固定种子 100-episode mean 持续提升
- 同时要求 std 明显收敛，而不是只拉高 mean

### 6.4 工作重点四：跑一个 canonical 默认入口 100k 实验

现在默认入口已经接上 `imag_continue_prob_cap=0.95`，但还缺一个“默认入口下重新生成的标准 100k 产物”。

下一步应新增一条 canonical run：

- 使用 `scripts/cartpole_train.py`
- `--steps 100000`
- `--imagination-only`
- 不显式传 `imag_continue_prob_cap`

意义：

- 验证默认交付路径是否与手工 override 的实验一致
- 形成新的标准对照产物
- 为下一轮算法修复提供稳定起点

## 7. 推荐的下一阶段执行方案

建议按下面顺序推进，而不是同时改很多东西：

### 阶段 A：默认入口基线固化

目标：

- 生成新的默认入口 100k canonical run
- 补齐 fixed-seed 100-episode re-eval
- 作为之后全部算法修复的统一 baseline

交付物：

- `outputs/exp_pureimag_defaultcap095_100k/*`
- 固定评估结果
- 对照 repaired baseline 与 manual-override cap run 的差异报告

### 阶段 B：自适应 continue stabilization

目标：

- 把静态 `0.95` 升级为指标驱动的动态机制

验收指标：

- 训练曲线振荡幅度下降
- fixed-seed 100-episode mean 不低于当前 `299.36`
- std 明显下降

### 阶段 C：自适应 target delta 控制

目标：

- 进一步抑制 late-stage target runaway

验收指标：

- `imag/value_target_gap_abs_mean` 后半程不再继续扩张
- `actor/target_mean` 不再出现异常抬升
- 训练中间 checkpoint 的质量下降幅度缩小

## 8. 不建议的方向

以下方向当前不建议作为主线：

- 再次回退到 stateless wrapper collector/eval 路线
- 再把问题解释为 replay buffer/collector 没接上纯 imag
- 只围绕 10-episode eval 做超参数碰运气
- 继续大范围同时改多个稳定化项，导致失去归因能力

## 9. 最终判断

当前项目状态应当被准确定义为：

- 训练链路关键 bug：已修复
- pure imag 路径：已真实恢复
- CartPole 交付 artifact：显著提升
- 稳定 solved：尚未完成

因此下一阶段的工作重点应明确为：

> 围绕 late-stage oscillation 做自适应稳定化，把“能跑出强 artifact”推进为“能够更稳定地 solved”。



## 10. New Finding from the Canonical Default-Entry Run
After the report was first written, a new default-entry `100k` CartPole pure-imag run was executed at `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_pureimag_defaultentry_100k`.

What it proved:

- The supported entrypoint now really injects `imag_continue_prob_cap=0.95` by default for `CartPole + imagination_only`; this was confirmed in the live run `config_resolved.json` without any explicit CLI override.
- The uninterrupted part of the run reproduced the earlier manual-override experiment exactly through `step 2000`, with evals `114.7 -> 31.7 -> 26.7 -> 252.3`.

What it exposed:

- The run did not finish cleanly on the first attempt.
- Resuming from `resume_latest.pt` restored the model and `global_step`, but did not preserve imagination-only training continuity.
- Immediately after resume, runtime logs reported `Imagination-only enabled but no seed data available`.
- In `train_metrics.jsonl`, steps `2100`, `2200`, and `2300` appear twice, and the second copies show `train/env_steps_collected` resetting from `16800/17600/18400` to `832/1632/2432`.
- `eval_history.jsonl` also shows the discontinuity clearly: `step 2000 mean=252.3` followed by `step 2500 mean=8.0`.

Implication:

- The finished artifact in `/outputs/exp_pureimag_defaultentry_100k` cannot be treated as the canonical 100k baseline, because the back half of the run is not a true continuation of the front half.
- This introduces a new high-priority repair item: resume must restore replay/buffer/collector seed state sufficiently for imagination-only training to continue without resetting effective data history.


## 11. Resume Continuity Fix
The canonical default-entry run also exposed a resume-chain defect. That defect has now been repaired.

Implemented changes:

- Resume checkpoints now persist replay-buffer contents and in-progress buffer state.
- Trainer resume now restores replay state, `env_steps_collected`, and episode-tracking EMA/counters.
- API resume now restores `best_eval_return/best_step` before continuing and truncates stale JSONL records above the resume step.

Validation:

- New trainer-level regression test proves imagination-only resume can continue from restored seed data without hitting `no seed data available`.
- New API-level regression test proves resumed runs keep the prior best eval and do not append on top of stale higher-step JSONL history.
- A real two-stage CLI smoke run at `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/smoke_resume_continuity_v2` verified continuity from `step 20` to `step 30` on the supported `cartpole_train.py` path.
- The resulting `eval_history.jsonl` is continuous at `5,10,15,20,25,30`, and `train_metrics.jsonl` shows monotonic `env_steps_collected` (`64,96,128,160,224,256`) with no duplicate step records.

Updated implication:

- Resume continuity is no longer the primary blocker. The project can return to the mainline objective: improving late-stage pure-imag stability and solved-rate robustness on long runs.


## 12. Clean Canonical Default-Entry Baseline
After repairing resume continuity, a fresh clean default-entry `100k` run was generated at `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_pureimag_defaultentry_100k_resumefixed`.

This run used the supported entrypoint only; no explicit `imag_continue_prob_cap` override was passed. The resolved config still shows `imag_continue_prob_cap=0.95` via the CartPole pure-imag default path.

Most importantly, the clean default-entry run reproduces the earlier manual-override run exactly:

- Eval history: `114.7 -> 31.7 -> 26.7 -> 252.3 -> 18.9 -> 35.8`
- Final promoted artifact: `283.8 +- 141.7`
- `best_eval_mean_during_train = 252.3 @ step 2000`

Implication:

- The current supported default path is now the canonical pure-imag CartPole baseline.
- Because this clean run exactly matches the earlier manual-cap run, the remaining failure mode is definitively algorithmic late-stage oscillation rather than any remaining entrypoint, resume, collector, or logging defect.


## 13. Adaptive Continue Controller: Wiring Verification and Next Gate
A pressure-based adaptive imagined-continue controller has now been implemented and smoke-verified on the supported CLI entrypoint.

Verified artifact:

- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/smoke_adaptive_continue_cap_200`

What this smoke proved:

- The adaptive overrides are correctly propagated through `scripts/cartpole_train.py` into the resolved runtime config.
- The controller is live in the real training loop rather than being a dead code path covered only by tests.
- At `step 50`, the controller tightened the continue cap from static `0.95` down to `0.9186410278327264` because raw imagined pressure was elevated:
  - `imag/continue_cap_gap_abs_raw = 5.2882`
  - `imag/continue_cap_continue_mean_raw = 0.9821`
  - `imag/continue_cap_pressure = 0.0318`
- By `step 200`, the cap relaxed back to `0.95` as those raw metrics cooled. This means the controller can both tighten and release, which is the intended control behavior.

What this smoke did not prove:

- It did not prove improved solved-rate or late-stage robustness. A 200-step smoke is only enough to verify live behavior and log integrity.

Current execution gate:

- Two same-seed `1500`-update CartPole pure-imag runs are now the next decision point:
  - static-cap baseline: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_staticcap_1500_seed42`
  - adaptive-cap experiment: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_adaptivecap_1500_seed42`
- The comparison will decide whether adaptive continue control is merely well-wired, or actually improves medium-stage stability enough to remain on the mainline repair path.


## 14. Medium-Run A/B Verdict: Static 0.95 vs Adaptive Continue Control
The next decision gate has now been completed with a same-seed medium-length CartPole pure-imag comparison.

Artifacts:

- static baseline: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_staticcap_1500_seed42`
- adaptive variant: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_adaptivecap_1500_seed42`

Protocol:

- `CartPole-v1`
- `--update-steps 1500`
- `--seed 42`
- `--imagination-only`
- `--eval-interval 250`
- `--eval-episodes 5`

Observed eval histories:

- static: `18.2 -> 112.0 -> 168.2 -> 61.2 -> 41.2 -> 74.2`
- adaptive: `11.2 -> 12.2 -> 113.8 -> 500.0 -> 148.6 -> 107.4`

What this means:

- The adaptive controller is not just wired correctly; it materially changes training behavior.
- Its current downside is clear: it harms early learning. On the first two eval points it is much worse than the repaired static baseline.
- Its upside is also clear: it suppresses the exact mid/late imagined-target runaway that destroys the static branch.

The strongest diagnostic evidence is the `850-1000` window:

- Static branch:
  - `actor/target_mean`: `5.80 @ 750 -> 12.07 @ 850 -> 24.74 @ 900 -> 46.96 @ 950`
  - `imag/value_target_gap_abs_mean`: `1.91 @ 750 -> 3.07 @ 850 -> 9.41 @ 900 -> 23.00 @ 950 -> 26.84 @ 1000`
  - Eval collapses from `168.2 @ 750` to `61.2 @ 1000`
- Adaptive branch:
  - pressure stays quiet earlier, then turns on when needed
  - `step 950`: `continue_prob_mean_raw = 0.9949`, `continue_prob_cap = 0.8755`, `cap_pressure = 0.1018`
  - `step 1000`: `continue_prob_cap = 0.8582`, `actor/target_mean = 7.79`, `imag/value_target_gap_abs_mean = 3.90`
  - Eval reaches `500.0 @ 1000` instead of collapsing

Updated verdict:

- The project is no longer blocked only by “whether any algorithmic fix helps.” One does help.
- The remaining problem has become more precise: current adaptive control preserves mid/late stability but over-regularizes too early.
- Therefore the next repair target should not be “invent another unrelated stabilization.” It should be “retain adaptive anti-runaway behavior while restoring early learning speed.”

Recommended next implementation direction:

- add an early-training warmup gate or gain schedule for adaptive pressure
- keep current pressure terms (`gap`, `continue`, `actor target`) as the late-stage trigger
- only allow strong cap tightening after a minimum step threshold or after early eval performance clears a baseline

This is the first result in the project that cleanly separates the two remaining algorithmic requirements:

1. early learning must not be choked
2. late imagined target runaway must still be suppressed


## 15. Warmup/Ramp Probe: Useful but Insufficient
After the A/B result above, a direct follow-up fix was implemented: the adaptive controller now supports step-based delayed activation.

Added controls:

- `adaptive_imag_continue_cap_warmup_steps`
- `adaptive_imag_continue_cap_ramp_steps`
- logging metric `imag/continue_cap_adaptive_scale`

Why this was added:

- The first adaptive controller clearly improved the `850-1000` runaway window, but it also harmed `250/500` learning badly.
- That strongly suggested the controller was intervening too early, not just too strongly.

Validation:

- New targeted regression test proves warmup can defer tightening even when raw pressure would otherwise clamp the cap.
- Full test suite now passes at `80` tests.

Experiment:

- artifact: `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_adaptivecap_warm_700_300_1500_seed42`
- settings: `warmup_steps=700`, `ramp_steps=300`

Observed eval history:

- warmup+ramp adaptive: `18.2 -> 112.0 -> 168.2 -> 280.2 -> 33.6 -> 25.6`

Interpretation:

- This confirms the early-learning diagnosis. With delayed activation, the first three eval points exactly match the static baseline rather than the weak early behavior of the original adaptive branch.
- It also retains some anti-runaway value at `step 1000`: `280.2` is far better than static `61.2`.
- But it does not solve post-1000 stability. The run still collapses hard by `1250/1500`.

Updated conclusion:

- Warmup/ramp is worth keeping as infrastructure because it cleanly separates “when should control start?” from “what should happen after control starts?”
- But warmup/ramp alone is not the full repair. The next algorithmic job is to stabilize the post-trigger regime, not keep retuning activation timing in isolation.

Most likely next direction:

- combine delayed activation with a second stabilizer that acts after the cap begins tightening
- likely candidates are stricter post-trigger target-delta control or an actor-target amplitude guard that activates only after the adaptive controller has entered pressure mode


## 16. Post-Trigger Stabilizer Probes: What Failed and What Survived
To move beyond warmup/ramp alone, two post-trigger stabilizer variants were implemented and tested.

### Variant A: Hard post-trigger return clip
Artifact:

- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_adaptivecap_warm_postclip6_1500_seed42`

Eval history:

- `18.2 -> 112.0 -> 168.2 -> 14.8 -> 118.2 -> 11.8`

Verdict:

- Not viable as a mainline fix.
- It preserved early learning because it only activates under pressure mode, but once it triggered it clipped imagined returns too aggressively and corrupted the target surface.
- Because it changes `returns` directly, it harms critic learning as well as actor behavior.

### Variant B: Actor-only post-trigger suppression
Artifact:

- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_adaptivecap_warm_actorguard_1500_seed42`

Eval history:

- `18.2 -> 112.0 -> 168.2 -> 88.8 -> 24.6 -> 247.2`

Verdict:

- More promising structurally than return clipping.
- It leaves critic targets intact and can still recover a strong late checkpoint.
- But the current fixed gain/floor is too coarse: once pressure exceeds the threshold, `actor/post_trigger_scale` drops straight to the floor and creates a different oscillation pattern instead of stable control.

Updated conclusion:

- The next repair should not use fixed hard thresholds or fixed hard floors in the post-trigger controller.
- The surviving idea is actor-only intervention, but it needs to be pressure-shaped and continuous rather than binary/floor-style.

In other words, the project has now ruled out one bad branch and narrowed the next branch:

- bad branch: hard clipping imagined returns after trigger
- viable branch: smoothly scaled actor suppression after trigger


## 17. Continuous Actor Guard: Stronger Branch, Still Not Stable Enough
The fixed-floor actor guard was then upgraded into a continuous pressure-shaped controller. Instead of dropping to a hard floor as soon as pressure crossed the threshold, actor suppression now decays smoothly with pressure excess.

Artifacts:

- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_adaptivecap_warm_actorcurve10_1500_seed42`
- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_adaptivecap_warm_actorcurve3_1500_seed42`

Observed eval histories:

- gain `10.0`: `18.2 -> 112.0 -> 168.2 -> 52.8 -> 490.2 -> 21.0`
- gain `3.0`: `18.2 -> 112.0 -> 168.2 -> 183.6 -> 452.6 -> 24.0`

What this proves:

- Continuous actor-only control is a better structure than fixed-floor control.
- It preserves early learning and can generate very strong late checkpoints.
- It is therefore more promising than both hard return clipping and fixed-floor actor suppression.

What it still does not solve:

- It still fails to hold a strong checkpoint across the next eval window.
- In practical terms, it creates a delayed high-quality peak rather than stable solved behavior.

Updated next-step recommendation:

- stop searching over single scalar gains in isolation
- add controller state / hysteresis so the system reacts differently when pressure is rising versus when it is already recovering
- that is the next logical step if the goal remains “convert strong transient best checkpoints into persistent solved performance”


## 18. Hysteresis Probe: Better Timing, Still No Persistence
The next control upgrade added hysteresis to the continuous actor-only guard. That means entering the risk zone and releasing out of it now use different dynamics.

Artifact:

- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_adaptivecap_warm_actorcurve3_hyst_1500_seed42`

Observed eval history:

- `18.2 -> 112.0 -> 168.2 -> 500.0 -> 26.0 -> 20.4`

Interpretation:

- This is meaningful progress in one narrow sense: hysteresis did exactly what it was supposed to do for timing. The strong checkpoint now appears earlier and more decisively than in the non-hysteretic continuous controller.
- But it still does not solve persistence. The system still produces a strong transient best checkpoint and then loses it by the next evaluation windows.

Updated conclusion:

- Pressure memory alone is not enough.
- The remaining missing signal is likely some notion of checkpoint quality persistence or recovery success, so the controller can decide when to keep protection on versus when to safely release it.

This narrows the next step further:

- not another gain sweep
- not another purely pressure-based state machine
- but a controller that combines pressure with recent quality trajectory


## 19. Quality-Aware Probes: Necessary Signal, Wrong Coupling So Far
The next step after hysteresis was to inject recent quality information into the controller. Two versions were tested.

### Version A: Quality-gated release
Artifact:

- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_adaptivecap_warm_actorcurve3_hyst_quality_1500_seed42`

Eval history:

- `18.2 -> 112.0 -> 168.2 -> 500.0 -> 26.0 -> 20.4`

Verdict:

- This proved that quality is a sensible signal to track, but using it only as a release gate did not materially change the collapse pattern.
- The reason is important: the dominant failure after the strong checkpoint is not caused only by premature release.

### Version B: Direct quality-modulated actor scaling
Artifact:

- `/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_adaptivecap_warm_actorcurve3_hyst_qscale_1500_seed42`

Eval history:

- `18.2 -> 112.0 -> 168.2 -> 111.6 -> 16.4 -> 21.4`

Verdict:

- This version changed the dynamics immediately, which proves quality should affect control strength.
- But the tested direct multiplicative coupling was too aggressive and hurt the strong `step 1000` behavior instead of preserving it.

Updated conclusion:

- The remaining problem is no longer “should quality be part of the controller?” The answer is yes.
- The remaining design problem is “how should quality be coupled?”
- Current evidence suggests the next useful version should use bounded or piecewise quality intervention that activates only in the most pathological zone, rather than continuously shrinking actor scale whenever quality degrades moderately.
