# Findings & Decisions

## Requirements
- 用户要求先一次性梳理 `aletheia_train.py` 剩余 correctness 风险，再给出完整修复清单、任务列表、最优雅修复方案。
- 方案需要先复核，再按顺序逐包实施。
- 实施节奏固定为：补测试、改实现、跑分组测试、打回滚提交点。
- 编辑必须用 `apply_patch`，提交用非交互 git。

## Research Findings
- `aletheia_train.py` 当前剩余 `except Exception` / `logger.warning` 口可以粗分为三类：
- 第一类：直接影响训练语义的 strict 候选，例如 `resolve_policy_wall_strength()`、slow-target regularization、post-solved anchor clone/restore、runtime task-cert telemetry sanitize。
- 第二类：显式兼容/恢复边界，例如 `steps_since_collect` 推断、checkpoint anchor fallback rebuild；这类要么已经显式开关，要么需要继续分层。
- 第三类：metrics-only 保护，例如 `loss_packet.to_metrics()`、部分 eval-anchor KL 统计；这些不应误收成训练 fail-fast。
- `resolve_policy_wall_strength()` 已确认属于第一类：异常会静默回退成 `1.0`，改变 `policy_from_wm_state()` 梯度隔离语义，已在提交 `05687f5` 收 strict。
- `slow-target regularization` 已确认属于第一类：异常会把 `slow_reg` 置 0 继续训练，已在提交 `8a20494` 收 strict，并在 `slow_value_reg_weight=0` 时显式短路。
- `post_solved actor/critic anchor` 链路里存在多处 clone/state_dict 失败后置空继续的口；一旦相关 anchor KL/pull/critic penalty 启用，这会让训练约束静默失效。
- `_sanitize_bootstrap_runtime_task_cert_telemetry()` 不是纯 metrics，它会回写运行态 `real_task_cert_state/alarm/recovery`，解析失败后写成 `0.0` 可能改变后续 gate 计算。
- `runtime compensation guard mismatch` 目前仍是 warning-only；其语义是“恢复到了 `post_solved` 阶段，但当前配置没启用 post-solved guards”，高度可疑，可能应默认 strict。
- `post_solved anchor` 在 `TrainingStep` 与 `TrainingLoop` 各有一份 `_capture_post_solved_actor_anchor()`，两处都在 `deepcopy` 失败时静默置空；这是重复实现漂移点，应统一成共享 strict helper。
- `_clone_frozen_actor()` 不仅服务于 behavior-policy eval-anchor，也服务于 `real_stability_certified_anchor` 注册表采集；后者会影响 `behavior_policy_*_registry_support_mask` 和 actor/critic contract，不是 metrics-only。
- `_compute_distribution_kl_tensor()` 只被两个调用点使用：
- `policy_bank` 路径：失败会让 certified registry support 退化为全 0，影响 contract 与 bootstrap。
- `behavior_policy_eval_anchor` 路径：主要生成审计/对比指标，可保留 best-effort。
- `compensation_kernel.restore_adaptive_compensation_state(..., strict=True)` 已经覆盖了 `partial_restore / fallback_recovered / degraded` 的 strict fail-fast，因此 restore-side degraded 不再是 `aletheia_train.py` 当前最优先缺口。
- `_module_state_dict_to_cpu()` 被 `compensation_kernel` 导出 `post_solved anchor`、`behavior_policy eval anchor`、`real_stability registry` 状态时调用；如果 `state_dict()` 失败当前会导出空 `{}`，属于 checkpoint 边界上的 silent degradation。
- `WP1` 已实施后的 contract：
- `post_solved anchor` capture 现在按 config 分层：pull 只要求 parameter snapshot，KL/real-KL 要求 actor policy snapshot，critic penalty 要求 critic snapshot。
- `TrainingStep` / `TrainingLoop` 的 `_capture_post_solved_actor_anchor()` 已统一到共享 helper，避免两份实现再次漂移。
- `real-stability certified anchor` clone/export 失败不再静默吞掉，会直接 fail-fast。
- restore 侧只对“当前配置明确依赖”的 missing snapshot 记 `partial_restore`；optional snapshot 失败仍允许 `fallback_recovered`，保持兼容恢复边界清晰。
- `test_training_loop_integration.py` 中有数条断言已落后于近期 strict 契约与 registry 语义，本轮已一并校正，确保后续 `WP2-WP4` 回归噪音下降。
- `WP2` 已实施后的 contract：
- `behavior_policy eval-anchor KL` 继续是 best-effort 审计路径，失败时不应把训练主流程打断。
- certified-registry KL 直接参与 `behavior_policy_*_registry_support_mask`、actor contract 与 bootstrap，失败时必须抛错而不是回退成“无 registry 支持”。
- KL helper 现在需要显式调用语义：`strict=True` 必须带上下文抛错，`strict=False` 才允许返回 `None`。
- `WP3` 已实施后的 contract：
- real-stability / bootstrap runtime task-cert telemetry 现在共用同一套 finite-number coercion helper。
- missing key 仍按原语义处理；但如果 telemetry 显式提供非法值或非有限值，不再默默写成 `0.0`，而是直接抛 `ValueError`。
- 这意味着 telemetry 生产侧现在必须对 `nan` / `inf` / 非数字字符串负责，不能再依赖训练侧 sanitize 偷偷兜底。
- `WP4` 已实施后的 contract：
- `runtime_compensation_guard_mismatch` 现在挂到 `validation_mode` 三态契约：默认 `strict` 直接抛错，`warn` 只告警一次，`off` 才允许静默兼容。
- 这条 mismatch 语义是“恢复态已经进入 `post_solved`，但当前 run 没有启用 post-solved guards”；它不再作为 warning-only 保活路径存在。
- 与此同步，测试夹具里那些显式想表达 `validation_mode="off"` 的 case 已改成 `config_mode="compat"`，避免被 `TrainingConfig.__post_init__` 自动收回 strict 后产生伪兼容测试。
- 仓库内 `config_mode="strict" + validation_mode="off"` 的历史测试夹具已在 `aletheia/` 范围内清零；最后一批 quality-scale 测试统一到了显式 `config_mode="compat"`，提交点为 `3f13063`。
- 本轮对 `WP1-WP4` 之外的全仓 warning-continue 口复扫后，剩余 training-side 路径大致分成三类：
- 显式兼容口：`_resolve_resume_steps_since_collect(... legacy_steps_since_collect_mode='infer')`、`runtime_compensation_guard_mismatch` 的 `validation_mode='warn'`、`compensation restore strict=False`。这些都已经需要显式选择兼容模式，不再属于默认 silent degrade。
- 诊断型 warning：`BranchValidator`、`Entropy collapse` 恢复提醒、`RemainingStepsComputer` 的 no-done 比例提示、`min_seq_len` 过短提示。它们会继续运行，但语义上更像数据/训练状态提示而不是恢复失败保活。
- 先前最值得收敛的默认 warning-continue 边界是 `_training_checkpoint_schema.py` 中 `optimizer_restore_mode='auto'` 默认降级 skip。这条已经在提交 `4462411` 收口：
- 默认 `TrainingCheckpointRestorePolicy.optimizer_restore_mode` 与 CLI `--resume-optimizer-restore-mode` 都改为 `strict`。
- strict 模式现在不仅拒绝 config drift / missing effective config metadata，也拒绝 requested restore layer 缺失 section 或 target object。
- `api.run_train()` 的 resume metadata 预读路径改成显式 `training_state only` policy，不再依赖默认宽松跳过。
- 当前剩余的 checkpoint 兼容口只剩显式 opt-in：`optimizer_restore_mode='auto'` 和 `skip`。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 用“训练语义影响面 + 默认启用程度 + 修复耦合度”来排序候选口 | 避免只按 grep 结果机械修复 |
| 优先收会改变损失、梯度、gate、恢复状态的 silent fallback | 这是最直接的 correctness 风险 |
| metrics-only 保护口单独列为保留项 | 避免把调试可观测性问题误修成训练中断 |
| 对恢复/兼容口先判断是否已有显式 restore mode | 避免重复发明另一个兼容开关 |
| `WP1` 与 `WP2` 分开 | frozen snapshot 和 registry KL 都属高风险，但前者是状态捕获，后者是运行时计算，拆开能降低单包耦合 |
| `runtime_compensation_guard_mismatch` 优先走 `validation_mode` 分层而不是硬编码新开关 | 项目已有 `strict/warn/off` 契约，复用更优雅 |
| 对 telemetry sanitize 采用共享 numeric coercion helper | 避免两套实现继续漂移，并显式区分 missing key 与 invalid value |
| `WP1` 不把 optional snapshot restore 全部收成 fail-fast | 避免把 schema/历史兼容恢复与“当前配置真的依赖该 snapshot”的 correctness 风险混为一谈 |
| `WP2` 只把 certified-registry KL 收成 strict，保留 eval-anchor 审计 best-effort | 两条路径的训练影响面不同，不能一刀切 |
| `WP3` 对 telemetry 采用“missing-key compatible, invalid-value strict” | 既保留兼容数据源缺省，又阻断 silent numeric corruption |
| `WP4` 对 runtime mismatch 采用 `validation_mode` 分层 | 训练语义风险默认 strict，兼容行为必须显式选择 `warn/off` |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 单靠 grep 无法判断异常口是否影响训练语义 | 继续沿读写链路追踪调用方、状态回写点和损失消费点 |
| `session-catchup.py` 无输出 | 当前项目无待恢复的 planning 上下文，直接进入新一轮规划 |
| `behavior_policy_eval_anchor` 表面上像 metrics-only，但其 sibling `registry` 路径会影响 contract | 继续把 eval-anchor 与 registry 支路拆开分析，避免误归类 |

## Resources
- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py`
- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_training_entrypoints.py`
- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_config.py`
- `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/training/compensation_kernel.py`
- commits: `8a20494`, `05687f5`, `e6e76ff`, `df1d909`, `04dd84c`, `7c4cfe9`

## Visual/Browser Findings
- 本轮无浏览器/图片发现。
