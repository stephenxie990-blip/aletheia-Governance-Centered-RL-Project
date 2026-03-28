# Task Plan: Aletheia Training Correctness Sweep

## Goal
完成 `aletheia/aletheia_train.py` 剩余“异常吞掉 / warning 后继续 / 默认值降级继续训练”路径的系统梳理，给出完整修复清单、最优修复方案与执行顺序，并按统一节奏逐包实施。

## Current Phase
Phase 5

## Phases
### Phase 1: Discovery & Inventory
- [x] 梳理用户约束与当前工作节奏
- [x] 盘点 `aletheia_train.py` 里剩余异常吞掉/降级运行路径
- [x] 记录当前回滚点与最近已完成修复
- **Status:** complete

### Phase 2: Risk Classification & Repair Plan
- [x] 区分 strict fail-fast、显式兼容、metrics-only 三类边界
- [x] 形成完整修复清单、优先级和最优修复策略
- [x] 记录测试策略、提交粒度和回滚点规则
- **Status:** complete

### Phase 3: Plan Review
- [x] 复核每个候选口是否真影响训练语义
- [x] 复核是否存在重复实现，必要时统一契约
- [x] 锁定执行顺序，避免交叉改动扩大 blast radius
- **Status:** complete

### Phase 4: Incremental Execution
- [x] `WP1` 每一包先补测试
- [x] `WP1` 再改实现
- [x] `WP1` 再跑分组测试
- [x] `WP1` 再打回滚提交点
- [x] `WP2` registry KL strictness 先补测试
- [x] `WP2` registry KL strictness 再改实现
- [x] `WP2` registry KL strictness 再跑分组测试
- [x] `WP2` registry KL strictness 再打回滚提交点
- [x] `WP3` numeric telemetry strictness 先补测试
- [x] `WP3` numeric telemetry strictness 再改实现
- [x] `WP3` numeric telemetry strictness 再跑分组测试
- [x] `WP3` numeric telemetry strictness 再打回滚提交点
- [x] `WP4` runtime mismatch policy
- **Status:** complete

### Phase 5: Final Verification & Delivery
- [x] 汇总剩余风险与明确保留的兼容口
- [x] 更新计划文件状态
- [x] 清理 `strict + off` 测试夹具冲突并单独验证
- [x] 复扫 `WP1-WP4` 之外 warning-continue 训练侧路径
- [x] 收紧 training checkpoint restore 默认 strict 契约
- [ ] 向用户交付修复清单、状态与下一步建议
- **Status:** in_progress

## Key Questions
1. 哪些异常吞掉口会直接改变训练更新、损失项、梯度隔离或恢复语义？
2. 哪些口只是 telemetry/metrics 保护，不应被误收成 fail-fast？
3. 哪些“兼容恢复”必须继续保留，但需要显式开关而不是默认静默降级？
4. 哪些重复实现应统一成单一 strict contract，避免后续再次漂移？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 继续采用 `测试 -> 实现 -> 分组验证 -> 提交回滚点` 的包级节奏 | 降低 blast radius，保持每次变更可回退 |
| 本轮先停手扩散修改，先做完整 inventory 和风险分层 | 用户要求先一次性梳理清楚再执行 |
| 优先处理真正改变训练语义的吞异常口 | 这些口比 metrics-only warning 更可能造成 silent correctness drift |
| 使用项目内规划文件持续记录 | 避免后续 sweep 过程丢上下文或重复扫描 |
| 采用“三层修复模型” | Tier 1 训练语义 invariant 默认 strict；Tier 2 配置/恢复一致性问题走 `validation_mode` 分层；Tier 3 metrics-only 保留 best-effort 但必须与训练语义隔离 |
| 将后续修复按 4 个 work package 落地 | 降低单包复杂度，便于逐包测试和回滚 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `uv run pytest ...` 在 Python 3.14 + torch/libomp 组合下直接 abort | 1 | 通过 `KMP_DUPLICATE_LIB_OK=TRUE` 跑测试，记录为当前本地测试前提 |

## Notes
- 当前已完成回滚点：`8a20494` slow-target strict、`05687f5` policy wall strength strict。
- 已锁定后续 4 个 work package：
- `WP1` frozen snapshot strictness：`post_solved anchor` / `real_stability registry` 的 clone、capture、state export。
- `WP2` registry KL strictness：`_compute_distribution_kl_tensor` / `_compute_policy_bank_min_kl_tensor` 分离 strict 与 metrics-only 调用。
- `WP3` numeric telemetry strictness：`_sanitize_real_stability_telemetry` / `_sanitize_bootstrap_runtime_task_cert_telemetry`。
- `WP4` config/runtime mismatch policy：`runtime_compensation_guard_mismatch` 按 `validation_mode` 默认 strict 分层。
- `WP1` 已完成并提交：`e6e76ff`。
- `WP2` 已完成并提交：`df1d909`。
- `WP3` 已完成并提交：`04dd84c`。
- `WP4` 已完成并提交：`7c4cfe9`。
- `strict/off` 历史测试夹具统一已完成并提交：`3f13063`。
- `training checkpoint restore` 默认 strict 收口已完成并提交：`4462411`。
