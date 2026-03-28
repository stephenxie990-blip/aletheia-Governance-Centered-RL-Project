# Progress Log

## Session: 2026-03-28

### Phase 1: Discovery & Inventory
- **Status:** complete
- **Started:** 2026-03-28
- Actions taken:
  - 检查技能说明并启用文件化规划流程。
  - 运行 session catchup，确认当前项目没有待恢复 planning 上下文。
  - 重新扫描 `aletheia_train.py` 中 `except Exception`、`logger.warning`、`return {}`、`return None`、置零降级路径。
  - 复读 `slow-target`、`policy wall strength`、post-solved anchor、runtime task-cert telemetry、runtime compensation guard mismatch 等链路上下文。
  - 核对当前工作树、最近回滚点和已完成修复提交。
- Files created/modified:
  - `task_plan.md` (created)
  - `findings.md` (created)
  - `progress.md` (created)

### Phase 2: Risk Classification & Repair Plan
- **Status:** complete
- Actions taken:
  - 将剩余候选口按 strict fail-fast / explicit compatibility / metrics-only 三类初步分层。
  - 识别 `post_solved anchor clone/restore` 与 `bootstrap runtime task-cert telemetry sanitize` 为下一批高风险候选。
  - 确认 `runtime compensation guard mismatch` 是潜在 medium/high 风险 warning-only 口。
  - 进一步确认 `compensation_kernel` restore 侧已经在 `strict=True` 下拦截 degraded restore，因此本轮优先级应回到 train 入口层剩余口。
  - 确认 certified registry KL 失败会影响 contract，不属于 metrics-only。
  - 锁定 4 个 work package：frozen snapshot strictness、registry KL strictness、numeric telemetry strictness、runtime mismatch policy。
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 3: Plan Review
- **Status:** complete
- Actions taken:
  - 复核 `partial_restore/fallback_recovered` 是否已由 compensation kernel 的 strict restore 接管。
  - 复核 `behavior_policy eval-anchor` 与 certified registry 两条支路的真实影响面。
  - 复核 `runtime_compensation_guard_mismatch` 更适合挂到 `validation_mode` 而非新增 ad-hoc 兼容开关。
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 4A: WP1 Frozen Snapshot Strictness
- **Status:** complete (`e6e76ff`)
- Actions taken:
  - 为 `post_solved actor/critic anchor` 和 frozen snapshot export/restore 增补 strict 回归测试。
  - 抽出共享 frozen snapshot helper，统一 `TrainingStep` / `TrainingLoop` 两份 `_capture_post_solved_actor_anchor()`。
  - 将 `post_solved actor/critic` snapshot requirement 与当前 config 绑定，避免启用 KL / critic anchor 时 silent `None`。
  - 将 `real-stability certified anchor` clone 失败改成 fail-fast。
  - 将 `_module_state_dict_to_cpu()` 从 `state_dict()` 失败返回空字典改为带上下文 fail-fast。
  - 收紧 restore 语义：required snapshot 缺失时返回 `partial_restore`，optional snapshot 失败仍保留 `fallback_recovered`。
  - 顺手修正 `test_training_loop_integration.py` 内多条已被现有 strict 契约淘汰的测试夹具和断言，确保整文件回归重新可用。
- Files created/modified:
  - `aletheia/aletheia_train.py` (updated)
  - `aletheia/training/compensation_kernel.py` (updated)
  - `aletheia/tests/test_training_entrypoints.py` (updated)
  - `aletheia/tests/test_training_loop_integration.py` (updated)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 4B: WP2 Registry KL Strictness
- **Status:** complete (`df1d909`)
- Actions taken:
  - 为 eval-anchor 与 certified-registry 两条 KL 链路补了 strict/best-effort 分层测试。
  - 将 `_compute_distribution_kl_tensor()` / `_compute_policy_bank_min_kl_tensor()` 增加 `strict/context` 语义。
  - 保持 `behavior_policy eval-anchor` 审计路径 best-effort。
  - 将 certified-registry 合同路径切到 strict，并统一异常上下文为 `certified registry policy KL`。
- Files created/modified:
  - `aletheia/aletheia_train.py` (updated)
  - `aletheia/tests/test_training_loop_integration.py` (updated)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 4C: WP3 Numeric Telemetry Strictness
- **Status:** complete (`04dd84c`)
- Actions taken:
  - 为 `_sanitize_real_stability_telemetry()` 与 `_sanitize_bootstrap_runtime_task_cert_telemetry()` 补了 invalid / non-finite fail-fast 测试。
  - 抽出共享 finite-number coercion helper，并将两条 sanitize 路径统一到同一个 numeric contract。
  - 保留 missing key 的原有语义，但一旦 telemetry 提供非法值或非有限值，直接抛 `ValueError`。
- Files created/modified:
  - `aletheia/aletheia_train.py` (updated)
  - `aletheia/tests/test_training_loop_integration.py` (updated)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 4D: WP4 Runtime Mismatch Policy
- **Status:** complete (`7c4cfe9`)
- Actions taken:
  - 为 `runtime_compensation_guard_mismatch` 新增 strict / warn / off 三态测试，明确其绑定到 `validation_mode` 契约。
  - 在 `train_step()` 中将原先硬编码的 warning-only 逻辑抽到 `_handle_runtime_compensation_guard_mismatch()`，默认 strict fail-fast。
  - 修正一批旧测试夹具里 `config_mode="strict"` 与 `validation_mode="off"` 的自相矛盾配置，改为显式 `config_mode="compat"` 表达兼容路径。
  - 先跑 7 条直接受影响的 targeted tests，再跑 `training_loop_integration.py` 与 `training_entrypoints.py` 整文件回归，并补 compile check。
- Files created/modified:
  - `aletheia/aletheia_train.py` (updated)
  - `aletheia/tests/test_training_loop_integration.py` (updated)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 5A: Strict/Off Fixture Cleanup + Warning Sweep
- **Status:** complete
- Actions taken:
  - 全量复扫 `aletheia/` 下 `config_mode="strict"` 与 `validation_mode="off"` 的冲突夹具，确认只剩 3 处 quality-scale 测试语义冲突。
  - 将剩余 3 处夹具统一改成显式 `config_mode="compat"`，并跑 targeted + full integration 回归。
  - 为本次夹具清理创建独立回滚点：`3f13063` `Normalize compat validation-off test fixtures`。
  - 全仓复扫 `logger.warning` / `warnings.warn` / warning-continue 训练侧路径，完成显式兼容口 vs 诊断 warning vs 默认兼容边界分层。
- Files created/modified:
  - `aletheia/tests/test_training_loop_integration.py` (updated)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 5B: Training Checkpoint Restore Strictness
- **Status:** complete (`4462411`)
- Actions taken:
  - 为 training checkpoint restore 增补 strict 契约测试：默认 config drift fail-fast、requested section 缺失 fail-fast、requested target 缺失 fail-fast、explicit `optimizer_restore_mode='auto'` 兼容降级。
  - 将 `TrainingCheckpointRestorePolicy.optimizer_restore_mode` 与 CLI `--resume-optimizer-restore-mode` 默认值从 `auto` 改为 `strict`。
  - 收紧 `restore_training_checkpoint_payload()`：默认请求的 `training_state/model/optimizer_bundle/replay_buffer` 都必须同时具备 checkpoint section 与 target object；不再通过 `skip_missing_sections/objects` 静默跳过。
  - 将 `api.run_train()` 的 resume metadata 预读改成显式 `training_state only` restore policy，避免 metadata-only 路径依赖默认宽松语义。
- Files created/modified:
  - `aletheia/_training_checkpoint_schema.py` (updated)
  - `aletheia/aletheia_api.py` (updated)
  - `aletheia/tests/test_run_train_contracts.py` (updated)
  - `aletheia/tests/test_training_loop_integration.py` (updated)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| training entrypoints full suite | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_entrypoints.py -q` | 当前已落地 strict 修复无回归 | `42 passed` | ✓ |
| slow-target strict tests | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_entrypoints.py -q -k slow_target_regularization` | 新增 slow-target strict 合同测试通过 | `2 passed` | ✓ |
| wall strength strict tests | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_entrypoints.py -q -k wall_strength` | wall strength 解析失败应 fail-fast | `2 passed` | ✓ |
| WP1 entrypoint targeted tests | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_entrypoints.py -q -k 'required_post_solved_actor_anchor_clone_failures or required_post_solved_critic_anchor_clone_failures or pull_only_post_solved_anchor_without_policy_clone'` | 新增 post-solved anchor strict tests 通过 | `3 passed` | ✓ |
| WP1 loop targeted tests | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q -k 'compensation_restore_rejects_missing_required_post_solved_anchor_snapshots or capture_real_stability_anchor_rejects_required_clone_failures or export_adaptive_compensation_state_rejects_snapshot_state_dict_failures'` | restore/capture/export strict tests 通过 | `3 passed` | ✓ |
| WP1 full entrypoints file | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_entrypoints.py -q` | entrypoint full file 无回归 | `45 passed` | ✓ |
| WP1 full integration file | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q` | integration full file 无回归 | `220 passed` | ✓ |
| WP1 compile check | `uv run python -m compileall aletheia/aletheia_train.py aletheia/training/compensation_kernel.py aletheia/tests/test_training_entrypoints.py aletheia/tests/test_training_loop_integration.py` | 修改文件可编译 | `pass` | ✓ |
| WP2 targeted KL split tests | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q -k 'keeps_eval_anchor_kl_best_effort or rejects_certified_registry_kl_failures'` | eval-anchor best-effort / registry strict 分层通过 | `2 passed` | ✓ |
| WP2 full entrypoints file | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_entrypoints.py -q` | helper 改签名后 entrypoints 无回归 | `45 passed` | ✓ |
| WP2 full integration file | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q` | integration full file 无回归 | `222 passed` | ✓ |
| WP2 compile check | `uv run python -m compileall aletheia/aletheia_train.py aletheia/tests/test_training_loop_integration.py` | 修改文件可编译 | `pass` | ✓ |
| WP3 targeted telemetry tests | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q -k 'sanitize_real_stability_telemetry_rejects_invalid_numeric_values or sanitize_bootstrap_runtime_task_cert_telemetry_rejects_nonfinite_values'` | 非法 telemetry 数值默认 fail-fast | `2 passed` | ✓ |
| WP3 full entrypoints file | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_entrypoints.py -q` | telemetry strictness 不影响 entrypoints | `45 passed` | ✓ |
| WP3 full integration file | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q` | integration full file 无回归 | `224 passed` | ✓ |
| WP3 compile check | `uv run python -m compileall aletheia/aletheia_train.py aletheia/tests/test_training_loop_integration.py` | 修改文件可编译 | `pass` | ✓ |
| WP4 targeted runtime mismatch tests | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q -k 'negative_imagined_advantage_guard_downscales_actor_and_boosts_critic_in_post_solved or post_solved_stage_specific_negative_adv_guard_triggers_without_global_guard or post_solved_negative_adv_guard_latch_holds_recovery_pressure_after_sign_flip or post_solved_actor_base_return_cap_blocks_false_negative_imagined_advantage or post_solved_target_critic_actor_base_clears_false_negative_imagined_advantage or imagination_only_resume_restores_buffer_and_loop_counters or train_step_logs_real_runtime_compensation_phase'` | 直接受 `validation_mode` 契约影响的夹具全部恢复正确语义 | `7 passed` | ✓ |
| WP4 full entrypoints file | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_entrypoints.py -q` | runtime mismatch 分层不影响 entrypoints | `45 passed` | ✓ |
| WP4 full integration file | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q` | integration full file 无回归 | `225 passed` | ✓ |
| WP4 compile check | `uv run python -m compileall aletheia/aletheia_train.py aletheia/tests/test_training_loop_integration.py` | 修改文件可编译 | `pass` | ✓ |
| strict/off fixture targeted tests | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q -k 'quality_scale_further_reduces_actor_scale_under_bad_quality or piecewise_quality_scale_stays_inactive_below_mild_threshold or piecewise_quality_scale_bounds_bad_quality_without_full_collapse'` | 最后一批 compat/off 夹具统一后仍通过 | `3 passed` | ✓ |
| strict/off fixture full integration file | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q` | 夹具统一不引入整文件回归 | `225 passed` | ✓ |
| checkpoint restore targeted contracts | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_run_train_contracts.py -q -k 'resume_from_trims_histories_and_preserves_best_eval or restore_training_checkpoint_payload_rejects_optimizer_restore_on_config_drift_by_default or restore_training_checkpoint_payload_auto_downgrades_optimizer_restore_on_config_drift or restore_training_checkpoint_payload_rejects_missing_requested_training_state_section or restore_training_checkpoint_payload_rejects_missing_requested_model_section or restore_training_checkpoint_payload_rejects_missing_requested_optimizer_section or restore_training_checkpoint_payload_rejects_missing_requested_buffer_section or restore_training_checkpoint_payload_rejects_missing_requested_optimizer_target or training_state_manager_load_uses_unified_training_checkpoint_reader'` | checkpoint restore strict/default vs explicit compatibility 通过 | `8 passed` | ✓ |
| checkpoint restore model-only targeted test | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q -k 'training_state_load_materializes_lazy_world_model_components'` | 显式 model-only policy 通过 | `1 passed` | ✓ |
| checkpoint restore full run_train contracts | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_run_train_contracts.py -q` | 入口与 checkpoint 合同整包无回归 | `173 passed, 20 subtests passed` | ✓ |
| checkpoint restore full integration file | `KMP_DUPLICATE_LIB_OK=TRUE uv run pytest aletheia/tests/test_training_loop_integration.py -q` | integration full file 无回归 | `225 passed` | ✓ |
| checkpoint restore compile check | `uv run python -m compileall aletheia/_training_checkpoint_schema.py aletheia/aletheia_api.py aletheia/tests/test_run_train_contracts.py aletheia/tests/test_training_loop_integration.py` | 修改文件可编译 | `pass` | ✓ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-03-28 | `uv run pytest` 在 Python 3.14 导入 torch 时触发 libomp abort | 1 | 改用 `KMP_DUPLICATE_LIB_OK=TRUE` 运行 pytest |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5：`WP1-WP4` 已完成实现、验证与回滚提交点 |
| Where am I going? | 向用户交付 checkpoint restore 收严结果与当前剩余风险分层 |
| What's the goal? | 清理 `aletheia_train.py` 剩余训练侧 silent correctness 风险 |
| What have I learned? | training checkpoint restore 的真正风险不只是 optimizer `auto`，还包括 requested layer 缺失 section/target 时的宽松跳过；这些必须统一成同一套 strict contract |
| What have I done? | 已完成 `WP1`、`WP2`、`WP3`、`WP4`、strict/off 夹具清理，以及 training checkpoint restore strictness 收口 |
