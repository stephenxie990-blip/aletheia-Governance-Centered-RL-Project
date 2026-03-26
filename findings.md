# Findings & Decisions

## Requirements
- 先对当前项目的模型源码做完整备份，不包含数据、测试、脚本等非模型源码内容。
- 用正式的项目级方案把 SAI 主线固化下来，覆盖目标规划、阶段控制、任务编排、审查安排、回滚纪律。
- 可以使用 subagent，但主代理必须负责总编排、验收和偏航控制。
- 后续实施要“方向锁死、逐步验证、每步可回滚”，不能再回到散装补丁式推进。

## Archive Governance Round 13 Findings (2026-03-22)
- 第十三轮的正确目标不是继续做开放式“卫生治理”，而是宣布前 `1-12` 轮已形成闭环，并将后续工作收敛为“例行审计 + 定向退役”。
- 当前归档治理层已经具备稳定制度能力：
  - `docs/` 根层只保留正式入口文档
  - `docs/archive/` 已形成 `README -> 子 README -> 文件` 的导航链
  - [NAMING-CONVENTIONS.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/NAMING-CONVENTIONS.md) 与 [AUDIT-CHECKLIST.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/AUDIT-CHECKLIST.md) 已形成正式规则面
  - `bash scripts/archive_audit.sh` 已成为标准巡检入口
- 本轮新增闭环文档：
  - [归档治理闭环报告-2026-03-22.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/归档治理闭环报告-2026-03-22.md)
- 当前仓库中“可退役清理”的对象可以分成三层：
  - 已在 `Batch R1` 物理退役的对象：
    - `aletheia/tests/cartpole_last_result.json`
    - `scripts/cartpole_mbrl_full.py`
    - `scripts/cartpole_test.py`
    - `scripts/cartpole_imagination_test.py`
    - `scripts/cartpole_sweep.py`
  - 已在 `Batch R1` 一并清理的非实验卫生项：
    - `./.pytest_cache`
    - `./scripts/__pycache__`
  - 仍需按场景再清理一次的卫生项：
    - `./aletheia/__pycache__`
    - 说明：该目录在缓存清理后又因 `compileall` 验收重新生成，因此它属于“可随时清理、也会被验证流程再生”的派生物，而不是未完成项
  - 暂不应直接删除的历史代码：
    - [aletheia/_scaffold/stages.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/_scaffold/stages.py)
    - [aletheia/_scaffold/post_trigger.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/_scaffold/post_trigger.py)
    - [aletheia/_scaffold/post_entry.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/_scaffold/post_entry.py)
    - 它们虽然名字属于旧阶段语义集中区，但当前仍被 [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py) 和 [test_contract_convergence.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_contract_convergence.py) 直接消费。
- 文档层还存在一个明确的内容去重机会：
  - [统一训练入口方案与技术路径.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/undated/统一训练入口方案与技术路径.md)
  - 原 `统一训练入口方案.md` 已在 `Batch R2` 中并入并删除，当前只保留一个正文入口。
- `config_mode="compat"` 仍大量存在于测试与部分实验脚本中，不应在本轮被误判成可立即删除：
  - live 训练入口已经以 strict 解析链为主
  - 但测试口径和部分旧研究脚本仍依赖 compat 场景覆盖，因此它更像“后续工程化收口对象”，不是“现在就能硬删”的死代码。

## Deep-Water Inventory Findings (2026-03-22)
- 深水区残留不能再按“看到 `compat/legacy/scaffold` 字样就想删”的方式处理；当前至少要拆成三类：

### A. Live Path 依赖

- `_scaffold` 当前仍是 live training path 的直接依赖，不是孤儿目录：
  - [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py) 仍直接导入：
    - [post_trigger.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/_scaffold/post_trigger.py)
    - [post_entry.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/_scaffold/post_entry.py)
    - [stages.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/_scaffold/stages.py)
  - 直接证据：
    - `_scaffold` 导入区在 [aletheia_train.py:168](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py:168)
    - `_resolve_imag_compensation_phase()` 仍是主训练期相位解析真相源，在 [aletheia_train.py:9990](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py:9990)
    - `_build_imagined_batch` 侧继续调用 `_scaffold_resolve_persistence_escape_soft_floors` 与 `_scaffold_resolve_standard_soft_fallback_trigger_release_progress`，在 [aletheia_train.py:10958](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py:10958)
    - `run_step` 侧同样继续调用这些 helper，在 [aletheia_train.py:11774](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py:11774)
- [compensation.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/training/compensation.py) 也仍依赖 `_scaffold/stages.py` 做 canonicalization：
  - 导入在 [compensation.py:6](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/training/compensation.py:6)
  - `CompensationDecision.__post_init__` 和 `MinimalCompensationState.from_payload/apply_to_runtime` 都通过它做标准化，在 [compensation.py:116](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/training/compensation.py:116)、[compensation.py:221](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/training/compensation.py:221)、[compensation.py:340](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/training/compensation.py:340)
- 结论：
  - `_scaffold/` 当前不能作为 `Batch R3` 直接物理删除对象
  - 它需要先被“语义回迁/内联/降面”后，才谈删除

### B. Fail-Fast 守门

- [aletheia_api.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py) 已把正式训练入口固定在 strict 模式：
  - `run_train()` 里构造 `TrainingConfig(config_mode="strict", validation_mode="strict")`，见 [aletheia_api.py:3324](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py:3324)
- 同文件中的 legacy phase override 识别逻辑不是兼容放行，而是入口拒绝：
  - `is_legacy_compensation_phase_field()` 命中后进入 `rejected_legacy_override_keys`
  - 最终直接 `ValueError("Legacy phase overrides are no longer supported: ...")`
  - 位置在 [aletheia_api.py:3691](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py:3691) 和 [aletheia_api.py:3696](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py:3696)
- [aletheia_config.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_config.py) 中两类残留也属于 fail-fast 守门而不是“兼容通道”：
  - `is_legacy_compensation_phase_field()`：只负责识别旧字段名，在 [aletheia_config.py:118](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_config.py:118)
  - `allow_legacy_field_aliases` / removed aliases / removed fields：全部直接报错，在 [aletheia_config.py:2590](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_config.py:2590)
- checkpoint / training restore 侧还有一类“安全降级”守门：
  - [restore_training_checkpoint_payload](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/_training_checkpoint_schema.py) 在严格加载失败时，会 warning 后退到 `strict=False` 模型加载，位置在 [ _training_checkpoint_schema.py:60 ](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/_training_checkpoint_schema.py:60)
  - 这类不是旧 controller/phase 兼容，而是 checkpoint 形状不完全匹配时的恢复兜底
- 结论：
  - 这层代码当前不该被作为“可删 compat”处理
  - 真正能做的是后续判断：哪些 fail-fast 识别规则还在被测试依赖，哪些可以合并收口成单点

### C. 真正可删或可迁出的 Compat 壳

- [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py) 里有一批明确标注为 compatibility alias 的薄壳：
  - 见注释区 [aletheia_train.py:421](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py:421)
  - `_compute_bootstrap_*` / `_compute_post_transition_*` 这些 `_...` 名称只是转发到 `contracts.*`
  - 现有主要消费者是测试，如 [test_contract_convergence.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_contract_convergence.py) 的 `test_train_helper_aliases_now_point_to_contract_modules`
- `_compute_policy_outputs()` 是一个“Backward-compatible 3-return wrapper”，在 [aletheia_train.py:2974](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py:2974)
  - 如果没有外部调用点依赖旧 3-return 形状，它属于可迁移/可删候选
- `wrap_training_components()` 明确标注为 `legacy training entrypoints` adapter，在 [aletheia_train.py:7974](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py:7974)
  - 这层是典型 compat 壳，适合优先盘点其真实调用面，再决定是否删除
- `TrainingLoop.__init__` 里仍保留 `optimizer / log_fn / checkpoint_dir` 等 `Legacy aliases` 参数，在 [aletheia_train.py:8096](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py:8096)
  - 这一层比 `_scaffold` 更像真正的 Batch R3 删除对象
- [aletheia/__init__.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/__init__.py) 仍把 `_scaffold` 暴露在包级 `__all__` 中，在 [__init__.py:23](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/__init__.py:23)
  - 这不是 live path 必需，而是外部可见面扩张；等 `_scaffold` 不再被直接 import 后，可以作为第一批可删出口
- world model 侧还有一类较独立的 compat 壳：
  - `WMState.from_legacy_state()` / `WMState.coerce()` 仍支持旧 `(h_shared, z)` tuple 形状，在 [aletheia_world_model.py:4629](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_world_model.py:4629)
  - 当前真实生产调用在 [aletheia_world_model.py:7539](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_world_model.py:7539)
  - 这说明它现在不是“只有测试在用”的纯死壳；若要删，必须先把生产调用侧彻底标准化为只传 `WMState`

## Batch R3 Recommendation
- `Batch R3` 不应该先砍 `_scaffold/`
- 最小且优雅的顺序应是：
  1. `R3.1` 删除 `aletheia_train.py` 中纯 compatibility alias / test-only wrapper / legacy init 参数
  2. `R3.2` 压缩 `config_mode="compat"` 的测试面，只保留真正覆盖 fail-fast / canonical completion 所必需的用例
  3. `R3.3` 单独做 `_scaffold` 回迁评估，把 `stages/post_trigger/post_entry` 中 live 仍需部分内联到 `training/compensation.py` 或 `aletheia_train.py`
  4. `R3.4` 在 `_scaffold` 不再被 live path 与测试直接引用后，再删包级导出与物理目录
- 明确不建议：
  - 直接删 `is_legacy_compensation_phase_field()` 或 API 中的 legacy override 拒绝逻辑
  - 直接删 `WMState.coerce()` / `from_legacy_state()`，因为生产路径当前还会走到
  - 直接删 `_scaffold/` 目录，因为主训练链和 compensation state 还在依赖
- 进一步修正：
  - `_compute_policy_outputs()` 是本批最干净的 compat 删除候选；它是显式 `Backward-compatible 3-return wrapper`，且当前命中只在测试侧
  - `wrap_training_components()` 不能归入本批直接删除对象；虽然它是 adapter，但 `run_train()` 当前仍直接通过它构造 `loop_model`
  - `config_mode="compat"` 的测试面主要集中在：
    - [test_training_loop_integration.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_training_loop_integration.py)
    - [test_pure_imagination_path.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_pure_imagination_path.py)
  - 因此 `Batch R3` 的实际顺序应细化为：
    1. 先删 `_compute_policy_outputs()` 这类纯测试 compat 别名
    2. 再盘 `TrainingLoop.__init__` 的 legacy aliases，并同步收口 API/test 调用
    3. 再压 `config_mode="compat"` 测试面
    4. `wrap_training_components()` 单列为后续“去 adapter 化”迁移项
    5. 最后才进入 `_scaffold` 语义回迁

## Step 2.1 Full-Chain Audit Findings (2026-03-22)
- 当前 `Step 2.1` 失败，不能再继续被表述成“authority replacement 某个阈值不对”。
- 对比正式 baseline `v252` 与 probe `step21d/step21e` 后，第一失真点已经明确落在 `750 -> 1000` 的整条训练链，而不是单个 helper：
  - `step21d @750` 的真实行为已经高于 baseline：
    - `eval=359.8` vs baseline `324.87`
  - 但 `step21d @1000` 出现断崖：
    - `eval=141.87` vs baseline `483.6`
    - `real_reward_health=0.394`
    - `real_task_cert_gate=0.394`
    - `real_policy_kl_to_certified_anchor_mean=1.061`
    - `real_policy_kl_to_certified_registry_mean=0.983`
- 这说明系统不是“bootstrap 没接上”，而是：
  - `750` 前接法已经产生短期收益
  - `750 -> 1000` 间真实行为/任务认证先坏掉
  - critic bootstrap 之后只是把一个已经失真的后程 value regime 更稳定地送进 critic
- 从训练指标看，`step21d @1000` 的问题是“质量失真被稳定注入”，而不是“authority 太弱”：
  - `critic/critic_contract_bootstrap_effective_contact_mean=0.3104`，和 baseline `0.3083` 同级
  - `contract/bootstrap_external_authority_on_valid_mean=0.7452`，和 baseline `0.7464` 同级
  - 但：
    - `critic/value_target_gap_abs_mean=21.52` vs baseline `1.21`
    - `imag/open_loop_audit_value_gap_mean=10.02` vs baseline `0.27`
    - `wm/policy_open_loop_consistency_feature_l1_mean=0.1231` vs baseline `0.0671`
    - `wm/policy_open_loop_consistency_cosine_gap_mean=0.0722` vs baseline `0.0169`
    - `actor/corridor_semantic_common_gap_abs_mean=16.26` vs baseline `8.96`
    - `actor/corridor_semantic_inflation_excess_mean=12.11` vs baseline `4.20`
- 这组对账表明：
  - authority/contact 层已经“有力”
  - 但它所承载的 source/value/actor 语义质量显著更差
  - 因此继续只在 `bonus` 或 `floor` 上打磨，属于把坏信号更稳定地放大
- `step21d @1000` 的另一个关键事实是：
  - `critic_contract_bootstrap_source_hold_persistence_gate_mean=0.0`
  - `critic_contract_bootstrap_source_consumer_retention_gate_mean=0.0`
  - `critic_contract_bootstrap_post_transition_*` 仍全 0
  - 所以当前失败点并不在 `Step 2.2 / 2.3`，而是在它们启动之前，`Step 2.1` 自身已经把训练带入坏后程
- 当前最合理的根因表述应改为：
  - `Step 2.1` 的主病灶不是“接触面积不足”
  - 而是“quality-weighted effective contact 失真”：系统在 `750 -> 1000` 区间让低质量/高错配的 bootstrap source 进入了主 critic target 链，并同时触发了真实 eval 质量坍塌
- 因此下一刀不应继续窄盯 `compute_bootstrap_authority_source_replacement_contract()` 一处；
  - 必须把以下子链视为一个联动故障面共同审计：
    - trigger entry
    - requested floor / on-valid floor
    - source replacement quality
    - external value injection
    - actor/corridor semantic inflation
    - eval feedback degradation

## Step 2.1 v3 Execution Findings (2026-03-22)
- 已按 `Step 2.1 全链路收口执行方案 v3` 完成代码落地：
  - [authority.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/contracts/authority.py)
    - `compute_bootstrap_trigger_entry_contract(...)` 新增 `regime_quality_gate`
    - `compute_bootstrap_authority_source_replacement_contract(...)` 新增 regime-quality 输入
  - [consumers.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/contracts/consumers.py)
    - 新增 `compute_bootstrap_final_external_value_quality_contract(...)`
  - [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py)
    - 完成 `trigger -> authority replacement -> final external value clamp` 接线
    - 新增 Step 2.1 v3 telemetry
  - [test_training_loop_integration.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_training_loop_integration.py)
    - 新增 regime-veto / final-value-clamp focused tests
- 固定静态验收已全部通过：
  - `./.venv/bin/python -m compileall aletheia scripts`
  - `./.venv/bin/python -m unittest aletheia.tests.test_contract_convergence -q`
  - `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration -q`
  - `./.venv/bin/python -m unittest aletheia.tests.test_run_train_contracts -q`
- smoke run 已通过：
  - 输出目录：
    - [exp_seed42_step21f_smoke_gate_250_20260322_0056](/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_step21f_smoke_gate_250_20260322_0056)
  - 结果：
    - `eval@250 = 269.27`
  - 结论：
    - 本轮接线没有打坏训练/评估/落盘主链
- probe run 已执行到 `1000` 并判定失败：
  - 输出目录：
    - [exp_seed42_step21f_probe_to1000_gate_2500_20260322_0100](/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_step21f_probe_to1000_gate_2500_20260322_0100)
  - eval：
    - `250 = 46.67`
    - `500 = 218.13`
    - `750 = 112.67`
    - `1000 = 63.67`
  - 对比门限：
    - `750 >= 359.8` 未通过
    - `1000 > 141.87` 未通过
    - `1000 >= 220` 更未通过
- 本轮失败形态与上一轮不同，且更清楚：
  - 不是“veto 没生效”
  - 而是“veto 生效过早且过强，但又没有真正咬到 live-path 的坏 value source”
- 关键结构证据：
  - `step21f @600`：
    - `critic_contract_bootstrap_regime_quality_gate_mean = 0.327`
    - `critic_contract_bootstrap_final_value_injection_gate_mean = 0.0227`
  - `step21f @700`：
    - `critic_contract_bootstrap_final_value_behavior_health_mean = 0.0`
    - `critic_contract_bootstrap_final_value_injection_gate_mean = 0.0`
  - 说明质量闸在 `600-700` 就已几乎完全关死
- 但同时：
  - `contract/bootstrap_external_bonus_value_quality_delta_abs_mean` 在 `600/700/800/900/1000` 都是 `0.0`
  - 这说明 final value clamp 虽然 gate 很低，但实际没有改变 live-path value
  - 即：
    - `bootstrap_external_bonus_value_consumer_retained`
    - 与最终 clamp 的对照锚值
    - 在实际主路径上已经收敛成同一来源或近同一来源
- 因此本轮正式裁决：
  - `Step 2.1 v3` 失败
  - 失败点不在“是否需要 clamp”，而在“clamp 插得太晚，未命中真实坏 source”
  - 下一轮若继续留在 `Step 2.1`，主刀口必须上移到：
    - `bonus terminal truth source`
    - `bonus source value replacement`
    - 以及这些 source 是如何在 `700-1000` 提前滑向坏 regime 的

## PR1-PR6 Execution Findings (2026-03-21)
- 当前剩余算法主任务应正式拆成 4 类，而不是继续混成“最后三件事”：
  - `historical/certified authority dominance`
  - `actor/corridor/bootstrap unified consumer contract`
  - `plateau stabilization via contract outcome`
  - `controller final eradication`
- `controller` 已不再是目录级问题，而是“运行时概念残留”问题：
  - `aletheia/controller/` 目录本体几乎已空
  - 但 `controller_stage`、`controller_info`、`imag/controller_*` telemetry、`MinimalCompensationState.apply_to_controller()` 仍让 controller 作为一等语义活在主训练链里
- `post-transition capture` 当前存在 canonicalization 缺口：
  - `contracts/authority.py` 已有 `compute_post_transition_retention_capture(...)`
  - 但 `aletheia_train.py` 中仍保留了一段 duplicated 的 `current_authority / historical_authority / capture_support / certified_capture_source` 手工公式
  - 因此必须先做 `authority core dominance + train glue canonicalization`
- `PR-1` 的结构性切口已经明确：
  - 不能只继续调 `capture_support`
  - 必须让 `historical_authority` 在明确条件下成为真实 winner，而不是永远被 `current_authority` 下界锁成 contemporaneous mirror
- `PR-1` 已完成并通过 focused 验收：
  - `RetentionCaptureDecision` 已扩成可审查裁决对象，新增：
    - `historical_takeover_gate`
    - `dominant_source`
    - `dominance_margin`
    - `is_historical_winner`
    - `release_blocked`
  - `compute_post_transition_retention_capture(...)` 现已支持：
    - 有支持时历史 authority 真实胜出
    - 释放压力过高且支持不足时回退到 current authority
  - 新增 focused tests 已证明：
    - supported history 可成为 winner
    - 弱支持 + 高 release pressure 不会形成伪 dominance
- `PR-2` 的执行重点已锁定：
  - `aletheia_train.py` 仍保留 duplicated 的 post-transition authority 手工计算
  - 下一阶段必须把 train glue 改成只消费 canonical `RetentionCaptureDecision`
- `PR-2` 已完成并通过 focused 验收：
  - `aletheia_train.py` 现已显式导入并消费 `compute_post_transition_retention_capture(...)`
  - train 中那段 duplicated 的 `current_authority / historical_authority / capture_support / certified_capture_source` 手工公式已删除
  - 搜索确认旧重复公式已不再存在于主训练文件
  - focused 回归 `19 tests ... OK`
- `PR-3` 的执行重点已锁定：
  - 需要把 consumer 看到的 post-transition 胜出语义收敛成单一 contract view
  - 为后续 plateau stabilization 和 controller 退场准备统一消费入口
- `PR-3` 已完成并通过 focused 验收：
  - `contracts/core.py` 新增 `ConsumerContractView` 与 `select_consumer_contract_view(...)`
  - train 中的 corridor / bootstrap 观察面已接入统一 consumer-facing selection
  - 新增测试证明 task-certified 合同可优先于 geometry-only 合同
- `PR-4` 已完成并通过 focused 验收：
  - `contracts/consumers.py` 现已在 `capture.is_historical_winner` 时直接抬升 floor/persistence gate
  - plateau stabilization 的最小闭环已绑定到 contract winner，而不是旧 controller 阶段
- `PR-5` 已完成并通过 focused 验收：
  - `MinimalCompensationState` payload 已切换为 `runtime_phase`
  - 旧 `controller_stage` payload 现在 fail-fast
  - roundtrip / restore / run_train contract focused tests 通过
- `PR-6` 已完成并通过工程验收：
  - `aletheia/controller/` 目录已物理删除
  - `aletheia_train.py` 正式 metrics 面已清掉大批 `post_entry / handoff / fallback` 的 controller telemetry
  - `compileall aletheia`、focused tests、以及真实 smoke gate run 均通过
- 本轮真实 smoke gate run：
  - 输出目录：[outputs/exp_seed42_v251_pr1_pr6_smoke_gate_250_20260321_1924](/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v251_pr1_pr6_smoke_gate_250_20260321_1924)
  - 结果：`250 -> 37.20`
  - 结论：训练链路在 `PR1-PR6` 之后仍能正常启动、评估、保存 best/final/checkpoint/resume 产物
- 新增治理裁决：
  - `controller` 最终目标不再是“降级”
  - 而是“rename-out + protocol cutover + physical eradication”

## Cleanup Findings (2026-03-20)
- `Batch 0 -> Batch 5` 已按 stop-loss cleanup track 完整执行，`Phase 5/6` 不再停留在“计划中”。
- 真正需要保留的 live controller 语义已经收敛为四态：
  - `idle`
  - `trigger`
  - `persistence`
  - `post_solved`
- `post_entry / internal_post_entry / handoff / entry_probe / pretrigger / persistence_release / standard soft fallback / late-trigger rescue` 已不再被允许主导训练行为。
- `Batch 4` 暴露了一个真实工程问题：`TrainingConfig` 先删掉了 `persistence / post_solved` 字段，但 `aletheia_train.py` 的最小补偿主线还在读这些配置。
- 因此最终收口不是“把所有 controller 相关配置删空”，而是：
  - 删掉旧阶段机与脚手架专属配置入口
  - 补回 live path 真实还需要的最小 `persistence / post_solved` 字段
- `aletheia/tests/test_training_loop_integration.py` 已按新主线收缩：
  - 删除了依赖旧 controller 阶段机的大批测试块
  - 保留并验证了最小主线所需的 `trigger / persistence / persistence tail mismatch / persistence base return cap`
  - 旧 controller 只剩 resume-state 兼容属性断言，不再承担行为验收职责
- 最终验证口径固定为：
  - `compileall aletheia`
  - `run_train_contracts` 的 resume/checkpoint 合同测试
  - 5 个最小 controller 行为回归
- 最终验证结果：
  - `./.venv/bin/python -m compileall aletheia` 通过
  - `./.venv/bin/python -m unittest ...` 聚焦回归共 `Ran 6 tests ... OK`

## Contract Convergence Findings (2026-03-21)
- `contracts/` 目录已经成为合同收敛后的 authoritative 模块边界：
  - [core.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/contracts/core.py)
  - [evidence.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/contracts/evidence.py)
  - [certification.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/contracts/certification.py)
  - [authority.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/contracts/authority.py)
  - [consumers.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/contracts/consumers.py)
  - [telemetry.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/contracts/telemetry.py)
- `semantic_contract.py` 已退回 compatibility layer；`SemanticContract / SemanticArbiter` 的 authoritative 定义转移到 [contracts/core.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/contracts/core.py)
- `aletheia_train.py` 中原先散落的 authority / consumer helper 现在由模块级别别名重定向到 `contracts/authority.py` 与 `contracts/consumers.py`，因此：
  - 主训练流与旧测试入口不必重写
  - 运行时行为已经切到新模块
  - 旧 helper 物理定义仍留在大文件里，作为最后一层兼容壳而非 authoritative 实现
- `MinimalCompensationState` 已落到 [training/compensation.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/training/compensation.py)，当前 resume/export 已新增 `minimal_compensation_state`：
  - 导出位置：[aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py)
  - 保存位置：[aletheia_api.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py)
- `TrainingConfig` 已新增 4 组合同域访问面，旧平铺字段仍保留，避免 breaking change：
  - `certification_contract_config()`
  - `authority_contract_config()`
  - `consumer_contract_config()`
  - `compensation_config()`
- bootstrap summary telemetry key 列表已从 [aletheia_api.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py) 的超长硬编码字面量收敛到 [contracts/telemetry.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/contracts/telemetry.py)
- 当前工程判断：
  - 合同层的“7-8 层合同感”已经从训练大文件的结构复杂度，收缩到少数模块边界和显式对象
  - 行为问题的主因仍不是 `WM core`，而是 Phase 4 主线上 authority / bridge / eval protocol 的行为闭环没有被修通

## Research Findings
- 核心模型与训练源码位于 `aletheia/` 目录，主文件包括：
  - `aletheia_train.py`
  - `aletheia_world_model.py`
  - `aletheia_actor_critic.py`
  - `aletheia_config.py`
  - `aletheia_foundation.py`
  - `aletheia_api.py`
  - `aletheia_twohot.py`
  - `aletheia_vec_env.py`
  - `controller/` 子模块
  - 若干 `post_entry_*` 审计模块
- `aletheia/tests/` 属于测试，不在本次正式备份范围内。
- 项目当前已经存在大量训练输出、checkpoint、tmp 与脚本目录，因此必须明确排除：
  - `aletheia/tests`
  - `scripts`
  - `outputs`
  - `checkpoints`
  - `tmp`
  - `__pycache__`
  - `*.pyc`
- 当前可恢复的正式源码快照已创建在：
  - `/Users/zhangsan/Desktop/缸中之脑v5.6/备份/SAI项目源码备份-2026-03-17-143958`
- 子代理独立审计确认：
  - “严格 source-only backup” 的正确定义是 `aletheia/` 包内可导入 `.py` 源文件
  - `post_entry_*.py` 与 `controller/` 子模块必须保留在源码备份中
  - `tests / scripts / outputs / checkpoints / tmp / .venv / docs / 备份` 都不应混入源码快照
- 项目当前不是 git repo，也未发现 `pyproject.toml`、`requirements.txt`、`poetry.lock`、`uv.lock` 等锁定文件，因此仅靠源码快照不足以完整复现环境。
- Step 1 的最小落点已确认：
  - 新增独立模块 `aletheia/semantic_contract.py`
  - 保持为纯数据结构与本地结构校验，不接训练逻辑
  - 测试采用现有 `unittest` 风格，独立文件 `aletheia/tests/test_semantic_contract.py`
- Step 1 结构 cut 已实际落地：
  - `SemanticContract` 已显式承载 `value / source / coverage / confidence / authority / trust / task_agreement / registry_support / semantic_debt / freshness / certified_by`
  - 本地校验已覆盖张量形状一致、[0,1] 有界约束、外生 source 识别与 geometry/task 认证通道区分
  - `summary()` 已作为行为中立的摘要接口存在，可供同阶段后续旁路观测接入复用
- Phase 1B 旁路观测接入已实际落地：
  - `clean anchor / corridor / bootstrap internal / bootstrap external` 四类 contract 摘要已在 imagined batch 中可见
  - 训练日志已新增 `contract/clean_anchor_*`、`contract/corridor_*`、`contract/bootstrap_internal_*`、`contract/bootstrap_external_*`
  - 所有 contract 旁路张量都经 `detach()` 进入摘要与日志，不参与训练梯度或权重更新
- 零行为改动的关键边界已确认：
  - 不在 `aletheia_train.py` 中接线
  - 不改现有指标名、不新增训练期副作用
  - 不让 `SemanticContract` 依赖 controller / train / API 层
- 本地验证已确认 Step 1 仍停留在结构层：
  - `./.venv/bin/python -m unittest aletheia.tests.test_semantic_contract` 通过
  - `./.venv/bin/python -m unittest aletheia.tests.test_controller_signals` 通过
  - `./.venv/bin/python -m compileall aletheia/semantic_contract.py` 通过
- Phase 1 完整验证已通过：
  - `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_build_imagined_batch_emits_phase1b_semantic_contract_observation_summaries` 通过
  - `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_train_step_prefers_bootstrap_rewritten_critic_target_branch_when_available` 通过
  - `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_clean_mix_tracks_critic_trust_independently_of_actor_contract_path` 通过
  - `./.venv/bin/python -m compileall aletheia/semantic_contract.py aletheia/aletheia_train.py` 通过
- 子代理对 Phase 2 的只读审计确认：
  - `critic_contract_authority / critic_contract_trust / critic_contract_bootstrap_semantic_debt / critic_contract_bootstrap_contact_drive` 与 `late/precontact gate` 是最自然的 takeover floor 输入
  - `anchor coverage / anchor confidence / row support / dense surface gain` 更适合作为 modulation-only 输入，而不是 floor veto
  - 现有指标名 `critic_contract_bootstrap_clean_mix_mean / critic_contract_bootstrap_contact_floor_mean / critic_contract_bootstrap_mix_surface_mean / critic_contract_bootstrap_contact_modulation_mean` 可以保留，只需重映射其内部语义
- Phase 2 权威接管合同已实际落地：
  - `aletheia/semantic_contract.py` 新增 `SemanticArbiterDecision` 与 `SemanticArbiter`
  - `aletheia_train.py` 中 critic bootstrap 已从“单段 surface 公式”改写为 `takeover_floor -> modulation_bonus -> external_authority`
  - `takeover_floor` 只由 `precontact/late gate + authority + semantic_debt` 决定
  - `coverage / confidence / support / dense surface` 只影响 `modulation_bonus`
  - actor / corridor / controller 主行为链路未改动，Phase 2 仍是纯 critic bootstrap-only 改写
- Phase 2.5 第一刀已实际落地：
  - 在 `aletheia_train.py` 的 bootstrap sustained-contact 段新增 `bootstrap_precontact_persistent_bonus`
  - 该 bonus 只由 `precontact_gate * surface_memory * remaining_authority_budget` 决定，不引入 actor/corridor 新依赖
  - 作用是让已建立过的 bootstrap surface 在 pre-contact 阶段持续占有一部分剩余外生接触预算，而不是只留下“状态记忆”却不改变最终 `clean_mix`
  - 新增观测指标 `critic/bootstrap_persistent_bonus_mean` 与 `critic_contract_bootstrap_persistent_bonus_mean`
  - 现有失败用例已转绿，证明 `surface_memory` 现在不仅被记录，而且会把 `sustained_floor / clean_mix` 真正抬高
- Phase 2.5 暂时确认的结构病灶：
  - 集成测试辅助场景里 `critic_contract_authority_mean=1.0` 的直接原因不是 `surface_memory` 失效，而是 `task_corridor_gate=0` 使 `semantic_mismatch=1.0`，进而把 `authority/contact_drive` 顶满
  - 这说明原先的持续接触测试刚好落在一个“fresh 已经到上限”的姿态上；但它同样暴露出旧合同里 `surface_memory` 对最终输出没有额外抓手的问题
  - `persistent_bonus` 的设计正是为了解决这个缺口：即便当前 contact drive 已高，历史 surface 仍能在剩余 authority budget 上形成持续加成
- Phase 2 定向验证已通过：
  - `./.venv/bin/python -m unittest aletheia.tests.test_semantic_contract` 通过
  - `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_clean_mix_tracks_critic_trust_independently_of_actor_contract_path aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_clean_mix_no_longer_collapses_under_partial_anchor_coverage aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_semantic_debt_persists_when_reward_alarm_recovers_but_semantic_gap_remains aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_train_step_prefers_bootstrap_rewritten_critic_target_branch_when_available aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_contract_stays_dormant_before_late_stage_gate aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_precontact_floor_engages_before_late_gate_opens aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_build_imagined_batch_emits_phase1b_semantic_contract_observation_summaries` 通过
  - `./.venv/bin/python -m compileall aletheia/semantic_contract.py aletheia/aletheia_train.py` 通过
- Phase 2.5 当前局部验证已通过：
  - `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_surface_memory_raises_sustained_floor_before_late_gate_opens` 通过
  - `./.venv/bin/python -m unittest aletheia.tests.test_semantic_contract aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_clean_mix_tracks_critic_trust_independently_of_actor_contract_path aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_clean_mix_no_longer_collapses_under_partial_anchor_coverage aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_semantic_debt_persists_when_reward_alarm_recovers_but_semantic_gap_remains aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_train_step_prefers_bootstrap_rewritten_critic_target_branch_when_available aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_contract_stays_dormant_before_late_stage_gate aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_precontact_floor_engages_before_late_gate_opens aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_surface_memory_raises_sustained_floor_before_late_gate_opens aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_build_imagined_batch_emits_phase1b_semantic_contract_observation_summaries` 通过
  - `./.venv/bin/python -m compileall aletheia/aletheia_train.py aletheia/tests/test_training_loop_integration.py` 通过
- `v196` 真实 run 已完成，结论明确：
  - 行为曲线：`250=15.4 -> 500=57.0 -> 750=402.4 -> 1000=129.0 -> 1250=70.0 -> 1500=17.4 -> 1750=167.6 -> 2000=27.6 -> 2250=26.4 -> 2500=39.0`
  - 这说明 Phase 2.5 第一刀没有通过；它修到了“接触能持续”，但没有修到“中后段不易先掉坑”
  - fake corridor 仍然存在：`2250` 时 `corridor occupancy=1.0 / persistence=1.0 / registry KL 很低`，但 eval 只有 `26.4`
  - 最关键的结构证据不是 authority 不足，而是 authority surface 分配错位：
    - `step=2000`：`bootstrap_mix_surface_mean=0.8`
    - 但 `bootstrap_effective_contact_mean=0.123`
    - 同时 `bootstrap_dense_effective_contact_mean=0.524`
    - 说明大部分 external authority 仍铺在 dense fallback，而不是真实 coverage-backed anchor
  - `step=2250/2500` 也保持同样形态：dense effective contact 显著高于真实 anchor contact，行为没有恢复
- 由 `v196` 得出的下一刀原则已经锁定：
  - 不再继续加 `gate / floor / bonus`
  - 不回 actor / corridor / controller
  - 只在 bootstrap 内重写 `mix surface` 的预算分配律：
    - `authority / trust / semantic_debt` 先决定总外生接触预算
    - `anchor_coverage / anchor_confidence / dense_surface_gain` 只调制预算如何在真实 coverage 与 dense fallback 间分配
  - 新增直接观测量：`bootstrap_anchor_contact_density_mean` 与 `bootstrap_dense_contact_density_mean`
- Phase 2.5 第二刀 `surface budget rebalance` 已落地：
  - `aletheia_train.py` 不再把 `external_authority` 直接均匀作为最终 `mix_surface`
  - 当前实现改成：
    - 先保留 dense 最小底座
    - 再把可重分配 budget 按 coverage/confidence/dense surface 做归一化分配
    - 使 coverage-backed anchor 获得更高的 contact density，而不把 coverage 重新抬成是否能接触的 veto gate
  - 本地探针已显示：
    - `clean_mix_mean=0.8`
    - `mix_surface_mean≈0.653`
    - `anchor_contact_density≈0.8`
    - `dense_contact_density≈0.505`
    - 说明 surface 分配已明显向 coverage-backed anchor 倾斜
- `v197` 真实 run 已完成，结论比 `v196` 更收敛：
  - 行为曲线：`250=15.4 -> 500=57.0 -> 750=402.4 -> 1000=51.6 -> 1250=39.6 -> 1500=99.2 -> 1750=91.2 -> 2000=96.0 -> 2250=22.2 -> 2500=22.0`
  - 这说明 `surface budget rebalance` 修到了“掉坑后没那么容易彻底趴底”，但没有修到“中后段不容易先掉坑”
  - 最关键的新病灶是：当前 `base_mix_surface` 仍被当成平均预算去再分配，而不是 local base authority
    - `1500/1750/2000` 的恢复说明 direction 对了
    - `2250/2500` 再次塌到 `22.x` 说明 contract base 仍可被 surface 层再次预算化，持续接触不具备真正刚性
  - 因而下一刀不再是调预算，而是改 surface law：
    - `authority / trust / semantic_debt` 决定 local base authority
    - `anchor_coverage / dense_surface_gain / support` 只在剩余 headroom 上做单调加码
    - surface 层不得再把 base authority 向下稀释
- Phase 2.5 第三刀 `monotonic base-preserving surface lift` 已落地：
  - `base_mix_surface` 现在作为 bootstrap local base authority 保留下来
  - anchor/dense 的 priority surface 只分配 bonus headroom，不再重新瓜分 base
  - 新增显式观测：
    - `bootstrap_surface_bonus_mean`
    - `bootstrap_surface_floor_violation_mean`
  - 新增结构性断言：
    - `surface_floor_violation_mean == 0`
    - `mix_surface_mean >= base_mix_surface_mean`
- 当前已确认的结构性收益：
  - `clean_mix/contact_floor` 与 `contact_modulation` 已从同一条乘法链中拆开
  - bootstrap external authority 现在来自显式仲裁结果，而不是隐含的 surface 混合副产物
  - 老指标名与 Phase 1B 合同旁路摘要保持连续，可直接和历史 run 对账
- `v195` 真实训练验证已完成，关键事实如下：
  - 行为曲线：`750=402.4 -> 1750=37.8 -> 2250=500.0 -> 2500=93.8`
  - contract 链路：`1250+` 后 `bootstrap_clean_mix_mean=0.2`、`mix_surface=0.8`、`contact_floor=0.8` 持续稳定，说明接管合同已真正进入 bootstrap
  - modulation 分支：post-contact 区间 `bootstrap_contact_modulation_mean` 基本恒为 `0.0`，当前有效作用几乎全部来自 floor
  - critic 语义：post-contact 区间 `real_mc_value_gap_abs_mean` 平均约 `21.45`、峰值约 `33.45`；`teacher value -> short return` 平均约 `32.30`；`imag value -> short return` 平均约 `36.82`
  - fake health 仍存在：`2250=500.0` 与 `2500=93.8` 时 `corridor occupancy=1.0`、`persistence=1.0`，说明几何认证仍不能代表任务语义健康
  - 结论：Phase 2 已修通“稳定接触”，但尚未修通“持续主导”；当前合同更像恢复合同，还不是持续接触合同
- `v205/v206` 的窄口径 Phase 3 安全验证已经收敛出新的上游病灶：
  - `v205` 的回归来自 actor-side 常驻 soft blend，不是 unified contract 主通路本身
  - `v206` 虽恢复到更健康的轨迹形态，但新增 actor-side tail relief 全程 dormant
  - 最关键的结构证据是：`critic/bootstrap_late_gate = 0.0`、`critic/bootstrap_mix_surface_mean = 0.0`、`actor/actor_contract_unified_tail_relief_mean = 0.0`
  - 这说明当前主病灶不在 actor，而在 upstream bootstrap trigger surface 没有真正打开
- `bootstrap trigger surface rewrite` 已实际落地：
  - 保留 `critic_contract_bootstrap_eval_gate` 作为观测项
  - 新增 `critic_contract_bootstrap_trigger_gate = max(eval_gate, task_degradation, release_guard_mean, negative_adv_pressure_mean, trigger_surface_mean)`
  - `late_gate = step_gate * trigger_gate`
  - `precontact_gate = precontact_step_gate * trigger_gate`
  - `semantic_alarm / reward_alarm` 改为由 `precontact_gate` 驱动，使 semantic debt 可在旧 eval gate 未开时提前累积
  - 新增观测指标：
    - `critic/bootstrap_eval_gate`
    - `critic/bootstrap_trigger_gate`
    - `critic/bootstrap_negative_adv_pressure_mean`
    - `critic/bootstrap_trigger_surface_mean`
    - 以及对应 `critic_contract_*` batch 摘要
- trigger-surface rewrite 的本地验证已通过：
  - `./.venv/bin/python -m compileall aletheia/aletheia_train.py aletheia/tests/test_training_loop_integration.py` 通过
  - `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_contract_stays_dormant_before_late_stage_gate aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_trigger_surface_can_open_without_eval_gate_once_time_ready aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_precontact_floor_engages_before_late_gate_opens aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_train_step_prefers_bootstrap_rewritten_critic_target_branch_when_available` 通过
  - `./.venv/bin/python -m unittest aletheia.tests.test_semantic_contract aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_clean_mix_tracks_critic_trust_independently_of_actor_contract_path aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_surface_bonus_preserves_contract_base_authority aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_clean_mix_no_longer_collapses_under_partial_anchor_coverage aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_semantic_debt_persists_when_reward_alarm_recovers_but_semantic_gap_remains aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_train_step_prefers_bootstrap_rewritten_critic_target_branch_when_available aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_contract_stays_dormant_before_late_stage_gate aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_trigger_surface_can_open_without_eval_gate_once_time_ready aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_bootstrap_precontact_floor_engages_before_late_gate_opens aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_build_imagined_batch_emits_phase1b_semantic_contract_observation_summaries` 通过
- 下一条真实验证 run 已固定为 `v207`：
  - overrides：`/Users/zhangsan/Desktop/缸中之脑v5.6/tmp/v207_phase25_bootstrap_trigger_surface_rewrite_overrides.json`
  - 目标：优先验证 `eval_gate=0` 时 `trigger_gate / late_gate / mix_surface` 是否仍能在时间成熟后打开
  - 归因纪律：保持 `Phase 2.5 bootstrap-only` 基线，不把新的 actor 变量混入这一轮验证
- `v207` 的正式裁决已经完成：
  - 误配置 run `_01` 未带 eval，被停止并仅保留为结构旁证
  - 官方 run `_02` 结果为 `250=15.4, 500=57.0, 750=402.4, 1000=138.4, 1250=50.0, 1500=35.4, 1750=92.6, 2000=297.2, 2250=25.8, 2500=65.2`
  - `700/750` 时 `bootstrap_eval_gate=0` 但 `bootstrap_trigger_gate≈0.86`，证明 trigger surface 已摆脱旧 eval gate 单点卡死
  - `1250+` 后 `trigger_gate=1 / late_gate=1 / precontact_gate=1` 稳定成立，说明“门开了”这件事已经修对
  - 但 `2250` 仍深跌到 `25.8`，`2500` 只回到 `65.2`，因此 Phase 2.5 不能判定通过
  - 更关键的是：`clean_mix_mean≈0.48~0.53` 已经不低，但 `effective_contact_mean` 在尾段只有 `0.13~0.19`，同时 `anchor_coverage_mean` 从 `0.525` 掉到 `0.150`
  - 这说明当前剩余病灶不再是 trigger timing，而是 coverage 下行时关键态外生 authority 的持续主导仍会塌缩
- `v208` 的实现假设已经落地：
  - base floor 不再让 `anchor_coverage` 决定“关键态能不能接管”，而改由 `anchor_valid_mask` 决定 valid positions 上的持续接管存在性
  - `anchor_coverage / confidence` 只保留在 bonus priority 与 modulation/support 链上，用来决定“已建立接管后还能加多少码”
  - `bootstrap_surface_state` 不再跟随全局 `mix_surface_mean`，而改为跟随 valid + debt-high 区域的 `focus_effective_contact_mean`
  - 这刀的目标不是让早期开门更早，而是让中后段一旦进入 debt-high 且 anchor-valid 的状态，surface memory 能追踪真正关键的接触质量
- 新增的结构观测已经可用：
  - `critic/bootstrap_anchor_valid_mean`
  - `critic/bootstrap_debt_focus_mean`
  - `critic/bootstrap_anchor_valid_effective_contact_mean`
  - `critic/bootstrap_focus_effective_contact_mean`
- `v208` 正式 run 已启动：
  - overrides：`/Users/zhangsan/Desktop/缸中之脑v5.6/tmp/v208_phase25_bootstrap_effective_contact_persistence_rewrite_overrides.json`
  - 输出目录：`/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v208_phase25_bootstrap_effective_contact_persistence_rewrite_2500_20260318_01`
  - 当前早段观测：`250` 前后 `trigger_gate` 在抬升，但 `precontact/late gate` 尚未打开，因此 `focus_effective_contact_mean=0` 仍属正常
- `v208` 的正式结果已经确认：
  - eval：`250=15.4, 500=57.0, 750=402.4, 1000=172.4, 1250=113.8, 1500=229.4, 1750=125.6, 2000=17.8, 2250=500.0, 2500=51.2`
  - 相比 `v207`，中段 `1000-1750` 明显改善，说明 bootstrap 的 valid/debt-high 持续接触合同并非无效修改
  - 但 `2000` 深跌、`2250` 又 solved、`2500` 再掉，说明系统仍存在强烈的错误盆地往返
  - 更重要的是，bootstrap 侧的关键受控量已经打中：
    - `1000`: `anchor_valid_effective_contact_mean=0.906`, `focus_effective_contact_mean=0.938`
    - `1500`: `1.000 / 1.000`
    - `2000`: `1.000 / 1.000`
  - 这意味着“critical valid positions 上外生 authority 持续接触不足”已不再是首要解释
  - 与之相对，行为侧却出现更硬的假 corridor 证据：
    - `2000` eval `17.8` 时 corridor occupancy/persistence 仍约 `0.955/0.95`
    - `2500` eval `51.2` 时 occupancy/persistence 已是 `1.0/1.0`
  - 因此当前最硬判断是：主病灶正在从 bootstrap 持续接触，迁移到 fake corridor / geometry-only certification / 下游统一语义消费错位
- `v209` 的最小 reward-semantic registry recertification 已实际落地：
  - `set_external_eval_feedback()` 现在会把 `real_reward_health / real_reward_degradation` 写入 real stability telemetry
  - registry support 被显式拆成 raw geometry support 与 reward-semantic recertified support
  - 下游实际消费的 `behavior_policy_certified_registry_support_mask` 变为 `raw_geometry_support * real_reward_health`
  - actor / critic 主公式保持冻结，因此该 cut 仍然是纯 corridor/certification 层
- `v209` 的结构结论是“诊断通过、验收失败”：
  - eval 轨迹为 `250=15.4, 500=57.0, 750=402.4, 1000=172.4, 1250=86.8, 1500=133.4, 1750=174.8, 2000=40.4, 2250=500.0, 2500=43.0`
  - `1000 eval` 回撤后，训练侧 `behavior_policy_certified_registry_support_fraction` 会从 geometry 的 `0.508` 压低到 `0.218`
  - `1250 eval=86.8` 时 health 仅 `0.216`，对应 effective support 已被压到 `0.143 / 0.071`
  - 这说明 fake corridor 的“去认证”链路第一次真正接通，系统不再在低回报时无条件把 geometry registry 当成高可信输入
- `v209` 同时暴露出当前 Phase 4 实现的核心不足：
  - 它本质上是 eval-lagged 的 reward-health 折扣器，而不是独立 task-cert 主合同
- `2026-03-20` 的 Phase 5/6 清理盘点已经固定：
  - `live training path`：`aletheia_train.py` 中按 `controller_stage` 分段控制 `continue cap / base_return_cap / actor scale / hysteresis` 的逻辑仍直接改训练行为，当前不可直接删除
  - `audit-only`：`post_entry_audit.py`、`post_entry_template_compare.py`、`post_entry_precursor_audit.py`、`controller/relation_audit.py`、`controller/signal_audit.py` 只消费 run 产物做审计和报表，可作为第一批删除对象
  - `signal/schema-only`：`controller/signals.py`、`controller/schema.py`、`controller/invariants.py` 主要服务解释与校验，可作为第二批删除对象
  - `config/test surface`：`aletheia_config.py` 与 `aletheia/tests/test_training_loop_integration.py` 是主要跟随面，另有 `test_controller_*`、`test_post_entry_*`、`test_run_train_contracts.py` 需要同步清理
- 已形成正式清理文档：
  - `/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase5-6-清理收口执行计划-2026-03-20.md`
- 推荐清理批次已锁定：
  1. 冻结基线与 manifest
  2. 删除 audit-only 模块与 API 出口
  3. 删除 signal/schema/invariants 解释层
  4. 折叠 live controller 为最小补偿层
  5. 清理 config / test / metrics 表面
  6. 做最终验证与对账
  - `2250 eval=500.0` 之后，训练侧 `2300-2500` 会重新满额信任 registry，直到 `2500 eval=43.0` 才再次意识到崩落
  - 因此它能拆掉错误背书，却不能在同一 eval 区间内阻止再次掉坑
- `v209` 进一步强化了新的主判断：
  - bootstrap 持续接触合同已经不是首要短板
  - 现在缺的是“geometry cert 不能直接推出 task truth”的正式 corridor 主合同
  - 以及一条比 eval 更即时的 task-agreement / reward-agreement 输入，用来避免 reward-health 的单区间滞后
- Phase 4 的最小 `explicit task cert` 通道已经接通，并通过了训练循环级别回归：
  - `_compute_task_cert_gate(...)` 现在将 imagined `task_corridor_gate + task_corridor_confidence` 与 real-side `real_task_cert_gate` 汇成独立 task-cert 信号
  - 最终 `behavior_policy_certified_registry_support_mask` 已从旧的 `reward_semantic_registry_support_mask` 切换为 `min(geometry_registry_support_mask, task_cert_support_mask)`
  - 这意味着：几何 registry support 可以保持高位，但只要 task corridor 明确分歧，最终有效认证就必须下行
  - 对应集成测试已覆盖三种关键面：
    - 几何 support 高但 task mismatch 时，actor contract trust 仍应激活
    - real reward health 回撤时，real-side task cert 会同步下行
    - task corridor disagrees 时，即便 geometry support 与 reward-semantic support 仍高，final certified support 也必须下降
- 这次实现还顺手暴露了两个真实但次级的问题，并已一并修复：
  - `real_task_cert_gate` 与 `behavior_preinflation_mask` 的取值顺序原本在 task-cert 统计之后，导致接线顺序存在 `UnboundLocalError` 风险
  - `behavior_f_policy_*` 这组纯观测指标此前被意外挂在 registry availability 下面，导致无 registry 时错误掉成 `0.0`
- 当前仍保留两项 Phase 4 语义债，需要在后续观测/图表清理时处理：
  - `task-cert` 相关命名里仍混用 `gate / cert / support`，容易让仪表盘把“最终 cert 支持”误读成“原始 gate”
  - `geometry_registry_support_mask` 与 `reward_semantic_registry_support_mask` 还没有和 `task-cert` 一样对称导出 raw tensor，逐点对账面仍不够硬
- `v210` 的正式 run 结论是“结构通过、阶段未过”：
  - eval 轨迹为 `250=15.4, 500=73.0, 750=340.6, 1000=500.0, 1250=260.8, 1500=172.4, 1750=128.6, 2000=217.2, 2250=178.2, 2500=66.6`
  - 与 `v209` 相比，显式 task-cert 已经更早进入最终认证主链路：
    - `500`: geometry `0.117`，final certified `0.110`
    - `750`: geometry `0.075`，final certified `0.071`
    - `1000`: geometry `0.550`，final certified `0.526`
  - 这证明“几何认证不能单独推出任务真理”的主合同已经真正生效，而不再只是观测层伪信号
  - 但它前中段的压制幅度仍只有约 `4%-6%`，属于“早介入的弱收缩”，还不是强任务接管
  - 更关键的是，`critic/real_mc_value_gap_abs_mean` 在 `1000-2000` 区间仍升到 `22.98 / 30.61 / 42.84 / 49.84` 这一高位，说明显式 task-cert 没有阻断更深层的 critic 语义膨胀
  - 因此 `v210` 的硬结论是：
    - fake corridor 的最终认证，确实比 `v209` 更早开始被压
    - 但它还没有被“更早且更强地压住”
    - 当前下一刀仍应留在 Phase 4，补更即时、更强的 task-agreement / reward-agreement 主输入，而不是回去继续堆 reward-health 乘法或过早切 Phase 3
- `v211` 的 immediate reward-agreement cut 已实际落地：
  - `aletheia_train.py` 中新增 `_compute_reference_reward_agreement(...)`
  - real-side task-cert 现在由 `min(real_eval_gate, real_batch_reward_agreement)` 驱动，而不再完全等待下一次 eval 更新
  - `aletheia_config.py` 新增 `adaptive_imag_task_cert_reward_agreement_quantile`
  - 日志新增：
    - `behavior_policy_task_cert_real_eval_gate_mean`
    - `behavior_policy_task_cert_real_reward_agreement_mean`
    - `behavior_policy_task_cert_real_gate_mean`
  - 集成回归 `test_registry_support_uses_immediate_real_batch_reward_agreement_before_next_eval` 已确认：即便 eval-side gate 仍为 `1.0`，即时 real-batch reward-agreement 也能提前压低 final certified support
- `v211` 的正式 run 结论是“结构更真，但行为仍未过关”：
  - eval 轨迹为 `250=15.4, 500=57.0, 750=309.8, 1000=117.6, 1250=143.2, 1500=124.8, 1750=81.4, 2000=424.0, 2250=16.8, 2500=95.8`
  - 相比 `v210`，它第一次真正做到了“同一 eval 区间内”的任务去认证：
    - `500`: `real_eval_gate=1.0`，但 `real_reward_agreement=0.852`，`final certified support=0.170`
    - `750`: `real_eval_gate=1.0`，但 `real_reward_agreement=0.246`，`final certified support=0.0368`
    - `1000-2500`: `real_task_cert_gate` 基本稳定在 `0.035~0.048`
  - 它同时也证明了当前主合同仍缺恢复律：
    - `critic/real_mc_value_gap_abs_mean` 被压在 `4~10`，说明 critic 语义膨胀已被明显遏制
    - 但 `final certified support / cert-geom ratio` 几乎整段贴近 `0.03~0.05`
    - 系统因此不是回到高质量任务吸引子，而是被压进“低错但低效”的保守盆地，只在 `2000` 短暂冲高后再次崩落
  - `v211` 的硬结论是：
    - 现在 Phase 4 的主问题不再是“任务认证来得太晚”
    - 而是“任务认证一旦收紧，缺少可恢复的主合同斜率，导致长期近零撤销”
    - 下一刀仍应留在 Phase 4，把 task-cert 从硬撤销器升级为真正的任务认证主合同，而不是回退去扩 bootstrap / actor 改动面
- 当前代码已完成一个更窄的 Phase 4 切口：
  - 显式 `task-cert` 不再只停留在 final certified support / outer revocation 层，而是直接进入 critic bootstrap 的 `takeover floor / internal cap` 合同。
  - 具体实现复用了已有的 `behavior_policy_task_cert_real_gate` 与 `behavior_policy_task_cert_support_mask`，构造 `bootstrap_task_cert_revocation`，并把它写入 bootstrap floor 分配前的 authority 主链。
  - 新增的结构观测为：
    - `critic/bootstrap_task_cert_revocation_mean`
    - `critic/bootstrap_task_cert_scope_mean`
    - `critic/bootstrap_task_cert_takeover_floor_mean`
    - `critic/bootstrap_task_cert_internal_cap_mean`
  - 当前真实存在的回归测试是 `test_low_real_task_cert_raises_bootstrap_external_floor_and_caps_internal_authority`，它显式固定了一个更硬的因果面：
    - `real_eval_gate` 保持健康
    - 即时 `real_batch reward_agreement` 将 real-side task-cert 压低
    - bootstrap external authority 必须随之上升
    - bootstrap internal authority 必须随之下降
  - 这说明当前系统第一次具备了“task-cert 直接约束 critic bootstrap 自举权威”的显式合同，而不是只在下游 support 层做事后折扣
- `v207-v242` 全量实验对账已经完成，并固定成单独矩阵文档：
  - `WM core` 不是这段窗口的主故障层
  - `imagined geometry` 在这段窗口只是背景层，不是当前实验刀口
  - 主故障迁移路径已经固定为 `authority -> bridge -> eval protocol`
  - 对应文档：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/v207-v242全量实验对账表与根因分层矩阵-2026-03-20.md`
- `v240 / v241 / v242` 现在可以被正式视为同一个 `15 ep` 新窗口里的连续证据链：
  - `v240` 证明 availability/open gates 不等于 durability
  - `v241` 证明 floor 接上但仍只是复述当前退化 authority
  - `v242` 完整跑完后，又进一步证明当前 capture-source replacement 仍未形成净结构增益或净行为增益
- `v242` 的硬证据已经足够把问题收缩到更小：
  - eval 轨迹为 `250=46.7, 500=218.1, 750=62.1, 1000=31.1, 1250=203.9, 1500=92.2, 1750=51.9, 2000=264.3, 2250=25.0, 2500=22.5`
  - `1500` 时：
    - `current_authority = 0.1798`
    - `historical_authority = 0.1582`
    - `capture_source = 0.1798`
  - `2000` 时：
    - `current_authority = 0.0762`
    - `historical_authority = 0.0759`
    - `capture_source = 0.0762`
  - `2500` 时：
    - `current_authority = 0.0531`
    - `historical_authority = 0.0504`
    - `capture_source = 0.0531`
  - 这说明历史 authority 虽已进入 helper，但还没有在任何关键点真正高于当前 authority；当前实现仍是 contemporaneous mirror
- 因此，当前唯一允许继续推进的主线已经进一步缩成：
  - 只在 [`_compute_post_transition_certified_retention_floor_contract(...)`](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py) 内继续修 `post_transition_certified_capture_source`
  - 不再重开 `WM core` / imagined geometry / corridor / entry / trigger / actor / controller
  - 后续实验压缩为 2-3 个必须跑的 A/B，首先验证“历史 authority 是否能在有真实支撑时，至少一次把 capture source 顶到高于 current authority”
- `Phase 4.22B` 的 `historical-authority dominance` 代码 cut 已实际落地，并通过了定向 TDD 回归：
  - helper 从“`support` 作为总上限”改为“`current authority` 作为下界，`historical authority` 允许在 `current + support` 的窗口内形成 lift”
  - `historical authority` 现在会对 `prev_floor_state` 做一次 release attenuation debias，而不再直接等于 `0.95 * prev_floor_state`
  - 对应测试已覆盖：
    - supported history 能高于 current authority
    - low support 仍会压住 lift
    - 贴 `v242` 数值时也能打破 `capture_source == current_authority`
- `v243 + v244` 两次同口径 run 已把 `capture source` 层正式打穿：
  - 两次 run 的 eval 轨迹都与 `v242` 逐点同型：
    - `250=46.7, 500=218.1, 750=62.1, 1000=31.1, 1250=203.9, 1500=92.2, 1750=51.9, 2000=264.3, 2250=25.0, 2500=22.5`
  - 但两次 run 的结构量都出现相同 lift：
    - `1500`: `capture_source = 0.2501 > current = 0.1798`
    - `2000`: `capture_source = 0.1524 > current = 0.0762`
    - `2500`: `capture_source = 0.1062 > current = 0.0531`
  - 且 `2000 / 2500` 时 `persistence_gate_floored` 均高于原始 gate
- 因此当前正式结论已经进一步上移：
  - `capture source` 本体问题已不再是当前最窄瓶颈
  - 当前瓶颈已经变成 `floor-to-gate coupling`
  - 后续允许层不再是继续调 `capture source`，而是只改 `hold_state_floored / persistence_gate_floored` 的 downstream coupling

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 使用 `task_plan.md / findings.md / progress.md` 作为持久规划层 | 符合 `pi-planning-with-files` 技能，防止多轮实施过程中的目标漂移 |
| 用 RFC + 控制计划双文档固化主线 | RFC 锁定方向，控制计划锁定执行与审查节奏 |
| 先定义 `SemanticContract`，再实现 `SemanticArbiter` | 先固化接口，再逐层接入，避免一次性重构导致因果混乱 |
| Step 2 只改 critic bootstrap 主合同 | 这是当前证据最强、改动最小且最接近根因的第一落点 |
| controller 仅在后续阶段做降级与拆除，不参与主线修复 | 避免再次让补偿层遮蔽主病灶 |
| 在治理期额外记录最小环境快照 | 仓库无版本控制与 lockfile，需要保留实现时的环境口径 |
| Step 1 先做独立模块 + 独立测试，不做训练接线 | 保证第一次落地只验证接口结构，不污染当前主线因果 |
| 在 `aletheia/__init__.py` 导出 `semantic_contract` | 与包现有 `import *` 风格保持一致，同时保持模块本身无训练耦合 |
| Phase 1B 只接旁路摘要与日志，不接 live contract 对象 | 避免把训练 batch 和现有主逻辑改成 contract-first，先完成观测照亮 |
| Phase 2 之前不改变任何 bootstrap / actor / corridor 权重公式 | 保持 Phase 1 的信息增益纯净，确保下一刀仍能归因到 critic bootstrap 主合同 |
| Phase 2 采用薄 `SemanticArbiter` 落 bootstrap 接管合同 | 目标是把“最低接触面积”和“接触后加码”显式拆层，同时保持指标与行为边界稳定 |
| `v195` 后冻结进入 Phase 3 的计划 | 真实 run 已证明主病灶仍在 critic bootstrap 持续性，不应提前把因果扩散到 actor |
| 下一步定义为 Phase 2.5 持续接触合同重写 | 重点从“接没接进去”转为“能否在掉坑前形成持续主导”，方向仍锁在 critic bootstrap |
| `v207` 必须使用 Phase 2.5 bootstrap-only 基线 | 这轮实验的唯一目的，是验证 trigger surface 是否摆脱旧 eval gate 的单点卡死，不能再让 actor 变量污染因果 |
| `v207` 的通过条件必须拆成“结构通过”和“阶段通过”两层 | trigger surface 已真实打开，但行为仍深跌，因此不能把“门开了”误写成“持续接触已修好” |
| 下一刀收敛为 `bootstrap effective-contact persistence rewrite` | 当 `clean_mix_mean` 已高而 `effective_contact_mean` 仍会随 `anchor_coverage` 一起塌缩时，受控量必须改成 valid/debt-high 区域的持续 authority 接触，而不是继续盯全局均值 |
| `v208` 说明 bootstrap 侧的持续接触合同已接近局部闭环 | 当 `anchor_valid_effective_contact_mean` 和 `focus_effective_contact_mean` 已在关键步点接近 1，而端到端仍不稳时，不应继续把全部责任归给 bootstrap |
| 接下来需要重新打开 Phase 4/Phase 3 的门禁评估 | `v208` 暴露的是 fake corridor 与统一语义消费错位，而不是 bootstrap 局部接触仍然太弱 |
| `v209` 只应被视为 Phase 4 的最小诊断 cut，而不是最终 corridor 方案 | 它证明 reward-semantic 去认证方向正确，但也证明“用 eval-lagged reward health 乘一下 geometry support”不足以形成稳定 task authority |
| 下一步的 corridor 改造必须从“折扣 geometry cert”升级到“显式 task cert” | 否则系统会在单次高分后立即恢复对 geometry registry 的满额信任，并在下一个 eval 前再次掉回假 corridor |
| 显式 task cert 的第一版保持 bootstrap/actor 公式冻结 | 当前目标是先验证上游认证通道重写本身是否产生净信息增益，避免又把因果扩散到下游消费者 |
| `v210` 后继续留在 Phase 4，而不切回 Phase 2.5/3 | 当前证据表明显式 task-cert 已经产生净信息增益，但不足以独立稳定系统；下一步应增强 task-cert 输入质量与时效性，而不是回退到旧 bootstrap 或过早扩大到 actor/controller |
| `v211` 后继续留在 Phase 4，而不切回 bootstrap / actor | 即时 reward-agreement 已证明“任务认证来得更早”不是主要缺口；当前缺口是 task-cert 缺少恢复斜率与再授权机制，仍应在同一层解决 |
| `v212` 后必须把 task-cert 接进 bootstrap authority，而不是继续只调 support/reward-health | 真实 run 已表明 real-side task-cert 会下行，但 bootstrap internal authority 仍能长期主导；缺口已收敛到 bootstrap 自举合同本身 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 初次备份过滤不严，误包含测试文件 | 重新生成严格排除 `tests/` 的正式快照，并保留失败快照用于审计 |
| 需要将现有思考正式化，避免后续实施偏航 | 新增 RFC 与控制计划文档，并建立 planning files 作为过程记忆 |

## Resources
- 核心源码目录：`/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia`
- 正式备份路径：`/Users/zhangsan/Desktop/缸中之脑v5.6/备份/SAI项目源码备份-2026-03-17-143958`
- 项目级 RFC：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/RFC-SAI-001-语义权威接口.md`
- 实施控制计划：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/SAI-实施控制与审查计划.md`
- 环境快照：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/SAI-环境快照-2026-03-17.md`
- Step 1 模块：`/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/semantic_contract.py`
- Step 1 测试：`/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_semantic_contract.py`
- Step 1 导出入口：`/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/__init__.py`
- Phase 1B 主接线：`/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py`
- Phase 2 结构接管实现：`/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/semantic_contract.py`
- Phase 2 bootstrap 主接线：`/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py`
- Phase 2.5 trigger surface 重写主接线：`/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py`
- `v207` 运行档案：`/Users/zhangsan/Desktop/缸中之脑v5.6/tmp/v207_phase25_bootstrap_trigger_surface_rewrite_overrides.json`
- `v207` 官方运行输出：`/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v207_phase25_bootstrap_trigger_surface_rewrite_2500_20260317_02`
- `v209` overrides：`/Users/zhangsan/Desktop/缸中之脑v5.6/tmp/v209_phase4_reward_semantic_registry_recertification_overrides.json`
- `v209` 官方运行输出：`/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v209_phase4_reward_semantic_registry_recertification_2500_20260318_01`
- `v210` overrides：`/Users/zhangsan/Desktop/缸中之脑v5.6/tmp/v210_phase4_explicit_task_cert_channel_overrides.json`
- `v210` 官方运行输出：`/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v210_phase4_explicit_task_cert_channel_2500_20260318_01`
- `v211` overrides：`/Users/zhangsan/Desktop/缸中之脑v5.6/tmp/v211_phase4_immediate_reward_agreement_task_cert_overrides.json`
- `v211` 官方运行输出：`/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v211_phase4_immediate_reward_agreement_task_cert_2500_20260318_01`
- `v207-v242` 全量对账矩阵：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/v207-v242全量实验对账表与根因分层矩阵-2026-03-20.md`
- 最小主线文档：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-最小主线与唯一允许A-B-2026-03-20.md`
- 决策版摘要：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-决策版摘要-2026-03-20.md`
- `v242` 官方运行输出：`/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v242_phase422_post_transition_certified_capture_source_replacement_2500_20260320_103210`
- `Phase 4.22B` 结果文档：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-22B-Historical-Authority-Dominance-A-B-结果-2026-03-20.md`
- `v243` 官方运行输出：`/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v243_phase422b_post_transition_historical_authority_dominance_2500_20260320_112727`
- `v244` 官方 rerun 输出：`/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v244_phase422b_historical_authority_dominance_rerun_2500_20260320_115208`

## Visual/Browser Findings
- 本任务未使用浏览器/图片信息源。

## Contract Convergence Findings (2026-03-21)
- 当前“合同层太多”的本质不是 7-8 个平级 contract，而是：
  - 证据输入
  - 认证状态
  - authority 决策
  - consumer 消费
  - compensation/resume 状态
  被同时抬成 train 顶层 helper 与 metrics 名字。
- 本轮收敛的目标对象已经固定：
  - 非合同：`EvidenceBundle`、`MinimalCompensationState`
  - 一等合同：`CertificationState`、`AuthorityDecision`、`ConsumerDirective`
- 当前工程不是 git repo，因此必须把回滚节点前置成文档纪律，而不能等到失败后再补。
- Batch A 已完成，后续所有 Batch 必须复用同一套验收程序：
  - `compileall`
  - focused unit tests
  - focused training-loop regressions
  - resume/checkpoint regression
  - 批次人工审查

## Contract Convergence Findings (Batch J Closeout)
- `aletheia_train.py` 里的旧 helper 之所以造成“合同层很多”的观感，关键不是调用太多，而是实体实现和兼容入口混在一起。把这些旧实现降级成 `__legacy_*` 后，当前唯一 authoritative 入口已经清晰落在 `contracts/*` 与 `training/compensation.py`。
- `compute_bootstrap_trigger_entry_contract` 属于 authority，而不是 consumer。把这条链从 `consumers` 侧桥接收回后，authority / consumer 的边界比之前更干净，也更符合 Phase 4 的根因分层。
- 这轮收敛没有产生新的证据把当前问题重新归因到 `WM core`。工程侧混乱已经大幅下降，后续如果继续看不过线主因，应优先检查 authority / bridge / eval protocol，而不是再扩 controller 或回退世界模型精度归因。
- 但当前不能把“模块收敛完成”误写成“legacy surface 已全部删除”。`aletheia_train.py` 里仍保留整块 `__legacy_*` 实体实现，`aletheia_api.py` / `aletheia_config.py` / `test_run_train_contracts.py` 仍暴露旧阶段配置面，Batch J 只能算进入最后收尾，而不是彻底封口。

## Contract Convergence Findings (Batch J.1)
- `run_train` 外部入口是 legacy surface 最容易继续扩散的地方。先把 legacy controller-stage overrides 在 API 层隔离掉，比直接硬删 train 内部状态机更安全，也更容易监控。
- 这轮隔离后，legacy stage override 的行为已经从“可注入正式配置”变成“被忽略并告警”。这是一个很重要的边界变化：旧字段还可能存在于对象或代码里，但它们不再是外部允许继续推进的正式合同。
- 当前 warning 日志已经变成有效的过程监控器。如果后续批次还频繁触发这条 warning，就说明调用面或测试面仍在依赖旧阶段合同。

## Engineering Governance Findings (2026-03-21)
- `semantic_contract.py` 已经不再承载 authoritative 定义，只是纯 re-export shim，属于可以立即物理删除的 compat 壳。
- `controller stage` 的 canonicalization 属于边界性定义，不应该继续住在 `aletheia_train.py` 顶部。它适合先迁入 `_scaffold/`，作为后续集中删除的第一块。
- `MinimalCompensationState` 的 legacy 映射目前不是普通兼容层，而是 checkpoint/resume 产物协议的一部分。它不能与纯 shim 同批硬删，必须先完成 canonical payload 迁移与 focused roundtrip 验证。
- 当前最合理的治理排序是：
  - 先删纯 shim
  - 再集中 stage/scaffold 边界
  - 然后切 API/config 外部 legacy 面
  - 最后再切 resume/checkpoint 协议
- `Harvey` 的只读盘点确认：
  - 第一批最适合继续迁入 `_scaffold/` 的是 `_resolve_imag_controller_stage` 周边 helper、`post_entry` 软保护 profile、`post_solved` 相关高水位/anchor 计算器。
  - 不能先删的是真正 live 的三类消费面：
    - `controller_info` 生产链
    - `rl_batch["controller_*"]` 消费链
    - post-solved anchor 在优化器里的执行链
- `Carver` 的只读盘点确认：
  - 生产代码里的 compat 面，绝大多数都不是“立即删除”级别，而是“先改协议/测试再删”。
  - 当前最大的 compat 聚集地不是单个模块，而是 `test_training_loop_integration.py` 的 `config_mode="compat"` 与 `controller_stage` 测试矩阵。
  - `semantic_contract.py` 属于极少数可以立即物理删除的纯 shim，已完成。

## Engineering Governance Findings (Q3-Q5 Closeout 2026-03-21)
- `Q3` 当前最该收死的不是所有 compat 行为，而是“已经不再生效却还允许输入”的 legacy controller-stage overrides。把 `run_train` 从 ignore+warning 改成 fail-fast，是低风险且高收益的边界收口。
- `Q4` 当前最有价值的删除对象不是 live path，而是 `_resolve_imag_controller_stage()` 里首个 `return` 之后整段旧阶段机死代码。它是严格的零语义风险删除，删掉后 runtime 边界清晰了很多。
- `Q5` 的真实瓶颈不是缺少新的 state 设计，而是 canonical payload 仍带着 compat 痕迹。`schema_version`、严格三通道 `bonus_hold_state`、非法 payload fail-fast，这三件事做完后，`MinimalCompensationState` 才算从“兼容载体”进入“正式协议”。
- `_restore_adaptive_controller_state()` 必须拒绝旧的 duplicate minimal keys。否则 export 已经单写 canonical subpayload，但 restore 仍可能从旧 artefact 恢复出半残状态，协议边界会再次变脏。
- `aletheia_train.py` 里的纯薄转发 helper 与恒 false/stub 占位符已经完成物理删除。现在主训练文件里剩下的 controller 相关逻辑，基本都属于 live path，而不是“看起来像逻辑、实际上只是壳”的脚手架。
- 当前没有新增证据表明问题重新回到 `WM core`。工程治理层这轮的主要收益，是把“旧阶段机还在暗中影响当前运行时”这个疑点正式排除掉。

## Engineering Governance Findings (Global Strict Closeout 2026-03-21)
- “剩下三项兼容口子”的最优雅收口，不是继续在入口做假对齐，而是把正式合同直接定成一句话：
  - 正式入口一律 strict，compat 只允许在测试/fixture 中显式声明
- `run_train` 的 strict 化卡点并不在 strict 本身，而在旧合同把 `total_steps` 和 `total_env_steps` 错当成同义字段。真正的规范语义应当是：
  - `total_steps / num_train_steps` = 训练更新预算
  - `total_env_steps` = 环境采样预算
- 因此需要删掉的不只是 `run_train` 的 compat 默认，还包括：
  - `TrainingConfig.from_dict()` 内部对 `total_steps <-> total_env_steps` 的追踪镜像
  - `_normalize_training_config()` 中把 `total_steps` 自动别名成 `total_env_steps` 的旧兼容入口
- `load_agent(strict=True)` 默认恢复与 ConfigPolicy canonical/atomic apply 本身没有暴露新的结构风险；本轮真正暴露出来并已修掉的，是“训练预算”和“环境预算”被 legacy 镜像绑死。
- 本轮之后，这三项兼容口子已经完成治理收口：
  - `run_train`：strict 默认已落地，并保留 distinct update/env budgets
  - `load_agent`：strict restore 默认已落地，非 strict 只能显式 opt-out
  - ConfigPolicy：canonical + atomic apply 已落地，失败不再 partial-apply，也不再伪造 `_policy_*` 默认值

## Step 2.1 v4 Findings (2026-03-22)
- 本轮把 cut 从 `final external value clamp` 前移到了 `source seed` 入口，并新增了 live-path telemetry：
  - `critic/bootstrap_source_seed_quality_gate_mean`
  - `contract/bootstrap_external_bonus_value_source_seed_delta_abs_mean`
- focused tests、`compileall`、`test_contract_convergence`、`test_training_loop_integration`、`test_run_train_contracts` 全部通过，说明这次改动在合同层是干净的。
- smoke run：
  - [exp_seed42_step21g_smoke_gate_250_20260322_0935](/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_step21g_smoke_gate_250_20260322_0935)
  - `eval@250 = 126.07`
- probe run：
  - [exp_seed42_step21g_probe_to1000_gate_2500_20260322_0937](/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_step21g_probe_to1000_gate_2500_20260322_0937)
  - `250 = 10.73`
  - `500 = 153.60`
  - `750 = 72.60`
  - `1000 = 473.47`
- 这轮最关键的新证据不是 reward，而是结构面：
  - `500/600/700/800/900/1000` 上：
    - `critic/bootstrap_source_seed_quality_gate_mean = 1.0`
    - `contract/bootstrap_external_bonus_value_source_seed_delta_abs_mean = 0.0`
    - `contract/bootstrap_external_authority_mean = 0.0`
    - `contract/bootstrap_external_authority_on_valid_mean = 0.0`
    - `critic/bootstrap_effective_contact_mean = 0.0`
- 这说明：
  - 新的 `source seed veto` 并没有像设计那样命中 `step21f` 的坏 regime
  - 它实际把系统推成了更保守的另一种失败形态：`external bootstrap authority/contact` 在 `500-1000` 整段都没有真正接上
  - 所以这轮失败不是“veto 太晚”，而是“veto 与 activation/contact 没有解耦，直接把 Step 2.1 要建立的外生接管面一起掐灭了”
- 与 `step21f` 的正式对比结论：
  - `step21f`：`regime gate` 低，`external authority/contact` 存在，但坏 value 继续注入
  - `step21g`：`regime gate` 恒为 `1.0`，`external authority/contact` 归零，因此不再是同一种坏相位
- 正式裁决：
  - `step21g` 没有通过 `Step 2.1 v3` 的中间门，因为 `eval@750 = 72.60 << 359.8`
  - 虽然 `eval@1000 = 473.47` 很高，但这不能覆盖 `750` 的结构失真
  - 当前不能启动 `Step 2.2`
- 当前最窄根因继续收敛为：
  - 需要把 `external authority/contact activation` 与 `external value quality veto` 明确拆开
  - 允许 `authority/contact` 存在
  - 但不能让坏 `external value seed` 跟着穿透

## Step 2 Refocus Findings (2026-03-22)
- `Batch R3` 已完成并通过固定验收：
  - `compileall`
  - `test_training_entrypoints`
  - `test_training_loop_integration`
  - `test_pure_imagination_path`
  - `test_run_train_contracts`
  - `test_contract_convergence`
  - `_scaffold` 物理目录已删除
- 因此工程兼容壳不再是当前第一问题，主线应明确回到 `Step 2 authority/bootstrap`。
- 对比正式 baseline `v252`、`step21d`、`step21f`、`step21g` 后，当前最关键的新结论是：
  - `step21d` 证明 external bootstrap 接上后，`750` 可以优于 baseline，但 `1000` 会因坏 value regime 穿透而塌掉。
  - `step21f` 证明仅加 `regime veto / final clamp` 仍不够，因为当前 clamp 只约束了 bonus value，没有约束 `floor_authority` 这条底座注入。
  - `step21g` 证明把 external bootstrap 全关掉并不会让 `1000` 必然失败，但会把 `750` 打穿，说明当前需要的不是“完全禁用 external”，而是“保留 contact/authority，切断坏 value 穿透”。
- 已从 live path 确认的关键实现问题：
  - 在 [aletheia_train.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py) 中，`bootstrap_external_value_final` 目前由：
    - `floor_authority * anchor_bootstrap_values_next`
    - `bonus_authority * bootstrap_external_bonus_value_quality_clamped`
    共同组成。
  - 但 [compute_bootstrap_final_external_value_quality_contract](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/contracts/consumers.py) 只裁 `bonus` 路径，不裁 `floor_authority` 路径。
  - 结果是 `step21f @1000` 虽然：
    - `critic/critic_contract_bootstrap_final_value_injection_gate_mean = 0.0083`
    - 但 `contract/bootstrap_external_value_final_mean = 26.33`
    - 同时 `contract/bootstrap_external_authority_floor_mean = 0.117`
  - 这说明当前 “quality clamp 生效” 只停留在 telemetry 语义上，没有真实切断 external floor value 注入。
- 当前正式裁决：
  - `Step 2.1` 的第一主刀口应从“authority optimization”继续收窄为：
    - `external floor injection` 的质量门控
    - 保留 authority/contact
    - 仅切断低质量 external value 对 critic target 的穿透
  - 在这件事修完前，不允许启动 `Step 2.2/2.3`

## Repository Hygiene Findings (2026-03-22)
- 根目录文档卫生的真实问题不是“文件太多”，而是“活跃工作记忆”和“历史阶段报告”混住在同一层级，导致导航和审计都变得噪声很高。
- 当前最干净的工程收口方式，是把根目录只保留活跃工作记忆文件：
  - [task_plan.md](/Users/zhangsan/Desktop/缸中之脑v5.6/task_plan.md)
  - [findings.md](/Users/zhangsan/Desktop/缸中之脑v5.6/findings.md)
  - [progress.md](/Users/zhangsan/Desktop/缸中之脑v5.6/progress.md)
- 已确认历史性根目录报告共 35 份，现已归档到：
  - [README.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/README.md)
  - [2026-03](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03)
  - [undated](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/undated)
- 这轮不能直接删文档的原因已经被证实：
  - `task_plan.md`、`findings.md`、`progress.md` 以及部分阶段文档之间存在大量绝对路径交叉引用
  - 正确动作是“先迁移，再批量修正路径”，而不是“先删再补”
- 本轮没有触碰：
  - `outputs/`
  - `tmp/`
  - `.venv/`
  - 任何训练/实验产物

## Docs Hygiene Findings (2026-03-22)
- `docs/` 根层之前的主要问题，不是“文档太多”，而是“正式入口文档”和“阶段性 workstream 稿件”混住，导致 `docs/` 根层失去了导航意义。
- 当前更优雅的层级是两层制：
  - `docs/` 根层只保留正式入口文档
  - 历史工作流文档进入 [README.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/README.md) 所描述的 workstream archive
- 已完成的 workstream 分层：
  - [cartpole-pure-imag](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag)：48 个文件
  - [actor-contract](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/actor-contract)：8 个文件
  - [control-stack](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/control-stack)：24 个文件
- 本轮扫描没有发现仓库其他 Markdown 仍在显式引用这些被迁移文档的旧 `docs/文件名` 绝对路径，因此本轮重排风险显著低于第三轮根目录归档。

## Content Dedup Findings (2026-03-22)
- `cartpole-pure-imag` 的主要噪声不是“有太多文档”，而是“有一整个高重复 family 被平铺展示”，尤其是 `signal_run_audit` 系列。
- 最优雅的去重方式不是删掉变体，而是把它们收成一个 family：
  - 主入口文档保留在同簇最显眼的位置
  - 变体和 JSON 底稿继续保留
  - `signal_window_findings`、`landing_guard_findings` 这类派生结论也和主文档放在一起
- 经过第五轮治理，`cartpole-pure-imag` 已经形成三分结构：
  - [analysis-design](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/analysis-design)
  - [run-notes](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/run-notes)
  - [signal-audit-family](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/signal-audit-family/README.md)
- 这轮去重没有删除任何历史内容，做的是“目录降噪 + 主入口显式化”。

## Archive Normalization Findings (2026-03-22)
- `workstreams` 目录在第五轮之后仍有一个不一致点：只有 `cartpole-pure-imag` 有完整的 README 和子目录规范，而 `actor-contract`、`control-stack` 仍然只是“平铺归档”。
- 第六轮已经把这个不一致收掉：
  - [actor-contract](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/actor-contract/README.md) 现在明确区分 `design-architecture` 与 `validation-diagnostics`
  - [control-stack](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/control-stack/README.md) 现在明确区分 `css-trust-line`、`corridor-takeover` 与 `runtime-diagnostics`
- 这一轮的价值不在于“减少文件数量”，而在于让三个 workstream 目录终于使用同一种治理语言：
  - 有 README
  - 有主入口建议
  - 有子目录语义
  - 有后续新增文档的落位规则

## Index Compression Findings (2026-03-22)
- `root-reports/2026-03` 的主要问题不是层级不够，而是“目录入口仍然只是一个文件堆”。当单月文档数量达到 30 份时，继续靠文件名浏览会快速失去导航能力。
- 第七轮最合理的收口方式不是再次分目录，而是补一份主题索引页：
  - [2026-03 索引](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/README.md)
- 当前索引已经把高密度历史文档压成几条主线：
  - `Start Here`
  - `Phase 4 Mainline`
  - `Phase 4 Experiments And A/B`
  - `Experiment Accounting And Audits`
  - `Cleanup And Governance`
  - `Project-Wide Program Docs`
- 这轮没有移动任何文件，做的是“入口压缩”和“主题重排”，目的是把阅读路径收短，而不是再改物理结构。

## Archive Index Unification Findings (2026-03-22)
- 在第七轮之后，归档区还差最后两个入口：
  - `docs/archive` 顶层没有总索引
  - `root-reports/undated` 没有自己的 README
- 第八轮已经把这两个缺口补齐：
  - [archive index](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/README.md)
  - [undated 索引](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/undated/README.md)
- 这轮的核心收益是导航链被打通了：
  - 从 [docs/README.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/README.md) 可以进入 archive 总入口
  - 从 archive 总入口可以进入 `root-reports` 和 `workstreams`
  - 从各分区又能继续下钻到月索引、主题索引和子簇 README

## Naming Hygiene Findings (2026-03-22)
- 这轮最值得修的，不是“把所有历史命名风格洗成一种”，而是先清掉真正会影响路径稳定性和可读性的异常名。
- 当前已修掉的两类高风险异常：
  - 前导符号：`# 统一训练入口方案与技术路径.md`
  - 文件名空格：`actor主合同重绑定与certified corridor再认证系统设计图-2026-03-16-1420.md`
- 本轮治理原则已经固定：
  - 先修高风险异常
  - 再通过 README 显式写下命名规则
  - 不把“规范化”误做成“大规模历史文档重命名”

## Naming Convention Findings (2026-03-22)
- 命名治理如果只停留在“这次改了两个文件”，后面很容易再次回退；因此第十轮必须把规则正式写成文档，而不是继续分散在 README 和进度记录里。
- [naming conventions](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/NAMING-CONVENTIONS.md) 现在已经把以下内容落成正式规则：
  - 什么属于高风险异常
  - 哪些历史风格允许并存
  - 什么时候应该重命名历史文件
  - 重命名的固定步骤
  - README 最低要求
- 这一轮的意义，是把前九轮实践提炼成“以后继续治理时可以直接照着执行”的标准。

## Archive Audit Automation Findings (2026-03-22)
- 仅有规则文档还不够，后面继续归档时仍然可能退化；因此第十一轮需要把规则转成“机器能先扫一遍”的半自动巡检草案。
- 当前已经落地：
  - [audit checklist](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/AUDIT-CHECKLIST.md)
  - [archive_audit.py](/Users/zhangsan/Desktop/缸中之脑v5.6/tools/archive_audit.py)
- 这份草案的定位很明确：
  - 自动检查结构、命名、链接
  - 人工判断例外是否合理
  - 不把归档治理误做成“只看脚本 exit code”

## Archive Audit Integration Findings (2026-03-22)
- 仅有 `tools/archive_audit.py` 还不够，因为团队协作时大家不一定会默认直接进 `tools/` 找底层脚本。
- 第十二轮已经把审计入口固定为：
  - [archive_audit.sh](/Users/zhangsan/Desktop/缸中之脑v5.6/scripts/archive_audit.sh)
- 这样当前接入层次就明确了：
  - `scripts/` 负责标准执行入口
  - `tools/` 负责底层检查实现
  - `AUDIT-CHECKLIST` 负责解释何时跑、怎样判定通过
