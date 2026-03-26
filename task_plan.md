# Task Plan: SAI 语义权威接口项目实施

## Goal
在不再偏离主线的前提下，将项目从“散装 gate + controller 补偿”推进到“统一语义权威接口驱动的 critic / actor / corridor 协同控制”，并按固定阶段逐步落地、验证、回滚。

## Current Phase
`2026-03-22` 当前阶段已切回 `Step 2 authority/bootstrap` 主线：
- `Batch R3` 已完成，compat / wrapper / `_scaffold` 收口不再是当前主阻塞。
- 当前第一主阻塞重新确认为 `Step 2.1 bootstrap quality control`。
- 当前固定结论：
  - 不能提前进入 `Step 2.2 historical dominance`
  - 不能提前进入 `Step 2.3 consumer retention`
  - 需要先把 `external authority/contact activation` 与 `external value injection veto` 解耦
  - 目标不是再把 authority 抬高，而是阻止坏 external value 穿透 critic 主链

## PR Execution Track

### PR-1: Authority Core Dominance
- [x] 重写 `post-transition historical/certified authority` 的 dominance 规则
- [x] 让 `capture_source` 在结构上可稳定超过 `current_authority`
- [x] 将 dominance 决策显式化为 canonical contract decision
- [x] 新增 focused tests 覆盖历史 authority 胜出 / 释放 / 支持不足三类情况
- **Verification:** `compileall` + `test_contract_convergence` + focused `test_training_loop_integration`
- **Review checkpoint:** 已通过。结构层已确认存在 `historical` 胜出与 `release_blocked` 回退两种裁决，不再只有 contemporaneous mirror。
- **Status:** complete

### PR-2: Train Glue Canonicalization
- [x] 删除 `aletheia_train.py` 中 duplicated authority 公式
- [x] 训练主文件改为只消费 `contracts/authority.py` 的 canonical 决策
- [x] 收缩重复 telemetry，确保 post-transition 只由单点 authoritative 逻辑产生
- **Verification:** `compileall` + focused `test_training_loop_integration` + `test_run_train_contracts`
- **Review checkpoint:** 已通过。`aletheia_train.py` 中 duplicated authority 公式已删除，post-transition 指标改由 canonical decision 派生。
- **Status:** complete

### PR-3: Unified Consumer Contract
- [x] 统一 actor / corridor / bootstrap consumer 的 winner contract 入口
- [x] 将 geometry support 与 task truth 明确分离
- [x] 建立 consumer-facing canonical contract view
- **Verification:** `compileall` + focused `test_contract_convergence` + focused `test_training_loop_integration`
- **Review checkpoint:** 已通过。已建立最小 consumer-facing contract view，并在 corridor / bootstrap 侧接入统一选择口径，task-certified 合同可优先于 geometry-only 合同。
- **Status:** complete

### PR-4: Plateau Stabilization Without Controller
- [x] 将后段 hold / floor / release 的判据收束到 contract winner
- [x] 去掉对 controller 阶段分支的行为依赖
- [x] 用真实基线或同口径 rerun 验证“结构打穿后不再立即塌回旧型”
- **Verification:** `compileall` + focused tests + 一条真实 baseline 或同口径 gate run
- **Review checkpoint:** 已通过。post-transition historical winner 现在可直接抬高 floor/persistence gate，稳定化判据绑定到 contract outcome，而不是绑定到旧 controller 分支。
- **Status:** complete

### PR-5: Controller Rename-Out and Protocol Cutover
- [x] 将 `controller_stage` 重命名为 runtime-neutral compensation phase
- [x] 将 `MinimalCompensationState` 切换到 runtime-neutral payload
- [x] checkpoint/resume/export/restore 协议切到新命名，旧 payload fail-fast
- **Verification:** `compileall` + focused `test_run_train_contracts` + focused resume roundtrip tests
- **Review checkpoint:** 已通过。payload 已切到 `runtime_phase`，旧 `controller_stage` payload 会显式报错，协议面已不再默默接受旧 controller 键。
- **Status:** complete

### PR-6: Controller Physical Eradication
- [x] 删除 `imag/controller_*`、`actor/controller_stage`、`controller_info` 类 telemetry
- [x] 收缩 config surface，删除仅服务旧 controller 语义的入口
- [x] 清理测试中对 controller 的历史依赖
- [x] 物理删除 `aletheia/controller/`
- **Verification:** `compileall` + focused tests + 一条真实 baseline gate run
- **Review checkpoint:** 已通过。`aletheia/controller/` 已物理删除，正式 telemetry 面已清掉 post-entry / handoff / fallback 遗留 controller 指标，并通过真实 smoke gate run。
- **Status:** complete

## Phases

### Phase 0: 备份与治理固化
- [x] 明确“模型源码”备份边界
- [x] 创建只含模型源码的备份快照
- [x] 建立项目级计划文件
- [x] 起草 RFC 与实施控制计划
- [x] 汇总子代理审计意见并并入治理文档
- [x] 补最小环境快照，记录非 git / 无 lockfile 风险
- **Status:** complete

### Phase 1: 语义合同对象显式化
- [x] 新增 `SemanticContract` 最小字段集
- [x] 在 clean anchor / bootstrap / corridor 链路旁路生成 contract
- [x] 新增 contract 结构单测
- [x] 日志中输出 source / coverage / confidence / authority
- **Status:** complete

### Phase 2: Critic Bootstrap 权威接管合同
- [x] 引入 `SemanticArbiter`
- [x] 将 bootstrap surface 从“可接触”改为“最低接触面积合同”
- [x] 保持 actor 不动
- [x] 用结构指标验证接触面积与语义债压制是否改善
- **Status:** complete

### Phase 2.5: Critic Bootstrap 持续接触合同
- [ ] 将 bootstrap contract 从“后段恢复接管”升级为“中后段持续主导”
- [x] 完成 `v205/v206` 裁决：actor unified contract 主通路本身可行，但 actor-side tail relief 未提供新增实效
- [x] 将下一刀收敛为 `bootstrap trigger surface rewrite`
- [x] 将 late / precontact gate 从“step × eval”升级为“time base × semantic trigger surface”
- [x] 用 `semantic_debt / release_guard / negative_adv_pressure / task_degradation` 驱动中后段连续开门
- [x] 新增 `bootstrap_eval_gate / bootstrap_trigger_gate / bootstrap_trigger_surface / bootstrap_negative_adv_pressure` 结构观测与回归测试
- [ ] 让 `authority / trust / semantic_debt` 决定 pre-contact 阶段的最低外生接触占比
- [ ] 让 `coverage / confidence / support / dense surface gain` 仅作为加码调制，不再影响主导权是否存在
- [x] 让 `surface_memory` 从“只记录状态”升级为“能占用剩余 authority budget 的持续 bonus”
- [x] 完成 `v196` 裁决：确认失败点不是接触预算不足，而是 authority surface 分配错位
- [x] 将第二刀收敛为 `bootstrap surface budget rebalance`
- [x] 启动 `v207` 真实 pure-imag run，并完成正式裁决（误配置 `_01` 归档为旁证，`_02` 为正式结果）
- [x] 验证 trigger surface 已摆脱旧 `eval_gate` 单点卡死：`700/750` 时 `eval_gate=0` 但 `trigger_gate≈0.86`
- [ ] 验证 `1000-1750` 区间不再出现“接触已稳定但 eval 继续深跌”的形态
- [ ] 验证 solved 后段的 value semantics 不再在 `corridor=1.0` 下继续显著膨胀
- [ ] 将受控量从 `clean_mix_mean` 升级为 coverage-valid / debt-high 区域的 `effective_contact` 持续主导，避免“均值看似接入、关键态仍失联”
- [ ] 让 `anchor_coverage` 不再把外生 authority 的主导权压回稀疏偶发接触，而只影响已建立接触后的加码幅度
- [x] 完成 `effective-contact persistence rewrite` 第一版代码落地：base floor 改用 `anchor_valid_mask`，`anchor_coverage` 退到 bonus 调制链，`surface_state` 改读 focus effective contact
- [x] 新增 fractional coverage 回归测试，防止 coverage 权重重新压掉 valid debt-high 接管
- [x] 启动 `v208` 真实 pure-imag run，验证 `focus_effective_contact_mean` 是否开始主导 `surface_state`
- [x] `v208` 证明 valid/debt-high 区域的持续接触合同已经真正生效：`1000/1250/1500/2000` 的 `anchor_valid_effective_contact_mean≈0.91/0.93/1.0/1.0`
- [ ] 判定 Phase 2.5 是否应继续留在 bootstrap，还是主病灶已经迁移到 fake corridor / 下游统一语义消费
- **Status:** in_progress

### Phase 3: Actor 消费统一合同
- [x] 完成 `v205/v206` 窄口径安全性验证：actor 读取 unified contract 主通路不会额外打坏系统
- [x] 识别并修复 `v205` 中的常驻 soft blend 回归
- [ ] 让 actor 读取同源 trust / authority 语义的正式阶段闭环建立在“upstream bootstrap trigger surface 已真实激活”之上
- [ ] 验证 actor distrust 与 critic authority 在真实 run 中不再长期错位
- [ ] 验证 actor-side relief/fallback 有真实增益，而不是持续 dormant
- **Status:** narrow-pass, blocked_by_upstream_bootstrap

### Phase 4: Corridor 几何/任务双认证
- [x] 以 `real_reward_health` 对 raw geometry registry support 做最小 reward-semantic recertification
- [x] 在 imagined batch / actor metrics / eval telemetry 中同时暴露 geometry support 与 recertified support
- [x] 用真实 run `v209` 验证“假 corridor 会被去认证”这一最小链路
- [x] 将 task cert 从“reward-health 乘子”升级为独立认证通道，而不是 geometry support 的事后折扣
- [x] 禁止 geometry cert 单独推出任务真理
- [x] 用真实 run `v210` 验证显式 task-cert 通道已进入主链路，并能比 `v209` 更早收缩最终有效认证
- [x] 将 `reward_health` 的 eval 驱动滞后问题收敛到更即时的 task-agreement / reward-agreement 观测
- [x] 用真实 run `v211` 验证即时 `real_reward_agreement` 已进入 real-side task-cert 主链，并能在同一 eval 区间内压缩最终有效认证
- [x] 将 explicit `task-cert` 直接接入 critic bootstrap 的 `takeover floor / internal cap` 合同，避免 real-side 去认证只停留在 outer support/revocation
- [x] 新增 bootstrap-task-cert 回归测试，确认 real-side task-cert 下行会同步抬升 external floor 并压低 internal authority
- [ ] 将 task-cert 从“近零硬撤销”升级为“可恢复的任务认证主合同”，避免 `cert/geom` 在中后段长期贴近 `0.03~0.05`
- [ ] 用真实 pure-imag run 验证 `task-cert -> bootstrap authority` 新合同能否比 `v212` 更早压低 internal authority，并改善 `2000+` 掉坑形态
- [ ] 验证 fake corridor 能被更早且更强地识别，同时不会把系统长期压进低中收益保守盆地
- [ ] 验证单次高分后不会在同一 eval 区间内重新接近几何满额认证，也不会在下一个区间快速失去任务吸引子
- **Status:** in_progress

### Phase 5: Controller 降级与脚手架收缩
- [x] 将 controller 从主合同降级为补偿层
- [x] 清理冗余 latch / bypass / rescue 逻辑
- [x] 保证主行为与语义指标不退化
- **Status:** complete

### Phase 6: 终局固化与清理
- [x] 形成统一的 contract / arbiter 工程边界
- [x] 收缩遗留脚手架与重复指标
- [x] 完成终局验证并出对账图
- **Status:** complete

## Cleanup Execution Track
- `2026-03-20` 已形成正式 `Phase 5/6` 清理收口执行计划：
  - 文档路径：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase5-6-清理收口执行计划-2026-03-20.md`
  - 清理批次固定为：
    1. `Batch 0` 冻结基线与 manifest
    2. `Batch 1` 删除 audit-only 模块与 API 报表出口
    3. `Batch 2` 删除 signal/schema/invariants 解释层
    4. `Batch 3` 折叠 live controller 为最小补偿层
    5. `Batch 4` 清理 config / test / metrics 面
    6. `Batch 5` 做最终验证与对账
- 清理边界固定为三类：
  - `live training path`：不可直接删，只能折叠
  - `audit-only`：优先删除
  - `config/test surface`：必须跟随清理
- `2026-03-20` 执行结果：
  - `Batch 0` 到 `Batch 5` 已全部完成
  - `controller` live path 已收缩为 `idle / trigger / persistence / post_solved` 最小补偿层
  - `TrainingConfig` 仅补回 live path 真实仍在读取的最小 `persistence / post_solved` 字段
  - `aletheia/tests/test_training_loop_integration.py` 中依赖 `pretrigger / entry_probe / post_entry / late_trigger / fallback / handoff / persistence_release` 的旧阶段机测试已清退
  - 最终核验口径固定为：
    - `./.venv/bin/python -m compileall aletheia`
    - `./.venv/bin/python -m unittest aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_run_train_uses_agent_collector_and_writes_resume_artifacts`
    - `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_post_trigger_confirmation_steps_delay_guard_activation aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_persistence_min_step_blocks_early_drawdown_arm aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_persistence_tail_mismatch_repairs_actor_targets_weights_and_critic aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_persistence_tail_mismatch_stays_off_outside_persistence aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_persistence_actor_base_return_cap_applies_when_configured`

## Contract Convergence Refactor Track
- `2026-03-21` 已完成 `Batch A -> Batch I`，并完成 `Batch J` 的兼容层收口：
  - canonical contracts:
    - `EvidenceBundle`
    - `CertificationState`
    - `AuthorityDecision`
    - `ConsumerDirective`
  - non-contract carriers:
    - `MinimalCompensationState`
  - 工程落点：
    - `aletheia/contracts/*.py`
    - `aletheia/training/compensation.py`
  - `aletheia_train.py` 中旧 helper 已降级为 `__legacy_*`，运行入口改由 canonical alias 托管
  - `compute_bootstrap_trigger_entry_contract` 已归回 authority 层
  - config / telemetry / resume 接口已同步收敛
- 本轮收敛的固定结论：
  - 后续 Phase 4 行为问题不再需要回到散装 helper / controller 脚手架里找因果
  - 当前不过线主因仍不回打到 `WM core`
  - 后续根因分析和实验裁决应继续锁定 authority / bridge / eval protocol
- 当前剩余尾项：
  - 物理删除 `aletheia_train.py` 中 `__legacy_*` 实体实现
  - 收掉 `aletheia_api.py` / `aletheia_config.py` / `test_run_train_contracts.py` 中的旧阶段配置合同
- `2026-03-21` 当前轮次已完成 `Batch J.1 legacy override quarantine`：
  - 在 [aletheia_config.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_config.py) 显式定义 legacy controller-stage 字段识别规则
  - 在 [aletheia_api.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_api.py) 拦截 legacy stage overrides，不再把它们当正式 `run_train` 合同入口
  - 在 [test_run_train_contracts.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_run_train_contracts.py) 将旧阶段 override 测试改为“忽略并保持 canonical config 不被污染”
  - 过程监控信号固定为：
    - API warning: `Ignoring legacy controller-stage overrides during contract convergence`
    - focused run-train override regression
  - 下一刀仍是：
    - 压缩 `aletheia_train.py` 中 live stage 分支
    - 再之后才是物理删除 `__legacy_*`

## Contract Convergence Track
- `2026-03-21` 已完成 `Batch A -> Batch J` 的合同收敛重构执行：
  - `Batch A`：建立正式实施/验收/回滚文档与基线快照
  - `Batch B`：`EvidenceBundle` 正式收敛到 `aletheia/contracts/evidence.py`
  - `Batch C`：`CertificationState` 与 task-cert 计算正式收敛到 `aletheia/contracts/certification.py`
  - `Batch D`：authority helper 正式收敛到 `aletheia/contracts/authority.py`
  - `Batch E`：consumer / hold / trigger helper 正式收敛到 `aletheia/contracts/consumers.py`
  - `Batch F`：`MinimalCompensationState` 与 hold helpers 正式收敛到 `aletheia/training/compensation.py`
  - `Batch G`：`TrainingConfig` 新增 `certification_contract_config / authority_contract_config / consumer_contract_config / compensation_config`
  - `Batch H`：bootstrap contract summary key 列表收敛到 `aletheia/contracts/telemetry.py`
  - `Batch I`：resume state 新增 `minimal_compensation_state` 显式导出/恢复口径
  - `Batch J`：完成 compile + focused tests + resume tests 验收；保留 `aletheia_train.py` 旧 helper 名作为兼容入口，但运行时 authoritative 实现已切到新模块

## Key Questions
1. `SemanticContract` 的第一版最小字段集应包含哪些字段，既足够表达权威语义，又不引入过早抽象？
2. critic bootstrap 的“最低接触面积合同”应如何与 coverage / confidence 解耦，避免再次退回多层乘法稀释？
3. actor 与 critic 如何共享同一套 trust 语义而不引入不必要的行为扰动？
4. corridor 的 task cert 该在什么时点接入，才能既压住 fake corridor，又不污染当前主线因果？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 主线锁定为 SAI Program，不再回到 actor-first 或 controller-first | 跨版本证据已将主病灶收敛到语义权威接口与 critic bootstrap 主合同 |
| actor 在 Step 2 前冻结 | trust-only actor 已证明有效，当前继续改 actor 会打脏因果 |
| 第一个工程落点是 critic bootstrap，而不是 corridor | critic 仍会默认信自己，是当前最窄病灶 |
| corridor recertification 放在 critic bootstrap 稳住之后 | 避免同时改两层导致难以归因 |
| 备份范围限定为 `aletheia/` 下非测试 Python 源码 | 满足“只备份模型源码，不备份数据/测试/脚本”的用户要求 |
| 额外补一份最小环境快照，但不混入源码备份包 | 仓库不是 git repo，且未发现 lockfile，仅有源码快照不足以支撑后续审计与复现 |
| Step 1 先只落结构对象与测试，不接训练主行为 | 先把 contract 变成显式对象，再进入 critic bootstrap 主合同改写 |
| Step 2 采用“takeover floor + modulation bonus + arbiter decision”薄实现 | 保持 critic bootstrap-only 改动边界，同时把“最低接管权”和“调制加成”从同一乘法链中拆开 |
| v195 后不进入 Phase 3 | 实测证明 Phase 2 已修通“接触建立”，但还未修通“持续主导”；当前主病灶仍在 critic bootstrap |
| 下一刀定义为“持续接触合同”而非 actor/corridor 改造 | `clean_mix/floor` 已稳定，剩余问题是中后段持续性不足，不是 actor 未共享合同 |
| Phase 2.5 第一刀采用 `surface_memory -> persistent_bonus` 而不采用 `support` 截断 authority | 前者保持“support 只做调制、不是 veto”的主线原则；后者会把 coverage/support 又抬回主接触门控 |
| `v196` 后不继续加 gate/floor，而是改 `mix surface` 分配律 | 证据显示接触预算已足，失败点在 coverage-backed anchor 与 dense fallback 之间的 authority 分配错位 |
| `v197` 后不再把 `base_mix_surface` 当平均预算重分配 | 真实 run 证明方向修对了，但 surface 层仍会把 contract base 重新预算化，导致首跌后段仍可再次塌陷 |
| Phase 2.5 下一刀定义为 `monotonic base-preserving surface lift` | `authority / trust / semantic_debt` 必须先决定 local base authority；coverage/support 只能在 headroom 上做加码，不能再向下稀释 |
| `v205/v206` 只作为 Phase 3 窄口径安全性验证，不改变主阶段门禁 | `v206` 证明 actor unified contract 主通路可行，但上游 bootstrap trigger surface 仍为 0，不能据此宣布 Phase 2.5 结束 |
| 当前下一刀定义为 `bootstrap trigger surface rewrite` | `v206` 的关键证据是 `bootstrap_late_gate=0 / mix_surface=0 / tail_relief=0`，说明上游 trigger 没开，actor 无法拿到真实 bootstrap authority |
| `v207` 使用 Phase 2.5 bootstrap-only 基线 overrides，而不继承 Phase 3 actor 实验名义 | 本轮只验证 upstream trigger surface 是否被真正打开，避免 actor 变量再次污染归因 |
| `v207` 不能判定为 Phase 2.5 通过 | 官方结果 `250=15.4, 500=57.0, 750=402.4, 1000=138.4, 1250=50.0, 1500=35.4, 1750=92.6, 2000=297.2, 2250=25.8, 2500=65.2` 证明结构开门已成，但中后段持续接触仍未稳住 |
| 下一刀收敛为 `bootstrap effective-contact persistence rewrite` | `clean_mix≈0.48~0.53` 已不再解释失败；真正塌缩的是 `anchor_coverage` 下行时的 `effective_contact_mean` 与关键态持续 authority |
| `v208` 不能简单归类为“bootstrap 继续失败” | 这轮已经把 valid/debt-high 区域的持续接触打到接近饱和，但行为仍会在 `2000` 深跌后于 `2250` 再回到 `500`，说明主病灶可能已迁移到 fake corridor / 下游消费层 |
| `v209` 不能判定为 Phase 4 通过 | 最小 reward-semantic recertification 已成功压低假 corridor 的有效 registry support，但端到端曲线仍未稳住，且 `2250 -> 2500` 暴露出 eval 驱动 `reward_health` 的一整段滞后窗口 |
| `v210` 后仍不进入 Phase 3/5 | 显式 task-cert 已证明方向正确，但当前收益主要是“更早收缩认证”而不是“彻底压住 fake corridor”；下一步仍应留在 Phase 4，补更即时、更强的 task cert 输入 |
| `v211` 后仍不切回 bootstrap / actor | 即时 `real_reward_agreement` 已证明能在同一 eval 区间内强力撤销 fake corridor 的任务认证，但行为结果表明现在缺的是“认证恢复律”而不是再次扩大 bootstrap 或 actor 改动面 |
| `v212` 后下一刀必须把 task-cert 写进 bootstrap authority 合同 | 真实 run 已证明 real-side task-cert 会下行，但 bootstrap internal authority 仍能长期占优；继续只做 outer revocation 已无信息增益 |
| `v207-v242` 当前不过线主因不回打到 `WM core` | 全量对账矩阵显示主故障迁移路径是 `authority -> bridge -> eval protocol`，没有足够证据把最近窗口重新归因为世界模型核心精度 |
| `v240 / v241 / v242` 必须视为同一 `15 ep` 窗口内的连续负样本 | 三条 run 的行为轨迹几乎同型，说明当前真正未打穿的仍是 post-transition durable capture，而不是 entry / trigger / floor 接线缺失 |
| 后续唯一允许推进的主线收缩为 `post_transition_certified_capture_source` | `v242` 完整跑完后，历史 authority 在真实关键点上仍未高于 current authority；下一步只能继续缩在同一 helper 内做 capture source dominance A/B |
| `v243 / v244` 已完成 `capture source` 同口径 A/B 与 rerun | 两次 run 都复现了 `capture_source > current_authority` 与 `floored gate > raw gate`，但行为仍逐点贴着 `v242`；因此下一步允许层已上移到 `floor-to-gate coupling` |
| `Phase 5/6` 不再作为“模型修好后的庆功清理”处理 | 由于 Phase 4 未稳定收敛，Stage 5/6 现改为 stop-loss cleanup track：先删 audit-only，再删 signal-only，最后折叠 live controller |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 首次源码快照误包含 `aletheia/tests` | 1 | 保留失败快照用于审计，重新生成严格排除 `tests/` 的正式快照 |
| `pi-planning-with-files` 的 catchup 脚本使用 `python` 不存在 | 1 | 改用 `python3` 运行，成功完成 catchup |

## Notes
- 正式源码备份路径：`/Users/zhangsan/Desktop/缸中之脑v5.6/备份/SAI项目源码备份-2026-03-17-143958`
- 失败但保留的首次快照路径：`/Users/zhangsan/Desktop/缸中之脑v5.6/备份/SAI项目源码备份-2026-03-17-143907`
- 最小环境快照路径：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/SAI-环境快照-2026-03-17.md`
- Step 1 结构模块：`/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/semantic_contract.py`
- Step 1 结构测试：`/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_semantic_contract.py`
- Phase 1B 已将 contract 旁路接入 `clean anchor / corridor / bootstrap` 的日志层与 imagined batch 摘要
- Phase 2 已将 critic bootstrap 改写为 `takeover floor -> modulation bonus -> final external authority` 的显式仲裁合同
- `v195` 真实 run 已完成验证，结论是：Phase 2 合同确实生效，但目前更像恢复合同，还不是持续接触合同
- `v205/v206` 已完成 Phase 3 窄口径安全性验证：actor unified contract 主通路本身成立，但新增 actor-side tail relief 全程 dormant
- 当前主阶段仍是 Phase 2.5；`v207` 已完成 trigger surface 结构验证，但未通过持续接触验收
- `v208` 已证明 bootstrap 的 valid/debt-high 持续接触合同打中了目标，但端到端 solved hold 仍未建立
- `v209` 已证明 fake corridor 的 reward-semantic 去认证链路接通：当 `1000/1250/2000` 回撤出现时，`real_reward_health` 会显著下行并同步压低 `behavior_policy_certified_registry_support_fraction`
- `v209` 也同时暴露了新的主限制：`reward_health` 只在 eval 后更新，因此 `2250` 单次高分会让 `2300-2500` 整段训练重新满额信任 registry，直到 `2500 eval` 才看见再次崩落
- Phase 4 最小 `explicit task cert` 通道已落地：最终 `behavior_policy_certified_registry_support_mask` 现在由 `min(geometry_registry_support_mask, task_cert_support_mask)` 生成，旧 `reward_semantic_registry_*` 退回观测层
- `v210` 已确认：显式 task cert 确实比 `v209` 更早介入 fake corridor 去认证，但前中段压制幅度仍偏弱，后段仍会被 critic 语义膨胀反噬
- `v211` 已确认：新增 `real_reward_agreement` 分支确实把 real-side task-cert 提前到了同一 eval 区间内，`1000-2500` 的 `real_task_cert_gate` 基本稳定在 `0.035~0.048`，且 `critic/real_mc_value_gap_abs_mean` 被压在 `4~10`
- `v211` 也同时确认：当前 task-cert 主合同仍然过于保守，虽然一度在 `2000` 冲到 `424.0`，但最终 `2250 -> 2500` 仍会从 `16.8` 回到 `95.8`，表明系统缺少“去认证后恢复高任务质量”的主路径
- 当前代码已将 `behavior_policy_task_cert_real_gate / behavior_policy_task_cert_support_mask` 接入 bootstrap `takeover floor / internal cap`，新增观测：
  - `critic/bootstrap_task_cert_revocation_mean`
  - `critic/bootstrap_task_cert_scope_mean`
  - `critic/bootstrap_task_cert_takeover_floor_mean`
  - `critic/bootstrap_task_cert_internal_cap_mean`
- 当前待验证点进一步收敛为：这条新 authority 合同能否在真实 run 中把 `real_task_cert_gate` 的撤销及时传到 `bootstrap_internal_authority_mean`，从而减少 `2000+` 的先掉坑
- 当前待验证点已经收敛为：如何把显式 task cert 从“即时硬收缩”升级为“更强且可恢复的任务认证主合同”
- `v207-v242` 全量实验对账表已经固定：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/v207-v242全量实验对账表与根因分层矩阵-2026-03-20.md`
- 当前最小主线文档已经固定：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-最小主线与唯一允许A-B-2026-03-20.md`
- 当前一页式决策摘要已经固定：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/Phase4-决策版摘要-2026-03-20.md`
- `v242` 已完整跑完，官方 eval 轨迹为：`250=46.7, 500=218.1, 750=62.1, 1000=31.1, 1250=203.9, 1500=92.2, 1750=51.9, 2000=264.3, 2250=25.0, 2500=22.5`
- `v242` 的关键结构事实是：`1500/2000/2500` 时 `post_transition_certified_capture_source` 仍分别等于 `0.1798 / 0.0762 / 0.0531`，与 `current_authority` 完全同值，说明当前 capture-source replacement 仍未打破 contemporaneous mirror
- `v243` 官方运行输出：`/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v243_phase422b_post_transition_historical_authority_dominance_2500_20260320_112727`
- `v244` 官方 rerun 输出：`/Users/zhangsan/Desktop/缸中之脑v5.6/outputs/exp_seed42_v244_phase422b_historical_authority_dominance_rerun_2500_20260320_115208`
- `v243 / v244` 都复现了相同结构结果：
  - `1500`: `capture_source = 0.2501 > current_authority = 0.1798`
  - `2000`: `capture_source = 0.1524 > current_authority = 0.0762`
  - `2500`: `capture_source = 0.1062 > current_authority = 0.0531`
- `v243 / v244` 的 eval 轨迹与 `v242` 完全同型：`46.7 / 218.1 / 62.1 / 31.1 / 203.9 / 92.2 / 51.9 / 264.3 / 25.0 / 22.5`
- 当前允许的下一步已经从 `capture source` 上移到 `floor-to-gate coupling`
- `2026-03-20` 已完成 `Phase 5/6` 清理盘点：
  - `aletheia_train.py` 中的 `controller_stage -> continue cap / base_return_cap / actor scale / hysteresis` 仍是 live path，不可整包直删
  - `post_entry_audit.py / post_entry_template_compare.py / post_entry_precursor_audit.py / controller/relation_audit.py / controller/signal_audit.py` 属于 audit-only，可优先删除
  - `controller/signals.py / schema.py / invariants.py` 属于 signal/schema 解释层，可作为第二批删除
  - `aletheia_config.py` 与 `aletheia/tests/test_training_loop_integration.py` 是主要 config/test 跟随面
- `pi-planning-with-files` 已用于把 cleanup 计划固化到项目文档，子代理盘点结论已并入主计划
- 每一阶段都必须先看结构指标，再看 reward 曲线
- 允许调参数，不允许改变结构方向
- `2026-03-21` 已正式启动“合同收敛重构”执行轨：
  - 实施文档：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/合同收敛重构实施计划-2026-03-21.md`
  - 回滚节点：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/合同收敛重构回滚节点-2026-03-21.md`
  - 批次固定为 `Batch A -> Batch J`
  - 一等对象目标固定为：
    - `CertificationState`
    - `AuthorityDecision`
    - `ConsumerDirective`
  - 非合同载体固定为：
    - `EvidenceBundle`
    - `MinimalCompensationState`
- `2026-03-21` 已新增工程治理主计划：`/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/工程治理执行计划-Q1-Q5-2026-03-21.md`
- 当前执行阶段切换为：
  - `Q1`：抽离 stage canonicalization 与纯 shim 删除
  - `Q2`：盘点 train live runtime 与 `_scaffold/` 的进一步迁移边界
  - `Q5`：仅保留协议迁移设计，不提前删除 `MinimalCompensationState` 的 legacy payload
- `2026-03-21` 阶段审查结论更新：
  - `Q1`：`pass`
  - 证据：
    - `semantic_contract.py` 已物理删除
    - `controller stage canonicalization` 已迁入 `aletheia/_scaffold/stages.py`
    - `compileall + test_semantic_contract + test_contract_convergence` 通过
  - `Q2`：`pass_with_adjustment`
  - 证据：
    - `post-trigger quality` 与 `handoff landing guard` 已迁入 `aletheia/_scaffold/post_trigger.py`
    - `test_trigger_persistence_landing_guard_is_disabled_by_default`
    - `test_adaptive_controller_state_roundtrip_restores_post_transition_certified_floor_state`
    - 均已通过
  - 调整：
    - 下一阶段先做 API/config 外部 compat 入口盘点与收口设计
    - `MinimalCompensationState` 暂不物理删 legacy payload，因为它仍直接连着 checkpoint/resume 协议
- `2026-03-21` 工程治理第二轮执行已完成 `Q3/Q4/Q5` 的当前最小收口：
  - `Q3`：
    - `run_train` 不再“忽略并告警” legacy controller-stage overrides，改为入口 `ValueError`
    - `test_run_train_contracts.py` 同步改为显式拒绝旧键，并补一条 canonical-only run_train override 验证
  - `Q4`：
    - `aletheia_train.py::_resolve_imag_controller_stage()` 中首个 `return` 之后整段不可达旧阶段机已物理删除
    - 当前 live stage 解析正式收敛到 `idle / trigger / persistence / post_solved`
    - `aletheia_train.py` 中纯薄转发 / scaffold 壳已继续清理：
      - `_resolve_persistence_escape_soft_floors`
      - `_resolve_standard_soft_fallback_trigger_release_progress`
      - `_is_trigger_persistence_handoff_landing_guard_active`
      - `_is_standard_soft_fallback_trigger_release_active`
      - `_maybe_apply_effective_post_entry_commit_override`
  - `Q5`：
    - `MinimalCompensationState` payload 字段从 `version` 收口为 `schema_version`
    - `bonus_hold_state` 不再接受 scalar broadcast，必须是三通道 canonical mapping
    - `restore_minimal_compensation_state()` 对非法 payload 改为 fail-fast，不再 silent no-op
    - `_restore_adaptive_controller_state()` 新增 removed duplicate minimal keys 防线，拒绝旧双写协议
  - focused 验收结果：
    - `./.venv/bin/python -m compileall aletheia`
    - `./.venv/bin/python -m unittest aletheia.tests.test_contract_convergence aletheia.tests.test_semantic_contract`
    - `./.venv/bin/python -m unittest aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_run_train_rejects_legacy_post_entry_highwater_commit_overrides aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_explicit_eval_confirmation_overrides_reach_training_config aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_run_train_rejects_mixed_legacy_controller_stage_overrides aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_run_train_rejects_legacy_post_entry_highwater_eval_threshold_override aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_agent_save_persists_config_bundle_metadata aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_load_agent_uses_checkpoint_config_bundle_when_present aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_load_agent_without_config_bundle_uses_default_creation_overrides aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_create_agent_rejects_noncanonical_reserved_override_aliases`
    - `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_adaptive_controller_state_roundtrip_restores_post_transition_certified_floor_state aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_imagination_only_resume_restores_buffer_and_loop_counters`
    - `./.venv/bin/python -m unittest aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_run_train_uses_agent_collector_and_writes_resume_artifacts`
  - 当前裁决：
    - `Q3`：`pass`
    - `Q4`：`pass_with_adjustment`
    - `Q5`：`pass_with_adjustment`
  - 后续治理口径已更新为“正式入口全 strict”：
    - `run_train` 默认生成 `config_mode="strict"` / `validation_mode="strict"` 的训练合同
    - `load_agent()` 默认 `strict=True`，仅保留显式 `strict=False` 逃生口
    - ConfigPolicy 已切到 canonical + atomic apply，不再 partial-apply，也不再注入伪 `_policy_*` 默认值
    - `total_steps / num_train_steps` 统一表示训练更新预算，`total_env_steps` 独立表示环境采样预算；compat 只允许显式存在于测试/fixture

## Repository Hygiene
- `2026-03-22` 已完成第三轮“文档归档重排”：
  - 根目录历史性报告已整体迁入 [README.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/README.md) 所定义的归档区
  - [2026-03](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03) 收纳 30 份带日期文档
  - [undated](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/undated) 收纳 5 份无显式日期的历史文档
  - 仓库根目录当前仅保留活跃工作记忆文件：
    - `task_plan.md`
    - `findings.md`
    - `progress.md`
  - 仓库内对这些历史文档的绝对路径引用已同步修正到新归档路径
- `2026-03-22` 已完成第四轮“docs/ 层级卫生治理”：
  - `docs/` 根层只保留正式入口文档：
    - [RFC-SAI-001-语义权威接口.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/RFC-SAI-001-语义权威接口.md)
    - [SAI-实施控制与审查计划.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/SAI-实施控制与审查计划.md)
    - [SAI-环境快照-2026-03-17.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/SAI-环境快照-2026-03-17.md)
  - 历史 workstream 文档已迁入 [README.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/README.md) 定义的归档区
  - 当前 workstream 归档分层为：
    - [cartpole-pure-imag](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag)
    - [actor-contract](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/actor-contract)
    - [control-stack](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/control-stack)
- `2026-03-22` 已完成第五轮“cartpole-pure-imag 内容去重治理”：
  - [cartpole-pure-imag](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/README.md) 已从“48 个文件平铺”收口为：
    - [analysis-design](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/analysis-design)
    - [run-notes](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/run-notes)
    - [signal-audit-family](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/signal-audit-family/README.md)
  - `signal_run_audit` 高重复 family 已集中归档，并保留主入口 + 变体 + JSON 底稿
- `2026-03-22` 已完成第六轮“workstreams 归档规范化”：
  - [actor-contract](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/actor-contract/README.md) 已规范化为：
    - [design-architecture](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/actor-contract/design-architecture)
    - [validation-diagnostics](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/actor-contract/validation-diagnostics)
  - [control-stack](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/control-stack/README.md) 已规范化为：
    - [css-trust-line](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/control-stack/css-trust-line)
    - [corridor-takeover](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/control-stack/corridor-takeover)
    - [runtime-diagnostics](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/control-stack/runtime-diagnostics)
- `2026-03-22` 已完成第七轮“root-reports 索引压缩”：
  - [2026-03 索引](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/README.md) 已建立
  - `2026-03` 目录入口已从“30 份文件列表”压缩成主题导航：
    - `Start Here`
    - `Phase 4 Mainline`
    - `Phase 4 Experiments And A/B`
    - `Experiment Accounting And Audits`
    - `Cleanup And Governance`
- `2026-03-22` 已完成第八轮“archive 索引统一化”：
  - [archive index](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/README.md) 已建立为归档区总入口
  - [undated 索引](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/undated/README.md) 已建立
  - 当前导航链已经统一为：
    - `docs/README`
    - `docs/archive/README`
    - `root-reports/README` 或 `workstreams/README`
    - 各子簇 README
- `2026-03-22` 已完成第九轮“命名与文件名规范治理”：
  - 已修复两个高风险命名异常：
    - `# 统一训练入口方案与技术路径.md` -> `统一训练入口方案与技术路径.md`
    - `actor主合同重绑定与certified corridor再认证系统设计图-2026-03-16-1420.md` -> `actor主合同重绑定与certified-corridor再认证系统设计图-2026-03-16-1420.md`
  - 相关 README、索引页和入口链接已同步修正
  - 本轮只收口“前导符号”和“空格”这两类高风险异常，不强行改写全部历史命名风格
- `2026-03-22` 已完成第十轮“规则落盘”：
  - [naming conventions](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/NAMING-CONVENTIONS.md) 已建立为 `docs/archive` 级别的正式命名规范
  - 规则已经挂到：
    - [archive index](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/README.md)
    - [root-reports/README.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/README.md)
    - [workstreams/README.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/README.md)
  - 当前命名治理口径正式固定为：
    - 先修高风险异常
    - 再补 README 与索引规则
    - 最后才考虑风格统一
- `2026-03-22` 已完成第十一轮“归档审计自动化草案”：
  - [audit checklist](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/AUDIT-CHECKLIST.md) 已建立
  - 半自动巡检脚本已新增：
    - [archive_audit.py](/Users/zhangsan/Desktop/缸中之脑v5.6/tools/archive_audit.py)
  - 当前自动检查覆盖：
    - README 覆盖
    - 高风险文件名
    - 失效 archive 本地链接
    - 已知 legacy 名称热点
- `2026-03-22` 已完成第十二轮“归档审计接入方案”：
  - 标准命令入口已建立：
    - [archive_audit.sh](/Users/zhangsan/Desktop/缸中之脑v5.6/scripts/archive_audit.sh)
  - [AUDIT-CHECKLIST.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/AUDIT-CHECKLIST.md) 已补齐“标准接入方式 / 最小执行流程 / gate 节点”
  - 当前标准执行口径固定为：
    - 首选 `bash scripts/archive_audit.sh`
    - 需要紧凑输出时用 `bash scripts/archive_audit.sh --summary-only`
- `2026-03-22` 深水区盘点复核后，`Batch R3` 的执行顺序收紧为：
  1. 先删除纯测试 compat 别名：
     - `_compute_policy_outputs()`
  2. 再盘并收口 `TrainingLoop.__init__` 的 legacy aliases：
     - `optimizer`
     - `log_fn`
     - `checkpoint_dir`
  3. 再压缩 `config_mode="compat"` 测试面：
     - [test_training_loop_integration.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_training_loop_integration.py)
     - [test_pure_imagination_path.py](/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/tests/test_pure_imagination_path.py)
  4. 将 `wrap_training_components()` 单列为去 adapter 化迁移任务，不作为本批直接删除对象
  5. `_scaffold` 继续冻结，待独立语义回迁方案完成后再处理
