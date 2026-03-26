# SAI 实施控制与审查计划

## 1. 目的

本文件是 `RFC-SAI-001` 的执行版。  
它不重复讲抽象方向，而是规定：

- 每一阶段如何切分
- 谁负责什么
- 什么时候审查
- 用什么指标判定“通过 / 不通过”
- 遇到偏差如何回退

## 2. 执行规则

### 2.1 单位任务规则

每个实现单元必须满足：

- 预计 15 分钟内可验证
- 只有一个主风险
- 有单一 done condition

如果一个改动同时影响：

- contract 结构
- bootstrap 主逻辑
- actor trust 来源

则必须拆开，不允许在一个提交单元里混改。

### 2.2 Eval-First 规则

每阶段都必须先定义：

- 能力目标
- 回归风险
- 结构指标
- 行为指标

先看结构量，再看 reward。

### 2.3 变更纪律

允许：

- 调整 floor / cap / ramp / threshold
- 调整日志、测试、字段组织

不允许：

- 新增散装 gate 作为主修复手段
- 重新把 actor 当第一病灶
- 在 bootstrap 主线未稳定前先大改 corridor
- 扩张 controller stage 体系

### 2.4 当前阶段锁定

- 主阶段固定为 `Phase 4`
- `Phase 2.5` 已完成局部裁决：bootstrap 持续接触合同不再是第一病灶
- `Phase 3` 当前只完成了 `v205/v206` 的窄口径安全性验证，尚未进入正式闭环实施
- 当前下一刀固定为：`task-cert -> bootstrap authority` 的真实 run 验证，以及随后把 task-cert 从“强撤销”升级为“可恢复任务认证主合同`
- 在 Phase 4 未通过前，不回退到 actor / controller / 几何 corridor 的主逻辑扩刀

## 3. 阶段编排

### Stage 0：治理与防漂移

**产物**

- `task_plan.md`
- `findings.md`
- `progress.md`
- `RFC-SAI-001`
- 正式源码备份
- 最小环境快照

**审查点**

- 备份范围是否只含模型源码
- 是否补了环境口径，避免“只有源码、没有运行上下文”
- 方向、禁区、回滚纪律是否已写明

### Stage 1：`SemanticContract` 结构化

**实现范围**

- 新文件与最小字段集
- 旁路构造 contract
- 单元测试
- 日志输出

**主风险**

- 抽象过早过胖，造成实现噪音

**审查点**

- 字段是否最小且可计算
- 是否未引入训练行为变化
- 测试是否覆盖 source / coverage / authority 约束

### Stage 2：critic bootstrap 接管权重写

**实现范围**

- 用 authority / semantic debt 决定最低接触面积
- 让 coverage / confidence 只做 modulation

**主风险**

- 提前介入但伤害中段 climb

**审查点**

- `bootstrap_contact_floor_mean`
- `bootstrap_clean_mix_mean`
- `bootstrap_effective_contact_mean`
- `real_mc_value_gap_abs_mean`
- `eval@1000`

### Stage 2.5：critic bootstrap 持续接触 / trigger surface 重写

**实现范围**

- 保持 actor 不动
- 将 `late_gate / precontact_gate` 从“`step × eval`”改为“`time base × semantic trigger surface`”
- 用 `semantic_debt / release_guard / negative_adv_pressure / task_degradation` 驱动中后段连续开门
- 保留 `eval_gate` 作为观测量和触发面的一部分，但不再允许其单独决定 bootstrap 是否永远不开
- 只在 bootstrap 内重写 trigger surface 与持续接触合同

**主风险**

- 结构上门已经打开，但 coverage 下行时有效接触仍会塌缩，导致 `clean_mix_mean` 看似健康而关键态外生 authority 仍无法持续主导

**审查点**

- `bootstrap_eval_gate`
- `bootstrap_trigger_gate`
- `bootstrap_negative_adv_pressure_mean`
- `bootstrap_trigger_surface_mean`
- `bootstrap_late_gate`
- `bootstrap_effective_contact_mean`
- `bootstrap_mix_surface_mean`
- `bootstrap_clean_mix_mean`
- `bootstrap_anchor_coverage_mean`
- `bootstrap_contact_floor_mean`
- `eval@1000`
- `eval@1250`
- `eval@1500`
- `eval@1750`
- `eval@2250`
- `eval@2500`

### Stage 3：actor 接入统一语义

**实现范围**

- 统一 trust 来源
- 不重写 actor 主合同

**主风险**

- 因 trust 来源替换而引入额外行为扰动

**审查点**

- actor distrust 与 critic authority 的一致性
- reward 曲线是否无故变差

### Stage 4：corridor 双认证

**实现范围**

- geometry cert / task cert 分离
- combined cert 不再允许几何单独放行

**主风险**

- 过早去认证导致系统普遍保守

**审查点**

- fake corridor 是否减少
- `high_value_but_low_task_fraction`
- geometry/task disagreement 报警是否提前于行为崩塌

### Stage 5：controller 降级与脚手架收缩

**实现范围**

- 收缩补偿逻辑
- 移除已被主合同覆盖的 stage / latch / bypass

**主风险**

- 误删仍承担补偿职责的逻辑

**审查点**

- 行为是否退化
- 语义指标是否恶化
- 代码复杂度是否下降

## 4. 审查安排

### 4.1 主代理职责

主代理负责：

- 锁定主线
- 拆分阶段
- 控制修改边界
- 最终验收
- 生成对账图与复盘

### 4.2 子代理职责

子代理仅承担：

- 只读审计
- 范围核对
- 风险枚举
- 特定模块探索

子代理默认不负责：

- 改变总方向
- 合并多阶段变更
- 独立定义通过标准

### 4.3 审查节奏

每一阶段必须经过 4 个检查点：

1. 设计前检查  
   当前实现是否符合 RFC 边界
2. 代码后检查  
   是否只改了该阶段允许修改的层
3. 测试后检查  
   结构测试 / 回归测试是否通过
4. 训练后检查  
   结构指标与行为指标是否同时满足

## 5. 回滚规则

### 允许的回滚

- 回滚当前阶段新增公式
- 回滚当前阶段新增字段使用方式
- 回滚当前阶段新增日志/测试以外的主逻辑接入

### 不允许的回滚

- 回到 actor-first 主修
- 回到 controller-first 主修
- 重新发明平行的新主线

### 回滚触发条件

- 结构指标未改善且行为显著退化
- 修改越过当前阶段允许边界
- 同一问题连续 3 次重复失败且无新信息增益

## 6. 必备产物清单

每完成一个阶段，必须同步更新：

- `task_plan.md`
- `findings.md`
- `progress.md`
- 对应 RFC / 控制计划状态
- 测试结果
- 结构指标摘要
- 行为步点摘要

## 7. 当前已完成项

- 正式源码备份已完成：
  - `/Users/zhangsan/Desktop/缸中之脑v5.6/备份/SAI项目源码备份-2026-03-17-143958`
- 失败但保留的首轮快照：
  - `/Users/zhangsan/Desktop/缸中之脑v5.6/备份/SAI项目源码备份-2026-03-17-143907`
- 规划与治理文件已建立
- `SemanticContract` 与 `SemanticArbiter` 已落地，Phase 1 / Phase 2 已完成
- `v195` 已证明 bootstrap 接管合同能建立稳定接触，但更像“后段恢复合同”
- `v205/v206` 已完成 Phase 3 窄口径安全性验证，结论是 actor unified contract 主通路可行，但 actor-side tail relief 全程 dormant
- `bootstrap trigger surface rewrite` 已落地，正在准备 `v207` 真实 pure-imag 验证
- 子代理审计已发起并持续作为只读复核

## 8. 当前下一步

下一步固定为：

1. 同步 `task_plan.md / findings.md / progress.md` 到 `Phase 2.5 主阶段 + Phase 3 窄口径通过` 的真实状态
2. 以 `Phase 2.5 bootstrap-only` 基线创建 `v207` 运行档案，避免误用 Phase 3 actor 实验名义
3. 启动 `v207` 真实 pure-imag run
4. 优先核对 `bootstrap_trigger_gate / bootstrap_late_gate / bootstrap_mix_surface / bootstrap_clean_mix` 是否在旧 `eval_gate` 仍低时也能开启
5. 只有当 `v207` 证明 bootstrap trigger surface 已真实激活后，才重新评估是否推进 `Phase 3`
