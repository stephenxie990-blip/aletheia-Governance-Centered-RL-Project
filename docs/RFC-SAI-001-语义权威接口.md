# RFC-SAI-001：语义权威接口

## 1. 摘要

本 RFC 将项目后续主线正式锁定为：

> 建立一个统一的语义权威接口，使 actor、critic、corridor 不再直接消费“裸 value tensor”，而是消费一个携带来源、覆盖率、可信度、认证状态与接管权的语义合同对象。

这份 RFC 不解决所有实现细节，但它锁定后续实施的方向、边界、阶段顺序与不可变原则。

## 2. 问题定义

当前系统的主病灶不是单一超参或单一 gate，而是一个跨模块的接口漏洞：

- critic bootstrap 可以默认信自己
- actor 与 critic 的 trust 语义并非完全共享
- corridor 更像几何认证器，而不是任务语义认证器
- 外生真实锚存在，但没有作为统一的一等语义对象进入系统
- 为了弥补上述缺口，系统长出了大量 controller stage、latch、bypass 与补偿性 gate

这导致项目表现为：

- 分数可以冲高，但难以持续
- 行为有时稳定，语义却在膨胀
- fake corridor 可能长期存在
- 修复常常表现为“局部有效，但不终局”

## 3. 非目标

本 RFC 明确不以以下事项为第一阶段目标：

- 不重写 world model / bridge 主逻辑
- 不重新把 actor 作为第一主战场
- 不继续扩展 controller stage
- 不新造一套 critic loss family
- 不一次性全局替换所有模块接口

## 4. 核心对象

### 4.1 SemanticContract

`SemanticContract` 是跨模块流动的最小语义对象。第一版至少应包含：

- `value`
- `source`
- `coverage`
- `confidence`
- `authority`
- `trust`
- `task_agreement`
- `registry_support`
- `semantic_debt`
- `freshness`
- `certified_by`

### 4.2 SemanticArbiter

`SemanticArbiter` 是唯一的仲裁点，负责在多个 contract 之间决定最终接管权。

它不是新 loss，不是新 controller，而是统一回答：

- 当前谁更可信
- 谁拥有最低接管权
- 哪个 source 只能做 support，不能做 authority

## 5. Producer / Consumer 边界

### 5.1 Producers

- replay suffix clean target
- MC / short-return anchor
- critic bootstrap value
- task corridor certifier
- registry support recorder

### 5.2 Consumers

- critic bootstrap target
- lambda-return / target_critic
- actor main contract trust source
- corridor combined certification

## 6. 不可变原则

### 原则 1：外生真值在 coverage > 0 的位置永远有底线权威

外生来源可以不全覆盖，可以不总是占优，但不能在有 coverage 的位置被压成 0 权威。

### 原则 2：critic 永远不能拥有无限权威

critic 是内生自举来源，不允许在高风险阶段以 100% 权威垄断 bootstrap 语义。

### 原则 3：几何认证不能单独推出任务真理

geometry cert 只能作为 support，不足以单独授予任务真理或高 authority。

### 原则 4：actor / critic / corridor 最终必须消费同一套语义语言

允许阶段性存在不同 consumer 逻辑，但不允许长期存在完全平行的 trust 语义。

### 原则 5：调参数可以，改方向不可以

允许调整 floor、上限、阈值、ramp，但不允许回到：

- actor-first 主修
- controller-first 主修
- bridge-first 主修
- 多层散装 gate 补洞

## 7. 固定实施顺序

### Step 0：治理固化

- 建立 planning files
- 创建正式源码备份
- 记录最小环境快照
- 固化 RFC 与实施控制计划

### Step 1：定义 `SemanticContract`

- 先做数据结构与日志
- 不改训练主行为
- 加结构单测

### Step 2：critic bootstrap 引入 authority-driven takeover floor

- 第一个真正的主线改动
- 不动 actor
- 不动 corridor 主认证
- 只重写 bootstrap 接触合同

### Step 3：actor 消费统一 contract

- 保持 actor 主合同结构尽量不变
- 统一 trust 来源

### Step 4：corridor 双认证

- geometry cert
- task cert
- combined cert 不能再让 geometry 冒充 task truth

### Step 5：controller 降级

- controller 从主修复器退为补偿层
- 清理历史补偿逻辑

### Step 6：终局固化

- 统一 contract / arbiter 工程边界
- 清理重复指标与脚手架
- 输出终局对账图

## 8. 当前代码映射

当前主落点如下：

- clean target 来源：
  - `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py` 中 `_compute_seed_replay_suffix_targets`
- task corridor：
  - `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py` 中 `_compute_task_corridor_signals`
- actor semantic trust：
  - `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py` 中 `actor_contract_semantic_pressure / actor_contract_semantic_trust`
- critic bootstrap contract：
  - `/Users/zhangsan/Desktop/缸中之脑v5.6/aletheia/aletheia_train.py` 中 `critic_contract_bootstrap_*`

## 9. 阶段性验收标准

### Step 1

- 每个主语义对象都能输出 `source / coverage / confidence`
- 新增 contract 结构单测通过

### Step 2

- `bootstrap_clean_mix_mean` 不再长期停留在 0.04 级别
- `bootstrap_contact_floor_mean` 在关键阶段稳定非零
- `real_mc_value_gap_abs_mean` 中后段上升斜率下降
- `eval@1000` 不得显著差于当前最优基线

### Step 3

- actor distrust 与 critic authority 不再长期错位

### Step 4

- fake corridor 必须能被 geometry/task 双认证提前识别

### Step 5-6

- controller 状态收缩
- 冗余脚手架减少
- 主行为与语义指标不退化

## 10. 回滚纪律

- 每一步失败时，只回滚该步实现
- 不改变总方向
- 不允许因单次失败回到旧的散装补丁路线

## 11. 决策结论

从本 RFC 生效起，项目的正式主线固定为：

> `SemanticContract -> SemanticArbiter -> Critic Bootstrap -> Actor / Corridor 接入 -> Controller 退场`

任何新增实现都必须明确回答：

1. 它属于 contract、arbiter、consumer 还是补偿层？
2. 它是否新增了不受控的 authority 来源？
3. 它是否让 geometry 再次冒充了 task truth？
