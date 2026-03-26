# CartPole Pure-Imag Bridge Contract Phase 1 Implementation

Date: 2026-03-15

## 目标

本阶段只做第一刀，而且严格遵守当前设计原则：

- 不削弱 `iron wall`
- 不回退到 controller-rescue-first
- 不重新打开 imagined critic 对 world model 的主梯度污染路径
- 不继续保留主训练链路里的手写简化版 Huber critic loss

本阶段要完成的是：把项目里已经设计过、但主线没有真正闭合的 `bridge contract` 先恢复到“正式可训练、正式可验证”的状态。

## 本次实现内容

### 1. 删除主训练路径里的手写简化 critic loss

- `TrainingStep.run_step()` 不再在主链路里直接手写 `F.huber_loss(...)`
- critic 更新统一改为走 `critic.compute_loss(...)` 正式合约
- 若 critic 不实现该合约，训练直接报错，而不是偷偷退回旧逻辑
- 若 critic 支持 `include_router_loss` 参数，则当前显式传入 `False`

这一改动的意义不是“代码更整洁”而已，而是：

- 多 gamma critic
- double / target critic
- router-aware critic

这些高级 critic 能力以后终于不会再被主训练入口绕开

### 2. 恢复 world model 侧已存在但未真正接线的 bridge loss

本次重新接回并纳入主 loss 聚合的 bridge 项包括：

- `projection`
- `control_align`
- `control_reg`
- `consistency_head`
- `aux_inverse`
- `aux_value`
- `abstractor`

其中 `abstractor` 仍保持“开关和权重都打开才真正生效”的原设计，不做强行激活。

### 3. 给 bridge maintenance 一个非零晚期地板

新增：

- `bridge_maintenance_floor`
- `get_bridge_maintenance_scale()`

这一步的目的，是避免系统在 full wall 条件下把 `bridge maintenance` 一起衰减到零。  
也就是说：

- `truth isolation` 继续维持
- `bridge upkeep` 不再跟着一起彻底停摆

### 4. 保证 bridge 模块在 optimizer 建立前就物化

在 `setup_action_space()` 阶段提前 materialize：

- `projection`
- `control_head`
- `abstractor`
- 以及相关 bridge 子模块

这样它们的参数会在 optimizer 创建前进入参数集合，避免出现“逻辑上有 loss，实际上参数根本没进 optimizer”的隐性空转。

### 5. 让 loss packet 和训练指标重新反映 bridge 训练状态

本次把 bridge 相关损失重新放回 `PredictiveLossPacket`，并且：

- 计入 `core_loss`
- 回传到训练 metrics

这样后续真实训练时，bridge 是否真的在工作，不再只能靠猜。

### 6. 升级测试替身，而不是给生产代码加兼容回退

由于主训练入口现在正式要求 critic 合约，integration tests 里所有遗留的“裸 `nn.Linear` critic”或只实现 `forward()` 的 critic stub 都被升级成了正式 contract 版本。

这里刻意没有做的一件事是：

- 没有为了兼容旧测试，把生产代码重新塞回 legacy fallback

这保证了“删除手写简化 critic loss”不是表面动作，而是真正成为主线约束。

## 本次验证

通过的验证包括：

- `./.venv/bin/python -m unittest aletheia.tests.test_training_entrypoints -v`
- `./.venv/bin/python -m unittest aletheia.tests.test_core_components -v`
- `./.venv/bin/python -m unittest aletheia.tests.test_run_train_contracts -v`
- `./.venv/bin/python -m unittest aletheia.tests.test_pure_imagination_path -v`
- `./.venv/bin/python -m unittest aletheia.tests.test_training_loop_integration -v`
- `./.venv/bin/python -m py_compile aletheia/aletheia_train.py aletheia/aletheia_world_model.py aletheia/aletheia_config.py aletheia/tests/test_training_entrypoints.py aletheia/tests/test_core_components.py aletheia/tests/test_training_loop_integration.py`

结果：

- 单元与集成回归共 `283` 条测试全部通过
- 语法编译检查通过

## 当前判断

本阶段已经完成了“修桥合同”的第一步，但需要明确边界：

- 这次修的是训练契约缺口
- 不是已经证明真实训练成绩必然恢复到稳定 `500`

更准确地说，本次已经把以下结构性问题修正到位：

- 主训练入口不再绕开正式 critic 体系
- bridge loss 不再停留在“定义过但没真训练”
- bridge 参数不再因为 lazy materialization 漏进 optimizer
- full wall 不再把 bridge maintenance 一并衰减为零

## 下一阶段建议

如果继续按当前上游优先的路线推进，下一阶段应当直接做：

1. 把 replay open-loop audit 升级为真正的 `policy-space open-loop consistency loss`
2. 让 `teacher policy-space` 与 `imagined policy-space` 的错位直接进入 bridge 训练
3. 在此基础上再开新的真实训练 run，验证是否同时做到：
   - imagined 语义继续稳定
   - actor corridor 推进力不再被压钝
   - 不再出现 solved 后 `500 -> 88` 级别回撤

## 一句话结论

第一阶段已经落地完成：  
主线不再靠手写简化 critic loss 和未接线 bridge 组件勉强维持，而是恢复到了“critic 合约完整、bridge loss 真训练、bridge maintenance 不归零”的正式状态。
