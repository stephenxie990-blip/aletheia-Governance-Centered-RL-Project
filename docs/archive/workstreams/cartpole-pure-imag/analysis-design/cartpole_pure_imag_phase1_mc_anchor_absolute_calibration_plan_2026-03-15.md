# CartPole Pure-Imag Phase 1 Plan: MC Anchor Absolute Calibration

Date: 2026-03-15
Status: approved for implementation

## Overall Goal

第一阶段目标不是“把所有问题一次修完”，而是用最小变量、最强因果辨识力的方式，验证并修复当前最主要的结构病灶：

- `critic family common-mode inflation`
- 发生位置：`high-value actor-use corridor`
- 当前缺口：缺少真正外生的 absolute semantic anchor

## Phase 1 Scope

第一阶段只做：

1. 在 replay 真实轨迹上构造 **full Monte Carlo return** anchor
2. 将真实状态映射到 actor-use bridge / policy feature space
3. 对 online critic 增加 **corridor-gated absolute calibration loss**
4. 增加 common-mode inflation 相关监控指标

第一阶段明确不做：

- 不改 actor calibrated branch
- 不改 controller / rescue 路线
- 不加新的 bootstrap critic tail anchor
- 不恢复 legacy critic path
- 不把更多 feature regularization 当主线

## Why This Phase Is Minimal And Correct

选择 `full MC return` 而不是 `n-step + critic tail` 的原因：

- `full MC return` 不经过 critic，自身不受 common-mode inflation 污染
- CartPole replay 中有完整 episode，可直接构造真正外生锚
- 第一阶段要验证的是“critic family 是否因为缺绝对锚而共同漂移”
- 用 MC anchor 能给出最干净的 yes/no 结论

## Engineering Plan

### Stage A: Replay MC Anchor

在 replay buffer sampling 路径中，为 sampled real sequence 追加：

- `value_real_mc`
- 或直接把 `value_real` 切到 MC 版本

要求：

- 仅由真实 rewards / dones 计算
- done 后停止累计
- 零 critic bootstrap
- 与 sampled sequence 的时间对齐

### Stage B: Corridor-Gated Critic Calibration

复用现有 `value_real_anchor` critic 路径，不新开旁路。

新增：

- MC-based target ingestion
- 保守 corridor mask
- absolute calibration metrics

第一版 corridor 策略：

- 只在 replay batch 中高价值 top 段开启强校准
- 其余状态弱化或关闭

### Stage C: Verification

先验收机制，再验收分数。

先看：

- `critic/value_real_anchor`
- `critic/real_mc_value_gap_abs_mean`
- `critic/real_mc_corridor_gap_abs_mean`
- `critic/value_target_gap_abs_mean`
- `imag/open_loop_audit_teacher_value_abs_to_short_return_mean`
- `imag/open_loop_audit_imag_value_abs_to_short_return_mean`

再看：

- late-stage current-checkpoint eval 是否仍整体塌回低分

## Review Mechanism

本阶段采用四道审查：

1. **Structure review**
   - 是否只动 critic-side
   - 是否保持现有主训练链路整洁
2. **Contract review**
   - 是否复用现有 `value_real_anchor` contract
   - 是否没有偷偷引入 critic bootstrap 污染
3. **Test review**
   - RED/GREEN 覆盖：
     - replay MC target correctness
     - corridor gating correctness
     - critic anchor metrics exposure
4. **Behavior review**
   - 核心回归 tests 通过
   - 指标方向符合诊断

## Skills Arrangement

本阶段只使用最小技能集：

- `pi-planning-with-files`
  - 管理阶段目标、进度、发现、验证结果
- `python-testing`
  - 先写失败测试，再实现
- `verification-loop`
  - 在实现后跑定向测试与核心回归检查

不额外引入其他 skills，避免流程膨胀。

## Exit Criteria

Phase 1 完成的标准：

1. 代码中已经存在可配置的 MC-based critic absolute calibration
2. 定向测试覆盖：
   - replay MC anchor
   - corridor gating
   - critic metrics
3. 核心回归测试通过
4. 形成新的独立实现/验证报告与记录追加
