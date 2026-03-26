# CartPole Pure-Imag `v164` Bridge Contract Phase 1 Validation Run

Date: 2026-03-15

## Run

- Run dir:
  - `outputs/exp_seed42_v164_h15_light_sc_bridge_phase1_validation_2500_20260315_01`
- Command:
  - `./.venv/bin/python scripts/cartpole_train.py --env CartPole-v1 --update-steps 2500 --collect-steps-per-cycle 32 --train-steps-per-cycle 4 --seed 42 --device auto --save outputs/exp_seed42_v164_h15_light_sc_bridge_phase1_validation_2500_20260315_01 --imagination-only --imagination-horizon 15 --enable-eval --eval-interval 250 --eval-episodes 5 --eval-max-steps 500 --log-interval 50 --save-interval 250 --pretrain-ratio 0.05 --warmup-ratio 0.0 --overrides '{"rssm_msc":{"enabled":false},"rssm_shortcut_consistency":{"enabled":true,"horizons":[2],"loss_scale":0.1,"sample_ratio":0.25,"max_starts":2}}'`
- Validation intent:
  - 在不新增 target-ruler、不过度引入 controller 变量的前提下，验证 `Phase 1 / bridge contract restoration` 是否已经：
    - 真正接通训练
    - 不造成主线回归
    - 并足以单独修复纯 imag CartPole 的主问题

## Why This Run Was Chosen

这条 run 刻意使用了接近 `v151` 的轻量基线：

- `horizon=15`
- `MSC=false`
- light `SC` only
- 不启用 target-ruler
- controller stage 全程保持自然演化

这样做的目的，是尽量减少额外变量，让这次验证主要回答：

`Phase 1 修桥本身够不够`

## Eval Trajectory

| eval step | mean |
| --- | ---: |
| 250 | 9.6 |
| 500 | 9.6 |
| 750 | 9.2 |
| 1000 | 104.8 |
| 1250 | 13.2 |
| 1500 | 32.4 |
| 1750 | 49.6 |
| 2000 | 113.0 |
| 2250 | 50.4 |
| 2500 | 117.6 |

补充：

- best eval = `117.6 @ 2500`
- final eval = `117.6 @ 2500`
- `post_entry_audit` 显示 controller stage 全程为 `idle`

因此这条验证线没有靠 controller rescue。

## Direct Validation Result

## 1. Phase 1 is real, not nominal

这条 run 明确证明：

- `wm/bridge/maintenance_scale` 全程非零，且晚期仍维持在 `0.89+`
- `wm/bridge/control_align`
- `wm/bridge/aux_inverse`
- `wm/bridge/aux_value`

都持续非零

这说明：

- bridge contract restoration 已经真进入训练主线
- 不是“代码连上了，但实际上没在学”

## 2. Phase 1 does not solve the main problem by itself

虽然 bridge 真的在训练，但这条 run 仍然没有进入稳定 corridor。

更准确地说，它表现为：

- 早期长期低位：`250/500/750` 基本都在 `9~10`
- 中后段出现脉冲式抬升：`1000 -> 104.8`，`2000 -> 113.0`
- 但每次抬升后都守不住：`1250 -> 13.2`，`2250 -> 50.4`

所以这不是 solved 线，而是一条：

`bridge 有训练，但 actor-use corridor 仍然时通时断`

## 3. Actor-use semantic mismatch is still the dominant disease

这条 run 最关键的内部信号是：

- `actor/online_adv_mean`
  - `250`: `+4.10`
  - `750`: `-12.58`
  - `1500`: `-26.15`
  - `2000`: `-39.25`
  - `2500`: `-44.84`
- `critic/value_target_gap_abs_mean`
  - `250`: `4.13`
  - `750`: `13.95`
  - `1500`: `27.21`
  - `2250`: `41.12`
  - `2500`: `49.08`
- `imag/open_loop_audit_value_gap_mean`
  - `250`: `0.95`
  - `750`: `17.76`
  - `1500`: `22.56`
  - `2250`: `60.48`
  - `2500`: `94.94`

这三条线说明：

- bridge 在维护
- actor 训练真正使用的 imagined value geometry 仍然在持续跑偏
- 而且这种跑偏是越到后段越严重

因此本轮验证把结论钉得很清楚：

`Phase 1 修复了“桥没训练”这个结构错误，但没有修复“actor-use imagined 语义失真”这个机制错误。`

## Comparison With `v151`

| run | best eval | final eval | shape |
| --- | ---: | ---: | --- |
| `v151` | 500.0 @ 1250 | 18.4 @ 2500 | 能碰到 solved，但后段彻底掉穿 |
| `v164` | 117.6 @ 2500 | 117.6 @ 2500 | 明显没 solved，但后段不再完全掉穿到双位数以下 |

这意味着：

- `v164` 比 `v151` 更稳
- 但也明显更弱

所以对 `Phase 1` 的最准确评价不是“成功”或“失败”二选一，而是：

`结构上正确，训练上真实，性能上不够。`

## Judgment

## Elegant?

是。

因为它做的是：

- 恢复 post-wall bridge training contract
- 不靠 controller 救火
- 不靠重新开墙
- 不靠把 legacy critic path 塞回主线

这在设计上是对的。

## Accurate?

是。

因为这条 run 的信号和我们的结构诊断完全吻合：

- bridge 已经活了
- 但 actor-use imagined space 还没被真正对齐

它没有推翻诊断，反而强化了诊断。

## Correct?

部分正确。

更准确地说：

- 对“修复方向”而言是正确的
- 对“是否已经足够解决主问题”而言还不正确

它是必要条件，但不是充分条件。

## Final Conclusion

这条验证 run 结束后，关于 `Phase 1` 可以正式下结论：

1. `bridge contract restoration` 已经真实生效。
2. 它没有引入主线回归，也没有依赖 controller rescue。
3. 但它不能单独修复 pure-imag CartPole 的主病灶。
4. 下一步应当直接进入：
   - `policy-space open-loop consistency loss`
   - 用 replay open-loop teacher vs imagined policy-space 对账，直接约束 actor-use imagined manifold
5. 不应该回头：
   - 补 controller
   - 恢复 legacy critic path
   - 或把 target critic 当成主要真值来源
