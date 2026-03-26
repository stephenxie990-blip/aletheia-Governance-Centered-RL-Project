# CartPole Pure-Imag Archive

这个目录承接 `cartpole pure-imag` 工作流的历史文档，并按“主入口 + 子簇归档”的方式做第五轮内容去重治理。

## 目录分层

- [analysis-design](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/analysis-design)
  - 设计说明、机制分析、阶段复盘、实现方案与结构检查
- [run-notes](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/run-notes)
  - 单次运行记录、验证 run、观测 run 与问题表
- [signal-audit-family](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/signal-audit-family/README.md)
  - 高重复的 `signal_run_audit` 主文档、变体文档与配套 JSON

## 主入口建议

如果只是要快速恢复这一段工作的主脉络，优先看这几份：
- [cartpole_pure_imag_mechanism_analysis_2026-03-14.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/analysis-design/cartpole_pure_imag_mechanism_analysis_2026-03-14.md)
- [cartpole_pure_imag_next_mainline_analysis_2026-03-14.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/analysis-design/cartpole_pure_imag_next_mainline_analysis_2026-03-14.md)
- [cartpole_pure_imag_structural_verification_checklist_2026-03-15.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/analysis-design/cartpole_pure_imag_structural_verification_checklist_2026-03-15.md)
- [cartpole_pure_imag_signal_run_audit_2026-03-14.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/signal-audit-family/cartpole_pure_imag_signal_run_audit_2026-03-14.md)

## 去重规则

- 不删除历史内容，只压缩目录噪声
- 高重复 family 必须集中存放，不再平铺在 workstream 根层
- JSON 原始对账资产与对应 Markdown 一起保留，避免“只剩结论、丢了底稿”
- 之后如果继续新增 `pure-imag` 历史稿，默认进入对应子簇，而不是直接丢回本目录根层
