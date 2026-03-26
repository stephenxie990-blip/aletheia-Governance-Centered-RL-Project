# Archive Index

这个目录承接仓库中已经退出“活跃工作面”的历史文档，但这些内容仍然需要可回溯、可审计、可导航。

当前归档区分成两大块：

规则文档：
- [naming conventions](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/NAMING-CONVENTIONS.md)
- [audit checklist](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/AUDIT-CHECKLIST.md)

标准命令入口：
- [archive_audit.sh](/Users/zhangsan/Desktop/缸中之脑v5.6/scripts/archive_audit.sh)

## Root Reports

- [root-reports](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/README.md)
  - 原先散落在仓库根目录的阶段报告、审计、对账、治理计划
  - 适合按“时间窗口”和“治理阶段”恢复主线

其中：
- [2026-03 索引](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/README.md)
- [undated 索引](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/undated/README.md)

## Workstreams

- [workstreams](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/README.md)
  - 原先散落在 `docs/` 根层的历史工作流文档
  - 适合按主题和子系统恢复具体实现演化

其中：
- [cartpole-pure-imag](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/cartpole-pure-imag/README.md)
- [actor-contract](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/actor-contract/README.md)
- [control-stack](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/workstreams/control-stack/README.md)

## Archive Rule

- 归档区不等于垃圾桶；归档内容默认仍然可被计划链、发现链和审计链引用
- 新增历史文档时，优先先决定它属于 `root-reports` 还是 `workstreams`
- 新增一个新的归档簇时，应同时补 README，避免再次退回“靠目录名猜结构”
- 新增历史文件名时，遵循 [naming conventions](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/NAMING-CONVENTIONS.md)
- 归档巡检执行与通过标准见 [audit checklist](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/AUDIT-CHECKLIST.md)
