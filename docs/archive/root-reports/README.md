# Root Reports Archive

这个目录承接原先散落在仓库根目录的历史性报告、审计、对账与执行计划文档。

归档规则固定为：
- `2026-03/`：带明确 `2026-03-*` 时间戳的阶段性文档
- `undated/`：没有显式日期、但已经不属于当前活跃工作面的历史文档

本轮归档目标是“根目录只保留活跃工作记忆文件”，因此仓库根目录当前只保留：
- `task_plan.md`
- `findings.md`
- `progress.md`

归档结果：
- `2026-03/`：31 份历史文档
- `undated/`：4 份历史文档
- 合计：35 份历史文档

导航入口：
- [archive index](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/README.md)
- [naming conventions](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/NAMING-CONVENTIONS.md)
- [2026-03 索引](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/2026-03/README.md)
- [undated 索引](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/root-reports/undated/README.md)

治理约束：
- 不迁移 `outputs/`、`tmp/`、`.venv/` 下的实验或环境资产
- 不删除历史报告内容，只做集中归档与路径修正
- 仓库内原有绝对路径引用已同步改写到归档后的新位置
- 命名治理遵循 [naming conventions](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/NAMING-CONVENTIONS.md)
