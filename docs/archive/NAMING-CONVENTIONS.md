# Archive Naming Conventions

这份文档定义 `docs/archive/` 下历史文档与归档目录的命名规则。

目标不是“把所有历史名字洗成一种风格”，而是：
- 保证路径稳定
- 提高可读性
- 避免新增高风险异常名
- 在不破坏历史证据链的前提下逐步收口

## Core Rule

命名治理优先级固定为：
1. 先修高风险异常
2. 再补索引和 README 规则
3. 最后才考虑风格统一

高风险异常优先于美观统一。

## Must Avoid

今后新增归档文件或目录时，禁止出现：
- 文件名前导符号
  - 例如：`# foo.md`
- 文件名中的空格
  - 例如：`foo bar.md`
- 临时占位式名字
  - 例如：`新建文档.md`、`未命名.md`、`temp.md`
- 无语义缩写导致不可读
  - 除非该缩写已经在项目内稳定使用

## Allowed Styles

当前归档区允许并保留以下历史风格：
- 中文标题型
  - 例如：`项目系统性修复实施方案-2026-03-12.md`
- 英文/拼音 workstream 型
  - 例如：`cartpole_pure_imag_signal_run_audit_2026-03-14.md`
- 中英混合标题型
  - 例如：`Phase4-决策版摘要-2026-03-20.md`

这些风格可以并存，不要求一次性统一。

## Date Rule

若文档本身具有明确阶段时间窗，优先保留日期后缀：
- 推荐：`主题-YYYY-MM-DD.md`
- 已有英文 workstream 习惯时，可保留：
  - `topic_YYYY-MM-DD.md`
  - `topic_YYYY-MM-DD_variant.md`

没有显式日期的历史文档，不强制补日期，但应放在合适的 `undated/` 归档区。

## Separator Rule

分隔符规则如下：
- 中文标题链路优先使用连字符 `-`
- 英文 workstream 系列允许保留下划线 `_`
- 如果同一文件名里已经存在稳定的 `_` 系列表达，不强制改成 `-`
- 但应避免“空格作为结构分隔符”

允许：
- `cartpole_pure_imag_v165_policy_open_loop_consistency_validation_run_2026-03-15.md`
- `Phase5-6-清理收口执行计划-2026-03-20.md`

不推荐新增：
- `foo bar baz.md`

## When To Rename Historical Files

只有在以下情况才建议重命名历史文件：
- 文件名前导符号或空格会影响链接稳定性
- 名字会显著误导读者
- README / 索引层已经形成稳定入口，需要把异常名收口

不建议仅因为“风格不统一”就批量重命名历史文件。

## Rename Procedure

如果确实需要重命名，固定按这个顺序执行：
1. 先确认引用面是否可控
2. 物理重命名文件
3. 修正 README、索引页和治理记录中的链接
4. 复核旧路径是否只剩“改名说明”，不再作为活链接存在

## README Rule

每个归档簇 README 至少应明确：
- 该簇承接什么内容
- 当前分层
- 主入口建议
- 命名或落位规则

如果一个目录已经大到无法靠文件名直接导航，就应该先补 README，再考虑别的整理动作。

## Current Exceptions

当前刻意保留、不做强制统一的历史命名差异包括：
- 中文标题与英文 workstream 并存
- `Phase4-*` 与 `cartpole_pure_imag_*` 两套不同历史命名体系并存
- 已进入稳定索引链的旧标题风格

## Applied Fixes

本轮之前已经完成的高风险命名修复：
- `# 统一训练入口方案与技术路径.md` -> `统一训练入口方案与技术路径.md`
- `actor主合同重绑定与certified corridor再认证系统设计图-2026-03-16-1420.md` -> `actor主合同重绑定与certified-corridor再认证系统设计图-2026-03-16-1420.md`
