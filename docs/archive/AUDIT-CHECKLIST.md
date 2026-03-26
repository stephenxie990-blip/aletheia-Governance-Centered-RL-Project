# Archive Audit Checklist

这份清单把前十轮归档治理沉淀下来的规则，压成一套“可执行 + 可复核”的归档审计草案。

目标不是一次性把归档区变成全自动系统，而是先建立一套稳定的半自动巡检流程：
- 机器负责扫结构、命名、链接
- 人负责判断例外是否合理

## Scope

当前巡检范围固定为：
- `docs/archive/`

不纳入本清单的范围：
- `outputs/`
- `tmp/`
- `.venv/`
- 代码逻辑正确性

## Audit Command

推荐命令：

```bash
bash scripts/archive_audit.sh
```

如果需要更紧凑的结果：

```bash
bash scripts/archive_audit.sh --summary-only
```

底层实现仍然是：

```bash
python3 tools/archive_audit.py
```

## Automatic Checks

脚本当前会自动检查：

1. README coverage
- 归档目录中的每个非叶子簇目录，是否具备 `README.md`

2. High-risk filename violations
- 是否存在前导符号文件名
- 是否存在空格文件名
- 是否存在临时占位式名字

3. Broken local archive links
- Markdown 中指向本地 `docs/archive` 的绝对路径链接是否存在目标文件

4. Naming hotspot review
- 是否仍然出现已知高风险旧名字作为活链接目标

## Manual Review Checklist

自动检查通过后，仍要做一轮人工审查：

1. 新增文档是否放在了正确归档簇
2. 新增目录是否补了 README
3. README 是否写清：
- 目录承接内容
- 当前分层
- 主入口建议
- 命名或落位规则
4. 新的文件名是否只是“看起来规范”，但实际语义更差
5. 历史改名是否只为了美观，而不是为了修高风险异常

## Pass Criteria

一次巡检可以判为通过，需要同时满足：
- 自动检查无 `ERROR`
- `WARN` 仅为已知、已记录的历史例外
- 人工复核没有发现新目录缺 README
- 人工复核没有发现新的高风险异常命名

## Current Policy

当前命名治理口径固定为：
1. 先修高风险异常
2. 再补 README 与索引规则
3. 最后才考虑风格统一

对应正式规则见：
- [NAMING-CONVENTIONS.md](/Users/zhangsan/Desktop/缸中之脑v5.6/docs/archive/NAMING-CONVENTIONS.md)

## Known Safe Exceptions

当前允许保留的历史差异包括：
- 中文标题型与英文 workstream 型并存
- `Phase4-*` 与 `cartpole_pure_imag_*` 两套历史命名体系并存
- 已进入稳定索引链、且不构成路径风险的旧风格标题

## Suggested Cadence

建议在以下时机跑一次：
- 新增一批历史文档后
- 完成一轮归档重排后
- 准备继续做更深层治理前
- 担心 README 链、链接路径或命名规则再次退化时

## Standard Integration

当前标准接入方式固定为：
- 人工执行入口：
  - [archive_audit.sh](/Users/zhangsan/Desktop/缸中之脑v5.6/scripts/archive_audit.sh)
- 底层检查器：
  - [archive_audit.py](/Users/zhangsan/Desktop/缸中之脑v5.6/tools/archive_audit.py)

推荐把它当成以下节点的标准 gate：
- 每轮归档治理收口前
- 新增 README / 索引 / 重命名后
- 准备宣布“本轮文档治理完成”前

最小执行流程：
1. 跑 `bash scripts/archive_audit.sh`
2. 若 `errors > 0`，先修复再继续
3. 若只有 `warnings`，按人工复核清单判断是否可接受
4. 在 `progress.md` 中记录本轮审计结果
