#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "docs" / "archive"

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\((/Users/zhangsan/Desktop/缸中之脑v5\.6/docs/archive/[^)]+)\)")
GENERIC_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

TEMP_NAME_PATTERNS = (
    "新建文档",
    "未命名",
    "temp",
    "tmp",
)


def is_non_leaf_directory(path: Path) -> bool:
    return any(child.is_dir() for child in path.iterdir())


def readme_coverage_issues() -> list[str]:
    issues: list[str] = []
    for path in sorted(ARCHIVE_ROOT.rglob("*")):
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        if not is_non_leaf_directory(path):
            continue
        if not (path / "README.md").exists():
            issues.append(f"Missing README: {path}")
    return issues


def filename_issues() -> list[str]:
    issues: list[str] = []
    for path in sorted(ARCHIVE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        lowered = name.lower()
        if name.startswith("#") or name.startswith(" "):
            issues.append(f"Leading symbol/space: {path}")
        if " " in name:
            issues.append(f"Space in filename: {path}")
        if any(token in lowered for token in TEMP_NAME_PATTERNS):
            issues.append(f"Temporary placeholder name: {path}")
    return issues


def broken_archive_links() -> list[str]:
    issues: list[str] = []
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in {"outputs", "tmp", ".venv"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in MARKDOWN_LINK_RE.finditer(text):
            target = Path(match.group(1))
            if not target.exists():
                issues.append(f"Broken archive link in {path}: {target}")
    return issues


def hotspot_warnings() -> list[str]:
    warnings: list[str] = []
    legacy_hotspots = [
        "# 统一训练入口方案与技术路径.md",
        "actor主合同重绑定与certified corridor再认证系统设计图-2026-03-16-1420.md",
    ]
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in {"outputs", "tmp", ".venv"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        link_targets = [match.group(1) for match in GENERIC_LINK_RE.finditer(text)]
        for hotspot in legacy_hotspots:
            for target in link_targets:
                if hotspot in target:
                    warnings.append(f"Legacy name still used in link target in {path}: {hotspot}")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Semi-automated archive audit")
    parser.add_argument("--summary-only", action="store_true", help="Only print totals")
    args = parser.parse_args()

    error_groups = {
        "readme": readme_coverage_issues(),
        "filenames": filename_issues(),
        "broken_links": broken_archive_links(),
    }
    warn_groups = {
        "legacy_hotspots": hotspot_warnings(),
    }

    error_count = sum(len(v) for v in error_groups.values())
    warn_count = sum(len(v) for v in warn_groups.values())

    if args.summary_only:
        print(f"errors={error_count}")
        print(f"warnings={warn_count}")
        return 1 if error_count else 0

    print("ARCHIVE AUDIT REPORT")
    print("====================")
    print(f"archive_root: {ARCHIVE_ROOT}")
    print(f"errors: {error_count}")
    print(f"warnings: {warn_count}")
    print()

    for group_name, issues in error_groups.items():
        print(f"[ERROR:{group_name}] {len(issues)}")
        for issue in issues:
            print(issue)
        print()

    for group_name, issues in warn_groups.items():
        print(f"[WARN:{group_name}] {len(issues)}")
        for issue in issues:
            print(issue)
        print()

    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
