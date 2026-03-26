#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _transform_obj(
    obj: Any,
    *,
    rename_release_coupling_to_observed: bool,
    rename_release_coupling_to_historical: bool,
) -> Any:
    if isinstance(obj, dict):
        transformed: dict[str, Any] = {}
        for key, value in obj.items():
            new_key = key
            if key == "critic/critic_contract_bootstrap_source_recert_gate_mean":
                new_key = "critic/critic_contract_bootstrap_source_recert_gate_composite_mean"
            elif key == "critic_contract_bootstrap_source_recert_gate_mean":
                new_key = "critic_contract_bootstrap_source_recert_gate_composite_mean"
            elif key == "contract/bootstrap_external_authority_bonus_midlate_clamped_mean":
                new_key = "contract/bootstrap_external_authority_bonus_postrecert_mean"
            elif key == "bootstrap_external_authority_bonus_midlate_clamped_mean":
                new_key = "bootstrap_external_authority_bonus_postrecert_mean"
            elif key == "critic/bootstrap_external_authority_bonus_midlate_clamped_mean":
                new_key = "critic/bootstrap_external_authority_bonus_postrecert_mean"
            elif rename_release_coupling_to_observed:
                if key == "critic/critic_contract_bootstrap_bonus_authority_release_coupling_mean":
                    new_key = (
                        "critic/critic_contract_bootstrap_bonus_authority_release_coupling_observed_mean"
                    )
                elif key == "critic/bootstrap_bonus_authority_release_coupling_mean":
                    new_key = "critic/bootstrap_bonus_authority_release_coupling_observed_mean"
                elif key == "critic_contract_bootstrap_bonus_authority_release_coupling_mean":
                    new_key = (
                        "critic_contract_bootstrap_bonus_authority_release_coupling_observed_mean"
                    )
            elif rename_release_coupling_to_historical:
                if key == "critic/critic_contract_bootstrap_bonus_authority_release_coupling_mean":
                    new_key = (
                        "critic/critic_contract_bootstrap_bonus_authority_release_coupling_historical_mean"
                    )
                elif key == "critic/bootstrap_bonus_authority_release_coupling_mean":
                    new_key = "critic/bootstrap_bonus_authority_release_coupling_historical_mean"
                elif key == "critic_contract_bootstrap_bonus_authority_release_coupling_mean":
                    new_key = (
                        "critic_contract_bootstrap_bonus_authority_release_coupling_historical_mean"
                    )

            transformed_value = _transform_obj(
                value,
                rename_release_coupling_to_observed=rename_release_coupling_to_observed,
                rename_release_coupling_to_historical=rename_release_coupling_to_historical,
            )

            if (
                new_key in transformed
                and new_key.endswith("bonus_postrecert_mean")
                and transformed[new_key] == transformed_value
            ):
                continue
            transformed[new_key] = transformed_value
        return transformed
    if isinstance(obj, list):
        return [
            _transform_obj(
                item,
                rename_release_coupling_to_observed=rename_release_coupling_to_observed,
                rename_release_coupling_to_historical=rename_release_coupling_to_historical,
            )
            for item in obj
        ]
    return obj


def _rewrite_json(
    path: Path,
    *,
    rename_release_coupling_to_observed: bool,
    rename_release_coupling_to_historical: bool,
) -> bool:
    original = path.read_text(encoding="utf-8")
    data = json.loads(original)
    migrated = _transform_obj(
        data,
        rename_release_coupling_to_observed=rename_release_coupling_to_observed,
        rename_release_coupling_to_historical=rename_release_coupling_to_historical,
    )
    rendered = json.dumps(migrated, ensure_ascii=False, indent=2) + "\n"
    if rendered == original:
        return False
    path.write_text(rendered, encoding="utf-8")
    return True


def _rewrite_jsonl(
    path: Path,
    *,
    rename_release_coupling_to_observed: bool,
    rename_release_coupling_to_historical: bool,
) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = [line for line in original.splitlines() if line.strip()]
    migrated_lines = []
    for line in lines:
        data = json.loads(line)
        migrated = _transform_obj(
            data,
            rename_release_coupling_to_observed=rename_release_coupling_to_observed,
            rename_release_coupling_to_historical=rename_release_coupling_to_historical,
        )
        migrated_lines.append(json.dumps(migrated, ensure_ascii=False))
    rendered = "\n".join(migrated_lines) + ("\n" if migrated_lines else "")
    if rendered == original:
        return False
    path.write_text(rendered, encoding="utf-8")
    return True


def _migrate_dir(
    path: Path,
    *,
    rename_release_coupling_to_observed: bool,
    rename_release_coupling_to_historical: bool,
) -> list[Path]:
    changed: list[Path] = []
    for name in ("summary.json", "train_metrics.jsonl"):
        target = path / name
        if not target.exists():
            continue
        updated = (
            _rewrite_json(
                target,
                rename_release_coupling_to_observed=rename_release_coupling_to_observed,
                rename_release_coupling_to_historical=rename_release_coupling_to_historical,
            )
            if target.suffix == ".json"
            else _rewrite_jsonl(
                target,
                rename_release_coupling_to_observed=rename_release_coupling_to_observed,
                rename_release_coupling_to_historical=rename_release_coupling_to_historical,
            )
        )
        if updated:
            changed.append(target)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate Step 2.1 run artifacts to cleaned telemetry semantics.")
    parser.add_argument("run_dirs", nargs="+", help="Run output directories to migrate")
    parser.add_argument(
        "--rename-release-coupling-to-observed",
        action="store_true",
        help="Rename release coupling metric to *_observed_mean for runs where it was observation-only.",
    )
    parser.add_argument(
        "--rename-release-coupling-to-historical",
        action="store_true",
        help="Rename release coupling metric to *_historical_mean for runs with legacy historical semantics.",
    )
    args = parser.parse_args()

    total_changed: list[Path] = []
    for run_dir in args.run_dirs:
        changed = _migrate_dir(
            Path(run_dir),
            rename_release_coupling_to_observed=args.rename_release_coupling_to_observed,
            rename_release_coupling_to_historical=args.rename_release_coupling_to_historical,
        )
        total_changed.extend(changed)

    for path in total_changed:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
