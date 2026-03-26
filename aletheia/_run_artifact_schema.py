from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch

from .training.compensation import summarize_compensation_restore_report


@dataclass(frozen=True)
class RunArtifactPaths:
    save_root: Optional[Path]
    save_dir: Optional[Path]
    final_ckpt_path: Optional[Path]
    periodic_prefix: str
    eval_history_path: Optional[Path]
    train_metrics_path: Optional[Path]
    config_path: Optional[Path]
    summary_path: Optional[Path]
    best_trainer_path: Optional[Path]
    best_ckpt_path: Optional[Path]
    resume_latest_path: Optional[Path]
    trainer_state_final_path: Optional[Path]

    @classmethod
    def from_save_arg(cls, save_arg: Optional[str]) -> "RunArtifactPaths":
        if not save_arg:
            return cls(
                save_root=None,
                save_dir=None,
                final_ckpt_path=None,
                periodic_prefix="checkpoint",
                eval_history_path=None,
                train_metrics_path=None,
                config_path=None,
                summary_path=None,
                best_trainer_path=None,
                best_ckpt_path=None,
                resume_latest_path=None,
                trainer_state_final_path=None,
            )

        save_root = Path(save_arg).expanduser()
        periodic_prefix = "checkpoint"
        if save_root.suffix:
            save_dir = save_root.parent
            final_ckpt_path = save_root
            periodic_prefix = save_root.stem
        else:
            save_dir = save_root
            final_ckpt_path = save_dir / "final.pt"

        return cls(
            save_root=save_root,
            save_dir=save_dir,
            final_ckpt_path=final_ckpt_path,
            periodic_prefix=periodic_prefix,
            eval_history_path=save_dir / "eval_history.jsonl",
            train_metrics_path=save_dir / "train_metrics.jsonl",
            config_path=save_dir / "config_resolved.json",
            summary_path=save_dir / "summary.json",
            best_trainer_path=save_dir / "trainer_state_best.pt",
            best_ckpt_path=save_dir / "best.pt",
            resume_latest_path=save_dir / "resume_latest.pt",
            trainer_state_final_path=save_dir / "trainer_state_final.pt",
        )

    def periodic_checkpoint_path(self, step: int) -> Optional[Path]:
        if self.save_dir is None:
            return None
        return self.save_dir / f"{self.periodic_prefix}_step{int(step)}.pt"

    def periodic_trainer_state_path(self, step: int) -> Optional[Path]:
        if self.save_dir is None:
            return None
        return self.save_dir / f"trainer_state_step{int(step)}.pt"


def build_resolved_config_payload(
    *,
    saved_at_utc: str,
    project_root: Path,
    git_commit: Optional[str],
    argv: List[str],
    args: Mapping[str, Any],
    overrides_effective: Mapping[str, Any],
    effective_training_config: Mapping[str, Any],
    agent_bootstrap_bundle: Mapping[str, Any],
    profile: Optional[Any] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "saved_at_utc": saved_at_utc,
        "project_root": str(project_root),
        "git_commit": git_commit,
        "argv": list(argv),
        "args": dict(args),
        "overrides_effective": dict(overrides_effective),
        "effective_training_config": dict(effective_training_config),
        "agent_bootstrap_bundle": dict(agent_bootstrap_bundle),
    }
    if profile is not None:
        payload["env_expected_dims"] = {
            "vitals_dim": int(getattr(profile, "vitals_dim", 0)),
            "action_dim": int(getattr(profile, "action_dim", 0)),
            "action_embed_dim": int(getattr(profile, "action_embed_dim", 0)),
        }
    return payload


def _jsonify_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify_metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_metric_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonify_metric_value(value.detach().cpu().item())
        return _jsonify_metric_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonify_metric_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_train_metric_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    metrics_payload = normalized.get("metrics", {}) or {}
    step_value = int(normalized.get("step", 0))
    update_step = int(
        normalized.get(
            "update_step",
            metrics_payload.get(
                "train/update_step",
                metrics_payload.get("train/global_step", step_value),
            ),
        )
    )
    normalized["step"] = step_value
    normalized["update_step"] = update_step
    normalized["global_step"] = int(
        normalized.get("global_step", metrics_payload.get("train/global_step", update_step))
    )
    if "train/env_steps_collected" in metrics_payload and "env_steps_collected" not in normalized:
        normalized["env_steps_collected"] = int(metrics_payload["train/env_steps_collected"])
    if "train/batch_source" in metrics_payload and "batch_source" not in normalized:
        normalized["batch_source"] = str(metrics_payload["train/batch_source"])
    if "train/imag_ratio" in metrics_payload and "imag_ratio" not in normalized:
        normalized["imag_ratio"] = float(metrics_payload["train/imag_ratio"])
    if (
        "train/runtime_compensation_phase" in metrics_payload
        and "runtime_compensation_phase" not in normalized
    ):
        normalized["runtime_compensation_phase"] = str(
            metrics_payload["train/runtime_compensation_phase"]
        )
    if (
        "train/compensation_restore_status" in metrics_payload
        and "compensation_restore_status" not in normalized
    ):
        normalized["compensation_restore_status"] = str(
            metrics_payload["train/compensation_restore_status"]
        )
    if (
        "train/compensation_restore_degraded" in metrics_payload
        and "compensation_restore_degraded" not in normalized
    ):
        normalized["compensation_restore_degraded"] = float(
            metrics_payload["train/compensation_restore_degraded"]
        )
    if (
        "train/compensation_restore_post_solved_anchor_status" in metrics_payload
        and "compensation_restore_post_solved_anchor_status" not in normalized
    ):
        normalized["compensation_restore_post_solved_anchor_status"] = str(
            metrics_payload["train/compensation_restore_post_solved_anchor_status"]
        )
    if (
        "train/compensation_restore_behavior_policy_anchor_status" in metrics_payload
        and "compensation_restore_behavior_policy_anchor_status" not in normalized
    ):
        normalized["compensation_restore_behavior_policy_anchor_status"] = str(
            metrics_payload["train/compensation_restore_behavior_policy_anchor_status"]
        )
    if (
        "train/compensation_restore_real_stability_registry_status" in metrics_payload
        and "compensation_restore_real_stability_registry_status" not in normalized
    ):
        normalized["compensation_restore_real_stability_registry_status"] = str(
            metrics_payload["train/compensation_restore_real_stability_registry_status"]
        )
    if (
        "train/compensation_restore_real_stability_registry_restored_entries"
        in metrics_payload
        and "compensation_restore_real_stability_registry_restored_entries"
        not in normalized
    ):
        normalized["compensation_restore_real_stability_registry_restored_entries"] = int(
            metrics_payload[
                "train/compensation_restore_real_stability_registry_restored_entries"
            ]
        )
    if (
        "train/compensation_restore_real_stability_registry_skipped_entries"
        in metrics_payload
        and "compensation_restore_real_stability_registry_skipped_entries"
        not in normalized
    ):
        normalized["compensation_restore_real_stability_registry_skipped_entries"] = int(
            metrics_payload[
                "train/compensation_restore_real_stability_registry_skipped_entries"
            ]
        )
    return normalized


def build_train_metric_record(
    metrics: Mapping[str, Any],
    step: int,
    *,
    train_t0: float,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    step_int = int(step)
    update_step = int(metrics.get("train/update_step", metrics.get("train/global_step", step_int)))
    global_step = int(metrics.get("train/global_step", update_step))
    record: Dict[str, Any] = {
        "step": step_int,
        "update_step": update_step,
        "global_step": global_step,
        "elapsed_sec": float((time.time() if now is None else now) - train_t0),
        "metrics": _jsonify_metric_value(dict(metrics)),
    }
    if "train/env_steps_collected" in metrics:
        record["env_steps_collected"] = int(metrics["train/env_steps_collected"])
    if "train/batch_source" in metrics:
        record["batch_source"] = str(metrics["train/batch_source"])
    if "train/imag_ratio" in metrics:
        record["imag_ratio"] = float(metrics["train/imag_ratio"])
    if "train/runtime_compensation_phase" in metrics:
        record["runtime_compensation_phase"] = str(
            metrics["train/runtime_compensation_phase"]
        )
    if "train/compensation_restore_status" in metrics:
        record["compensation_restore_status"] = str(
            metrics["train/compensation_restore_status"]
        )
    if "train/compensation_restore_degraded" in metrics:
        record["compensation_restore_degraded"] = float(
            metrics["train/compensation_restore_degraded"]
        )
    if "train/compensation_restore_post_solved_anchor_status" in metrics:
        record["compensation_restore_post_solved_anchor_status"] = str(
            metrics["train/compensation_restore_post_solved_anchor_status"]
        )
    if "train/compensation_restore_behavior_policy_anchor_status" in metrics:
        record["compensation_restore_behavior_policy_anchor_status"] = str(
            metrics["train/compensation_restore_behavior_policy_anchor_status"]
        )
    if "train/compensation_restore_real_stability_registry_status" in metrics:
        record["compensation_restore_real_stability_registry_status"] = str(
            metrics["train/compensation_restore_real_stability_registry_status"]
        )
    if "train/compensation_restore_real_stability_registry_restored_entries" in metrics:
        record["compensation_restore_real_stability_registry_restored_entries"] = int(
            metrics["train/compensation_restore_real_stability_registry_restored_entries"]
        )
    if "train/compensation_restore_real_stability_registry_skipped_entries" in metrics:
        record["compensation_restore_real_stability_registry_skipped_entries"] = int(
            metrics["train/compensation_restore_real_stability_registry_skipped_entries"]
        )
    return normalize_train_metric_record(record)


def normalize_eval_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    step_value = int(normalized.get("step", 0))
    mean_reward = float(normalized.get("mean", normalized.get("mean_reward", 0.0)))
    normalized["step"] = step_value
    normalized["update_step"] = int(normalized.get("update_step", step_value))
    normalized["global_step"] = int(normalized.get("global_step", normalized["update_step"]))
    normalized["mean_reward"] = float(normalized.get("mean_reward", mean_reward))
    normalized["eval_reward"] = float(normalized.get("eval_reward", normalized["mean_reward"]))
    normalized["current_checkpoint_reward"] = float(
        normalized.get("current_checkpoint_reward", normalized["mean_reward"])
    )
    return normalized


def build_eval_record(
    *,
    results: Mapping[str, Any],
    step: int,
    telemetry: Optional[Mapping[str, Any]] = None,
    agent_step_count: Optional[Any] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "step": int(step),
        "update_step": int(step),
        "global_step": int(step),
        "mean": float(results["mean"]),
        "mean_reward": float(results["mean"]),
        "eval_reward": float(results["mean"]),
        "current_checkpoint_reward": float(results["mean"]),
        "std": float(results["std"]),
        "min": float(results["min"]),
        "max": float(results["max"]),
        "mean_length": float(results["mean_length"]),
    }
    if agent_step_count is not None:
        try:
            record["env_step_count"] = int(agent_step_count)
        except (TypeError, ValueError):
            pass
    for key, value in dict(telemetry or {}).items():
        record[str(key)] = float(value)
    return normalize_eval_record(record)


def append_jsonl_record(path_obj: Optional[Path], record: Mapping[str, Any]) -> None:
    if path_obj is None:
        return
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def read_jsonl_records(path_obj: Optional[Path]) -> List[Dict[str, Any]]:
    if path_obj is None or not path_obj.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path_obj.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def rewrite_jsonl_records(path_obj: Optional[Path], records: Iterable[Mapping[str, Any]]) -> None:
    if path_obj is None:
        return
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def trim_history_records_for_resume(
    path_obj: Optional[Path],
    *,
    resume_step: int,
    normalize_record: Callable[[Mapping[str, Any]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records = [
        normalize_record(record)
        for record in read_jsonl_records(path_obj)
        if int(record.get("step", -1)) <= int(resume_step)
    ]
    rewrite_jsonl_records(path_obj, records)
    return records


def build_artifact_manifest(
    paths: RunArtifactPaths,
    *,
    best_ckpt_exists: bool,
    alert_snapshots: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    return {
        "final_ckpt": str(paths.final_ckpt_path) if paths.final_ckpt_path is not None else None,
        "best_ckpt": str(paths.best_ckpt_path) if best_ckpt_exists and paths.best_ckpt_path is not None else None,
        "eval_history_jsonl": str(paths.eval_history_path) if paths.eval_history_path is not None else None,
        "train_metrics_jsonl": str(paths.train_metrics_path) if paths.train_metrics_path is not None else None,
        "resolved_config_json": str(paths.config_path) if paths.config_path is not None else None,
        "resume_latest": str(paths.resume_latest_path) if paths.resume_latest_path is not None else None,
        "trainer_state_final": str(paths.trainer_state_final_path) if paths.trainer_state_final_path is not None else None,
        "alert_snapshots": list(alert_snapshots or []),
    }


def build_summary_payload(
    *,
    saved_at_utc: str,
    env_id: Any,
    args: Any,
    seed: int,
    device: Any,
    collector_steps: int,
    train_steps_per_cycle: int,
    total_updates: int,
    expected_env_steps: int,
    effective_training_config: Mapping[str, Any],
    agent_bootstrap_bundle: Mapping[str, Any],
    paths: RunArtifactPaths,
    final_current: Mapping[str, Any],
    final_best: Optional[Mapping[str, Any]],
    final_model_source: str,
    best_eval_mean: float,
    best_eval_step: int,
    eval_history: List[Mapping[str, Any]],
    stats: Mapping[str, Any],
    latest_train_metric_record: Mapping[str, Any],
    bootstrap_contract_summary_keys: Iterable[str],
    loop: Any,
) -> Dict[str, Any]:
    latest_metrics_payload = dict(latest_train_metric_record.get("metrics", {}) or {})
    compensation_restore_summary = summarize_compensation_restore_report(
        getattr(loop, "_adaptive_compensation_restore_report", None)
    )
    bootstrap_contract_summary = {
        key: latest_metrics_payload.get(key)
        for key in bootstrap_contract_summary_keys
        if key in latest_metrics_payload
    }
    return {
        "saved_at_utc": saved_at_utc,
        "run": {
            "env": str(env_id),
            "steps_arg": int(getattr(args, "steps", 0)),
            "update_steps": int(total_updates),
            "expected_env_steps": int(expected_env_steps),
            "seed": int(seed),
            "device": str(device),
            "collect_steps_per_cycle": int(collector_steps),
            "train_steps_per_cycle": int(train_steps_per_cycle),
        },
        "effective_training_config": dict(effective_training_config),
        "agent_bootstrap_bundle": dict(agent_bootstrap_bundle),
        "artifacts": build_artifact_manifest(
            paths,
            best_ckpt_exists=bool(paths.best_ckpt_path is not None and paths.best_ckpt_path.exists()),
        ),
        "eval": {
            "final_current": final_current,
            "final_best": final_best,
            "final_model_source": final_model_source,
            "best_eval_mean_during_train": best_eval_mean if best_eval_mean > -float("inf") else None,
            "best_eval_step_during_train": int(best_eval_step),
            "num_eval_records": int(len(eval_history)),
        },
        "stats": dict(stats),
        "bootstrap_contract_last": {
            "step": latest_train_metric_record.get("step"),
            "update_step": latest_train_metric_record.get("update_step"),
            "global_step": latest_train_metric_record.get("global_step"),
            "metrics": bootstrap_contract_summary,
        },
        "bootstrap_external_eval_feedback": {
            "last_step": int(getattr(loop, "_bootstrap_external_eval_feedback_last_step", -1)),
            "last_mean": (
                float(getattr(loop, "_bootstrap_external_eval_feedback_last_mean"))
                if float(getattr(loop, "_bootstrap_external_eval_feedback_last_mean", -float("inf"))) > -float("inf")
                else None
            ),
            "best_mean": (
                float(getattr(loop, "_bootstrap_external_eval_feedback_best_mean"))
                if float(getattr(loop, "_bootstrap_external_eval_feedback_best_mean", -float("inf"))) > -float("inf")
                else None
            ),
            "telemetry": dict(getattr(loop, "_bootstrap_external_eval_feedback_telemetry", {})),
        },
        "compensation_restore": compensation_restore_summary,
    }
