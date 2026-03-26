import json
from pathlib import Path

from aletheia._run_artifact_schema import RunArtifactPaths


def effective_training_config_of(payload):
    return payload["effective_training_config"]


def agent_bootstrap_bundle_of(payload):
    return payload["agent_bootstrap_bundle"]


def training_state_payload_of(payload):
    return payload["training_state"]


def optimizer_bundle_payload_of(payload):
    return payload["optimizer_bundle"]


def replay_buffer_payload_of(payload):
    return payload["replay_buffer"]


def run_artifact_paths_for(root: str | Path) -> RunArtifactPaths:
    return RunArtifactPaths.from_save_arg(str(root))


def load_summary_payload(root: str | Path):
    paths = run_artifact_paths_for(root)
    return json.loads(paths.summary_path.read_text(encoding="utf-8"))


def load_resolved_config_payload(root: str | Path):
    paths = run_artifact_paths_for(root)
    return json.loads(paths.config_path.read_text(encoding="utf-8"))


def make_minimal_agent_checkpoint(
    *,
    agent_bootstrap_bundle=None,
    **extra,
):
    checkpoint = {
        "world_model": {},
        "actor": {},
        "critic": {},
    }
    if agent_bootstrap_bundle is not None:
        checkpoint["agent_bootstrap_bundle"] = agent_bootstrap_bundle
    checkpoint.update(extra)
    return checkpoint
