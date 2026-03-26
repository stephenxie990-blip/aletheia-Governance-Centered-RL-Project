from __future__ import annotations

import copy
from typing import Any, Dict, Optional


def serialize_effective_training_config(
    effective_training_config: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Serialize the effective training config into checkpoint-safe form."""
    if effective_training_config is None:
        return None
    if hasattr(effective_training_config, "to_dict"):
        return copy.deepcopy(effective_training_config.to_dict())
    if isinstance(effective_training_config, dict):
        return copy.deepcopy(effective_training_config)
    raise TypeError(
        "effective training config must be a TrainingConfig-like object or mapping"
    )
