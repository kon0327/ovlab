"""Dependency-light handling of optional local OpenVLA statistics sidecars."""

import json
from pathlib import Path
from typing import Any, Union


def load_local_dataset_statistics(model: Any, openvla_path: Union[str, Path]) -> bool:
    """Apply a local training-output sidecar without requiring it for HF snapshots."""
    stats_path = Path(openvla_path) / "dataset_statistics.json"
    if not stats_path.is_file():
        return False
    with stats_path.open("r", encoding="utf-8") as stream:
        model.norm_stats = json.load(stream)
    return True
