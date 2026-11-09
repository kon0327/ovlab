"""Portable experiment and node locations shared by run and export owners."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Mapping


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _segment(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{label} must be a portable filesystem identifier")
    return value


@dataclass(frozen=True, slots=True)
class ExperimentScope:
    experiment_id: str
    node_parts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _segment(self.experiment_id, "experiment ID")
        if self.experiment_id in {"train", "evaluate", "derived", "interactive", "isolated", "grouped"}:
            raise ValueError("experiment ID conflicts with a legacy run category")
        for part in self.node_parts:
            _segment(part, "experiment node ID")
            if part in {"train", "evaluate", "derived", "interactive", "isolated", "grouped"}:
                raise ValueError("experiment node ID conflicts with an artifact directory name")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> ExperimentScope | None:
        values = os.environ if environment is None else environment
        experiment_id = values.get("OVLAB_EXPERIMENT_ID", "")
        node = values.get("OVLAB_EXPERIMENT_NODE", "")
        if not experiment_id:
            if node:
                raise ValueError("OVLAB_EXPERIMENT_NODE requires OVLAB_EXPERIMENT_ID")
            return None
        if node.startswith("/") or node.endswith("/") or "//" in node:
            raise ValueError("OVLAB_EXPERIMENT_NODE must contain slash-separated node IDs")
        return cls(experiment_id, tuple(node.split("/")) if node else ())

    @property
    def node_path(self) -> str:
        return "/".join(self.node_parts)

    @classmethod
    def from_run_path(cls, runs_root: str | Path, run_path: str | Path, category: str) -> ExperimentScope | None:
        """Recover scope from a canonical run; legacy typed/flat paths return None."""
        if category not in {"train", "evaluate"}:
            raise ValueError("run category must be train or evaluate")
        root = Path(runs_root).expanduser().resolve()
        path = Path(run_path).expanduser().resolve()
        if not path.is_relative_to(root):
            raise ValueError("run path escapes its configured root")
        parts = path.relative_to(root).parts
        if len(parts) >= 3 and parts[0] not in {"train", "evaluate"} and parts[1] == category:
            return cls(parts[0], parts[2:-1])
        return None

    def runs_root(self, base: str | Path, category: str) -> Path:
        if category not in {"train", "evaluate"}:
            raise ValueError("run category must be train or evaluate")
        return Path(base) / self.experiment_id / category / Path(*self.node_parts)

    def exports_root(self, base: str | Path, category: str) -> Path:
        if category not in {"train", "evaluate"}:
            raise ValueError("export category must be train or evaluate")
        return Path(base) / self.experiment_id / category / Path(*self.node_parts)

