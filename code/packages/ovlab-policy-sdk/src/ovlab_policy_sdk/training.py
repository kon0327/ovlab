"""Optional training and checkpoint preparation extensions for plugin API v1.

These interfaces receive generic OVLAB documents and filesystem locations. A
plugin owns all model-specific schemas, loader names and runtime dependencies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Protocol

from .plugin import PolicyPluginCompatibilityError


class TrainingPlanningServices(Protocol):
    """Host resource callbacks available to model-specific plan compilers."""
    repository_root: Path
    model_data_root: Path
    def resolve_model(self, profile, **options) -> Mapping: ...
    def resolve_dataset(self, profile, **options) -> Mapping: ...
    def estimate_vram(self, profile) -> float: ...


class TrainingPlugin(ABC):
    """Stateless host hooks; create_trainer returns a fresh stateful trainer."""

    api_version = "1"

    @abstractmethod
    def validate_profile(self, document: Mapping) -> Mapping:
        """Return a normalized profile or reject invalid model/trainer settings."""

    def validate_profile_template(self, document: Mapping) -> Mapping:
        """Validate composable defaults; implementations define allowed omissions.

        The default remains strict. Template validation never authorizes model
        execution, which always requires validate_profile on the final document.
        """
        return self.validate_profile(document)

    @abstractmethod
    def estimated_vram(self, profile) -> float:
        """Return a documented planning estimate in GiB."""

    @abstractmethod
    def plan(self, services, profile, **options) -> dict:
        """Build scientific/execution plans using generic resource services."""

    @abstractmethod
    def create_trainer(self):
        """Return a new trainer implementing the OVLAB training lifecycle."""

    @abstractmethod
    def validate_staging(self, context) -> tuple[dict | None, list[dict]]:
        """Validate plugin-specific staged tensors before immutable publication."""

    @abstractmethod
    def expected_loader(self, scientific: Mapping) -> str:
        """Identify the registered loader required by this finalized artifact."""

    @abstractmethod
    def verify_checkpoint(self, root: Path, checkpoint: Mapping) -> None:
        """Verify model-specific artifact semantics after generic file hashes."""

    def checkpoint_format(self, scientific: Mapping) -> str:
        """Describe staged weight serialization in the immutable checkpoint manifest."""
        return "safetensors"

    def container_resources(self, plan: Mapping, dataset: Mapping) -> dict:
        """Declare read-only dataset layout and plugin-specific worker variables."""
        return {"environment": {"OVLAB_TRAINING_DATA_ROOT": "/datasets/resolved"},
                "mounts": [{"source": str(Path(dataset["host_path"]) / "prepared"),
                            "target": "/datasets/resolved"}]}

    def validate_ablation(self, profile: Mapping, factors: list[str]) -> None:
        """Validate model-owned experiment axes before generic suite orchestration."""
        raise PolicyPluginCompatibilityError("this trainer does not support ablation suites")

    def validate_checkpoint_binding(self, training: Mapping, policy: Mapping):
        raise PolicyPluginCompatibilityError("this trainer does not support evaluation checkpoint binding")

    def bind_training_checkpoint(self, *args, **kwargs):
        raise PolicyPluginCompatibilityError("this trainer does not support evaluation checkpoint binding")

    def prepare_dense_checkpoint(self, *args, **kwargs):
        raise PolicyPluginCompatibilityError("this trainer does not support dense checkpoint preparation")


class CheckpointPreparationPlugin(ABC):
    """Optional offline conversion hook, run inside a compatible plugin image."""

    api_version = "1"

    @abstractmethod
    def prepare(self, base: Path, checkpoint: Path, output: Path, *, dtype: str) -> None:
        """Publish inference weights without modifying base/training artifacts."""
