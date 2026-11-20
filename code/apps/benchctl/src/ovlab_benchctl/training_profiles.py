"""Strict versioned training profiles and deterministic resolved plans."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import math
from pathlib import Path
from typing import Mapping

from .checkpointing import checkpoint_spec_by_id, verify_checkpoint
from .datasets import DatasetRequest, DatasetStore
from .training_errors import (
    DatasetUnavailableError,
    TrainingPlanError,
    TrainingProfileError,
    TrainingResourceError,
)
from .training_identity import canonical_json, identity
from .policy_plugins import with_registry


TRAINING_PROFILE_SCHEMA = "ovlab.training-profile/v1"
TRAINING_PROFILE_KIND = "training"
TRAINING_PLAN_SCHEMA = "ovlab.training-plan/v1"


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TrainingProfileError(f"{label} must be a mapping")
    return dict(value)


def _keys(document: Mapping[str, object], allowed: set[str], required: set[str], label: str) -> None:
    unknown = sorted(set(document) - allowed)
    missing = sorted(required - set(document))
    if unknown:
        raise TrainingProfileError(f"{label} contains unknown fields: {', '.join(unknown)}")
    if missing:
        raise TrainingProfileError(f"{label} is missing required fields: {', '.join(missing)}")


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise TrainingProfileError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: object, label: str, minimum: float, maximum: float, *, lower_open=False) -> float:
    if type(value) not in {int, float}:
        raise TrainingProfileError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= minimum if lower_open else result < minimum) or result > maximum:
        bracket = "(" if lower_open else "["
        raise TrainingProfileError(f"{label} must be in {bracket}{minimum}, {maximum}]")
    return result


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    profile_id: str
    document: Mapping[str, object]

    @classmethod
    def from_document(cls, value: object) -> "TrainingProfile":
        document = _mapping(value, "training profile")
        for key in ("schema_version", "kind", "id", "trainer"):
            if key not in document:
                raise TrainingProfileError(f"training profile is missing required fields: {key}")
        if document["schema_version"] != TRAINING_PROFILE_SCHEMA:
            raise TrainingProfileError(f"training profile schema_version must be {TRAINING_PROFILE_SCHEMA}")
        if document["kind"] != TRAINING_PROFILE_KIND:
            raise TrainingProfileError(f"training profile kind must be {TRAINING_PROFILE_KIND}")
        profile_id = document["id"]
        if not isinstance(profile_id, str) or not profile_id or "/" in profile_id or ".." in profile_id:
            raise TrainingProfileError("training profile id must be a portable non-empty identifier")
        from .training_plugins import backend_for_profile
        normalized = backend_for_profile(document).validate_profile(document)
        return cls(profile_id=profile_id, document=json.loads(canonical_json(normalized)))

    def as_dict(self) -> dict[str, object]:
        return dict(self.document)


def _checkpoint_candidates(model_data_root: Path, repo_id: str, resource_id: str, revision: str) -> tuple[tuple[str, Path], ...]:
    from .checkpointing import default_global_cache
    global_root = Path(os.environ.get("OVLAB_GLOBAL_HF_CACHE", default_global_cache(os.environ))).expanduser().resolve()
    hub = global_root / "hub" if (global_root / "hub").is_dir() else global_root
    global_snapshot = hub / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
    managed = model_data_root / "checkpoints" / "huggingface" / resource_id / revision
    return (("global-huggingface-cache", global_snapshot), ("ovlab-managed-cache", managed))


def with_fitted_action_normalization(profile: TrainingProfile, enabled: bool) -> TrainingProfile:
    """Resolve the CLI override into scientific configuration without mutation."""
    if not enabled:
        return profile
    from copy import deepcopy
    document = deepcopy(dict(profile.document))
    partition = document["dataset"].get("partition")
    if not isinstance(partition, dict):
        raise TrainingProfileError("--fit-action-normalization requires dataset.partition; validation episodes must be excluded")
    partition["action_normalization"] = "train-q99"
    return TrainingProfile.from_document(document)


class TrainingPlanner:
    """Dependency-light planner: no network, model import, or GPU allocation."""

    def __init__(self, repository_root: Path, model_data_root: Path):
        self.repository_root = repository_root.resolve()
        self.model_data_root = model_data_root.expanduser().resolve()
        self.datasets = DatasetStore(self.model_data_root)

    def _model(self, profile: TrainingProfile, *, allow_checkpoint_download: bool = False) -> dict[str, object]:
        model = profile.document["model"]
        assert isinstance(model, dict)
        resource_id = str(model["base_checkpoint"])
        spec = checkpoint_spec_by_id(self.repository_root, resource_id)
        selector = model.get("revision", "local/main")
        if selector in {"local/main", "remote/main", "latest"}:
            from .checkpointing import CheckpointResolver, default_global_cache
            global_cache = Path(os.environ.get("OVLAB_GLOBAL_HF_CACHE", default_global_cache(os.environ))).expanduser().resolve()
            resolved = CheckpointResolver(
                global_cache=global_cache,
                managed_cache=self.model_data_root / "checkpoints" / "huggingface",
            ).resolve_latest(resource_id, spec.repo_id, mode="remote/main" if selector == "latest" else selector,
                             allow_download=allow_checkpoint_download)
            return {
                "resource_id": resource_id,
                "repository": spec.repo_id,
                "revision": resolved.spec.revision,
                "aggregate_sha256": resolved.spec.expected_sha256,
                "verified_file_count": resolved.verified_file_count,
                "source_kind": resolved.source_kind,
                "host_path": str(resolved.host_path),
                "revision_selection": selector,
            }
        from .checkpointing import LatestCheckpointSelection, checkpoint_spec_from_snapshot
        if isinstance(spec, LatestCheckpointSelection):
            for source_kind, candidate in _checkpoint_candidates(self.model_data_root, spec.repo_id, resource_id, selector):
                if candidate.is_dir():
                    spec = checkpoint_spec_from_snapshot(resource_id, spec.repo_id, selector, candidate)
                    break
            else:
                raise TrainingPlanError(f"base checkpoint {resource_id}@{selector} is unavailable locally")
        if model["revision"] != spec.revision:
            raise TrainingPlanError(
                f"profile model revision {model['revision']} does not match registry revision {spec.revision}"
            )
        for source_kind, candidate in _checkpoint_candidates(self.model_data_root, spec.repo_id, resource_id, spec.revision):
            if candidate.is_dir():
                count = verify_checkpoint(spec, candidate)
                return {
                    "resource_id": resource_id,
                    "repository": spec.repo_id,
                    "revision": spec.revision,
                    "aggregate_sha256": spec.expected_sha256,
                    "verified_file_count": count,
                    "source_kind": source_kind,
                    "host_path": str(candidate.resolve()),
                    "revision_selection": "pinned",
                }
        raise TrainingPlanError(
            f"base checkpoint {resource_id!r} is unavailable locally; training planning never downloads checkpoints"
        )

    def _dataset(self, profile: TrainingProfile, *, allow_dataset_download: bool) -> dict[str, object]:
        dataset = profile.document["dataset"]
        assert isinstance(dataset, dict)
        reference = str(dataset["ref"])
        if reference.startswith("dataset-"):
            return self.datasets.inspect(reference)
        try:
            provider, name = reference.split("/", 1)
        except ValueError as exc:
            raise TrainingPlanError("dataset.ref must be an immutable dataset ID or provider/name selector") from exc
        resolution = self.datasets.resolve(DatasetRequest(source=provider, name=name))
        ready = self.datasets.find_resolution(resolution.resolution_id)
        if ready is None:
            suffix = "; run an explicit dataset fetch first" if allow_dataset_download else "; pass --allow-dataset-download only to train run or fetch explicitly"
            raise DatasetUnavailableError(f"dataset selector {reference!r} is not locally ready{suffix}")
        self.datasets.verify(str(ready["dataset_id"]))
        return ready

    @staticmethod
    def _estimated_vram(profile: TrainingProfile) -> float:
        from .training_plugins import backend_for_profile
        return backend_for_profile(profile.document).estimated_vram(profile)

    def resolve_model(self, profile, **options):
        return self._model(profile, **options)

    def resolve_dataset(self, profile, **options):
        return self._dataset(profile, **options)

    def estimate_vram(self, profile):
        return self._estimated_vram(profile)

    @with_registry
    def plan(self, profile: TrainingProfile, **options) -> dict[str, object]:
        from .training_plugins import backend_for_profile
        return backend_for_profile(profile.document, self.repository_root).plan(self, profile, **options)
