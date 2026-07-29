"""Read-only inspection and integrity validation for finalized filesystem runs."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re

from .artifacts.codec import TraceCodec
from .errors import ArtifactError


class RunIntegrityError(ArtifactError):
    """A stored run is missing, inconsistent, or no longer immutable."""


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunIntegrityError(f"invalid JSON artifact: {path.name}") from exc


def _run_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RunIntegrityError(f"run path is not a directory: {path}")
    return path


def _resolved_hashes(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunIntegrityError("cannot read resolved_config.yaml") from exc
    result = {}
    for key in ("scientific_config_hash", "execution_config_hash"):
        match = re.search(rf"(?m)^{key}: ['\"]?([0-9a-f]{{64}})['\"]?\s*$", text)
        if match is None:
            raise RunIntegrityError(f"resolved configuration lacks a valid {key}")
        result[key] = match.group(1)
    return result


def inspect_run(value: str | Path) -> dict[str, object]:
    """Summarize an existing run without changing it."""
    path = _run_path(value)
    started_path = path / "manifest.started.json"
    if not started_path.is_file():
        raise RunIntegrityError("required artifact is missing: manifest.started.json")
    started = _json(started_path)
    finals = [item for item in (path / "manifest.completed.json", path / "manifest.failed.json") if item.is_file()]
    if len(finals) > 1:
        raise RunIntegrityError("run contains conflicting final manifests")
    final = _json(finals[0]) if finals else None
    episode_paths = sorted(path.glob("tasks/*/episodes/*"))
    metric_files = sorted(path.glob("tasks/*/episodes/*/metrics.episode.json"))
    statuses: dict[str, int] = {}
    for metric_path in metric_files:
        for result in _json(metric_path):
            status = result.get("status", "invalid")
            statuses[status] = statuses.get(status, 0) + 1
    connection = _json(path / "connection.json") if (path / "connection.json").is_file() else {}
    tasks = sorted(path.glob("tasks/*"))
    schema_versions = set()
    for episode in episode_paths:
        trace_path = episode / "trace.json"
        if trace_path.is_file():
            schema_versions.add(_json(trace_path).get("schema_version"))
    return {
        "run_id": started.get("run_id"),
        "status": "running" if final is None else final.get("status"),
        "policy": connection.get("policy"),
        "benchmark": connection.get("benchmark"),
        "scientific_config_hash": started.get("scientific_config_hash"),
        "execution_config_hash": started.get("execution_config_hash"),
        "task_count": len(tasks),
        "rollout_count": len(episode_paths),
        "metric_availability": dict(sorted(statuses.items())),
        "failure_type": None if final is None else final.get("failure_type"),
        "trace_schema_version": next(iter(schema_versions)) if len(schema_versions) == 1 else None,
        "trace_schema_versions": sorted(item for item in schema_versions if item is not None),
        "artifact_paths": {
            "run": ".",
            "started_manifest": "manifest.started.json",
            "final_manifest": None if not finals else finals[0].name,
            "source_config": "source_config.yaml" if (path / "source_config.yaml").is_file() else None,
            "resolved_config": "resolved_config.yaml" if (path / "resolved_config.yaml").is_file() else None,
            "plan": "plan.json" if (path / "plan.json").is_file() else None,
            "connection": "connection.json" if (path / "connection.json").is_file() else None,
        },
    }


def verify_run(value: str | Path) -> dict[str, object]:
    """Validate stored hashes, schemas, finalization markers, and consistency."""
    path = _run_path(value)
    required = (
        "manifest.started.json", "source_config.yaml", "resolved_config.yaml", "plan.json", "connection.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RunIntegrityError(f"required artifacts are missing: {', '.join(missing)}")
    summary = inspect_run(path)
    started = _json(path / "manifest.started.json")
    resolved = _resolved_hashes(path / "resolved_config.yaml")
    for key in ("scientific_config_hash", "execution_config_hash"):
        if not isinstance(resolved, dict) or resolved.get(key) != started.get(key):
            raise RunIntegrityError(f"{key} is inconsistent between manifest and resolved configuration")
    completed = path / "manifest.completed.json"
    failed = path / "manifest.failed.json"
    if completed.is_file():
        final = _json(completed)
        if final.get("status") != "completed" or final.get("failure_type") is not None:
            raise RunIntegrityError("completed manifest is inconsistent with recorded failure state")
    elif failed.is_file():
        final = _json(failed)
        if final.get("status") not in {"failed", "aborted", "interrupted"}:
            raise RunIntegrityError("failed manifest has an invalid run status")
    else:
        raise RunIntegrityError("run has no final manifest")
    integrity_path = path / "integrity.json"
    if not integrity_path.is_file():
        raise RunIntegrityError("required artifact is missing: integrity.json")
    integrity = _json(integrity_path)
    if integrity.get("schema_version") != "ovlab-integrity/1.0.0" or integrity.get("algorithm") != "sha256":
        raise RunIntegrityError("integrity manifest has an unsupported schema or algorithm")
    expected = {row.get("path"): row for row in integrity.get("files", [])}
    actual = {
        str(item.relative_to(path)): item
        for item in path.rglob("*")
        if item.is_file() and item != integrity_path and not item.name.endswith(".tmp")
    }
    if set(expected) != set(actual):
        raise RunIntegrityError("integrity inventory differs from stored run files")
    for relative, item in sorted(actual.items()):
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        row = expected[relative]
        if row.get("size_bytes") != item.stat().st_size or row.get("sha256") != digest:
            raise RunIntegrityError(f"integrity checksum mismatch: {relative}")
    codec = TraceCodec()
    episode_paths = sorted(path.glob("tasks/*/episodes/*"))
    for episode in episode_paths:
        if not (episode / "trace.finalized.json").is_file():
            raise RunIntegrityError(f"episode trace is not finalized: {episode.relative_to(path)}")
        if not (episode / "finalized.json").is_file():
            raise RunIntegrityError(f"episode metrics are not finalized: {episode.relative_to(path)}")
        try:
            codec.decode(episode)
        except Exception as exc:
            raise RunIntegrityError(f"trace validation failed: {episode.relative_to(path)}: {exc}") from exc
    if final.get("episode_count") != len(episode_paths):
        raise RunIntegrityError("final manifest episode_count differs from stored traces")
    return {
        **summary,
        "integrity": "verified",
        "verified_episode_count": len(episode_paths),
        "verified_file_count": len(actual),
    }
