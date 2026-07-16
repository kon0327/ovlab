"""Atomic local filesystem artifact store."""

import json
from pathlib import Path
import shutil

from ovlab_core.contracts import EpisodeId, RunId, TaskId
from ovlab_metrics import MetricResult, MetricScope, MetricStatus

from ..errors import ArtifactError
from .base import RunArtifactStore
from .codec import TraceCodec, _plain
from .layout import safe_key


class FilesystemRunArtifactStore(RunArtifactStore):
    def __init__(self, root):
        self.root = Path(root)
        self.codec = TraceCodec()

    def _run_path(self, run_id): return self.root / safe_key(str(run_id))
    def _task_path(self, run_id, task_id): return self._run_path(run_id) / "tasks" / safe_key(str(task_id))
    def _episode_path(self, run_id, task_id, episode_id): return self._task_path(run_id, task_id) / "episodes" / safe_key(str(episode_id))

    def create_run(self, run_id, started_manifest):
        path = self._run_path(run_id)
        if path.exists(): raise ArtifactError("run artifact already exists")
        path.mkdir(parents=True)
        self._atomic_json(path / "manifest.started.json", started_manifest)

    def write_plan(self, run_id, plan): self._atomic_json(self._run_path(run_id) / "plan.json", plan.canonical())
    def write_connection_report(self, run_id, report): self._atomic_json(self._run_path(run_id) / "connection.json", _connection(report))

    def write_episode_trace(self, run_id, trace):
        target = self._episode_path(run_id, trace.episode_context.task_id, trace.episode_context.episode_id)
        if target.exists(): raise ArtifactError("finalized episode artifact already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        if temporary.exists(): raise ArtifactError("partial episode artifact already exists")
        try:
            self.codec.encode(trace, temporary)
            self._atomic_json(temporary / "trace.finalized.json", {"raw_trace_finalized": True})
            temporary.rename(target)
        except Exception:
            raise

    def write_episode_metric_results(self, run_id, task_id, episode_id, results):
        path = self._episode_path(run_id, task_id, episode_id)
        if not (path / "trace.finalized.json").is_file(): raise ArtifactError("raw trace must be finalized before metrics")
        self._atomic_json(path / "metrics.episode.json", [_metric(result) for result in results])
        self._atomic_json(path / "finalized.json", {"episode_finalized": True})

    def write_task_metric_results(self, run_id, task_id, results):
        path = self._task_path(run_id, task_id)
        path.mkdir(parents=True, exist_ok=True)
        self._atomic_json(path / "metrics.task.json", [_metric(result) for result in results])

    def finalize_run(self, run_id, manifest): self._final_manifest(run_id, "manifest.completed.json", manifest)
    def mark_run_failed(self, run_id, manifest): self._final_manifest(run_id, "manifest.failed.json", manifest)

    def _final_manifest(self, run_id, name, manifest):
        path = self._run_path(run_id)
        if (path / "manifest.completed.json").exists() or (path / "manifest.failed.json").exists():
            raise ArtifactError("run already finalized")
        self._atomic_json(path / name, manifest)

    def read_episode_trace(self, run_id, task_id, episode_id):
        path = self._episode_path(run_id, task_id, episode_id)
        if not (path / "trace.finalized.json").is_file(): raise ArtifactError("episode raw trace is partial or missing")
        return self.codec.decode(path)

    def read_metric_results(self, run_id, task_id, episode_id=None):
        path = self._task_path(run_id, task_id) / "metrics.task.json" if episode_id is None else self._episode_path(run_id, task_id, episode_id) / "metrics.episode.json"
        if not path.is_file(): return ()
        return tuple(_decode_metric(item) for item in json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _atomic_json(path, value):
        if path.exists(): raise ArtifactError(f"finalized artifact already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)


def _metric(result):
    return {"metric_id": result.metric_id, "metric_version": result.metric_version, "scope": result.scope.value, "status": result.status.value, "value": _json_value(result.value), "unit": result.unit, "sample_count": result.sample_count, "run_id": str(result.run_id), "task_id": str(result.task_id), "episode_id": None if result.episode_id is None else str(result.episode_id), "reason": result.reason, "diagnostics": _json_value(result.diagnostics), "metric_config": _json_value(result.metric_config), "metric_config_hash": result.metric_config_hash, "metadata": _json_value(result.metadata)}


def _decode_metric(value):
    return MetricResult(value["metric_id"], value["metric_version"], MetricScope(value["scope"]), MetricStatus(value["status"]), value["value"], value["unit"], value["sample_count"], RunId(value["run_id"]), TaskId(value["task_id"]), None if value["episode_id"] is None else EpisodeId(value["episode_id"]), value["reason"], value["diagnostics"], value["metric_config"], value["metric_config_hash"], value["metadata"])


def _connection(report):
    return {"benchmark": {"name": report.benchmark_name, "version": report.benchmark_version}, "policy": {"name": report.policy_name, "version": report.policy_version}, "contract_version": report.contract_version, "compatible": report.compatibility_report.compatible, "compatibility_issues": [{"code": issue.code, "severity": issue.severity.value, "path": issue.path, "message": issue.message} for issue in report.compatibility_report.issues], "selected_task_ids": [str(task.task_id) for task in report.selected_tasks], "enabled_metrics": [{"metric_id": item.metric_id, "metric_version": item.metric_version} for item in report.enabled_metric_descriptors], "required_metrics": list(report.required_metric_ids), "statically_available_metrics": list(report.statically_available_metric_ids), "potentially_unavailable_metrics": list(report.potentially_unavailable_metric_ids), "plan_hash": report.plan_hash, "metadata": dict(report.metadata)}


def _json_value(value):
    import numpy as np
    if isinstance(value, np.ndarray): return value.tolist()
    if hasattr(value, "items"): return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_json_value(item) for item in value]
    return value
