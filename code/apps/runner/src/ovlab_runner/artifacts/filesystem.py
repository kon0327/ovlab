"""Atomic local filesystem artifact store."""

import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

from ovlab_core.contracts import EpisodeId, RunId, TaskId
from ovlab_metrics import MetricResult, MetricScope, MetricStatus

from ..errors import ArtifactError
from .base import RunArtifactStore
from .codec import TraceCodec, _plain
from .layout import run_key, safe_key


class FilesystemRunArtifactStore(RunArtifactStore):
    def __init__(self, root):
        self.root = Path(root)
        self.codec = TraceCodec()

    def _run_path(self, run_id): return self.root / run_key(str(run_id))
    def _task_path(self, run_id, task_id): return self._run_path(run_id) / "tasks" / safe_key(str(task_id))
    def _episode_path(self, run_id, task_id, episode_id): return self._task_path(run_id, task_id) / "episodes" / safe_key(str(episode_id))

    def create_run(self, run_id, started_manifest):
        path = self._run_path(run_id)
        if path.exists(): raise ArtifactError("run artifact already exists")
        path.mkdir(parents=True)
        self._atomic_json(path / "manifest.started.json", started_manifest)

    def write_configuration(self, run_id, snapshot):
        path = self._run_path(run_id)
        if (path / "source_config.yaml").exists() or (path / "resolved_config.yaml").exists():
            raise ArtifactError("run configuration already exists")
        self._atomic_text(path / "source_config.yaml", snapshot.portable_source_yaml)
        self._atomic_text(path / "resolved_config.yaml", snapshot.resolved_config_yaml)

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
        results = tuple(results)
        self._atomic_json(path / "metrics.episode.json", [_metric(result) for result in results])
        trace = self.codec.decode(path)
        self._atomic_json(
            path / "metadata.json",
            self._episode_metadata(run_id, trace, results),
        )
        self._atomic_json(path / "finalized.json", {"episode_finalized": True})

    def _episode_metadata(self, run_id, trace, results):
        run_path = self._run_path(run_id)
        started = _read_json(run_path / "manifest.started.json")
        plan = _read_json(run_path / "plan.json")
        context = trace.episode_context
        context_metadata = dict(context.metadata)
        environment = dict(context_metadata.get("environment", {}))
        reset = dict(trace.metadata.get("benchmark_reset", {}))
        plan_metadata = dict(plan.get("metadata", {}))
        deployment = dict(started.get("metadata", {}).get("deployment", {}))
        method = _method_descriptor(started)
        policy_configuration = dict(plan_metadata.get("policy_configuration", {}))
        success_metric = next(
            (result for result in results if result.metric_id == "task.success"),
            None,
        )
        success = trace.terminal_status.value == "success"
        metric_status = None if success_metric is None else success_metric.status.value
        objects = environment.get("objects", environment.get("object_names"))
        started_ns = trace.metadata.get("episode_started_wall_time_utc_ns")
        ended_ns = trace.metadata.get("episode_ended_wall_time_utc_ns")
        run_created_ns = plan.get("run_context", {}).get("created_wall_time_utc_ns")
        checkpoint_identity = dict(method.get("checkpoint_identity", {}))
        checkpoint = {
            "checkpoint_id": deployment.get(
                "checkpoint_id", policy_configuration.get("checkpoint_id")
            ),
            "repository": deployment.get(
                "checkpoint_repository", checkpoint_identity.get("repository")
            ),
            "revision": deployment.get(
                "checkpoint_revision", checkpoint_identity.get("revision")
            ),
            "sha256": deployment.get(
                "checkpoint_sha256", checkpoint_identity.get("aggregate_sha256")
            ),
            "normalization_key": policy_configuration.get("unnorm_key"),
        }
        return {
            "schema_version": "ovlab-episode-metadata/1.0.0",
            "identity": {
                "run_id": str(context.run_id),
                "task_id": str(context.task_id),
                "episode_id": str(context.episode_id),
                "rollout_index": context.rollout_index,
                "episode_seed": context.seed,
                "plan_hash": plan.get("metadata", {}).get("plan_hash", started.get("plan_hash")),
                "scientific_config_hash": started.get("scientific_config_hash"),
                "execution_config_hash": started.get("execution_config_hash"),
            },
            "experiment": {
                "id": plan_metadata.get(
                    "experiment_id", plan.get("run_context", {}).get("experiment_name")
                ),
                "name": plan_metadata.get(
                    "experiment_name", plan.get("run_context", {}).get("experiment_name")
                ),
                "tags": plan_metadata.get("experiment_tags", []),
            },
            "benchmark": {
                "name": _benchmark_family(context_metadata, started),
                "component": started.get("benchmark", {}).get("name"),
                "version": started.get("benchmark", {}).get("version"),
            },
            "scenario": {
                "suite": context_metadata.get("suite_name"),
                "task_name": context_metadata.get("task_name"),
                "task_index": context_metadata.get("task_index"),
                "task_id": str(context.task_id),
                "initial_state_index": reset.get("initial_state_index"),
            },
            "environment_description": {
                "availability": "available" if objects is not None else "partial",
                "objects": objects,
                "native_task_reference": environment.get(
                    "native_task_reference", reset.get("native_task_reference")
                ),
                "available_initial_state_count": environment.get(
                    "available_initial_state_count"
                ),
                "details": environment,
            },
            "mission": {
                "initial_instruction": context.initial_instruction.text,
                "instruction_id": str(context.initial_instruction.instruction_id),
                "source": context.initial_instruction.source.value,
            },
            "model": {
                "name": started.get("policy", {}).get("name"),
                "version": started.get("policy", {}).get("version"),
                "method": _method_summary(method),
                "configuration": policy_configuration,
            },
            "checkpoint": checkpoint,
            "execution": {
                "action_execution_mode": plan.get("action_execution_policy", {}).get("mode"),
                "maximum_episode_steps": trace.metadata.get("task_maximum_steps"),
                "renderer": (
                    started.get("runtime", {}).get("benchmark", {}).get("libero_renderer")
                ),
            },
            "datetime": {
                "run_created_utc": _iso_utc(run_created_ns),
                "episode_started_utc": _iso_utc(started_ns),
                "episode_ended_utc": _iso_utc(ended_ns),
            },
            "result": {
                "status": "OK" if success else "NOK",
                "success": success,
                "success_rate": 1.0 if success else 0.0,
                "success_metric_status": metric_status,
                "terminal_status": trace.terminal_status.value,
                "executed_step_count": len(trace.executed_actions),
                "prediction_count": len(trace.policy_predictions),
                "failure_type": trace.metadata.get("failure_type"),
                "failure_message": trace.metadata.get("failure_message"),
                "duration_ns": (
                    None if trace.end_timestamp_ns is None
                    else trace.end_timestamp_ns - trace.start_timestamp_ns
                ),
            },
            "artifacts": {
                "trace": "trace.json",
                "episode_metrics": "metrics.episode.json",
                "video": "video.mp4",
                "video_metadata": "video.json",
                "run_integrity": "../../../../integrity.json",
            },
            "source_of_truth": {
                "trace": "trace.json",
                "metrics": "metrics.episode.json",
                "configuration": "../../../../resolved_config.yaml",
                "note": "metadata.json is a derived human-readable episode index",
            },
        }

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
        from ..reporting import (
            generate_canonical_videos, integrity_document, remove_generated_outputs,
            write_canonical_report,
        )
        encoded = json.dumps(_json_value(manifest), sort_keys=True, separators=(",", ":")).encode()
        temporary = path / f"{name}.tmp"
        if temporary.exists():
            raise ArtifactError("partial final manifest already exists")
        try:
            generate_canonical_videos(path)
            write_canonical_report(path, final_manifest=manifest)
            document = integrity_document(path, virtual_files={name: encoded})
            self._atomic_json(path / "integrity.json", document)
            temporary.write_bytes(encoded)
            temporary.replace(path / name)
        except Exception:
            try: temporary.unlink()
            except FileNotFoundError: pass
            remove_generated_outputs(path)
            raise

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

    @staticmethod
    def _atomic_text(path, value):
        if path.exists(): raise ArtifactError(f"finalized artifact already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)


def _metric(result):
    return {"metric_id": result.metric_id, "metric_version": result.metric_version, "scope": result.scope.value, "status": result.status.value, "value": _json_value(result.value), "unit": result.unit, "sample_count": result.sample_count, "run_id": str(result.run_id), "task_id": str(result.task_id), "episode_id": None if result.episode_id is None else str(result.episode_id), "reason": result.reason, "diagnostics": _json_value(result.diagnostics), "metric_config": _json_value(result.metric_config), "metric_config_hash": result.metric_config_hash, "metadata": _json_value(result.metadata)}


def _decode_metric(value):
    return MetricResult(value["metric_id"], value["metric_version"], MetricScope(value["scope"]), MetricStatus(value["status"]), value["value"], value["unit"], value["sample_count"], RunId(value["run_id"]), TaskId(value["task_id"]), None if value["episode_id"] is None else EpisodeId(value["episode_id"]), value["reason"], value["diagnostics"], value["metric_config"], value["metric_config_hash"], value["metadata"])


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read episode metadata source: {path.name}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"episode metadata source must be an object: {path.name}")
    return value


def _method_descriptor(started):
    policy_runtime = started.get("runtime", {}).get("policy", {})
    remote = policy_runtime.get("remote_policy", {})
    descriptor = remote.get("method_descriptor", policy_runtime.get("method_descriptor", {}))
    return dict(descriptor) if isinstance(descriptor, dict) else {}


def _method_summary(method):
    keys = (
        "family", "id", "method_name", "version", "artifact_form",
        "backbone_adaptation", "backbone_merge_status", "quantization",
        "action_chunk_size", "action_dimension", "normalization", "objective",
        "parameter_counts", "training_step",
    )
    summary = {key: method[key] for key in keys if key in method}
    lora = method.get("lora")
    if isinstance(lora, dict):
        lora_keys = (
            "peft_type", "rank", "alpha", "scaling", "dropout", "bias",
            "target_modules", "modules_to_save", "trainable_parameter_count",
        )
        summary["lora"] = {key: lora[key] for key in lora_keys if key in lora}
    return summary


def _benchmark_family(context_metadata, started):
    suite = context_metadata.get("suite_name")
    if isinstance(suite, str) and suite.upper().startswith("LIBERO"):
        return "LIBERO"
    return started.get("benchmark", {}).get("name")


def _iso_utc(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    result = datetime.fromtimestamp(seconds, timezone.utc).replace(
        microsecond=nanoseconds // 1_000
    )
    return result.isoformat().replace("+00:00", "Z")


def _connection(report):
    return {"benchmark": {"name": report.benchmark_name, "version": report.benchmark_version}, "policy": {"name": report.policy_name, "version": report.policy_version}, "contract_version": report.contract_version, "compatible": report.compatibility_report.compatible, "compatibility_issues": [{"code": issue.code, "severity": issue.severity.value, "path": issue.path, "message": issue.message} for issue in report.compatibility_report.issues], "selected_task_ids": [str(task.task_id) for task in report.selected_tasks], "enabled_metrics": [{"metric_id": item.metric_id, "metric_version": item.metric_version} for item in report.enabled_metric_descriptors], "required_metrics": list(report.required_metric_ids), "statically_available_metrics": list(report.statically_available_metric_ids), "potentially_unavailable_metrics": list(report.potentially_unavailable_metric_ids), "plan_hash": report.plan_hash, "metadata": dict(report.metadata)}


def _json_value(value):
    import numpy as np
    if isinstance(value, np.ndarray): return value.tolist()
    if hasattr(value, "items"): return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_json_value(item) for item in value]
    return value
