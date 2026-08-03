"""Reproducible offline HTML reports over immutable canonical run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import hashlib
import html
import importlib.resources
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import uuid

import numpy as np

from .artifacts.codec import TraceCodec
from .artifacts.layout import run_key, safe_key
from .errors import ArtifactError, ReportingRendererError, ReportingSourceUnavailableError
from .inspection import verify_run
from .permissions import finalize_managed_directory, finalize_managed_tree


REPORT_PROFILE_SCHEMA = "ovlab.report-profile/v1"
REPORT_SCHEMA = "ovlab.report/v1"
REPORT_MANIFEST_SCHEMA = "ovlab.report-manifest/v1"
REPORT_RENDERER_ID = "ovlab-jinja-static-html"
REPORT_RENDERER_VERSION = "1.3.0"
CHART_BUILDERS = {
    "action_timeseries": "1.2.0",
    "latency_distribution": "1.2.0",
    "episode_outcomes": "1.2.0",
    "vram_timeseries": "1.0.0",
    "estimated_compute_timeseries": "1.0.0",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SECTIONS = {
    "overview", "episode_results", "task_metrics", "action_metrics",
    "system_metrics", "action_trajectories", "latency", "videos", "provenance",
}


def _plain(value):
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value) -> bytes:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest_value(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read canonical reporting source: {path}") from exc


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(_canonical_bytes(value))
    temporary.replace(path)


def _inside(path: Path, root: Path, label: str) -> Path:
    root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ArtifactError(f"{label} escapes its configured root: {path}")
    return resolved


@dataclass(frozen=True, slots=True)
class ReportProfile:
    identifier: str
    template: str
    sections: tuple[str, ...]
    charts: tuple[dict[str, str], ...]
    options: dict[str, bool]
    local_template_root: Path | None = None

    @classmethod
    def from_mapping(cls, document, *, template_base: str | Path | None = None) -> "ReportProfile":
        doc = _plain(document)
        if not isinstance(doc, dict) or doc.get("schema_version") != REPORT_PROFILE_SCHEMA:
            raise ArtifactError(f"report profile schema_version must equal {REPORT_PROFILE_SCHEMA}")
        allowed = {"schema_version", "id", "template", "sections", "charts", "options"}
        unknown = sorted(set(doc) - allowed)
        if unknown:
            raise ArtifactError(f"report profile contains unknown keys: {', '.join(unknown)}")
        for key in ("id", "template", "sections", "charts", "options"):
            if key not in doc:
                raise ArtifactError(f"report profile is missing required key: {key}")
        identifier = doc["id"]
        if not isinstance(identifier, str) or not _SAFE_ID.fullmatch(identifier):
            raise ArtifactError("report profile id must be a portable identifier")
        template = doc["template"]
        if not isinstance(template, str) or not template:
            raise ArtifactError("report profile template must be a built-in ID or local bundle path")
        local_template_root = None
        if template != "benchmark/task-v1":
            if template_base is None:
                raise ArtifactError("a local report template requires an explicit profile/template root")
            base = Path(template_base).expanduser().resolve()
            candidate = Path(template).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            local_template_root = candidate.resolve()
            if not local_template_root.is_relative_to(base):
                raise ArtifactError("local report template escapes its selected template root")
            if not local_template_root.is_dir():
                raise ArtifactError("local report template bundle does not exist")
            for name in ("run-v1.html", "task-v1.html", "style.css"):
                asset = (local_template_root / name).resolve()
                if not asset.is_relative_to(local_template_root) or not asset.is_file():
                    raise ArtifactError(f"local report template bundle is missing safe asset: {name}")
        sections = doc["sections"]
        if not isinstance(sections, list) or not sections or len(sections) != len(set(sections)):
            raise ArtifactError("report profile sections must be a non-empty unique list")
        invalid_sections = [value for value in sections if value not in _SECTIONS]
        if invalid_sections:
            raise ArtifactError(f"unsupported report sections: {', '.join(invalid_sections)}")
        charts = doc["charts"]
        if not isinstance(charts, list):
            raise ArtifactError("report profile charts must be a list")
        normalized_charts = []
        seen = set()
        for index, chart in enumerate(charts):
            if not isinstance(chart, dict) or set(chart) != {"id", "builder"}:
                raise ArtifactError(f"report profile charts[{index}] must contain only id and builder")
            if not isinstance(chart["id"], str) or not _SAFE_ID.fullmatch(chart["id"]):
                raise ArtifactError(f"report profile charts[{index}].id is invalid")
            if chart["id"] in seen:
                raise ArtifactError("report profile chart ids must be unique")
            if chart["builder"] not in CHART_BUILDERS:
                raise ArtifactError(f"unknown report chart builder: {chart['builder']}")
            seen.add(chart["id"])
            normalized_charts.append({"id": chart["id"], "builder": chart["builder"]})
        options = doc["options"]
        expected_options = {"show_units", "show_unavailable", "include_raw_provenance"}
        if not isinstance(options, dict) or set(options) != expected_options:
            raise ArtifactError("report profile options must contain show_units, show_unavailable, and include_raw_provenance")
        if any(type(value) is not bool for value in options.values()):
            raise ArtifactError("report profile options must be booleans")
        return cls(identifier, template, tuple(sections), tuple(normalized_charts), dict(options), local_template_root)

    def document(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_PROFILE_SCHEMA,
            "id": self.identifier,
            "template": "local-template-bundle" if self.local_template_root is not None else self.template,
            "sections": list(self.sections),
            "charts": [dict(item) for item in self.charts],
            "options": dict(self.options),
        }


def builtin_profile(identifier: str = "libero-task-default") -> ReportProfile:
    if identifier != "libero-task-default":
        raise ArtifactError(f"unknown built-in report profile: {identifier}")
    resource = importlib.resources.files("ovlab_runner").joinpath(
        "report_assets/profiles/libero-task-default.yaml"
    )
    return ReportProfile.from_mapping(json.loads(resource.read_text(encoding="utf-8")))


def report_profiles() -> tuple[dict[str, object], ...]:
    profile = builtin_profile()
    return ({
        "id": profile.identifier,
        "schema_version": REPORT_PROFILE_SCHEMA,
        "template": profile.template,
        "sections": list(profile.sections),
        "charts": [dict(item) for item in profile.charts],
        "source": "built-in",
    },)


def _template_assets(profile: ReportProfile) -> tuple[dict[str, object], ...]:
    if profile.local_template_root is not None:
        rows = []
        for name in ("run-v1.html", "task-v1.html", "style.css"):
            data = (profile.local_template_root / name).read_bytes()
            rows.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
        return tuple(rows)
    root = importlib.resources.files("ovlab_runner").joinpath("report_assets")
    rows = []
    for relative in (
        "templates/benchmark/run-v1.html",
        "templates/benchmark/task-v1.html",
        "static/style.css",
    ):
        data = root.joinpath(relative).read_bytes()
        rows.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
    return tuple(rows)


def _template_digest(profile: ReportProfile) -> str:
    return _digest_value(_template_assets(profile))


def _final_manifest(run_path: Path):
    candidates = [path for path in (run_path / "manifest.completed.json", run_path / "manifest.failed.json") if path.is_file()]
    if len(candidates) > 1:
        raise ArtifactError("canonical run contains multiple final manifests")
    return (_json(candidates[0]), candidates[0].name) if candidates else (None, None)


def _canonical_inputs(run_path: Path, task_key: str | None = None) -> tuple[dict[str, object], ...]:
    integrity = run_path / "integrity.json"
    if integrity.is_file() and task_key is None:
        return ({"path": "integrity.json", "sha256": _sha256(integrity), "size_bytes": integrity.stat().st_size},)
    candidates = []
    fixed = ("manifest.started.json", "plan.json", "connection.json", "source_config.yaml", "resolved_config.yaml")
    for name in fixed:
        path = run_path / name
        if path.is_file():
            candidates.append(path)
    final, final_name = _final_manifest(run_path)
    if final_name:
        candidates.append(run_path / final_name)
    task_root = run_path / "tasks"
    if task_key is None:
        candidates.extend(path for path in task_root.rglob("*") if path.is_file() and not path.name.endswith(".tmp"))
    else:
        selected = _inside(task_root / task_key, task_root, "task")
        candidates.extend(path for path in selected.rglob("*") if path.is_file() and not path.name.endswith(".tmp"))
    rows = []
    for path in sorted(set(candidates)):
        rows.append({"path": str(path.relative_to(run_path)), "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    return tuple(rows)


def _metric_category(metric_id: str) -> str:
    if metric_id.startswith("task.") or metric_id.startswith("episode."):
        return "task"
    if metric_id.startswith("system."):
        return "system"
    return "action"


def _normalized_metric(metric: dict[str, object]) -> dict[str, object]:
    result = dict(metric)
    result["category"] = _metric_category(str(metric.get("metric_id", "")))
    result["canonical_metric_ref"] = {
        "metric_id": metric.get("metric_id"),
        "metric_version": metric.get("metric_version"),
        "metric_config_hash": metric.get("metric_config_hash"),
        "scope": metric.get("scope"),
        "task_id": metric.get("task_id"),
        "episode_id": metric.get("episode_id"),
    }
    value = result.get("value")
    if result.get("status") != "available":
        result["value"] = None
    elif isinstance(value, dict):
        value = dict(value)
        n = value.get("valid_episode_count", result.get("sample_count", 0))
        if isinstance(n, int) and n < 2 and "standard_deviation" in value:
            value["standard_deviation"] = None
            value["standard_deviation_qualification"] = "unavailable for n < 2"
        result["value"] = value
    result["aggregation"] = {
        "n": result.get("sample_count", 0),
        "unit": result.get("unit"),
        "reduction": (
            "canonical task aggregation" if result.get("scope") == "task" else "canonical episode metric"
        ),
    }
    return result


def _success_summary(episodes: list[dict[str, object]], expected: int) -> dict[str, object]:
    finalized = len(episodes)
    successful = sum(item["terminal_status"] == "success" for item in episodes)
    failed = sum(item["terminal_status"] == "failure" for item in episodes)
    interrupted = sum(item["terminal_status"] in {"aborted", "policy_error", "benchmark_error"} for item in episodes)
    missing = max(expected - finalized, 0)
    base = {
        "eligible_episode_count": finalized,
        "successful_episode_count": successful,
        "failed_episode_count": failed,
        "interrupted_or_invalid_episode_count": interrupted,
        "missing_episode_count": missing,
        "denominator_semantics": "all finalized episodes; missing episodes are displayed separately and never silently excluded",
    }
    if finalized == 1:
        return {**base, "presentation": "binary_success", "success": successful == 1, "success_rate": None}
    if finalized > 1:
        return {**base, "presentation": "success_rate", "success": None, "success_rate": successful / finalized}
    return {**base, "presentation": "unavailable", "success": None, "success_rate": None}


def _downsample(values: list, maximum: int = 500) -> list:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return [values[int(index)] for index in indices]


def _descriptive_statistics(values: list[float]) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    count = int(finite.size)
    non_finite_count = int(array.size - finite.size)
    if count == 0:
        return {
            "n": 0, "non_finite_count": non_finite_count, "minimum": None,
            "p05": None, "median": None, "mean": None,
            "sample_standard_deviation": None, "p95": None, "maximum": None,
        }
    return {
        "n": count,
        "non_finite_count": non_finite_count,
        "minimum": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "sample_standard_deviation": float(np.std(finite, ddof=1)) if count >= 2 else None,
        "p95": float(np.quantile(finite, 0.95)),
        "maximum": float(np.max(finite)),
    }


def _action_component_names(dimension: int) -> tuple[str, ...]:
    canonical = ("tx", "ty", "tz", "rx", "ry", "rz", "gripper")
    return tuple(
        canonical[index] if index < len(canonical) else f"component_{index}"
        for index in range(dimension)
    )


def _trace_view(episode_path: Path, trace) -> dict[str, object]:
    actions = [list(map(float, item.applied_action)) for item in trace.executed_actions]
    inference = [item.inference_duration_ns / 1_000_000 for item in trace.policy_predictions]
    rpc = [item.metadata.get("rpc_round_trip_duration_ns") for item in trace.policy_predictions]
    rpc = [float(value) / 1_000_000 for value in rpc if isinstance(value, int)]
    closed = [item.metadata.get("closed_loop_step_duration_ns") for item in trace.executed_actions]
    closed = [float(value) / 1_000_000 for value in closed if isinstance(value, int)]
    transitions = 0
    gripper = [row[6] for row in actions if len(row) > 6]
    for previous, current in zip(gripper, gripper[1:]):
        transitions += (previous >= 0) != (current >= 0)
    component_names = _action_component_names(max((len(row) for row in actions), default=0))
    allocated, reserved, peak_allocated, peak_reserved, estimated_gflops = [], [], [], [], []
    compute_identity = None
    for prediction in trace.policy_predictions:
        runtime = prediction.metadata.get("runtime", {})
        performance = runtime.get("performance", {}) if isinstance(runtime, dict) else {}
        if not performance and isinstance(prediction.metadata.get("performance"), dict):
            performance = prediction.metadata["performance"]
        after = performance.get("cuda_memory_after", {}) if isinstance(performance, dict) else {}
        compute = performance.get("estimated_compute", {}) if isinstance(performance, dict) else {}
        if isinstance(after, dict) and after.get("status") == "available":
            for target, key in (
                (allocated, "allocated_bytes"), (reserved, "reserved_bytes"),
                (peak_allocated, "peak_allocated_bytes"), (peak_reserved, "peak_reserved_bytes"),
            ):
                value = after.get(key)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    target.append(float(value) / (1024 * 1024))
        value = compute.get("estimated_gflops") if isinstance(compute, dict) else None
        if isinstance(value, int | float) and not isinstance(value, bool):
            estimated_gflops.append(float(value))
        if compute_identity is None and isinstance(compute, dict) and compute.get("method"):
            compute_identity = {
                key: compute.get(key) for key in ("method", "formula", "qualification")
            }
    return {
        "action_series": _downsample(actions),
        "action_sample_count": len(actions),
        "action_component_statistics": {
            name: _descriptive_statistics([row[index] for row in actions if len(row) > index])
            for index, name in enumerate(component_names)
        },
        "gripper_series": _downsample(gripper),
        "gripper_transition_count": int(transitions),
        "inference_latency_ms": _downsample(inference),
        "inference_latency_statistics_ms": _descriptive_statistics(inference),
        "rpc_latency_ms": _downsample(rpc),
        "closed_loop_latency_ms": _downsample(closed),
        "vram_allocated_mib": _downsample(allocated),
        "vram_reserved_mib": _downsample(reserved),
        "vram_peak_allocated_mib": _downsample(peak_allocated),
        "vram_peak_reserved_mib": _downsample(peak_reserved),
        "vram_statistics_mib": {
            "allocated": _descriptive_statistics(allocated),
            "reserved": _descriptive_statistics(reserved),
            "peak_allocated": _descriptive_statistics(peak_allocated),
            "peak_reserved": _descriptive_statistics(peak_reserved),
        },
        "estimated_gflops": _downsample(estimated_gflops),
        "estimated_gflops_statistics": _descriptive_statistics(estimated_gflops),
        "estimated_compute_identity": compute_identity,
        "trace_reference": str(episode_path.name + "/trace.json"),
    }


def _policy_performance_identity(connection: dict[str, object]) -> dict[str, object]:
    metadata = connection.get("metadata", {})
    capabilities = metadata.get("policy_capabilities", {}) if isinstance(metadata, dict) else {}
    runtime = capabilities.get("runtime", {}) if isinstance(capabilities, dict) else {}
    method = capabilities.get("method_descriptor", {}) if isinstance(capabilities, dict) else {}
    parameter_counts = runtime.get("parameter_counts") if isinstance(runtime, dict) else None
    if not isinstance(parameter_counts, dict):
        parameter_counts = method.get("parameter_counts") if isinstance(method, dict) else None
    if not isinstance(parameter_counts, dict):
        total = None
        if isinstance(runtime, dict):
            total = runtime.get("total_parameter_count", runtime.get("total_runtime_parameter_count"))
        if total is None and isinstance(method, dict):
            total = method.get("total_runtime_parameter_count")
        parameter_counts = {"total": total} if total is not None else {}
    return {
        "schema_version": "ovlab.model-complexity/v1",
        "parameter_counts": parameter_counts,
        "method_family": method.get("family", method.get("method_family")) if isinstance(method, dict) else None,
        "method_id": method.get("method_id") if isinstance(method, dict) else None,
        "merge_status": method.get("merge_status", method.get("backbone_merge_status")) if isinstance(method, dict) else None,
        "quantization": method.get("quantization") if isinstance(method, dict) else None,
        "cuda_memory_after_load": runtime.get("cuda_memory_after_load") if isinstance(runtime, dict) else None,
        "adapter_parameter_semantics": (
            "live runtime adapter parameters; merged adapters remain zero/unavailable and historical training counts stay in method provenance"
        ),
    }


def build_report_model(run_path: str | Path, *, scope_task_id: str | None = None) -> dict[str, object]:
    path = Path(run_path).expanduser().resolve()
    started = _json(path / "manifest.started.json")
    plan = _json(path / "plan.json")
    connection = _json(path / "connection.json") if (path / "connection.json").is_file() else {}
    final, final_name = _final_manifest(path)
    final = final or {}
    expected_per_task = int(plan.get("rollout_count_per_task", 1))
    task_views = []
    codec = TraceCodec()
    for task_path in sorted((path / "tasks").glob("*")):
        if not task_path.is_dir():
            continue
        episode_views = []
        for episode_path in sorted((task_path / "episodes").glob("*")):
            if not (episode_path / "trace.finalized.json").is_file():
                continue
            trace = codec.decode(episode_path)
            task_id = str(trace.episode_context.task_id)
            if scope_task_id is not None and task_id != scope_task_id and task_path.name != scope_task_id:
                continue
            metrics = _json(episode_path / "metrics.episode.json") if (episode_path / "metrics.episode.json").is_file() else []
            video = _json(episode_path / "video.json") if (episode_path / "video.json").is_file() else {
                "status": "unavailable", "reason": "canonical video is not finalized yet",
            }
            episode_views.append({
                "episode_id": str(trace.episode_context.episode_id),
                "seed": trace.episode_context.seed,
                "rollout_index": trace.episode_context.rollout_index,
                "terminal_status": trace.terminal_status.value,
                "success": trace.terminal_status.value == "success",
                "instruction": trace.episode_context.initial_instruction.text,
                "initial_state_identity": trace.metadata.get("benchmark_reset", {}).get("initial_state_index"),
                "failure_type": trace.metadata.get("failure_type"),
                "failure_message": trace.metadata.get("failure_message"),
                "partial": trace.terminal_status.value in {"aborted", "policy_error", "benchmark_error"},
                "metrics": [_normalized_metric(item) for item in metrics],
                "video": video,
                "canonical_episode_path": str(episode_path.relative_to(path)),
                "trace_view": _trace_view(episode_path, trace),
            })
        if scope_task_id is not None and not episode_views:
            continue
        task_metrics_path = task_path / "metrics.task.json"
        task_metrics = _json(task_metrics_path) if task_metrics_path.is_file() else []
        task_id = (
            str(episode_views[0].get("canonical_episode_path", "")).split("/")[1]
            if False else (task_metrics[0].get("task_id") if task_metrics else None)
        )
        if task_id is None and episode_views:
            trace = codec.decode(next((task_path / "episodes").glob("*")))
            task_id = str(trace.episode_context.task_id)
        if scope_task_id is not None and task_id != scope_task_id and task_path.name != scope_task_id:
            continue
        instruction = episode_views[0]["instruction"] if episode_views else "unavailable"
        terminal_values = {item["terminal_status"] for item in episode_views}
        task_status = "completed"
        if terminal_values & {"benchmark_error", "policy_error", "failure"}:
            task_status = "failed"
        elif terminal_values & {"aborted"} or len(episode_views) < expected_per_task:
            task_status = "interrupted" if final.get("status") == "interrupted" else "partial"
        task_views.append({
            "task_id": task_id or task_path.name,
            "task_key": task_path.name,
            "instruction": instruction,
            "status": task_status,
            "partial": task_status in {"partial", "interrupted"},
            "outcome": _success_summary(episode_views, expected_per_task),
            "episodes": episode_views,
            "metrics": [_normalized_metric(item) for item in task_metrics],
        })
    if scope_task_id is not None and not task_views:
        raise ArtifactError(f"task is not present in canonical run: {scope_task_id}")
    renderer = final.get("metadata", {}).get("benchmark_runtime", {}).get(
        "libero_renderer", started.get("runtime", {}).get("benchmark", {}).get("libero_renderer")
    )
    detected = renderer.get("detected_renderer") if isinstance(renderer, dict) else None
    software = isinstance(detected, dict) and "llvmpipe" in str(detected.get("renderer", "")).lower()
    all_episodes = [episode for task in task_views for episode in task["episodes"]]
    vram_peak_allocated = [
        value for episode in all_episodes
        for value in episode["trace_view"]["vram_peak_allocated_mib"]
    ]
    vram_peak_reserved = [
        value for episode in all_episodes
        for value in episode["trace_view"]["vram_peak_reserved_mib"]
    ]
    estimated_compute = [
        value for episode in all_episodes
        for value in episode["trace_view"]["estimated_gflops"]
    ]
    compute_estimators = []
    for episode in all_episodes:
        identity = episode["trace_view"].get("estimated_compute_identity")
        if isinstance(identity, dict) and identity not in compute_estimators:
            compute_estimators.append(identity)
    run_status = final.get("status", "partial")
    complexity = _policy_performance_identity(connection)
    load_memory = complexity.get("cuda_memory_after_load")
    load_peak_allocated = None
    load_peak_reserved = None
    if isinstance(load_memory, dict) and load_memory.get("status") == "available":
        if isinstance(load_memory.get("peak_allocated_bytes"), int | float):
            load_peak_allocated = float(load_memory["peak_allocated_bytes"]) / (1024 * 1024)
        if isinstance(load_memory.get("peak_reserved_bytes"), int | float):
            load_peak_reserved = float(load_memory["peak_reserved_bytes"]) / (1024 * 1024)
    peak_allocated_candidates = [*vram_peak_allocated, *([] if load_peak_allocated is None else [load_peak_allocated])]
    peak_reserved_candidates = [*vram_peak_reserved, *([] if load_peak_reserved is None else [load_peak_reserved])]
    return {
        "schema_version": REPORT_SCHEMA,
        "scope": "task" if scope_task_id is not None else "run",
        "scope_task_id": scope_task_id,
        "completeness": "complete" if final_name is not None else "partial",
        "run": {
            "run_id": started.get("run_id"),
            "status": run_status,
            "infrastructure_status": run_status,
            "scientific_config_hash": started.get("scientific_config_hash"),
            "execution_config_hash": started.get("execution_config_hash"),
            "experiment": started.get("metadata", {}).get("experiment_id"),
            "experiment_name": started.get("metadata", {}).get("experiment_name"),
        },
        "benchmark": connection.get("benchmark"),
        "policy": connection.get("policy"),
        "performance": {
            "telemetry_schema": "ovlab.performance-telemetry/v1",
            "model_complexity": complexity,
            "vram_source": "PyTorch CUDA caching allocator in the isolated policy process",
            "vram_qualification": "not whole-device NVML memory usage",
            "compute_qualification": "estimated analytical proxy, not measured hardware FLOPs",
            "compute_estimators": compute_estimators,
            "summary": {
                "prediction_sample_count": sum(
                    int(episode["trace_view"]["inference_latency_statistics_ms"]["n"])
                    for episode in all_episodes
                ),
                "peak_allocated_mib": max(peak_allocated_candidates) if peak_allocated_candidates else None,
                "peak_reserved_mib": max(peak_reserved_candidates) if peak_reserved_candidates else None,
                "load_peak_allocated_mib": load_peak_allocated,
                "load_peak_reserved_mib": load_peak_reserved,
                "estimated_total_gflops": sum(estimated_compute) if estimated_compute else None,
            },
        },
        "renderer": {
            "configuration": renderer,
            "acceleration": "software" if software else "hardware-or-unknown",
            "description": "Mesa llvmpipe software rendering" if software else "configured renderer",
        },
        "outcome": _success_summary(all_episodes, len(task_views) * expected_per_task),
        "tasks": task_views,
        "metric_semantics": {
            "unavailable": "value is null and is never represented as zero",
            "sample_standard_deviation": "not displayed for n < 2",
            "source": "canonical metrics.task.json and metrics.episode.json",
        },
        "provenance": {
            "started_manifest": "manifest.started.json",
            "final_manifest": final_name,
            "connection": "connection.json",
            "plan": "plan.json",
            "configuration": ["source_config.yaml", "resolved_config.yaml"],
            "canonical_integrity": "integrity.json" if (path / "integrity.json").is_file() else None,
            "verification": "verified" if final_name and (path / "integrity.json").is_file() else "partial-input",
        },
    }


def _chart_axes(builder: str) -> dict[str, str]:
    if builder == "action_timeseries":
        return {"x": "Ordered control sample", "y": "Applied action (normalized command)"}
    if builder == "latency_distribution":
        return {"x": "Ordered prediction sample", "y": "Policy inference latency (ms)"}
    if builder == "vram_timeseries":
        return {"x": "Ordered prediction sample", "y": "PyTorch CUDA allocator memory (MiB)"}
    if builder == "estimated_compute_timeseries":
        return {"x": "Ordered prediction sample", "y": "Estimated compute (GFLOPs)"}
    return {"x": "Episode index", "y": "Outcome (0=failure, 1=success)"}


def _chart_statistics(builder: str, model: dict[str, object]) -> dict[str, object]:
    rows = []
    if builder == "action_timeseries":
        for task in model["tasks"]:
            for episode in task["episodes"]:
                for component, statistics in episode["trace_view"]["action_component_statistics"].items():
                    rows.append({
                        "task_id": task["task_id"],
                        "episode_id": episode["episode_id"],
                        "series": component,
                        "unit": "normalized_command",
                        **statistics,
                    })
        source = "full canonical trace executed_actions.applied_action; chart downsampling is not used"
    elif builder == "latency_distribution":
        for task in model["tasks"]:
            for episode in task["episodes"]:
                rows.append({
                    "task_id": task["task_id"],
                    "episode_id": episode["episode_id"],
                    "series": "policy_inference",
                    "unit": "ms",
                    **episode["trace_view"]["inference_latency_statistics_ms"],
                })
        source = "full canonical trace policy_predictions.inference_duration_ns converted to ms"
    elif builder == "vram_timeseries":
        for task in model["tasks"]:
            for episode in task["episodes"]:
                for series_name, statistics in episode["trace_view"]["vram_statistics_mib"].items():
                    rows.append({
                        "task_id": task["task_id"], "episode_id": episode["episode_id"],
                        "series": series_name, "unit": "MiB", **statistics,
                    })
        source = "full canonical prediction performance.cuda_memory_after; bytes converted to MiB"
    elif builder == "estimated_compute_timeseries":
        for task in model["tasks"]:
            for episode in task["episodes"]:
                rows.append({
                    "task_id": task["task_id"], "episode_id": episode["episode_id"],
                    "series": "estimated_compute", "unit": "GFLOPs",
                    **episode["trace_view"]["estimated_gflops_statistics"],
                })
        source = "full canonical prediction performance.estimated_compute; analytical estimate, not measurement"
    else:
        all_values = []
        for task in model["tasks"]:
            values = [1.0 if episode["success"] else 0.0 for episode in task["episodes"]]
            all_values.extend(values)
            rows.append({
                "task_id": task["task_id"],
                "episode_id": None,
                "series": "success_indicator",
                "unit": "binary",
                **_descriptive_statistics(values),
            })
        rows.insert(0, {
            "task_id": None,
            "episode_id": None,
            "series": "success_indicator_all_tasks",
            "unit": "binary",
            **_descriptive_statistics(all_values),
        })
        source = "canonical terminal_status mapped exactly as success=1 and every other finalized outcome=0"
    return {
        "schema_version": "ovlab.descriptive-statistics/v1",
        "source": source,
        "population": "all finite canonical samples in the stated task/episode scope",
        "standard_deviation": "sample standard deviation (ddof=1); unavailable for n < 2",
        "quantiles": "NumPy linear interpolation at 0.05 and 0.95",
        "rows": rows,
    }


def _sampled_points(values: list[float], maximum: int = 400) -> list[list[float]]:
    if not values:
        return []
    indices = (
        np.arange(len(values), dtype=int)
        if len(values) <= maximum else
        np.linspace(0, len(values) - 1, maximum, dtype=int)
    )
    return [[int(index), float(values[int(index)])] for index in indices]


def _chart_svg(chart_id: str, builder: str, model: dict[str, object]) -> str:
    width, height = 960, 430
    left, top, plot_width, plot_height = 78, 72, 850, 292
    colors = ("#155eef", "#d92d20", "#039855", "#7a5af8", "#dc6803", "#088ab2", "#667085")
    axes = _chart_axes(builder)
    series: list[dict[str, object]] = []
    labels: list[str] = []

    if builder == "episode_outcomes":
        values = []
        for task in model["tasks"]:
            for episode in task["episodes"]:
                labels.append(str(episode["episode_id"]))
                values.append(1.0 if episode["success"] else 0.0)
        series.append({"name": "success", "color": colors[0], "points": _sampled_points(values)})
        y_min, y_max = -0.05, 1.05
    elif builder == "latency_distribution":
        values = []
        for task in model["tasks"]:
            for episode in task["episodes"]:
                values.extend(episode["trace_view"]["inference_latency_ms"])
        labels = [str(index) for index in range(len(values))]
        series.append({"name": "inference_ms", "color": colors[0], "points": _sampled_points(values)})
        finite = [float(value) for value in values if math.isfinite(float(value))]
        y_min, y_max = (min(finite), max(finite)) if finite else (0.0, 1.0)
    elif builder == "vram_timeseries":
        combined = {"allocated": [], "reserved": [], "peak_allocated": [], "peak_reserved": []}
        for task in model["tasks"]:
            for episode in task["episodes"]:
                view = episode["trace_view"]
                combined["allocated"].extend(view["vram_allocated_mib"])
                combined["reserved"].extend(view["vram_reserved_mib"])
                combined["peak_allocated"].extend(view["vram_peak_allocated_mib"])
                combined["peak_reserved"].extend(view["vram_peak_reserved_mib"])
        labels = [str(index) for index in range(max((len(values) for values in combined.values()), default=0))]
        for index, (name, values) in enumerate(combined.items()):
            series.append({"name": name, "color": colors[index], "points": _sampled_points(values)})
        finite = [point[1] for item in series for point in item["points"] if math.isfinite(point[1])]
        y_min, y_max = (min(finite), max(finite)) if finite else (0.0, 1.0)
    elif builder == "estimated_compute_timeseries":
        values = []
        for task in model["tasks"]:
            for episode in task["episodes"]:
                values.extend(episode["trace_view"]["estimated_gflops"])
        labels = [str(index) for index in range(len(values))]
        series.append({"name": "estimated_gflops", "color": colors[0], "points": _sampled_points(values)})
        finite = [float(value) for value in values if math.isfinite(float(value))]
        y_min, y_max = (min(finite), max(finite)) if finite else (0.0, 1.0)
    else:
        rows = []
        for task in model["tasks"]:
            for episode in task["episodes"]:
                rows.extend(episode["trace_view"]["action_series"])
        labels = [str(index) for index in range(len(rows))]
        names = _action_component_names(max((len(row) for row in rows), default=0))
        for component, name in enumerate(names):
            values = [float(row[component]) for row in rows if len(row) > component]
            series.append({"name": name, "color": colors[component], "points": _sampled_points(values)})
        finite = [point[1] for item in series for point in item["points"] if math.isfinite(point[1])]
        y_min, y_max = (min(finite), max(finite)) if finite else (-1.0, 1.0)

    if not math.isfinite(y_min) or not math.isfinite(y_max):
        y_min, y_max = 0.0, 1.0
    if y_max == y_min:
        padding = max(abs(y_min) * 0.05, 0.5)
        y_min, y_max = y_min - padding, y_max + padding
    else:
        padding = (y_max - y_min) * 0.05
        y_min, y_max = y_min - padding, y_max + padding
    max_x = max((point[0] for item in series for point in item["points"]), default=1)
    max_x = max(int(max_x), 1)
    y_span = y_max - y_min

    paths = []
    legend = []
    for item_index, item in enumerate(series):
        coordinates = []
        for x_value, y_value in item["points"]:
            if not math.isfinite(y_value):
                continue
            y_plot = plot_height - (y_value - y_min) * plot_height / y_span
            coordinates.append(f"{x_value:.3f},{y_plot:.3f}")
        if coordinates:
            paths.append(
                f'<polyline class="data-series" data-series="{html.escape(str(item["name"]))}" '
                f'fill="none" stroke="{item["color"]}" stroke-width="2" vector-effect="non-scaling-stroke" '
                f'points="{" ".join(coordinates)}"/>'
            )
        legend_x = left + (item_index % 4) * 150
        legend_y = 47 + (item_index // 4) * 18
        legend.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" '
            f'stroke="{item["color"]}" stroke-width="3"/>'
            f'<text x="{legend_x + 28}" y="{legend_y + 4}" class="legend-label">{html.escape(str(item["name"]))}</text>'
        )

    y_ticks = []
    for index in range(5):
        ratio = index / 4
        y = top + ratio * plot_height
        value = y_max - ratio * y_span
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.3f}" x2="{left + plot_width}" y2="{y:.3f}" class="grid"/>'
            f'<text x="{left - 10}" y="{y + 4:.3f}" text-anchor="end" class="tick-label">{value:.4g}</text>'
        )
    x_ticks = []
    for index in range(5):
        x = left + index * plot_width / 4
        value = max_x * index / 4
        x_ticks.append(
            f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{top + plot_height}" class="grid"/>'
            f'<text id="x-tick-{index}" x="{x:.3f}" y="{top + plot_height + 21}" text-anchor="middle" '
            f'class="tick-label">{value:.4g}</text>'
        )

    payload = base64.b64encode(_canonical_bytes({
        "max_x": max_x,
        "labels": labels,
        "series": series,
        "plot": {"left": left, "top": top, "width": plot_width, "height": plot_height},
    })).decode("ascii")
    empty = (
        f'<text x="{left + plot_width / 2}" y="{top + plot_height / 2}" text-anchor="middle" class="empty">'
        'No canonical samples available</text>'
        if not any(item["points"] for item in series) else ""
    )
    script = r"""
<script><![CDATA[
(() => {
  const payload = JSON.parse(atob(document.getElementById("chart-data").textContent.trim()));
  const root = document.documentElement;
  const viewport = document.getElementById("viewport");
  const overlay = document.getElementById("interaction");
  const cursor = document.getElementById("cursor");
  const tooltip = document.getElementById("tooltip");
  const tooltipBox = document.getElementById("tooltip-box");
  const tooltipText = document.getElementById("tooltip-text");
  const p = payload.plot;
  const fullEnd = Math.max(payload.max_x, 1);
  let start = 0;
  let end = fullEnd;
  let dragX = null;
  let dragStart = 0;

  function localPoint(event) {
    const point = root.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(root.getScreenCTM().inverse());
  }
  function clampWindow(nextStart, nextEnd) {
    const width = nextEnd - nextStart;
    if (nextStart < 0) { nextEnd -= nextStart; nextStart = 0; }
    if (nextEnd > fullEnd) { nextStart -= nextEnd - fullEnd; nextEnd = fullEnd; }
    start = Math.max(0, nextStart);
    end = Math.min(fullEnd, Math.max(start + Math.min(width, fullEnd), nextEnd));
  }
  function updateView() {
    viewport.setAttribute("viewBox", `${start} 0 ${Math.max(end - start, 0.001)} ${p.height}`);
    for (let index = 0; index < 5; index += 1) {
      const value = start + (end - start) * index / 4;
      document.getElementById(`x-tick-${index}`).textContent = value.toFixed(value >= 100 ? 0 : 2);
    }
  }
  function nearestPoint(points, target) {
    let best = null;
    for (const point of points) {
      if (best === null || Math.abs(point[0] - target) < Math.abs(best[0] - target)) best = point;
    }
    return best;
  }
  function showCursor(event) {
    if (dragX !== null) return;
    const point = localPoint(event);
    const fraction = Math.max(0, Math.min(1, (point.x - p.left) / p.width));
    const target = start + fraction * (end - start);
    const first = payload.series.length ? nearestPoint(payload.series[0].points, target) : null;
    if (!first) return;
    const selectedX = first[0];
    const x = p.left + (selectedX - start) * p.width / Math.max(end - start, 0.001);
    cursor.setAttribute("x1", x); cursor.setAttribute("x2", x); cursor.hidden = false;
    const label = payload.labels[selectedX] || String(selectedX);
    const values = payload.series.map(item => {
      const selected = nearestPoint(item.points, selectedX);
      return `${item.name}=${selected ? Number(selected[1]).toPrecision(5) : "n/a"}`;
    });
    const lines = [`sample ${selectedX} · ${label}`];
    for (let index = 0; index < values.length; index += 3) lines.push(values.slice(index, index + 3).join(" · "));
    tooltipText.replaceChildren();
    lines.forEach((line, index) => {
      const span = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
      span.setAttribute("x", "9"); span.setAttribute("dy", index === 0 ? "16" : "17"); span.textContent = line;
      tooltipText.appendChild(span);
    });
    const boxWidth = Math.min(610, Math.max(190, Math.max(...lines.map(line => line.length)) * 7.1));
    const boxHeight = 9 + lines.length * 17;
    tooltipBox.setAttribute("width", boxWidth); tooltipBox.setAttribute("height", boxHeight);
    const tooltipX = Math.min(p.left + p.width - boxWidth, Math.max(p.left, x + 10));
    tooltip.setAttribute("transform", `translate(${tooltipX} ${p.top + 8})`); tooltip.hidden = false;
  }
  overlay.addEventListener("pointermove", event => {
    if (dragX !== null) {
      const point = localPoint(event);
      const shift = (dragX - point.x) * (end - start) / p.width;
      clampWindow(dragStart + shift, dragStart + shift + (end - start)); updateView();
    } else showCursor(event);
  });
  overlay.addEventListener("pointerleave", () => { if (dragX === null) { cursor.hidden = true; tooltip.hidden = true; } });
  overlay.addEventListener("pointerdown", event => {
    dragX = localPoint(event).x; dragStart = start; overlay.setPointerCapture(event.pointerId); overlay.classList.add("dragging");
  });
  overlay.addEventListener("pointerup", event => {
    dragX = null; overlay.releasePointerCapture(event.pointerId); overlay.classList.remove("dragging"); showCursor(event);
  });
  overlay.addEventListener("wheel", event => {
    event.preventDefault();
    const point = localPoint(event);
    const fraction = Math.max(0, Math.min(1, (point.x - p.left) / p.width));
    const width = end - start;
    const nextWidth = Math.max(fullEnd / 200, Math.min(fullEnd, width * (event.deltaY > 0 ? 1.25 : 0.8)));
    const anchor = start + fraction * width;
    clampWindow(anchor - fraction * nextWidth, anchor + (1 - fraction) * nextWidth); updateView(); showCursor(event);
  }, {passive: false});
  overlay.addEventListener("dblclick", () => { start = 0; end = fullEnd; updateView(); cursor.hidden = true; tooltip.hidden = true; });
  updateView();
})();
]]></script>"""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="chart-title chart-description">'
        '<style>text{font-family:system-ui,sans-serif;fill:#172033}.grid{stroke:#e4e7ec;stroke-width:1}'
        '.axis{stroke:#667085;stroke-width:1.25}.tick-label,.legend-label{font-size:12px}.axis-label{font-size:13px;font-weight:600}'
        '.hint{font-size:11px;fill:#667085}.empty{font-size:14px;fill:#667085}.interaction{fill:transparent;cursor:crosshair}'
        '.interaction.dragging{cursor:grabbing}.cursor{stroke:#344054;stroke-width:1;stroke-dasharray:4 3}'
        '.tooltip-box{fill:#101828;fill-opacity:.94;stroke:#fff}.tooltip-text{fill:#fff;font-size:12px}</style>'
        '<rect width="100%" height="100%" fill="#fff"/>'
        f'<title id="chart-title">{html.escape(chart_id)}</title>'
        '<desc id="chart-description">Interactive offline chart. Move the cursor for values, use the wheel to zoom, drag to pan, and double-click to reset.</desc>'
        f'<text x="24" y="28" font-size="17" font-weight="600">{html.escape(chart_id)}</text>'
        f'<text x="{width - 24}" y="27" text-anchor="end" class="hint">cursor: values · wheel: zoom · drag: pan · double-click: reset</text>'
        f'{"".join(legend)}{"".join(y_ticks)}{"".join(x_ticks)}'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>'
        f'<text x="{left + plot_width / 2}" y="{height - 17}" text-anchor="middle" class="axis-label">{html.escape(axes["x"])}</text>'
        f'<text transform="translate(19 {top + plot_height / 2}) rotate(-90)" text-anchor="middle" class="axis-label">{html.escape(axes["y"])}</text>'
        f'<svg id="viewport" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        f'viewBox="0 0 {max_x} {plot_height}" preserveAspectRatio="none" overflow="hidden">{"".join(paths)}</svg>{empty}'
        f'<rect id="interaction" class="interaction" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}"/>'
        f'<line id="cursor" class="cursor" y1="{top}" y2="{top + plot_height}" hidden="true"/>'
        '<g id="tooltip" hidden="true"><rect id="tooltip-box" class="tooltip-box" rx="4"/>'
        '<text id="tooltip-text" class="tooltip-text"/></g>'
        f'<script id="chart-data" type="application/json">{payload}</script>{script}</svg>\n'
    )


def _render_html(template_name: str, model, profile, charts, *, task=None, task_links=None) -> str:
    try:
        from jinja2 import DictLoader, Environment, PackageLoader, StrictUndefined, select_autoescape
    except ImportError as exc:
        raise ReportingRendererError("Jinja2 is required by the offline report renderer") from exc
    if profile.local_template_root is None:
        loader = PackageLoader("ovlab_runner", "report_assets/templates")
    else:
        loader = DictLoader({
            "benchmark/run-v1.html": (profile.local_template_root / "run-v1.html").read_text(encoding="utf-8"),
            "benchmark/task-v1.html": (profile.local_template_root / "task-v1.html").read_text(encoding="utf-8"),
        })
    environment = Environment(
        loader=loader,
        autoescape=select_autoescape(enabled_extensions=("html",), default_for_string=True),
        undefined=StrictUndefined,
    )
    try:
        template = environment.get_template(template_name)
        return template.render(model=model, profile=profile.document(), charts=charts, task=task, task_links=task_links or {})
    except Exception as exc:
        raise ReportingRendererError(f"offline HTML renderer failed for {template_name}: {exc}") from exc


class DerivedReportEngine:
    """Build and verify deterministic report bundles without writing to runs/."""

    def __init__(self, runs_root, derived_root, profile: ReportProfile | None = None):
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.derived_root = Path(derived_root).expanduser().resolve()
        display = os.environ.get("OVLAB_DERIVED_DISPLAY_ROOT")
        self.display_root = Path(display).expanduser() if display else self.derived_root
        self.profile = profile or builtin_profile()

    def _display_path(self, target: Path) -> str:
        return str(self.display_root / target.relative_to(self.derived_root))

    def resolve_run(self, run: str | Path) -> Path:
        candidate = Path(run)
        if candidate.is_absolute():
            return _inside(candidate, self.runs_root, "run")
        if len(candidate.parts) != 1 or candidate.name in {"", ".", ".."}:
            raise ArtifactError("run must be a run ID or a path within the configured runs root")
        return _inside(self.runs_root / run_key(candidate.name), self.runs_root, "run")

    def generate(self, run: str | Path, *, task_id: str | None = None) -> dict[str, object]:
        started_ns = time.perf_counter_ns()
        source = self.resolve_run(run)
        if not source.is_dir():
            raise ReportingSourceUnavailableError(f"canonical run is unavailable: {source.name}")
        model = build_report_model(source, scope_task_id=task_id)
        if model["completeness"] == "complete":
            verify_run(source)
        canonical_inputs = _canonical_inputs(source, safe_key(task_id) if task_id else None)
        identity_inputs = {
            "canonical_inputs": list(canonical_inputs),
            "profile": self.profile.document(),
            "template_bundle_sha256": _template_digest(self.profile),
            "report_schema": REPORT_SCHEMA,
            "manifest_schema": REPORT_MANIFEST_SCHEMA,
            "renderer": {"id": REPORT_RENDERER_ID, "version": REPORT_RENDERER_VERSION},
            "chart_builders": dict(sorted(CHART_BUILDERS.items())),
            "scope": model["scope"],
            "scope_task_id": task_id,
        }
        build_id = _digest_value(identity_inputs)[:20]
        run_id = str(model["run"]["run_id"])
        profile_root = _inside(self.derived_root / run_key(run_id) / self.profile.identifier, self.derived_root, "derived profile")
        target = profile_root / build_id
        if target.exists():
            result = self.verify(run_id, build_id=build_id)
            return {**result, "reused": True, "output": self._display_path(target)}
        stage = profile_root / f".{build_id}.{uuid.uuid4().hex}.partial"
        stage.mkdir(parents=True, exist_ok=False)
        try:
            assets = stage / "assets"
            charts_dir = assets / "charts"
            charts_dir.mkdir(parents=True)
            style = (
                (self.profile.local_template_root / "style.css").read_text(encoding="utf-8")
                if self.profile.local_template_root is not None else
                importlib.resources.files("ovlab_runner").joinpath("report_assets/static/style.css").read_text(encoding="utf-8")
            )
            (assets / "style.css").write_text(style, encoding="utf-8")
            chart_rows = []
            chart_map = {}
            for chart in self.profile.charts:
                relative = f"assets/charts/{chart['id']}.svg"
                svg = _chart_svg(chart["id"], chart["builder"], model)
                (stage / relative).write_text(svg, encoding="utf-8")
                row = {
                    **chart,
                    "builder_version": CHART_BUILDERS[chart["builder"]],
                    "path": relative,
                    "axes": _chart_axes(chart["builder"]),
                    "statistics": _chart_statistics(chart["builder"], model),
                    "interaction": {
                        "cursor": "nearest displayed canonical sample and values",
                        "zoom": "pointer-centered horizontal wheel zoom",
                        "pan": "horizontal pointer drag",
                        "reset": "double-click",
                        "runtime": "self-contained SVG; no network dependency",
                    },
                    "canonical_sources": (
                        ["tasks/*/episodes/*/trace.json:executed_actions.applied_action"]
                        if chart["builder"] == "action_timeseries" else
                        ["tasks/*/episodes/*/trace.json:policy_predictions.inference_duration_ns"]
                        if chart["builder"] == "latency_distribution" else
                        ["tasks/*/episodes/*/trace.json:policy_predictions.metadata.runtime.performance.cuda_memory_after"]
                        if chart["builder"] == "vram_timeseries" else
                        ["tasks/*/episodes/*/trace.json:policy_predictions.metadata.runtime.performance.estimated_compute"]
                        if chart["builder"] == "estimated_compute_timeseries" else
                        ["tasks/*/episodes/*/trace.json:terminal_status"]
                    ),
                    "transformation": "deterministic ordered sampling; at most 400 points",
                }
                chart_rows.append(row)
                chart_map[chart["id"]] = relative
            model["charts"] = chart_rows
            model["build"] = {
                "derived_build_id": build_id,
                "profile_id": self.profile.identifier,
                "renderer": f"{REPORT_RENDERER_ID}@{REPORT_RENDERER_VERSION}",
            }
            _atomic_json(stage / "report.json", model)
            task_links = {}
            for task in model["tasks"]:
                task_key = task["task_key"]
                task_dir = stage / "tasks" / task_key
                task_dir.mkdir(parents=True)
                task_chart_map = {key: f"../../{value}" for key, value in chart_map.items()}
                for episode in task["episodes"]:
                    video = episode["video"]
                    if video.get("status") == "available" and video.get("path"):
                        canonical_video = source / episode["canonical_episode_path"] / str(video["path"])
                        _inside(canonical_video, self.runs_root, "canonical video")
                        if not canonical_video.is_file():
                            raise ArtifactError(f"referenced canonical video is missing: {canonical_video}")
                        episode["video_href"] = os.path.relpath(canonical_video, task_dir)
                    else:
                        episode["video_href"] = None
                (task_dir / "index.html").write_text(
                    _render_html("benchmark/task-v1.html", model, self.profile, task_chart_map, task=task),
                    encoding="utf-8",
                )
                task_links[task["task_id"]] = f"tasks/{task_key}/index.html"
            (stage / "index.html").write_text(
                _render_html("benchmark/run-v1.html", model, self.profile, chart_map, task_links=task_links),
                encoding="utf-8",
            )
            file_rows = []
            for path in sorted(item for item in stage.rglob("*") if item.is_file()):
                relative = str(path.relative_to(stage))
                file_rows.append({"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
            manifest = {
                "schema_version": REPORT_MANIFEST_SCHEMA,
                "derived_build_id": build_id,
                "profile_id": self.profile.identifier,
                "run_id": run_id,
                "scope": model["scope"],
                "scope_task_id": task_id,
                "status": "complete" if model["completeness"] == "complete" else "partial",
                "identity_inputs": identity_inputs,
                "files": file_rows,
                "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "postprocessing_duration_ns": time.perf_counter_ns() - started_ns,
            }
            manifest["manifest_payload_sha256"] = _digest_value(manifest)
            _atomic_json(stage / "manifest.json", manifest)
            stage.rename(target)
            _atomic_json(profile_root / "latest.json", {
                "schema_version": "ovlab.report-latest/v1", "derived_build_id": build_id,
                "status": manifest["status"], "scope": model["scope"], "scope_task_id": task_id,
            })
            run_report_root = self.derived_root / run_key(run_id)
            finalize_managed_tree(run_report_root)
            finalize_managed_directory(run_report_root.parent)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        result = self.verify(run_id, build_id=build_id)
        return {**result, "reused": False, "output": self._display_path(target)}

    def verify(self, run: str | Path, *, build_id: str | None = None) -> dict[str, object]:
        source = self.resolve_run(run)
        run_id = _json(source / "manifest.started.json").get("run_id")
        profile_root = _inside(self.derived_root / run_key(str(run_id)) / self.profile.identifier, self.derived_root, "derived profile")
        if build_id is None:
            latest = _json(profile_root / "latest.json")
            build_id = latest.get("derived_build_id")
        if not isinstance(build_id, str) or not _SAFE_ID.fullmatch(build_id):
            raise ArtifactError("derived build ID is invalid")
        target = _inside(profile_root / build_id, profile_root, "derived build")
        manifest = _json(target / "manifest.json")
        if manifest.get("schema_version") != REPORT_MANIFEST_SCHEMA:
            raise ArtifactError("unsupported report manifest schema")
        recorded_manifest_digest = manifest.pop("manifest_payload_sha256", None)
        if recorded_manifest_digest != _digest_value(manifest):
            raise ArtifactError("report manifest payload checksum mismatch")
        manifest["manifest_payload_sha256"] = recorded_manifest_digest
        if manifest.get("status") == "complete":
            verify_run(source)
        if manifest.get("run_id") != run_id or manifest.get("profile_id") != self.profile.identifier:
            raise ArtifactError("report manifest source/profile identity mismatch")
        expected_id = _digest_value(manifest.get("identity_inputs"))[:20]
        if expected_id != build_id or manifest.get("derived_build_id") != build_id:
            raise ArtifactError("derived build identity mismatch")
        current_inputs = _canonical_inputs(source, safe_key(manifest["scope_task_id"]) if manifest.get("scope_task_id") else None)
        if list(current_inputs) != manifest["identity_inputs"].get("canonical_inputs"):
            raise ArtifactError("canonical report inputs no longer match their recorded checksums")
        expected_paths = {"manifest.json"}
        for row in manifest.get("files", []):
            relative = row.get("path")
            if not isinstance(relative, str):
                raise ArtifactError("report manifest contains an invalid file path")
            path = _inside(target / relative, target, "report file")
            if not path.is_file() or path.stat().st_size != row.get("size_bytes") or _sha256(path) != row.get("sha256"):
                raise ArtifactError(f"derived report file checksum mismatch: {relative}")
            expected_paths.add(relative)
        actual_paths = {str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()}
        if actual_paths != expected_paths:
            raise ArtifactError("derived report contains missing or unexpected files")
        report = _json(target / "report.json")
        if report.get("schema_version") != REPORT_SCHEMA or report.get("build", {}).get("derived_build_id") != build_id:
            raise ArtifactError("report JSON schema or build identity mismatch")
        for html in [target / "index.html", *target.glob("tasks/*/index.html")]:
            content = html.read_text(encoding="utf-8")
            if re.search(r"(?:https?:)?//", content, re.IGNORECASE):
                raise ArtifactError(f"offline report contains an external asset/link: {html.relative_to(target)}")
            for href in re.findall(r'(?:href|src|data)="([^"]+)"', content):
                if href.startswith("#"):
                    continue
                linked = (html.parent / href.split("#", 1)[0]).resolve()
                if not (linked.is_relative_to(target) or linked.is_relative_to(self.runs_root)) or not linked.exists():
                    raise ArtifactError(f"report contains an invalid internal link: {href}")
        return {
            "schema_version": REPORT_MANIFEST_SCHEMA,
            "run_id": run_id,
            "profile_id": self.profile.identifier,
            "derived_build_id": build_id,
            "status": manifest.get("status"),
            "verified_file_count": len(manifest.get("files", [])),
            "integrity": "verified",
            "output": self._display_path(target),
            "canonical_run_modified": False,
        }

    def record_failure(self, run_id, scope: str, exc: BaseException) -> None:
        root = _inside(self.derived_root / run_key(str(run_id)) / self.profile.identifier / "operational-failures", self.derived_root, "report failure")
        _atomic_json(root / f"{time.time_ns()}-{uuid.uuid4().hex[:8]}.json", {
            "schema_version": "ovlab.report-operational-failure/v1",
            "scope": scope,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "canonical_status_authoritative": True,
        })


class AutomaticDerivedReporter:
    """ExperimentRunner postprocessor that isolates reporting from benchmark state."""

    def __init__(
        self, engine: DerivedReportEngine | None, *, on_task_finalize=True, on_run_finalize=True,
        isolated_export_engine=None,
    ):
        self.engine = engine
        self.on_task_finalize = bool(on_task_finalize)
        self.on_run_finalize = bool(on_run_finalize)
        self.isolated_export_engine = isolated_export_engine

    def task_finalized(self, run_id, task_id) -> None:
        if self.engine is not None and self.on_task_finalize:
            self.engine.generate(str(run_id))

    def run_finalized(self, run_id, status: str) -> None:
        if self.engine is not None and self.on_run_finalize:
            self.engine.generate(str(run_id))
        if status == "completed" and self.isolated_export_engine is not None:
            self.isolated_export_engine.generate_isolated(str(run_id))

    def record_failure(self, run_id, scope: str, exc: BaseException) -> None:
        if self.engine is not None:
            self.engine.record_failure(run_id, scope, exc)
