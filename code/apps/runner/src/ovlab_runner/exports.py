"""Readable isolated and grouped exports from immutable canonical runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

from .artifacts.codec import TraceCodec
from .artifacts.layout import run_key, safe_key
from .derived import (
    _action_component_names,
    _canonical_bytes,
    _descriptive_statistics,
    _inside,
    _json,
    _sha256,
    build_report_model,
)
from .errors import ArtifactError
from .inspection import verify_run
from .permissions import finalize_managed_directory, finalize_managed_tree


EXPORT_METADATA_SCHEMA = "ovlab.export-metadata/v2"
EXPORT_ENGINE_VERSION = "2.0.0"
ISOLATED_TEMPLATE = "isolated-default-v1"
GROUPED_TEMPLATE = "grouped-default-v1"
LEGACY_EXPORT_SPEC_SCHEMA = "ovlab.export-spec/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FIGURE_FORMATS = ("png", "pdf")


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(_canonical_bytes(value))
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _portable_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ArtifactError(f"{label} must be a portable identifier")
    return value


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        values = {}
        for key in columns:
            value = row.get(key)
            if value is None:
                values[key] = ""
            elif isinstance(value, float):
                values[key] = format(value, ".12g")
            elif isinstance(value, (dict, list, tuple)):
                values[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            else:
                values[key] = str(value)
        writer.writerow(values)
    return stream.getvalue().encode("utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_csv_bytes(rows))


def _plot_backend():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ArtifactError("the export figure backend requires matplotlib") from exc
    return plt


def _save_figure(figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in _FIGURE_FORMATS:
        path = base.with_suffix(f".{suffix}")
        metadata = (
            {"Creator": "OVLAB", "Producer": "OVLAB", "CreationDate": None, "ModDate": None}
            if suffix == "pdf"
            else {"Software": "OVLAB"}
        )
        figure.savefig(path, format=suffix, dpi=160, bbox_inches="tight", metadata=metadata)
        outputs.append(path)
    _plot_backend().close(figure)
    return outputs


def _statistics_rows(category: str, series: dict[str, list[float]], **identity) -> list[dict[str, object]]:
    rows = []
    for name, values in series.items():
        rows.append({
            **identity,
            "category": category,
            "series": name,
            **_descriptive_statistics(values),
        })
    return rows


def _source_configuration(run_path: Path) -> dict[str, object]:
    rows = []
    for name in ("source_config.yaml", "resolved_config.yaml"):
        path = run_path / name
        if path.is_file():
            rows.append({"path": name, "sha256": _sha256(path)})
    return {"canonical_files": rows}


def _first_episode_metadata(run_path: Path) -> dict[str, object]:
    for path in sorted((run_path / "tasks").glob("*/episodes/*/metadata.json")):
        return _json(path)
    return {}


def _model_key(record: dict[str, object]) -> str:
    identity = {
        "model": record["identity"].get("model"),
        "checkpoint": record["identity"].get("checkpoint"),
    }
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _source_record(run_path: Path) -> dict[str, object]:
    verification = verify_run(run_path)
    model = build_report_model(run_path)
    metadata = _first_episode_metadata(run_path)
    started = _json(run_path / "manifest.started.json")
    episode_model = metadata.get("model", {}) if isinstance(metadata, dict) else {}
    episode_checkpoint = metadata.get("checkpoint", {}) if isinstance(metadata, dict) else {}
    policy = model.get("policy") or {}
    identity = {
        "experiment": metadata.get("experiment") or {
            "id": model["run"].get("experiment"),
            "name": model["run"].get("experiment_name"),
        },
        "model": episode_model or {
            "name": policy.get("name"),
            "version": policy.get("version"),
            "method": policy.get("method_descriptor"),
        },
        "checkpoint": episode_checkpoint or policy.get("model_identity") or {},
    }
    integrity = run_path / "integrity.json"
    return {
        "run_id": str(model["run"]["run_id"]),
        "path": run_path,
        "integrity_sha256": _sha256(integrity),
        "scientific_config_hash": model["run"].get("scientific_config_hash"),
        "status": model["run"].get("status"),
        "suite": _suite(model),
        "model": model,
        "identity": identity,
        "configuration": _source_configuration(run_path),
        "created_at_utc": started.get("created_at_utc"),
        "verification": verification,
    }


def _suite(model: dict[str, object]) -> str | None:
    for task in model["tasks"]:
        task_id = str(task["task_id"])
        for prefix, suite in (
            ("libero/10/", "libero_10"),
            ("libero/spatial/", "libero_spatial"),
            ("libero/object/", "libero_object"),
            ("libero/goal/", "libero_goal"),
        ):
            if task_id.startswith(prefix):
                return suite
    return None


def _all_records(runs_root: Path) -> list[dict[str, object]]:
    records = []
    if not runs_root.is_dir():
        raise ArtifactError(f"runs root is unavailable: {runs_root}")
    for path in sorted(item for item in runs_root.iterdir() if item.is_dir()):
        if not (path / "manifest.started.json").is_file():
            continue
        try:
            records.append(_source_record(path))
        except Exception as exc:
            raise ArtifactError(f"candidate run failed canonical verification: {path.name}: {exc}") from exc
    return records


def _record(runs_root: Path, run_id: str) -> dict[str, object]:
    if not isinstance(run_id, str) or Path(run_id).name != run_id or not run_id:
        raise ArtifactError(f"invalid run ID: {run_id}")
    path = _inside(runs_root / run_key(run_id), runs_root, "selected run")
    if not path.is_dir():
        raise ArtifactError(f"selected run is unavailable: {run_id}")
    return _source_record(path)


def _episode_data(record: dict[str, object], selected_episode_id: str | None = None) -> list[dict[str, object]]:
    codec = TraceCodec()
    rows = []
    for task in record["model"]["tasks"]:
        for episode in task["episodes"]:
            episode_id = str(episode["episode_id"])
            if selected_episode_id is not None and episode_id != selected_episode_id:
                continue
            episode_path = record["path"] / episode["canonical_episode_path"]
            trace = codec.decode(episode_path)
            actions = [list(map(float, item.applied_action)) for item in trace.executed_actions]
            requested = [list(map(float, item.requested_action)) for item in trace.executed_actions]
            inference = [item.inference_duration_ns / 1_000_000 for item in trace.policy_predictions]
            rpc = [item.metadata.get("rpc_round_trip_duration_ns") for item in trace.policy_predictions]
            rpc = [float(value) / 1_000_000 for value in rpc if isinstance(value, int)]
            closed = [item.metadata.get("closed_loop_step_duration_ns") for item in trace.executed_actions]
            closed = [float(value) / 1_000_000 for value in closed if isinstance(value, int)]
            positions = []
            for observation in trace.observations:
                for proprioception in observation.proprioception:
                    values = np.asarray(proprioception.values).reshape(-1)
                    if proprioception.name in {"robot.eef.position", "robot.proprioception"} and values.size >= 3:
                        positions.append([float(value) for value in values[:3]])
                        break
            rows.append({
                "run_id": record["run_id"],
                "task_id": str(task["task_id"]),
                "episode_id": episode_id,
                "episode_key": safe_key(episode_id),
                "rollout_index": episode["rollout_index"],
                "seed": episode["seed"],
                "terminal_status": episode["terminal_status"],
                "success": bool(episode["success"]),
                "instruction": episode["instruction"],
                "actions": actions,
                "requested_actions": requested,
                "inference_ms": inference,
                "rpc_ms": rpc,
                "closed_loop_ms": closed,
                "eef_positions": positions,
                "metrics": episode["metrics"],
            })
    if selected_episode_id is not None and not rows:
        raise ArtifactError(f"episode is not present in canonical run: {selected_episode_id}")
    return rows


def _episode_statistics(episode: dict[str, object]) -> list[dict[str, object]]:
    actions = episode["actions"]
    components = _action_component_names(max((len(row) for row in actions), default=0))
    series = {
        name: [row[index] for row in actions if len(row) > index]
        for index, name in enumerate(components)
    }
    rows = _statistics_rows(
        "applied_action", series,
        scope="episode", run_id=episode["run_id"], task_id=episode["task_id"],
        episode_id=episode["episode_id"],
    )
    rows.extend(_statistics_rows(
        "timing_ms",
        {
            "model_inference": episode["inference_ms"],
            "rpc_round_trip": episode["rpc_ms"],
            "closed_loop_step": episode["closed_loop_ms"],
        },
        scope="episode", run_id=episode["run_id"], task_id=episode["task_id"],
        episode_id=episode["episode_id"],
    ))
    return rows


def _aggregate_statistics(
    episodes: list[dict[str, object]], *, scope: str, run_id: str | None = None,
    group_name: str | None = None,
) -> list[dict[str, object]]:
    dimensions = max((len(row) for episode in episodes for row in episode["actions"]), default=0)
    names = _action_component_names(dimensions)
    identity = {"scope": scope, "group": group_name, "run_id": run_id, "task_id": None, "episode_id": None}
    rows = _statistics_rows(
        "applied_action",
        {
            name: [
                row[index] for episode in episodes for row in episode["actions"]
                if len(row) > index
            ]
            for index, name in enumerate(names)
        },
        **identity,
    )
    rows.extend(_statistics_rows(
        "timing_ms",
        {
            "model_inference": [value for episode in episodes for value in episode["inference_ms"]],
            "rpc_round_trip": [value for episode in episodes for value in episode["rpc_ms"]],
            "closed_loop_step": [value for episode in episodes for value in episode["closed_loop_ms"]],
        },
        **identity,
    ))
    rows.extend(_statistics_rows(
        "episode",
        {
            "success_indicator": [float(episode["success"]) for episode in episodes],
            "executed_steps": [float(len(episode["actions"])) for episode in episodes],
        },
        **identity,
    ))
    return rows


def _episode_timeseries(episode: dict[str, object]) -> list[dict[str, object]]:
    count = max(
        len(episode["actions"]), len(episode["requested_actions"]), len(episode["inference_ms"]),
        len(episode["rpc_ms"]), len(episode["closed_loop_ms"]), len(episode["eef_positions"]),
    )
    rows = []
    component_names = _action_component_names(max((len(row) for row in episode["actions"]), default=0))
    for index in range(count):
        row = {"step": index}
        if index < len(episode["actions"]):
            row.update({f"applied_{name}": value for name, value in zip(component_names, episode["actions"][index])})
        if index < len(episode["requested_actions"]):
            row.update({f"requested_{name}": value for name, value in zip(component_names, episode["requested_actions"][index])})
        for key, values in (
            ("model_inference_ms", episode["inference_ms"]),
            ("rpc_round_trip_ms", episode["rpc_ms"]),
            ("closed_loop_step_ms", episode["closed_loop_ms"]),
        ):
            if index < len(values):
                row[key] = values[index]
        if index < len(episode["eef_positions"]):
            row.update(dict(zip(("eef_x_m", "eef_y_m", "eef_z_m"), episode["eef_positions"][index])))
        rows.append(row)
    return rows


def _episode_figures(episode: dict[str, object], root: Path) -> list[str]:
    plt = _plot_backend()
    omitted = []
    key = episode["episode_key"]
    actions = np.asarray(episode["actions"], dtype=np.float64)
    names = _action_component_names(actions.shape[1] if actions.ndim == 2 else 0)

    figure, axis = plt.subplots(figsize=(10, 5))
    if actions.size:
        for index, name in enumerate(names):
            axis.plot(actions[:, index], label=name, linewidth=1.1)
        axis.legend(ncol=min(4, len(names)), fontsize=8)
    axis.set(title="Applied actions over time", xlabel="Control step", ylabel="Normalized command")
    axis.grid(alpha=.25)
    _save_figure(figure, root / f"{key}__actions-over-time")

    figure, axis = plt.subplots(figsize=(9, 5))
    if actions.size:
        axis.boxplot([actions[:, index] for index in range(actions.shape[1])], tick_labels=names, showfliers=True)
    axis.set(title="Applied action distributions", xlabel="Action component", ylabel="Normalized command")
    axis.grid(axis="y", alpha=.25)
    _save_figure(figure, root / f"{key}__action-boxplots")

    timing = {
        "Model inference": episode["inference_ms"],
        "RPC round trip": episode["rpc_ms"],
        "Closed loop": episode["closed_loop_ms"],
    }
    figure, axis = plt.subplots(figsize=(10, 5))
    for label, values in timing.items():
        if values:
            axis.plot(values, label=label, linewidth=1.2)
    if any(timing.values()):
        axis.legend()
    axis.set(title="Execution timing over time", xlabel="Ordered sample", ylabel="Duration (ms)")
    axis.grid(alpha=.25)
    _save_figure(figure, root / f"{key}__timing-over-time")

    positions = np.asarray(episode["eef_positions"], dtype=np.float64)
    if positions.ndim == 2 and positions.shape[0] >= 2 and positions.shape[1] == 3:
        figure = plt.figure(figsize=(8, 7))
        axis = figure.add_subplot(111, projection="3d")
        axis.plot(positions[:, 0], positions[:, 1], positions[:, 2], linewidth=1.5)
        axis.scatter(*positions[0], marker="o", label="start")
        axis.scatter(*positions[-1], marker="x", label="end")
        axis.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="End-effector trajectory")
        axis.legend()
        _save_figure(figure, root / f"{key}__eef-trajectory-3d")
    else:
        omitted.append("eef-trajectory-3d: canonical end-effector positions unavailable")
    return omitted


def _outcome_value(model: dict[str, object]) -> float | None:
    outcome = model["outcome"]
    if outcome["presentation"] == "success_rate":
        return outcome["success_rate"]
    if outcome["presentation"] == "binary_success":
        return float(bool(outcome["success"]))
    return None


def _metric_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        for task in record["model"]["tasks"]:
            for metric in task["metrics"]:
                rows.append({
                    "run_id": record["run_id"],
                    "task_id": task["task_id"],
                    "metric_id": metric["metric_id"],
                    "metric_version": metric["metric_version"],
                    "metric_config_hash": metric["metric_config_hash"],
                    "status": metric["status"],
                    "value": metric["value"],
                    "unit": metric["unit"],
                    "n": metric["sample_count"],
                })
    return rows


def _validate_metric_compatibility(records: list[dict[str, object]]) -> None:
    identities: dict[str, set[tuple[object, object, object]]] = {}
    for row in _metric_rows(records):
        identities.setdefault(str(row["metric_id"]), set()).add((
            row["metric_version"], row["metric_config_hash"], row["unit"],
        ))
    incompatible = sorted(metric_id for metric_id, values in identities.items() if len(values) > 1)
    if incompatible:
        raise ArtifactError(
            f"grouped export contains incompatible metric schemas/units: {incompatible}"
        )


def _episode_summary(episodes: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{
        "run_id": item["run_id"], "task_id": item["task_id"], "episode_id": item["episode_id"],
        "rollout_index": item["rollout_index"], "seed": item["seed"],
        "instruction": item["instruction"], "terminal_status": item["terminal_status"],
        "success": item["success"], "executed_steps": len(item["actions"]),
        "prediction_count": len(item["inference_ms"]),
    } for item in episodes]


def _run_overview_figures(record: dict[str, object], episodes: list[dict[str, object]], root: Path) -> list[str]:
    plt = _plot_backend()
    omitted = []
    task_groups: dict[str, list[dict[str, object]]] = {}
    for episode in episodes:
        task_groups.setdefault(episode["task_id"], []).append(episode)

    labels = list(task_groups)
    values = [sum(item["success"] for item in task_groups[label]) / len(task_groups[label]) for label in labels]
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * .8), 5))
    axis.bar(range(len(labels)), values, color="#155eef")
    axis.set(title="Success rate by task", xlabel="Task", ylabel="Success rate", ylim=(0, 1))
    axis.set_xticks(range(len(labels)), [label[-18:] for label in labels], rotation=35, ha="right")
    axis.grid(axis="y", alpha=.25)
    _save_figure(figure, root / "success-by-task")

    dimensions = max((len(row) for episode in episodes for row in episode["actions"]), default=0)
    names = _action_component_names(dimensions)
    component_values = [
        [row[index] for episode in episodes for row in episode["actions"] if len(row) > index]
        for index in range(dimensions)
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    if component_values:
        axis.boxplot(component_values, tick_labels=names, showfliers=True)
    axis.set(title="Action distributions across the run", xlabel="Action component", ylabel="Normalized command")
    axis.grid(axis="y", alpha=.25)
    _save_figure(figure, root / "action-boxplots")

    figure, axis = plt.subplots(figsize=(max(8, len(episodes) * .7), 5))
    latency = [item["inference_ms"] for item in episodes if item["inference_ms"]]
    latency_labels = [item["episode_id"][-10:] for item in episodes if item["inference_ms"]]
    if latency:
        axis.boxplot(latency, tick_labels=latency_labels, showfliers=True)
    axis.set(title="Inference latency by episode", xlabel="Episode", ylabel="Duration (ms)")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=.25)
    _save_figure(figure, root / "inference-latency-boxplots")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(range(len(episodes)), [len(item["actions"]) for item in episodes], color="#7a5af8")
    axis.set(title="Episode lengths", xlabel="Episode", ylabel="Executed control steps")
    axis.set_xticks(range(len(episodes)), [item["episode_id"][-10:] for item in episodes], rotation=35, ha="right")
    axis.grid(axis="y", alpha=.25)
    _save_figure(figure, root / "episode-lengths")

    trajectories = [item for item in episodes if len(item["eef_positions"]) >= 2]
    if trajectories:
        figure = plt.figure(figsize=(8, 7))
        axis = figure.add_subplot(111, projection="3d")
        for item in trajectories:
            positions = np.asarray(item["eef_positions"])
            axis.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=item["episode_id"][-8:])
        axis.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="End-effector trajectories")
        axis.legend(fontsize=7)
        _save_figure(figure, root / "eef-trajectories-3d")
    else:
        omitted.append("eef-trajectories-3d: canonical end-effector positions unavailable")
    return omitted


def _wilson(successes: int, count: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if count == 0:
        return None, None
    proportion = successes / count
    denominator = 1 + z * z / count
    centre = (proportion + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _run_summary(records: list[dict[str, object]], episode_map: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        episodes = episode_map[record["run_id"]]
        successes = sum(item["success"] for item in episodes)
        lower, upper = _wilson(successes, len(episodes))
        latency = [value for item in episodes for value in item["inference_ms"]]
        rows.append({
            "run_id": record["run_id"],
            "experiment": record["identity"]["experiment"].get("name") or record["identity"]["experiment"].get("id"),
            "model": record["identity"]["model"].get("name"),
            "checkpoint": record["identity"]["checkpoint"].get("checkpoint_id") or record["identity"]["checkpoint"].get("repository"),
            "suite": record["suite"],
            "episodes": len(episodes), "successes": successes,
            "success_rate": successes / len(episodes) if episodes else None,
            "success_rate_ci95_low": lower, "success_rate_ci95_high": upper,
            "median_inference_ms": float(np.median(latency)) if latency else None,
            "mean_episode_steps": float(np.mean([len(item["actions"]) for item in episodes])) if episodes else None,
        })
    return rows


def _group_figures(records, episode_map, run_rows, root: Path) -> list[str]:
    plt = _plot_backend()
    omitted = []
    labels = [row["run_id"][-14:] for row in run_rows]
    rates = [row["success_rate"] if row["success_rate"] is not None else 0.0 for row in run_rows]
    lower = [rate - (row["success_rate_ci95_low"] or rate) for rate, row in zip(rates, run_rows)]
    upper = [(row["success_rate_ci95_high"] or rate) - rate for rate, row in zip(rates, run_rows)]
    figure, axis = plt.subplots(figsize=(max(8, len(records) * 1.1), 5))
    axis.bar(range(len(records)), rates, yerr=[lower, upper], capsize=4, color="#155eef")
    axis.set(title="Success comparison with 95% Wilson intervals", xlabel="Run", ylabel="Success rate", ylim=(0, 1))
    axis.set_xticks(range(len(records)), labels, rotation=35, ha="right")
    axis.grid(axis="y", alpha=.25)
    _save_figure(figure, root / "success-comparison")

    task_ids = sorted({episode["task_id"] for episodes in episode_map.values() for episode in episodes})
    matrix = np.full((len(records), len(task_ids)), np.nan)
    for row_index, record in enumerate(records):
        for column_index, task_id in enumerate(task_ids):
            selected = [item for item in episode_map[record["run_id"]] if item["task_id"] == task_id]
            if selected:
                matrix[row_index, column_index] = sum(item["success"] for item in selected) / len(selected)
    figure, axis = plt.subplots(figsize=(max(8, len(task_ids) * .8), max(4, len(records) * .55)))
    image = axis.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axis.set(title="Task success heatmap", xlabel="Task", ylabel="Run")
    axis.set_xticks(range(len(task_ids)), [value[-15:] for value in task_ids], rotation=35, ha="right")
    axis.set_yticks(range(len(records)), labels)
    figure.colorbar(image, ax=axis, label="Success rate")
    _save_figure(figure, root / "task-success-heatmap")

    latency = [[value for item in episode_map[record["run_id"]] for value in item["inference_ms"]] for record in records]
    available = [(label, values) for label, values in zip(labels, latency) if values]
    figure, axis = plt.subplots(figsize=(max(8, len(available) * 1.1), 5))
    if available:
        axis.boxplot([values for _, values in available], tick_labels=[label for label, _ in available], showfliers=True)
    axis.set(title="Inference latency comparison", xlabel="Run", ylabel="Duration (ms)")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=.25)
    _save_figure(figure, root / "inference-latency-boxplots")

    figure, axis = plt.subplots(figsize=(8, 5))
    for label, values in available:
        ordered = np.sort(values)
        axis.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post", label=label)
    if available:
        axis.legend(fontsize=8)
    axis.set(title="Inference latency ECDF", xlabel="Duration (ms)", ylabel="Cumulative probability", ylim=(0, 1))
    axis.grid(alpha=.25)
    _save_figure(figure, root / "inference-latency-ecdf")

    figure, axis = plt.subplots(figsize=(8, 5))
    for row in run_rows:
        if row["median_inference_ms"] is not None and row["success_rate"] is not None:
            axis.scatter(row["median_inference_ms"], row["success_rate"], label=row["run_id"][-14:])
    if any(row["median_inference_ms"] is not None for row in run_rows):
        axis.legend(fontsize=8)
    axis.set(title="Success-latency trade-off", xlabel="Median inference latency (ms)", ylabel="Success rate", ylim=(0, 1))
    axis.grid(alpha=.25)
    _save_figure(figure, root / "success-latency-pareto")

    statuses = sorted({item["terminal_status"] for episodes in episode_map.values() for item in episodes})
    figure, axis = plt.subplots(figsize=(max(8, len(records) * 1.1), 5))
    bottom = np.zeros(len(records))
    for status in statuses:
        counts = [sum(item["terminal_status"] == status for item in episode_map[record["run_id"]]) for record in records]
        axis.bar(range(len(records)), counts, bottom=bottom, label=status)
        bottom += counts
    axis.set(title="Terminal outcome composition", xlabel="Run", ylabel="Episode count")
    axis.set_xticks(range(len(records)), labels, rotation=35, ha="right")
    axis.legend(fontsize=8)
    _save_figure(figure, root / "terminal-outcome-composition")

    trajectories = [
        (record["run_id"], item)
        for record in records for item in episode_map[record["run_id"]]
        if len(item["eef_positions"]) >= 2
    ]
    if trajectories:
        figure = plt.figure(figsize=(8, 7))
        axis = figure.add_subplot(111, projection="3d")
        for run_id, item in trajectories:
            values = np.asarray(item["eef_positions"])
            axis.plot(values[:, 0], values[:, 1], values[:, 2], alpha=.75, label=run_id[-8:])
        axis.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="End-effector trajectory comparison")
        _save_figure(figure, root / "eef-trajectories-3d")
    else:
        omitted.append("eef-trajectories-3d: canonical end-effector positions unavailable")
    return omitted


def _file_inventory(stage: Path) -> list[dict[str, object]]:
    return [{
        "path": str(path.relative_to(stage)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    } for path in sorted(item for item in stage.rglob("*") if item.is_file() and item.name != "metadata.json")]


def _publish(stage: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        stage.rename(target)
    else:
        backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.previous")
        target.rename(backup)
        try:
            stage.rename(target)
        except Exception:
            if not target.exists() and backup.exists():
                backup.rename(target)
            raise
        else:
            shutil.rmtree(backup, ignore_errors=True)
    finalize_managed_tree(target)
    finalize_managed_directory(target.parent)


def validate_export_spec(document) -> dict[str, object]:
    """Validate the legacy grouped-selection file without executable builders.

    Kept for compatibility with the first H.2 CLI. New callers should use the
    explicit ``export grouped`` command.
    """
    doc = json.loads(json.dumps(document))
    if not isinstance(doc, dict) or doc.get("schema_version") != LEGACY_EXPORT_SPEC_SCHEMA:
        raise ArtifactError(f"export spec schema_version must equal {LEGACY_EXPORT_SPEC_SCHEMA}")
    required = {"schema_version", "id", "selection"}
    if not required <= set(doc):
        raise ArtifactError("legacy export spec requires id and selection")
    if set(doc) - {"schema_version", "id", "selection", "group_by", "outputs", "template"}:
        raise ArtifactError("legacy export spec contains unsupported fields")
    _portable_id(doc["id"], "export id")
    selection = doc["selection"]
    if not isinstance(selection, dict) or set(selection) - {"suite", "run_ids", "filters"}:
        raise ArtifactError("legacy selection supports only suite, run_ids, and filters")
    run_ids = selection.get("run_ids", [])
    if not isinstance(run_ids, list) or any(not isinstance(value, str) or not value for value in run_ids):
        raise ArtifactError("selection.run_ids must be a list of run IDs")
    filters = selection.get("filters", {})
    if not isinstance(filters, dict) or set(filters) - {"status", "policy", "experiment"}:
        raise ArtifactError("selection.filters contains an unsupported filter")
    return doc


class ExportEngine:
    """Generate readable exports while preserving canonical runs as read-only sources."""

    def __init__(self, runs_root, exports_root):
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.exports_root = Path(exports_root).expanduser().resolve()
        display = os.environ.get("OVLAB_EXPORTS_DISPLAY_ROOT")
        self.display_root = Path(display).expanduser() if display else self.exports_root

    def _display_path(self, target: Path) -> str:
        return str(self.display_root / target.relative_to(self.exports_root))

    def generate_isolated(
        self, run_id: str, *, episode_id: str | None = None, template: str = ISOLATED_TEMPLATE,
    ) -> dict[str, object]:
        if template != ISOLATED_TEMPLATE:
            raise ArtifactError(f"unknown isolated export template: {template}")
        record = _record(self.runs_root, run_id)
        episodes = _episode_data(record, episode_id)
        target = _inside(self.exports_root / "isolated" / run_key(run_id), self.exports_root, "isolated export")
        stage = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
        stage.mkdir(parents=True)
        omitted = []
        try:
            episode_tables = stage / "episodes" / "tables"
            episode_figures = stage / "episodes" / "figures"
            for episode in episodes:
                key = episode["episode_key"]
                _write_csv(episode_tables / f"{key}__statistics.csv", _episode_statistics(episode))
                _write_csv(episode_tables / f"{key}__timeseries.csv", _episode_timeseries(episode))
                omitted.extend(f"{episode['episode_id']}: {value}" for value in _episode_figures(episode, episode_figures))

            if episode_id is None:
                overview_tables = stage / "overview" / "tables"
                overview_figures = stage / "overview" / "figures"
                _write_csv(overview_tables / "episode-summary.csv", _episode_summary(episodes))
                _write_csv(
                    overview_tables / "descriptive-statistics.csv",
                    _aggregate_statistics(episodes, scope="run", run_id=record["run_id"]),
                )
                _write_csv(overview_tables / "metric-summary.csv", _metric_rows([record]))
                omitted.extend(_run_overview_figures(record, episodes, overview_figures))

            metadata = {
                "schema_version": EXPORT_METADATA_SCHEMA,
                "export_type": "isolated",
                "source": {
                    "run": record["run_id"],
                    "episodes": [item["episode_id"] for item in episodes],
                    "scope": "episode" if episode_id is not None else "run",
                    "canonical_integrity_sha256": record["integrity_sha256"],
                },
                "experiment": record["identity"]["experiment"],
                "model": record["identity"]["model"],
                "checkpoint": record["identity"]["checkpoint"],
                "config": record["configuration"],
                "template": {"id": template, "engine_version": EXPORT_ENGINE_VERSION},
                "datetime": {"source_run_created_utc": record["created_at_utc"], "exported_at_utc": _utc_now()},
                "omitted_figures": omitted,
                "files": _file_inventory(stage),
            }
            _atomic_json(stage / "metadata.json", metadata)
            _publish(stage, target)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return {**self.verify("isolated", run_id), "reused": False}

    def _group_records(
        self, *, all_runs=False, run_ids=(), same_model_as: str | None = None, suite: str | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        modes = int(bool(all_runs)) + int(bool(run_ids)) + int(same_model_as is not None)
        if modes != 1:
            raise ArtifactError("grouped export requires exactly one selector: all runs, same model, or manual runs")
        if run_ids:
            records = [_record(self.runs_root, run_id) for run_id in sorted(set(run_ids))]
            selection = {"mode": "manual", "run_ids": [item["run_id"] for item in records]}
        else:
            records = _all_records(self.runs_root)
            if same_model_as is not None:
                reference = _record(self.runs_root, same_model_as)
                key = _model_key(reference)
                records = [record for record in records if _model_key(record) == key]
                selection = {"mode": "same-model", "reference_run": same_model_as, "model_key": key}
            else:
                selection = {"mode": "all-runs"}
        records = [record for record in records if record["status"] == "completed"]
        if suite is not None:
            records = [record for record in records if record["suite"] == suite]
            selection["suite"] = suite
        records.sort(key=lambda item: item["run_id"])
        if not records:
            raise ArtifactError("grouped export selection is empty")
        suites = sorted({record["suite"] for record in records})
        if len(suites) > 1:
            raise ArtifactError(f"grouped export spans incompatible benchmark suites: {suites}")
        return records, selection

    def generate_grouped(
        self, group_name: str, *, all_runs=False, run_ids=(), same_model_as=None,
        suite=None, template: str = GROUPED_TEMPLATE,
    ) -> dict[str, object]:
        _portable_id(group_name, "group name")
        if template != GROUPED_TEMPLATE:
            raise ArtifactError(f"unknown grouped export template: {template}")
        records, selection = self._group_records(
            all_runs=all_runs, run_ids=tuple(run_ids), same_model_as=same_model_as, suite=suite,
        )
        _validate_metric_compatibility(records)
        episode_map = {record["run_id"]: _episode_data(record) for record in records}
        run_rows = _run_summary(records, episode_map)
        target = _inside(self.exports_root / "grouped" / group_name, self.exports_root, "grouped export")
        stage = target.parent / f".{group_name}.{uuid.uuid4().hex}.partial"
        stage.mkdir(parents=True)
        try:
            tables, figures = stage / "tables", stage / "figures"
            _write_csv(tables / "run-summary.csv", run_rows)
            _write_csv(tables / "episode-summary.csv", [
                row for record in records for row in _episode_summary(episode_map[record["run_id"]])
            ])
            run_statistics = [
                row for record in records
                for row in _aggregate_statistics(
                    episode_map[record["run_id"]], scope="run", run_id=record["run_id"],
                )
            ]
            group_statistics = _aggregate_statistics(
                [episode for record in records for episode in episode_map[record["run_id"]]],
                scope="group", group_name=group_name,
            )
            _write_csv(tables / "descriptive-statistics.csv", run_statistics + group_statistics)
            _write_csv(tables / "metric-summary.csv", _metric_rows(records))
            omitted = _group_figures(records, episode_map, run_rows, figures)
            model_identities = []
            checkpoint_identities = []
            for record in records:
                if record["identity"]["model"] not in model_identities:
                    model_identities.append(record["identity"]["model"])
                if record["identity"]["checkpoint"] not in checkpoint_identities:
                    checkpoint_identities.append(record["identity"]["checkpoint"])
            metadata = {
                "schema_version": EXPORT_METADATA_SCHEMA,
                "export_type": "grouped",
                "source": {
                    "group": group_name,
                    "runs": [record["run_id"] for record in records],
                    "selection": selection,
                    "canonical_integrity": {
                        record["run_id"]: record["integrity_sha256"] for record in records
                    },
                },
                "experiment": [record["identity"]["experiment"] for record in records],
                "model": model_identities,
                "checkpoint": checkpoint_identities,
                "config": {
                    "selection": selection,
                    "sources": {record["run_id"]: record["configuration"] for record in records},
                },
                "template": {"id": template, "engine_version": EXPORT_ENGINE_VERSION},
                "datetime": {"exported_at_utc": _utc_now()},
                "omitted_figures": omitted,
                "files": _file_inventory(stage),
            }
            _atomic_json(stage / "metadata.json", metadata)
            _publish(stage, target)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return {**self.verify("grouped", group_name), "reused": False}

    def generate(self, spec_document) -> dict[str, object]:
        """Compatibility bridge: a legacy spec now produces a grouped export."""
        spec = validate_export_spec(spec_document)
        selection = spec["selection"]
        run_ids = tuple(selection.get("run_ids", ()))
        suite = selection.get("suite")
        if run_ids:
            return self.generate_grouped(spec["id"], run_ids=run_ids, suite=suite)
        filters = selection.get("filters", {})
        if set(filters) - {"status"} or filters.get("status", "completed") != "completed":
            raise ArtifactError("legacy metadata filters require the explicit grouped CLI")
        return self.generate_grouped(spec["id"], all_runs=True, suite=suite)

    def verify(self, kind: str, name: str | None = None, *, build_id=None) -> dict[str, object]:
        # Accept the former verify(export_id, build_id=...) call as grouped lookup.
        if name is None:
            name, kind = kind, "grouped"
        if kind not in {"isolated", "grouped"}:
            raise ArtifactError("export kind must be isolated or grouped")
        if kind == "grouped":
            key = _portable_id(name, "group name")
        else:
            key = run_key(name)
        target = _inside(self.exports_root / kind / key, self.exports_root, "export")
        metadata = _json(target / "metadata.json")
        if metadata.get("schema_version") != EXPORT_METADATA_SCHEMA or metadata.get("export_type") != kind:
            raise ArtifactError("unsupported export metadata schema or type")
        source = metadata.get("source", {})
        run_ids = [source.get("run")] if kind == "isolated" else source.get("runs", [])
        recorded = (
            {source.get("run"): source.get("canonical_integrity_sha256")}
            if kind == "isolated"
            else source.get("canonical_integrity", {})
        )
        for run_id in run_ids:
            try:
                record = _record(self.runs_root, run_id)
            except Exception as exc:
                raise ArtifactError(
                    f"canonical export source verification failed: {run_id}: {exc}"
                ) from exc
            if record["integrity_sha256"] != recorded.get(run_id):
                raise ArtifactError(f"canonical export source checksum mismatch: {run_id}")
        expected = {"metadata.json"}
        for item in metadata.get("files", []):
            path = _inside(target / item["path"], target, "export file")
            if not path.is_file() or path.stat().st_size != item["size_bytes"] or _sha256(path) != item["sha256"]:
                raise ArtifactError(f"export file checksum mismatch: {item['path']}")
            expected.add(item["path"])
        actual = {str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()}
        if actual != expected:
            raise ArtifactError("export contains missing or unexpected files")
        return {
            "schema_version": EXPORT_METADATA_SCHEMA,
            "export_type": kind,
            "name": name,
            "integrity": "verified",
            "verified_file_count": len(expected),
            "source_run_ids": run_ids,
            "output": self._display_path(target),
        }
