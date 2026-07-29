"""Deterministic reports, videos, and whole-run integrity inventories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .artifacts.codec import TraceCodec
from .errors import ArtifactError


REPORT_SCHEMA_VERSION = "ovlab-report/1.0.0"
VIDEO_SCHEMA_VERSION = "ovlab-video/1.0.0"
INTEGRITY_SCHEMA_VERSION = "ovlab-integrity/1.0.0"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value) -> None:
    if path.exists():
        raise ArtifactError(f"finalized artifact already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_video(episode: Path, trace) -> dict[str, object]:
    frames = []
    for observation in trace.observations:
        candidates = [image for image in observation.images if image.name == "camera.primary.rgb"]
        if not candidates and observation.images:
            candidates = [observation.images[0]]
        if candidates:
            frames.append(np.asarray(candidates[0].data))
    relationship = {
        "reset_frame_count": 1 if frames else 0,
        "accepted_action_count": len(trace.executed_actions),
        "post_action_frame_count": max(len(frames) - 1, 0),
        "expected_frame_count": len(trace.executed_actions) + 1 if frames else 0,
        "actual_frame_count": len(frames),
        "aligned": bool(frames) and len(frames) == len(trace.executed_actions) + 1,
    }
    base = {
        "schema_version": VIDEO_SCHEMA_VERSION,
        "source": "canonical_episode_trace",
        "camera": "camera.primary.rgb",
        "frame_relationship": relationship,
    }
    if not frames:
        return {**base, "status": "unavailable", "reason": "trace contains no recorded RGB frames"}
    first = frames[0]
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ArtifactError("video source frames must be HWC uint8 RGB")
    if any(frame.shape != first.shape or frame.dtype != np.uint8 for frame in frames):
        raise ArtifactError("video source frames must have one consistent shape and dtype")
    try:
        import cv2
    except ImportError:
        return {**base, "status": "unavailable", "reason": "OpenCV video runtime is unavailable"}

    height, width, _ = first.shape
    frequencies = {
        prediction.action_spec.control_frequency_hz
        for prediction in trace.policy_predictions
        if prediction.action_spec.control_frequency_hz is not None
    }
    fps = float(next(iter(frequencies))) if len(frequencies) == 1 else 20.0
    target = episode / "video.mp4"
    if target.exists():
        raise ArtifactError("finalized video artifact already exists")
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True,
    )
    if not writer.isOpened():
        raise ArtifactError("OpenCV could not open the MP4 video encoder")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not target.is_file() or target.stat().st_size == 0:
        raise ArtifactError("video encoder produced an empty artifact")

    # Decode through a new reader instance after the writer has been closed.
    reader = cv2.VideoCapture(str(target))
    decoded = 0
    decoded_width = decoded_height = None
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise ArtifactError("video decoder returned an empty frame")
            decoded += 1
            decoded_height, decoded_width = frame.shape[:2]
    finally:
        reader.release()
    if decoded != len(frames) or (decoded_width, decoded_height) != (width, height):
        raise ArtifactError("independent video decode disagrees with encoded frames")
    return {
        **base,
        "status": "available",
        "path": "video.mp4",
        "codec": "mp4v",
        "pixel_format": "RGB source; decoder-native BGR verification",
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": len(frames),
        "decoded_frame_count": decoded,
        "duration_seconds": len(frames) / fps,
        "decode_verified": True,
        "size_bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }


def generate_canonical_videos(run_path: str | Path) -> tuple[dict[str, object], ...]:
    path = Path(run_path)
    results = []
    codec = TraceCodec()
    for episode in sorted(path.glob("tasks/*/episodes/*")):
        trace = codec.decode(episode)
        metadata = _episode_video(episode, trace)
        _atomic_json(episode / "video.json", metadata)
        results.append({"episode_path": str(episode.relative_to(path)), **metadata})
    return tuple(results)


def build_run_report(run_path: str | Path, *, final_manifest=None) -> dict[str, object]:
    path = Path(run_path)
    started = _json(path / "manifest.started.json")
    if final_manifest is None:
        candidates = [item for item in (path / "manifest.completed.json", path / "manifest.failed.json") if item.is_file()]
        if len(candidates) != 1:
            raise ArtifactError("report generation requires exactly one final run manifest")
        final_manifest = _json(candidates[0])
    connection = _json(path / "connection.json") if (path / "connection.json").is_file() else {}
    codec = TraceCodec()
    episodes = []
    all_metrics = []
    for episode in sorted(path.glob("tasks/*/episodes/*")):
        trace = codec.decode(episode)
        metrics_path = episode / "metrics.episode.json"
        metrics = _json(metrics_path) if metrics_path.is_file() else []
        all_metrics.extend(metrics)
        video = _json(episode / "video.json") if (episode / "video.json").is_file() else {
            "schema_version": VIDEO_SCHEMA_VERSION, "status": "unavailable", "reason": "video was not generated",
        }
        episodes.append({
            "episode_id": str(trace.episode_context.episode_id),
            "task_id": str(trace.episode_context.task_id),
            "seed": trace.episode_context.seed,
            "instruction": trace.episode_context.initial_instruction.text,
            "terminal_status": trace.terminal_status.value,
            "counts": {
                "observations": len(trace.observations),
                "predictions": len(trace.policy_predictions),
                "executed_actions": len(trace.executed_actions),
            },
            "counts_source": "canonical_episode_trace",
            "metrics": metrics,
            "video": video,
        })
    task_metrics = []
    for metric_file in sorted(path.glob("tasks/*/metrics.task.json")):
        values = _json(metric_file)
        task_metrics.extend(values)
    all_metrics.extend(task_metrics)
    renderer = (
        final_manifest.get("metadata", {})
        .get("benchmark_runtime", {})
        .get("libero_renderer")
    )
    detected = renderer.get("detected_renderer") if isinstance(renderer, dict) else None
    renderer_classification = "unknown"
    if isinstance(detected, dict) and "llvmpipe" in str(detected.get("renderer", "")).lower():
        renderer_classification = "software"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": started.get("run_id"),
        "status": final_manifest.get("status"),
        "qualification_label": started.get("metadata", {}).get("qualification", "production"),
        "scientific_config_hash": started.get("scientific_config_hash"),
        "execution_config_hash": started.get("execution_config_hash"),
        "benchmark": connection.get("benchmark"),
        "policy": connection.get("policy"),
        "renderer": renderer,
        "renderer_acceleration_classification": renderer_classification,
        "episodes": episodes,
        "task_metrics": task_metrics,
        "metric_results": all_metrics,
        "metric_status_semantics": {
            "available": "measured or derived as identified by metric metadata",
            "unavailable": "not measured; value must be null",
            "insufficient_data": "defined but trace has too few samples; value must be null",
            "error": "metric evaluation failure; value must be null",
        },
        "sources": {
            "configuration": ["source_config.yaml", "resolved_config.yaml"],
            "connection": "connection.json",
            "plan": "plan.json",
            "traces": "tasks/*/episodes/*/trace.json",
            "episode_metrics": "tasks/*/episodes/*/metrics.episode.json",
            "task_metrics": "tasks/*/metrics.task.json",
        },
    }


def human_report(document: dict[str, object]) -> str:
    lines = [
        f"OVLAB run {document.get('run_id')}",
        f"status: {document.get('status')}",
        f"scientific_config_hash: {document.get('scientific_config_hash')}",
        f"execution_config_hash: {document.get('execution_config_hash')}",
        f"renderer_acceleration: {document.get('renderer_acceleration_classification')}",
        "metrics:",
    ]
    for metric in document.get("metric_results", []):
        value = "unavailable" if metric.get("value") is None else metric.get("value")
        lines.append(
            f"  {metric.get('scope')} {metric.get('metric_id')}={value} "
            f"{metric.get('unit')} [{metric.get('status')}]"
        )
    return "\n".join(lines) + "\n"


def write_canonical_report(run_path: str | Path, *, final_manifest) -> dict[str, object]:
    path = Path(run_path)
    report = build_run_report(path, final_manifest=final_manifest)
    reports = path / "reports"
    _atomic_json(reports / "report.json", report)
    text_path = reports / "report.txt"
    if text_path.exists():
        raise ArtifactError("finalized report artifact already exists: report.txt")
    text_path.write_text(human_report(report), encoding="utf-8")
    return report


def regenerate_report(run_path: str | Path, output_path: str | Path) -> dict[str, object]:
    source = Path(run_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output == source or output.is_relative_to(source):
        raise ArtifactError("derived report output must be outside the immutable run")
    if output.exists():
        raise ArtifactError("derived report output already exists")
    report = build_run_report(source)
    output.mkdir(parents=True)
    _atomic_json(output / "report.json", report)
    (output / "report.txt").write_text(human_report(report), encoding="utf-8")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_run": str(source),
        "output": str(output),
        "canonical_semantic_match": (
            (source / "reports/report.json").is_file()
            and _json(source / "reports/report.json") == report
        ),
        "source_modified": False,
    }


def integrity_document(run_path: str | Path, *, virtual_files=None) -> dict[str, object]:
    path = Path(run_path)
    virtual_files = dict(virtual_files or {})
    rows = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = str(item.relative_to(path))
        if relative == "integrity.json" or relative.endswith(".tmp"):
            continue
        rows.append({"path": relative, "size_bytes": item.stat().st_size, "sha256": _sha256(item)})
    for relative, data in sorted(virtual_files.items()):
        rows.append({"path": relative, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    rows.sort(key=lambda row: row["path"])
    return {"schema_version": INTEGRITY_SCHEMA_VERSION, "algorithm": "sha256", "files": rows}


def remove_generated_outputs(run_path: str | Path) -> None:
    path = Path(run_path)
    shutil.rmtree(path / "reports", ignore_errors=True)
    for episode in path.glob("tasks/*/episodes/*"):
        for name in ("video.mp4", "video.json"):
            try:
                (episode / name).unlink()
            except FileNotFoundError:
                pass
    try:
        (path / "integrity.json").unlink()
    except FileNotFoundError:
        pass
