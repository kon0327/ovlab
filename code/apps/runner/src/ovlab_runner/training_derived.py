"""Offline system-performance reports over finalized canonical training runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import uuid

from .derived import _chart_statistics, _chart_svg, _descriptive_statistics
from .errors import ArtifactError, ReportingSourceUnavailableError


TRAINING_REPORT_SCHEMA = "ovlab.training-performance-report/v1"
TRAINING_REPORT_MANIFEST_SCHEMA = "ovlab.training-performance-report-manifest/v1"
TRAINING_REPORT_RENDERER = "ovlab-training-performance-html@1.0.0"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read canonical training source: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"canonical training document must be an object: {path}")
    return value


def _read_metrics(path: Path) -> list[dict[str, object]]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("metric row is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactError(f"cannot read canonical training metrics: {path}") from exc
    return rows


def _verify_training_run(root: Path) -> dict[str, object]:
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != "ovlab.training-run/v1" or manifest.get("status") != "completed":
        raise ArtifactError("training performance reports require a finalized completed training run")
    for entry in manifest.get("files", []):
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactError("training manifest contains an unsafe file path")
        path = root / relative
        if not path.is_file() or path.stat().st_size != entry.get("size") or _sha256(path) != entry.get("sha256"):
            raise ArtifactError(f"training run integrity failure: {relative}")
    return manifest


def build_training_performance_model(training_run: str | Path) -> dict[str, object]:
    root = Path(training_run).expanduser().resolve()
    manifest = _verify_training_run(root)
    result = _read_json(root / "result.json")
    plan = _read_json(root / "training-plan.json")
    metrics = _read_metrics(root / "metrics.jsonl")
    allocated = [float(row["gpu_memory_allocated_bytes"]) / (1024 * 1024) for row in metrics if isinstance(row.get("gpu_memory_allocated_bytes"), int | float)]
    reserved = [float(row["gpu_memory_reserved_bytes"]) / (1024 * 1024) for row in metrics if isinstance(row.get("gpu_memory_reserved_bytes"), int | float)]
    peak_allocated = [float(row["gpu_memory_peak_bytes"]) / (1024 * 1024) for row in metrics if isinstance(row.get("gpu_memory_peak_bytes"), int | float)]
    peak_reserved = [float(row["gpu_memory_peak_reserved_bytes"]) / (1024 * 1024) for row in metrics if isinstance(row.get("gpu_memory_peak_reserved_bytes"), int | float)]
    gflops = [float(row["estimated_gflops"]) for row in metrics if isinstance(row.get("estimated_gflops"), int | float)]
    loss = [float(row["training_loss"]) for row in metrics if isinstance(row.get("training_loss"), int | float)]
    duration = [float(row["step_duration_ms"]) for row in metrics if isinstance(row.get("step_duration_ms"), int | float)]
    compute_identity = None
    for row in metrics:
        row_performance = row.get("performance", {})
        compute = row_performance.get("estimated_compute", {}) if isinstance(row_performance, dict) else {}
        if isinstance(compute, dict) and compute.get("method"):
            compute_identity = {
                key: compute.get(key) for key in ("method", "formula", "qualification")
            }
            break
    performance = result.get("performance", {})
    parameter_counts = performance.get("parameter_counts", {}) if isinstance(performance, dict) else {}
    if not isinstance(parameter_counts, dict) or not parameter_counts:
        parameter_counts = {
            "total": result.get("total_parameter_count"),
            "trainable": result.get("trainable_parameter_count"),
            "frozen": result.get("frozen_parameter_count"),
            "adapter": result.get("adapter_parameter_count"),
            "trainable_adapter": result.get("trainable_adapter_parameter_count"),
        }
    return {
        "schema_version": TRAINING_REPORT_SCHEMA,
        "training_run_id": manifest.get("run_id", root.name),
        "status": result.get("status"),
        "scientific_training_id": manifest.get("scientific_training_id"),
        "execution_plan_id": manifest.get("execution_plan_id"),
        "checkpoint_id": manifest.get("checkpoint_id"),
        "profile_id": plan.get("scientific", {}).get("profile", {}).get("id") if isinstance(plan.get("scientific"), dict) else None,
        "parameter_counts": parameter_counts,
        "system_metrics": {
            "schema_version": "ovlab.system-metrics-summary/v1",
            "sample_count": len(metrics),
            "vram_source": "pytorch-cuda-caching-allocator",
            "vram_qualification": "policy/trainer process allocator counters; not whole-device NVML memory usage",
            "peak_allocated_bytes": result.get("peak_vram_bytes"),
            "peak_reserved_bytes": result.get("peak_reserved_vram_bytes"),
            "estimated_compute_method": performance.get("estimated_compute_method") if isinstance(performance, dict) else None,
            "compute_qualification": "analytical dense-parameter/token proxy; not measured hardware FLOPs",
            "compute_estimator": compute_identity,
            "estimated_total_gflops": sum(gflops) if gflops else None,
            "statistics": {
                "allocated_mib": _descriptive_statistics(allocated),
                "reserved_mib": _descriptive_statistics(reserved),
                "peak_allocated_mib": _descriptive_statistics(peak_allocated),
                "peak_reserved_mib": _descriptive_statistics(peak_reserved),
                "estimated_gflops_per_step": _descriptive_statistics(gflops),
                "training_loss": _descriptive_statistics(loss),
                "optimizer_step_duration_ms": _descriptive_statistics(duration),
            },
        },
        "series": {
            "allocated_mib": allocated, "reserved_mib": reserved,
            "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
            "estimated_gflops": gflops, "training_loss": loss,
            "optimizer_step_duration_ms": duration,
        },
        "source": {
            "training_manifest_sha256": _sha256(root / "manifest.json"),
            "metrics_sha256": _sha256(root / "metrics.jsonl"),
            "result_sha256": _sha256(root / "result.json"),
        },
    }


def _chart_model(model: dict[str, object]) -> dict[str, object]:
    series = model["series"]
    return {"tasks": [{"task_id": "training", "episodes": [{
        "episode_id": "optimizer-steps",
        "trace_view": {
            "vram_allocated_mib": series["allocated_mib"],
            "vram_reserved_mib": series["reserved_mib"],
            "vram_peak_allocated_mib": series["peak_allocated_mib"],
            "vram_peak_reserved_mib": series["peak_reserved_mib"],
            "vram_statistics_mib": {
                "allocated": _descriptive_statistics(series["allocated_mib"]),
                "reserved": _descriptive_statistics(series["reserved_mib"]),
                "peak_allocated": _descriptive_statistics(series["peak_allocated_mib"]),
                "peak_reserved": _descriptive_statistics(series["peak_reserved_mib"]),
            },
            "estimated_gflops": series["estimated_gflops"],
            "estimated_gflops_statistics": _descriptive_statistics(series["estimated_gflops"]),
        },
    }]}]}


def _html(model: dict[str, object], chart_statistics: dict[str, dict[str, object]]) -> str:
    counts = model["parameter_counts"]
    system = model["system_metrics"]
    rows = []
    for chart_id in ("vram_tracking", "estimated_compute"):
        rows.append(
            f'<h3>{html.escape(chart_id.replace("_", " ").title())}</h3>'
            f'<object class="chart" type="image/svg+xml" data="assets/{chart_id}.svg"></object>'
            '<table><thead><tr><th>Series</th><th>Unit</th><th>n</th><th>Min</th><th>Median</th><th>Mean</th><th>P95</th><th>Max</th></tr></thead><tbody>'
            + "".join(
                "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
                    row.get("series"), row.get("unit"), row.get("n"), row.get("minimum"),
                    row.get("median"), row.get("mean"), row.get("p95"), row.get("maximum"),
                )) + "</tr>"
                for row in chart_statistics[chart_id]["rows"]
            ) + "</tbody></table>"
        )
    estimator = system.get("compute_estimator")
    estimator_html = "" if not isinstance(estimator, dict) else (
        f'<p><strong>{html.escape(str(estimator.get("method")))}</strong>: '
        f'<code>{html.escape(str(estimator.get("formula")))}</code>. '
        f'{html.escape(str(estimator.get("qualification")))}</p>'
    )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OVLAB training performance {html.escape(str(model["training_run_id"]))}</title><style>
body{{font-family:system-ui,sans-serif;background:#f2f4f7;color:#172033;margin:0}}main{{max-width:1180px;margin:auto;padding:28px}}section{{background:#fff;border:1px solid #d0d5dd;border-radius:10px;padding:18px;margin:16px 0}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #e4e7ec;padding:8px;text-align:left}}code{{overflow-wrap:anywhere}}.chart{{width:100%;height:430px}}
</style></head><body><main><h1>OVLAB training performance report</h1><section><h2>Identity</h2><p><strong>Training run:</strong> <code>{html.escape(str(model["training_run_id"]))}</code><br><strong>Status:</strong> {html.escape(str(model["status"]))}<br><strong>Checkpoint:</strong> <code>{html.escape(str(model["checkpoint_id"]))}</code></p></section>
<section><h2>Model complexity</h2><table><thead><tr><th>Total</th><th>Trainable</th><th>Frozen</th><th>Adapter</th><th>Trainable adapter</th></tr></thead><tbody><tr><td>{counts.get("total", "unavailable")}</td><td>{counts.get("trainable", "unavailable")}</td><td>{counts.get("frozen", "unavailable")}</td><td>{counts.get("adapter", "unavailable")}</td><td>{counts.get("trainable_adapter", "unavailable")}</td></tr></tbody></table></section>
<section><h2>System telemetry</h2><p>VRAM source: {html.escape(str(system["vram_source"]))}; {html.escape(str(system["vram_qualification"]))}. Compute: {html.escape(str(system["compute_qualification"]))}.</p>{estimator_html}<p><strong>Peak allocated:</strong> {system.get("peak_allocated_bytes", "unavailable")} bytes · <strong>Peak reserved:</strong> {system.get("peak_reserved_bytes", "unavailable")} bytes · <strong>Estimated total:</strong> {system.get("estimated_total_gflops", "unavailable")} GFLOPs</p>{''.join(rows)}</section>
<section><h2>Provenance</h2><p>Canonical training evidence is read-only. Full data, estimator formulas and source checksums are in <a href="report.json">report.json</a>.</p></section></main></body></html>'''


class TrainingDerivedReportEngine:
    def __init__(self, training_runs_root: str | Path, derived_root: str | Path):
        self.training_runs_root = Path(training_runs_root).expanduser().resolve()
        self.derived_root = Path(derived_root).expanduser().resolve()
        display = os.environ.get("OVLAB_DERIVED_DISPLAY_ROOT")
        self.display_root = Path(display).expanduser() if display else self.derived_root

    def _display_path(self, target: Path) -> str:
        return str(self.display_root / target.relative_to(self.derived_root))

    def _source(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise ArtifactError("training run must be a portable run ID")
        source = (self.training_runs_root / run_id).resolve()
        if not source.is_relative_to(self.training_runs_root) or not source.is_dir():
            raise ReportingSourceUnavailableError(f"canonical training run is unavailable: {run_id}")
        return source

    def generate(self, run_id: str) -> dict[str, object]:
        source = self._source(run_id)
        model = build_training_performance_model(source)
        identity = {"source": model["source"], "renderer": TRAINING_REPORT_RENDERER, "schema": TRAINING_REPORT_SCHEMA}
        build_id = _digest(identity)[:20]
        profile_root = self.derived_root / "training" / run_id / "system-performance"
        target = profile_root / build_id
        if target.is_dir():
            return {**self.verify(run_id, build_id=build_id), "reused": True, "output": self._display_path(target)}
        stage = profile_root / f".{build_id}.{uuid.uuid4().hex}.partial"
        (stage / "assets").mkdir(parents=True)
        try:
            chart_model = _chart_model(model)
            chart_stats = {}
            for chart_id, builder in (("vram_tracking", "vram_timeseries"), ("estimated_compute", "estimated_compute_timeseries")):
                (stage / "assets" / f"{chart_id}.svg").write_text(_chart_svg(chart_id, builder, chart_model), encoding="utf-8")
                chart_stats[chart_id] = _chart_statistics(builder, chart_model)
            model["charts"] = chart_stats
            (stage / "report.json").write_bytes(_canonical(model))
            (stage / "index.html").write_text(_html(model, chart_stats), encoding="utf-8")
            files = [{"path": str(path.relative_to(stage)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)} for path in sorted(stage.rglob("*")) if path.is_file()]
            manifest = {
                "schema_version": TRAINING_REPORT_MANIFEST_SCHEMA, "training_run_id": run_id,
                "derived_build_id": build_id, "identity_inputs": identity, "files": files,
                "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            manifest["manifest_payload_sha256"] = _digest(manifest)
            (stage / "manifest.json").write_bytes(_canonical(manifest))
            target.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(target)
            (profile_root / "latest.json").write_bytes(_canonical({"schema_version": "ovlab.training-report-latest/v1", "derived_build_id": build_id}))
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return {**self.verify(run_id, build_id=build_id), "reused": False, "output": self._display_path(target)}

    def verify(self, run_id: str, *, build_id: str | None = None) -> dict[str, object]:
        source = self._source(run_id)
        _verify_training_run(source)
        profile_root = self.derived_root / "training" / run_id / "system-performance"
        if build_id is None:
            build_id = str(_read_json(profile_root / "latest.json")["derived_build_id"])
        target = (profile_root / build_id).resolve()
        if not target.is_relative_to(profile_root.resolve()):
            raise ArtifactError("training report build escapes the derived root")
        manifest = _read_json(target / "manifest.json")
        if (
            manifest.get("schema_version") != TRAINING_REPORT_MANIFEST_SCHEMA
            or manifest.get("training_run_id") != run_id
            or manifest.get("derived_build_id") != build_id
            or _digest(manifest.get("identity_inputs"))[:20] != build_id
        ):
            raise ArtifactError("training report manifest identity mismatch")
        recorded = manifest.pop("manifest_payload_sha256", None)
        if recorded != _digest(manifest):
            raise ArtifactError("training report manifest checksum mismatch")
        current = build_training_performance_model(source)
        if current["source"] != manifest.get("identity_inputs", {}).get("source"):
            raise ArtifactError("canonical training report inputs changed")
        expected = {"manifest.json"}
        for row in manifest.get("files", []):
            path = (target / str(row["path"])).resolve()
            if not path.is_relative_to(target) or not path.is_file() or path.stat().st_size != row["size_bytes"] or _sha256(path) != row["sha256"]:
                raise ArtifactError(f"training report file checksum mismatch: {row['path']}")
            expected.add(str(row["path"]))
        actual = {str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()}
        if actual != expected:
            raise ArtifactError("training report contains missing or unexpected files")
        return {
            "schema_version": TRAINING_REPORT_MANIFEST_SCHEMA, "training_run_id": run_id,
            "derived_build_id": build_id, "integrity": "verified", "canonical_training_run_modified": False,
        }
