"""Experiment-scoped public reports with one complete interactive artifact."""

from __future__ import annotations

import os
from pathlib import Path

from .artifacts.layout import run_key, safe_key
from .derived import (
    CHART_BUILDERS, REPORT_SCHEMA, ReportProfile, _canonical_inputs, _chart_axes,
    _chart_statistics, build_report_model, builtin_profile,
)
from .errors import ArtifactError, ReportingSourceUnavailableError
from .inspection import verify_run
from .interactive import _read, publish_unified_dashboard, verify_unified_dashboard
from .training_derived import (
    _chart_model, _chart_statistics as _training_chart_statistics,
    build_training_performance_model,
)


def _source(root: Path, run: str | Path, *, training: bool) -> Path:
    candidate = Path(run)
    if candidate.is_absolute():
        source = candidate.resolve()
    elif len(candidate.parts) == 1 and candidate.name not in {"", ".", ".."}:
        direct = (root / candidate.name).resolve()
        source = direct if direct.is_relative_to(root) and direct.is_dir() else (root / run_key(candidate.name)).resolve()
    else:
        raise ArtifactError("report run must be an absolute canonical path or a run ID")
    if not source.is_relative_to(root) or not source.is_dir():
        raise ReportingSourceUnavailableError(f"canonical {'training ' if training else ''}run is unavailable: {run}")
    return source


def _display_path(target: Path) -> str:
    """Translate reporting-container paths to the host's configured export root."""
    display = os.environ.get("OVLAB_EXPORTS_DISPLAY_ROOT")
    container_exports = Path("/var/lib/ovlab/exports")
    if display and target.is_relative_to(container_exports):
        return str(Path(display).expanduser() / target.relative_to(container_exports))
    return str(target)


class UnifiedInteractiveReportEngine:
    """Publish evaluation evidence under an exact experiment/node interactive root."""

    def __init__(self, runs_root: str | Path, interactive_root: str | Path,
                 profile: ReportProfile | None = None):
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.interactive_root = Path(interactive_root).expanduser().resolve()
        self.profile = profile or builtin_profile()

    def _model(self, source: Path, task_id: str | None):
        model = build_report_model(source, scope_task_id=task_id, key_parameters=self.profile.key_parameters)
        if model["completeness"] == "complete":
            verify_run(source)
        model["charts"] = [{
            **chart,
            "builder_version": CHART_BUILDERS[chart["builder"]],
            "axes": _chart_axes(chart["builder"]),
            "statistics": _chart_statistics(chart["builder"], model),
            "rendering": "interactive full-series chart; evidence remains accessible on demand",
        } for chart in self.profile.charts]
        return model

    def _identity(self, source: Path, task_id: str | None):
        return {
            "canonical_inputs": list(_canonical_inputs(source, safe_key(task_id) if task_id else None)),
            "profile": self.profile.document(),
            "report_schema": REPORT_SCHEMA,
            "scope_task_id": task_id,
        }

    def generate(self, run: str | Path, *, task_id: str | None = None) -> dict[str, object]:
        source = _source(self.runs_root, run, training=False)
        model = self._model(source, task_id)
        identity = self._identity(source, task_id)
        target, reused = publish_unified_dashboard(source, self.interactive_root, model,
            source_identity=identity, profile_id=self.profile.identifier)
        return {
            "schema_version": "ovlab.interactive-report/v1", "run_id": model["run"]["run_id"],
            "profile_id": self.profile.identifier, "status": model["run"]["status"],
            "interactive_build_id": target.name, "interactive_output": _display_path(target),
            "output": _display_path(target), "integrity": "verified", "interactive_integrity": "verified",
            "canonical_run_modified": False, "reused": reused,
        }

    def verify(self, run: str | Path, *, build_id: str | None = None,
               task_id: str | None = None) -> dict[str, object]:
        source = _source(self.runs_root, run, training=False)
        if task_id is None:
            profile_root = self.interactive_root / run_key(source.name) / self.profile.identifier
            if build_id is None:
                build_id = _read(profile_root / "latest.json").get("build_id")
            if not isinstance(build_id, str) or not build_id or Path(build_id).name != build_id:
                raise ArtifactError("interactive report build ID is invalid")
            manifest = _read(profile_root / build_id / "manifest.json")
            task_id = manifest.get("identity_inputs", {}).get("source_identity", {}).get("scope_task_id")
            if task_id is not None and not isinstance(task_id, str):
                raise ArtifactError("interactive report task scope is invalid")
        model = self._model(source, task_id)
        target = verify_unified_dashboard(source, self.interactive_root, model,
            source_identity=self._identity(source, task_id), profile_id=self.profile.identifier,
            build_id=build_id)
        return {
            "schema_version": "ovlab.interactive-report/v1", "run_id": model["run"]["run_id"],
            "profile_id": self.profile.identifier, "interactive_build_id": target.name,
            "interactive_output": _display_path(target), "output": _display_path(target),
            "integrity": "verified", "interactive_integrity": "verified",
            "canonical_run_modified": False,
        }


class TrainingInteractiveReportEngine:
    """Publish training evidence under an exact experiment/node interactive root."""

    def __init__(self, training_runs_root: str | Path, interactive_root: str | Path, *, key_parameters=None):
        self.training_runs_root = Path(training_runs_root).expanduser().resolve()
        self.interactive_root = Path(interactive_root).expanduser().resolve()
        self.key_parameters = key_parameters

    def _model(self, source: Path):
        model = build_training_performance_model(source, key_parameters=self.key_parameters)
        chart_model = _chart_model(model)
        model["charts"] = {
            name: _training_chart_statistics(builder, chart_model)
            for name, builder in (("vram_tracking", "vram_timeseries"),
                                  ("estimated_compute", "estimated_compute_timeseries"))
        }
        return model

    def generate(self, run: str | Path) -> dict[str, object]:
        source = _source(self.training_runs_root, run, training=True)
        model = self._model(source)
        target, reused = publish_unified_dashboard(source, self.interactive_root, model,
            source_identity=self._identity(model),
            profile_id="system-performance", training=True)
        return {
            "schema_version": "ovlab.interactive-report/v1", "training_run_id": model["training_run_id"],
            "interactive_build_id": target.name, "interactive_output": _display_path(target),
            "output": _display_path(target), "integrity": "verified", "interactive_integrity": "verified",
            "canonical_training_run_modified": False, "reused": reused,
        }

    def verify(self, run: str | Path, *, build_id: str | None = None) -> dict[str, object]:
        source = _source(self.training_runs_root, run, training=True)
        model = self._model(source)
        target = verify_unified_dashboard(source, self.interactive_root, model,
            source_identity=self._identity(model),
            profile_id="system-performance", build_id=build_id)
        return {
            "schema_version": "ovlab.interactive-report/v1", "training_run_id": model["training_run_id"],
            "interactive_build_id": target.name, "interactive_output": _display_path(target),
            "output": _display_path(target), "integrity": "verified", "interactive_integrity": "verified",
            "canonical_training_run_modified": False,
        }

    def _identity(self, model):
        identity = {"canonical_inputs": model["source"]}
        if self.key_parameters is not None:
            identity["key_parameters"] = self.key_parameters
        return identity
