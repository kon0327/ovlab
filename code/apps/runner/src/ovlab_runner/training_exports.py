"""Verified raw and grouped training exports inside an experiment tree."""

from __future__ import annotations

import csv
import html
import io
import json
import os
from pathlib import Path
import shutil
import uuid

from ovlab_core.experiment_scope import ExperimentScope

from .errors import ArtifactError
from .isolated_index import render_isolated_index
from .permissions import finalize_managed_directory, finalize_managed_tree
from .training_derived import build_training_performance_model, _read_json, _read_metrics, _sha256


SCHEMA = "ovlab.training-export/v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for char in value) or value in {".", ".."}:
        raise ArtifactError(f"{label} must be a portable identifier")
    return value


def _csv(rows: list[dict[str, object]]) -> bytes:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def _chart(values: list[float], label: str) -> bytes:
    points = [(index, float(value)) for index, value in enumerate(values) if isinstance(value, int | float)]
    if not points:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 160" role="img" aria-label="{html.escape(label)}"><text x="20" y="80">No recorded values</text></svg>').encode()
    minimum = min(value for _, value in points)
    maximum = max(value for _, value in points)
    span = maximum - minimum or 1.0
    width = max(len(values) - 1, 1)
    line = " ".join(f"{20 + index * 600 / width:.1f},{140 - (value - minimum) * 120 / span:.1f}" for index, value in points)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 160" role="img" aria-label="{html.escape(label)}"><rect width="640" height="160" fill="#f7faff"/><polyline points="{line}" fill="none" stroke="#1557a2" stroke-width="3"/><text x="20" y="15" font-family="sans-serif" font-size="12">{html.escape(label)}</text></svg>').encode()


def _files(root: Path) -> list[dict[str, object]]:
    return [{"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(root.rglob("*")) if path.is_file()]


class TrainingExportEngine:
    def __init__(self, runs_root: str | Path, exports_root: str | Path, *, scope: ExperimentScope | None = None):
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.exports_root = Path(exports_root).expanduser().resolve()
        self.scope = scope if scope is not None else ExperimentScope.from_environment()
        self.search_root = self.scope.runs_root(self.runs_root, "train") if self.scope else self.runs_root
        self.output_root = self.scope.exports_root(self.exports_root, "train") if self.scope else self.exports_root / "train"
        display = os.environ.get("OVLAB_EXPORTS_DISPLAY_ROOT")
        self.display_root = Path(display).expanduser() if display else self.exports_root

    def _source(self, run_id: str) -> Path:
        _safe_id(run_id, "training run ID")
        if not self.search_root.is_dir():
            raise ArtifactError(f"training run is unavailable: {run_id}")
        matches = []
        for plan in self.search_root.rglob("training-plan.json"):
            root = plan.parent
            if root.name != run_id or not (root / "manifest.json").is_file():
                continue
            resolved = root.resolve()
            if not resolved.is_relative_to(self.search_root.resolve()) or plan.is_symlink():
                raise ArtifactError("training run path escapes the selected experiment")
            relative = resolved.relative_to(self.search_root.resolve()).parts
            if self.scope is None and (relative[0] == "evaluate" or len(relative) >= 2 and relative[1] == "evaluate"):
                continue
            matches.append(resolved)
        if len(matches) != 1:
            message = "ambiguous across experiments" if matches else "unavailable"
            raise ArtifactError(f"training run {message}: {run_id}")
        return matches[0]

    def _record(self, run_id: str) -> dict[str, object]:
        source = self._source(run_id)
        model = build_training_performance_model(source)
        plan = _read_json(source / "training-plan.json")
        scientific = plan.get("scientific", {})
        profile = scientific.get("profile", {}) if isinstance(scientific, dict) else {}
        dataset = scientific.get("dataset", {}) if isinstance(scientific, dict) else {}
        profile_dataset = profile.get("dataset", {}) if isinstance(profile, dict) else {}
        dataset_id = dataset.get("dataset_id") if isinstance(dataset, dict) else None
        dataset_ref = profile_dataset.get("ref") if isinstance(profile_dataset, dict) else None
        tasks = profile_dataset.get("tasks") if isinstance(profile_dataset, dict) else None
        training_science = scientific.get("training", {}) if isinstance(scientific, dict) else {}
        objective = training_science.get("objective") if isinstance(training_science, dict) else None
        if objective is None:
            contract = scientific.get("trainer_contract", {}) if isinstance(scientific, dict) else {}
            objective = contract.get("objective") if isinstance(contract, dict) else None
        return {"source": source, "model": model, "dataset_id": dataset_id, "dataset_ref": dataset_ref,
                "task_ids": tasks if isinstance(tasks, list) else None, "objective": objective}

    def _publish(self, kind: str, name: str, files: dict[str, bytes], source: dict[str, str]) -> dict[str, object]:
        _safe_id(name, f"{kind} export name")
        target = self.output_root / kind / name
        if target.is_dir():
            verified = self.verify(kind, name)
            if verified["source_integrity"] != source:
                raise ArtifactError(f"existing {kind} export has different canonical sources: {target}")
            if kind == "isolated" and not (target / "index.html").is_file():
                # Preserve all previously verified bytes while adding the new,
                # checksum-covered browser page to an older isolated export.
                if any(path.is_symlink() for path in target.rglob("*")):
                    raise ArtifactError("cannot upgrade an isolated export containing symbolic links")
                stage = target.parent / f".{name}.{uuid.uuid4().hex}.partial"
                backup = target.parent / f".{name}.{uuid.uuid4().hex}.previous"
                try:
                    shutil.copytree(target, stage)
                    metadata = _read_json(stage / "metadata.json")
                    (stage / "metadata.json").unlink()
                    (stage / "index.html").write_text(
                        render_isolated_index(stage, category="train", metadata=metadata), encoding="utf-8",
                    )
                    metadata["files"] = _files(stage)
                    (stage / "metadata.json").write_bytes(_canonical(metadata))
                    finalize_managed_tree(stage)
                    target.rename(backup)
                    try:
                        stage.rename(target)
                    except BaseException:
                        backup.rename(target)
                        raise
                    shutil.rmtree(backup)
                finally:
                    if stage.exists():
                        shutil.rmtree(stage, ignore_errors=True)
                return {**self.verify(kind, name), "reused": True, "index_added": True}
            return {**verified, "reused": True}
        self.exports_root.mkdir(parents=True, exist_ok=True)
        current = self.exports_root
        for part in target.parent.relative_to(self.exports_root).parts:
            current = current / part
            if current.is_symlink():
                raise ArtifactError(f"training export path contains a symbolic link: {current}")
            current.mkdir(exist_ok=True)
            if current.stat().st_uid == os.getuid():
                finalize_managed_directory(current)
        stage = target.parent / f".{name}.{uuid.uuid4().hex}.partial"
        stage.mkdir()
        try:
            for relative, payload in files.items():
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            metadata = {"schema_version": SCHEMA, "export_type": kind, "name": name,
                        "source_integrity": source}
            if kind == "isolated":
                (stage / "index.html").write_text(
                    render_isolated_index(stage, category="train", metadata=metadata), encoding="utf-8",
                )
            metadata["files"] = _files(stage)
            (stage / "metadata.json").write_bytes(_canonical(metadata))
            stage.rename(target)
            finalize_managed_tree(target)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return {**self.verify(kind, name), "reused": False}

    def generate_isolated(self, run_id: str) -> dict[str, object]:
        record = self._record(run_id)
        source, model = record["source"], record["model"]
        metrics = _read_metrics(source / "metrics.jsonl")
        summary = {key: value for key, value in record.items() if key not in {"source", "model"}}
        summary.update({"run_id": run_id, "model": model})
        return self._publish("isolated", run_id, {
            "summary.json": _canonical(summary),
            "metrics.jsonl": (source / "metrics.jsonl").read_bytes(),
            "metrics.csv": _csv(metrics),
            "figures/training-loss.svg": _chart(model["series"]["training_loss"], "Training loss"),
            "figures/vram-allocated.svg": _chart(model["series"]["allocated_mib"], "Allocated VRAM (MiB)"),
        }, {run_id: model["source"]["training_manifest_sha256"]})

    def generate_grouped(self, group_name: str, *, run_ids: tuple[str, ...] | list[str]) -> dict[str, object]:
        _safe_id(group_name, "group name")
        if len(set(run_ids)) < 2:
            raise ArtifactError("training grouped export requires at least two distinct runs")
        records = [self._record(run_id) for run_id in sorted(set(run_ids))]
        rows = []
        for record in records:
            model = record["model"]
            loss = model["series"]["training_loss"]
            rows.append({"run_id": model["training_run_id"], "profile_id": model["profile_id"],
                         "dataset_id": record["dataset_id"], "dataset_ref": record["dataset_ref"],
                         "task_ids": record["task_ids"], "objective": record["objective"],
                         "optimizer_steps": model["system_metrics"]["sample_count"],
                         "final_training_loss": loss[-1] if loss else None,
                         "peak_allocated_bytes": model["system_metrics"]["peak_allocated_bytes"],
                         "estimated_total_gflops": model["system_metrics"]["estimated_total_gflops"]})
        dataset_ids = {row["dataset_id"] for row in rows}
        objectives = {row["objective"] for row in rows}
        task_sets = {None if row["task_ids"] is None else tuple(sorted(map(str, row["task_ids"]))) for row in rows}
        task_compatible = task_sets == {None} or len(task_sets) == 1 and None not in task_sets
        comparable = (len(dataset_ids) == 1 and None not in dataset_ids
                      and len(objectives) == 1 and None not in objectives and task_compatible)
        comparison = {"status": "comparable" if comparable else "not_comparable",
                      "basis": "same recorded dataset ID, task coverage and training objective" if comparable else "dataset, task coverage or objective differs or was not recorded",
                      "dataset_coverage": rows,
                      "task_coverage": {row["run_id"]: row["task_ids"] for row in rows},
                      "paired_final_loss": ({row["run_id"]: row["final_training_loss"] for row in rows} if comparable else None)}
        source = {record["model"]["training_run_id"]: record["model"]["source"]["training_manifest_sha256"] for record in records}
        return self._publish("grouped", group_name, {
            "run-summary.csv": _csv(rows), "coverage.csv": _csv([{key: row.get(key) for key in ("run_id", "dataset_id", "dataset_ref", "task_ids", "objective")} for row in rows]),
            "comparison.json": _canonical(comparison),
        }, source)

    def verify(self, kind: str, name: str) -> dict[str, object]:
        if kind not in {"isolated", "grouped"}:
            raise ArtifactError("training export kind must be isolated or grouped")
        _safe_id(name, "training export name")
        target = self.output_root / kind / name
        metadata = _read_json(target / "metadata.json")
        if metadata.get("schema_version") != SCHEMA or metadata.get("export_type") != kind or metadata.get("name") != name:
            raise ArtifactError("training export metadata identity mismatch")
        source = metadata.get("source_integrity", {})
        if not isinstance(source, dict):
            raise ArtifactError("training export source inventory is invalid")
        for run_id, recorded in source.items():
            current = self._record(run_id)["model"]["source"]["training_manifest_sha256"]
            if current != recorded:
                raise ArtifactError(f"training export canonical source changed: {run_id}")
        expected = {"metadata.json"}
        for entry in metadata.get("files", []):
            relative = Path(str(entry.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ArtifactError("training export metadata contains an unsafe file path")
            path = target / relative
            if not path.is_file() or path.stat().st_size != entry.get("size_bytes") or _sha256(path) != entry.get("sha256"):
                raise ArtifactError(f"training export checksum mismatch: {relative}")
            expected.add(str(relative))
        actual = {str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()}
        if actual != expected:
            raise ArtifactError("training export contains missing or unexpected files")
        return {"schema_version": SCHEMA, "export_type": kind, "name": name, "integrity": "verified",
                "source_integrity": source, "source_run_ids": sorted(source),
                "output": str(self.display_root / target.relative_to(self.exports_root))}
