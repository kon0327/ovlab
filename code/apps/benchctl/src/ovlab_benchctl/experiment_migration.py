"""Two-phase migration of legacy runs into one-run experiment trees.

Apply only copies. Finalize quarantines exact originals after the new interactive
reports verify; cleanup removes the quarantine after another verification pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import stat
import uuid

from ovlab_core.experiment_scope import ExperimentScope


SCHEMA = "ovlab.experiment-migration/v1"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class MigrationError(RuntimeError):
    """Migration cannot proceed without risking existing evidence."""


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_json_bytes(value))
        temporary.chmod(0o640)
        if os.geteuid() == 0:
            parent = path.parent.stat()
            os.chown(temporary, parent.st_uid, parent.st_gid)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _portable(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE.fullmatch(value) or value in {".", ".."}:
        raise MigrationError(f"{label} is not a portable identifier: {value!r}")
    return value


def _inventory(root: Path) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise MigrationError(f"migration source is not a regular directory: {root}")
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise MigrationError(f"migration source contains a symbolic link: {path}")
        relative = str(path.relative_to(root))
        if path.is_dir():
            digest.update(b"D\0" + relative.encode("utf-8") + b"\0")
        elif path.is_file():
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
            size = path.stat().st_size
            file_count += 1
            size_bytes += size
            digest.update(b"F\0" + relative.encode("utf-8") + b"\0" + str(size).encode() + b"\0" + checksum.digest())
        else:
            raise MigrationError(f"migration source contains an unsupported file type: {path}")
    return {"sha256": digest.hexdigest(), "file_count": file_count, "size_bytes": size_bytes}


class ExperimentMigration:
    def __init__(self, data_root: str | Path, *, manifest: str | Path | None = None,
                 verify_evaluation=None, verify_training=None, verify_report=None):
        self.data_root = Path(data_root).expanduser().resolve()
        self.runs_root = self.data_root / "runs"
        self.exports_root = self.data_root / "exports"
        self.manifest_path = Path(manifest).expanduser().resolve() if manifest else self.data_root / "experiment-migration-v1.json"
        self.verify_evaluation = verify_evaluation
        self.verify_training = verify_training
        self.verify_report_callback = verify_report

    def _inside(self, relative: str, kind: str) -> Path:
        parts = Path(relative).parts
        if (not parts or Path(relative).is_absolute() or any(part in {".", ".."} for part in parts)
                or any(not _SAFE.fullmatch(part) for part in parts)):
            raise MigrationError(f"invalid {kind} relative path: {relative!r}")
        path = self.data_root.joinpath(*parts)
        parent = self.data_root
        for part in parts:
            parent = parent / part
            if parent.is_symlink() or not parent.resolve().is_relative_to(self.data_root):
                raise MigrationError(f"{kind} path contains a symbolic link or escapes data root: {relative}")
        return path

    @staticmethod
    def _experiment_id(category: str, run_id: str) -> str:
        identifier = f"legacy-{category}-{run_id}"
        if len(identifier.encode("utf-8")) > 240:
            identifier = identifier[:190] + "-" + hashlib.sha256(identifier.encode()).hexdigest()[:16]
        return ExperimentScope(identifier).experiment_id

    def _legacy_reports(self, category: str, run_id: str) -> list[str]:
        roots = ([f"exports/{category}/derived/{run_id}", f"exports/{category}/interactive/{run_id}"]
                 + ([f"exports/derived/{run_id}", f"exports/interactive/{run_id}"] if category == "evaluate" else
                    [f"exports/derived/training/{run_id}", f"exports/interactive/training/{run_id}"]))
        return [relative for relative in roots if self._inside(relative, "legacy report").is_dir()]

    def _legacy_isolated(self, category: str, run_id: str) -> list[str]:
        roots = [f"exports/{category}/isolated/{run_id}"]
        if category == "evaluate":
            roots.append(f"exports/isolated/{run_id}")
        return [relative for relative in roots if self._inside(relative, "legacy isolated export").is_dir()]

    def _run_status(self, category: str, path: Path) -> str:
        if category == "evaluate":
            if (path / "manifest.completed.json").is_file():
                return "completed"
            if (path / "manifest.failed.json").is_file():
                return "failed"
            return "incomplete"
        manifest = path / "manifest.json"
        if manifest.is_file():
            try:
                document = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise MigrationError(f"invalid training run manifest: {manifest}") from exc
            if document.get("status") == "completed":
                return "completed"
        result = path / "result.json"
        if result.is_file():
            try:
                document = json.loads(result.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise MigrationError(f"invalid training run result: {result}") from exc
            if document.get("status") in {"failed", "interrupted"}:
                return document["status"]
        return "incomplete"

    def _plan(self) -> dict[str, object]:
        entries: list[dict[str, object]] = []
        runs: list[dict[str, object]] = []
        known_isolated: set[str] = set()
        known_reports: set[str] = set()
        for category in ("train", "evaluate"):
            legacy_root = self.runs_root / category
            if not legacy_root.is_dir():
                continue
            for source in sorted(legacy_root.iterdir()):
                if source.is_symlink():
                    raise MigrationError(f"legacy run root contains a symbolic link: {source}")
                if not source.is_dir():
                    continue
                run_id = _portable(source.name, "run ID")
                experiment_id = self._experiment_id(category, run_id)
                destination = f"runs/{experiment_id}/{category}/{run_id}"
                reports = self._legacy_reports(category, run_id)
                isolated = self._legacy_isolated(category, run_id)
                known_reports.update(reports)
                known_isolated.update(isolated)
                status = self._run_status(category, source)
                partial_traces = (category == "evaluate" and
                                  next(source.glob("tasks/*/episodes/*/trace.finalized.json"), None) is not None)
                reportable = status == "completed" or bool(reports) or partial_traces
                run = {"category": category, "run_id": run_id, "experiment_id": experiment_id,
                       "source": f"runs/{category}/{run_id}", "destination": destination,
                       "destination_path": str(self.data_root / destination),
                       "interactive_root": str(self.exports_root / experiment_id / category / "interactive"),
                       "status": status, "reportable": reportable, "report_required": reportable,
                       "report_reason": ("completed canonical run" if status == "completed" else
                                         "existing legacy report" if reports else
                                         "finalized partial evaluation traces" if partial_traces else
                                         "incomplete or failed run without report"),
                       "source_export_paths": isolated + reports,
                       "legacy_reports": reports}
                runs.append(run)
                entries.append({"type": "run", "source": run["source"], "destination": destination,
                                "category": category, "run_id": run_id})
                for index, relative in enumerate(isolated):
                    name = f"{run_id}--legacy" if len(isolated) == 1 else f"{run_id}--legacy-{'typed' if index == 0 else 'flat'}"
                    entries.append({"type": "isolated", "source": relative,
                                    "destination": f"exports/{experiment_id}/{category}/isolated/{name}",
                                    "category": category, "run_id": run_id})
        # Legacy grouped comparisons have no trustworthy parent experiment.
        # Keep each complete artifact under an explicit historical collection.
        for category, legacy_roots in (("evaluate", ("exports/evaluate/grouped", "exports/grouped")),
                                       ("train", ("exports/train/grouped",))):
            for root_index, relative_root in enumerate(legacy_roots):
                root = self._inside(relative_root, "legacy grouped root")
                if not root.is_dir():
                    continue
                for source in sorted(root.iterdir()):
                    if source.is_symlink():
                        raise MigrationError(f"legacy grouped root contains a symbolic link: {source}")
                    if not source.is_dir():
                        continue
                    name = _portable(source.name, "grouped export ID")
                    if any(entry["destination"] == f"exports/legacy-comparisons/{category}/grouped/{name}" for entry in entries):
                        name = f"{name}--flat-{root_index}"
                    entries.append({"type": "grouped", "source": str(source.relative_to(self.data_root)),
                                    "destination": f"exports/legacy-comparisons/{category}/grouped/{name}",
                                    "category": category})
        # Preserve unmatched isolated exports rather than guessing a run owner.
        for category, roots in (("evaluate", ("exports/evaluate/isolated", "exports/isolated")),
                                ("train", ("exports/train/isolated",))):
            for root_index, relative_root in enumerate(roots):
                root = self._inside(relative_root, "legacy isolated root")
                if not root.is_dir():
                    continue
                for source in sorted(root.iterdir()):
                    if source.is_symlink():
                        raise MigrationError(f"legacy isolated root contains a symbolic link: {source}")
                    if not source.is_dir():
                        continue
                    relative = str(source.relative_to(self.data_root))
                    if relative in known_isolated:
                        continue
                    name = _portable(source.name, "orphan isolated export ID")
                    if any(entry["destination"] == f"exports/legacy-comparisons/{category}/isolated/{name}" for entry in entries):
                        name = f"{name}--flat-{root_index}"
                    entries.append({"type": "orphan-isolated", "source": relative,
                                    "destination": f"exports/legacy-comparisons/{category}/isolated/{name}",
                                    "category": category})
        unmatched_reports = []
        for category, roots in (("evaluate", ("exports/evaluate/derived", "exports/evaluate/interactive",
                                               "exports/derived", "exports/interactive")),
                                ("train", ("exports/train/derived", "exports/train/interactive",
                                           "exports/derived/training", "exports/interactive/training"))):
            for relative_root in roots:
                root = self._inside(relative_root, "legacy report root")
                if root.is_dir():
                    for path in sorted(root.iterdir()):
                        if path.is_symlink():
                            raise MigrationError(f"legacy report root contains a symbolic link: {path}")
                        relative = str(path.relative_to(self.data_root))
                        if path.is_dir() and relative not in known_reports and path.name != "training":
                            unmatched_reports.append(relative)
        return {"schema_version": SCHEMA, "data_root": str(self.data_root),
                "runs": runs, "entries": entries, "unmatched_reports": sorted(set(unmatched_reports))}

    def preview(self) -> dict[str, object]:
        plan = self._plan()
        return {**plan, "state": "preview", "run_count": len(plan["runs"]),
                "copy_count": len(plan["entries"]), "report_count": sum(bool(run["report_required"]) for run in plan["runs"])}

    def _verify_canonical(self, category: str, path: Path, status: str) -> str:
        if status != "completed":
            return "byte_inventory_only"
        if category == "evaluate":
            if self.verify_evaluation is None:
                from ovlab_runner.inspection import verify_run
                verifier = verify_run
            else:
                verifier = self.verify_evaluation
        else:
            if self.verify_training is None:
                from ovlab_runner.training_derived import _verify_training_run
                verifier = _verify_training_run
            else:
                verifier = self.verify_training
        verifier(path)
        return "canonical_verified"

    def _copy_verified(self, source: Path, destination: Path, expected: dict[str, object]) -> None:
        if destination.exists():
            raise MigrationError(f"migration destination already exists: {destination}")
        parent = self.data_root
        for part in destination.parent.relative_to(self.data_root).parts:
            parent = parent / part
            if parent.is_symlink():
                raise MigrationError(f"migration destination parent is a symbolic link: {parent}")
            parent.mkdir(exist_ok=True)
            if parent.stat().st_uid == os.getuid() and parent != self.data_root:
                parent.chmod(0o2770)
        stage = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
        try:
            shutil.copytree(source, stage, symlinks=False)
            if os.geteuid() == 0:
                # copytree preserves bytes and modes, but not ownership. The
                # migration may need root to read restrictive failed-run files.
                # Restore their original UID/GID before publishing the copy.
                for original in (source, *sorted(source.rglob("*"))):
                    copied = stage / original.relative_to(source)
                    original_stat = original.stat(follow_symlinks=False)
                    os.chown(copied, original_stat.st_uid, original_stat.st_gid, follow_symlinks=False)
                    copied.chmod(stat.S_IMODE(original_stat.st_mode))
            if _inventory(stage) != expected:
                raise MigrationError(f"staged copy differs from source: {source}")
            stage.rename(destination)
            if _inventory(destination) != expected:
                raise MigrationError(f"published copy differs from source: {source}")
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _write_experiment_record(self, run: dict[str, object], inventory: dict[str, object], *, status: str) -> None:
        experiment_id = run["experiment_id"]
        run_root = self.runs_root / experiment_id
        export_root = self.exports_root / experiment_id
        evidence = run_root / ".experiment"
        for directory in (run_root, evidence, export_root):
            if directory.is_symlink():
                raise MigrationError(f"experiment metadata path is a symbolic link: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            if directory.stat().st_uid == os.getuid():
                directory.chmod(0o2770)
        spec = {"schema_version": "ovlab.experiment/v1", "kind": "workflow", "id": experiment_id,
                "name": f"Migrated {run['category']} run {run['run_id']}", "execute": []}
        plan = {"schema_version": "ovlab.experiment-plan/v1", "experiment_id": experiment_id,
                "name": spec["name"], "nodes": [{"path": "", "id": experiment_id}], "steps": [],
                "migration_only": True}
        migration = {"schema_version": SCHEMA, "legacy_source": run["source"],
                     "canonical_destination": run["destination"], "source_inventory": inventory,
                     "reportable": run["reportable"], "report_reason": run["report_reason"],
                     "source_export_paths": run["source_export_paths"]}
        result = {"schema_version": "ovlab.experiment-result/v1", "experiment_id": experiment_id,
                  "status": status, "runs_root": str(run_root), "exports_root": str(export_root),
                  "nodes": {"": {run["category"]: {"run_id": run["run_id"], "status": run["status"],
                                                         "run_path": run["destination_path"]}}},
                  "grouped": {}, "migration": migration,
                  "overview": str(export_root / "index.html")}
        _atomic_json(evidence / "spec.json", spec)
        _atomic_json(evidence / "plan.json", plan)
        _atomic_json(evidence / "migration.json", migration)
        _atomic_json(evidence / "result.json", result)
        _atomic_json(export_root / "experiment.json", result)
        profile = "system-performance" if run["category"] == "train" else "libero-task-default"
        latest = export_root / run["category"] / "interactive" / run["run_id"] / profile / "latest.json"
        report_link = ""
        if latest.is_file():
            try:
                build_id = json.loads(latest.read_text(encoding="utf-8")).get("build_id")
            except (OSError, json.JSONDecodeError):
                build_id = None
            if isinstance(build_id, str) and _SAFE.fullmatch(build_id) and (latest.parent / build_id / "index.html").is_file():
                href = f"{run['category']}/interactive/{run['run_id']}/{profile}/{build_id}/index.html"
                report_link = f'<p><a href="{html.escape(href, quote=True)}">Open interactive report</a></p>'
        title = html.escape(spec["name"])
        body = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{title}</title><style>body{{font:16px/1.5 system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#10243b}}'
                'main{border:1px solid #cbd8e6;border-radius:12px;padding:1.5rem}code{overflow-wrap:anywhere}</style></head><body><main>'
                f'<h1>{title}</h1><p>Status: <strong>{html.escape(status)}</strong></p>'
                f'<p>Run: <code>{html.escape(run["run_id"])}</code> · Phase: {html.escape(run["category"])}</p>'
                f'<p>Evidence: {html.escape(run["report_reason"])}</p>{report_link}'
                '<p><a href="experiment.json">Experiment record</a></p></main></body></html>')
        temporary = export_root / f".index.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(body, encoding="utf-8")
            temporary.chmod(0o640)
            temporary.replace(export_root / "index.html")
        finally:
            temporary.unlink(missing_ok=True)

    def _verify_experiment_records(self, document: dict[str, object]) -> None:
        for run in document["runs"]:
            experiment_id = run["experiment_id"]
            run_root = self.runs_root / experiment_id
            export_root = self.exports_root / experiment_id
            spec = json.loads((run_root / ".experiment/spec.json").read_text(encoding="utf-8"))
            migration = json.loads((run_root / ".experiment/migration.json").read_text(encoding="utf-8"))
            result = json.loads((export_root / "experiment.json").read_text(encoding="utf-8"))
            if (spec.get("schema_version") != "ovlab.experiment/v1" or spec.get("id") != experiment_id
                    or migration.get("canonical_destination") != run["destination"]
                    or result.get("experiment_id") != experiment_id
                    or not (export_root / "index.html").is_file()):
                raise MigrationError(f"migrated experiment metadata is inconsistent: {experiment_id}")

    def apply(self) -> dict[str, object]:
        if self.manifest_path.exists():
            raise MigrationError(f"migration manifest already exists: {self.manifest_path}")
        plan = self._plan()
        if not plan["runs"]:
            raise MigrationError("no legacy runs were found")
        for entry in plan["entries"]:
            if self._inside(entry["destination"], "migration destination").exists():
                raise MigrationError(f"migration destination already exists: {entry['destination']}")
        for run in plan["runs"]:
            for root in (self.runs_root, self.exports_root):
                experiment = root / run["experiment_id"]
                if experiment.is_symlink():
                    raise MigrationError(f"migration experiment path is a symbolic link: {experiment}")
                if experiment.is_dir():
                    # A failed earlier copy can leave only empty parents. Use
                    # rmdir so any real artifact or unknown file stops here.
                    for directory in sorted((path for path in experiment.rglob("*") if path.is_dir()),
                                            key=lambda path: len(path.parts), reverse=True):
                        try:
                            directory.rmdir()
                        except OSError:
                            break
                    try:
                        experiment.rmdir()
                    except OSError as exc:
                        raise MigrationError(f"migration experiment already exists: {experiment}") from exc
        total_bytes = sum(_inventory(self._inside(entry["source"], "migration source"))["size_bytes"] for entry in plan["entries"])
        if shutil.disk_usage(self.data_root).free < total_bytes:
            raise MigrationError(f"insufficient free space for migration copy: {total_bytes} bytes required")
        created = []
        try:
            for entry in plan["entries"]:
                source = self._inside(entry["source"], "migration source")
                destination = self._inside(entry["destination"], "migration destination")
                inventory = _inventory(source)
                if entry["type"] == "run":
                    run = next(run for run in plan["runs"] if run["source"] == entry["source"])
                    entry["canonical_verification"] = self._verify_canonical(entry["category"], source, run["status"])
                self._copy_verified(source, destination, inventory)
                created.append(destination)
                if _inventory(source) != inventory:
                    raise MigrationError(f"legacy source changed during migration: {source}")
                if entry["type"] == "run":
                    self._verify_canonical(entry["category"], destination, run["status"])
                entry["inventory"] = inventory
            for run in plan["runs"]:
                run_entry = next(entry for entry in plan["entries"] if entry["type"] == "run" and entry["source"] == run["source"])
                self._write_experiment_record(run, run_entry["inventory"], status="migrating")
            plan.update({"state": "prepared", "migration_id": uuid.uuid4().hex,
                         "prepared_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})
            _atomic_json(self.manifest_path, plan)
        except Exception:
            for destination in reversed(created):
                if destination.is_dir():
                    shutil.rmtree(destination)
            for run in plan["runs"]:
                for root in (self.runs_root, self.exports_root):
                    experiment = root / run["experiment_id"]
                    if experiment.is_dir():
                        shutil.rmtree(experiment)
            raise
        return {**plan, "manifest_path": str(self.manifest_path)}

    def _manifest(self, expected_state: str) -> dict[str, object]:
        try:
            document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MigrationError(f"cannot read migration manifest: {self.manifest_path}") from exc
        if not isinstance(document, dict) or document.get("schema_version") != SCHEMA or document.get("state") != expected_state or document.get("data_root") != str(self.data_root):
            raise MigrationError("migration manifest identity or state mismatch")
        migration_id = document.get("migration_id")
        if not isinstance(migration_id, str) or re.fullmatch(r"[0-9a-f]{32}", migration_id) is None:
            raise MigrationError("migration manifest ID is invalid")
        for entry in document.get("entries", []):
            self._inside(entry["source"], "migration source")
            self._inside(entry["destination"], "migration destination")
            if not isinstance(entry.get("inventory"), dict):
                raise MigrationError("migration manifest lacks verified source inventory")
        return document

    def _verify_report(self, run: dict[str, object]) -> None:
        if not run["report_required"]:
            return
        source = self._inside(run["destination"], "migrated run")
        interactive = self.exports_root / run["experiment_id"] / run["category"] / "interactive"
        if self.verify_report_callback is not None:
            self.verify_report_callback(run["category"], source, interactive)
        elif run["category"] == "train":
            from ovlab_runner.unified_reports import TrainingInteractiveReportEngine
            TrainingInteractiveReportEngine(self.runs_root, interactive).verify(source)
        else:
            from ovlab_runner.unified_reports import UnifiedInteractiveReportEngine
            UnifiedInteractiveReportEngine(self.runs_root, interactive).verify(source)

    def _verify_prepared(self, document: dict[str, object], *, originals: bool) -> None:
        self._verify_experiment_records(document)
        for entry in document["entries"]:
            destination = self._inside(entry["destination"], "migration destination")
            if _inventory(destination) != entry["inventory"]:
                raise MigrationError(f"migrated copy changed: {destination}")
            if originals:
                source = self._inside(entry["source"], "migration source")
                if _inventory(source) != entry["inventory"]:
                    raise MigrationError(f"legacy source changed after migration: {source}")
            if entry["type"] == "run":
                run = next(run for run in document["runs"] if run["source"] == entry["source"])
                self._verify_canonical(entry["category"], destination, run["status"])
        for run in document["runs"]:
            self._verify_report(run)

    def finalize(self) -> dict[str, object]:
        document = self._manifest("prepared")
        if document.get("unmatched_reports"):
            raise MigrationError("unmatched legacy reports require manual assignment before cleanup")
        expected = self._plan()
        if (expected["runs"] != document["runs"] or
                [{key: value for key, value in entry.items() if key not in {"inventory", "canonical_verification"}}
                 for entry in document["entries"]] != expected["entries"]):
            raise MigrationError("legacy migration plan changed after prepare")
        self._verify_prepared(document, originals=True)
        sources = [entry["source"] for entry in document["entries"]]
        sources.extend(report for run in document["runs"] for report in run["legacy_reports"])
        if len(sources) != len(set(sources)):
            raise MigrationError("migration plan selects a legacy directory more than once")
        quarantine = self.data_root / ".experiment-migration-quarantine" / document["migration_id"]
        if quarantine.exists():
            raise MigrationError(f"migration quarantine already exists: {quarantine}")
        moved = []
        try:
            for relative in sources:
                source = self._inside(relative, "legacy source")
                target = quarantine / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
                moved.append((source, target))
            document.update({"state": "quarantined", "quarantine_root": str(quarantine),
                             "finalized_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})
            _atomic_json(self.manifest_path, document)
        except Exception:
            for source, target in reversed(moved):
                if target.exists() and not source.exists():
                    target.rename(source)
            raise
        return {**document, "manifest_path": str(self.manifest_path)}

    def cleanup(self) -> dict[str, object]:
        document = self._manifest("quarantined")
        quarantine = self.data_root / ".experiment-migration-quarantine" / document["migration_id"]
        if document.get("quarantine_root") != str(quarantine):
            raise MigrationError("migration quarantine identity mismatch")
        self._verify_prepared(document, originals=False)
        if quarantine.is_dir():
            for entry in document["entries"]:
                archived = quarantine / entry["source"]
                if archived.exists() and _inventory(archived) != entry["inventory"]:
                    raise MigrationError(f"quarantined legacy source changed: {archived}")
            # Record refresh can fail without deleting originals. A retry is
            # safe because the archived canonical source is still present.
            for run in document["runs"]:
                run_entry = next(entry for entry in document["entries"] if entry["type"] == "run" and entry["source"] == run["source"])
                self._write_experiment_record(run, run_entry["inventory"], status="migrated")
            for relative in (entry["source"] for entry in document["entries"]):
                archived = quarantine / relative
                if archived.is_dir():
                    shutil.rmtree(archived)
            for run in document["runs"]:
                for relative in run["legacy_reports"]:
                    archived = quarantine / relative
                    if archived.is_dir():
                        shutil.rmtree(archived)
            for directory in sorted((path for path in quarantine.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
                directory.rmdir()
            quarantine.rmdir()
        elif any(self._inside(entry["source"], "legacy source").exists() for entry in document["entries"]):
            raise MigrationError("quarantine is missing while legacy sources remain")
        # Remove only old shared directories which are empty after moving the
        # exact listed artifacts. Unknown content prevents rmdir and survives.
        shared = [self.runs_root / "train", self.runs_root / "evaluate"]
        for category in ("train", "evaluate"):
            shared.extend(self.exports_root / category / family for family in ("derived", "interactive", "isolated", "grouped"))
        shared.extend(self.exports_root / family / "training" for family in ("derived", "interactive"))
        shared.extend(self.exports_root / family for family in ("derived", "interactive", "isolated", "grouped"))
        shared.extend((self.exports_root / "train", self.exports_root / "evaluate",
                       self.data_root / ".experiment-migration-quarantine"))
        for directory in shared:
            try:
                directory.rmdir()
            except FileNotFoundError:
                pass
            except OSError as exc:
                if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                    raise
        document.update({"state": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})
        _atomic_json(self.manifest_path, document)
        return {**document, "manifest_path": str(self.manifest_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy runs into one-run experiment trees")
    parser.add_argument("action", choices=("preview", "apply", "finalize", "cleanup"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    migration = ExperimentMigration(args.data_root, manifest=args.manifest)
    try:
        result = getattr(migration, args.action)()
    except (MigrationError, ValueError, OSError) as exc:
        parser.exit(1, f"migration: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
