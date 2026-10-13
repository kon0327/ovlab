"""Safe host-side lifecycle operations for runs, reports, and exports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from .run_references import (
    RunReferenceAmbiguousError,
    RunReferenceError,
    RunReferenceUnavailableError,
    resolve_run_reference,
    resolve_run_reference_across,
    run_hash,
)


DATA_LIST_SCHEMA = "ovlab.data-list/v1"
DATA_OPERATION_SCHEMA = "ovlab.data-operation/v1"
ARCHIVE_ENTRY_SCHEMA = "ovlab.archive-entry/v1"
EXPORT_METADATA_SCHEMA = "ovlab.export-metadata/v2"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DataManagementError(RuntimeError):
    """A requested data operation cannot be completed safely."""


class DataSourceUnavailableError(DataManagementError):
    """A selected run or report does not exist."""


class DataSafetyError(DataManagementError):
    """An operation would violate an artifact or filesystem safety boundary."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    temporary.replace(path)


def _directory_inventory(root: Path, *, hashes: bool) -> dict[str, object]:
    if root.is_symlink():
        raise DataSafetyError(f"managed data contains a forbidden symbolic link: {root}")
    file_count = 0
    size_bytes = 0
    digest = hashlib.sha256() if hashes else None
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DataSafetyError(f"managed data contains a forbidden symbolic link: {path}")
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        size = path.stat().st_size
        file_count += 1
        size_bytes += size
        if digest is not None:
            file_digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\0")
            digest.update(file_digest.digest())
    return {
        "file_count": file_count,
        "size_bytes": size_bytes,
        "content_sha256": None if digest is None else digest.hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class DataTarget:
    kind: str
    identifier: str
    path: Path
    archive_relative: Path
    state: str
    run_type: str | None = None
    companion_path: Path | None = None
    companion_relative: Path | None = None

    @property
    def run_hash(self) -> str | None:
        return run_hash(self.identifier.split(":", 1)[-1]) if self.kind == "run" else None

    def document(self, *, detail: bool = False) -> dict[str, object]:
        value = {
            "kind": self.kind,
            "id": self.identifier,
            "state": self.state,
            "path": str(self.path),
        }
        if self.run_hash is not None:
            value["run_hash"] = self.run_hash
        if self.run_type is not None:
            value["run_type"] = self.run_type
        if detail:
            value.update(_directory_inventory(self.path, hashes=False))
        return value


class DataManager:
    """Manage canonical runs plus regenerable derived reports and exports."""

    def __init__(
        self, data_root: str | Path, runs_root: str | Path,
        *, exports_root: str | Path | None = None,
        archive_root: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.exports_root = Path(
            exports_root if exports_root is not None else self.data_root / "exports"
        ).expanduser().resolve()
        self.derived_root = self.exports_root / "derived"
        self.interactive_root = self.exports_root / "interactive"
        self.archive_root = Path(
            archive_root if archive_root is not None else self.data_root / "archive"
        ).expanduser().resolve()
        managed_roots = (
            ("runs", self.runs_root),
            ("exports", self.exports_root),
            ("archive", self.archive_root),
        )
        for index, (left_name, left) in enumerate(managed_roots):
            for right_name, right in managed_roots[index + 1:]:
                if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                    raise DataSafetyError(
                        f"configured {left_name} and {right_name} roots overlap: "
                        f"{left} and {right}"
                    )

    @staticmethod
    def _identifier(value: str, label: str) -> str:
        if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
            raise DataSafetyError(f"{label} must be one portable filesystem identifier")
        return value

    @staticmethod
    def _managed_directory(path: Path, root: Path, label: str) -> Path:
        resolved_root = root.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise DataSafetyError(f"{label} escapes its configured root")
        if path.is_symlink() or not path.is_dir():
            raise DataSourceUnavailableError(f"{label} does not exist as a managed directory: {path.name}")
        return resolved

    @staticmethod
    def _run_state(path: Path) -> str:
        if (path / "manifest.completed.json").is_file():
            return "completed"
        failed = path / "manifest.failed.json"
        if failed.is_file():
            try:
                status = json.loads(failed.read_text(encoding="utf-8")).get("status")
            except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
                status = None
            if status in {"aborted", "interrupted"}:
                return "aborted"
            return "failed"
        return "active-or-incomplete"

    @staticmethod
    def _report_state(path: Path) -> str:
        if next(path.rglob("*.partial"), None) is not None:
            return "active-or-incomplete"
        if any(item.is_file() for item in path.rglob("manifest.json")):
            return "generated"
        return "incomplete"

    @staticmethod
    def _export_state(path: Path, family: str) -> str:
        if next(path.rglob("*.partial"), None) is not None:
            return "active-or-incomplete"
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return "incomplete"
        if (
            isinstance(metadata, dict)
            and metadata.get("schema_version") == EXPORT_METADATA_SCHEMA
            and metadata.get("export_type") == family
        ):
            return "generated"
        return "incomplete"

    def _run_target(self, identifier: str, *, archived: bool = False) -> DataTarget:
        root = self.archive_root / "runs" if archived else self.runs_root
        requested_type = None
        plain = identifier
        if ":" in identifier:
            requested_type, plain = identifier.split(":", 1)
            if requested_type not in {"train", "evaluate"} or not plain:
                raise DataSafetyError("run ID prefix must be train: or evaluate:")
        legacy_train = self.archive_root / "training-runs" if archived else self.data_root / "training-runs"
        roots = (
            (root / requested_type, legacy_train) if requested_type == "train" else
            (root / requested_type,) if requested_type else
            (root / "train", legacy_train, root / "evaluate", root)
        )
        try:
            resolved = resolve_run_reference_across(roots, plain, label="run")
        except RunReferenceUnavailableError as exc:
            raise DataSourceUnavailableError(str(exc)) from exc
        except (RunReferenceAmbiguousError, RunReferenceError) as exc:
            raise DataSafetyError(str(exc)) from exc
        DataManager._identifier(resolved.run_id, "run ID")
        variant = requested_type or ("train" if resolved.path == legacy_train.resolve() / resolved.run_id else resolved.path.parent.name if resolved.path.parent.name in {"train", "evaluate"} else "evaluate")
        allowed_root = root if resolved.path.is_relative_to(root) else resolved.path.parent
        path = self._managed_directory(resolved.path, allowed_root, "run")
        return DataTarget(
            "run", resolved.run_id, path, Path("runs") / variant / resolved.run_id,
            self._run_state(path), variant,
        )

    @staticmethod
    def _run_reference(root: Path, reference: str, label: str):
        try:
            resolved = resolve_run_reference(root, reference, label=label)
            DataManager._identifier(resolved.run_id, f"{label} ID")
            return resolved
        except RunReferenceUnavailableError as exc:
            raise DataSourceUnavailableError(str(exc)) from exc
        except (RunReferenceAmbiguousError, RunReferenceError) as exc:
            raise DataSafetyError(str(exc)) from exc

    def _report_target(self, identifier: str, *, archived: bool = False) -> DataTarget:
        training = identifier.startswith("training:")
        plain = identifier.split(":", 1)[1] if training else identifier
        root = self.archive_root / "exports" / "derived" if archived else self.derived_root
        if training:
            resolved = self._run_reference(root / "training", plain, "training report")
            plain = resolved.run_id
        else:
            resolved = self._run_reference(root, plain, "report")
            plain = resolved.run_id
        relative = Path("training") / plain if training else Path(plain)
        path = self._managed_directory(root / relative, root, "report")
        display = f"training:{plain}" if training else plain
        companion_relative = Path("interactive") / relative
        companion_path = (self.archive_root / "exports" / companion_relative) if archived else (self.exports_root / companion_relative)
        if companion_path.is_symlink():
            raise DataSafetyError(f"managed interactive report contains a symbolic link: {companion_path}")
        return DataTarget(
            "report", display, path,
            Path("exports") / "derived" / relative,
            self._report_state(path),
            companion_path=companion_path,
            companion_relative=Path("exports") / companion_relative,
        )

    def _export_target(self, identifier: str, *, archived: bool = False) -> DataTarget:
        if not isinstance(identifier, str) or ":" not in identifier:
            raise DataSafetyError("export ID must use isolated:NAME or grouped:NAME")
        family, plain = identifier.split(":", 1)
        if family not in {"isolated", "grouped"}:
            raise DataSafetyError("export ID must use isolated:NAME or grouped:NAME")
        root = self.archive_root / "exports" if archived else self.exports_root
        if family == "isolated":
            plain = self._run_reference(root / family, plain, "isolated export").run_id
        else:
            plain = self._identifier(plain, "export ID")
        relative = Path(family) / plain
        path = self._managed_directory(root / relative, root, "export")
        display = f"{family}:{plain}"
        return DataTarget(
            "export", display, path, Path("exports") / relative,
            self._export_state(path, family),
        )

    def _list_runs(self, *, archived: bool) -> list[DataTarget]:
        root = self.archive_root / "runs" if archived else self.runs_root
        if not root.is_dir():
            return []
        targets = []
        for run_type in ("train", "evaluate"):
            typed_root = root / run_type
            if typed_root.is_symlink():
                raise DataSafetyError(f"managed {run_type} run root contains a symbolic link: {typed_root}")
            if not typed_root.is_dir():
                continue
            for path in sorted(typed_root.iterdir()):
                if path.is_symlink():
                    raise DataSafetyError(f"managed {run_type} run root contains a symbolic link: {path}")
                if path.is_dir():
                    targets.append(self._run_target(f"{run_type}:{path.name}", archived=archived))
        legacy_train = self.archive_root / "training-runs" if archived else self.data_root / "training-runs"
        if legacy_train.is_dir():
            for path in sorted(legacy_train.iterdir()):
                if path.is_symlink():
                    raise DataSafetyError(f"legacy training run root contains a symbolic link: {path}")
                if path.is_dir():
                    targets.append(self._run_target(f"train:{path.name}", archived=archived))
        for path in sorted(root.iterdir()):
            if path.is_symlink():
                raise DataSafetyError(f"managed run root contains a symbolic link: {path}")
            if path.is_dir() and path.name not in {"train", "evaluate"}:
                targets.append(self._run_target(path.name, archived=archived))
        return targets

    def _list_reports(self, *, archived: bool) -> list[DataTarget]:
        root = self.archive_root / "exports" / "derived" if archived else self.derived_root
        if not root.is_dir():
            return []
        targets = []
        for path in sorted(root.iterdir()):
            if path.is_symlink():
                raise DataSafetyError(f"managed report root contains a symbolic link: {path}")
            if not path.is_dir():
                continue
            if path.name == "training":
                for item in sorted(path.iterdir()):
                    if item.is_symlink():
                        raise DataSafetyError(f"managed training report root contains a symbolic link: {item}")
                    if item.is_dir():
                        targets.append(self._report_target(f"training:{item.name}", archived=archived))
            else:
                targets.append(self._report_target(path.name, archived=archived))
        return targets

    def _list_exports(self, *, archived: bool) -> list[DataTarget]:
        root = self.archive_root / "exports" if archived else self.exports_root
        if not root.is_dir():
            return []
        targets = []
        for family in ("isolated", "grouped"):
            family_root = root / family
            if family_root.is_symlink():
                raise DataSafetyError(f"managed export root contains a symbolic link: {family_root}")
            if not family_root.is_dir():
                continue
            for path in sorted(family_root.iterdir()):
                if path.is_symlink():
                    raise DataSafetyError(f"managed export root contains a symbolic link: {path}")
                if not path.is_dir():
                    continue
                if not _IDENTIFIER.fullmatch(path.name):
                    raise DataSafetyError(
                        f"managed export root contains an active or invalid entry: {path}"
                    )
                targets.append(self._export_target(f"{family}:{path.name}", archived=archived))
        return targets

    def list(self, *, kind: str = "all", archived: bool = False, detail: bool = False) -> dict[str, object]:
        if kind not in {"runs", "reports", "exports", "all"}:
            raise DataSafetyError("data kind must be runs, reports, exports, or all")
        targets = []
        if kind in {"runs", "all"}:
            targets.extend(self._list_runs(archived=archived))
        if kind in {"reports", "all"}:
            targets.extend(self._list_reports(archived=archived))
        if kind in {"exports", "all"}:
            targets.extend(self._list_exports(archived=archived))
        return {
            "schema_version": DATA_LIST_SCHEMA,
            "scope": "archive" if archived else "active",
            "data_root": str(self.data_root),
            "archive_root": str(self.archive_root),
            "items": [target.document(detail=detail) for target in targets],
        }

    def select(
        self, *, run_id: str | None = None, report_id: str | None = None,
        export_id: str | None = None, all_data: bool = False,
        allow_incomplete: bool = False,
    ) -> tuple[DataTarget, ...]:
        selected = sum(
            value is not None for value in (run_id, report_id, export_id)
        ) + int(all_data)
        if selected != 1:
            raise DataSafetyError("select exactly one of --run, --report, --export, or --all")
        if run_id is not None:
            targets = [self._run_target(run_id)]
        elif report_id is not None:
            targets = [self._report_target(report_id)]
        elif export_id is not None:
            targets = [self._export_target(export_id)]
        else:
            targets = [
                *self._list_runs(archived=False),
                *self._list_reports(archived=False),
                *self._list_exports(archived=False),
            ]
        if not targets:
            raise DataSourceUnavailableError("the selected data set is empty")
        for target in targets:
            if not allow_incomplete and target.state in {"active-or-incomplete", "incomplete"}:
                raise DataSafetyError(
                    f"refusing to modify {target.kind} {target.identifier!r} in state {target.state!r}"
                )
            _directory_inventory(target.path, hashes=False)
            if target.companion_path is not None and target.companion_path.is_dir():
                _directory_inventory(target.companion_path, hashes=False)
        return tuple(targets)

    def preview(self, action: str, *, force: bool = False, **selector) -> dict[str, object]:
        if force and action != "delete":
            raise DataSafetyError("--force is supported only for delete operations")
        targets = self.select(allow_incomplete=force, **selector)
        if action == "delete":
            self._require_delete_permissions(targets)
        return {
            "schema_version": DATA_OPERATION_SCHEMA,
            "action": action,
            "status": "planned",
            "dry_run": True,
            "force": force,
            "target_count": len(targets),
            "targets": [target.document(detail=False) for target in targets],
            "archive_root": str(self.archive_root),
        }

    def delete(self, *, force: bool = False, **selector) -> dict[str, object]:
        targets = self.select(allow_incomplete=force, **selector)
        self._require_delete_permissions(targets)
        deleted = []
        for target in targets:
            try:
                shutil.rmtree(target.path)
                if target.companion_path is not None and target.companion_path.is_dir():
                    shutil.rmtree(target.companion_path)
            except PermissionError as exc:
                raise DataSafetyError(
                    f"delete permission changed after preflight for {target.kind} "
                    f"{target.identifier!r}: {exc.filename or target.path}"
                ) from exc
            deleted.append(target.document(detail=False))
        return {
            "schema_version": DATA_OPERATION_SCHEMA,
            "action": "delete", "status": "completed", "dry_run": False,
            "force": force,
            "target_count": len(deleted), "targets": deleted,
            "archive_root": str(self.archive_root),
        }

    @staticmethod
    def _require_delete_permissions(targets: tuple[DataTarget, ...]) -> None:
        blocked: list[Path] = []
        seen: set[Path] = set()
        for target in targets:
            roots = [target.path]
            if target.companion_path is not None and target.companion_path.is_dir():
                roots.append(target.companion_path)
            directories = [directory for root in roots for directory in (root.parent, root)]
            directories.extend(
                path for root in roots for path in root.rglob("*")
                if path.is_dir() and not path.is_symlink()
            )
            for directory in directories:
                resolved = directory.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                if not os.access(resolved, os.W_OK | os.X_OK):
                    blocked.append(resolved)
        if blocked:
            examples = ", ".join(str(path) for path in blocked[:3])
            remainder = len(blocked) - min(len(blocked), 3)
            suffix = "" if remainder == 0 else f" (and {remainder} more)"
            raise DataSafetyError(
                "delete permission preflight failed before any data was removed; "
                f"directories require write and execute permission: {examples}{suffix}. "
                "Repair ownership or group-write permissions for the managed data roots first"
            )

    def _archive_one(self, target: DataTarget) -> dict[str, object]:
        destination = (self.archive_root / target.archive_relative).resolve()
        if not destination.is_relative_to(self.archive_root):
            raise DataSafetyError("archive destination escapes OVLAB archive root")
        if destination.exists():
            raise DataSafetyError(
                f"archive destination already exists for {target.kind} {target.identifier!r}"
            )
        companion_destination = None
        if target.companion_path is not None and target.companion_path.is_dir():
            assert target.companion_relative is not None
            companion_destination = (self.archive_root / target.companion_relative).resolve()
            if not companion_destination.is_relative_to(self.archive_root) or companion_destination.exists():
                raise DataSafetyError(
                    f"archive destination already exists for interactive report {target.identifier!r}"
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        cross_filesystem = False
        try:
            os.replace(target.path, destination)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            cross_filesystem = True
            stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
            source_inventory = _directory_inventory(target.path, hashes=True)
            try:
                shutil.copytree(target.path, stage, symlinks=False)
                copied_inventory = _directory_inventory(stage, hashes=True)
                if copied_inventory != source_inventory:
                    raise DataSafetyError("cross-filesystem archive verification failed")
                stage.rename(destination)
                shutil.rmtree(target.path)
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                raise
        inventory = _directory_inventory(destination, hashes=False)
        companion_inventory = None
        if companion_destination is not None and target.companion_path is not None:
            companion_destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(target.companion_path, companion_destination)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                cross_filesystem = True
                stage = companion_destination.with_name(f".{companion_destination.name}.{uuid.uuid4().hex}.partial")
                source_inventory = _directory_inventory(target.companion_path, hashes=True)
                try:
                    shutil.copytree(target.companion_path, stage, symlinks=False)
                    if _directory_inventory(stage, hashes=True) != source_inventory:
                        raise DataSafetyError("cross-filesystem interactive archive verification failed")
                    stage.rename(companion_destination)
                    shutil.rmtree(target.companion_path)
                except Exception:
                    shutil.rmtree(stage, ignore_errors=True)
                    raise
            companion_inventory = _directory_inventory(companion_destination, hashes=False)
        entry = {
            "schema_version": ARCHIVE_ENTRY_SCHEMA,
            "kind": target.kind,
            "id": target.identifier,
            "source_path": str(target.path),
            "archive_path": str(destination),
            "archive_relative": str(target.archive_relative),
            "archived_at_utc": _utc_now(),
            "cross_filesystem_copy": cross_filesystem,
            "interactive_archive_path": None if companion_destination is None else str(companion_destination),
            "interactive_file_count": None if companion_inventory is None else companion_inventory["file_count"],
            "interactive_size_bytes": None if companion_inventory is None else companion_inventory["size_bytes"],
            **inventory,
        }
        manifest_name = target.identifier.replace(":", "--") + ".json"
        manifest_root = self.archive_root / "manifests" / target.kind
        if target.run_type is not None:
            entry["run_type"] = target.run_type
            manifest_root = manifest_root / target.run_type
        _atomic_json(manifest_root / manifest_name, entry)
        return entry

    def archive(self, **selector) -> dict[str, object]:
        targets = self.select(**selector)
        destinations = [self.archive_root / target.archive_relative for target in targets]
        destinations.extend(
            self.archive_root / target.companion_relative
            for target in targets
            if target.companion_relative is not None
            and target.companion_path is not None
            and target.companion_path.is_dir()
        )
        collisions = [str(path) for path in destinations if path.exists()]
        if collisions:
            raise DataSafetyError("archive destination already exists: " + ", ".join(collisions))
        archived = [self._archive_one(target) for target in targets]
        return {
            "schema_version": DATA_OPERATION_SCHEMA,
            "action": "archive", "status": "completed", "dry_run": False,
            "target_count": len(archived), "targets": archived,
            "archive_root": str(self.archive_root),
        }
