"""Trainer contract, canonical training runs, and immutable checkpoint bundles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
from typing import Mapping
import uuid

from ovlab_core.experiment_scope import ExperimentScope

from ovlab_policy_sdk.training_support import (TrainingRunContext, TrainerAdapter, inspect_safetensors, safetensors_header as _safetensors_header)

from .run_references import RunReferenceError, resolve_run_reference_tree
from .strict_yaml import dumps
from .training_errors import CheckpointBundleError, TrainingRuntimeError
from .training_identity import atomic_json, canonical_json, identity, inventory, safe_relative, sha256_file


TRAINING_RUN_SCHEMA = "ovlab.training-run/v1"
TRAINING_RESULT_SCHEMA = "ovlab.training-result/v1"
CHECKPOINT_SCHEMA = "ovlab.checkpoint/v1"
CHECKPOINT_MANIFEST_SCHEMA = "ovlab.checkpoint-manifest/v1"
TRAINER_ADAPTER_VERSION = "1.0.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id(profile_id: str, scientific_id: str, scope: ExperimentScope | None = None) -> str:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", profile_id).strip("._-") or "training"
    suffix = scientific_id.rsplit("-", 1)[-1][:8]
    if scope is not None:
        # The final eight characters are a lookup locator, not the scientific
        # identity. Include per-execution entropy so repeated or sibling nodes
        # remain distinct while preserving the canonical readable run-key form.
        payload = f"{scientific_id}:{scope.experiment_id}:{scope.node_path}:{uuid.uuid4().hex}"
        suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{stamp}_{suffix}"


class TrainingRunStore:
    def __init__(
        self, model_data_root: str | Path, *, runs_root: str | Path | None = None,
        training_runs_root: str | Path | None = None, scope: ExperimentScope | None = None,
    ):
        self.model_data_root = Path(model_data_root).expanduser().resolve()
        self.runs_root = Path(runs_root).expanduser().resolve() if runs_root is not None else self.model_data_root / "runs"
        self.scope = scope if scope is not None else ExperimentScope.from_environment()
        self.explicit_root = training_runs_root is not None
        self.root = (Path(training_runs_root).expanduser().resolve() if training_runs_root is not None
                     else self.scope.runs_root(self.runs_root, "train") if self.scope is not None
                     else self.runs_root / "train")

    def _run_root(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or Path(run_id).parts != (run_id,) or run_id in {".", ".."}:
            raise TrainingRuntimeError("training run ID must be one portable path segment")
        direct = self.root / run_id
        if direct.is_symlink():
            raise TrainingRuntimeError(f"training run resolves to a forbidden symbolic link: {run_id}")
        if direct.is_dir():
            return direct.resolve()
        if self.explicit_root:
            raise TrainingRuntimeError(f"training run does not exist: {run_id}")
        try:
            search_root = self.root if self.scope is not None else self.runs_root
            found = resolve_run_reference_tree(search_root, run_id,
                                               category=None if self.scope is not None else "train",
                                               label="training run")
        except RunReferenceError as exc:
            raise TrainingRuntimeError(str(exc)) from exc
        if self.scope is not None and not found.path.is_relative_to(self.root):
            raise TrainingRuntimeError(f"training run does not exist in the selected experiment node: {run_id}")
        return found.path

    def create(
        self,
        profile: Mapping[str, object],
        resolved_profile: Mapping[str, object],
        plan: Mapping[str, object],
        provenance: Mapping[str, object],
    ) -> TrainingRunContext:
        scientific_id = str(plan["scientific_training_id"])
        run_id = _run_id(str(resolved_profile["id"]), scientific_id, self.scope)
        if self.scope is not None:
            candidate = self.runs_root
            for part in (self.scope.experiment_id, "train", *self.scope.node_parts):
                candidate = candidate / part
                if candidate.is_symlink() or not candidate.resolve().is_relative_to(self.runs_root):
                    raise TrainingRuntimeError("experiment run path contains a symbolic link or escapes the runs root")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.scope is not None:
            scoped_parent = self.runs_root / self.scope.experiment_id
            parent = scoped_parent
            for part in ("train", *self.scope.node_parts):
                # The created hierarchy must be writable by the isolated worker.
                # Existing unrelated experiment roots are never touched.
                if parent.is_symlink() or not parent.resolve().is_relative_to(self.runs_root):
                    raise TrainingRuntimeError("experiment run path contains a symbolic link or escapes the runs root")
                parent.chmod(0o2770)
                parent = parent / part
            if parent.is_symlink() or not parent.resolve().is_relative_to(self.runs_root):
                raise TrainingRuntimeError("experiment run path contains a symbolic link or escapes the runs root")
            parent.chmod(0o2770)
        root = self.root / run_id
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise TrainingRuntimeError(f"completed or staged training run already exists: {run_id}") from exc
        # mkdir/touch modes are filtered by the caller's umask. Set modes
        # explicitly so the isolated worker can write through its host group.
        # setgid keeps worker-created artifacts in that shared group too.
        root.chmod(0o2770)
        for directory in ("logs", "staging-checkpoints"):
            child = root / directory
            child.mkdir(mode=0o2770)
            child.chmod(0o2770)
        (root / "original-profile.yaml").write_text(dumps(dict(profile)), encoding="utf-8")
        (root / "resolved-profile.yaml").write_text(dumps(dict(resolved_profile)), encoding="utf-8")
        (root / "original-profile.yaml").chmod(0o660)
        (root / "resolved-profile.yaml").chmod(0o660)
        atomic_json(root / "training-plan.json", dict(plan), mode=0o644)
        atomic_json(root / "provenance.json", {
            "schema_version": TRAINING_RUN_SCHEMA,
            "run_id": run_id,
            "created_at": _utc_now(),
            **({"experiment_scope": {"experiment_id": self.scope.experiment_id,
                                       "node_path": self.scope.node_path}} if self.scope is not None else {}),
            **dict(provenance),
        }, mode=0o644)
        for name in ("events.jsonl", "metrics.jsonl"):
            path = root / name
            path.touch(mode=0o660)
            path.chmod(0o660)
        self.event(root, "planned", {"scientific_training_id": scientific_id})
        return TrainingRunContext(run_id=run_id, root=root, plan=plan)

    @staticmethod
    def _append(path: Path, document: Mapping[str, object]) -> None:
        payload = canonical_json(document)
        with path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def event(self, root: Path, state: str, details: Mapping[str, object] | None = None) -> None:
        self._append(root / "events.jsonl", {
            "schema_version": "ovlab.training-event/v1",
            "state": state,
            "timestamp": _utc_now(),
            "details": dict(details or {}),
        })

    def fail(self, context: TrainingRunContext, exc: BaseException, *, interrupted=False) -> dict[str, object]:
        status = "interrupted" if interrupted else "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)[:4000], "first_raw_excerpt": str(exc)[:1000]}
        self.event(context.root, status, failure)
        result = {
            "schema_version": TRAINING_RESULT_SCHEMA,
            "run_id": context.run_id,
            "status": status,
            "checkpoint_id": None,
            "failure": failure,
        }
        atomic_json(context.root / "result.json", result, exclusive=False, mode=0o644)
        return result

    def list(self) -> list[dict[str, object]]:
        results = []
        roots = (self.root,) if self.scope is not None or self.explicit_root else (self.root, self.runs_root)
        seen: set[Path] = set()
        for path in sorted(path for root in roots if root.is_dir() for path in root.rglob("result.json")):
            if path in seen or not (path.parent / "training-plan.json").is_file() or path.is_symlink():
                continue
            seen.add(path)
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            results.append({**document, "host_path": str(path.parent)})
        return results

    def inspect(self, run_id: str) -> dict[str, object]:
        root = self._run_root(run_id)
        result_path = root / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {"status": "running"}
        plan = json.loads((root / "training-plan.json").read_text(encoding="utf-8"))
        return {"schema_version": TRAINING_RUN_SCHEMA, "run_id": run_id, "host_path": str(root), "result": result, "plan": plan}

    def verify(self, run_id: str) -> dict[str, object]:
        root = self._run_root(run_id)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise TrainingRuntimeError(f"training run is not finalized: {run_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != TRAINING_RUN_SCHEMA or manifest.get("status") != "completed":
            raise TrainingRuntimeError(f"training run is not completed: {run_id}")
        for entry in manifest.get("files", ()):
            path = root / safe_relative(str(entry["path"]))
            if not path.is_file() or path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
                raise TrainingRuntimeError(f"training run integrity failure: {entry['path']}")
        return {"schema_version": "ovlab.training-verification/v1", "run_id": run_id, "status": "verified", "verified_file_count": len(manifest.get("files", ()))}


class CheckpointBundleStore:
    def __init__(self, model_data_root: str | Path):
        self.model_data_root = Path(model_data_root).expanduser().resolve()
        self.legacy_root = self.model_data_root / "checkpoints"
        self.root = self.legacy_root / "local"

    def _existing_path(self, checkpoint_id: str) -> Path | None:
        if not re.fullmatch(r"checkpoint-[0-9a-f]{32}", checkpoint_id):
            raise CheckpointBundleError("checkpoint ID must be an immutable checkpoint-<32 hex> ID")
        candidates = [root / checkpoint_id for root in (self.root, self.legacy_root)]
        for path in candidates:
            if path.is_symlink():
                raise CheckpointBundleError(f"checkpoint path must not be a symlink: {path}")
        found = [path for path in candidates if path.exists()]
        if len(found) > 1:
            raise CheckpointBundleError(f"checkpoint exists in both local and legacy locations: {checkpoint_id}")
        return found[0] if found else None

    @contextmanager
    def _publication_lock(self, checkpoint_id: str):
        # Finalizers in separate containers may publish identical checkpoint
        # content at the same time. Lock only the atomic publication step.
        import fcntl
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f".{checkpoint_id}.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDONLY, 0o644)
        with os.fdopen(descriptor, "rb") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def finalize(self, context: TrainingRunContext) -> dict[str, object]:
        plan = context.plan
        scientific = plan.get("scientific")
        if not isinstance(scientific, dict):
            raise CheckpointBundleError("training plan lacks scientific identity")
        output = scientific.get("output")
        training = scientific.get("training")
        base = scientific.get("base_checkpoint")
        if not isinstance(output, dict) or not isinstance(training, dict) or not isinstance(base, dict):
            raise CheckpointBundleError("training plan lacks checkpoint semantics")
        staging = context.staging_checkpoint
        weights = staging / "weights-or-adapter"
        processor = staging / "processor"
        if not weights.is_dir():
            raise CheckpointBundleError("staged checkpoint lacks weights-or-adapter")
        kind = str(output.get("output_kind"))
        from .training_plugins import backend_for_plan
        backend = backend_for_plan(plan)
        adapter_config, tensors = backend.validate_staging(context)
        files, content_digest, total_size = inventory(staging)
        staged_result_path = context.root / "result.json"
        staged_result = (
            json.loads(staged_result_path.read_text(encoding="utf-8"))
            if staged_result_path.is_file()
            else {}
        )
        parameter_counts = {
            "total": staged_result.get("total_parameter_count"),
            "trainable": staged_result.get("trainable_parameter_count"),
            "frozen": staged_result.get("frozen_parameter_count"),
            "adapter": staged_result.get("adapter_parameter_count"),
            "trainable_adapter": staged_result.get("trainable_adapter_parameter_count"),
        }
        semantics = {
            "schema_version": CHECKPOINT_SCHEMA,
            "kind": "peft_adapter" if kind == "adapter" else "full_checkpoint",
            "output_format": backend.checkpoint_format(scientific),
            "base_checkpoint": base,
            "adapter": adapter_config,
            "planned_peft": training.get("peft") if kind == "adapter" else None,
            "parameter_counts": parameter_counts,
            "tensor_inventory": tensors,
            "content_digest": content_digest,
            "processor": "inherited" if not processor.is_dir() else "bundled",
            "merge_status": "unmerged" if kind == "adapter" else "not_applicable",
            "expected_loader": backend.expected_loader(scientific),
        }
        checkpoint_id = identity("checkpoint", semantics, 32)
        final = self.root / checkpoint_id
        existing_path = self._existing_path(checkpoint_id)
        if existing_path is not None:
            existing = self.inspect(checkpoint_id)
            self.verify(checkpoint_id)
            result = self._complete_run(context, staged_result, checkpoint_id, existing_path)
            return {**result, "checkpoint": existing["checkpoint"], "reused": True}
        temporary = self.root / f".{checkpoint_id}.{uuid.uuid4().hex}.staging"
        self.root.mkdir(parents=True, exist_ok=True)
        temporary.mkdir()
        try:
            for source in sorted(staging.rglob("*")):
                relative = source.relative_to(staging)
                target = temporary / relative
                if source.is_symlink():
                    raise CheckpointBundleError(f"checkpoint staging contains symlink: {relative}")
                if source.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif source.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(source, target)
                    except OSError as exc:
                        if exc.errno != errno.EXDEV:
                            raise
                        # Distinct Docker bind mounts cannot be hardlinked,
                        # even when they share a host filesystem. Publish the
                        # copied bytes through the same verified staging path.
                        shutil.copyfile(source, target)
            checkpoint_document = {**semantics, "checkpoint_id": checkpoint_id, "training_run_id": context.run_id}
            atomic_json(temporary / "checkpoint.json", checkpoint_document, mode=0o644)
            manifest_files, _, manifest_size = inventory(temporary, exclude=("manifest.json",))
            manifest = {
                "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
                "checkpoint_id": checkpoint_id,
                "state": "finalized",
                "kind": semantics["kind"],
                "base_checkpoint": base,
                "merge_status": semantics["merge_status"],
                "adapter": semantics["adapter"],
                "planned_peft": semantics["planned_peft"],
                "parameter_counts": semantics["parameter_counts"],
                "expected_loader": semantics["expected_loader"],
                "content_digest": content_digest,
                "files": manifest_files,
                "total_size": manifest_size,
                "published_at": _utc_now(),
            }
            # The manifest is created last in the hidden staging directory;
            # one rename then publishes a complete, independently verifiable
            # bundle with no visible half-finalized checkpoint window.
            atomic_json(temporary / "manifest.json", manifest)
            for path in sorted(temporary.rglob("*"), reverse=True):
                path.chmod(0o444 if path.is_file() else 0o555)
            temporary.chmod(0o555)
            with self._publication_lock(checkpoint_id):
                existing_path = self._existing_path(checkpoint_id)
                if existing_path is not None:
                    existing = self.inspect(checkpoint_id)
                    self.verify(checkpoint_id)
                    temporary.chmod(0o755)
                    for path in temporary.rglob("*"):
                        if path.is_dir():
                            path.chmod(0o755)
                    shutil.rmtree(temporary)
                    result = self._complete_run(context, staged_result, checkpoint_id, existing_path)
                    return {**result, "checkpoint": existing["checkpoint"], "reused": True}
                os.replace(temporary, final)
                final.chmod(0o555)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        result = self._complete_run(context, staged_result, checkpoint_id, final)
        return {**result, "checkpoint": {**semantics, "checkpoint_id": checkpoint_id}, "reused": False}

    @staticmethod
    def _complete_run(context, staged_result, checkpoint_id, final):
        """Finalize this run even when its immutable checkpoint already exists."""
        if (context.root / "manifest.json").is_file():
            TrainingRunStore(context.root.parent, training_runs_root=context.root.parent).verify(context.run_id)
            previous = json.loads((context.root / "result.json").read_text(encoding="utf-8"))
            if previous.get("checkpoint_id") != checkpoint_id:
                raise CheckpointBundleError("completed training run points to a different checkpoint")
            return previous
        result = {
            **staged_result,
            "schema_version": TRAINING_RESULT_SCHEMA,
            "run_id": context.run_id,
            "status": "completed",
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": str(final),
            "failure": None,
        }
        atomic_json(context.root / "result.json", result, exclusive=False, mode=0o444)
        # Safetensors uses private temporary files. The worker owns these;
        # grant the inherited host group access before sealing the inventory
        # so host-side verification can read copied staging evidence too.
        for path in context.root.rglob("*"):
            if path.is_symlink():
                continue
            info = path.stat()
            shared = stat.S_IRGRP | (stat.S_IXGRP if path.is_dir() else 0)
            if info.st_uid == os.getuid() and info.st_mode & shared != shared:
                path.chmod(stat.S_IMODE(info.st_mode) | shared)
        run_files, run_digest, _ = inventory(context.root, exclude=("manifest.json",))
        atomic_json(context.root / "manifest.json", {
            "schema_version": TRAINING_RUN_SCHEMA,
            "run_id": context.run_id,
            "status": "completed",
            "scientific_training_id": context.plan["scientific_training_id"],
            "execution_plan_id": context.plan["execution_plan_id"],
            "checkpoint_id": checkpoint_id,
            "files": run_files,
            "content_digest": run_digest,
            "finalized_at": _utc_now(),
        })
        return result

    def _manifest(self, checkpoint_id: str) -> tuple[Path, dict[str, object]]:
        root = self._existing_path(checkpoint_id)
        if root is None:
            raise CheckpointBundleError(f"finalized checkpoint does not exist: {checkpoint_id}")
        parent = root.parent
        manifest_path = root / "manifest.json"
        if not root.is_dir() or not root.resolve().is_relative_to(parent.resolve()) or not manifest_path.is_file():
            raise CheckpointBundleError(f"finalized checkpoint does not exist: {checkpoint_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA or manifest.get("state") != "finalized":
            raise CheckpointBundleError(f"checkpoint is not finalized: {checkpoint_id}")
        return root, manifest

    def list(self) -> list[dict[str, object]]:
        results = []
        seen = set()
        for root in (self.root, self.legacy_root):
            if not root.is_dir():
                continue
            for path in sorted(root.glob("checkpoint-*/manifest.json")):
                if path.parent.is_symlink():
                    continue
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if document.get("state") == "finalized":
                    checkpoint_id = path.parent.name
                    if checkpoint_id in seen:
                        raise CheckpointBundleError(f"checkpoint exists in both local and legacy locations: {checkpoint_id}")
                    seen.add(checkpoint_id)
                    results.append({**document, "host_path": str(path.parent)})
        return results

    def inspect(self, checkpoint_id: str) -> dict[str, object]:
        root, manifest = self._manifest(checkpoint_id)
        checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
        return {"manifest": manifest, "checkpoint": checkpoint, "host_path": str(root)}

    def verify(self, checkpoint_id: str) -> dict[str, object]:
        root, manifest = self._manifest(checkpoint_id)
        for entry in manifest.get("files", ()):
            path = root / safe_relative(str(entry["path"]))
            if not path.is_file() or path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
                raise CheckpointBundleError(f"checkpoint integrity failure: {entry['path']}")
        checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
        from .training_plugins import backend_for_checkpoint
        backend_for_checkpoint(checkpoint).verify_checkpoint(root, checkpoint)
        return {
            "schema_version": "ovlab.checkpoint-verification/v1",
            "checkpoint_id": checkpoint_id,
            "status": "verified",
            "verified_file_count": len(manifest.get("files", ())),
            "host_path": str(root),
        }




def __getattr__(name):
    # Compatibility exports are supplied by a registered project, never by a
    # model import in OVLAB's storage/orchestration layer.
    from .training_plugins import extension_symbol
    return extension_symbol("training_runs", name)
