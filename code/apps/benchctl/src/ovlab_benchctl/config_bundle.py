"""Validated, minimal configuration bundles for isolated deployments."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterator

from .schema import validate
from .strict_yaml import load


_COMPONENT_KINDS = {
    "benchmark": "benchmark",
    "policy": "policy",
    "metrics": "metric_set",
    "protocol": "protocol",
    "action_interface": "action_interface",
    "artifacts": "artifact_store",
}


def _merge(parent: dict[str, object], child: dict[str, object]) -> dict[str, object]:
    result = deepcopy(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ConfigBundle:
    """Immutable identity and file closure for one deployment configuration."""

    experiment: str
    execution_profile: str
    files: tuple[dict[str, object], ...]
    sha256: str

    def document(self) -> dict[str, object]:
        return {
            "schema_version": "ovlab-config-bundle/1.0.0",
            "experiment": self.experiment,
            "execution_profile": self.execution_profile,
            "files": [dict(item) for item in self.files],
            "bundle_sha256": self.sha256,
        }

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": "ovlab-config-bundle/1.0.0",
            "bundle_sha256": self.sha256,
            "file_count": len(self.files),
            "container_root": "/opt/ovlab/configs",
            "container_experiment": f"/opt/ovlab/configs/{self.experiment}",
            "container_execution_profile": f"/opt/ovlab/configs/{self.execution_profile}",
            "read_only": True,
        }


class ConfigBundleBuilder:
    """Collect and materialize only the transitive YAML closure for one run."""

    def __init__(self, repository_root: str | Path) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.config_root = (self.repository_root / "configs").resolve()
        if not self.config_root.is_dir():
            raise ValueError(f"config root does not exist: {self.config_root}")

    def _relative(self, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.config_root):
            raise ValueError(f"configuration escapes configs/: {path}")
        return resolved.relative_to(self.config_root).as_posix()

    def _extends_closure(
        self,
        reference: str,
        expected_kind: str,
        selected: set[str],
        stack: tuple[Path, ...] = (),
    ) -> dict[str, object]:
        path = (self.config_root / reference).resolve()
        relative = self._relative(path)
        if path in stack:
            chain = " -> ".join(self._relative(item) for item in (*stack, path))
            raise ValueError(f"configuration extends cycle: {chain}")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"configuration must be a regular file: {path}")
        document = load(path)
        if document.get("kind") != expected_kind:
            raise ValueError(
                f"{relative}.kind must equal {expected_kind!r}"
            )
        selected.add(relative)
        parent = document.get("extends")
        if parent is None:
            resolved = document
        else:
            if not isinstance(parent, str) or not parent:
                raise ValueError(f"{relative}.extends must be a non-empty string")
            parent_path = (path.parent / parent).resolve()
            parent_document = self._extends_closure(
                self._relative(parent_path), expected_kind, selected, (*stack, path)
            )
            if document.get("type") != parent_document.get("type"):
                raise ValueError(f"{relative}.type must match its parent")
            resolved = _merge(parent_document, document)
        validate(resolved, relative, expected_kind)
        return resolved

    def build(self, experiment: str, execution_profile: str) -> ConfigBundle:
        experiment_reference = experiment.removeprefix("configs/")
        execution_reference = execution_profile.removeprefix("configs/")
        selected: set[str] = set()
        experiment_document = self._extends_closure(
            experiment_reference, "experiment", selected
        )
        for name, kind in _COMPONENT_KINDS.items():
            reference = experiment_document["components"][name]
            self._extends_closure(reference, kind, selected)
        registry = experiment_document["resources"]["registry"]
        self._extends_closure(registry, "resource_registry", selected)
        self._extends_closure(
            execution_reference, "execution_profile", selected
        )

        files = tuple(
            {
                "path": relative,
                "size": (self.config_root / relative).stat().st_size,
                "sha256": _sha256(self.config_root / relative),
            }
            for relative in sorted(selected)
        )
        identity = {
            "schema_version": "ovlab-config-bundle/1.0.0",
            "experiment": experiment_reference,
            "execution_profile": execution_reference,
            "files": files,
        }
        canonical = json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return ConfigBundle(
            experiment_reference,
            execution_reference,
            files,
            hashlib.sha256(canonical).hexdigest(),
        )

    @contextmanager
    def materialize(self, bundle: ConfigBundle) -> Iterator[Path]:
        root = Path(tempfile.mkdtemp(prefix="ovlab-config-bundle-"))
        try:
            root.chmod(0o755)
            for item in bundle.files:
                relative = str(item["path"])
                source = self.config_root / relative
                target = root / relative
                target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                target.chmod(0o444)
            manifest = root / ".ovlab-bundle.json"
            manifest.write_text(
                json.dumps(bundle.document(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest.chmod(0o444)
            yield root
        finally:
            shutil.rmtree(root, ignore_errors=False)
