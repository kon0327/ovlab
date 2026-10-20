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
from .strict_yaml import dumps, load


_COMPONENT_KINDS = {
    "benchmark": "benchmark",
    "policy": "policy",
    "metrics": "metric_set",
    "protocol": "protocol",
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
    execution_profile: str | None
    files: tuple[dict[str, object], ...]
    sha256: str
    quantization_override: str | None = None
    overlays: tuple[tuple[str, str], ...] = ()

    def document(self) -> dict[str, object]:
        document = {
            "schema_version": "ovlab-config-bundle/1.0.0",
            "experiment": self.experiment,
            "execution_profile": self.execution_profile,
            "files": [dict(item) for item in self.files],
            "bundle_sha256": self.sha256,
        }
        if self.quantization_override is not None:
            document["quantization_override"] = self.quantization_override
        return document

    def summary(self) -> dict[str, object]:
        summary = {
            "schema_version": "ovlab-config-bundle/1.0.0",
            "bundle_sha256": self.sha256,
            "file_count": len(self.files),
            "container_root": "/opt/ovlab/configs",
            "container_experiment": f"/opt/ovlab/configs/{self.experiment}",
            "container_execution_profile": (
                f"/opt/ovlab/configs/{self.execution_profile}" if self.execution_profile else None
            ),
            "read_only": True,
        }
        if self.quantization_override is not None:
            summary["quantization_override"] = self.quantization_override
        return summary


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

    def load_component(self, reference: str, expected_kind: str) -> dict[str, object]:
        """Resolve inheritance for host orchestration using only the stdlib."""
        return self._extends_closure(reference, expected_kind, set())

    def build(
        self, experiment: str, execution_profile: str | None, *, quantization: str | None = None,
        kind: str = "experiment",
    ) -> ConfigBundle:
        experiment_reference = experiment.removeprefix("configs/")
        execution_reference = execution_profile.removeprefix("configs/") if execution_profile else None
        selected: set[str] = set()
        experiment_document = self._extends_closure(
            experiment_reference, kind, selected
        )
        overlays: dict[str, str] = {}
        if kind == "deployment" and quantization is not None:
            deployment_leaf = load(self.config_root / experiment_reference)
            deployment_leaf.setdefault("runtime", {})["quantization"] = quantization
            overlays[experiment_reference] = dumps(deployment_leaf)
        component_kinds = {"policy": "policy"} if kind == "deployment" else _COMPONENT_KINDS
        for name, component_kind in component_kinds.items():
            reference = experiment_document["components"][name]
            component = self._extends_closure(reference, component_kind, selected)
            if name == "policy":
                if quantization is not None:
                    component["settings"]["runtime"]["quantization"] = quantization
                    validate(component, reference, "policy")
                    policy_leaf = load(self.config_root / reference)
                    policy_leaf.setdefault("settings", {}).setdefault("runtime", {})[
                        "quantization"
                    ] = quantization
                    overlays[reference] = dumps(policy_leaf)
                self._extends_closure(
                    component["settings"]["action"]["action_interface"],
                    "action_interface", selected,
                )
        registry = experiment_document["resources"]["registry"]
        self._extends_closure(registry, "resource_registry", selected)
        if execution_reference is not None:
            self._extends_closure(execution_reference, "execution_profile", selected)

        files = tuple(
            {
                "path": relative,
                "size": (
                    len(overlays[relative].encode("utf-8"))
                    if relative in overlays else (self.config_root / relative).stat().st_size
                ),
                "sha256": (
                    hashlib.sha256(overlays[relative].encode("utf-8")).hexdigest()
                    if relative in overlays else _sha256(self.config_root / relative)
                ),
                **({"source_sha256": _sha256(self.config_root / relative)} if relative in overlays else {}),
            }
            for relative in sorted(selected)
        )
        identity = {
            "schema_version": "ovlab-config-bundle/1.0.0",
            "experiment": experiment_reference,
            "execution_profile": execution_reference,
            "files": files,
        }
        if quantization is not None:
            identity["quantization_override"] = quantization
        canonical = json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return ConfigBundle(
            experiment_reference,
            execution_reference,
            files,
            hashlib.sha256(canonical).hexdigest(),
            quantization,
            tuple(sorted(overlays.items())),
        )

    @contextmanager
    def materialize(self, bundle: ConfigBundle) -> Iterator[Path]:
        root = Path(tempfile.mkdtemp(prefix="ovlab-config-bundle-"))
        try:
            root.chmod(0o755)
            overlays = dict(bundle.overlays)
            for item in bundle.files:
                relative = str(item["path"])
                source = self.config_root / relative
                target = root / relative
                target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                if relative in overlays:
                    target.write_text(overlays[relative], encoding="utf-8")
                else:
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
