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

from .policy_plugins import with_registry

from .schema import validate
from .strict_yaml import dumps, load
from .config_references import ConfigReferences


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
    sources: tuple[tuple[str, str], ...] = ()

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

    def __init__(self, repository_root: str | Path, *, environment=None) -> None:
        self.environment = environment
        self.repository_root = Path(repository_root).resolve()
        self.config_root = (self.repository_root / "configs").resolve()
        self.references = ConfigReferences(self.repository_root, self.config_root)

    @with_registry
    def _source(self, reference: str) -> Path:
        return self.references.source(reference)

    def _relative(self, path: Path) -> str:
        return self.references.bundle_path(path)

    def _extends_closure(
        self,
        reference: str,
        expected_kind: str,
        selected: set[str],
        stack: tuple[Path, ...] = (),
        *, validate_document: bool = True,
    ) -> dict[str, object]:
        path = self._source(reference).resolve()
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
            parent_path = self.references.parent(path, parent)
            parent_document = self._extends_closure(
                self._relative(parent_path), expected_kind, selected, (*stack, path),
                validate_document=False,
            )
            if "type" in document and document["type"] != parent_document.get("type"):
                raise ValueError(f"{relative}.type must match its parent")
            resolved = _merge(parent_document, document)
        if validate_document:
            validate(resolved, relative, expected_kind, repository_root=self.repository_root)
        return resolved

    @with_registry
    def load_component(self, reference: str, expected_kind: str, *, validate_document: bool = True) -> dict[str, object]:
        """Resolve inheritance for host orchestration using only the stdlib."""
        return self._extends_closure(reference, expected_kind, set(), validate_document=validate_document)

    @with_registry
    def build(
        self, experiment: str, execution_profile: str | None, *, quantization: str | None = None,
        kind: str = "experiment",
    ) -> ConfigBundle:
        experiment_reference = self._relative(self._source(experiment))
        execution_reference = self._relative(self._source(execution_profile)) if execution_profile else None
        selected: set[str] = set()
        experiment_document = self._extends_closure(
            experiment_reference, kind, selected
        )
        if kind == "experiment" and execution_reference is None:
            configured_execution = experiment_document.get("deployment", {}).get("execution_profile")
            if configured_execution:
                execution_reference = self._relative(self._source(configured_execution))
        overlays: dict[str, str] = {}
        if kind == "experiment" and "offline" in experiment_document.get("deployment", {}):
            # Checkpoint acquisition is a host responsibility. Keep its setting
            # in the deployment plan and source hash, while workers receive the
            # portable runtime document. Flatten extends so a parent cannot
            # reintroduce host-only options in older policy images.
            runtime_document = deepcopy(experiment_document)
            runtime_document.pop("extends", None)
            runtime_document["deployment"].pop("offline")
            overlays[experiment_reference] = dumps(runtime_document)
        if kind == "deployment" and quantization is not None:
            deployment_leaf = load(self._source(experiment_reference))
            deployment_leaf.setdefault("runtime", {})["quantization"] = quantization
            overlays[experiment_reference] = dumps(deployment_leaf)
        component_kinds = {"policy": "policy"} if kind == "deployment" else _COMPONENT_KINDS
        benchmark_provider = None
        for name, component_kind in component_kinds.items():
            reference = experiment_document["components"][name]
            component = self._extends_closure(reference, component_kind, selected)
            if name == "benchmark" and component.get("provider"):
                benchmark_provider = component["provider"]
                from .benchmark_plugins import benchmark_for_document, call_benchmark
                owner = benchmark_for_document(component, self.repository_root)
                for dependency in call_benchmark(owner, "bundle_references", component):
                    self._extends_closure(dependency.reference, dependency.kind, selected)
            if name == "policy":
                if quantization is not None:
                    component["settings"]["runtime"]["quantization"] = quantization
                    validate(component, reference, "policy", repository_root=self.repository_root)
                    policy_leaf = load(self._source(reference))
                    policy_leaf.setdefault("settings", {}).setdefault("runtime", {})[
                        "quantization"
                    ] = quantization
                    overlays[self._relative(self._source(reference))] = dumps(policy_leaf)
                self._extends_closure(
                    component["settings"]["action"]["action_interface"],
                    "action_interface", selected,
                )
        registry = experiment_document["resources"]["registry"]
        self._extends_closure(registry, "resource_registry", selected)
        if execution_reference is not None:
            execution_document = self._extends_closure(execution_reference, "execution_profile", selected)
            if benchmark_provider is not None and execution_document.get("provider") != benchmark_provider:
                raise ValueError("execution profile belongs to a different benchmark provider")

        from .policy_plugins import registry_document, plugin_project_root
        extra_sources = {}
        registry_snapshot = deepcopy(registry_document(self.repository_root))
        for project_id, project in registry_snapshot.get("projects", {}).items():
            project["configs_root"] = f"configs/plugins/configs/{project_id}"
        policy_ref = experiment_document["components"]["policy"]
        policy_doc = self.load_component(policy_ref, "policy")
        from .policy_plugins import plugin_for_type
        plugin = (plugin_for_type(policy_doc["type"], self.repository_root, provider=policy_doc.get("provider")).descriptor
                  if policy_doc["type"] != "mock" else {})
        provider_refs = {plugin["provider_ref"]} if plugin.get("provider_ref") else set()
        project_ids = {Path(reference).parts[2] for reference in selected
                       if Path(reference).parts[:2] in {("plugins", "configs"), ("modules", "configs")} and len(Path(reference).parts) > 3}
        if plugin.get("project"):
            project_ids.add(plugin["project"])
        if kind == "experiment":
            benchmark_doc = self.load_component(experiment_document["components"]["benchmark"], "benchmark")
            if benchmark_doc.get("provider"):
                from .extensions import ProviderRef
                project_ids.add(ProviderRef.parse(benchmark_doc["provider"]).project)
                provider_refs.add(benchmark_doc["provider"])
        # A selected catalog may contain artifact metadata validated by another
        # registered project. Its lightweight control code belongs in the worker.
        resource_doc = self.load_component(experiment_document["resources"]["registry"], "resource_registry")
        from .policy_plugins import checkpoint_validator
        for entry in resource_doc["checkpoints"].values():
            owner = checkpoint_validator(entry, self.repository_root, registry=registry_snapshot)
            if owner is not None and owner.descriptor.get("project"):
                project_ids.add(owner.descriptor["project"])
                if owner.descriptor.get("provider_ref"):
                    provider_refs.add(owner.descriptor["provider_ref"])
        from .plugin_projects import read_document
        from .policy_plugins import registry_path
        registration = read_document(registry_path(self.repository_root))
        catalog = None
        if registration.get("schema_version") == "ovlab.plugins/v1" and any(
                value.get("provider_ref") for table in ("plugins", "benchmarks", "datasets")
                for value in registration.get(table, {}).values()):
            from .benchmark_plugins import extension_catalog
            registration = extension_catalog(self.repository_root).document
        if registration.get("schema_version") == "ovlab.extensions/v1":
            from .extensions import ExtensionCatalog
            catalog = ExtensionCatalog(self.repository_root, document=registration)
        for project_id in sorted(project_ids):
            category = catalog.project_manifest(project_id)["category"] if catalog else "plugin"
            project_root = (catalog.project_root(project_id) if catalog else Path(plugin_project_root(project_id, self.repository_root))).resolve()
            manifest_name = catalog.project_metadata(project_id)["manifest"] if catalog else "plugin.json"
            candidates = [project_root / name for name in (manifest_name, "pyproject.toml")]
            if (project_root / "src").is_dir():
                candidates.extend((project_root / "src").rglob("*"))
            # Flat and mixed-layout projects can also expose root modules or
            # packages, independently of model libraries under src/.
            for path in project_root.iterdir():
                if path.is_file() and path.suffix == ".py":
                    candidates.append(path)
                elif path.is_dir() and (path / "__init__.py").is_file():
                    candidates.extend(path.rglob("*"))
            for source in candidates:
                if source.is_symlink():
                    raise ValueError(f"plugin control source cannot be a symlink: {source}")
                if not source.is_file() or "__pycache__" in source.parts or source.suffix in {".pyc", ".pyo"}:
                    continue
                relative = f"{category}s/projects/{project_id}/{source.relative_to(project_root).as_posix()}"
                selected.add(relative)
                extra_sources[relative] = source

        # Plugin-side shared control packages are explicit source dependencies.
        # Freeze their Python code alongside the owning project so workers never
        # need the host checkout or a network fetch to validate configuration.
        from .plugin_projects import control_source_root
        control_sources = set()
        for table in ("plugins", "benchmarks", "datasets"):
            for descriptor in registry_snapshot.get(table, {}).values():
                if descriptor.get("provider_ref") in provider_refs or (not descriptor.get("provider_ref") and descriptor.get("project") in project_ids):
                    control_sources.update(descriptor.get("control_sources", []))
        for source_id in sorted(control_sources):
            source_root = Path(control_source_root(source_id, self.repository_root, registry_snapshot)).resolve()
            candidates = [source_root / "pyproject.toml"]
            if (source_root / "src").is_dir():
                candidates.extend((source_root / "src").rglob("*"))
            for path in source_root.iterdir():
                if path.is_file() and path.suffix == ".py":
                    candidates.append(path)
                elif path.is_dir() and (path / "__init__.py").is_file():
                    candidates.extend(path.rglob("*"))
            for source in candidates:
                if source.is_symlink():
                    raise ValueError(f"plugin control source cannot be a symlink: {source}")
                if not source.is_file() or "__pycache__" in source.parts or source.suffix in {".pyc", ".pyo"}:
                    continue
                relative = f"plugins/sources/{source_id}/{source.relative_to(source_root).as_posix()}"
                selected.add(relative)
                extra_sources[relative] = source

        from .plugin_identity import registered_source_identities
        descriptors = [value for table in ("plugins", "benchmarks", "datasets")
                       for value in registry_snapshot.get(table, {}).values()
                       if value.get("provider_ref") in provider_refs or (not value.get("provider_ref") and value.get("project") in project_ids)]
        registry_snapshot["source_identities"] = registered_source_identities(
            self.repository_root,
            list(dict.fromkeys([name for descriptor in descriptors for name in descriptor.get("sources", [])]
                               + [value["source_id"] for name, value in registry_snapshot.get("projects", {}).items()
                                  if name in project_ids])),
            registry=registry_snapshot,
        )
        for source_id in control_sources:
            if registry_snapshot["sources"][source_id]["kind"] == "local":
                registry_snapshot["sources"][source_id]["path"] = f"configs/plugins/sources/{source_id}"
        # Keep only declarations whose code/resources belong to this run.
        registry_snapshot["projects"] = {name: value for name, value in registry_snapshot.get("projects", {}).items() if name in project_ids}
        registry_snapshot["plugins"] = {name: value for name, value in registry_snapshot["plugins"].items()
                                        if value in descriptors}
        for table in ("benchmarks", "datasets"):
            registry_snapshot[table] = {name: value for name, value in registry_snapshot.get(table, {}).items()
                                        if value in descriptors}
        required_roles = {value["deployment"]["role"] for value in descriptors if value.get("deployment")}
        registry_snapshot["builds"] = {name: value for name, value in registry_snapshot["builds"].items()
                                       if name in required_roles}
        kept_sources = {source for table in ("plugins", "benchmarks", "datasets")
                        for value in registry_snapshot.get(table, {}).values() for source in value.get("sources", [])}
        kept_sources.update(source for value in registry_snapshot["builds"].values() for source in value.get("sources", []))
        kept_sources.update(value["source_id"] for value in registry_snapshot["projects"].values())
        registry_snapshot["sources"] = {name: value for name, value in registry_snapshot["sources"].items() if name in kept_sources}
        for name, value in registry_snapshot["sources"].items():
            if value["kind"] != "local" or name in control_sources:
                continue
            owner = next((project_id for project_id, metadata in registry_snapshot["projects"].items()
                          if metadata["source_id"] == name), None)
            value["path"] = f"configs/plugins/projects/{owner}" if owner else f"external/{name}"
        if catalog is not None:
            refs = [value["provider_ref"] for value in descriptors if value.get("provider_ref")]
            config_refs = [f"{Path(reference).parts[0][:-1]}://{Path(reference).parts[2]}/{'/'.join(Path(reference).parts[3:])}"
                           for reference in selected if Path(reference).parts[:2] in {("plugins", "configs"), ("modules", "configs")}]
            snapshot = catalog.snapshot(provider_refs=refs, config_refs=config_refs)
            portable_locations = {"projects": {}, "sources": {}}
            for project_id, value in snapshot.lock["projects"].items():
                portable_locations["projects"][project_id] = f"configs/{value['category']}s/projects/{project_id}"
                portable_locations["sources"][f"{project_id}/self"] = portable_locations["projects"][project_id]
                for source_name in catalog.project_manifest(project_id).get("sources", {}):
                    source_id = f"{project_id}/{source_name}"
                    if source_id in control_sources:
                        portable_locations["sources"][source_id] = f"configs/plugins/sources/{source_id}"
            selected_registration = {**registration, "projects": {name: value for name, value in registration["projects"].items() if name in project_ids}}
            selected_registration = deepcopy(selected_registration)
            selected_registration["infrastructure"] = {"sources": {}, "builds": {}}
            for project_id, value in selected_registration["projects"].items():
                if value["source"]["kind"] == "local":
                    value["source"]["path"] = portable_locations["projects"][project_id]
            for reference, value in (("extensions/registry.json", selected_registration),
                                     ("extensions/lock.json", {"lock": snapshot.lock, "locations": portable_locations})):
                selected.add(reference)
                overlays[reference] = json.dumps(value, sort_keys=True, indent=2) + "\n"
        policy_gpu = plugin.get("deployment", {}).get("gpu", bool(plugin))
        benchmark_gpu = any(value.get("kind") == "benchmark" and value.get("deployment", {}).get("gpu", False)
                            for value in descriptors)
        gpu = experiment_document.get("runtime", {}).get("gpu", "all" if policy_gpu or benchmark_gpu else None)
        runtime_profile = "plugins/runtime-profile.yaml"
        selected.add(runtime_profile)
        overlays[runtime_profile] = dumps({
            "schema_version": "0.1.0", "kind": "local_profile", "id": "plugin-container-runtime",
            "paths": {"checkpoint_root": "/checkpoints/resolved", "dataset_root": "/datasets", "runs_root": "/var/lib/ovlab/runs"},
            "devices": {"primary_gpu": "cpu" if gpu is None else "cuda:0"},
        })
        plugin_reference = "plugins/registry.json"
        selected.add(plugin_reference)
        overlays[plugin_reference] = json.dumps(registry_snapshot, sort_keys=True, indent=2) + "\n"

        files = tuple(
            {
                "path": relative,
                "size": (
                    len(overlays[relative].encode("utf-8"))
                    if relative in overlays else (extra_sources.get(relative) or self._source(relative)).stat().st_size
                ),
                "sha256": (
                    hashlib.sha256(overlays[relative].encode("utf-8")).hexdigest()
                    if relative in overlays else _sha256(extra_sources.get(relative) or self._source(relative))
                ),
                **({"source_sha256": _sha256(self._source(relative))} if relative in overlays and self._source(relative).is_file() else {}),
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
            tuple((relative, str(source)) for relative, source in sorted(extra_sources.items())),
        )

    @contextmanager
    def materialize(self, bundle: ConfigBundle) -> Iterator[Path]:
        root = Path(tempfile.mkdtemp(prefix="ovlab-config-bundle-"))
        try:
            root.chmod(0o755)
            overlays = dict(bundle.overlays)
            sources = dict(bundle.sources)
            for item in bundle.files:
                relative = str(item["path"])
                source = Path(sources[relative]) if relative in sources else self._source(relative)
                target = root / relative
                target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                if relative in overlays:
                    target.write_text(overlays[relative], encoding="utf-8")
                else:
                    shutil.copyfile(source, target)
                if target.stat().st_size != item["size"] or _sha256(target) != item["sha256"]:
                    raise ValueError(f"configuration or plugin source changed after bundling: {relative}")
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
