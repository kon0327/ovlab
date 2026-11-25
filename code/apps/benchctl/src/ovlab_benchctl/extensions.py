"""Declarative extension discovery, explicit selection and portable provenance.

Reading this catalog never fetches a repository or imports provider code. Source
acquisition is explicit. SDK-specific validation belongs to the selected kind's
adapter, while this module owns names, source bindings and configuration roots.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

from .errors import ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError

REGISTRY_SCHEMA = "ovlab.extensions/v1"
MANIFEST_SCHEMA = "ovlab.extension/v1"
LOCK_SCHEMA = "ovlab.extension-lock/v1"
PLUGIN_KINDS = frozenset({"policy", "benchmark", "service", "dataset", "training"})
MODULE_KINDS = frozenset({"metric", "report"})
_CATEGORIES = {"plugin": PLUGIN_KINDS, "module": MODULE_KINDS}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENTRYPOINT = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _identifier(value, label):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ConfigSchemaError(f"invalid {label}: {value!r}")
    return value


def _relative(value, label):
    if (not isinstance(value, str) or not value or Path(value).is_absolute()
            or ".." in Path(value).parts or "\\" in value or "\0" in value):
        raise ConfigSchemaError(f"{label} must stay inside its project")
    return value


def _inside(root, relative, label):
    relative = _relative(relative, label)
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ConfigReferenceError(f"{label} escapes its project")
    return path


def _strings(value, label):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ConfigSchemaError(f"{label} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ConfigSchemaError(f"{label} contains duplicate references")
    return value


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ConfigSchemaError(f"{label} must be a mapping")
    return value


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result
    try:
        return _mapping(json.loads(Path(path).read_text(), object_pairs_hook=unique), "extension document")
    except (OSError, ValueError) as exc:
        raise ConfigSchemaError(f"cannot read extension document {path}: {exc}") from exc


def _source(source, label):
    _mapping(source, label)
    kind = source.get("kind")
    if kind == "local":
        if set(source) - {"kind", "path"}:
            raise ConfigSchemaError(f"{label} contains unknown local source fields")
        if not isinstance(source.get("path"), str) or not source["path"] or "\0" in source["path"]:
            raise ConfigSchemaError(f"{label} requires a filesystem path")
    elif kind == "git":
        # A legacy path is a staging hint, never part of source identity.
        if set(source) - {"kind", "url", "commit", "path"}:
            raise ConfigSchemaError(f"{label} contains unknown Git source fields")
        if (not isinstance(source.get("url"), str) or not source["url"]
                or source["url"].startswith("-") or "\0" in source["url"]
                or not isinstance(source.get("commit"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", source["commit"])):
            raise ConfigSchemaError(f"{label} requires a Git URL and full lowercase Git commit")
        if "path" in source:
            _relative(source["path"], f"{label} staging path")
    else:
        raise ConfigSchemaError(f"{label} requires kind local or git")
    return source


def registry_path(repository_root, explicit=None, *, environment=None):
    """Choose one registration document; never merge documents by precedence."""
    root = Path(repository_root).resolve()
    values = os.environ if environment is None else environment
    configured = explicit or values.get("OVLAB_EXTENSION_REGISTRY") or values.get("OVLAB_PLUGIN_REGISTRY")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else root / path
    candidates = (
        root / "configs/extensions/registry.json", root / "configs/plugins/registry.json",
        root / "plugins/registry.json", Path("/usr/share/ovlab/extension-registry.json"),
        Path("/usr/share/ovlab/plugin-registry.json"),
        Path(__file__).with_name("plugins") / "registry.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[-1])


def acquire_manifest(repository_root, source, manifest="extension.json", *, offline=False):
    """Acquire a pinned/local manifest before its declared project ID is known.

    This is an explicitly mutating registration primitive. The returned manifest
    is validated under its own ID; no registration document is written.
    """
    _source(source, "extension registration source")
    _relative(manifest, "extension manifest")
    registration = {"source": deepcopy(source), "manifest": manifest}
    pending = ExtensionCatalog(repository_root, document={"schema_version": REGISTRY_SCHEMA,
        "projects": {"acquisition": registration}}, offline=offline)
    path = pending.project_root("acquisition", acquire=True)
    document = _read_json(_inside(path, manifest, "extension manifest"))
    project = _identifier(document.get("id"), "extension project identifier")
    bound = ExtensionCatalog(repository_root, document={"schema_version": REGISTRY_SCHEMA,
        "projects": {project: registration}}, offline=offline)
    bound._validate_manifest(project, document, path)
    return path, document


@dataclass(frozen=True, order=True)
class ProviderRef:
    category: str
    kind: str
    project: str
    provider: str

    def __post_init__(self):
        if self.category not in _CATEGORIES or self.kind not in _CATEGORIES[self.category]:
            raise ConfigSchemaError(f"invalid extension category/kind: {self.category}/{self.kind}")
        _identifier(self.project, "provider project")
        _identifier(self.provider, "provider identifier")

    def __str__(self):
        return f"{self.category}:{self.kind}:{self.project}/{self.provider}"

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ConfigSchemaError("provider reference must be an explicit typed identifier")
        parts = value.split(":")
        owner = parts[-1].split("/")
        if len(parts) != 3 or len(owner) != 2:
            raise ConfigSchemaError(f"invalid qualified provider reference: {value!r}")
        return cls(parts[0], parts[1], owner[0], owner[1])


@dataclass(frozen=True)
class ResolvedProvider:
    ref: ProviderRef
    descriptor: dict
    project_root: Path
    manifest: dict


class ResolvedSnapshot:
    """Keep frozen identity independent of mutable dictionaries handed to callers."""

    def __init__(self, lock, locations):
        self._lock = deepcopy(lock)
        self._locations = deepcopy(locations)

    @property
    def lock(self):
        return deepcopy(self._lock)

    @property
    def locations(self):
        return deepcopy(self._locations)

    @property
    def sha256(self):
        return hashlib.sha256(_canonical(self._lock)).hexdigest()


class UnsupportedProviderKind(ConfigCompatibilityError):
    """The catalog understands metadata for a kind without implementing its SDK."""


def validate_provider_api(provider, *, expected_kind=None, api_version="1"):
    """Check a selected runtime header without importing an SDK or implementation."""
    ref = provider.ref
    if ref.category != "plugin" or ref.kind not in PLUGIN_KINDS:
        raise UnsupportedProviderKind(f"runtime loading is not supported for {ref.category}/{ref.kind}")
    if expected_kind is not None and ref.kind != expected_kind:
        raise ConfigCompatibilityError(f"provider {ref} has kind {ref.kind!r}, expected {expected_kind!r}")
    if provider.descriptor["api_version"] != api_version:
        raise ConfigCompatibilityError(f"provider {ref} requires unsupported {ref.kind} API {provider.descriptor['api_version']!r}")
    if provider.descriptor.get("runtime_status", "implemented") == "unavailable":
        raise ConfigCompatibilityError(f"provider {ref} is unavailable")
    return deepcopy(provider.descriptor)


def _validate_builds(builds, sources, label):
    _mapping(builds, label)
    for name, build in builds.items():
        _identifier(name, f"{label} identifier")
        _mapping(build, f"{label}.{name}")
        _relative(build.get("dockerfile"), f"{label}.{name}.dockerfile")
        image = build.get("image")
        if not isinstance(image, str) or not image or image.startswith("-") or any(c.isspace() for c in image):
            raise ConfigSchemaError(f"{label}.{name} requires a Docker image reference")
        for path in _strings(build.get("locks", []), f"{label}.{name}.locks"):
            _relative(path, f"{label}.{name} dependency lock")
        refs = _strings(build.get("sources", []), f"{label}.{name}.sources")
        if set(refs) - sources:
            raise ConfigSchemaError(f"{label}.{name} references unknown sources")


class ExtensionCatalog:
    """One explicit registration snapshot with lazy per-project discovery."""

    def __init__(self, repository_root, document=None, registry_path=None, *, environment=None, offline=False):
        self.root = Path(repository_root).resolve()
        self.path = globals()["registry_path"](self.root, registry_path, environment=environment)
        self.offline = offline
        self.frozen = (self.root / "configs/.ovlab-bundle.json").is_file()
        self._frozen_snapshot = None
        if self.frozen:
            lock_path = self.root / "configs/extensions/lock.json"
            if lock_path.is_file():
                self._frozen_snapshot = _read_json(lock_path)
        self._document = deepcopy(document) if document is not None else _read_json(self.path)
        self._validate_registry()

    @property
    def document(self):
        return deepcopy(self._document)

    def _validate_registry(self):
        document = _mapping(self._document, "extension registry")
        if document.get("schema_version") != REGISTRY_SCHEMA:
            raise ConfigSchemaError(f"extension registry schema_version must be {REGISTRY_SCHEMA}")
        if set(document) - {"schema_version", "projects", "infrastructure"}:
            raise ConfigSchemaError("extension registry contains unknown fields")
        for name, registration in _mapping(document.get("projects"), "extension projects").items():
            _identifier(name, "extension project identifier")
            _mapping(registration, f"extension {name} registration")
            if set(registration) - {"source", "manifest"}:
                raise ConfigSchemaError(f"extension {name} registration contains unknown fields")
            _source(registration.get("source"), f"extension {name} source")
            _relative(registration.get("manifest", "extension.json"), "extension manifest")
        infrastructure = _mapping(document.get("infrastructure", {}), "extension infrastructure")
        if set(infrastructure) - {"sources", "builds"}:
            raise ConfigSchemaError("extension infrastructure contains unknown fields")
        sources = _mapping(infrastructure.get("sources", {}), "infrastructure sources")
        for name, source in sources.items():
            _identifier(name, "infrastructure source identifier")
            _source(source, f"infrastructure source {name}")
        _validate_builds(infrastructure.get("builds", {}), set(sources), "infrastructure builds")

    def _registration(self, project):
        _identifier(project, "extension project identifier")
        try:
            return self._document["projects"][project]
        except KeyError as exc:
            raise ConfigReferenceError(f"unknown extension project: {project}") from exc

    def _cache(self, source):
        selection = {key: source[key] for key in ("kind", "url", "commit")}
        digest = hashlib.sha256(_canonical(selection)).hexdigest()
        cache = self.root / "plugins/.cache/extensions" / digest
        current = self.root
        for part in cache.relative_to(self.root).parts:
            current /= part
            if current.is_symlink():
                raise ConfigReferenceError("extension cache cannot contain symlinks")
        return cache

    def _git_root(self, source, *, acquire):
        cache = self._cache(source)
        path = cache / "external/project"
        if path.is_symlink() or (cache / "external").is_symlink():
            raise ConfigReferenceError("extension checkout cannot be a symlink")
        if path.is_dir():
            from .plugin_sources import _git, _uncommitted_files
            if not (path / ".git").exists():
                raise ConfigCompatibilityError(f"extension cache is not a Git checkout: {path}")
            origin = _git(path, "remote", "get-url", "origin")
            normalize = lambda value: value.rstrip("/").removesuffix(".git")
            if normalize(origin) != normalize(source["url"]):
                raise ConfigCompatibilityError("cached extension origin differs from its registered source")
            if _git(path, "rev-parse", "HEAD") != source["commit"] or _uncommitted_files(path):
                raise ConfigCompatibilityError("cached extension differs from its pinned commit; register edited code as a local source")
            return path
        if not acquire:
            raise ConfigReferenceError(f"extension source {source['url']}@{source['commit']} has no cached manifest; acquire it explicitly")
        from .plugin_sources import prepared_sources
        with prepared_sources(cache, ["project"], registry={"sources": {"project": source}}, offline=self.offline):
            return path

    def project_root(self, project, *, acquire=False):
        if self.frozen:
            if self._frozen_snapshot is None or project not in self._frozen_snapshot["lock"]["projects"]:
                raise ConfigReferenceError(f"project {project} is outside the frozen extension closure")
            relative = self._frozen_snapshot["locations"]["projects"].get(project)
            if relative is None:
                raise ConfigReferenceError(f"frozen project {project} has no bundled location")
            path = _inside(self.root, relative, "frozen project")
            if not path.is_dir():
                raise ConfigReferenceError(f"frozen extension project is missing: {project}")
            return path
        source = self._registration(project)["source"]
        if source["kind"] == "git":
            return self._git_root(source, acquire=acquire)
        path = Path(source["path"]).expanduser()
        path = (path if path.is_absolute() else self.root / path).resolve()
        if not path.is_dir():
            raise ConfigReferenceError(f"extension project does not exist: {path}")
        return path

    def project_manifest(self, project, *, acquire=False):
        root = self.project_root(project, acquire=acquire)
        relative = self._registration(project).get("manifest", "extension.json")
        manifest = _read_json(_inside(root, relative, "extension manifest"))
        self._validate_manifest(project, manifest, root)
        if self.frozen:
            expected = self._frozen_snapshot["lock"]["projects"][project]["manifest_sha256"]
            if hashlib.sha256(_canonical(manifest)).hexdigest() != expected:
                raise ConfigCompatibilityError(f"frozen extension manifest changed: {project}")
        return manifest

    def _validate_manifest(self, project, manifest, root):
        allowed = {"schema_version", "id", "category", "version", "description", "providers", "sources",
                   "builds", "configs", "config_aliases", "capabilities", "training", "preparation"}
        if set(manifest) - allowed:
            raise ConfigSchemaError(f"extension {project} manifest contains unknown fields")
        if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("id") != project:
            raise ConfigSchemaError(f"extension {project} requires a matching {MANIFEST_SCHEMA} manifest")
        category = manifest.get("category")
        if category not in _CATEGORIES:
            raise ConfigSchemaError(f"extension {project} requires category plugin or module")
        if not isinstance(manifest.get("version"), str) or not manifest["version"]:
            raise ConfigSchemaError(f"extension {project} requires a package version")
        for field in ("capabilities", "training", "preparation"):
            if field in manifest:
                _mapping(manifest[field], f"extension {project}.{field}")
        sources = _mapping(manifest.get("sources", {}), f"extension {project} sources")
        if "self" in sources:
            raise ConfigSchemaError(f"extension {project} source self is reserved")
        for name, source in sources.items():
            _identifier(name, "extension source identifier")
            _source(source, f"extension {project} source {name}")
            if source["kind"] == "local" and self._registration(project)["source"]["kind"] == "git":
                path = Path(source["path"]).expanduser()
                path = (path if path.is_absolute() else root / path).resolve()
                if not path.is_relative_to(root):
                    raise ConfigCompatibilityError(f"remote extension {project} local dependencies must stay inside its project")
        known_sources = {"self", *sources}
        _validate_builds(manifest.get("builds", {}), known_sources, f"extension {project} builds")
        for name, provider in _mapping(manifest.get("providers"), f"extension {project} providers").items():
            _identifier(name, "extension provider identifier")
            _mapping(provider, f"extension {project} provider {name}")
            if provider.get("kind") not in _CATEGORIES[category]:
                raise ConfigSchemaError(f"extension {project}/{name} kind does not match category {category}")
            if not isinstance(provider.get("api_version"), str) or not provider["api_version"]:
                raise ConfigSchemaError(f"extension {project}/{name} requires a provider API version")
            status = provider.get("runtime_status", "implemented")
            if status not in {"implemented", "experimental", "unavailable"}:
                raise ConfigSchemaError(f"extension {project}/{name} has an invalid runtime_status")
            if status != "unavailable" and (not isinstance(provider.get("entrypoint"), str) or not _ENTRYPOINT.fullmatch(provider["entrypoint"])):
                raise ConfigSchemaError(f"extension {project}/{name} requires module:object entrypoint")
            if "config_type" in provider:
                _identifier(provider["config_type"], "provider configuration type")
            refs = _strings(provider.get("sources", []), f"extension {project}/{name}.sources")
            if set(refs) - known_sources:
                raise ConfigSchemaError(f"extension {project}/{name} references unknown sources")
            controls = _strings(provider.get("control_sources", []), f"extension {project}/{name}.control_sources")
            if set(controls) - {"self", *refs}:
                raise ConfigSchemaError(f"extension {project}/{name}.control_sources must select declared sources")
            deployment = provider.get("deployment")
            if deployment is not None:
                _mapping(deployment, f"extension {project}/{name} deployment")
                role = deployment.get("role")
                if role not in manifest.get("builds", {}):
                    raise ConfigSchemaError(f"extension {project}/{name} references an unknown build role")
                build = manifest["builds"][role]
                if deployment.get("image") != build["image"]:
                    raise ConfigSchemaError(f"extension {project}/{name} image differs from its build recipe")
                if set(refs) - {"self", *build.get("sources", [])}:
                    raise ConfigSchemaError(f"extension {project}/{name} build omits required provider sources")
                if not isinstance(deployment.get("image_env"), str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", deployment["image_env"]):
                    raise ConfigSchemaError(f"extension {project}/{name} requires an image environment variable")
        configs = _mapping(manifest.get("configs", {}), f"extension {project} configs")
        if set(configs) - {"root", "templates", "checkpoints"}:
            raise ConfigSchemaError(f"extension {project} configs contains unknown fields")
        config_root = _inside(root, configs.get("root", "configs"), "extension configuration root")
        for category, paths in _mapping(configs.get("templates", {}), "extension templates").items():
            _identifier(category, "template category")
            for relative in _strings(paths, "extension templates"):
                _inside(config_root, relative, "extension template")
        for relative in _strings(configs.get("checkpoints", []), "extension checkpoint catalogs"):
            _inside(config_root, relative, "extension checkpoint catalog")
        aliases = manifest.get("config_aliases", {})
        if isinstance(aliases, str):
            aliases = _read_json(_inside(root, aliases, "extension configuration aliases"))
        for source, target in _mapping(aliases, "extension configuration aliases").items():
            _relative(source, "extension alias")
            _inside(config_root, target, "extension alias target")

    def project_metadata(self, project, *, acquire=False):
        manifest = self.project_manifest(project, acquire=acquire)
        configs = manifest.get("configs", {})
        aliases = manifest.get("config_aliases", {})
        if isinstance(aliases, str):
            aliases = _read_json(_inside(self.project_root(project), aliases, "extension configuration aliases"))
        return {"id": project, "category": manifest["category"], "version": manifest["version"],
                "description": manifest.get("description", ""), "source_id": f"{project}/self",
                "manifest": self._registration(project).get("manifest", "extension.json"),
                "configs_root": configs.get("root", "configs"), "config_aliases": deepcopy(aliases),
                "checkpoint_catalogs": deepcopy(configs.get("checkpoints", [])),
                "config_templates": deepcopy(configs.get("templates", {})),
                "capabilities": deepcopy(manifest.get("capabilities", {})),
                "providers": [str(ProviderRef(manifest["category"], item["kind"], project, name))
                              for name, item in manifest["providers"].items()]}

    def list_projects(self, category=None):
        if category is not None and category not in _CATEGORIES:
            raise ConfigSchemaError(f"unknown extension category: {category}")
        rows = []
        for project in sorted(self._document["projects"]):
            row = {"id": project, "source": deepcopy(self._registration(project)["source"])}
            try:
                row.update(self.project_metadata(project), status="available")
            except ConfigReferenceError as exc:
                row.update(category=None, status="unavailable", error=str(exc))
            except (ConfigSchemaError, ConfigCompatibilityError) as exc:
                row.update(category=None, status="invalid", error=str(exc))
            if category is None or row["category"] == category:
                rows.append(row)
        return rows

    def resolve_provider(self, reference, *, expected_kind=None, acquire=False):
        ref = ProviderRef.parse(reference)
        if self.frozen and (self._frozen_snapshot is None or
                            str(ref) not in self._frozen_snapshot.get("lock", {}).get("providers", {})):
            raise ConfigReferenceError(f"provider {ref} is absent from the frozen selection")
        if expected_kind is not None and ref.kind != expected_kind:
            raise ConfigCompatibilityError(f"provider {ref} has kind {ref.kind!r}, expected {expected_kind!r}")
        manifest = self.project_manifest(ref.project, acquire=acquire)
        descriptor = manifest["providers"].get(ref.provider)
        if manifest["category"] != ref.category or not isinstance(descriptor, dict) or descriptor["kind"] != ref.kind:
            raise ConfigReferenceError(f"unknown qualified extension provider: {ref}")
        descriptor = deepcopy(descriptor)
        descriptor["sources"] = [f"{ref.project}/{name}" for name in dict.fromkeys(["self", *descriptor.get("sources", [])])]
        descriptor["control_sources"] = [f"{ref.project}/{name}" for name in descriptor.get("control_sources", [])]
        def qualify_roles(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "role" and isinstance(item, str):
                        value[key] = f"{ref.project}/{item}"
                    else:
                        qualify_roles(item)
            elif isinstance(value, list):
                for item in value:
                    qualify_roles(item)
        for section in ("deployment", "training", "preparation"):
            qualify_roles(descriptor.get(section))
        return ResolvedProvider(ref, descriptor, self.project_root(ref.project), deepcopy(manifest))

    def project_sources(self, project, *, acquire=False):
        manifest = self.project_manifest(project, acquire=acquire)
        root = self.project_root(project)
        result = {f"{project}/self": deepcopy(self._registration(project)["source"])}
        if result[f"{project}/self"]["kind"] == "local":
            result[f"{project}/self"]["path"] = str(root)
        for name, selection in manifest.get("sources", {}).items():
            selection = deepcopy(selection)
            if selection["kind"] == "local":
                path = Path(selection["path"]).expanduser()
                selection["path"] = str((path if path.is_absolute() else root / path).resolve())
            result[f"{project}/{name}"] = selection
        return result

    def project_builds(self, project, *, acquire=False):
        manifest = self.project_manifest(project, acquire=acquire)
        result = {}
        for name, recipe in manifest.get("builds", {}).items():
            recipe = deepcopy(recipe)
            recipe["project"] = project
            recipe["sources"] = [f"{project}/{source}" for source in dict.fromkeys(["self", *recipe.get("sources", [])])]
            result[f"{project}/{name}"] = recipe
        return result

    def infrastructure_sources(self):
        result = {}
        for name, source in self._document.get("infrastructure", {}).get("sources", {}).items():
            source = deepcopy(source)
            if source["kind"] == "local":
                path = Path(source["path"]).expanduser()
                source["path"] = str((path if path.is_absolute() else self.root / path).resolve())
            result[f"@infrastructure/{name}"] = source
        return result

    def infrastructure_builds(self):
        result = {}
        for name, build in self._document.get("infrastructure", {}).get("builds", {}).items():
            build = deepcopy(build)
            build["sources"] = [f"@infrastructure/{source}" for source in build.get("sources", [])]
            result[f"@infrastructure/{name}"] = build
        return result

    def source_root(self, source_id, *, acquire=False):
        if self.frozen:
            if self._frozen_snapshot is None or source_id not in self._frozen_snapshot["lock"]["sources"]:
                raise ConfigReferenceError(f"source {source_id} is outside the frozen extension closure")
            relative = self._frozen_snapshot["locations"]["sources"].get(source_id)
            if relative is None:
                raise ConfigReferenceError(f"source {source_id} has no bundled control location")
            path = _inside(self.root, relative, "frozen control source")
            if not path.is_dir():
                raise ConfigReferenceError(f"frozen extension source is missing: {source_id}")
            return path
        owner, name = self._owned_ref(source_id, "source")
        sources = self.infrastructure_sources() if owner == "@infrastructure" else self.project_sources(owner, acquire=acquire)
        if source_id not in sources:
            raise ConfigReferenceError(f"unknown extension source: {source_id}")
        selection = sources[source_id]
        if selection["kind"] == "git":
            return self._git_root(selection, acquire=acquire)
        path = Path(selection["path"])
        if not path.is_dir():
            raise ConfigReferenceError(f"extension source does not exist: {path}")
        return path.resolve()

    @staticmethod
    def _owned_ref(reference, label):
        if not isinstance(reference, str) or len(reference.split("/")) != 2:
            raise ConfigSchemaError(f"{label} requires a project/local identifier")
        owner, name = reference.split("/")
        if owner != "@infrastructure":
            _identifier(owner, f"{label} owner")
        _identifier(name, f"{label} identifier")
        return owner, name

    @staticmethod
    def config_parts(reference):
        if not isinstance(reference, str):
            raise ConfigReferenceError("extension configuration reference must be a URI")
        parsed = urlsplit(reference)
        if (parsed.scheme not in _CATEGORIES or not _NAME.fullmatch(parsed.netloc)
                or parsed.query or parsed.fragment or not parsed.path.startswith("/")
                or not parsed.path.lstrip("/") or "%" in parsed.path or "\\" in parsed.path
                or ".." in Path(parsed.path).parts):
            raise ConfigReferenceError(f"invalid extension configuration reference: {reference}")
        return parsed.scheme, parsed.netloc, parsed.path.lstrip("/")

    def config_path(self, reference, *, acquire=False):
        category, project, relative = self.config_parts(reference)
        manifest = self.project_manifest(project, acquire=acquire)
        if manifest["category"] != category:
            raise ConfigReferenceError(f"configuration URI category differs from extension {project}")
        root = (_inside(self.root, f"configs/{category}s/configs/{project}", "frozen configurations") if self.frozen else
                _inside(self.project_root(project), manifest.get("configs", {}).get("root", "configs"), "extension configuration root"))
        return _inside(root, relative, "extension configuration")

    def config_roots(self, category=None):
        roots = {}
        if self.frozen:
            if self._frozen_snapshot is None:
                raise ConfigReferenceError("frozen extension lock is missing")
            for project, metadata in self._frozen_snapshot["lock"]["projects"].items():
                if category is None or metadata["category"] == category:
                    roots[project] = _inside(self.root, f"configs/{metadata['category']}s/configs/{project}", "frozen configurations")
            return roots
        for row in self.list_projects(category=category):
            if row["status"] == "available":
                roots[row["id"]] = _inside(self.project_root(row["id"]), row["configs_root"], "extension configuration root")
        return roots

    def snapshot(self, project_ids=(), provider_refs=(), config_refs=(), *, build_refs=(), acquire=False):
        """Freeze explicit selected declarations, independently of host locations.

        The caller supplies the transitive YAML/provider selection closure. A
        project selected directly includes its providers/builds; a config URI
        selects only its owner. Git dependency identity needs its pin, not fetch.
        """
        projects = set(project_ids)
        full_projects = set(project_ids)
        refs = {ProviderRef.parse(ref) for ref in provider_refs}
        projects.update(ref.project for ref in refs)
        for uri in config_refs:
            self.config_path(uri, acquire=acquire)
            projects.add(self.config_parts(uri)[1])
        selected_builds = set(build_refs)
        for reference in selected_builds:
            owner, _ = self._owned_ref(reference, "build")
            if owner != "@infrastructure":
                projects.add(owner)
        lock = {"schema_version": LOCK_SCHEMA, "projects": {}, "providers": {}, "sources": {}, "builds": {}}
        locations = {"projects": {}, "sources": {}}
        sources = {}
        selected_sources = set()
        for project in sorted(projects):
            manifest = self.project_manifest(project, acquire=acquire)
            root = self.project_root(project)
            locations["projects"][project] = str(root)
            lock["projects"][project] = {"category": manifest["category"], "version": manifest["version"],
                "manifest": self._registration(project).get("manifest", "extension.json"),
                "manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(), "source_id": f"{project}/self"}
            sources.update(self.project_sources(project))
            selected_sources.add(f"{project}/self")
            if project in full_projects:
                refs.update(ProviderRef(manifest["category"], item["kind"], project, name)
                            for name, item in manifest["providers"].items())
                selected_builds.update(self.project_builds(project))
        for ref in sorted(refs):
            provider = self.resolve_provider(ref)
            lock["providers"][str(ref)] = deepcopy(provider.descriptor)
            selected_sources.update(provider.descriptor["sources"])
        infrastructure_builds = self.infrastructure_builds()
        for reference in sorted(selected_builds):
            owner, _ = self._owned_ref(reference, "build")
            builds = infrastructure_builds if owner == "@infrastructure" else self.project_builds(owner)
            if reference not in builds:
                raise ConfigReferenceError(f"unknown extension build: {reference}")
            lock["builds"][reference] = builds[reference]
            selected_sources.update(builds[reference]["sources"])
        sources.update(self.infrastructure_sources())
        from .plugin_sources import tree_identity
        for source_id in sorted(selected_sources):
            source = sources[source_id]
            if source["kind"] == "git":
                lock["sources"][source_id] = {key: source[key] for key in ("kind", "url", "commit")}
                # Locators are optional for pinned dependencies not yet acquired.
                try:
                    locations["sources"][source_id] = str(self.source_root(source_id))
                except ConfigReferenceError:
                    pass
            else:
                path = Path(source["path"])
                lock["sources"][source_id] = {"kind": "local", **tree_identity(path)}
                locations["sources"][source_id] = str(path)
        return ResolvedSnapshot(lock, locations)

    def load_provider(self, reference, *, expected_kind=None, api_version="1", contract=None, acquire=False):
        """Load selected control code; the kind adapter supplies its SDK contract."""
        provider = self.resolve_provider(reference, expected_kind=expected_kind, acquire=acquire)
        validate_provider_api(provider, expected_kind=expected_kind, api_version=api_version)
        roots = [provider.project_root]
        roots.extend(self.source_root(source, acquire=acquire) for source in provider.descriptor["control_sources"])
        candidates = [candidate for root in roots for candidate in (root, root / "src") if candidate.is_dir()]
        # The adapter's own package takes priority over its explicit dependencies.
        for candidate in reversed(candidates):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        module, attribute = provider.descriptor["entrypoint"].split(":")
        expected = set()
        for candidate in candidates:
            target = candidate.joinpath(*module.split("."))
            expected.update(path.resolve() for path in (target.with_suffix(".py"), target / "__init__.py") if path.is_file())
        if not expected:
            raise ConfigReferenceError(f"provider {provider.ref} entrypoint is missing from its selected sources")
        try:
            loaded = importlib.import_module(module)
            loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
            if loaded_file not in expected:
                from .plugin_sources import tree_identity
                namespace = module.split('.')[0]
                loaded_package = next((parent for parent in loaded_file.parents if parent.name == namespace), None)
                matches = loaded_package is not None and any(
                    (candidate / namespace).is_dir() and tree_identity(candidate / namespace) == tree_identity(loaded_package)
                    for candidate in candidates)
                if not matches:
                    raise ConfigCompatibilityError(f"provider {provider.ref} was imported from a different source; restart with the selected catalog")
            implementation = getattr(loaded, attribute)
            if isinstance(implementation, type):
                implementation = implementation()
        except (ImportError, AttributeError, TypeError) as exc:
            raise ConfigReferenceError(f"cannot load provider {provider.ref}: {exc}") from exc
        if str(getattr(implementation, "api_version", "")) != api_version:
            raise ConfigCompatibilityError(f"provider {provider.ref} implementation has incompatible API version")
        if contract is not None and not isinstance(implementation, contract):
            raise ConfigCompatibilityError(f"provider {provider.ref} does not implement the selected SDK contract")
        return implementation
