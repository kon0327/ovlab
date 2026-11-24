"""Acquire standalone plugin projects and expand their declarative manifests."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

from .errors import ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError

_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def read_document(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result
    try:
        document = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    except (OSError, ValueError) as exc:
        raise ConfigSchemaError(f"cannot read plugin document {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigSchemaError(f"plugin document must be a mapping: {path}")
    return document


def _source(source):
    if not isinstance(source, dict) or source.get("kind") not in {"local", "git"}:
        raise ConfigSchemaError("plugin project requires a local or git source")
    if source["kind"] == "local":
        if not isinstance(source.get("path"), str) or not source["path"]:
            raise ConfigSchemaError("local plugin project requires a filesystem path")
    elif (not isinstance(source.get("url"), str) or not source["url"] or source["url"].startswith("-")
          or not isinstance(source.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", source["commit"])):
        raise ConfigSchemaError("remote plugin project requires a Git URL and full lowercase Git commit")
    return source


def acquire_project(root, name, source, *, offline=False):
    """Read local projects directly; cache pinned remote projects independently."""
    root = Path(root).resolve()
    _source(source)
    if not _NAME.fullmatch(name):
        raise ConfigSchemaError(f"invalid plugin project identifier: {name}")
    if source["kind"] == "local":
        path = Path(source["path"]).expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_dir():
            raise ConfigReferenceError(f"plugin project does not exist: {path}")
        return path
    cache = root / "plugins/.cache" / name / source["commit"]
    if (root / "plugins").is_symlink() or (root / "plugins/.cache").is_symlink():
        raise ConfigReferenceError("plugin project cache cannot be a symlink")
    # Source preparation accepts an explicit registry and does not recurse into
    # project discovery. Its lock and atomic publication protect this cache.
    from .plugin_sources import prepared_sources
    source_id = f"plugin-{name}"
    project = cache / "external" / source_id
    if project.is_dir() and (project / "plugin.json").is_file():
        from .plugin_sources import _git, _uncommitted_files
        if _git(project, "rev-parse", "HEAD") != source["commit"] or _uncommitted_files(project):
            raise ConfigCompatibilityError(f"cached plugin {name} differs from its pinned commit; register edited code as a local project")
        return project
    if offline:
        raise ConfigReferenceError(f"plugin {name}@{source['commit']} has no cached manifest; acquire it before offline or dry-run use")
    with prepared_sources(cache, [source_id], registry={"sources": {source_id: source}}, offline=offline):
        return project


def _inside(root, relative, label):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ConfigSchemaError(f"{label} must stay inside the plugin project")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ConfigReferenceError(f"{label} escapes the plugin project")
    return path


def _merge(target, values, label):
    if not isinstance(values, dict):
        raise ConfigSchemaError(f"plugin {label} must be a mapping")
    for name, value in values.items():
        if name in target and target[name] != value:
            raise ConfigCompatibilityError(f"conflicting plugin {label} registration: {name}")
        target[name] = value


def _strings(value, label):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ConfigSchemaError(f"{label} must be a list of non-empty strings")
    return value


def expand_projects(document, root, *, offline=False, catalog=None):
    """Convert compact project registrations into the common runtime registry."""
    registrations = document.get("plugins", {})
    if not isinstance(registrations, dict):
        raise ConfigSchemaError("plugin project registrations must be a mapping")
    if set(document) - {"schema_version", "plugins", "sources", "builds"}:
        raise ConfigSchemaError("plugin project registry contains unknown fields")
    if any(not isinstance(document.get(key, {}), dict) for key in ("sources", "builds")):
        raise ConfigSchemaError("plugin project sources and builds must be mappings")
    result = {"schema_version": "ovlab.plugins/v1", "sources": deepcopy(document.get("sources", {})),
              "builds": deepcopy(document.get("builds", {})), "plugins": {}, "projects": {}}
    for name, registration in registrations.items():
        if not isinstance(registration, dict) or not isinstance(registration.get("source"), dict):
            raise ConfigSchemaError(f"plugin {name} requires a project source")
        if set(registration) - {"source", "manifest", "offline"} or type(registration.get("offline", False)) is not bool:
            raise ConfigSchemaError(f"plugin {name} has invalid project registration fields")
        source = deepcopy(_source(registration["source"]))
        project = (catalog.project_root(name) if catalog is not None else
                   acquire_project(root, name, source, offline=offline or registration.get("offline", False)))
        manifest_name = registration.get("manifest", "plugin.json")
        manifest = read_document(_inside(project, manifest_name, "plugin manifest"))
        from .extension_compat import policy_manifest
        manifest = policy_manifest(manifest)
        if (manifest.get("schema_version") != "ovlab.plugin/v1" or manifest.get("api_version") != "1"
                or manifest.get("kind") != "policy" or manifest.get("id") != name):
            raise ConfigSchemaError(f"plugin {name} requires a matching policy manifest with SDK API 1")
        if not isinstance(manifest.get("version"), str) or not manifest["version"]:
            raise ConfigSchemaError(f"plugin {name} requires a version")
        source_id = f"plugin-{name}"
        _merge(result["sources"], {source_id: source}, "sources")
        upstreams = deepcopy(manifest.get("sources", {}))
        if not isinstance(upstreams, dict):
            raise ConfigSchemaError(f"plugin {name} sources must be a mapping")
        for upstream in upstreams.values():
            _source(upstream)
            if upstream["kind"] == "local":
                path = Path(upstream["path"]).expanduser()
                path = (path if path.is_absolute() else project / path).resolve()
                if source["kind"] == "git" and not path.is_relative_to(project):
                    raise ConfigCompatibilityError(f"remote plugin {name} local dependencies must stay inside its project; register external code as a pinned Git source")
                upstream["path"] = str(path)
        _merge(result["sources"], upstreams, "sources")
        builds = deepcopy(manifest.get("builds", {}))
        if not isinstance(builds, dict) or any(not isinstance(build, dict) for build in builds.values()):
            raise ConfigSchemaError(f"plugin {name} builds must be a mapping of recipes")
        for build in builds.values():
            for key in ("dockerfile",):
                _inside(project, build.get(key), f"plugin build {key}")
                build[key] = f"external/{source_id}/{build[key]}"
            build["locks"] = [f"external/{source_id}/{path}" for path in _strings(build.get("locks", []), "plugin dependency locks")
                              if _inside(project, path, "plugin dependency lock")]
            build["sources"] = list(dict.fromkeys([source_id, *_strings(build.get("sources", []), "plugin build sources")]))
            build["project"] = name
        _merge(result["builds"], builds, "builds")
        descriptors = deepcopy(manifest.get("plugins", {}))
        if not isinstance(descriptors, dict) or any(not isinstance(item, dict) for item in descriptors.values()):
            raise ConfigSchemaError(f"plugin {name} policy types must be a mapping")
        for descriptor in descriptors.values():
            descriptor["sources"] = list(dict.fromkeys([source_id, *_strings(descriptor.get("sources", []), "plugin policy sources")]))
            descriptor["project"] = name
            descriptor.setdefault("capabilities", manifest.get("capabilities", {}))
            for hook in ("training", "preparation"):
                if hook in manifest:
                    descriptor.setdefault(hook, deepcopy(manifest[hook]))
        _merge(result["plugins"], descriptors, "policy types")
        configs = manifest.get("configs", {})
        if not isinstance(configs, dict) or not isinstance(manifest.get("capabilities", {}), dict):
            raise ConfigSchemaError(f"plugin {name} configs and capabilities must be mappings")
        config_root = configs.get("root", "configs")
        _inside(project, config_root, "plugin configuration root")
        templates = configs.get("templates", {})
        if not isinstance(templates, dict):
            raise ConfigSchemaError(f"plugin {name} config templates must map categories to paths")
        for category, paths in templates.items():
            if not isinstance(category, str) or not _NAME.fullmatch(category):
                raise ConfigSchemaError(f"plugin {name} has an invalid template category")
            for path in _strings(paths, "plugin configuration templates"):
                _inside(project / config_root, path, "plugin configuration template")
        aliases = manifest.get("config_aliases", {})
        if isinstance(aliases, str):
            aliases = read_document(_inside(project, aliases, "plugin configuration aliases"))
        if not isinstance(aliases, dict):
            raise ConfigSchemaError(f"plugin {name} configuration aliases must be a mapping")
        result["projects"][name] = {
            "kind": manifest["kind"], "description": manifest.get("description", ""),
            "source_id": source_id, "manifest": manifest_name, "version": manifest["version"],
            "configs_root": config_root, "config_aliases": aliases,
            "capabilities": deepcopy(manifest.get("capabilities", {})),
            "checkpoint_catalogs": deepcopy(configs.get("checkpoints", [])),
            "config_templates": deepcopy(templates),
        }
    return result


def project_root(name, root, registry, *, acquire=True):
    root = Path(root).resolve()
    metadata = registry.get("projects", {}).get(name)
    if not isinstance(metadata, dict):
        raise ConfigReferenceError(f"unknown plugin project: {name}")
    bundled = root / "configs/plugins/projects" / name
    if (root / "configs/.ovlab-bundle.json").is_file() and bundled.is_dir():
        return bundled
    source_id = metadata["source_id"]
    source = registry["sources"][source_id]
    if source["kind"] == "local":
        original = Path(source["path"]).expanduser()
        if not original.is_absolute():
            original = root / original
        if original.is_dir():
            return original.resolve()
    if source["kind"] == "git":
        cached = root / "plugins/.cache" / name / source["commit"] / "external" / source_id
        if cached.is_dir():
            return cached.resolve()
    staged = root / "external" / source_id
    if staged.is_dir() and source_id in registry.get("source_identities", {}):
        return staged.resolve()
    from importlib.metadata import entry_points
    for entry in entry_points(group="ovlab.plugin_projects"):
        if entry.name == name:
            try:
                installed = Path(entry.load()()).resolve()
            except (ImportError, OSError) as exc:
                raise ConfigReferenceError(f"cannot locate installed plugin {name}: {exc}") from exc
            if (installed / "plugin.json").is_file():
                return installed
    if not acquire:
        raise ConfigReferenceError(f"plugin project {name} is unavailable in this runtime")
    return acquire_project(root, name, source)


def control_source_root(name, root, registry, *, acquire=True):
    """Locate an explicit Python control dependency independently of external/.

    Pinned dependencies use the same isolated source preparation as builds. They
    may be shared across model projects without depending on a development
    checkout, and frozen workers use their bundled local source directly.
    """
    root = Path(root).resolve()
    from .policy_plugins import _OFFLINE
    acquire = acquire and not _OFFLINE.get()
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ConfigSchemaError(f"invalid control source identifier: {name}")
    try:
        source = registry["sources"][name]
    except KeyError as exc:
        raise ConfigReferenceError(f"unknown control source: {name}") from exc
    _source(source)
    bundled = root / "configs/plugins/sources" / name
    if (root / "configs/.ovlab-bundle.json").is_file() and bundled.is_dir():
        if bundled.is_symlink() or not bundled.resolve().is_relative_to(root):
            raise ConfigReferenceError("bundled control source escapes its runtime root")
        return bundled.resolve()
    if source["kind"] == "local":
        path = Path(source["path"]).expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_dir():
            raise ConfigReferenceError(f"control source does not exist: {path}")
        return path
    cache = root / "plugins/.cache/control" / name / source["commit"]
    for parent in (root / "plugins", root / "plugins/.cache", root / "plugins/.cache/control",
                   cache.parent, cache):
        if parent.is_symlink():
            raise ConfigReferenceError("control source cache cannot contain symlinks")
    from .plugin_sources import prepared_sources
    with prepared_sources(cache, [name], registry={"sources": {name: source}}, offline=not acquire):
        return cache / "external" / name


def register_project(root, location, *, commit=None, registry_path=None, dry_run=False):
    """Add a complete project after validating its manifest and registrations."""
    candidate = Path(location).expanduser()
    candidate = candidate if candidate.is_absolute() else Path(root) / candidate
    for manifest_name in ('extension.json', 'plugin.json'):
        manifest_path = candidate / manifest_name
        if manifest_path.is_file() and read_document(manifest_path).get('schema_version') == 'ovlab.extension/v1':
            return register_extension_project(root, location, commit=commit, registry_path=registry_path, dry_run=dry_run)
    root = Path(root).resolve()
    path = Path(location).expanduser()
    local = path if path.is_absolute() else root / path
    if local.is_dir() and commit is None:
        source = {"kind": "local", "path": str(local.resolve())}
        project = local.resolve()
    else:
        source = {"kind": "git", "url": str(location), "commit": commit}
        _source(source)
        if dry_run:
            return {"project": None, "registry": str(registry_path or root / "configs/plugins/registry.json"),
                    "source": source, "action": "fetch-manifest-and-register", "side_effects_performed": False}
        # Acquisition is keyed by a temporary name until the manifest declares ID.
        project = acquire_project(root, "registration", source)
    manifest = read_document(project / "plugin.json")
    name = manifest.get("id")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ConfigSchemaError("plugin manifest requires a valid project id")
    target = Path(registry_path) if registry_path else root / "configs/plugins/registry.json"
    if not target.is_absolute():
        target = root / target
    if target.is_file():
        registry = read_document(target)
    else:
        from .policy_plugins import registry_path as selected_registry
        registry = read_document(selected_registry(root))
    if registry.get("schema_version") != "ovlab.plugins/v2":
        raise ConfigCompatibilityError("project registration requires an ovlab.plugins/v2 registry")
    if not isinstance(registry.setdefault("plugins", {}), dict):
        raise ConfigSchemaError("plugin project registrations must be a mapping")
    existing = registry["plugins"].get(name)
    if existing is not None and (not isinstance(existing, dict) or not isinstance(existing.get("source"), dict)):
        raise ConfigSchemaError(f"plugin {name} has invalid existing registration")
    registration = {"source": source}
    equivalent = existing == registration
    if existing is not None and source["kind"] == "local" and existing.get("source", {}).get("kind") == "local":
        previous = Path(existing["source"]["path"]).expanduser()
        previous = previous if previous.is_absolute() else root / previous
        equivalent = previous.resolve() == local.resolve() and existing.get("manifest", "plugin.json") == "plugin.json"
    if existing is not None and not equivalent:
        raise ConfigCompatibilityError(f"plugin {name} is already registered with another source")
    registry["plugins"][name] = existing if existing is not None else registration
    from .policy_plugins import registry_document
    expanded = registry_document(root, document=registry)
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        import os
        import tempfile
        descriptor, temporary = tempfile.mkstemp(prefix=".plugin-registry-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(registry, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {"project": name, "registry": str(target), "source": source,
            "policy_types": [key for key, value in expanded["plugins"].items() if value.get("project") == name],
            "side_effects_performed": not dry_run}


def register_extension_project(root, location, *, commit=None, registry_path=None,
                               dry_run=False, category='plugin'):
    """Register one explicitly acquired project in the shared extension registry."""
    from .extensions import ExtensionCatalog, acquire_manifest, registry_path as default_registry
    root = Path(root).resolve()
    local = Path(location).expanduser()
    local = local if local.is_absolute() else root / local
    manifest_name = 'extension.json'
    if local.is_dir() and commit is None:
        if not (local / manifest_name).is_file() and (local / 'plugin.json').is_file():
            manifest_name = 'plugin.json'
        manifest = read_document(local / manifest_name)
        if manifest.get('schema_version') != 'ovlab.extension/v1':
            if category == 'plugin':
                return register_project(root, location, registry_path=registry_path, dry_run=dry_run)
            raise ConfigSchemaError('module registration requires an ovlab.extension/v1 manifest')
        source = {'kind': 'local', 'path': str(local.resolve())}
    else:
        source = {'kind': 'git', 'url': str(location), 'commit': commit}
        _source(source)
        if dry_run:
            return {'project': None, 'source': source, 'category': category,
                    'action': 'acquire-manifest-and-register', 'side_effects_performed': False}
        _, manifest = acquire_manifest(root, source, manifest_name)
    name = manifest.get('id')
    if manifest.get('category') != category:
        raise ConfigCompatibilityError(f'expected a {category} project, found {manifest.get("category")!r}')
    target = Path(registry_path).expanduser() if registry_path else root / 'configs/extensions/registry.json'
    target = target if target.is_absolute() else root / target
    document = read_document(target) if target.is_file() else read_document(default_registry(root))
    if document.get('schema_version') != 'ovlab.extensions/v1':
        raise ConfigCompatibilityError('extension registration requires an ovlab.extensions/v1 registry')
    registration = {'source': source, 'manifest': manifest_name}
    existing = document['projects'].get(name)
    if existing is not None:
        prior = deepcopy(existing)
        if prior.get('source', {}).get('kind') == source['kind'] == 'local':
            original = Path(prior['source']['path']).expanduser()
            prior['source']['path'] = str((original if original.is_absolute() else root / original).resolve())
        if prior != registration:
            raise ConfigCompatibilityError(f'extension {name} is already registered with another source')
    document['projects'][name] = existing or registration
    catalog = ExtensionCatalog(root, document=document)
    checked = catalog.project_manifest(name)
    if not dry_run:
        import os
        import tempfile
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix='.extension-registry-', dir=target.parent)
        try:
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(document, stream, indent=2)
                stream.write('\n')
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {'project': name, 'category': category, 'registry': str(target), 'source': source,
            'providers': list(checked['providers']), 'side_effects_performed': not dry_run}
