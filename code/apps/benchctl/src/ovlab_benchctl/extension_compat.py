"""Temporary policy facade while callers migrate to the neutral extension catalog.

The catalog owns registration and discovery. This projection preserves the
current policy worker format during the phased SDK migration; it is not a second
registration authority.
"""
from copy import deepcopy

from .errors import ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError


def policy_manifest(manifest):
    """Project a neutral manifest onto the current policy SDK descriptor format."""
    if manifest.get("schema_version") != "ovlab.extension/v1":
        return manifest
    result = {key: deepcopy(value) for key, value in manifest.items()
              if key not in {"category", "providers"}}
    result.update(schema_version="ovlab.plugin/v1", api_version="1", kind="policy", plugins={})
    for local_id, declaration in manifest.get("providers", {}).items():
        if declaration.get("kind") != "policy":
            continue
        descriptor = deepcopy(declaration)
        descriptor.pop("kind", None)
        descriptor.setdefault("id", local_id)
        descriptor["provider_ref"] = f"plugin:policy:{manifest['id']}/{local_id}"
        config_type = descriptor.get("config_type")
        if not isinstance(config_type, str) or not config_type:
            raise ConfigSchemaError(f"policy provider {local_id} requires config_type")
        if config_type in result["plugins"]:
            raise ConfigSchemaError(f"ambiguous policy config type {config_type!r} in {manifest['id']}")
        result["plugins"][config_type] = descriptor
    return result


def expand_extension_registry(document, root):
    """Project neutral registrations without losing provider/source/build scope.

    Only the worker data shape is transitional. Every runtime identifier remains
    owned by its project, so unrelated equal local names cannot collide.
    """
    from .extensions import ExtensionCatalog, ProviderRef
    from pathlib import Path
    if (Path(root) / "configs/.ovlab-bundle.json").is_file():
        from .plugin_projects import read_document
        return read_document(Path(root) / "configs/plugins/registry.json")
    catalog = ExtensionCatalog(root, document=document)
    result = {"schema_version": "ovlab.plugins/v1", "sources": catalog.infrastructure_sources(),
              "builds": catalog.infrastructure_builds(), "plugins": {}, "benchmarks": {}, "datasets": {}, "projects": {}}
    for name in document["projects"]:
        try:
            manifest = catalog.project_manifest(name)
        except (ConfigReferenceError, ConfigSchemaError, ConfigCompatibilityError):
            # This is an inventory projection, not explicit selection. A broken
            # unrelated checkout must not prevent using a healthy provider.
            # Explicit catalog resolution still reports that project's error.
            continue
        if manifest["category"] != "plugin": continue
        result["projects"][name] = catalog.project_metadata(name)
        result["projects"][name]["kind"] = ",".join(dict.fromkeys(value["kind"] for value in manifest["providers"].values()))
        sources = catalog.project_sources(name)
        for source_id, selection in sources.items():
            if selection["kind"] == "git": selection["path"] = f"external/extensions/{source_id}"
        result["sources"].update(sources)
        for role, recipe in catalog.project_builds(name).items():
            recipe["dockerfile"] = f"external/extensions/{name}/self/{recipe['dockerfile']}"
            recipe["locks"] = [f"external/extensions/{name}/self/{path}" for path in recipe.get("locks", [])]
            result["builds"][role] = recipe
        for local_id, declaration in manifest["providers"].items():
            kind = declaration["kind"]
            if kind not in {"policy", "benchmark", "dataset"}: continue
            ref = ProviderRef("plugin", kind, name, local_id)
            descriptor = catalog.resolve_provider(ref).descriptor
            descriptor.setdefault("id", local_id)
            descriptor.update(project=name, provider_ref=str(ref))
            descriptor.setdefault("capabilities", deepcopy(manifest.get("capabilities", {})))
            for hook in ("training", "preparation"):
                if hook in manifest: descriptor.setdefault(hook, deepcopy(manifest[hook]))
            table = {"policy": "plugins", "benchmark": "benchmarks", "dataset": "datasets"}[kind]
            result[table][str(ref)] = descriptor
    return result
