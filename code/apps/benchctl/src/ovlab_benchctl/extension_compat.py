"""Temporary policy facade while callers migrate to the neutral extension catalog.

The catalog owns registration and discovery. This projection preserves the
current policy worker format during the phased SDK migration; it is not a second
registration authority.
"""
from copy import deepcopy

from .errors import ConfigReferenceError, ConfigSchemaError


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
    """Adapt available policy registrations; unrelated unavailable projects stay lazy."""
    from .extensions import ExtensionCatalog
    from .plugin_projects import expand_projects
    catalog = ExtensionCatalog(root, document=document)
    infrastructure = deepcopy(document.get("infrastructure", {}))
    selected = {}
    for name, registration in document["projects"].items():
        try:
            manifest = catalog.project_manifest(name)
        except ConfigReferenceError:
            # Explicit config/provider selection reports the unavailable project.
            # Inventory of unrelated registrations must never acquire it.
            continue
        if manifest["category"] == "plugin" and any(
            value["kind"] == "policy" for value in manifest.get("providers", {}).values()
        ):
            selected[name] = registration
    legacy = {"schema_version": "ovlab.plugins/v2", "plugins": selected,
              "sources": infrastructure.get("sources", {}), "builds": infrastructure.get("builds", {})}
    return expand_projects(legacy, root, offline=True, catalog=catalog)
