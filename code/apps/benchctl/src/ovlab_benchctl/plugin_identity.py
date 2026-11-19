"""Portable identities binding policy images to their registered implementation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .errors import ConfigCompatibilityError, ConfigReferenceError
from .policy_plugins import registry_document
from .plugin_sources import inspect_sources, tree_identity

PLUGIN_BUILD_LABEL = "io.github.ovlab.plugin-build.sha256"


def source_identities(registry, rows):
    identities = {}
    for row in rows:
        name = row["id"]
        source = registry["sources"][name]
        if source["kind"] == "git":
            identities[name] = {"kind": "git", "url": source["url"], "revision": source["commit"]}
        else:
            saved = registry.get("source_identities", {}).get(name, {})
            digest = row.get("content_sha256", saved.get("content_sha256"))
            count = row.get("file_count", saved.get("file_count"))
            if not isinstance(digest, str) or len(digest) != 64 or type(count) is not int:
                raise ConfigReferenceError(f"local source {name} has no verifiable content identity; sync its project first")
            identities[name] = {"kind": "local", "content_sha256": digest, "file_count": count}
    return identities


def registered_source_identities(repository_root, names, *, registry):
    """Resolve portable provenance without inspecting Git checkouts at runtime.

    Bundles freeze local content identities on the host. Only an unbundled local
    project needs filesystem inspection; remote identities come from the pins.
    """
    saved = registry.get("source_identities", {})
    local_names = [name for name in names
                   if registry["sources"][name]["kind"] == "local" and name not in saved]
    local_rows = {}
    for name in local_names:
        path = Path(registry["sources"][name]["path"]).expanduser()
        if not path.is_absolute():
            path = Path(repository_root) / path
        if path.is_dir():
            local_rows[name] = {"id": name, **tree_identity(path)}
        else:
            local_rows[name] = inspect_sources(repository_root, [name], registry=registry)[0]
    rows = [local_rows.get(name, {"id": name}) for name in names]
    return source_identities(registry, rows)


def scientific_source_identities(repository_root, names, *, registry):
    identities = registered_source_identities(repository_root, names, registry=registry)
    for identity in identities.values():
        if identity["kind"] == "git":
            # Preserve the scientific configuration format used before bundles.
            identity["commit"] = identity.pop("revision")
    return identities


def plugin_build_identity(repository_root, role, *, registry=None, rows=None):
    root = Path(repository_root).resolve()
    registry = registry if registry is not None else registry_document(root)
    build = registry["builds"][role]
    names = build.get("sources", [])
    if rows is None:
        identities = registered_source_identities(root, names, registry=registry)
    else:
        identities = source_identities(registry, [row for row in rows if row["id"] in names])
    recipes = {}
    for relative in ("deploy/docker/base-images.lock", build["dockerfile"], *build.get("locks", [])):
        # Files inside registered sources are already bound by commit/content.
        if any(Path(relative).is_relative_to(Path("external") / name) for name in names):
            continue
        path = root / relative
        if not path.is_file():
            raise ConfigReferenceError(f"plugin build input is missing: {relative}")
        recipes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "schema_version": "ovlab.plugin-build/v1", "role": role, "build": build,
        "plugins": {key: value for key, value in registry["plugins"].items()
                    if value.get("deployment", {}).get("role") == role},
        "sources": identities, "recipes": recipes,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_plugin_image(labels, repository_root, role):
    if labels.get("io.github.ovlab.role") != role:
        raise ConfigCompatibilityError(f"policy image role differs from registered role {role!r}; rebuild it")
    expected = plugin_build_identity(repository_root, role)
    if labels.get(PLUGIN_BUILD_LABEL) != expected:
        raise ConfigCompatibilityError(f"policy image does not match registered plugin sources/recipe for {role}; rebuild it")
