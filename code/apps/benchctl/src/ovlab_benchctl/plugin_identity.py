"""Portable identities binding policy images to their registered implementation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .errors import ConfigCompatibilityError, ConfigReferenceError
from .policy_plugins import registry_document
from .plugin_sources import inspect_sources

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


def plugin_build_identity(repository_root, role, *, registry=None, rows=None):
    root = Path(repository_root).resolve()
    registry = registry if registry is not None else registry_document(root)
    build = registry["builds"][role]
    names = build.get("sources", [])
    if rows is None:
        rows = inspect_sources(root, names, registry=registry)
    rows = [row for row in rows if row["id"] in names]
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
        "sources": source_identities(registry, rows), "recipes": recipes,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_plugin_image(labels, repository_root, role):
    if labels.get("io.github.ovlab.role") != role:
        raise ConfigCompatibilityError(f"policy image role differs from registered role {role!r}; rebuild it")
    expected = plugin_build_identity(repository_root, role)
    if labels.get(PLUGIN_BUILD_LABEL) != expected:
        raise ConfigCompatibilityError(f"policy image does not match registered plugin sources/recipe for {role}; rebuild it")
