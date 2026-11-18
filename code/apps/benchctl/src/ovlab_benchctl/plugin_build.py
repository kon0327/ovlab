"""Registry-driven Docker builds, shared by the shell helper and public CLI."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

from .errors import ConfigCompatibilityError, ConfigReferenceError
from .policy_plugins import registry_document, registry_identity
from .plugin_identity import PLUGIN_BUILD_LABEL, plugin_build_identity
from .plugin_sources import inspect_sources, prepared_sources


def _file(root, relative, *, required=True):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ConfigReferenceError(f"build input must stay below the repository: {relative}")
    if required and not path.is_file():
        raise ConfigReferenceError(f"build input is missing: {relative}")
    return path


def _plans(root, selected, registry, rows, *, dry_run):
    plans = []
    for role in selected:
        build = registry["builds"][role]
        dockerfile = _file(root, build["dockerfile"], required=not dry_run)
        for lock in build.get("locks", []):
            _file(root, lock, required=not dry_run)
        plans.append({
            "role": role, "image": build["image"], "dockerfile": str(dockerfile),
            "sources": [row for row in rows if row["id"] in build.get("sources", [])],
            "side_effects_performed": not dry_run,
            "command": ["docker", "build", "--pull=false", "--file", str(dockerfile),
                        "--tag", build["image"], str(root)],
        })
    return plans


def build_images(repository_root, roles=None, *, offline=False, dry_run=False, runner=subprocess.run):
    root = Path(repository_root).resolve()
    registry = registry_document(root)
    registry_sha = registry_identity(root)
    selected = list(registry["builds"]) if roles is None else list(dict.fromkeys(roles))
    unknown = set(selected) - registry["builds"].keys()
    if unknown:
        raise ConfigReferenceError(f"unknown image roles: {', '.join(sorted(unknown))}")
    sources = list(dict.fromkeys(name for role in selected for name in registry["builds"][role].get("sources", [])))
    if dry_run:
        rows = inspect_sources(root, sources, registry=registry)
        return [{**plan, "registry_sha256": registry_sha} for plan in _plans(root, selected, registry, rows, dry_run=True)]
    with prepared_sources(root, sources, offline=offline, registry=registry) as rows:
        plans = _plans(root, selected, registry, rows, dry_run=False)
        spec = importlib.util.spec_from_file_location("ovlab_build_manifest", root / "deploy/scripts/source_manifest.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = module.build_manifest(root, registry=registry, source_rows=rows)
        with tempfile.TemporaryDirectory(prefix="ovlab-plugin-build-") as temporary:
            manifest_path = Path(temporary) / "source-manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            registry_path = Path(temporary) / "plugin-registry.json"
            registry_path.write_text(json.dumps(registry, sort_keys=True))
            for plan in plans:
                role, image = plan["role"], plan["image"]
                build = registry["builds"][role]
                signature = plugin_build_identity(root, role, registry=registry, rows=rows)
                lock_digest = hashlib.sha256()
                for name in ["deploy/docker/base-images.lock", *build.get("locks", [])]:
                    path = _file(root, name)
                    lock_digest.update(name.encode() + b"\0" + path.read_bytes())
                arguments = {
                    "OVLAB_REVISION": manifest["repository_revision"],
                    "OVLAB_SOURCE_MANIFEST_SHA256": manifest["source_content_sha256"],
                    "OVLAB_SOURCE_DIRTY": str(manifest["dirty"]).lower(),
                    "OVLAB_LOCK_SHA256": lock_digest.hexdigest(),
                    "OVLAB_DOCKERFILE_SHA256": hashlib.sha256(Path(plan["dockerfile"]).read_bytes()).hexdigest(),
                    "OVLAB_IMAGE_REFERENCE": image,
                    "OVLAB_VERSION": (root / "VERSION").read_text().strip(),
                }
                command = plan["command"][:-1] + [
                    "--secret", f"id=ovlab_source_manifest,src={manifest_path}",
                    "--secret", f"id=ovlab_plugin_registry,src={registry_path}",
                ]
                for key, value in arguments.items():
                    command += ["--build-arg", f"{key}={value}"]
                labels = {
                    PLUGIN_BUILD_LABEL: signature,
                    "io.github.ovlab.plugin-registry.sha256": registry_sha,
                    "io.github.ovlab.role": role,
                    "io.github.ovlab.source-manifest.sha256": manifest["source_content_sha256"],
                }
                for key, value in labels.items():
                    command += ["--label", f"{key}={value}"]
                command.append(str(root))
                result = runner(command, cwd=root, check=False)
                if result.returncode:
                    raise RuntimeError(f"Docker build failed for {role} ({result.returncode})")
                after = inspect_sources(root, sources, registry=registry, staged=True)
                if after != rows or registry_identity(root) != registry_sha:
                    raise ConfigCompatibilityError("plugin sources or registry changed during build; image must be rebuilt")
                plan.update(command=command, source_sha256=manifest["source_content_sha256"],
                            registry_sha256=registry_sha, plugin_build_sha256=signature)
        return plans
