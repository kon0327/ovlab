"""Registry-driven Docker builds, shared by the shell helper and public CLI."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tarfile
import tempfile

from .errors import ConfigCompatibilityError, ConfigReferenceError
from .policy_plugins import registry_document, registry_identity
from .plugin_identity import PLUGIN_BUILD_LABEL, plugin_build_identity
from .plugin_sources import inspect_sources, prepared_sources


def _copy_core_context(root, destination):
    """Snapshot OVLAB inputs without copying any managed source workspace."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        text=True, capture_output=True, check=True,
    )
    copied = []
    for relative in sorted(set(result.stdout.split("\0")) - {""}):
        path = Path(relative)
        if path.parts[0] in {"external", ".git"}:
            continue
        source, target = root / path, destination / path
        if source.is_symlink():
            if not source.resolve().is_relative_to(root):
                raise ConfigReferenceError(f"build input symlink escapes the repository: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(source))
            copied.append(relative)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(relative)
    return copied


def _verify_core_context(workspace, copied, manifest, included):
    """Ensure provenance describes the bytes frozen before source acquisition."""
    files = {row["path"]: row for row in manifest["files"]}
    if {name for name in copied if included(name)} != files.keys():
        raise ConfigCompatibilityError("OVLAB inputs changed while preparing the build context; retry the build")
    for name, row in files.items():
        path = workspace / name
        matches = (path.is_symlink() and path.readlink().as_posix() == row["symlink"]
                   if "symlink" in row else
                   path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"])
        if not matches:
            raise ConfigCompatibilityError(f"OVLAB input changed while preparing the build context: {name}; retry the build")


def _registered_rows(rows, registry):
    # Resolving relative local paths for an isolated workspace must not alter
    # the portable registration recorded in provenance.
    return [{**row, "selection": dict(registry["sources"][row["id"]])} for row in rows]


@contextmanager
def _build_sources(root, sources, registry, *, offline, isolated):
    if not isolated:
        with prepared_sources(root, sources, offline=offline, registry=registry) as rows:
            yield root, registry, rows, None
        return
    with tempfile.TemporaryDirectory(prefix="ovlab-isolated-build-") as temporary:
        workspace = Path(temporary)
        copied = _copy_core_context(root, workspace)
        source_registry = deepcopy(registry)
        for name in sources:
            source = source_registry["sources"][name]
            if source["kind"] == "local":
                path = Path(source["path"]).expanduser()
                source["path"] = str(path if path.is_absolute() else root / path)
        with prepared_sources(workspace, sources, offline=offline, registry=source_registry,
                              git_cache_root=root if offline else None) as rows:
            yield workspace, source_registry, _registered_rows(rows, registry), copied


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


def build_images(repository_root, roles=None, *, offline=False, dry_run=False, isolated_sources=True,
                 config=None, execution_profile=None, runner=subprocess.run):
    root = Path(repository_root).resolve()
    registry = registry_document(root, offline=offline or dry_run)
    selected = list(registry["builds"]) if roles is None else list(dict.fromkeys(roles))
    bundle = builder = None
    if config is not None:
        from .config_bundle import ConfigBundleBuilder
        from .strict_yaml import load
        from .policy_plugins import plugin_for_type
        builder = ConfigBundleBuilder(root)
        reference = str(config).removeprefix("configs/")
        kind = load(builder.references.source(reference)).get("kind")
        if kind not in {"experiment", "deployment"}:
            raise ConfigReferenceError("image configuration must be an experiment or deployment")
        document = builder.load_component(reference, kind)
        policy = builder.load_component(document["components"]["policy"], "policy")
        configured_role = plugin_for_type(policy["type"], root, provider=policy.get("provider")).descriptor.get("deployment", {}).get("role")
        if configured_role not in selected:
            raise ConfigCompatibilityError("selected image roles do not provide the configured policy")
        selected = [role for role in selected if role == configured_role or not registry["builds"][role].get("project")]
        if roles is None:
            selected = [configured_role]
        if execution_profile is None and kind == "experiment" and "deployment" in document:
            execution_profile = {"egl": "ovlab://profiles/egl.yaml", "glfw": "ovlab://profiles/glfw.yaml"}.get(document["deployment"].get("renderer", "egl"))
        bundle = builder.build(reference, execution_profile, kind=kind)
    unknown = set(selected) - registry["builds"].keys()
    if unknown:
        raise ConfigReferenceError(f"unknown image roles: {', '.join(sorted(unknown))}")
    registry_sha = registry_identity(root, roles=selected)
    sources = list(dict.fromkeys(name for role in selected for name in registry["builds"][role].get("sources", [])))
    if dry_run:
        workspace = Path("/<temporary-build-context>") if isolated_sources else root
        rows = [{"id": name, "kind": registry["sources"][name]["kind"],
                 "selection": dict(registry["sources"][name]),
                 "acquisition": "fetch" if registry["sources"][name]["kind"] == "git" else "copy"}
                for name in sources] if isolated_sources else inspect_sources(root, sources, registry=registry)
        return [{**plan, "registry_sha256": registry_sha, "isolated_sources": isolated_sources,
                 **({"config_bundle_sha256": bundle.sha256} if bundle is not None else {})}
                for plan in _plans(workspace, selected, registry, rows, dry_run=True)]
    with _build_sources(root, sources, registry, offline=offline, isolated=isolated_sources) as (workspace, source_registry, rows, copied):
        plans = _plans(workspace, selected, registry, rows, dry_run=False)
        module = runpy.run_path(str(root / "deploy/scripts/source_manifest.py"))
        manifest = module["build_manifest"](root, registry=registry, source_rows=rows)
        if copied is not None:
            _verify_core_context(workspace, copied, manifest, module["_included"])
        with tempfile.TemporaryDirectory(prefix="ovlab-plugin-build-") as temporary:
            manifest_path = Path(temporary) / "source-manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            registry_path = Path(temporary) / "plugin-registry.json"
            image_registry = deepcopy(registry)
            from .plugin_identity import source_identities
            image_registry["source_identities"] = source_identities(registry, rows)
            for row in rows:
                if row["kind"] == "local":
                    image_registry["sources"][row["id"]]["path"] = f"/opt/ovlab/external/{row['id']}"
            registry_path.write_text(json.dumps(image_registry, sort_keys=True))
            config_path = None
            if bundle is not None:
                config_path = Path(temporary) / "config-bundle.tar"
                with builder.materialize(bundle) as configuration:
                    with tarfile.open(config_path, "w") as archive:
                        for path in sorted(configuration.rglob("*")):
                            if path.is_file():
                                archive.add(path, arcname=path.relative_to(configuration).as_posix(), recursive=False)
            for plan in plans:
                role, image = plan["role"], plan["image"]
                build = registry["builds"][role]
                recipe = Path(plan["dockerfile"]).read_text()
                if bundle is not None and any(value not in recipe for value in ("ovlab_config_bundle", "OVLAB_CONFIG_BUNDLE_SHA256")):
                    raise ConfigCompatibilityError(f"configured build recipe for {role} must consume ovlab_config_bundle and its OVLAB_CONFIG_BUNDLE_SHA256 build argument")
                signature = plugin_build_identity(workspace, role, registry=registry, rows=rows)
                lock_digest = hashlib.sha256()
                for name in ["deploy/docker/base-images.lock", *build.get("locks", [])]:
                    path = _file(workspace, name)
                    lock_digest.update(name.encode() + b"\0" + path.read_bytes())
                arguments = {
                    "OVLAB_REVISION": manifest["repository_revision"],
                    "OVLAB_SOURCE_MANIFEST_SHA256": manifest["source_content_sha256"],
                    "OVLAB_SOURCE_DIRTY": str(manifest["dirty"]).lower(),
                    "OVLAB_LOCK_SHA256": lock_digest.hexdigest(),
                    "OVLAB_DOCKERFILE_SHA256": hashlib.sha256(Path(plan["dockerfile"]).read_bytes()).hexdigest(),
                    "OVLAB_IMAGE_REFERENCE": image,
                    "OVLAB_VERSION": (workspace / "VERSION").read_text().strip(),
                    "OVLAB_CONFIG_BUNDLE_SHA256": bundle.sha256 if bundle is not None else "none",
                }
                command = plan["command"][:-1] + [
                    "--secret", f"id=ovlab_source_manifest,src={manifest_path}",
                    "--secret", f"id=ovlab_plugin_registry,src={registry_path}",
                ]
                if config_path is not None:
                    command += ["--secret", f"id=ovlab_config_bundle,src={config_path}"]
                for key, value in arguments.items():
                    command += ["--build-arg", f"{key}={value}"]
                labels = {
                    PLUGIN_BUILD_LABEL: signature,
                    "io.github.ovlab.plugin-registry.sha256": registry_sha,
                    "io.github.ovlab.role": role,
                    "io.github.ovlab.source-manifest.sha256": manifest["source_content_sha256"],
                }
                if bundle is not None:
                    labels["io.github.ovlab.config-bundle.sha256"] = bundle.sha256
                for key, value in labels.items():
                    command += ["--label", f"{key}={value}"]
                command.append(str(workspace))
                result = runner(command, cwd=workspace, check=False)
                if result.returncode:
                    raise RuntimeError(f"Docker build failed for {role} ({result.returncode})")
                after = _registered_rows(inspect_sources(workspace, sources, registry=source_registry, staged=True), registry)
                if after != rows or registry_identity(root, roles=selected) != registry_sha:
                    raise ConfigCompatibilityError("plugin sources or registry changed during build; image must be rebuilt")
                plan.update(command=command, source_sha256=manifest["source_content_sha256"],
                            registry_sha256=registry_sha, plugin_build_sha256=signature,
                            isolated_sources=isolated_sources)
                if bundle is not None:
                    plan["config_bundle_sha256"] = bundle.sha256
        return plans
