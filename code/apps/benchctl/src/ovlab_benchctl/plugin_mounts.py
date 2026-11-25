"""Validate the shared contract for plugin-provided read-only directories."""
import csv
import io
from pathlib import Path, PurePosixPath
import re

from .errors import ConfigCompatibilityError

_RESERVED = tuple(map(PurePosixPath, ("/opt/ovlab/configs", "/checkpoints/resolved", "/checkpoints/base", "/var/lib/ovlab/training-run", "/run/ovlab", "/tmp")))
_ENV_RESERVED = {"OVLAB_PLUGIN_REGISTRY", "OVLAB_CONFIG_BUNDLE_SHA256", "OVLAB_RESOLVED_CHECKPOINT_ID",
                 "OVLAB_RESOLVED_CHECKPOINT_CONTAINER_PATH", "OVLAB_CHECKPOINT_REVISION", "OVLAB_CHECKPOINT_SHA256",
                 "OVLAB_LOCAL_PROFILE", "OVLAB_EXECUTION_PROFILE", "OVLAB_DEPLOYMENT_PROFILE"}


def validate_runtime_mounts(mounts, *, check_files=False, protect_tmp=True):
    if not isinstance(mounts, (list, tuple)):
        raise ConfigCompatibilityError("plugin runtime_mounts must return a list of directory mappings")
    result, targets, environment = [], [], set()
    for mount in mounts:
        if not isinstance(mount, dict) or set(mount) - {"source", "target", "environment", "required_files"}:
            raise ConfigCompatibilityError("invalid plugin runtime mount fields")
        source, target = mount.get("source"), mount.get("target")
        if not isinstance(source, str) or not Path(source).is_absolute() or "\0" in source:
            raise ConfigCompatibilityError("plugin mount source must be an absolute host directory")
        if not isinstance(target, str) or not target.startswith("/") or "\0" in target or "$" in target:
            raise ConfigCompatibilityError("plugin mount target must be an absolute literal container path")
        container = PurePosixPath(target)
        if ".." in container.parts or container.as_posix() != target:
            raise ConfigCompatibilityError("plugin mount target must be normalized")
        reserved = _RESERVED if protect_tmp else tuple(path for path in _RESERVED if path != PurePosixPath("/tmp"))
        if any(container.is_relative_to(path) or path.is_relative_to(container) for path in (*reserved, *targets)):
            raise ConfigCompatibilityError(f"plugin mount target overlaps a reserved or another mount: {target}")
        targets.append(container)
        key = mount.get("environment")
        if key is not None:
            if not isinstance(key, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) or key in _ENV_RESERVED or key.startswith(("OVLAB_CHECKPOINT_", "OVLAB_RESOLVED_")) or key in environment:
                raise ConfigCompatibilityError("plugin mount environment variable is invalid, reserved or duplicated")
            environment.add(key)
        required = mount.get("required_files", [])
        if not isinstance(required, list) or any(not isinstance(name, str) or not name or Path(name).is_absolute()
                                                 or ".." in Path(name).parts for name in required):
            raise ConfigCompatibilityError("plugin required_files must contain relative file paths")
        host = Path(source).resolve()
        if check_files:
            missing = [name for name in required if not (host / name).is_file() or not (host / name).resolve().is_relative_to(host)]
            if not host.is_dir() or missing:
                raise ConfigCompatibilityError(f"plugin runtime directory {host} is missing {', '.join(missing) or 'directory'}")
        result.append({**mount, "source": str(host)})
    return result


def docker_bind(source, target):
    stream = io.StringIO()
    csv.writer(stream, lineterminator="").writerow(["type=bind", f"source={source}", f"target={target}", "readonly"])
    return stream.getvalue()
