"""Read-only inventory of portable checkpoint registry entries."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .checkpointing import default_global_cache, local_checkpoint_override


def _remote_revision(repo_id, revision, environment):
    headers = {"User-Agent": "OpenVLABenchmark/0.1 checkpoint-catalog"}
    token = environment.get("HF_TOKEN") or environment.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://huggingface.co/api/models/{quote(repo_id, safe='/')}/revision/{quote(revision, safe='')}"
    try:
        with urlopen(Request(url, headers=headers), timeout=10) as response:
            sha = json.load(response).get("sha")
        if not isinstance(sha, str) or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            return "unknown", None
        return "available", sha
    except HTTPError as exc:
        return ("not_found_or_inaccessible" if exc.code in (401, 403, 404) else "unknown"), None
    except (URLError, OSError, ValueError, TypeError, AttributeError):
        return "unknown", None


def registry_checkpoints(registry, *, environment, model_data_root, online=False, profile=None):
    global_root = Path(environment.get("OVLAB_GLOBAL_HF_CACHE", default_global_cache(environment))).expanduser().resolve()
    hub = global_root / "hub" if (global_root / "hub").is_dir() else global_root
    runs = Path(environment.get("OVLAB_RUNS_ROOT", model_data_root / "runs")).expanduser()
    managed = Path(environment.get("OVLAB_MANAGED_CHECKPOINTS_ROOT", runs.parent / "checkpoints/huggingface")).expanduser().resolve()
    rows = []
    for resource_id, entry in registry["checkpoints"].items():
        repo = entry["repo_id"]
        selector = entry.get("revision", "local/main")
        repo_dir = hub / ("models--" + repo.replace("/", "--"))
        revision = selector
        if selector == "local/main":
            try:
                revision = (repo_dir / "refs/main").read_text(encoding="utf-8").strip()
            except OSError:
                revision = None
            if revision is not None and (len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)):
                revision = None
        elif selector == "remote/main":
            revision = None
        remote_status, remote_sha = ("not_checked", None)
        if online and (revision or selector == "remote/main"):
            remote_status, remote_sha = _remote_revision(repo, revision or "main", environment)
            if revision and remote_sha and revision != remote_sha:
                remote_status = "unknown"
            if selector == "remote/main":
                revision = remote_sha
        files = entry.get("files") or entry.get("artifact", {}).get("files", {})
        candidates = []
        if revision:
            candidates.append(("global-huggingface-cache", repo_dir / "snapshots" / revision))
        override = local_checkpoint_override(profile, resource_id)
        if override is not None and (selector not in ("local/main", "remote/main") or (revision and override.name == revision)):
            candidates.append(("local-profile", override))
        if revision:
            candidates.append(("ovlab-managed-cache", managed / resource_id / revision))
        local_status, local_source, local_path = "not_found", None, None
        if revision is None:
            local_status = "revision_unknown"
        for source, path in candidates:
            if not path.is_dir():
                continue
            try:
                complete = all((path / name).is_file() and (path / name).stat().st_size == info["size"] for name, info in files.items())
                if not files:
                    complete = any(item.is_file() for item in path.rglob("*"))
            except OSError:
                complete = False
            # Match resolver precedence: an incomplete higher-priority snapshot blocks later ones.
            local_status = "present_unverified" if complete else "incomplete"
            local_source, local_path = source, str(path)
            break
        rows.append({"checkpoint_id": resource_id, "repo_id": repo, "revision": selector,
                     "resolved_revision": revision, "local_status": local_status,
                     "local_source": local_source, "local_path": local_path,
                     "remote_status": remote_status})
    return {"schema_version": "ovlab.registry-checkpoint-list/v1", "source": "registry",
            "network": "online" if online else "offline", "checkpoints": rows}
