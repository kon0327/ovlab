"""CPU-only deployment checkpoint selection scenarios."""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import ovlab_benchctl.checkpointing as checkpointing
from ovlab_benchctl.deployment import ComposeDeployment, ComposeDeploymentError


ROOT = Path(__file__).resolve().parents[4]


def _experiment(tmp_path: Path, selector: str) -> Path:
    root = tmp_path / "repository"
    shutil.copytree(ROOT / "configs", root / "configs")
    registry = root / "configs/resources/registry.yaml"
    content = registry.read_text(encoding="utf-8")
    start = content.index("  openvla-7b:\n")
    end = content.index("  openvla-7b-finetuned-libero-10:\n")
    entry = f"  openvla-7b:\n    repo_id: openvla/openvla-7b\n    revision: {selector}\n"
    registry.write_text(content[:start] + entry + content[end:], encoding="utf-8")
    return root


def _plan(root: Path, tmp_path: Path, *, offline: bool):
    env_file = tmp_path / "empty.env"
    env_file.write_text("", encoding="utf-8")
    return SimpleNamespace(
        experiment="configs/experiments/libero-vanilla-smoke.yaml",
        local_profile=None,
        offline=offline,
        env_file=str(env_file),
        compose_files=(str(root / "deploy/compose/compose.yaml"),),
    )


def test_deploy_resolves_cached_local_main_while_offline(tmp_path):
    root = _experiment(tmp_path, "local/main")
    revision = "a" * 40
    repo = tmp_path / "cache/hub/models--openvla--openvla-7b"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs/main").write_text(revision, encoding="utf-8")
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"eval fixture")
    deployment = ComposeDeployment(root, environment={
        "OVLAB_GLOBAL_HF_CACHE": str(tmp_path / "cache"),
        "OVLAB_MANAGED_CHECKPOINTS_ROOT": str(tmp_path / "managed"),
    })

    with patch.object(checkpointing, "urlopen", side_effect=AssertionError("local/main must stay offline")):
        result = deployment._resolve_checkpoint(_plan(root, tmp_path, offline=True))

    assert result.spec.revision == revision
    assert result.spec.expected_sha256


def test_deploy_resolves_remote_main_and_downloads_selected_commit(tmp_path, monkeypatch):
    root = _experiment(tmp_path, "remote/main")
    revision = "b" * 40
    monkeypatch.setattr(
        checkpointing, "_download_snapshot",
        lambda _repo, _revision, directory, progress=None: (directory / "model.bin").write_bytes(b"eval fixture"),
    )
    deployment = ComposeDeployment(root, environment={
        "OVLAB_GLOBAL_HF_CACHE": str(tmp_path / "cache"),
        "OVLAB_MANAGED_CHECKPOINTS_ROOT": str(tmp_path / "managed"),
    })
    response = io.BytesIO(json.dumps({"sha": revision}).encode())

    with patch.object(checkpointing, "urlopen", return_value=response) as remote:
        result = deployment._resolve_checkpoint(_plan(root, tmp_path, offline=False))

    remote.assert_called_once()
    assert result.spec.revision == revision
    assert result.source_kind == "ovlab-managed-cache"


def test_deploy_rejects_remote_main_when_offline(tmp_path):
    root = _experiment(tmp_path, "remote/main")
    deployment = ComposeDeployment(root, environment={
        "OVLAB_MANAGED_CHECKPOINTS_ROOT": str(tmp_path / "managed"),
    })

    with pytest.raises(ComposeDeploymentError, match="remote/main requires online deployment"):
        deployment._resolve_checkpoint(_plan(root, tmp_path, offline=True))
