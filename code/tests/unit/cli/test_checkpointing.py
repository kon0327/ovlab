"""Dependency-light immutable checkpoint resolution regressions."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import pytest

import ovlab_benchctl.checkpointing as checkpointing
from ovlab_benchctl.checkpointing import (
    CheckpointResolutionError,
    CheckpointResolver,
    CheckpointSpec,
    checkpoint_spec,
    local_checkpoint_override,
    verify_checkpoint,
)


REPOSITORY = Path(__file__).resolve().parents[4]
EXPERIMENT = "configs/experiments/libero10-openvla-oft-rpc-smoke.yaml"


def _spec(payload: bytes = b"immutable checkpoint") -> CheckpointSpec:
    digest = hashlib.sha256(payload).hexdigest()
    line = f"model.bin {len(payload)} {digest}\n"
    return CheckpointSpec(
        resource_id="test-model",
        repo_id="owner/test-model",
        revision="a" * 40,
        expected_sha256=hashlib.sha256(line.encode()).hexdigest(),
        files=(("model.bin", len(payload), digest),),
    )


def _snapshot(path: Path, payload: bytes = b"immutable checkpoint") -> Path:
    path.mkdir(parents=True)
    (path / "model.bin").write_bytes(payload)
    return path


def test_portable_experiment_resolves_pinned_registry_identity():
    spec = checkpoint_spec(REPOSITORY, EXPERIMENT)

    assert spec.resource_id == "openvla-oft-7b-finetuned-libero-10"
    assert spec.repo_id == "moojink/openvla-7b-oft-finetuned-libero-10"
    assert spec.revision == "95220f9a3421a7ff12d4218e73d09ade830fa9a3"
    assert spec.expected_sha256 == "c08526071fc0303069532386493ed95483c466f4cdd07848482e2758e7f33c61"
    assert len(spec.files) > 10


def test_vanilla_base_checkpoint_is_also_pinned_and_verifiable():
    spec = checkpoint_spec(
        REPOSITORY, "configs/experiments/libero-vanilla-smoke.yaml"
    )

    assert spec.resource_id == "openvla-7b"
    assert spec.revision == "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
    assert spec.expected_sha256 == "f00cf60094c5df1b7656bec4fe0060830a8fe168d5fabd6626e3964842481f86"
    assert len(spec.files) == 16


def test_checkpoint_identity_resolution_needs_only_system_python():
    source = REPOSITORY / "code/apps/benchctl/src"
    completed = subprocess.run(
        [
            "/usr/bin/env", "-i", "PATH=/usr/bin:/bin", "HOME=/tmp",
            f"PYTHONPATH={source}", "python3", "-c",
            "from pathlib import Path; from ovlab_benchctl.checkpointing import checkpoint_spec; "
            f"s=checkpoint_spec(Path({str(REPOSITORY)!r}), {EXPERIMENT!r}); print(s.resource_id)",
        ],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "openvla-oft-7b-finetuned-libero-10"


def test_global_hf_snapshot_has_first_precedence_and_is_hardlinked_without_copy(tmp_path):
    spec = _spec()
    global_snapshot = _snapshot(
        tmp_path / "global/hub/models--owner--test-model/snapshots" / spec.revision
    )
    local = _snapshot(tmp_path / "local")
    resolver = CheckpointResolver(
        global_cache=tmp_path / "global", managed_cache=tmp_path / "managed",
        progress=(progress := []).append,
    )

    resolved = resolver.resolve(spec, local_path=local, offline=True)
    resolved_again = resolver.resolve(spec, local_path=local, offline=True)

    assert resolved.source_kind == "global-huggingface-cache"
    assert resolved.container_path == "/checkpoints/resolved/test-model"
    assert resolved.materialized_without_copy is True
    assert resolved.host_path.is_relative_to(tmp_path / "managed")
    assert resolved.host_path.stat().st_mode & 0o777 == 0o755
    assert os.stat(global_snapshot / "model.bin").st_ino == os.stat(
        resolved.host_path / "model.bin"
    ).st_ino
    assert resolved_again.host_path == resolved.host_path
    assert sum(message.startswith("Verified [") for message in progress) == 2


def test_local_profile_override_precedes_managed_cache(tmp_path):
    spec = _spec()
    local = _snapshot(tmp_path / "local")
    local.chmod(0o700)
    _snapshot(tmp_path / "managed" / spec.resource_id / spec.revision)
    resolved = CheckpointResolver(
        global_cache=tmp_path / "absent-global", managed_cache=tmp_path / "managed"
    ).resolve(spec, local_path=local, offline=True)

    assert resolved.source_kind == "local-profile"
    assert resolved.host_path == (
        tmp_path / "managed" / spec.resource_id / spec.revision
    ).resolve()
    assert resolved.materialized_without_copy is True
    assert resolved.host_path.stat().st_mode & 0o777 == 0o755
    assert local.stat().st_mode & 0o777 == 0o700


def test_external_local_snapshot_symlinks_are_materialized_with_hardlinks(tmp_path):
    payload = b"immutable checkpoint"
    spec = _spec(payload)
    blob = tmp_path / "blobs/model.bin"
    blob.parent.mkdir()
    blob.write_bytes(payload)
    local = tmp_path / "local"
    local.mkdir()
    (local / "model.bin").symlink_to(blob)

    resolved = CheckpointResolver(
        global_cache=tmp_path / "global", managed_cache=tmp_path / "managed"
    ).resolve(spec, local_path=local, offline=True)

    assert resolved.source_kind == "local-profile"
    assert resolved.materialized_without_copy is True
    assert not (resolved.host_path / "model.bin").is_symlink()
    assert os.stat(blob).st_ino == os.stat(resolved.host_path / "model.bin").st_ino


def test_managed_cache_is_used_before_download(tmp_path):
    spec = _spec()
    managed = _snapshot(tmp_path / "managed" / spec.resource_id / spec.revision)
    downloads = []
    resolver = CheckpointResolver(
        global_cache=tmp_path / "global",
        managed_cache=tmp_path / "managed",
        downloader=lambda *args: downloads.append(args),
    )

    resolved = resolver.resolve(spec)

    assert resolved.source_kind == "ovlab-managed-cache"
    assert resolved.host_path == managed.resolve()
    assert downloads == []


def test_online_missing_checkpoint_downloads_pinned_revision_atomically(tmp_path):
    spec = _spec()
    calls = []
    progress = []

    def download(repo_id, revision, destination):
        calls.append((repo_id, revision))
        (destination / "model.bin").write_bytes(b"immutable checkpoint")

    resolved = CheckpointResolver(
        global_cache=tmp_path / "global",
        managed_cache=tmp_path / "managed",
        downloader=download,
        progress=progress.append,
    ).resolve(spec)

    assert calls == [(spec.repo_id, spec.revision)]
    assert resolved.source_kind == "ovlab-managed-download"
    assert resolved.host_path == (tmp_path / "managed" / spec.resource_id / spec.revision).resolve()
    assert not tuple(resolved.host_path.parent.glob(f".{spec.revision}.*"))
    assert any("not found locally; starting download" in message for message in progress)
    assert any("verifying 1 files and SHA-256 hashes" in message for message in progress)
    assert progress[-1] == "Checkpoint 'test-model' is verified and ready."


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, *, content_length: int | None = None):
        super().__init__(payload)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


@pytest.mark.parametrize("size_metadata", [True, False])
def test_standard_library_download_reports_file_progress_with_or_without_known_size(
    tmp_path, monkeypatch, size_metadata
):
    payload = b"checkpoint bytes"
    revision = "a" * 40
    sibling = {"rfilename": "model.bin"}
    if size_metadata:
        sibling["size"] = len(payload)
    metadata = json.dumps({"sha": revision, "siblings": [sibling]}).encode()
    responses = [
        _Response(metadata),
        _Response(payload, content_length=len(payload) if size_metadata else None),
    ]
    monkeypatch.setattr(checkpointing, "urlopen", lambda *_args, **_kwargs: responses.pop(0))
    progress = []
    destination = tmp_path / "download"
    destination.mkdir()

    checkpointing._download_snapshot(
        "owner/model", revision, destination, progress=progress.append
    )

    assert (destination / "model.bin").read_bytes() == payload
    assert any("Downloading 1 checkpoint files" in message for message in progress)
    expected_size = "16 B" if size_metadata else "unknown size"
    assert any(f"[1/1] model.bin: starting ({expected_size})" in message for message in progress)
    assert progress[-1] == "[1/1] model.bin: 16 B complete."


def test_offline_missing_checkpoint_fails_without_downloader(tmp_path):
    spec = _spec()
    calls = []
    resolver = CheckpointResolver(
        global_cache=tmp_path / "global",
        managed_cache=tmp_path / "managed",
        downloader=lambda *args: calls.append(args),
    )

    with pytest.raises(CheckpointResolutionError, match="--offline"):
        resolver.resolve(spec, offline=True)

    assert calls == []


def test_corrupt_higher_precedence_snapshot_is_rejected_not_silently_replaced(tmp_path):
    spec = _spec()
    global_snapshot = _snapshot(
        tmp_path / "global/hub/models--owner--test-model/snapshots" / spec.revision,
        b"corrupt",
    )
    assert global_snapshot.is_dir()
    local = _snapshot(tmp_path / "local")

    with pytest.raises(CheckpointResolutionError, match="size mismatch"):
        CheckpointResolver(
            global_cache=tmp_path / "global", managed_cache=tmp_path / "managed"
        ).resolve(spec, local_path=local, offline=True)


def test_local_profile_checkpoint_path_is_absolute_and_host_only(tmp_path):
    snapshot = tmp_path / "quic"
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "schema_version: \"0.1.0\"\nkind: local_profile\nid: test\n"
        "paths:\n  checkpoint_root: /unused\n  dataset_root: /unused\n  runs_root: /unused\n"
        "devices:\n  primary_gpu: cuda:0\n"
        "resources:\n  checkpoints:\n    quic-libero10-v1:\n"
        f"      local_path: {snapshot}\n",
        encoding="utf-8",
    )

    assert local_checkpoint_override(profile, "quic-libero10-v1") == snapshot.resolve()
    assert local_checkpoint_override(profile, "another-model") is None


def test_file_and_aggregate_hashes_are_both_enforced(tmp_path):
    spec = _spec()
    snapshot = _snapshot(tmp_path / "snapshot")
    assert verify_checkpoint(spec, snapshot) == 1
    with pytest.raises(CheckpointResolutionError, match="registry aggregate SHA-256 is inconsistent"):
        CheckpointSpec(
            resource_id=spec.resource_id,
            repo_id=spec.repo_id,
            revision=spec.revision,
            expected_sha256="0" * 64,
            files=spec.files,
        )
