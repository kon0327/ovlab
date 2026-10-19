"""CPU-only training-plan scenarios for moving and pinned model revisions."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import ovlab_benchctl.checkpointing as checkpointing
from ovlab_benchctl.checkpointing import CheckpointResolutionError
from ovlab_benchctl.strict_yaml import load
from ovlab_benchctl.training_profiles import TrainingPlanner, TrainingProfile


ROOT = Path(__file__).resolve().parents[4]
PROFILE_PATH = ROOT / "configs/training/openvla-libero10-quic-smoke.yaml"


def _write_registry(repository: Path, selector: str | None, *, pinned: tuple[str, str] | None = None) -> None:
    registry = repository / "configs/resources/registry.yaml"
    registry.parent.mkdir(parents=True)
    revision = f"    revision: {selector}\n" if selector is not None else ""
    identity = ""
    if pinned is not None:
        pinned_revision, aggregate = pinned
        revision = f'    revision: "{pinned_revision}"\n'
        payload_sha = hashlib.sha256(b"tiny-model-bytes").hexdigest()
        identity = (
            f'    expected_sha256: "{aggregate}"\n    files:\n      model.bin:\n'
            f'        size: 16\n        sha256: "{payload_sha}"\n'
        )
    registry.write_text(
        'schema_version: "0.1.0"\nkind: resource_registry\ncheckpoints:\n'
        '  scenario-model:\n    repo_id: owner/scenario-model\n'
        f"{revision}{identity}"
        "repositories:\n  source:\n    path: external/source\n",
        encoding="utf-8",
    )


def _profile(selector: str | None, pinned_revision: str | None = None) -> TrainingProfile:
    document = load(PROFILE_PATH)
    document["model"]["base_checkpoint"] = "scenario-model"
    document["model"].pop("revision", None)
    if selector is not None:
        document["model"]["revision"] = selector
    if pinned_revision is not None:
        document["model"]["revision"] = pinned_revision
    return TrainingProfile.from_document(document)


def _snapshot(cache: Path, revision: str, payload: bytes = b"tiny-model-bytes") -> Path:
    model = cache / "hub/models--owner--scenario-model"
    refs = model / "refs/main"
    refs.parent.mkdir(parents=True, exist_ok=True)
    refs.write_text(revision, encoding="utf-8")
    snapshot = model / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(payload)
    return snapshot


def _plan(planner: TrainingPlanner, profile: TrainingProfile, *, allow_download: bool) -> dict[str, object]:
    planner._dataset = lambda _profile, allow_dataset_download: {
        "dataset_id": "dataset-test-fixture",
        "raw_content_digest": "a" * 64,
        "prepared_content_digest": "b" * 64,
        "preparation": {"format": "openvla-rlds"},
        "host_path": "/fixture/dataset",
    }
    return planner.plan(
        profile,
        available_gpu_count=1,
        available_vram_gib=32.0,
        allow_checkpoint_download=allow_download,
    )


@pytest.mark.parametrize("selector", [None, "local/main"])
def test_train_plan_uses_cached_main_when_revision_is_omitted_or_local(selector, tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    _write_registry(repository, selector)
    revision = "a" * 40
    _snapshot(tmp_path / "cache", revision)
    monkeypatch.setenv("OVLAB_GLOBAL_HF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(
        checkpointing, "urlopen", lambda *_args, **_kwargs: pytest.fail("local/main must stay offline"),
    )

    plan = _plan(
        TrainingPlanner(repository, tmp_path / "model-data"), _profile(selector), allow_download=True,
    )

    base = plan["scientific"]["base_checkpoint"]
    assert base["revision"] == revision
    assert base["revision_selection"] == "local/main"
    assert len(base["aggregate_sha256"]) == 64


def test_train_run_remote_main_resolves_downloaded_commit_and_records_hashes(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    _write_registry(repository, "remote/main")
    revision = "c" * 40
    cache = tmp_path / "cache"
    monkeypatch.setenv("OVLAB_GLOBAL_HF_CACHE", str(cache))
    response = io.BytesIO(json.dumps({"sha": revision}).encode())
    with patch.object(checkpointing, "urlopen", return_value=response) as remote:
        monkeypatch.setattr(
            checkpointing, "_download_snapshot",
            lambda _repo, _revision, directory, progress=None: (directory / "model.bin").write_bytes(b"tiny-model-bytes"),
        )
        plan = _plan(
            TrainingPlanner(repository, tmp_path / "model-data"), _profile("remote/main"), allow_download=True,
        )

    remote.assert_called_once()
    base = plan["scientific"]["base_checkpoint"]
    assert base["revision"] == revision
    assert base["revision_selection"] == "remote/main"
    assert base["aggregate_sha256"] == hashlib.sha256(
        f"model.bin 16 {hashlib.sha256(b'tiny-model-bytes').hexdigest()}\n".encode()
    ).hexdigest()


def test_train_plan_remote_main_queries_but_does_not_download(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    _write_registry(repository, "remote/main")
    revision = "e" * 40
    monkeypatch.setenv("OVLAB_GLOBAL_HF_CACHE", str(tmp_path / "empty-cache"))
    response = io.BytesIO(json.dumps({"sha": revision}).encode())
    with patch.object(checkpointing, "urlopen", return_value=response):
        with pytest.raises(CheckpointResolutionError, match="unavailable locally"):
            _plan(
                TrainingPlanner(repository, tmp_path / "model-data"), _profile("remote/main"),
                allow_download=False,
            )


def test_train_plan_explicit_pinned_revision_uses_registered_file_manifest(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    revision = "f" * 40
    payload = b"tiny-model-bytes"
    file_sha = hashlib.sha256(payload).hexdigest()
    manifest_line = f"model.bin {len(payload)} {file_sha}\n"
    aggregate = hashlib.sha256(manifest_line.encode()).hexdigest()
    _write_registry(repository, None, pinned=(revision, aggregate))
    snapshot = _snapshot(tmp_path / "cache", revision, payload)
    monkeypatch.setenv("OVLAB_GLOBAL_HF_CACHE", str(tmp_path / "cache"))

    plan = _plan(
        TrainingPlanner(repository, tmp_path / "model-data"), _profile(None, pinned_revision=revision),
        allow_download=False,
    )

    base = plan["scientific"]["base_checkpoint"]
    assert base["revision"] == revision
    assert "revision_selection" not in base
    assert base["aggregate_sha256"] == aggregate
    assert snapshot.is_dir()
