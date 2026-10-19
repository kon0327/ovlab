"""Host-created evidence must be writable by the isolated worker's group."""
import os
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ovlab_benchctl.training_deployment import TrainingDeployment
from ovlab_benchctl.training_runs import TrainingRunStore
from ovlab_benchctl.training_identity import identity


def test_training_image_accepts_custom_role_with_training_contract(tmp_path):
    from ovlab_benchctl.image_metadata import IMAGE_CONTRACT_LABEL
    from ovlab_benchctl.training_errors import TrainingRuntimeError

    def inspect(command, **kwargs):
        del command, kwargs
        labels = {IMAGE_CONTRACT_LABEL: "offline-dataset-ro-base-ro-training-run-rw-v1",
                  "io.github.ovlab.role": "my-vla-trainer"}
        return SimpleNamespace(returncode=0, stdout=json.dumps(labels) + "|sha256:custom")

    deployment = TrainingDeployment(tmp_path, environment={"OVLAB_TRAINING_IMAGE": "custom:local"}, runner=inspect)
    assert deployment._docker_image("OVLAB_TRAINING_IMAGE", "unused", "offline-dataset-ro-base-ro-training-run-rw-v1") == (
        "custom:local", "sha256:custom"
    )
    with pytest.raises(TrainingRuntimeError, match="deployment contract"):
        deployment._docker_image("OVLAB_TRAINING_IMAGE", "unused", "incompatible-v2")


@pytest.mark.parametrize("mask", [0o022, 0o077])
def test_training_run_keeps_worker_group_access_despite_host_umask(tmp_path, mask):
    store = TrainingRunStore(tmp_path)
    previous = os.umask(mask)
    try:
        context = store.create({"id": "permissions"}, {"id": "permissions"},
            {"scientific_training_id": "training-plan-permissions", "execution_plan_id": "execution-permissions"}, {})
    finally:
        os.umask(previous)
    for path in (context.root, context.root / "logs", context.root / "staging-checkpoints"):
        assert path.stat().st_mode & 0o7777 == 0o2770
        assert path.stat().st_gid == context.root.stat().st_gid
    for name in ("events.jsonl", "metrics.jsonl", "original-profile.yaml", "resolved-profile.yaml"):
        assert (context.root / name).stat().st_mode & 0o777 == 0o660
    assert (context.root / "training-plan.json").is_file()


def test_worker_receives_actual_mount_groups_in_addition_to_primary_group(monkeypatch):
    monkeypatch.setattr(os, "getgid", lambda: 1000)
    monkeypatch.setattr(Path, "stat", lambda path: SimpleNamespace(st_gid={"run": 4321, "checkpoints": 1000}[path.name]))
    assert TrainingDeployment._container_groups(Path("run"), Path("checkpoints"), Path("run")) == [
        "--group-add", "1000", "--group-add", "4321",
    ]


def test_training_materializes_hf_blob_links_without_changing_scientific_identity(tmp_path, monkeypatch):
    from ovlab_benchctl import checkpointing

    payload = b'{"model_type":"openvla"}'
    digest = hashlib.sha256(payload).hexdigest()
    spec = checkpointing.CheckpointSpec(
        resource_id="test-model", repo_id="owner/model", revision="a" * 40,
        expected_sha256=hashlib.sha256(f"config.json {len(payload)} {digest}\n".encode()).hexdigest(),
        files=(("config.json", len(payload), digest),),
    )
    snapshot = tmp_path / "global/hub/models--owner--model/snapshots" / spec.revision
    snapshot.mkdir(parents=True)
    blob = snapshot.parent.parent / "blobs" / digest
    blob.parent.mkdir()
    blob.write_bytes(payload)
    (snapshot / "config.json").symlink_to(f"../../blobs/{digest}")
    monkeypatch.setattr(checkpointing, "checkpoint_spec_by_id", lambda *args: spec)
    deployment = TrainingDeployment(tmp_path, environment={
        "OVLAB_MODEL_DATA_ROOT": str(tmp_path / "managed"),
        "OVLAB_GLOBAL_HF_CACHE": str(tmp_path / "global"),
    })
    plan = {
        "scientific_training_id": "same-science",
        "scientific": {"base_checkpoint": {"resource_id": spec.resource_id}},
        "execution": {"model_host_path": str(snapshot), "model_source_kind": "global-huggingface-cache"},
    }
    prepared = deployment._prepare_base_checkpoint(plan)
    selected = Path(prepared["execution"]["model_host_path"]) / "config.json"
    assert selected.read_bytes() == payload
    assert not selected.is_symlink()
    assert selected.stat().st_ino == blob.stat().st_ino
    assert prepared["scientific"] == plan["scientific"]
    assert prepared["scientific_training_id"] == plan["scientific_training_id"]
    assert prepared["execution_plan_id"] == identity("training-execution", prepared["execution"], 32)
    assert plan["execution"]["model_host_path"] == str(snapshot)


@pytest.mark.parametrize("trainer_status", [0, 1])
@pytest.mark.parametrize("interactive_progress", [False, True])
def test_isolated_lifecycle_preserves_run_id_and_records_early_failure(tmp_path, monkeypatch, trainer_status, interactive_progress):
    from ovlab_benchctl import training_deployment
    from ovlab_benchctl.training_errors import TrainingRuntimeError

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("id: fixture\n")
    profile = SimpleNamespace(document={"id": "fixture", "training": {}, "dataset": {"ref": "dataset-fixture"}})
    plan = {"scientific_training_id": "training-plan-fixture", "execution_plan_id": "execution-fixture",
            "scientific": {"dataset": {"dataset_id": "dataset-fixture"}},
            "execution": {"model_host_path": str(tmp_path / "base")}}
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        if "train" in command and "--run-root" in command:
            return SimpleNamespace(returncode=trainer_status)
        return SimpleNamespace(returncode=0, stdout='{"result": {}}')

    deployment = TrainingDeployment(tmp_path, runner=runner, environment={"OVLAB_MODEL_DATA_ROOT": str(tmp_path / "data")})
    monkeypatch.setattr(deployment, "_profile", lambda _: (profile_path, profile))
    monkeypatch.setattr(deployment, "_gpu", lambda: (1, 24))
    monkeypatch.setattr(deployment, "_docker_image", lambda variable, default, role: (default, "sha256:fixture"))
    monkeypatch.setattr(deployment, "_prepare_base_checkpoint", lambda value: value)
    monkeypatch.setattr(training_deployment.TrainingPlanner, "plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(training_deployment.DatasetStore, "inspect", lambda *args: {
        "host_path": str(tmp_path / "dataset"), "source_metadata": {"tfds_name": "fixture"},
    })
    if trainer_status:
        with pytest.raises(TrainingRuntimeError, match="trainer failed"):
            deployment.run(profile_path, interactive_progress=interactive_progress)
    else:
        deployment.run(profile_path, interactive_progress=interactive_progress)
    run_root, = (tmp_path / "data/runs/train").iterdir()
    trainer_command = next(command for command in commands if "train" in command and "--run-root" in command)
    assert ("--tty" in trainer_command) is interactive_progress
    for command in commands:
        if "--run-root" in command:
            mount_root = Path(command[command.index("--run-root") + 1])
            assert mount_root.name == run_root.name
            assert f"type=bind,source={run_root},target={mount_root}" in command
    if trainer_status:
        result = json.loads((run_root / "result.json").read_text())
        assert result["status"] == "failed"
        assert result["run_id"] == run_root.name
