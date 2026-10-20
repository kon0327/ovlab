import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ovlab_benchctl import ConfigResolver
from ovlab_benchctl.schema import validate
from ovlab_benchctl.strict_yaml import load
from ovlab_openvla_common import OpenVlaModelSource, libero_target_action_spec
from ovlab_openvla_quic import (
    OpenVlaQuICSettings, QuICDescriptorError, descriptor_from_settings,
)


ROOT = Path(__file__).resolve().parents[5]


def _settings(tmp_path):
    directory = tmp_path / "adapter"
    directory.mkdir()
    tensors = b"unit-test-manifest-only-fixture-no-model"
    (directory / "adapter.safetensors").write_bytes(tensors)
    payload = {
        "format_version": 1,
        "host_identity": {"family": "openvla_token", "revision": "a" * 40,
                          "checkpoint_id": "test-host", "checkpoint_sha256": "b" * 64,
                          "quic_method_sha256": "c" * 64},
        "tensor_sha256": hashlib.sha256(tensors).hexdigest(),
        "manifest": [{"name": "projector.fc1", "in_features": 4, "out_features": 8,
                      "config": {"orders": [1, 2], "block_count": 1, "base_size": 3}}],
        "metadata": {"normalization_contract": {"unnorm_key": "test_norm", "action_statistics": {}},
                     "preprocessing": {"prompt_style": "modern", "center_crop": False}},
    }
    manifest = json.dumps(payload).encode()
    (directory / "adapter.json").write_bytes(manifest)
    return OpenVlaQuICSettings(
        model=OpenVlaModelSource("test/host", "a" * 40, local_path=str(tmp_path)),
        unnorm_key="test_norm", adapter_path=str(directory),
        target_action_spec=libero_target_action_spec(),
        adapter_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        adapter_tensor_sha256=hashlib.sha256(tensors).hexdigest(),
    )


def test_runtime_descriptor_binds_method_artifact_normalization_and_interfaces_without_torch(tmp_path):
    settings = _settings(tmp_path)
    descriptor = descriptor_from_settings(settings)
    descriptor.require_runtime_ready()
    assert descriptor.generic_compound_backend_status == "portable_determinant_backend"
    assert descriptor.runtime_validated is False
    assert descriptor.training_validated is False
    assert descriptor.libero_validated is False
    assert descriptor.compression_verified is False
    assert descriptor.capability_identity["action_transform"]["kind"] == "identity"
    pose = descriptor_from_settings(replace(settings, action_codec="openvla-to-libero-v1"))
    assert pose.capability_identity["action_transform"]["above"] == -1.0
    assert descriptor.scientific_hash != pose.scientific_hash


def test_runtime_rejects_wrong_normalization_preprocessing_or_artifact_bytes(tmp_path):
    settings = _settings(tmp_path)
    for mutation, error in (({"unnorm_key": "other"}, "normalization"),
                            ({"center_crop": True}, "preprocessing"),
                            ({"adapter_manifest_sha256": "0" * 64}, "manifest"),
                            ({"adapter_tensor_sha256": "0" * 64}, "tensor")):
        with pytest.raises(QuICDescriptorError, match=error):
            descriptor_from_settings(replace(settings, **mutation))
    (Path(settings.adapter_path) / "adapter.safetensors").write_bytes(b"changed")
    with pytest.raises(QuICDescriptorError, match="tensor"):
        descriptor_from_settings(settings)


def test_fitted_normalization_is_bound_to_host_identity_and_rejects_tampering(tmp_path):
    settings = _settings(tmp_path)
    path = Path(settings.adapter_path) / "adapter.json"
    payload = json.loads(path.read_text())
    contract = {"unnorm_key": "test_norm", "action_statistics": {"q01": [-1.] * 7, "q99": [1.] * 7},
                "mode": "train-q99"}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
    payload["host_identity"]["normalization_contract_sha256"] = digest
    payload["metadata"]["normalization_contract"] = {**contract, "sha256": digest}

    def write_settings():
        path.write_text(json.dumps(payload))
        return replace(settings, adapter_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    descriptor = descriptor_from_settings(write_settings())
    host = descriptor.base_model_identity["host"]
    assert host["normalization_contract_sha256"] == digest
    assert host["action_normalization"]["test_norm"]["action"]["q99"] == (1.,) * 7
    payload["metadata"]["normalization_contract"]["action_statistics"]["q99"][0] = 2.
    with pytest.raises(QuICDescriptorError, match="SHA256 or host identity"):
        descriptor_from_settings(write_settings())


def test_quic_policy_resolves_digest_identity_and_container_paths_separately(tmp_path):
    profile = tmp_path / "local.yaml"
    profile.write_text(f'''schema_version: "0.1.0"
kind: local_profile
id: test
paths:
  checkpoint_root: {tmp_path}/checkpoints
  dataset_root: {tmp_path}/datasets
  runs_root: {tmp_path}/runs
devices:
  primary_gpu: cuda:0
''')
    resolver = ConfigResolver(ROOT / "configs", repository_root=ROOT)
    experiment = "configs/experiments/evaluation/libero10-quic-adapter-rpc-smoke.yaml"
    a = resolver.resolve(experiment, local_profile=profile, environment={"OVLAB_QUIC_ADAPTER_PATH": "/adapters/a"})
    b = resolver.resolve(experiment, local_profile=profile, environment={"OVLAB_QUIC_ADAPTER_PATH": "/adapters/b"})
    assert isinstance(a.policy_settings, OpenVlaQuICSettings)
    assert a.policy_settings.adapter_path == "/adapters/a"
    assert a.scientific_config_hash == b.scientific_config_hash
    assert a.execution_config_hash != b.execution_config_hash


def test_quantized_quic_and_unsupported_codec_fail_during_schema_validation():
    policy = load(ROOT / "configs/policies/openvla-quic/adapter-libero10.yaml")
    policy["settings"]["runtime"]["quantization"] = "4bit"
    with pytest.raises(ValueError, match="dense"):
        validate(policy, "test", "policy")
