"""CPU lifecycle contract using an explicit tiny model, without OpenVLA weights."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from safetensors.torch import save_file

from quic_petf_vla import AdapterConfig, inject, save_adapter
from ovlab_openvla_common import OpenVlaModelSource, libero_target_action_spec
from ovlab_openvla_quic import OpenVlaQuICSettings, descriptor_from_settings


ROOT = Path(__file__).resolve().parents[4]


class TinyDenseHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 8)
        self.norm_stats = {"fixture": {"action": {"q01": [-1.] * 7, "q99": [1.] * 7}}}

    @property
    def device(self):
        return self.projection.weight.device

    @property
    def dtype(self):
        return self.projection.weight.dtype

    def predict_action(self, **kwargs):
        assert kwargs["unnorm_key"] == "fixture" and kwargs["do_sample"] is False
        return np.asarray([.1, .2, .3, 0., 0., 0., .75], dtype=np.float32)


class Batch(dict):
    def to(self, *args, **kwargs):
        return self


class Processor:
    def __call__(self, prompt, image):
        assert prompt == "In: What action should the robot take to move cube?\nOut:"
        assert image.mode == "RGB"
        return Batch()


@pytest.mark.parametrize("normalization_mode", ("host", "train-q99"))
def test_dense_provider_load_predict_close_and_host_integrity(tmp_path, monkeypatch, normalization_mode):
    monkeypatch.syspath_prepend(str(ROOT / "external/openvla-quic"))
    from quic_petf import host as backend
    from ovlab_openvla_quic.runtime import create_provider

    checkpoint = tmp_path / "host"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "openvla"}))
    save_file({"projection.weight": torch.ones(8, 4)}, str(checkpoint / "model.safetensors"))
    host = {"checkpoint": str(checkpoint), "checkpoint_id": "tiny-dense-fixture", "revision": "a" * 40,
            "adaptation_state": "none", "unnorm_key": "fixture", "dtype": "float32", "device": "cpu"}
    model = TinyDenseHost()
    original = copy.deepcopy(model)
    stats = copy.deepcopy(model.norm_stats["fixture"]["action"])
    if normalization_mode == "train-q99":
        stats = {"q01": [-2.] * 7, "q99": [2.] * 7, "mask": [True] * 6 + [False]}
    normalization = {"unnorm_key": "fixture", "action_statistics": stats, "mode": normalization_mode}
    normalization["sha256"] = hashlib.sha256(json.dumps(normalization, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if normalization_mode == "train-q99":
        host["normalization_contract_sha256"] = normalization["sha256"]
        host["action_normalization"] = {"fixture": {"action": stats}}
    identity = backend.checkpoint_identity(host)
    inject(model, {"projection": AdapterConfig(block_count=1, base_size=3)})
    artifact = tmp_path / "adapter"
    save_adapter(model, artifact, identity, metadata={
        "normalization_contract": normalization,
        "preprocessing": {"prompt_style": "modern", "center_crop": False},
    })
    settings = OpenVlaQuICSettings(
        model=OpenVlaModelSource("tiny-dense-fixture", "a" * 40, local_path=str(checkpoint)),
        unnorm_key="fixture", adapter_path=str(artifact), target_action_spec=libero_target_action_spec(),
        adapter_manifest_sha256=hashlib.sha256((artifact / "adapter.json").read_bytes()).hexdigest(),
        adapter_tensor_sha256=hashlib.sha256((artifact / "adapter.safetensors").read_bytes()).hexdigest(),
        input_image_shape=(2, 2, 3), device="cpu", model_dtype="float32", action_codec="openvla-to-libero-v1",
    )
    descriptor = descriptor_from_settings(settings)
    provider = create_provider(variant="quic-peft", method_descriptor=descriptor.as_metadata())
    request = {"method_descriptor": descriptor.canonical_dict(), "scientific_identity_hash": descriptor.scientific_hash}
    def load_tiny_host(config):
        loaded = copy.deepcopy(original)
        backend.apply_action_normalization(loaded, config)
        return loaded, Processor(), {"fixture": True}

    monkeypatch.setattr(backend, "load_host", load_tiny_host)
    result = provider.load(request)
    assert result["host_identity"] == identity
    assert provider.model.norm_stats["fixture"]["action"] == stats
    assert original.norm_stats["fixture"]["action"]["q99"] == [1.] * 7
    assert provider.load_counts() == {"base_model": 1, "quic_artifact": 1, "processor": 1}
    with pytest.raises(RuntimeError, match="already loaded"):
        provider.load(request)
    provider.reset_episode({"episode_id": "ep", "seed": 1})
    prediction = provider.predict({"episode_id": "ep", "instruction": "Move Cube", "proprioception": {},
                                   "images": {"camera.primary.rgb": {"data": np.zeros((2, 2, 3), dtype=np.uint8)}}})
    assert prediction["actions"].shape == (1, 7)
    assert prediction["actions"][0, 6] == -1.0
    assert prediction["actions"].dtype == np.float32
    provider.close()
    assert provider.model is None and provider.processor is None
    if normalization_mode == "train-q99":
        changed_host = copy.deepcopy(host)
        changed_host["action_normalization"]["fixture"]["action"]["q99"][0] += 0.1
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            backend.checkpoint_identity(changed_host)
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "openvla", "changed": True}))
    with pytest.raises(ValueError, match="identity mismatch"):
        provider.load(request)


def test_inline_normalization_can_supply_a_missing_host_key_and_rejects_unbound_stats(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "external/openvla-quic"))
    from quic_petf.host import apply_action_normalization

    model = TinyDenseHost()
    stats = {"q01": [-2.] * 7, "q99": [2.] * 7, "mask": [True] * 6 + [False]}
    contract = {"unnorm_key": "new_key", "action_statistics": stats, "mode": "train-q99"}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
    host = {"unnorm_key": "new_key", "action_normalization": {"new_key": {"action": stats}},
            "normalization_contract_sha256": digest}
    apply_action_normalization(model, host)
    assert model.norm_stats["new_key"]["action"] == stats
    assert "fixture" in model.norm_stats
    host.pop("normalization_contract_sha256")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        apply_action_normalization(model, host)
