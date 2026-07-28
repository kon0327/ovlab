from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from helpers.contexts import make_episode_context, make_run_context
from helpers.fake_openvla import FakeOpenVlaRuntime, SequenceClock
from ovlab_benchctl import ConfigResolver
from ovlab_core.contracts import (
    ColorSpace, ImageEncoding, ImageObservation, Instruction, InstructionId,
    InstructionSource, PolicyObservation, StepId,
)
from ovlab_openvla_common import (
    CheckpointFileIdentity,
    OpenVlaArtifactForm,
    OpenVlaMethodFamily,
    OpenVlaModelSource,
    OpenVlaRuntimeArtifact,
    method_descriptor_from_registry,
)
from ovlab_openvla_lora_merged import OpenVlaMergedLoraAdapter
from ovlab_openvla_vanilla import OpenVlaVanillaAdapter, OpenVlaVanillaSettings


REPOSITORY = Path(__file__).resolve().parents[5]
CONFIGS = REPOSITORY / "configs"


@pytest.fixture
def observation():
    instruction = Instruction(
        InstructionId("instruction-0"), "Put The Object In The Basket", 1,
        InstructionSource.BENCHMARK,
    )
    data = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    image = ImageObservation(
        "camera.primary.rgb", data, 2, ImageEncoding.RAW, ColorSpace.RGB, "agentview"
    )
    return PolicyObservation(StepId("episode-0-step-0"), 2, instruction, (image,))


def registered_entry():
    registry = ConfigResolver(CONFIGS, repository_root=REPOSITORY).load_component(
        "resources/registry.yaml", "resource_registry"
    )
    return registry["checkpoints"]["openvla-7b-finetuned-libero-10"]


def test_registered_identity_is_merged_lora_not_vanilla_or_qp():
    descriptor = method_descriptor_from_registry(registered_entry())
    metadata = descriptor.as_metadata()
    assert descriptor.family is OpenVlaMethodFamily.LORA
    assert descriptor.artifact_form is OpenVlaArtifactForm.MERGED_FULL_WEIGHTS
    assert metadata["merge_status"] == "merged"
    assert metadata["active_peft_adapter"] is False
    assert metadata["runtime_peft_modules"] is False
    assert metadata["quantization"] == "none"
    assert metadata["qp_profile"] is None
    assert metadata["lora_configuration"] == {
        "alpha": 16,
        "bias": "none",
        "dropout": 0.0,
        "merge_procedure": "merge_and_unload()+save_pretrained()",
        "modules_to_save": None,
        "rank": 32,
        "scaling": 0.5,
        "target_policy": "all-linear",
    }


@pytest.mark.parametrize(
    "change,match",
    [
        ({"quantization": "4bit"}, "QLoRA or quantization"),
        ({"active_peft_adapter": True}, "must not report active PEFT"),
        ({"artifact_form": OpenVlaArtifactForm.FULL_WEIGHTS}, "merged_full_weights"),
    ],
)
def test_method_identity_rejects_qlora_active_peft_and_full_weight_misclassification(change, match):
    descriptor = method_descriptor_from_registry(registered_entry())
    with pytest.raises(ValueError, match=match):
        replace(descriptor, **change)


def local_artifact(tmp_path):
    contents = {
        "config.json": b"{}",
        "model.safetensors.index.json": b"{}\n",
        "model-00001-of-00001.safetensors": b"stable-weights",
    }
    files = []
    for name, content in contents.items():
        (tmp_path / name).write_bytes(content)
        files.append(CheckpointFileIdentity(name, len(content), hashlib.sha256(content).hexdigest()))
    aggregate = hashlib.sha256("".join(item.manifest_line() for item in files).encode()).hexdigest()
    artifact = OpenVlaRuntimeArtifact(
        "test-merged-lora", str(tmp_path), "a" * 40, "merged_full_weights", "merged",
        "not_present_in_published_artifact", "not_recoverable_from_published_artifact",
        tuple(files), aggregate,
    )
    return artifact


def test_immutable_artifact_manifest_accepts_absent_adapter_config_and_detects_mutation(tmp_path):
    artifact = local_artifact(tmp_path)
    verified = artifact.verify(tmp_path)
    assert verified["adapter_config"] == "not_present_in_published_artifact"
    assert not (tmp_path / "adapter_config.json").exists()
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpectedly contains adapter_config"):
        artifact.verify(tmp_path)
    (tmp_path / "adapter_config.json").unlink()
    (tmp_path / "config.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size mismatch"):
        artifact.verify(tmp_path)


def test_merged_adapter_requires_manifest_and_reuses_exact_action_contract(tmp_path, observation):
    descriptor = method_descriptor_from_registry(registered_entry())
    source = OpenVlaModelSource(str(tmp_path), "a" * 40)
    missing = OpenVlaVanillaSettings(source, "libero_10", method_descriptor=descriptor)
    with pytest.raises(ValueError, match="artifact manifest"):
        OpenVlaMergedLoraAdapter(missing, FakeOpenVlaRuntime(keys=("libero_10",))).initialize(
            make_run_context()
        )

    artifact = local_artifact(tmp_path)
    source = OpenVlaModelSource(str(tmp_path), artifact.revision, artifact.aggregate_sha256)
    settings = OpenVlaVanillaSettings(
        source,
        "libero_10",
        method_descriptor=descriptor,
        runtime_artifact=artifact,
    )
    runtime = FakeOpenVlaRuntime(keys=("libero_10",))
    adapter = OpenVlaMergedLoraAdapter(settings, runtime, clock_ns=SequenceClock())
    capabilities = adapter.initialize(make_run_context())
    adapter.reset_episode(make_episode_context())
    prediction = adapter.predict(observation)
    assert capabilities.component_name == "ovlab-openvla-lora-merged"
    assert capabilities.metadata["policy_family"] == "openvla-lora-merged"
    assert capabilities.metadata["method_descriptor"]["family"] == "lora"
    assert prediction.actions.dtype == np.float32
    assert prediction.actions.shape == (1, 7)
    np.testing.assert_array_equal(prediction.actions[0], [0, 0, 0, 0, 0, 0, -1])
    assert capabilities.metadata["action_codec_owner"] == "OpenVlaMergedLoraAdapter"
    adapter.close()


def test_vanilla_adapter_rejects_lora_method_identity(tmp_path):
    descriptor = method_descriptor_from_registry(registered_entry())
    settings = OpenVlaVanillaSettings(
        OpenVlaModelSource(str(tmp_path)), "libero_10", method_descriptor=descriptor
    )
    with pytest.raises(ValueError, match="requires method family 'vanilla'"):
        OpenVlaVanillaAdapter(settings, FakeOpenVlaRuntime(keys=("libero_10",))).initialize(
            make_run_context()
        )
