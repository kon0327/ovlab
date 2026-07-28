from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess

import numpy as np
import pytest

from ovlab_benchctl import ConfigResolver
from ovlab_core.contracts import (
    ColorSpace, EpisodeContext, EpisodeId, ImageEncoding, ImageObservation, Instruction,
    InstructionId, InstructionSource, PolicyObservation, RunContext, RunId, StepId, TaskId,
    ProprioceptiveObservation,
)
from ovlab_openvla_common import OpenVlaDecodedActionChunk
from ovlab_openvla_oft import OpenVlaOftAdapter, OpenVlaOftArtifact, validate_oft_method
from ovlab_openvla_oft.runtime import OftRuntimePrediction
from ovlab_openvla_oft.runtime import OPENVLA_OFT_GIT_COMMIT

REPOSITORY = Path(__file__).resolve().parents[5]


def test_policy_service_entrypoint_imports_without_loading_the_model():
    from ovlab_openvla_oft.service import main

    assert callable(main)


def test_recorded_oft_source_commit_matches_the_pinned_submodule_gitlink():
    actual = subprocess.check_output(
        ["git", "-C", str(REPOSITORY / "external/openvla-oft"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert OPENVLA_OFT_GIT_COMMIT == actual
    assert _resolved().policy_settings.artifact.method["training_provenance"]["source_commit"] == actual


def _resolved():
    return ConfigResolver(REPOSITORY / "configs", repository_root=REPOSITORY).resolve(
        "configs/experiments/libero10-openvla-oft-rpc-smoke.yaml",
        local_profile=REPOSITORY / "configs/local/gate-b-showrack.yaml",
        execution_profile="profiles/libero-bench-egl.yaml",
        environment={},
    )


def test_registered_method_is_strict_native_oft_and_accounts_all_adaptation_components():
    settings = _resolved().policy_settings
    assert isinstance(settings.artifact, OpenVlaOftArtifact)
    names = {item.path for item in settings.artifact.files}
    assert len(names) == 25
    assert {
        "configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py",
        "processor_config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer.model",
        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    } <= names
    method = settings.artifact.method
    validate_oft_method(method)
    assert method["acronym_expansion"] == "optimized_fine_tuning"
    assert method["family"] == "openvla_oft"
    assert method["film"] is method["diffusion"] is False
    assert method["quantization"] == "none"
    assert not any(key.lower().startswith("qp") for key in method)
    counts = _resolved().scientific_config["resource_registry"]["checkpoints"][
        "openvla-oft-7b-finetuned-libero-10"
    ]["artifact"]["parameter_counts"]
    assert counts["auxiliary"] == counts["action_head"] + counts["proprio_projector"]
    assert counts["complete_adaptation"] == counts["lora_trainable"] + counts["auxiliary"]
    assert counts["total_runtime"] == counts["merged_backbone"] + counts["auxiliary"]
    lora = method["lora"]
    assert lora["tensor_dtype_counts"] == {"BF16": 879}
    assert sum(lora["tensor_shape_counts"].values()) == lora["published_tensor_count"]


@pytest.mark.parametrize("field,value", [
    ("family", "lora"), ("film", True), ("diffusion", True),
    ("quantization", "4bit"), ("qp_classification", "QP0"),
])
def test_wrong_method_classifications_are_rejected(field, value):
    method = dict(_resolved().policy_settings.artifact.method)
    method[field] = value
    with pytest.raises(ValueError, match="classification"):
        validate_oft_method(method)


class FakeOftRuntime:
    def __init__(self):
        self.load_counts = {"backbone": 0, "processor": 0, "published_peft_adapter": 0,
                            "action_head": 0, "proprio_projector": 0}
        self.predict_count = 0
        self.closed = False

    def load(self, settings):
        del settings
        for key in ("backbone", "processor", "action_head", "proprio_projector"):
            self.load_counts[key] += 1
        return {
            "load_counts": dict(self.load_counts), "runtime_active_adapter": False,
            "cold_component_loading_duration_ns": 11, "warmup_duration_ns": 7,
            "timing_method": "fake synchronized timing",
        }

    def reset_episode(self, seed):
        del seed

    def predict(self, primary, wrist, proprio, instruction):
        assert instruction == "put the object in the basket"
        assert primary.shape == wrist.shape == (256, 256, 3)
        assert proprio.shape == (8,)
        self.predict_count += 1
        decoded = np.zeros((8, 7), dtype=np.float32)
        decoded[:, 6] = np.linspace(-0.01, 1.01, 8, dtype=np.float32)
        return OftRuntimePrediction(
            decoded.copy(), OpenVlaDecodedActionChunk(decoded), proprio.copy(), 2, 5,
            {"processor_calls": [], "prompt": "In: What action should the robot take to put the object in the basket?\nOut:"},
        )

    def close(self):
        self.closed = True

    def runtime_metadata(self):
        return {"load_counts": dict(self.load_counts), "prediction_count": self.predict_count}


def test_adapter_negotiates_native_inputs_and_returns_exact_chunk_with_one_codec_stage():
    runtime = FakeOftRuntime()
    adapter = OpenVlaOftAdapter(_resolved().policy_settings, runtime)
    run = RunContext(RunId("oft-unit"), 1, "OFT unit", 42)
    instruction = Instruction(InstructionId("i"), "put the object in the basket", 2, InstructionSource.BENCHMARK)
    episode = EpisodeContext(run.run_id, TaskId("libero/10/0"), EpisodeId("e"), 0, 42, instruction)
    images = (
        ImageObservation("camera.primary.rgb", np.zeros((256, 256, 3), dtype=np.uint8), 3,
                         ImageEncoding.RAW, ColorSpace.RGB, "agentview"),
        ImageObservation("camera.wrist.rgb", np.ones((256, 256, 3), dtype=np.uint8), 3,
                         ImageEncoding.RAW, ColorSpace.RGB, "robot0_eye_in_hand"),
    )
    proprio = ProprioceptiveObservation(
        "robot.proprioception", np.zeros(8, dtype=np.float32), 3, ("m",) * 3 + ("rad",) * 5,
    )
    observation = PolicyObservation(StepId("s"), 3, instruction, images, (proprio,))
    capabilities = adapter.initialize(run)
    assert capabilities.minimum_action_horizon == capabilities.maximum_action_horizon == 8
    assert not capabilities.supports_single_action and capabilities.supports_action_chunks
    assert [item.name for item in capabilities.observation_requirements.images] == [
        "camera.primary.rgb", "camera.wrist.rgb",
    ]
    adapter.reset_episode(episode)
    prediction = adapter.predict(observation)
    assert prediction.actions.shape == (8, 7) and prediction.actions.dtype == np.float32
    assert prediction.metadata["action_offsets"] == tuple(range(8))
    assert prediction.metadata["codec_application_count_per_action"] == 1
    assert set(np.unique(prediction.actions[:, 6])) <= {-1.0, 1.0}
    assert runtime.predict_count == 1
    assert runtime.load_counts == {"backbone": 1, "processor": 1, "published_peft_adapter": 0,
                                   "action_head": 1, "proprio_projector": 1}
    adapter.close()
    assert runtime.closed
