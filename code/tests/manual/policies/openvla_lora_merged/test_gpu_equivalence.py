"""D3: accepted OpenVLA preprocessing equivalence for the merged LoRA artifact."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # The validated openvla service environment intentionally has no pytest.
    pytest = None

from ovlab_benchctl import ConfigResolver
from ovlab_core.contracts import (
    ColorSpace, EpisodeContext, EpisodeId, ImageEncoding, ImageObservation, Instruction,
    InstructionId, InstructionSource, PolicyObservation, RunContext, RunId, StepId, TaskId,
)
from ovlab_openvla_common import LiberoActionCodec, OpenVlaDecodedAction, OpenVlaPromptFormatter
from ovlab_openvla_lora_merged import OpenVlaMergedLoraAdapter
from ovlab_openvla_vanilla import HuggingFaceOpenVlaRuntime, ModelDType


if pytest is not None:
    pytestmark = [pytest.mark.openvla, pytest.mark.lora, pytest.mark.gpu, pytest.mark.manual]

REPOSITORY = Path(__file__).resolve().parents[5]
def _settings():
    profile = os.environ.get("OVLAB_LOCAL_PROFILE")
    if not profile:
        raise RuntimeError("D3 requires OVLAB_LOCAL_PROFILE")
    resolved = ConfigResolver(REPOSITORY / "configs", repository_root=REPOSITORY).resolve(
        "configs/experiments/libero10-lora-merged-rpc-smoke.yaml",
        local_profile=Path(profile).resolve(),
        execution_profile="profiles/libero-bench-egl.yaml",
        environment={},
    )
    return replace(resolved.policy_settings, record_raw_output=True)


def _accepted_gate_c_input():
    trace_path = os.environ.get("OVLAB_GATE_C_TRACE")
    if not trace_path:
        raise RuntimeError("D3 requires OVLAB_GATE_C_TRACE pointing to an accepted Gate C trace.json")
    trace = Path(trace_path).resolve()
    payload = json.loads(trace.read_text(encoding="utf-8"))
    observation = payload["observations"][0]
    image_reference = observation["images"][0]["data"]
    image_path = trace.parent / image_reference["$array"]
    image = np.load(image_path, allow_pickle=False)
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == image_reference["sha256"]
    prediction_reference = payload["policy_predictions"][0]["actions"]
    prediction_path = trace.parent / prediction_reference["$array"]
    accepted_final = np.load(prediction_path, allow_pickle=False)[0]
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == prediction_reference["sha256"]
    assert observation["images"][0]["metadata"]["transform"] == "rotate_180"
    return image, observation["instruction"]["text"], accepted_final, image_reference["sha256"]


def _contexts(image, instruction_text):
    run = RunContext(RunId("gate-d-equivalence"), 1, "merged LoRA equivalence", 7)
    instruction = Instruction(
        InstructionId("gate-d-instruction"), instruction_text, 2,
        InstructionSource.BENCHMARK,
    )
    episode = EpisodeContext(
        run.run_id, TaskId("libero/10/0"), EpisodeId("gate-d-equivalence-episode"),
        0, 7, instruction,
    )
    image = ImageObservation(
        "camera.primary.rgb", image, 3,
        ImageEncoding.RAW, ColorSpace.RGB, "canonical",
    )
    observation = PolicyObservation(StepId("gate-d-equivalence-step-0"), 3, instruction, (image,))
    return run, episode, observation


def test_merged_lora_matches_accepted_autoclass_inference_path():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    settings = _settings()
    runtime = HuggingFaceOpenVlaRuntime()
    adapter = OpenVlaMergedLoraAdapter(settings, runtime)
    image, instruction, accepted_final, accepted_image_sha256 = _accepted_gate_c_input()
    run, episode, observation = _contexts(image, instruction)
    source_pixels = observation.images[0].data.copy()
    try:
        capabilities = adapter.initialize(run)
        assert runtime.load_count == 1
        assert runtime.processor_load_count == 1
        assert runtime.peft_adapter_load_count == 0
        assert runtime.runtime_metadata()["active_peft_adapter"] is False
        assert runtime.runtime_metadata()["runtime_peft_modules"] is False
        assert runtime.runtime_metadata()["quantized"] is False
        assert capabilities.metadata["method_descriptor"]["family"] == "lora"
        assert capabilities.metadata["method_descriptor"]["merge_status"] == "merged"
        assert capabilities.metadata["method_descriptor"]["qp_profile"] is None

        prompt = OpenVlaPromptFormatter().format(observation.instruction.text)
        assert prompt == f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        image = np.array(observation.images[0].data, copy=True)
        dtype = {
            ModelDType.BFLOAT16: runtime._torch.bfloat16,
            ModelDType.FLOAT16: runtime._torch.float16,
            ModelDType.FLOAT32: runtime._torch.float32,
        }[settings.model_dtype]
        reference_inputs = runtime._processor(prompt, runtime._Image.fromarray(image).convert("RGB"))
        reference_shapes = {
            key: tuple(int(size) for size in value.shape)
            for key, value in reference_inputs.items()
            if hasattr(value, "shape")
        }
        reference_dtypes = {
            key: str(value.dtype) for key, value in reference_inputs.items() if hasattr(value, "dtype")
        }
        reference_fingerprints = {
            key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in reference_inputs.items()
            if hasattr(value, "detach")
        }
        assert set(reference_inputs) >= {"input_ids", "pixel_values"}
        reference_inputs = reference_inputs.to(settings.device, dtype=dtype)
        runtime._synchronize()
        with runtime._torch.inference_mode():
            reference_decoded = np.asarray(runtime._model.predict_action(
                **reference_inputs,
                unnorm_key=settings.unnorm_key,
                do_sample=False,
            ))
        runtime._synchronize()
        reference_final = LiberoActionCodec().encode(OpenVlaDecodedAction(reference_decoded))

        adapter.reset_episode(episode)
        current = adapter.predict(observation)
        np.testing.assert_allclose(current.raw_output.value, reference_decoded, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(reference_final, accepted_final, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(current.actions[0], reference_final, rtol=1e-5, atol=1e-6)
        assert current.actions.shape == (1, 7) and current.actions.dtype == np.float32
        assert np.all(np.isfinite(current.actions))
        assert np.all(current.actions >= -1) and np.all(current.actions <= 1)
        assert current.actions[0, 6] == -1.0
        assert dict(current.metadata["runtime"]["processor_input_shapes"]) == reference_shapes
        assert dict(current.metadata["runtime"]["processor_input_dtypes"]) == reference_dtypes
        assert dict(current.metadata["runtime"]["processor_input_sha256"]) == reference_fingerprints
        assert reference_dtypes["input_ids"].startswith("torch.int")
        assert reference_dtypes["pixel_values"].startswith("torch.float")
        np.testing.assert_array_equal(observation.images[0].data, source_pixels)
        assert hashlib.sha256(observation.images[0].data.tobytes()).hexdigest() == hashlib.sha256(
            source_pixels.tobytes()
        ).hexdigest()
        assert len(accepted_image_sha256) == 64
        assert runtime.load_count == 1 and runtime.processor_load_count == 1
    finally:
        adapter.close()
    assert runtime._model is None and runtime._processor is None and runtime._torch is None


if __name__ == "__main__":
    test_merged_lora_matches_accepted_autoclass_inference_path()
    print("Gate D3 merged-LoRA AutoClass equivalence: PASS")
