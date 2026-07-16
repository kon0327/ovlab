"""Explicitly opt-in tests for the tested OpenVLA Conda environment."""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest

from ovlab_core.contracts import (
    ColorSpace, EpisodeContext, EpisodeId, ImageEncoding, ImageObservation, Instruction,
    InstructionId, InstructionSource, PolicyObservation, RunContext, RunId, StepId, TaskId,
)
from ovlab_openvla_common import LiberoActionCodec, OpenVlaDecodedAction, OpenVlaModelSource, OpenVlaPromptFormatter
from ovlab_openvla_vanilla import (
    HuggingFaceOpenVlaRuntime, ModelDType, OpenVlaVanillaAdapter, OpenVlaVanillaSettings,
)

pytestmark = [pytest.mark.openvla, pytest.mark.gpu, pytest.mark.manual]


def _configuration(record_raw_output=False):
    source = os.environ.get("OVLAB_OPENVLA_CHECKPOINT")
    key = os.environ.get("OVLAB_OPENVLA_UNNORM_KEY")
    if not source or not key:
        pytest.skip("set OVLAB_OPENVLA_CHECKPOINT and OVLAB_OPENVLA_UNNORM_KEY")
    attention = os.environ.get("OVLAB_OPENVLA_ATTENTION", "flash_attention_2") or None
    return OpenVlaVanillaSettings(
        OpenVlaModelSource(source, os.environ.get("OVLAB_OPENVLA_REVISION")),
        key,
        device=os.environ.get("OVLAB_OPENVLA_DEVICE", "cuda:0"),
        model_dtype=ModelDType(os.environ.get("OVLAB_OPENVLA_DTYPE", "bfloat16")),
        attention_implementation=attention,
        record_raw_output=record_raw_output,
    )


def _image():
    path = os.environ.get("OVLAB_OPENVLA_IMAGE_NPY")
    value = np.load(path, allow_pickle=False) if path else np.zeros((256, 256, 3), dtype=np.uint8)
    assert value.shape == (256, 256, 3) and value.dtype == np.uint8
    return value


def _contexts():
    run = RunContext(RunId("manual-openvla"), 1, "manual OpenVLA smoke", 7)
    instruction = Instruction(InstructionId("manual-instruction"), "put the object in the basket", 2,
                              InstructionSource.USER)
    episode = EpisodeContext(run.run_id, TaskId("manual-task"), EpisodeId("manual-episode"), 0, 7, instruction)
    image = ImageObservation("camera.primary.rgb", _image(), 3, ImageEncoding.RAW, ColorSpace.RGB, "manual")
    observation = PolicyObservation(StepId("manual-step-0"), 3, instruction, (image,))
    return run, episode, observation


def test_real_gpu_smoke_uses_only_local_checkpoint():
    adapter = OpenVlaVanillaAdapter(_configuration())
    run, episode, observation = _contexts()
    try:
        capabilities = adapter.initialize(run)
        adapter.reset_episode(episode)
        prediction = adapter.predict(observation)
        assert prediction.actions.shape == (1, 7)
        assert prediction.actions.dtype == np.float32
        assert np.all(np.isfinite(prediction.actions))
        assert prediction.action_spec is capabilities.output_action_spec
        assert prediction.inference_duration_ns > 0
    finally:
        adapter.close()


def test_regression_matches_legacy_smoke_stages():
    settings = _configuration(record_raw_output=True)
    runtime = HuggingFaceOpenVlaRuntime()
    adapter = OpenVlaVanillaAdapter(settings, runtime)
    run, episode, observation = _contexts()
    try:
        adapter.initialize(run)
        image = np.array(observation.images[0].data, copy=True)
        legacy_prompt = f"In: What action should the robot take to {observation.instruction.text.lower()}?\nOut:"
        assert OpenVlaPromptFormatter().format(observation.instruction.text) == legacy_prompt

        dtype = {
            ModelDType.BFLOAT16: runtime._torch.bfloat16,
            ModelDType.FLOAT16: runtime._torch.float16,
            ModelDType.FLOAT32: runtime._torch.float32,
        }[settings.model_dtype]
        legacy_inputs = runtime._processor(legacy_prompt, runtime._Image.fromarray(image).convert("RGB"))
        legacy_shapes = {key: tuple(int(size) for size in value.shape)
                         for key, value in legacy_inputs.items() if hasattr(value, "shape")}
        legacy_inputs = legacy_inputs.to(settings.device, dtype=dtype)
        runtime._synchronize()
        with runtime._torch.inference_mode():
            legacy_decoded = np.asarray(runtime._model.predict_action(
                **legacy_inputs, unnorm_key=settings.unnorm_key, do_sample=False
            ))
        runtime._synchronize()
        legacy_final = LiberoActionCodec().encode(OpenVlaDecodedAction(legacy_decoded))

        adapter.reset_episode(episode)
        current = adapter.predict(observation)
        current_decoded = current.raw_output.value
        np.testing.assert_allclose(current_decoded, legacy_decoded, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(current.actions[0, 6:], legacy_final[6:])
        np.testing.assert_allclose(current.actions[0, :6], legacy_final[:6], rtol=1e-5, atol=1e-6)
        assert current.actions.shape == (1, 7) and current.actions.dtype == np.float32
        assert dict(current.metadata["runtime"]["processor_input_shapes"]) == legacy_shapes
    finally:
        adapter.close()
