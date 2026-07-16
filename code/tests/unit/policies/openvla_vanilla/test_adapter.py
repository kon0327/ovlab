from dataclasses import replace

import numpy as np
import pytest

from helpers.contexts import make_episode_context, make_run_context
from ovlab_core.contracts import (
    AdapterState, ColorSpace, ImageEncoding, ImageObservation, PolicyObservation, StepId,
)
from ovlab_openvla_common import OpenVlaModelSource
from ovlab_openvla_vanilla import (
    HuggingFaceOpenVlaRuntime, OpenVlaCheckpointError, OpenVlaDependencyError,
    OpenVlaInferenceError, OpenVlaPreprocessingError, OpenVlaVanillaAdapter,
    OpenVlaVanillaSettings,
)
from ovlab_policy_sdk import PolicyLifecycleError

from helpers.fake_openvla import FakeOpenVlaRuntime, SequenceClock


def test_capabilities_and_checkpoint_identity(settings, runtime):
    adapter = OpenVlaVanillaAdapter(settings, runtime)
    with pytest.raises(PolicyLifecycleError):
        _ = adapter.capabilities
    caps = adapter.initialize(make_run_context())
    requirement = caps.observation_requirements
    assert len(requirement.images) == 1
    assert requirement.images[0].name == "camera.primary.rgb"
    assert requirement.images[0].shapes == ((256, 256, 3),)
    assert requirement.images[0].dtype == "uint8"
    assert not caps.supports_action_chunks and caps.minimum_action_horizon == caps.maximum_action_horizon == 1
    assert caps.supports_dynamic_instructions and not caps.exposes_raw_policy_output
    assert caps.metadata["checkpoint_identity"]["settings_hash"] == settings.settings_hash
    assert runtime.loaded


def test_prediction_contract_prompt_camera_unnorm_timing_and_no_mutation(settings, runtime, observation):
    source_before = observation.images[0].data.copy()
    adapter = OpenVlaVanillaAdapter(settings, runtime, clock_ns=SequenceClock(), wall_clock_ns=lambda: 500)
    adapter.initialize(make_run_context())
    episode = make_episode_context()
    adapter.reset_episode(episode)
    prediction = adapter.predict(observation)
    assert prediction.actions.shape == (1, 7) and prediction.actions.dtype == np.float32
    np.testing.assert_array_equal(prediction.actions[0], [0, 0, 0, 0, 0, 0, -1])
    assert prediction.horizon == 1 and prediction.inference_duration_ns == 100
    assert dict(prediction.metadata) == {
        "preprocessing_duration_ns": 20, "model_duration_ns": 30, "postprocessing_duration_ns": 10,
    }
    image, prompt, key = runtime.calls[-1]
    assert image.shape == (256, 256, 3) and key == "bridge_orig"
    assert prompt == "In: What action should the robot take to put the object in the basket?\nOut:"
    np.testing.assert_array_equal(observation.images[0].data, source_before)
    assert prediction.raw_output is None


def test_dynamic_instruction_is_formatted_each_prediction(settings, runtime, observation):
    clock = SequenceClock((100, 160, 170, 200, 300, 360, 370, 400))
    adapter = OpenVlaVanillaAdapter(settings, runtime, clock_ns=clock, wall_clock_ns=lambda: 500)
    adapter.initialize(make_run_context()); adapter.reset_episode(make_episode_context())
    adapter.predict(observation)
    updated_instruction = replace(observation.instruction, text="Pick up the cup")
    adapter.predict(replace(observation, instruction=updated_instruction, step_id=StepId("episode-0-step-1")))
    assert runtime.calls[0][1] != runtime.calls[1][1]
    assert runtime.calls[1][1].endswith("pick up the cup?\nOut:")


def test_missing_ambiguous_and_invalid_camera(settings, runtime, observation):
    adapter = OpenVlaVanillaAdapter(settings, runtime, clock_ns=SequenceClock())
    adapter.initialize(make_run_context()); adapter.reset_episode(make_episode_context())
    with pytest.raises(Exception, match="camera"):
        adapter.predict(replace(observation, images=()))
    # New adapter/clock for each failed attempt because timing clocks are deliberately finite.
    duplicate = replace(observation, images=(observation.images[0], observation.images[0]))
    adapter = OpenVlaVanillaAdapter(settings, runtime, clock_ns=SequenceClock())
    adapter.initialize(make_run_context()); adapter.reset_episode(make_episode_context())
    with pytest.raises(Exception, match="ambiguous"):
        adapter.predict(duplicate)


@pytest.mark.parametrize("shape,dtype", [((255, 256, 3), np.uint8), ((256, 256, 1), np.uint8), ((256, 256, 3), np.float32)])
def test_invalid_image_shape_or_dtype(settings, runtime, observation, shape, dtype):
    image = ImageObservation("camera.primary.rgb", np.zeros(shape, dtype=dtype), 2,
                             ImageEncoding.RAW, ColorSpace.RGB, "agentview")
    adapter = OpenVlaVanillaAdapter(settings, runtime, clock_ns=SequenceClock())
    adapter.initialize(make_run_context()); adapter.reset_episode(make_episode_context())
    with pytest.raises(Exception):
        adapter.predict(replace(observation, images=(image,)))


@pytest.mark.parametrize("action", [np.zeros(6), np.r_[np.zeros(6), np.nan], np.r_[np.zeros(6), np.inf]])
def test_wrong_or_nonfinite_decoded_action_fails(settings, observation, action):
    adapter = OpenVlaVanillaAdapter(settings, FakeOpenVlaRuntime(action), clock_ns=SequenceClock())
    adapter.initialize(make_run_context()); adapter.reset_episode(make_episode_context())
    with pytest.raises(OpenVlaInferenceError):
        adapter.predict(observation)


def test_lifecycle_close_and_deterministic_episode_ids(settings, observation):
    runtime = FakeOpenVlaRuntime()
    adapter = OpenVlaVanillaAdapter(settings, runtime, clock_ns=SequenceClock())
    adapter.initialize(make_run_context())
    with pytest.raises(PolicyLifecycleError): adapter.predict(observation)
    episode = make_episode_context(); adapter.reset_episode(episode)
    first = adapter.predict(observation); adapter.end_episode(episode)
    with pytest.raises(PolicyLifecycleError): adapter.predict(observation)
    adapter.reset_episode(episode)
    adapter._clock_ns = SequenceClock()
    second = adapter.predict(observation)
    assert first.prediction_id == second.prediction_id
    adapter.close(); adapter.close()
    assert runtime.closed and adapter.state is AdapterState.CLOSED
    with pytest.raises(PolicyLifecycleError): adapter.predict(observation)


def test_raw_output_is_bounded_numpy_when_enabled(settings, observation):
    configured = replace(settings, record_raw_output=True)
    adapter = OpenVlaVanillaAdapter(configured, FakeOpenVlaRuntime(), clock_ns=SequenceClock(), wall_clock_ns=lambda: 500)
    adapter.initialize(make_run_context()); adapter.reset_episode(make_episode_context())
    prediction = adapter.predict(observation)
    assert isinstance(prediction.raw_output.value, np.ndarray)
    assert prediction.raw_output.value.shape == (7,) and not prediction.raw_output.value.flags.writeable


def test_missing_action_statistics_fails_during_initialize(settings):
    adapter = OpenVlaVanillaAdapter(settings, FakeOpenVlaRuntime(keys=("libero_10",)))
    with pytest.raises(OpenVlaCheckpointError, match="statistics"):
        adapter.initialize(make_run_context())


def test_missing_local_snapshot_is_rejected_without_model_loader(tmp_path):
    settings = OpenVlaVanillaSettings(OpenVlaModelSource("not/a/cached-repo"), "bridge_orig")
    def missing(**kwargs):
        raise FileNotFoundError(kwargs["repo_id"])
    with pytest.raises(OpenVlaCheckpointError, match="local Hugging Face cache"):
        HuggingFaceOpenVlaRuntime._resolve(settings.model, missing, True)


def test_runtime_dependency_error_preserves_cause(monkeypatch):
    import builtins
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "torch": raise ImportError("blocked for test")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(OpenVlaDependencyError) as caught:
        HuggingFaceOpenVlaRuntime._runtime_imports()
    assert isinstance(caught.value.__cause__, ImportError)
