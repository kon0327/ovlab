"""Synchronous OpenVLA Vanilla PolicyAdapter."""

import time
from collections.abc import Callable

import numpy as np

from ovlab_core.contracts import (
    ActionPrediction,
    ColorSpace,
    EpisodeContext,
    ImageEncoding,
    ImageObservationSpec,
    ObservationRequirements,
    OVLAB_CONTRACT_VERSION,
    PolicyCapabilities,
    PolicyObservation,
    PredictionId,
    PredictionValidity,
    RawPolicyOutput,
    RunContext,
)
from ovlab_openvla_common import (
    LiberoActionCodec,
    OpenVlaActionCodecError,
    OpenVlaObservationError,
    OpenVlaPromptFormatter,
    select_canonical_rgb,
)
from ovlab_policy_sdk import PolicyAdapter

from .errors import OpenVlaActionDecodeError, OpenVlaInferenceError, OpenVlaPreprocessingError
from .runtime import HuggingFaceOpenVlaRuntime, OpenVlaRuntime
from .settings import OpenVlaVanillaSettings


class OpenVlaVanillaAdapter(PolicyAdapter):
    def __init__(
        self,
        settings: OpenVlaVanillaSettings,
        runtime: OpenVlaRuntime | None = None,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        wall_clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        super().__init__()
        if not isinstance(settings, OpenVlaVanillaSettings):
            raise TypeError("settings must be OpenVlaVanillaSettings")
        self.settings = settings
        self._runtime = HuggingFaceOpenVlaRuntime() if runtime is None else runtime
        self._clock_ns = clock_ns
        self._wall_clock_ns = wall_clock_ns
        self._formatter = OpenVlaPromptFormatter(settings.prompt_template)
        self._codec = LiberoActionCodec(settings.action_codec)
        self._prediction_index = 0
        self._identity = None

    def _initialize(self, run_context: RunContext) -> PolicyCapabilities:
        del run_context
        self._identity = self._runtime.load(self.settings)
        image_spec = ImageObservationSpec(
            name=self.settings.canonical_camera_name,
            shapes=(self.settings.input_image_shape,),
            dtype="uint8",
            encodings=(ImageEncoding.RAW,),
            color_spaces=(ColorSpace.RGB,),
        )
        metadata = {
            "policy_family": "openvla-vanilla",
            "prompt_template": self._formatter.identifier,
            "action_codec": self.settings.action_codec.identifier,
            "checkpoint_identity": self._identity.as_metadata(),
            "timing": "local preprocessing + model execution + target codec",
            "determinism": "greedy evaluation; bitwise GPU determinism is not claimed",
        }
        return PolicyCapabilities(
            component_name="ovlab-openvla-vanilla",
            component_version="0.1.0",
            contract_version=OVLAB_CONTRACT_VERSION,
            observation_requirements=ObservationRequirements(
                images=(image_spec,), minimum_image_count=1, maximum_image_count=1,
                minimum_proprioception_count=0, maximum_proprioception_count=None,
            ),
            output_action_spec=self.settings.target_action_spec,
            supports_single_action=True,
            supports_action_chunks=False,
            minimum_action_horizon=1,
            maximum_action_horizon=1,
            supports_dynamic_instructions=True,
            supports_deterministic_reset=self.settings.deterministic_inference,
            exposes_raw_policy_output=self.settings.record_raw_output,
            metadata=metadata,
        )

    def _reset_episode(self, episode_context: EpisodeContext) -> None:
        self._prediction_index = 0
        self._runtime.reset_episode(episode_context.seed)

    def _predict(self, observation: PolicyObservation) -> ActionPrediction:
        started = self._clock_ns()
        try:
            image = select_canonical_rgb(observation, self.settings.canonical_camera_name)
            if image.shape != self.settings.input_image_shape:
                raise OpenVlaPreprocessingError(
                    f"camera shape {image.shape} differs from configured {self.settings.input_image_shape}"
                )
            prompt = self._formatter.format(observation.instruction.text)
            runtime_result = self._runtime.predict(image, prompt, self.settings.unnorm_key)
            post_started = self._clock_ns()
            action = self._codec.encode(runtime_result.decoded_action)
            post_finished = self._clock_ns()
        except OpenVlaObservationError as exc:
            raise OpenVlaPreprocessingError(str(exc)) from exc
        except OpenVlaActionCodecError as exc:
            raise OpenVlaActionDecodeError(str(exc)) from exc
        except (OpenVlaPreprocessingError, OpenVlaInferenceError):
            raise
        except Exception as exc:
            raise OpenVlaActionDecodeError("failed to validate or convert OpenVLA action") from exc
        finished = self._clock_ns()
        total = finished - started
        post_duration = post_finished - post_started
        phases = runtime_result.preprocessing_duration_ns + runtime_result.model_duration_ns + post_duration
        if any(value < 0 for value in (total, runtime_result.preprocessing_duration_ns,
                                       runtime_result.model_duration_ns, post_duration)):
            raise OpenVlaInferenceError("inference clock moved backwards")
        if phases > total:
            raise OpenVlaInferenceError("phase durations exceed total inference duration")
        episode = self._episode_context
        assert episode is not None
        prediction_id = PredictionId(f"{episode.episode_id}:prediction:{self._prediction_index}")
        self._prediction_index += 1
        timestamp = self._wall_clock_ns()
        raw = None
        if self.settings.record_raw_output:
            raw = RawPolicyOutput(
                prediction_id=prediction_id,
                value=runtime_result.decoded_action.value,
                timestamp_ns=timestamp,
                metadata={"stage": "decoded-before-target-codec"},
            )
        prediction_metadata = {
            "preprocessing_duration_ns": runtime_result.preprocessing_duration_ns,
            "model_duration_ns": runtime_result.model_duration_ns,
            "postprocessing_duration_ns": post_duration,
        }
        if runtime_result.metadata:
            prediction_metadata["runtime"] = dict(runtime_result.metadata)
        return ActionPrediction(
            prediction_id=prediction_id,
            step_id=observation.step_id,
            actions=action[np.newaxis, :],
            action_spec=self.settings.target_action_spec,
            timestamp_ns=timestamp,
            inference_duration_ns=total,
            horizon=1,
            validity=PredictionValidity.VALID,
            raw_output=raw,
            metadata=prediction_metadata,
        )

    def _end_episode(self, episode_context: EpisodeContext) -> None:
        del episode_context
        self._prediction_index = 0

    def _close(self) -> None:
        self._runtime.close()
        self._identity = None
