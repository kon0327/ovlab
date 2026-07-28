"""OVLAB PolicyAdapter for native OpenVLA-OFT action-chunk inference."""

from __future__ import annotations

import hashlib
import time

import numpy as np

from ovlab_core.contracts import (
    ActionPrediction, ColorSpace, ImageEncoding, ImageObservationSpec, ObservationRequirements,
    OVLAB_CONTRACT_VERSION, PolicyCapabilities, PredictionId, PredictionValidity,
    ProprioceptiveObservationSpec, RawPolicyOutput,
)
from ovlab_openvla_common import LiberoActionChunkCodec
from ovlab_policy_sdk import PolicyAdapter

from .runtime import OpenVlaOftRuntime
from .settings import OpenVlaOftSettings


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class OpenVlaOftAdapter(PolicyAdapter):
    component_name = "ovlab-openvla-oft"

    def __init__(self, settings: OpenVlaOftSettings, runtime=None, *, clock_ns=time.perf_counter_ns,
                 wall_clock_ns=time.time_ns) -> None:
        super().__init__()
        self.settings = settings
        self.runtime = runtime or OpenVlaOftRuntime()
        self._codec = LiberoActionChunkCodec()
        self._clock_ns, self._wall_clock_ns = clock_ns, wall_clock_ns
        self._prediction_index = 0
        self._runtime_identity = {}

    def _initialize(self, run_context):
        del run_context
        self._runtime_identity = self.runtime.load(self.settings)
        image_specs = tuple(ImageObservationSpec(
            name, (self.settings.input_image_shape,), "uint8", (ImageEncoding.RAW,), (ColorSpace.RGB,),
        ) for name in (self.settings.primary_camera_name, self.settings.wrist_camera_name))
        proprio = ProprioceptiveObservationSpec(
            self.settings.proprioception_name, ((8,),), "float32", ("m",) * 3 + ("rad",) * 5,
        )
        method = {**self.settings.artifact.method, **{
            "method_name": "OpenVLA-OFT", "published_unmerged_adapter": "available",
            "runtime_active_adapter": False, "backbone_merge_status": "merged",
            "checkpoint_identity": self.settings.artifact.as_metadata(),
            "load_counts": dict(self.runtime.load_counts),
            "parameter_counts": dict(self.settings.artifact.parameter_counts),
            "byte_counts": dict(self.settings.artifact.byte_counts),
            "cold_component_loading_duration_ns": self._runtime_identity["cold_component_loading_duration_ns"],
            "warmup_duration_ns": self._runtime_identity["warmup_duration_ns"],
            "timing_method": self._runtime_identity["timing_method"],
        }}
        return PolicyCapabilities(
            self.component_name, "0.1.0", OVLAB_CONTRACT_VERSION,
            ObservationRequirements(image_specs, (proprio,), 2, 2, 1, 1),
            self.settings.target_action_spec,
            False, True, 8, 8, True, True, self.settings.record_raw_output,
            {
                "policy_family": "openvla_oft", "method_descriptor": method,
                "prompt_template": "openvla-v1@1.0.0",
                "action_codec": "openvla-decoded-to-libero-osc-pose@1.0.0",
                "action_codec_owner": type(self).__name__, "runtime": self._runtime_identity,
            },
        )

    def _reset_episode(self, episode_context):
        self._prediction_index = 0
        self.runtime.reset_episode(episode_context.seed)

    def _predict(self, observation):
        images = {item.name: item for item in observation.images}
        proprios = {item.name: item for item in observation.proprioception}
        expected_images = {self.settings.primary_camera_name, self.settings.wrist_camera_name}
        if set(images) != expected_images or set(proprios) != {self.settings.proprioception_name}:
            raise ValueError("OFT prediction requires exactly negotiated primary, wrist, and proprio channels")
        primary, wrist = (images[name].data for name in (
            self.settings.primary_camera_name, self.settings.wrist_camera_name,
        ))
        proprio = proprios[self.settings.proprioception_name].values
        for name, image in (("primary", primary), ("wrist", wrist)):
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError(f"OFT {name} image must be HWC uint8 [256,256,3]")
        if proprio.shape != (8,) or proprio.dtype != np.float32:
            raise ValueError("OFT proprioception must be float32 [8]")
        started = self._clock_ns()
        result = self.runtime.predict(primary, wrist, proprio, observation.instruction.text)
        post_started = self._clock_ns()
        actions = self._codec.encode(result.decoded_actions)
        post_finished = self._clock_ns()
        prediction_id = PredictionId(
            f"{self._episode_context.episode_id}:prediction:{self._prediction_index}"
        )
        self._prediction_index += 1
        timestamp = self._wall_clock_ns()
        metadata = {
            "chunk_id": str(prediction_id), "prediction_request_step_id": str(observation.step_id),
            "action_offsets": list(range(8)), "generated_chunk_length": 8,
            "raw_normalized_chunk_sha256": _sha(result.normalized_actions),
            "unnormalized_chunk_sha256": _sha(result.decoded_actions.value),
            "final_libero_chunk_sha256": _sha(actions),
            "primary_rgb_sha256": _sha(primary), "wrist_rgb_sha256": _sha(wrist),
            "proprioception_sha256": _sha(proprio),
            "normalized_proprioception_sha256": _sha(result.normalized_proprioception),
            "preprocessing_duration_ns": result.preprocessing_duration_ns,
            "model_duration_ns": result.model_duration_ns,
            "postprocessing_duration_ns": post_finished - post_started,
            "amortized_generation_per_predicted_action_ns": result.model_duration_ns // 8,
            "codec_application_count_per_action": 1,
            "runtime": result.metadata,
        }
        raw = None
        if self.settings.record_raw_output:
            raw = RawPolicyOutput(prediction_id, result.decoded_actions.value, timestamp,
                                  {"stage": "unnormalized-before-target-codec", "horizon": 8})
        return ActionPrediction(
            prediction_id, observation.step_id, actions, self.settings.target_action_spec,
            timestamp, self._clock_ns() - started, 8, PredictionValidity.VALID,
            raw_output=raw, metadata=metadata,
        )

    def _end_episode(self, episode_context):
        del episode_context
        self._prediction_index = 0

    def _close(self):
        self.runtime.close()
        self._runtime_identity = {}

    def runtime_metadata(self):
        return self.runtime.runtime_metadata()
