"""Test-image-only deterministic provider for transport and LIBERO qualification."""

import json
import os
import time

import numpy as np

from ovlab_benchctl.application import OvlabApplication
from ovlab_core.contracts import (
    ColorSpace,
    ImageEncoding,
    ImageObservationSpec,
    ObservationRequirements,
    OVLAB_CONTRACT_VERSION,
    ActionPrediction,
    PolicyCapabilities,
    PredictionId,
)
from ovlab_openvla_common import libero_target_action_spec
from ovlab_policy_sdk import PolicyAdapter


class TransportSmokePolicy(PolicyAdapter):
    """Deterministic action provider that exists only in the non-production image."""

    def __init__(self):
        super().__init__()
        raw = os.environ.get("OVLAB_TEST_ACTION_SEQUENCE_JSON")
        values = json.loads(raw) if raw else [[0, 0, 0, 0, 0, 0, -1]]
        sequence = np.asarray(values, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1:] != (7,) or len(sequence) < 1:
            raise ValueError("OVLAB_TEST_ACTION_SEQUENCE_JSON must be a non-empty float32 [N,7] array")
        if not np.all(np.isfinite(sequence)) or np.any(sequence < -1) or np.any(sequence > 1):
            raise ValueError("qualification actions must be finite and within [-1,1]")
        self.sequence = sequence
        self.sequence_index = 0
        self.prediction_count = 0

    def _initialize(self, _context):
        return PolicyCapabilities(
            component_name="qualification-test-policy",
            component_version="qualification/1.0.0",
            contract_version=OVLAB_CONTRACT_VERSION,
            observation_requirements=ObservationRequirements(
                images=(ImageObservationSpec(
                    name="camera.primary.rgb",
                    shapes=((256, 256, 3),),
                    dtype="uint8",
                    encodings=(ImageEncoding.RAW,),
                    color_spaces=(ColorSpace.RGB,),
                ),),
                minimum_image_count=1,
                maximum_image_count=1,
            ),
            output_action_spec=libero_target_action_spec(),
            supports_single_action=True,
            supports_action_chunks=False,
            minimum_action_horizon=1,
            maximum_action_horizon=1,
            supports_dynamic_instructions=True,
            supports_deterministic_reset=True,
            exposes_raw_policy_output=False,
            metadata={
                "checkpoint_identity": {
                    "unnorm_key": "libero_10",
                    "action_statistics_identity": "test-only:deterministic-sequence",
                },
                "prompt_template": "qualification/authoritative-instruction-passthrough@1",
                "action_codec": "qualification/canonical-libero-action@1",
                "action_codec_owner": "test-only-provider",
                "qualification_only": True,
                "action_sequence": self.sequence.tolist(),
            },
        )

    def _reset_episode(self, _context):
        self.sequence_index = 0

    def _predict(self, observation):
        started = time.monotonic_ns()
        index = min(self.sequence_index, len(self.sequence) - 1)
        action = self.sequence[index:index + 1]
        self.sequence_index += 1
        self.prediction_count += 1
        finished = time.monotonic_ns()
        return ActionPrediction(
            PredictionId(f"qualification-{self.prediction_count:08d}"),
            observation.step_id,
            action,
            self.capabilities.output_action_spec,
            max(observation.timestamp_ns, finished),
            max(finished - started, 1),
            1,
            metadata={
                "qualification_only": True,
                "action_sequence_index": index,
                "model_duration_ns": max(finished - started, 1),
            },
        )

    def _close(self):
        pass


def _test_provider(_settings):
    return TransportSmokePolicy()


OvlabApplication._policy_adapter = staticmethod(_test_provider)
