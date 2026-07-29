"""Test-image-only provider injection for the two-container transport smoke."""

import numpy as np

from ovlab_benchctl.application import OvlabApplication
from ovlab_core.contracts import (
    ColorSpace,
    ImageEncoding,
    ImageObservationSpec,
    ObservationRequirements,
    OVLAB_CONTRACT_VERSION,
    PolicyCapabilities,
)
from ovlab_openvla_common import libero_target_action_spec
from ovlab_policy_sdk import PolicyAdapter


class TransportSmokePolicy(PolicyAdapter):
    """Handshake-only provider that must never execute model inference."""

    def _initialize(self, _context):
        return PolicyCapabilities(
            component_name="handshake-only-policy",
            component_version="deployment-smoke/1.0.0",
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
                    "action_statistics_identity": "test-only:no-weights",
                },
                "prompt_template": "transport-smoke/no-prompt@1",
                "action_codec": "transport-smoke/no-codec@1",
                "action_codec_owner": "test-only-provider",
            },
        )

    def _reset_episode(self, _context):
        raise AssertionError("transport smoke must not start an episode")

    def _predict(self, _observation):
        raise AssertionError("transport smoke must not perform prediction")

    def _close(self):
        pass


def _test_provider(_settings):
    return TransportSmokePolicy()


OvlabApplication._policy_adapter = staticmethod(_test_provider)
