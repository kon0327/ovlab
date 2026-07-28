from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from helpers.contexts import make_episode_context, make_run_context
from ovlab_core.contracts import (
    ActionPrediction,
    ColorSpace,
    EpisodeContext,
    ImageEncoding,
    ImageObservation,
    ImageObservationSpec,
    Instruction,
    InstructionId,
    InstructionSource,
    ObservationRequirements,
    OVLAB_CONTRACT_VERSION,
    PolicyCapabilities,
    PolicyObservation,
    PredictionId,
    RunContext,
    StepId,
)
from ovlab_openvla_common import libero_target_action_spec
from ovlab_policy_sdk import PolicyAdapter
from ovlab_remote_policy.service import PolicyService


class SevenDPolicy(PolicyAdapter):
    def _initialize(self, run_context: RunContext) -> PolicyCapabilities:
        del run_context
        return PolicyCapabilities(
            component_name="test-seven-d-policy",
            component_version="1.0.0",
            contract_version=OVLAB_CONTRACT_VERSION,
            observation_requirements=ObservationRequirements(
                images=(ImageObservationSpec(
                    name="camera.primary.rgb",
                    shapes=((2, 2, 3),),
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
            metadata={"policy_family": "test"},
        )

    def _reset_episode(self, episode_context: EpisodeContext) -> None:
        del episode_context

    def _predict(self, observation: PolicyObservation) -> ActionPrediction:
        return ActionPrediction(
            prediction_id=PredictionId(f"{observation.step_id}:prediction"),
            step_id=observation.step_id,
            actions=np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]], dtype=np.float32),
            action_spec=self.capabilities.output_action_spec,
            timestamp_ns=observation.timestamp_ns + 1,
            inference_duration_ns=17,
            horizon=1,
            metadata={"model_duration_ns": 11},
        )

    def _close(self) -> None:
        pass


def identity_provider(capabilities):
    return {
        "model_identity": {"checkpoint": "test-checkpoint", "component": capabilities.component_name},
        "normalization_identity": {"unnorm_key": "test", "action_statistics_identity": "sha256:test"},
        "prompt_template_identity": "test-template@1",
        "action_codec_identity": {
            "identifier": "test-codec@1",
            "conversion_owner": "SevenDPolicy",
            "application_count": 1,
            "output_gripper_convention": "closed_positive",
        },
        "runtime_versions": {"python": "test", "policy_component": "test-seven-d-policy@1.0.0"},
    }


@pytest.fixture
def contexts():
    return make_run_context(), make_episode_context()


@pytest.fixture
def observation():
    instruction = Instruction(
        InstructionId("instruction-0"), "move deterministically", 2, InstructionSource.BENCHMARK
    )
    image = ImageObservation(
        "camera.primary.rgb",
        np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        3,
        ImageEncoding.RAW,
        ColorSpace.RGB,
        "agentview",
    )
    return PolicyObservation(StepId("episode-0-step-0"), 3, instruction, (image,))


@pytest.fixture
def running_service(tmp_path):
    socket_path = tmp_path / "p.sock"
    service = PolicyService(socket_path, SevenDPolicy(), identity_provider=identity_provider)
    thread = threading.Thread(target=service.serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while not socket_path.exists() and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert socket_path.is_socket()
    yield socket_path, thread
    thread.join(timeout=2)


@pytest.fixture
def running_method_service(tmp_path):
    socket_path = tmp_path / "method.sock"

    def with_method(capabilities):
        identity = identity_provider(capabilities)
        identity["method_descriptor"] = {
            "family": "lora",
            "artifact_form": "merged_full_weights",
            "merge_status": "merged",
            "active_peft_adapter": False,
        }
        return identity

    service = PolicyService(socket_path, SevenDPolicy(), identity_provider=with_method)
    thread = threading.Thread(target=service.serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while not socket_path.exists() and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert socket_path.is_socket()
    yield socket_path, thread
    thread.join(timeout=2)
