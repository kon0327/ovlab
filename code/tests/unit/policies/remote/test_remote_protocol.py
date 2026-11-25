from __future__ import annotations

import base64
import socket
import struct

import numpy as np
import pytest

from ovlab_core.contracts import (
    ActionPrediction, ColorSpace, ImageEncoding, ImageObservation, PolicyObservation,
    PredictionId, ProprioceptiveObservation, StepId,
)
from ovlab_openvla_common import libero_target_action_spec
from ovlab_remote_policy import RemotePolicyProtocolError
from ovlab_remote_policy.protocol import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    encode_frame,
    observation_to_predict_payload,
    predict_payload_to_observation,
    prediction_from_wire,
    prediction_to_wire,
    recv_frame,
    validate_request_envelope,
)


def test_length_prefixed_frame_round_trip_and_version():
    left, right = socket.socketpair()
    try:
        message = {"protocol_version": PROTOCOL_VERSION, "value": "ok"}
        left.sendall(encode_frame(message))
        assert recv_frame(right) == message
    finally:
        left.close()
        right.close()


def test_frame_rejects_declared_oversize_before_body_read():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!I", MAX_FRAME_BYTES + 1))
        with pytest.raises(RemotePolicyProtocolError, match="declares"):
            recv_frame(right)
    finally:
        left.close()
        right.close()


def test_incompatible_protocol_version_is_rejected_clearly():
    with pytest.raises(RemotePolicyProtocolError, match="unsupported protocol version"):
        validate_request_envelope({
            "protocol_version": "ovlab-policy-rpc/9.0.0",
            "request_id": "request-0",
            "operation": "health",
            "payload": {},
        })


def test_prediction_schema_round_trip_contains_only_public_fields(observation):
    payload = observation_to_predict_payload(observation, episode_id="episode-0")
    assert set(payload) == {"episode_id", "step_id", "timestamp_ns", "instruction", "images", "proprioception"}
    assert set(payload["images"][0]) == {"name", "camera_name", "timestamp_ns", "shape", "dtype", "encoding", "color_space", "data_b64"}
    episode_id, restored = predict_payload_to_observation(payload)
    assert episode_id == "episode-0"
    assert restored.step_id == observation.step_id
    np.testing.assert_array_equal(restored.images[0].data, observation.images[0].data)


def test_image_provenance_metadata_is_not_transmitted(observation):
    image = observation.images[0]
    enriched = type(image)(
        image.name,
        image.data,
        image.timestamp_ns,
        image.encoding,
        image.color_space,
        image.camera_name,
        {"native_key": "agentview_image", "transform": "rotate_180", "reward": 99},
    )
    enriched_observation = PolicyObservation(
        observation.step_id,
        observation.timestamp_ns,
        observation.instruction,
        (enriched,),
    )
    payload = observation_to_predict_payload(enriched_observation, episode_id="episode-0")
    serialized = str(payload)
    assert "native_key" not in serialized
    assert "transform" not in serialized
    assert "reward" not in serialized


@pytest.mark.parametrize("field", ["reward", "success", "simulator_state", "object_poses", "contacts"])
def test_prediction_schema_rejects_every_privileged_field(observation, field):
    payload = observation_to_predict_payload(observation, episode_id="episode-0")
    payload[field] = False
    with pytest.raises(RemotePolicyProtocolError, match="forbidden fields"):
        predict_payload_to_observation(payload)


def test_prediction_schema_rejects_metadata_but_round_trips_negotiated_modalities(observation):
    with_metadata = PolicyObservation(
        observation.step_id,
        observation.timestamp_ns,
        observation.instruction,
        observation.images,
        metadata={"reward": 1.0},
    )
    with pytest.raises(RemotePolicyProtocolError, match="metadata is forbidden"):
        observation_to_predict_payload(with_metadata, episode_id="episode-0")
    wrist = ImageObservation(
        "camera.wrist.rgb", np.full((2, 2, 3), 9, dtype=np.uint8), 3,
        ImageEncoding.RAW, ColorSpace.RGB, "robot0_eye_in_hand",
    )
    proprio = ProprioceptiveObservation("robot.proprioception", np.zeros(8, dtype=np.float32), 3, ("unitless",) * 8)
    with_proprio = PolicyObservation(
        observation.step_id,
        observation.timestamp_ns,
        observation.instruction,
        observation.images + (wrist,),
        (proprio,),
    )
    payload = observation_to_predict_payload(with_proprio, episode_id="episode-0")
    assert set(payload) == {"episode_id", "step_id", "timestamp_ns", "instruction", "images", "proprioception"}
    serialized = str(payload)
    for forbidden in ("reward", "success", "simulator_state", "object_poses", "contacts"):
        assert forbidden not in serialized
    _, restored = predict_payload_to_observation(payload)
    assert [item.name for item in restored.images] == ["camera.primary.rgb", "camera.wrist.rgb"]
    assert [item.name for item in restored.proprioception] == ["robot.proprioception"]
    np.testing.assert_array_equal(restored.proprioception[0].values, proprio.values)


def test_prediction_schema_rejects_mismatched_image_size(observation):
    payload = observation_to_predict_payload(observation, episode_id="episode-0")
    payload["images"][0]["data_b64"] = base64.b64encode(b"short").decode()
    with pytest.raises(RemotePolicyProtocolError, match="does not match declared shape"):
        predict_payload_to_observation(payload)


def test_action_chunk_wire_round_trip_preserves_horizon_dtype_and_offsets():
    actions = np.arange(56, dtype=np.float32).reshape(8, 7) / 100
    prediction = ActionPrediction(
        PredictionId("chunk-0"), StepId("step-0"), actions, libero_target_action_spec(),
        4, 3, 8, metadata={"action_offsets": list(range(8)), "chunk_id": "chunk-0"},
    )
    wire = prediction_to_wire(prediction)
    assert set(wire) >= {"actions", "shape", "horizon"}
    assert wire["shape"] == [8, 7] and wire["horizon"] == 8
    restored = prediction_from_wire(wire)
    assert restored["actions"].dtype == np.float32 and restored["actions"].shape == (8, 7)
    np.testing.assert_array_equal(restored["actions"], actions)


def test_prediction_round_trip_without_images_preserves_observation_time(observation):
    source = PolicyObservation(observation.step_id, 900, observation.instruction)
    payload = observation_to_predict_payload(source, episode_id='opaque:episode')
    assert payload['images'] == payload['proprioception'] == []
    episode, restored = predict_payload_to_observation(payload)
    assert episode == 'opaque:episode' and restored.timestamp_ns == 900
    assert restored.instruction.timestamp_ns == observation.instruction.timestamp_ns


def test_named_depth_and_float64_proprioception_round_trip(observation):
    image = ImageObservation('sensor.depth', np.arange(6, dtype=np.float32).reshape(2, 3),
                             701, ImageEncoding.RAW, ColorSpace.DEPTH, 'tool-mounted-depth')
    vector = ProprioceptiveObservation('arm.angles', np.arange(4, dtype=np.float64), 702, ('rad',) * 4)
    source = PolicyObservation(observation.step_id, 703, observation.instruction, (image,), (vector,))
    payload = observation_to_predict_payload(source, episode_id='opaque:episode')
    _, restored = predict_payload_to_observation(payload)
    assert restored.images[0].name == 'sensor.depth'
    assert restored.images[0].camera_name == 'tool-mounted-depth'
    assert restored.images[0].color_space is ColorSpace.DEPTH
    assert restored.images[0].timestamp_ns == 701
    assert restored.proprioception[0].timestamp_ns == 702
    np.testing.assert_array_equal(restored.images[0].data, image.data)
    assert restored.proprioception[0].values.dtype == np.float64
    np.testing.assert_array_equal(restored.proprioception[0].values, vector.values)


@pytest.mark.parametrize('mutation', ['overflow-shape', 'object-dtype', 'duplicate-name'])
def test_generic_modality_limits_reject_invalid_arrays(observation, mutation):
    payload = observation_to_predict_payload(observation, episode_id='opaque:episode')
    if mutation == 'overflow-shape': payload['images'][0]['shape'] = [2**62, 2**62]
    if mutation == 'object-dtype': payload['images'][0]['dtype'] = 'object'
    if mutation == 'duplicate-name': payload['images'].append(dict(payload['images'][0]))
    with pytest.raises(RemotePolicyProtocolError):
        predict_payload_to_observation(payload)


def test_two_joint_float64_action_and_gripper_none_survive_rpc():
    from dataclasses import replace
    from ovlab_core.contracts import ActionRepresentation, ActionSpec, GripperConvention, RotationRepresentation
    spec = ActionSpec(2, ActionRepresentation.JOINT_DELTA, (), (), (), RotationRepresentation.NONE,
                      GripperConvention.NONE, ('rad', 'rad'), np.full(2, -1., np.float64),
                      np.full(2, 1., np.float64), 'float64', 5.)
    for horizon in (1, 3):
        actions = np.full((horizon, 2), 0.123456789123, dtype=np.float64)
        prediction = ActionPrediction(PredictionId('two-joint'), StepId('step'), actions, spec, 2, 1, horizon)
        result = prediction_from_wire(prediction_to_wire(prediction))
        assert result['actions'].dtype == np.float64 and result['horizon'] == horizon
        assert result['action_spec'].gripper_convention is GripperConvention.NONE
        np.testing.assert_array_equal(result['actions'], actions)
