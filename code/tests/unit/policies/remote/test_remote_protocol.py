from __future__ import annotations

import base64
import socket
import struct

import numpy as np
import pytest

from ovlab_core.contracts import PolicyObservation, ProprioceptiveObservation
from ovlab_remote_policy import RemotePolicyProtocolError
from ovlab_remote_policy.protocol import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    encode_frame,
    observation_to_predict_payload,
    predict_payload_to_observation,
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
    assert set(payload) == {"episode_id", "step_id", "instruction", "image"}
    assert set(payload["image"]) == {"shape", "layout", "dtype", "data_b64"}
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


def test_prediction_schema_rejects_metadata_and_proprioception(observation):
    with_metadata = PolicyObservation(
        observation.step_id,
        observation.timestamp_ns,
        observation.instruction,
        observation.images,
        metadata={"reward": 1.0},
    )
    with pytest.raises(RemotePolicyProtocolError, match="metadata is forbidden"):
        observation_to_predict_payload(with_metadata, episode_id="episode-0")
    proprio = ProprioceptiveObservation("robot", np.zeros(1), 3, ("unitless",))
    with_proprio = PolicyObservation(
        observation.step_id,
        observation.timestamp_ns,
        observation.instruction,
        observation.images,
        (proprio,),
    )
    with pytest.raises(RemotePolicyProtocolError, match="proprioception is forbidden"):
        observation_to_predict_payload(with_proprio, episode_id="episode-0")


def test_prediction_schema_rejects_mismatched_image_size(observation):
    payload = observation_to_predict_payload(observation, episode_id="episode-0")
    payload["image"]["data_b64"] = base64.b64encode(b"short").decode()
    with pytest.raises(RemotePolicyProtocolError, match="does not match declared shape"):
        predict_payload_to_observation(payload)
