from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from ovlab_core.contracts import GripperConvention
from ovlab_remote_policy import (
    RemotePolicyAdapter,
    RemotePolicyError,
    RemotePolicyProtocolError,
    RemotePolicyTimeoutError,
    UnixPolicyClient,
)
from ovlab_remote_policy.protocol import make_request, recv_frame, send_frame


def test_mock_remote_lifecycle_handshake_action_and_separate_timing(
    running_service, contexts, observation
):
    socket_path, thread = running_service
    run, episode = contexts
    policy = RemotePolicyAdapter(UnixPolicyClient(socket_path, request_timeout_s=2))
    capabilities = policy.initialize(run)
    assert policy.client.health()["state"] == "ready"
    assert capabilities.output_action_spec.gripper_convention is GripperConvention.CLOSED_POSITIVE
    assert policy.handshake["protocol_version"] == "ovlab-policy-rpc/1.0.0"
    assert policy.handshake["action_codec_identity"]["application_count"] == 1
    policy.reset_episode(episode)
    prediction = policy.predict(observation)
    assert prediction.actions.dtype == np.float32
    assert prediction.actions.shape == (1, 7)
    np.testing.assert_array_equal(
        prediction.actions[0], np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0], dtype=np.float32)
    )
    assert prediction.inference_duration_ns == 17
    assert prediction.metadata["service_inference_duration_ns"] == 17
    assert prediction.metadata["rpc_round_trip_duration_ns"] > 0
    policy.end_episode(episode)
    policy.close()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not socket_path.exists()


def test_optional_method_descriptor_preserves_vanilla_protocol_compatibility(
    running_method_service, contexts
):
    socket_path, thread = running_method_service
    run, _ = contexts
    policy = RemotePolicyAdapter(UnixPolicyClient(socket_path, request_timeout_s=2))
    policy.initialize(run)
    assert policy.handshake["protocol_version"] == "ovlab-policy-rpc/1.0.0"
    assert policy.handshake["method_descriptor"] == {
        "family": "lora",
        "artifact_form": "merged_full_weights",
        "merge_status": "merged",
        "active_peft_adapter": False,
    }
    policy.close()
    thread.join(timeout=2)
    assert not thread.is_alive()


def _one_shot_server(socket_path, handler):
    def serve():
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        server.listen(1)
        connection, _ = server.accept()
        with connection:
            handler(connection)
        server.close()
        socket_path.unlink(missing_ok=True)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while not socket_path.exists() and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    return thread


def test_request_timeout_is_bounded(tmp_path):
    socket_path = tmp_path / "timeout.sock"
    thread = _one_shot_server(socket_path, lambda connection: (recv_frame(connection), time.sleep(0.2)))
    client = UnixPolicyClient(socket_path, request_timeout_s=0.02)
    client.connect()
    with pytest.raises(RemotePolicyTimeoutError, match="health timed out"):
        client.health()
    client.close_socket()
    thread.join(timeout=1)


def test_service_crash_is_reported(tmp_path):
    socket_path = tmp_path / "crash.sock"
    thread = _one_shot_server(socket_path, lambda connection: recv_frame(connection))
    client = UnixPolicyClient(socket_path, request_timeout_s=1)
    client.connect()
    with pytest.raises(RemotePolicyError, match="connection failed"):
        client.health()
    client.close_socket()
    thread.join(timeout=1)


def test_malformed_stale_response_is_rejected(tmp_path):
    socket_path = tmp_path / "malformed.sock"

    def wrong_request_id(connection):
        request = recv_frame(connection)
        send_frame(connection, {
            "protocol_version": request["protocol_version"],
            "request_id": "stale-request",
            "status": "ok",
            "payload": {"state": "ready", "pid": 1},
        })

    thread = _one_shot_server(socket_path, wrong_request_id)
    client = UnixPolicyClient(socket_path, request_timeout_s=1)
    client.connect()
    with pytest.raises(RemotePolicyProtocolError, match="stale response"):
        client.health()
    client.close_socket()
    thread.join(timeout=1)


def test_duplicate_request_is_rejected_and_service_remains_closeable(running_service):
    socket_path, thread = running_service
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(socket_path))
    request = make_request("duplicate-id", "health", {})
    send_frame(connection, request)
    assert recv_frame(connection)["status"] == "ok"
    send_frame(connection, request)
    duplicate = recv_frame(connection)
    assert duplicate["status"] == "error"
    assert duplicate["error"]["code"] == "duplicate_request"
    send_frame(connection, make_request("close-id", "close", {}))
    assert recv_frame(connection)["status"] == "ok"
    connection.close()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_incompatible_client_receives_clear_protocol_error(running_service):
    socket_path, thread = running_service
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(socket_path))
    send_frame(connection, {
        "protocol_version": "ovlab-policy-rpc/9.0.0",
        "request_id": "incompatible-id",
        "operation": "health",
        "payload": {},
    })
    response = recv_frame(connection)
    assert response["request_id"] == "incompatible-id"
    assert response["status"] == "error"
    assert response["error"]["code"] == "protocol_error"
    assert "unsupported protocol version" in response["error"]["message"]
    connection.close()
    thread.join(timeout=2)
    assert not thread.is_alive()
