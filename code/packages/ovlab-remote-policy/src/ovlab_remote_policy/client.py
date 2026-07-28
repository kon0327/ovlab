"""Synchronous client for the local policy service."""

from __future__ import annotations

import socket
import time
import uuid
from pathlib import Path
from typing import Any

from ovlab_core.contracts import EpisodeContext, RunContext

from ovlab_remote_policy.errors import (
    RemotePolicyError,
    RemotePolicyProtocolError,
    RemotePolicyServiceError,
    RemotePolicyTimeoutError,
)
from ovlab_remote_policy.protocol import (
    capabilities_from_wire,
    episode_context_to_wire,
    make_request,
    observation_to_predict_payload,
    prediction_from_wire,
    recv_frame,
    require_exact_keys,
    run_context_to_wire,
    send_frame,
    validate_response_envelope,
)


class UnixPolicyClient:
    """One-connection, strictly ordered AF_UNIX policy client."""

    def __init__(self, socket_path: str | Path, *, request_timeout_s: float = 60.0) -> None:
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        self.socket_path = Path(socket_path)
        self.request_timeout_s = float(request_timeout_s)
        self._socket: socket.socket | None = None
        self._prefix = uuid.uuid4().hex
        self._request_index = 0
        self._episode_id: str | None = None
        self.handshake: dict[str, Any] | None = None

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        if self._socket is not None:
            raise RemotePolicyError("policy client is already connected")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.request_timeout_s)
        try:
            sock.connect(str(self.socket_path))
        except socket.timeout as exc:
            sock.close()
            raise RemotePolicyTimeoutError(f"timed out connecting to {self.socket_path}") from exc
        except OSError:
            sock.close()
            raise
        self._socket = sock

    def initialize(self, context: RunContext):
        payload, _ = self._request("initialize", {"run_context": run_context_to_wire(context)})
        require_exact_keys(
            payload,
            required={
                "protocol_version", "capabilities", "model_identity", "normalization_identity",
                "prompt_template_identity", "action_codec_identity", "runtime_versions",
            },
            optional={"method_descriptor"},
            where="initialize response",
        )
        capabilities = capabilities_from_wire(payload["capabilities"])
        self.handshake = dict(payload)
        return capabilities

    def health(self) -> dict[str, Any]:
        payload, _ = self._request("health", {})
        require_exact_keys(payload, required={"state", "pid"}, where="health response")
        return payload

    def reset_episode(self, context: EpisodeContext) -> None:
        payload, _ = self._request("reset_episode", {"episode_context": episode_context_to_wire(context)})
        require_exact_keys(payload, required={"episode_id"}, where="reset_episode response")
        if payload["episode_id"] != str(context.episode_id):
            raise RemotePolicyProtocolError("reset response contains a stale episode_id")
        self._episode_id = str(context.episode_id)

    def predict(self, observation):
        if self._episode_id is None:
            raise RemotePolicyError("predict requires an active remote episode")
        request_payload = observation_to_predict_payload(observation, episode_id=self._episode_id)
        payload, round_trip_ns = self._request("predict", request_payload)
        parsed = prediction_from_wire(payload)
        parsed["rpc_round_trip_duration_ns"] = round_trip_ns
        return parsed

    def end_episode(self, context: EpisodeContext) -> None:
        payload, _ = self._request("end_episode", {"episode_context": episode_context_to_wire(context)})
        require_exact_keys(payload, required={"episode_id"}, where="end_episode response")
        if payload["episode_id"] != str(context.episode_id):
            raise RemotePolicyProtocolError("end response contains a stale episode_id")
        self._episode_id = None

    def request_close(self) -> None:
        if self._socket is None:
            return
        payload, _ = self._request("close", {})
        require_exact_keys(payload, required={"closed"}, where="close response")
        if payload["closed"] is not True:
            raise RemotePolicyProtocolError("service did not acknowledge close")

    def close_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
                self._episode_id = None

    def _request(self, operation: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        if self._socket is None:
            raise RemotePolicyError("policy client is not connected")
        request_id = f"{self._prefix}:{self._request_index}"
        self._request_index += 1
        message = make_request(request_id, operation, payload)
        started = time.perf_counter_ns()
        try:
            send_frame(self._socket, message)
            response = recv_frame(self._socket)
        except socket.timeout as exc:
            raise RemotePolicyTimeoutError(
                f"remote policy {operation} timed out after {self.request_timeout_s:g}s"
            ) from exc
        except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as exc:
            raise RemotePolicyError(f"remote policy connection failed during {operation}: {exc}") from exc
        except RemotePolicyProtocolError as exc:
            if "peer closed the socket" in str(exc):
                raise RemotePolicyError(
                    f"remote policy connection failed during {operation}: peer closed the socket"
                ) from exc
            raise
        finished = time.perf_counter_ns()
        response = validate_response_envelope(response, request_id=request_id)
        if response["status"] == "error":
            error = response["error"]
            raise RemotePolicyServiceError(f"{error['code']}: {error['message']}")
        return response["payload"], finished - started
