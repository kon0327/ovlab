"""Generic service loop backed by an ordinary PolicyAdapter."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Mapping
from pathlib import Path

from ovlab_policy_sdk import PolicyAdapter

from ovlab_remote_policy.errors import RemotePolicyProtocolError
from ovlab_remote_policy.protocol import (
    PROTOCOL_VERSION,
    capabilities_to_wire,
    episode_context_from_wire,
    make_error,
    make_success,
    predict_payload_to_observation,
    prediction_to_wire,
    recv_frame,
    require_exact_keys,
    run_context_from_wire,
    send_frame,
    validate_request_envelope,
)


class PolicyService:
    """Strict, single-client, local policy service."""

    def __init__(
        self,
        socket_path: str | Path,
        adapter: PolicyAdapter,
        *,
        identity_provider: Callable[[object], Mapping[str, object]],
    ) -> None:
        if not isinstance(adapter, PolicyAdapter):
            raise TypeError("adapter must be a PolicyAdapter")
        if not callable(identity_provider):
            raise TypeError("identity_provider must be callable")
        self.socket_path = Path(socket_path)
        self.adapter = adapter
        self.identity_provider = identity_provider
        self._seen_requests: set[str] = set()
        self._seen_steps: set[str] = set()
        self._episode_id: str | None = None

    def serve(self) -> None:
        if self.socket_path.exists():
            raise RuntimeError(f"socket path already exists: {self.socket_path}")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            server.listen(1)
            connection, _ = server.accept()
            with connection:
                self._serve_connection(connection)
        finally:
            try:
                self.adapter.close()
            finally:
                server.close()
                self.socket_path.unlink(missing_ok=True)

    def _serve_connection(self, connection: socket.socket) -> None:
        while True:
            try:
                raw_request = recv_frame(connection)
            except RemotePolicyProtocolError:
                return
            try:
                request = validate_request_envelope(raw_request)
            except RemotePolicyProtocolError as exc:
                request_id = raw_request.get("request_id")
                if isinstance(request_id, str) and request_id and len(request_id) <= 128:
                    send_frame(connection, make_error(request_id, "protocol_error", str(exc)))
                return
            request_id = request["request_id"]
            if request_id in self._seen_requests:
                send_frame(connection, make_error(request_id, "duplicate_request", "request_id was already used"))
                continue
            self._seen_requests.add(request_id)
            try:
                response, should_close = self._dispatch(request["operation"], request["payload"])
            except Exception as exc:
                code = "protocol_error" if isinstance(exc, RemotePolicyProtocolError) else "service_error"
                send_frame(connection, make_error(request_id, code, str(exc)))
                continue
            send_frame(connection, make_success(request_id, response))
            if should_close:
                return

    def _dispatch(self, operation: str, payload: dict[str, object]) -> tuple[dict[str, object], bool]:
        if operation == "initialize":
            require_exact_keys(payload, required={"run_context"}, where="initialize payload")
            capabilities = self.adapter.initialize(run_context_from_wire(payload["run_context"]))
            if capabilities.observation_requirements.proprioception:
                raise RemotePolicyProtocolError("remote service policies may not require proprioception")
            identity = dict(self.identity_provider(capabilities))
            required_identity = {
                "model_identity", "normalization_identity", "prompt_template_identity",
                "action_codec_identity", "runtime_versions",
            }
            missing = required_identity - set(identity)
            unexpected = set(identity) - required_identity
            if missing or unexpected:
                raise RemotePolicyProtocolError(
                    f"service identity mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "capabilities": capabilities_to_wire(capabilities),
                **identity,
            }
            return response, False
        if operation == "health":
            require_exact_keys(payload, required=set(), where="health payload")
            return {"state": self.adapter.state.value, "pid": os.getpid()}, False
        if operation == "reset_episode":
            require_exact_keys(payload, required={"episode_context"}, where="reset_episode payload")
            context = episode_context_from_wire(payload["episode_context"])
            self.adapter.reset_episode(context)
            self._episode_id = str(context.episode_id)
            self._seen_steps.clear()
            return {"episode_id": self._episode_id}, False
        if operation == "predict":
            episode_id, observation = predict_payload_to_observation(payload)
            if self._episode_id is None:
                raise RemotePolicyProtocolError("predict requires an active episode")
            if episode_id != self._episode_id:
                raise RemotePolicyProtocolError(
                    f"stale episode_id {episode_id!r}; active episode is {self._episode_id!r}"
                )
            step_id = str(observation.step_id)
            if step_id in self._seen_steps:
                raise RemotePolicyProtocolError(f"duplicate or stale step_id: {step_id!r}")
            self._seen_steps.add(step_id)
            return prediction_to_wire(self.adapter.predict(observation)), False
        if operation == "end_episode":
            require_exact_keys(payload, required={"episode_context"}, where="end_episode payload")
            context = episode_context_from_wire(payload["episode_context"])
            if str(context.episode_id) != self._episode_id:
                raise RemotePolicyProtocolError("end_episode context does not match the active episode")
            self.adapter.end_episode(context)
            episode_id = self._episode_id
            self._episode_id = None
            self._seen_steps.clear()
            return {"episode_id": episode_id}, False
        if operation == "close":
            require_exact_keys(payload, required=set(), where="close payload")
            self.adapter.close()
            return {"closed": True}, True
        raise AssertionError(f"unhandled operation {operation}")
