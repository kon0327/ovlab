"""Generic runner-side remote PolicyAdapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from ovlab_core.contracts import ActionPrediction, EpisodeContext, RunContext
from ovlab_policy_sdk import PolicyAdapter

from ovlab_remote_policy.client import UnixPolicyClient


class RemotePolicyAdapter(PolicyAdapter):
    """Expose a remote ordinary PolicyAdapter through the local OVLAB contract."""

    def __init__(
        self,
        client: UnixPolicyClient,
        *,
        start_service: Callable[[], None] | None = None,
        stop_service: Callable[[], None] | None = None,
        transport_metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(client, UnixPolicyClient):
            raise TypeError("client must be a UnixPolicyClient")
        self.client = client
        self._start_service = start_service
        self._stop_service = stop_service
        self._transport_metadata = {} if transport_metadata is None else dict(transport_metadata)
        self._service_health = None

    @property
    def handshake(self):
        return self.client.handshake

    def _initialize(self, run_context: RunContext):
        if self._start_service is not None:
            self._start_service()
        self.client.connect()
        capabilities = self.client.initialize(run_context)
        self._service_health = self.client.health()
        handshake_metadata = {
            key: value for key, value in self.client.handshake.items() if key != "capabilities"
        }
        handshake_metadata["transport"] = dict(self._transport_metadata)
        metadata = dict(capabilities.metadata)
        metadata["remote_policy"] = handshake_metadata
        metadata["remote_policy"]["health_at_initialize"] = dict(self._service_health)
        return replace(capabilities, metadata=metadata)

    def _reset_episode(self, episode_context: EpisodeContext) -> None:
        self.client.reset_episode(episode_context)

    def _predict(self, observation) -> ActionPrediction:
        parsed = self.client.predict(observation)
        metadata = dict(parsed["metadata"])
        metadata.update(
            {
                "service_inference_duration_ns": parsed["inference_duration_ns"],
                "rpc_round_trip_duration_ns": parsed["rpc_round_trip_duration_ns"],
                "rpc_protocol_version": self.client.handshake["protocol_version"],
            }
        )
        return ActionPrediction(
            prediction_id=parsed["prediction_id"],
            step_id=parsed["step_id"],
            actions=np.asarray(parsed["actions"], dtype=np.float32),
            action_spec=parsed["action_spec"],
            timestamp_ns=parsed["timestamp_ns"],
            inference_duration_ns=parsed["inference_duration_ns"],
            horizon=parsed["horizon"],
            validity=parsed["validity"],
            confidence=parsed["confidence"],
            metadata=metadata,
        )

    def _end_episode(self, episode_context: EpisodeContext) -> None:
        self.client.end_episode(episode_context)

    def _close(self) -> None:
        try:
            self.client.request_close()
        finally:
            self.client.close_socket()
            if self._stop_service is not None:
                self._stop_service()

    def runtime_metadata(self) -> dict[str, object]:
        if self.client.handshake is None:
            return {}
        return {
            "remote_policy": {
                **{key: value for key, value in self.client.handshake.items() if key != "capabilities"},
                "health_at_initialize": self._service_health,
                "transport": dict(self._transport_metadata),
            }
        }
