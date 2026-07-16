"""Deterministic policy adapter used only by tests."""

import numpy as np

from ovlab_core.contracts import (
    OVLAB_CONTRACT_VERSION,
    ActionPrediction,
    EpisodeContext,
    PolicyCapabilities,
    PolicyObservation,
    PredictionId,
    RawPolicyOutput,
    RunContext,
)
from ovlab_policy_sdk import PolicyAdapter

from .mock_specs import mock_action_spec, mock_observation_requirements


class MockPolicy(PolicyAdapter):
    def __init__(
        self,
        *,
        horizon: int = 1,
        capabilities_override: PolicyCapabilities | None = None,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.capabilities_override = capabilities_override
        self._seed = 0
        self._prediction_index = 0

    def _initialize(self, run_context: RunContext) -> PolicyCapabilities:
        if self.capabilities_override is not None:
            return self.capabilities_override
        return PolicyCapabilities(
            component_name="mock-policy",
            component_version="1.0.0",
            contract_version=OVLAB_CONTRACT_VERSION,
            observation_requirements=mock_observation_requirements(),
            output_action_spec=mock_action_spec(),
            supports_single_action=self.horizon == 1,
            supports_action_chunks=self.horizon > 1,
            minimum_action_horizon=self.horizon,
            maximum_action_horizon=self.horizon,
            supports_dynamic_instructions=False,
            supports_deterministic_reset=True,
            exposes_raw_policy_output=True,
        )

    def _reset_episode(self, episode_context: EpisodeContext) -> None:
        self._seed = episode_context.seed
        self._prediction_index = 0

    def _predict(self, observation: PolicyObservation) -> ActionPrediction:
        prediction_id = PredictionId(f"prediction-{self._seed}-{self._prediction_index}")
        base = np.float32((self._seed % 10 + self._prediction_index) / 100.0)
        actions = np.empty((self.horizon, 3), dtype=np.float32)
        for chunk_index in range(self.horizon):
            actions[chunk_index] = (base + chunk_index / 100.0, base, -base)
        raw = RawPolicyOutput(prediction_id, {"token_ids": [self._prediction_index]}, observation.timestamp_ns)
        prediction = ActionPrediction(
            prediction_id,
            observation.step_id,
            actions,
            self.capabilities.output_action_spec,
            observation.timestamp_ns + 1,
            1,
            self.horizon,
            raw_output=raw,
        )
        self._prediction_index += 1
        return prediction

    def _close(self) -> None:
        pass
