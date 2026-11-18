"""A constant action adapter for testing integration, never scientific evaluation."""
import numpy as np
from ovlab_core.contracts import ActionPrediction, PredictionId, PolicyCapabilities, ObservationRequirements, OVLAB_CONTRACT_VERSION
from ovlab_policy_sdk import PolicyAdapter


class ConstantPolicy(PolicyAdapter):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.counter = 0

    def _initialize(self, run_context):
        return PolicyCapabilities('example-policy', '1.0.0', OVLAB_CONTRACT_VERSION,
            ObservationRequirements(), self.settings.action_spec, True, False, 1, 1,
            True, True, False, {'checkpoint': self.settings.checkpoint})

    def _reset_episode(self, episode_context):
        self.counter = 0

    def _predict(self, observation):
        self.counter += 1
        return ActionPrediction(PredictionId(f'constant-{self.counter}'), observation.step_id,
            np.zeros((1, self.settings.action_spec.dimension), dtype=np.float32),
            self.settings.action_spec, observation.timestamp_ns, 0, 1)

    def _close(self):
        pass
