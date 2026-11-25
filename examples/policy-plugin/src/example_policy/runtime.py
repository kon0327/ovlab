"""A constant action adapter for testing integration, never scientific evaluation."""
import numpy as np
from ovlab_core.contracts import ActionPrediction, PredictionId
from ovlab_policy_sdk import PolicyAdapter


class ConstantPolicy(PolicyAdapter):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.counter = 0

    def _initialize(self, run_context):
        from .plugin import plugin
        return plugin.describe_capabilities(self.settings)

    def _reset_episode(self, episode_context):
        self.counter = 0

    def _predict(self, observation):
        self.counter += 1
        return ActionPrediction(PredictionId(f'constant-{self.counter}'), observation.step_id,
            np.zeros((1, self.settings.action_spec.dimension), dtype=np.float32),
            self.settings.action_spec, observation.timestamp_ns, 0, 1)

    def _close(self):
        pass
