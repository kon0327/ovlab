"""Abstract synchronous policy adapter lifecycle."""

from abc import ABC, abstractmethod

import numpy as np

from ovlab_core.contracts import (
    ActionPrediction,
    ActionSpec,
    AdapterState,
    EpisodeContext,
    PolicyCapabilities,
    PolicyObservation,
    RunContext,
)

from .errors import PolicyInferenceError, PolicyLifecycleError


def _action_specs_equal(first: ActionSpec, second: ActionSpec) -> bool:
    fields = (
        "dimension",
        "representation",
        "translation_indices",
        "rotation_indices",
        "gripper_indices",
        "rotation_representation",
        "gripper_convention",
        "units",
        "dtype",
    )
    if any(getattr(first, name) != getattr(second, name) for name in fields):
        return False
    for name in ("minimum", "maximum"):
        left, right = getattr(first, name), getattr(second, name)
        if (left is None) != (right is None):
            return False
        if left is not None and not np.array_equal(left, right):
            return False
    return True


class PolicyAdapter(ABC):
    """Stateful policy boundary; benchmark termination is signaled by end_episode()."""

    def __init__(self) -> None:
        self._state = AdapterState.CREATED
        self._capabilities: PolicyCapabilities | None = None
        self._run_context: RunContext | None = None
        self._episode_context: EpisodeContext | None = None

    @property
    def state(self) -> AdapterState:
        return self._state

    @property
    def capabilities(self) -> PolicyCapabilities:
        if self._capabilities is None:
            raise PolicyLifecycleError("capabilities", self._state, "adapter is not initialized")
        return self._capabilities

    def initialize(self, run_context: RunContext) -> PolicyCapabilities:
        self._require_state("initialize", AdapterState.CREATED)
        if not isinstance(run_context, RunContext):
            raise TypeError("run_context must be a RunContext")
        capabilities = self._initialize(run_context)
        if not isinstance(capabilities, PolicyCapabilities):
            raise TypeError("_initialize() must return PolicyCapabilities")
        self._run_context = run_context
        self._capabilities = capabilities
        self._state = AdapterState.READY
        return capabilities

    def reset_episode(self, episode_context: EpisodeContext) -> None:
        self._require_state("reset_episode", AdapterState.READY)
        self._validate_episode_context("reset_episode", episode_context)
        self._reset_episode(episode_context)
        self._episode_context = episode_context
        self._state = AdapterState.EPISODE_ACTIVE

    def predict(self, observation: PolicyObservation) -> ActionPrediction:
        self._require_state("predict", AdapterState.EPISODE_ACTIVE)
        if not isinstance(observation, PolicyObservation):
            raise TypeError("observation must be a PolicyObservation")
        prediction = self._predict(observation)
        if not isinstance(prediction, ActionPrediction):
            raise PolicyInferenceError("_predict() must return ActionPrediction")
        capabilities = self.capabilities
        if not _action_specs_equal(prediction.action_spec, capabilities.output_action_spec):
            raise PolicyInferenceError("prediction action specification differs from declared capabilities")
        if not capabilities.minimum_action_horizon <= prediction.horizon <= capabilities.maximum_action_horizon:
            raise PolicyInferenceError("prediction horizon is outside declared capability limits")
        if prediction.horizon == 1 and not capabilities.supports_single_action:
            raise PolicyInferenceError("single-action output was not declared")
        if prediction.horizon > 1 and not capabilities.supports_action_chunks:
            raise PolicyInferenceError("action-chunk output was not declared")
        return prediction

    def end_episode(self, episode_context: EpisodeContext) -> None:
        self._require_state("end_episode", AdapterState.EPISODE_ACTIVE)
        if episode_context != self._episode_context:
            raise PolicyLifecycleError("end_episode", self._state, "episode_context does not match active episode")
        self._end_episode(episode_context)
        self._episode_context = None
        self._state = AdapterState.READY

    def close(self) -> None:
        if self._state is AdapterState.CLOSED:
            return
        self._close()
        self._episode_context = None
        self._state = AdapterState.CLOSED

    def _require_state(self, operation: str, *states: AdapterState) -> None:
        if self._state not in states:
            raise PolicyLifecycleError(operation, self._state)

    def _validate_episode_context(self, operation: str, context: EpisodeContext) -> None:
        if not isinstance(context, EpisodeContext):
            raise TypeError("episode_context must be an EpisodeContext")
        assert self._run_context is not None
        if context.run_id != self._run_context.run_id:
            raise PolicyLifecycleError(operation, self._state, "run_id does not match initialized run")

    @abstractmethod
    def _initialize(self, run_context: RunContext) -> PolicyCapabilities: ...

    @abstractmethod
    def _reset_episode(self, episode_context: EpisodeContext) -> None: ...

    @abstractmethod
    def _predict(self, observation: PolicyObservation) -> ActionPrediction: ...

    def _end_episode(self, episode_context: EpisodeContext) -> None:
        """Optional hook for clearing implementation-specific episode state."""

    @abstractmethod
    def _close(self) -> None: ...
