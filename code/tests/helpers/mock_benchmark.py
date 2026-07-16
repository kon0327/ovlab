"""Deterministic benchmark adapter used only by tests."""

import numpy as np

from ovlab_benchmarks import (
    BenchmarkActionRequest,
    BenchmarkAdapter,
    BenchmarkResetResult,
    BenchmarkStepResult,
    TaskDescriptor,
)
from ovlab_core.contracts import (
    OVLAB_CONTRACT_VERSION,
    BenchmarkCapabilities,
    ColorSpace,
    EpisodeContext,
    ExecutedAction,
    ImageEncoding,
    ImageObservation,
    Instruction,
    PolicyObservation,
    ProprioceptiveObservation,
    RunContext,
    SignalAccess,
    SignalRegistry,
    SignalSpec,
    SignalValue,
    StepId,
    TaskId,
)

from .mock_specs import mock_action_spec, mock_observation_spec


class MockBenchmark(BenchmarkAdapter):
    def __init__(
        self,
        *,
        maximum_steps: int = 3,
        modify_actions: bool = False,
        capabilities_override: BenchmarkCapabilities | None = None,
    ) -> None:
        super().__init__()
        self.maximum_steps = maximum_steps
        self.modify_actions = modify_actions
        self.capabilities_override = capabilities_override
        self._step_index = 0
        self._seed = 0

    def _initialize(self, run_context: RunContext) -> BenchmarkCapabilities:
        if self.capabilities_override is not None:
            return self.capabilities_override
        return BenchmarkCapabilities(
            component_name="mock-benchmark",
            component_version="1.0.0",
            contract_version=OVLAB_CONTRACT_VERSION,
            observation_spec=mock_observation_spec(),
            action_spec=mock_action_spec(),
            signal_registry=SignalRegistry(
                (
                    SignalSpec(
                        "success", "bool", (), "", SignalAccess.EVALUATION_ONLY, "Terminal success"
                    ),
                    SignalSpec(
                        "hidden_target",
                        "float32",
                        (2,),
                        "m",
                        SignalAccess.PRIVILEGED,
                        "Privileged target position",
                    ),
                )
            ),
            supports_seeded_reset=True,
            supports_dynamic_instructions=False,
            supports_privileged_evaluation=True,
            task_suites=("mock-suite",),
        )

    def _list_tasks(self) -> tuple[TaskDescriptor, ...]:
        return (
            TaskDescriptor(
                "mock-suite",
                TaskId("mock-task-0"),
                "deterministic task",
                0,
                "move deterministically",
                self.maximum_steps,
            ),
        )

    def _reset_episode(self, episode_context: EpisodeContext) -> BenchmarkResetResult:
        self._step_index = 0
        self._seed = episode_context.seed
        observation = self._observation(episode_context, 0)
        signals = self._signals(observation.step_id, 0, False)
        return BenchmarkResetResult(episode_context, observation, signals, 0)

    def _step(self, request: BenchmarkActionRequest) -> BenchmarkStepResult:
        expected_step_id = StepId(f"{request.step_context.episode_id}-step-{self._step_index}")
        if request.step_context.step_id != expected_step_id or request.step_context.step_index != self._step_index:
            raise ValueError("mock benchmark received an unexpected step context")
        requested = request.requested_action
        if requested.shape != (self.capabilities.action_spec.dimension,):
            raise ValueError("mock benchmark requires exactly one action vector")
        applied = requested.copy()
        reason = None
        if self.modify_actions:
            applied[0] += np.float32(0.125)
            reason = "mock action modification"
        executed = ExecutedAction(
            request.prediction_id,
            request.step_context.step_id,
            request.chunk_index,
            requested,
            applied,
            request.timestamp_ns,
            reason,
        )
        self._step_index += 1
        terminated = self._step_index >= self.maximum_steps
        next_observation = None if terminated else self._observation_from_step(request.step_context, self._step_index)
        signals = self._signals(request.step_context.step_id, request.timestamp_ns, terminated)
        return BenchmarkStepResult(
            request.step_context,
            executed,
            next_observation,
            signals,
            None,
            terminated,
            False,
            True if terminated else None,
            request.timestamp_ns,
        )

    def _observation(self, episode_context: EpisodeContext, step_index: int) -> PolicyObservation:
        step_id = StepId(f"{episode_context.episode_id}-step-{step_index}")
        return self._make_observation(step_id, episode_context.initial_instruction, step_index)

    def _observation_from_step(self, step_context, step_index: int) -> PolicyObservation:
        assert self._episode_context is not None
        return self._observation(self._episode_context, step_index)

    def _make_observation(self, step_id: StepId, instruction: Instruction, step_index: int) -> PolicyObservation:
        timestamp = step_index * 10
        image_value = (self._seed + step_index) % 256
        image = ImageObservation(
            "front_rgb",
            np.full((4, 4, 3), image_value, dtype=np.uint8),
            timestamp,
            ImageEncoding.RAW,
            ColorSpace.RGB,
            "mock-camera",
        )
        proprioception = ProprioceptiveObservation(
            "robot_state",
            np.array([self._seed, step_index], dtype=np.float32),
            timestamp,
            ("rad", "rad"),
        )
        return PolicyObservation(step_id, timestamp, instruction, (image,), (proprioception,))

    @staticmethod
    def _signals(step_id: StepId, timestamp_ns: int, success: bool) -> tuple[SignalValue, ...]:
        return (
            SignalValue("success", success, timestamp_ns, "mock-benchmark", step_id),
            SignalValue(
                "hidden_target",
                np.array([0.25, 0.75], dtype=np.float32),
                timestamp_ns,
                "mock-benchmark",
                step_id,
            ),
        )

    def _close(self) -> None:
        pass
