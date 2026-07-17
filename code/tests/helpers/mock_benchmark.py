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
        task_count: int = 1,
        terminal_outcomes: tuple[str, ...] | None = None,
        capabilities_override: BenchmarkCapabilities | None = None,
    ) -> None:
        super().__init__()
        if type(task_count) is not int or task_count <= 0:
            raise ValueError("task_count must be a positive integer")
        if terminal_outcomes is None:
            terminal_outcomes = ("success",) + ("time_limit",) * (task_count - 1)
        if len(terminal_outcomes) != task_count or any(
            outcome not in {"success", "failure", "time_limit"} for outcome in terminal_outcomes
        ):
            raise ValueError("terminal_outcomes must define success, failure, or time_limit for every task")
        self.maximum_steps = maximum_steps
        self.modify_actions = modify_actions
        self.task_count = task_count
        self.terminal_outcomes = tuple(terminal_outcomes)
        self.capabilities_override = capabilities_override
        self._step_index = 0
        self._seed = 0
        self._terminal_outcome = "success"

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
                        "benchmark.task_success", "bool", (), "", SignalAccess.EVALUATION_ONLY,
                        "Authoritative terminal task success",
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
        return tuple(
            TaskDescriptor(
                "mock-suite",
                TaskId(f"mock-task-{index}"),
                f"deterministic task {index}",
                index,
                f"move deterministically for task {index}",
                self.maximum_steps,
                {"terminal_outcome": self.terminal_outcomes[index]},
            )
            for index in range(self.task_count)
        )

    def _reset_episode(self, episode_context: EpisodeContext) -> BenchmarkResetResult:
        self._step_index = 0
        self._seed = episode_context.seed
        try:
            task_index = int(str(episode_context.task_id).removeprefix("mock-task-"))
            self._terminal_outcome = self.terminal_outcomes[task_index]
        except (ValueError, IndexError) as exc:
            raise ValueError("mock benchmark received an unknown task ID") from exc
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
        terminal = self._step_index >= self.maximum_steps
        truncated = terminal and self._terminal_outcome == "time_limit"
        terminated = terminal and not truncated
        success = terminal and self._terminal_outcome == "success"
        next_observation = None if terminal else self._observation_from_step(request.step_context, self._step_index)
        signals = self._signals(request.step_context.step_id, request.timestamp_ns, success)
        return BenchmarkStepResult(
            request.step_context,
            executed,
            next_observation,
            signals,
            None,
            terminated,
            truncated,
            success if terminal else None,
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
            SignalValue(
                "benchmark.task_success", success, timestamp_ns, "mock-benchmark", step_id,
                access=SignalAccess.EVALUATION_ONLY,
            ),
            SignalValue(
                "hidden_target",
                np.array([0.25, 0.75], dtype=np.float32),
                timestamp_ns,
                "mock-benchmark",
                step_id,
                access=SignalAccess.PRIVILEGED,
            ),
        )

    def _close(self) -> None:
        pass
