"""Concrete synchronous benchmark adapter for pinned LIBERO."""

from collections.abc import Mapping

import numpy as np

from ovlab_benchmarks.adapter import BenchmarkAdapter
from ovlab_benchmarks.contracts import BenchmarkActionRequest, BenchmarkResetResult, BenchmarkStepResult, TaskDescriptor
from ovlab_core.contracts import (
    OVLAB_CONTRACT_VERSION,
    BenchmarkCapabilities,
    EpisodeContext,
    ExecutedAction,
    InstructionSource,
    RunContext,
    StepId,
)

from .actions import libero_action_spec, settling_action, validate_action, validate_runtime_action_spec
from .environment import PinnedLiberoBackend, native_action_spec
from .errors import LiberoConfigurationError, LiberoEnvironmentError
from .observations import map_observation, observation_spec
from .settings import LiberoAdapterSettings
from .signals import map_signals, signal_registry
from .tasks import LIBERO_PINNED_COMMIT, NativeTaskRecord, resolve_native_suite, select_initial_state_index


class LiberoBenchmarkAdapter(BenchmarkAdapter):
    def __init__(self, settings: LiberoAdapterSettings | None = None, *, backend=None) -> None:
        super().__init__()
        self.settings = settings or LiberoAdapterSettings()
        self._backend = backend
        self._tasks: tuple[NativeTaskRecord, ...] = ()
        self._tasks_by_id: dict[object, NativeTaskRecord] = {}
        self._environment = None
        self._native_observation: Mapping[str, object] | None = None
        self._native_step_index = 0
        self._initial_state_index = 0

    def _initialize(self, run_context: RunContext) -> BenchmarkCapabilities:
        backend = self._backend or PinnedLiberoBackend()
        records = []
        for suite_name in self.settings.suite_names:
            resolve_native_suite(suite_name)
            suite_records = tuple(backend.discover_suite(suite_name))
            selected = self.settings.task_indices
            if selected is not None:
                invalid = [index for index in selected if index >= len(suite_records)]
                if invalid:
                    raise LiberoConfigurationError(
                        f"task indices {invalid} are outside suite {suite_name!r} with {len(suite_records)} tasks"
                    )
                suite_records = tuple(suite_records[index] for index in selected)
            records.extend(suite_records)
        self._backend = backend
        self._tasks = tuple(records)
        self._tasks_by_id = {record.task_id: record for record in self._tasks}
        action_spec = libero_action_spec()
        registry = signal_registry(self.settings.observation_profile)
        return BenchmarkCapabilities(
            "libero-benchmark-adapter",
            "0.1.0",
            OVLAB_CONTRACT_VERSION,
            observation_spec(self.settings),
            action_spec,
            registry,
            True,
            False,
            any(spec.access.value == "privileged" for spec in registry),
            self.settings.suite_names,
            {
                "libero_commit": LIBERO_PINNED_COMMIT,
                "controller": "OSC_POSE",
                "observation_profile": self.settings.observation_profile.value,
            },
        )

    def _list_tasks(self) -> tuple[TaskDescriptor, ...]:
        return tuple(record.descriptor(self.settings.maximum_episode_steps) for record in self._tasks)

    def _reset_episode(self, episode_context: EpisodeContext) -> BenchmarkResetResult:
        task = self._tasks_by_id[episode_context.task_id]
        if episode_context.initial_instruction.text != task.language:
            raise LiberoConfigurationError("episode instruction must equal the authoritative LIBERO task language")
        if episode_context.initial_instruction.source is not InstructionSource.BENCHMARK:
            raise LiberoConfigurationError("LIBERO episode instruction source must be BENCHMARK")
        self._release_environment()
        assert self._backend is not None
        try:
            environment = self._backend.create_environment(task, self.settings)
            validate_runtime_action_spec(native_action_spec(environment), self.capabilities.action_spec)
            environment.reset()
            state_index = select_initial_state_index(
                episode_context,
                len(task.initial_states),
                self.settings.initial_state_selection,
                self.settings.base_seed,
            )
            raw = environment.set_init_state(task.initial_states[state_index])
            for _ in range(self.settings.initialization_settling_steps):
                raw, _, _, _ = environment.step(settling_action())
        except Exception:
            if "environment" in locals():
                try:
                    environment.close()
                except Exception:
                    pass
            raise
        self._environment = environment
        self._native_observation = raw
        self._native_step_index = 0
        self._initial_state_index = state_index
        step_id = StepId(f"{episode_context.episode_id}-step-0")
        timestamp = episode_context.initial_instruction.timestamp_ns
        observation = map_observation(raw, self.settings, step_id, episode_context.initial_instruction, timestamp)
        success = bool(environment.check_success())
        signals = map_signals(
            raw,
            self.settings.observation_profile,
            step_id,
            timestamp,
            reward=0.0,
            success=success,
            terminated=False,
            truncated=False,
            native_step_index=0,
            initial_state_index=state_index,
        )
        return BenchmarkResetResult(
            episode_context,
            observation,
            signals,
            timestamp,
            {
                "suite": task.suite_name,
                "task_id": str(task.task_id),
                "task_index": task.task_index,
                "rollout_index": episode_context.rollout_index,
                "episode_seed": episode_context.seed,
                "initial_state_index": state_index,
                "settling_steps": self.settings.initialization_settling_steps,
                "settling_action": settling_action().tolist(),
                "native_task_reference": f"{task.problem_folder}/{task.bddl_file}",
                "libero_commit": LIBERO_PINNED_COMMIT,
            },
        )

    def _step(self, request: BenchmarkActionRequest) -> BenchmarkStepResult:
        if self._environment is None:
            raise LiberoEnvironmentError("LIBERO environment is not active")
        action = validate_action(request.requested_action, self.capabilities.action_spec)
        try:
            raw, reward, native_done, _ = self._environment.step(action)
            success = bool(self._environment.check_success())
        except Exception as exc:
            raise LiberoEnvironmentError("native LIBERO step failed") from exc
        self._native_step_index += 1
        terminated = bool(native_done) or success
        truncated = self._native_step_index >= self.settings.maximum_episode_steps and not terminated
        executed = ExecutedAction(
            request.prediction_id,
            request.step_context.step_id,
            request.chunk_index,
            action,
            action,
            request.timestamp_ns,
            metadata={"native_command_semantics": "normalized OSC_POSE input"},
        )
        signals = map_signals(
            raw,
            self.settings.observation_profile,
            request.step_context.step_id,
            request.timestamp_ns,
            reward=float(reward),
            success=success,
            terminated=terminated,
            truncated=truncated,
            native_step_index=self._native_step_index,
            initial_state_index=self._initial_state_index,
        )
        next_observation = None
        if not terminated and not truncated:
            assert self._episode_context is not None
            next_step_id = StepId(f"{self._episode_context.episode_id}-step-{self._native_step_index}")
            next_observation = map_observation(
                raw,
                self.settings,
                next_step_id,
                self._episode_context.initial_instruction,
                request.timestamp_ns,
            )
        self._native_observation = raw
        if terminated or truncated:
            self._release_environment()
        return BenchmarkStepResult(
            request.step_context,
            executed,
            next_observation,
            signals,
            float(reward),
            terminated,
            truncated,
            success,
            request.timestamp_ns,
        )

    def _release_environment(self) -> None:
        environment, self._environment = self._environment, None
        if environment is not None:
            try:
                environment.close()
            except Exception as exc:
                raise LiberoEnvironmentError("failed to close native LIBERO environment") from exc

    def _close(self) -> None:
        self._release_environment()

    def _abort_episode(self) -> None:
        self._release_environment()
