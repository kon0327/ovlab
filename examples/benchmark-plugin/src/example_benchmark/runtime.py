"""Small deterministic CPU environment implementing the public adapter lifecycle."""
import numpy as np

from ovlab_benchmarks import BenchmarkAdapter, BenchmarkResetResult, BenchmarkStepResult
from ovlab_benchmarks.plugin import BenchmarkPluginCompatibilityError
from ovlab_core.contracts import (
    ExecutedAction, InstructionSource, PolicyObservation, ProprioceptiveObservation,
    SignalAccess, SignalValue, StepId,
)


class PlanarBenchmarkAdapter(BenchmarkAdapter):
    def __init__(self, tasks, capabilities):
        super().__init__()
        self._tasks = tuple(tasks)
        self._declared_capabilities = capabilities
        self._task = None
        self._step_index = 0
        self._position = np.zeros(2, dtype=np.float32)

    def _initialize(self, run_context):
        return self._declared_capabilities

    def _list_tasks(self):
        return self._tasks

    def _observation(self, episode, timestamp):
        return PolicyObservation(StepId(f'{episode.episode_id}-step-{self._step_index}'), timestamp,
            episode.initial_instruction, proprioception=(ProprioceptiveObservation(
                'fixture.joint_position', self._position, timestamp, ('rad', 'rad')),))

    @staticmethod
    def _signals(step_id, timestamp, success):
        return (SignalValue('benchmark.task_success', success, timestamp, 'example-planar', step_id,
                            access=SignalAccess.EVALUATION_ONLY),)

    def _reset_episode(self, episode):
        self._task = next(task for task in self._tasks if task.task_id == episode.task_id)
        if episode.initial_instruction.text != self._task.natural_language_instruction or episode.initial_instruction.source is not InstructionSource.BENCHMARK:
            raise BenchmarkPluginCompatibilityError('fixture instruction must match the authoritative catalog')
        self._step_index = 0
        self._position = np.asarray(np.random.default_rng(episode.seed).normal(0, 0.01, 2), dtype=np.float32)
        timestamp = episode.initial_instruction.timestamp_ns
        observation = self._observation(episode, timestamp)
        return BenchmarkResetResult(episode, observation, self._signals(observation.step_id, timestamp, False),
                                    timestamp, {'episode_seed': episode.seed})

    def _step(self, request):
        action = request.requested_action
        if action.shape != (2,) or action.dtype != np.float32 or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1):
            raise BenchmarkPluginCompatibilityError('fixture expects a finite float32 two-joint action in [-1, 1]')
        self._position = self._position + action
        self._step_index += 1
        success_after = self._task.metadata['success_after_steps']
        success = bool(success_after and self._step_index >= success_after)
        truncated = self._step_index >= self._task.maximum_steps and not success
        executed = ExecutedAction(request.prediction_id, request.step_context.step_id, request.chunk_index,
                                  action, action, request.timestamp_ns)
        observation = self._observation(self._episode_context, request.timestamp_ns)
        return BenchmarkStepResult(request.step_context, executed, observation,
            self._signals(request.step_context.step_id, request.timestamp_ns, success),
            float(success), success, truncated, success, request.timestamp_ns)

    def _close(self):
        self._task = None

    def _abort_episode(self):
        self._task = None

    def runtime_metadata(self):
        return {'execution': {'backend': 'cpu'}}
