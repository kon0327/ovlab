"""Abstract synchronous benchmark adapter lifecycle."""

from abc import ABC, abstractmethod

from ovlab_core.contracts import AdapterState, BenchmarkCapabilities, EpisodeContext, RunContext

from .contracts import BenchmarkActionRequest, BenchmarkResetResult, BenchmarkStepResult, TaskDescriptor
from .errors import BenchmarkLifecycleError


class BenchmarkAdapter(ABC):
    def __init__(self) -> None:
        self._state = AdapterState.CREATED
        self._capabilities: BenchmarkCapabilities | None = None
        self._run_context: RunContext | None = None
        self._episode_context: EpisodeContext | None = None

    @property
    def state(self) -> AdapterState:
        return self._state

    @property
    def capabilities(self) -> BenchmarkCapabilities:
        if self._capabilities is None:
            raise BenchmarkLifecycleError("capabilities", self._state, "adapter is not initialized")
        return self._capabilities

    def initialize(self, run_context: RunContext) -> BenchmarkCapabilities:
        self._require_state("initialize", AdapterState.CREATED)
        if not isinstance(run_context, RunContext):
            raise TypeError("run_context must be a RunContext")
        capabilities = self._initialize(run_context)
        if not isinstance(capabilities, BenchmarkCapabilities):
            raise TypeError("_initialize() must return BenchmarkCapabilities")
        self._run_context = run_context
        self._capabilities = capabilities
        self._state = AdapterState.READY
        return capabilities

    def list_tasks(self) -> tuple[TaskDescriptor, ...]:
        self._require_state("list_tasks", AdapterState.READY, AdapterState.EPISODE_ACTIVE)
        tasks = tuple(self._list_tasks())
        if any(not isinstance(task, TaskDescriptor) for task in tasks):
            raise TypeError("_list_tasks() must return TaskDescriptor values")
        return tasks

    def reset_episode(self, episode_context: EpisodeContext) -> BenchmarkResetResult:
        self._require_state("reset_episode", AdapterState.READY)
        self._validate_episode_context("reset_episode", episode_context)
        result = self._reset_episode(episode_context)
        if not isinstance(result, BenchmarkResetResult):
            raise TypeError("_reset_episode() must return BenchmarkResetResult")
        if result.episode_context != episode_context:
            raise BenchmarkLifecycleError("reset_episode", self._state, "result episode_context does not match request")
        self._episode_context = episode_context
        self._state = AdapterState.EPISODE_ACTIVE
        return result

    def step(self, request: BenchmarkActionRequest) -> BenchmarkStepResult:
        self._require_state("step", AdapterState.EPISODE_ACTIVE)
        if not isinstance(request, BenchmarkActionRequest):
            raise TypeError("request must be a BenchmarkActionRequest")
        assert self._episode_context is not None
        expected = (
            self._episode_context.run_id,
            self._episode_context.task_id,
            self._episode_context.episode_id,
        )
        actual = (request.step_context.run_id, request.step_context.task_id, request.step_context.episode_id)
        if actual != expected:
            raise BenchmarkLifecycleError("step", self._state, "request IDs do not match the active episode")
        result = self._step(request)
        if not isinstance(result, BenchmarkStepResult):
            raise TypeError("_step() must return BenchmarkStepResult")
        if result.terminated or result.truncated:
            self._episode_context = None
            self._state = AdapterState.READY
        return result

    def abort_episode(self) -> None:
        """Release an active episode after a runner or policy failure."""
        self._require_state("abort_episode", AdapterState.EPISODE_ACTIVE)
        self._abort_episode()
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
            raise BenchmarkLifecycleError(operation, self._state)

    def _validate_episode_context(self, operation: str, context: EpisodeContext) -> None:
        if not isinstance(context, EpisodeContext):
            raise TypeError("episode_context must be an EpisodeContext")
        assert self._run_context is not None
        if context.run_id != self._run_context.run_id:
            raise BenchmarkLifecycleError(operation, self._state, "run_id does not match initialized run")
        task_ids = {task.task_id for task in self.list_tasks()}
        if context.task_id not in task_ids:
            raise BenchmarkLifecycleError(operation, self._state, "task_id is not declared by this adapter")

    @abstractmethod
    def _initialize(self, run_context: RunContext) -> BenchmarkCapabilities: ...

    @abstractmethod
    def _list_tasks(self) -> tuple[TaskDescriptor, ...]: ...

    @abstractmethod
    def _reset_episode(self, episode_context: EpisodeContext) -> BenchmarkResetResult: ...

    @abstractmethod
    def _step(self, request: BenchmarkActionRequest) -> BenchmarkStepResult: ...

    def _abort_episode(self) -> None:
        """Optional concrete hook for releasing active episode resources."""

    @abstractmethod
    def _close(self) -> None: ...
