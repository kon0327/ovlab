"""Lazy compatibility boundary around the pinned LIBERO API."""

from pathlib import Path
from typing import Any

from .errors import LiberoConfigurationError, LiberoDependencyError, LiberoEnvironmentError
from .settings import LiberoAdapterSettings
from .tasks import NativeTaskRecord, resolve_native_suite


class PinnedLiberoBackend:
    """The only production module that imports LIBERO."""

    def __init__(self) -> None:
        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except (ImportError, ModuleNotFoundError) as exc:
            raise LiberoDependencyError(
                "LiberoBenchmarkAdapter requires the tested pinned LIBERO installation"
            ) from exc
        self._benchmark = benchmark
        self._get_libero_path = get_libero_path
        self._environment_type = OffScreenRenderEnv

    def discover_suite(self, suite_name: str) -> tuple[NativeTaskRecord, ...]:
        native_name = resolve_native_suite(suite_name)
        try:
            suite_type = self._benchmark.get_benchmark_dict()[native_name]
            suite = suite_type(task_order_index=0)
            records = []
            for index in range(suite.get_num_tasks()):
                task = suite.get_task(index)
                states = tuple(suite.get_task_init_states(index))
                records.append(
                    NativeTaskRecord(
                        suite_name,
                        native_name,
                        index,
                        task.name,
                        task.language,
                        task.problem_folder,
                        task.bddl_file,
                        states,
                        task,
                    )
                )
            return tuple(records)
        except Exception as exc:
            raise LiberoEnvironmentError(f"failed to discover native suite {native_name!r}") from exc

    def create_environment(self, task: NativeTaskRecord, settings: LiberoAdapterSettings) -> Any:
        bddl_path = Path(self._get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        arguments: dict[str, object] = {
            "bddl_file_name": str(bddl_path),
            "camera_names": list(settings.camera_names),
            "camera_heights": settings.camera_height,
            "camera_widths": settings.camera_width,
        }
        if settings.render_gpu_device_id is not None:
            arguments["render_gpu_device_id"] = settings.render_gpu_device_id
        if settings.controller_configuration_override is not None:
            raise LiberoConfigurationError(
                "the pinned OffScreenRenderEnv wrapper fixes OSC_POSE and does not safely expose controller overrides"
            )
        try:
            environment = self._environment_type(**arguments)
            environment.seed(settings.base_seed)
            return environment
        except Exception as exc:
            raise LiberoEnvironmentError(f"failed to create LIBERO environment for {task.task_id}") from exc


def native_action_spec(environment: object) -> object:
    if hasattr(environment, "action_spec"):
        return environment.action_spec
    wrapped = getattr(environment, "env", None)
    if wrapped is not None and hasattr(wrapped, "action_spec"):
        return wrapped.action_spec
    raise LiberoDependencyError("native LIBERO environment does not expose action_spec")
