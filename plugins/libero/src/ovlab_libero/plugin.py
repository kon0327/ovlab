"""LIBERO-owned control hooks; native libraries load only in the worker."""
from importlib.resources import files
import json

from ovlab_benchmarks import (
    BenchmarkPlugin, BenchmarkPluginValidationError, BenchmarkPluginCompatibilityError,
    BenchmarkDeploymentSpec, BenchmarkRuntimeMount, ConfigReference,
)
from .renderer import resolve_renderer_settings
LIBERO_PINNED_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
from .validation import validate_benchmark, validate_execution_profile


def catalog_data():
    document = json.loads(files('ovlab_libero').joinpath('task_catalog.json').read_text())
    if document['upstream_commit'] != LIBERO_PINNED_COMMIT:
        raise BenchmarkPluginCompatibilityError('LIBERO task catalog differs from its pinned upstream')
    return document


class LiberoPlugin(BenchmarkPlugin):
    def validate_benchmark(self, document, path):
        validate_benchmark(document, path)
        if not document['settings']['privileged_signals']['enabled']:
            raise BenchmarkPluginCompatibilityError('LIBERO always exposes its declared privileged signal registry')

    def validate_execution_profile(self, document, path):
        validate_execution_profile(document, path)

    def validate_local_execution(self, overrides, path):
        renderer = overrides.get('renderer', {})
        if set(renderer) - {'device_id'}:
            raise BenchmarkPluginValidationError(f'{path}: unknown renderer override')
        if 'device_id' in renderer and (type(renderer['device_id']) is not int or renderer['device_id'] < 0):
            raise BenchmarkPluginValidationError(f'{path}: device_id must be non-negative')

    def resolve_settings(self, document, context):
        from .settings import LiberoAdapterSettings, LiberoObservationProfile, InitialStateSelection
        self.validate_benchmark(document, 'benchmark')
        settings = document['settings']; obs = settings['observation']
        renderer_settings = self._renderer(context.execution.get('profile', {}),
            context.execution.get('local', {}), context.devices, context.environment,
            context.execution.get('deployment', {}))
        native_to_suite = {item['native_suite']: item['suite'] for item in catalog_data()['tasks']}
        indices = None if settings['task_indices'] == 'all' else tuple(settings['task_indices'])
        cameras = tuple(item['native_name'] for item in obs['cameras'].values())
        return LiberoAdapterSettings(
            suite_names=(native_to_suite[settings['suite']],), task_indices=indices,
            camera_names=cameras, camera_width=obs['width'], camera_height=obs['height'],
            observation_profile=LiberoObservationProfile(obs['profile']),
            maximum_episode_steps=context.maximum_episode_steps,
            initialization_settling_steps=settings['initialization']['settling_steps'],
            initial_state_selection=InitialStateSelection(settings['initialization']['state_selection']),
            base_seed=context.base_seed,
            renderer=renderer_settings,
        )

    def _renderer(self, execution, local, devices, environment, deployment):
        renderer = execution.get('execution', {}).get('libero', {}).get('renderer', {})
        backend = renderer.get('backend', deployment.get('renderer', 'egl'))
        local = local.get('libero', {})
        self.validate_local_execution(local, 'local execution')
        device_id = local.get('renderer', {}).get('device_id', renderer.get('device_id'))
        if device_id is None and backend == 'egl':
            device = devices.get('primary_gpu')
            if device is not None:
                if not device.startswith('cuda:') or not device.removeprefix('cuda:').isdigit():
                    raise BenchmarkPluginCompatibilityError('LIBERO EGL rendering requires primary_gpu: cuda:<index>')
                device_id = int(device.split(':')[1])
        return resolve_renderer_settings(backend, device_id, environment)

    def resolve_deployment(self, document, *, execution_profile, local_execution, devices, environment, deployment):
        self.validate_benchmark(document, 'benchmark')
        from types import SimpleNamespace
        renderer = self._renderer(execution_profile, local_execution, devices, environment, deployment)
        return self.deployment_spec(SimpleNamespace(renderer=renderer), SimpleNamespace(environment=environment))

    def identity(self, settings):
        from ovlab_core.contracts import BenchmarkIdentity
        return BenchmarkIdentity('libero', 'libero', '0.2.0', LIBERO_PINNED_COMMIT, {'family': 'LIBERO'})

    def execution_metadata(self, settings):
        return {"renderer": settings.renderer.as_dict()}

    def method_claims(self, settings):
        data = catalog_data()['tasks']
        selected = [item for item in data if item['suite'] in settings.suite_names]
        suites = tuple(dict.fromkeys(item['native_suite'] for item in selected))
        return {'libero/protocol@1': {'suites': suites, 'observation_profile': settings.observation_profile.value,
                                    'upstream_commit': LIBERO_PINNED_COMMIT}}

    def task_catalog(self, settings):
        from ovlab_benchmarks import TaskCatalog, TaskDescriptor
        from ovlab_core.contracts import TaskId
        return TaskCatalog('libero-tasks-v1', LIBERO_PINNED_COMMIT, tuple(
            TaskDescriptor(item['suite'], TaskId(item['id']), item['name'], item['index'],
                item['instruction'], settings.maximum_episode_steps,
                {'native_suite': item['native_suite'], 'native_task_reference': item['bddl'],
                 'available_initial_state_count': item['initial_state_count'],
                 'state_dimension': item['state_dimension'], 'libero_commit': LIBERO_PINNED_COMMIT})
            for item in catalog_data()['tasks'] if item['suite'] in settings.suite_names))

    def select_tasks(self, settings, catalog):
        if settings.task_indices is None:
            return catalog.select()
        tasks = [task.task_id for suite in settings.suite_names for index in settings.task_indices
                 for task in catalog.tasks if task.suite_name == suite and task.task_index == index]
        if len(tasks) != len(settings.suite_names) * len(settings.task_indices):
            raise BenchmarkPluginValidationError('task indices are outside the selected suite')
        return catalog.select(tuple(tasks))

    def describe_capabilities(self, settings):
        from .adapter import configured_capabilities
        selected = self.select_tasks(settings, self.task_catalog(settings))
        dimensions = {task.metadata['state_dimension'] for task in selected.tasks}
        dimension = next(iter(dimensions)) if len(dimensions) == 1 else None
        return configured_capabilities(settings, simulator_state_dimension=dimension)

    def bundle_references(self, document, execution_profile=None):
        return (ConfigReference(document['settings']['action']['interface_ref'], 'action_interface'),)

    def deployment_spec(self, settings, context):
        environment = {'MUJOCO_GL': settings.renderer.resolved_backend.value}
        mounts = []
        if settings.renderer.device_id is not None:
            environment['MUJOCO_EGL_DEVICE_ID'] = str(settings.renderer.device_id)
        if settings.renderer.resolved_backend.value == 'glfw':
            environment['DISPLAY'] = context.environment.get('DISPLAY', ':0')
            mounts.append(BenchmarkRuntimeMount('/tmp/.X11-unix', '/tmp/.X11-unix'))
        return BenchmarkDeploymentSpec('benchmark', 'ovlab-benchmark-libero:local', 'OVLAB_BENCHMARK_IMAGE',
                                       True, environment, tuple(mounts))

    def create_adapter(self, settings):
        from .adapter import LiberoBenchmarkAdapter
        return LiberoBenchmarkAdapter(settings)


plugin = LiberoPlugin()
