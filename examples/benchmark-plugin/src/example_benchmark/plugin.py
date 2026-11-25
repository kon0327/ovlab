"""Stdlib control hooks; no simulator, policy project or orchestration imports."""
from dataclasses import dataclass
from importlib.resources import files
import json

from ovlab_benchmarks.plugin import (
    BenchmarkDeploymentSpec, BenchmarkPlugin, BenchmarkPluginValidationError, ConfigReference,
)


def catalog_data():
    return json.loads(files('example_benchmark').joinpath('task_catalog.json').read_text(encoding='utf-8'))


@dataclass(frozen=True, slots=True)
class Settings:
    maximum_episode_steps: int
    selected_task_ids: tuple[str, ...] | None


class PlanarBenchmarkPlugin(BenchmarkPlugin):
    def validate_benchmark(self, document, path):
        if not isinstance(document, dict) or document.get('type') != 'example_planar':
            raise BenchmarkPluginValidationError(f'{path}: expected example_planar benchmark')
        settings = document.get('settings')
        if not isinstance(settings, dict) or set(settings) != {'task_ids', 'action'}:
            raise BenchmarkPluginValidationError(f'{path}.settings requires task_ids and action')
        selected = settings['task_ids']
        if selected != 'all':
            if not isinstance(selected, list) or not selected or any(not isinstance(item, str) or not item.strip() for item in selected):
                raise BenchmarkPluginValidationError(f'{path}.settings.task_ids must be all or a non-empty ID list')
            if len(set(selected)) != len(selected):
                raise BenchmarkPluginValidationError(f'{path}.settings.task_ids must be unique')
            known = {task['id'] for task in catalog_data()['tasks']}
            if set(selected) - known:
                raise BenchmarkPluginValidationError(f'{path}.settings.task_ids contains unknown opaque IDs')
        action = settings['action']
        if not isinstance(action, dict) or set(action) != {'interface_ref'} or not isinstance(action['interface_ref'], str) or not action['interface_ref']:
            raise BenchmarkPluginValidationError(f'{path}.settings.action requires an interface_ref')

    def resolve_settings(self, document, context):
        self.validate_benchmark(document, 'benchmark')
        self.validate_local_execution(context.execution.get('local', {}).get('example-benchmark', {}), 'local execution')
        selected = document['settings']['task_ids']
        return Settings(context.maximum_episode_steps, None if selected == 'all' else tuple(selected))

    def identity(self, settings):
        from ovlab_core.contracts import BenchmarkIdentity
        return BenchmarkIdentity('example-benchmark', 'example-planar', '1.0.0', 'example-planar-v1',
                                 {'family': 'Planar fixture'})

    def describe_capabilities(self, settings):
        from .contracts import capabilities
        return capabilities()

    def task_catalog(self, settings):
        from ovlab_benchmarks import TaskCatalog, TaskDescriptor
        from ovlab_core.contracts import TaskId
        data = catalog_data()
        return TaskCatalog(data['catalog_id'], data['revision'], tuple(
            TaskDescriptor('Planar fixture', TaskId(task['id']), task['name'], task['index'],
                task['instruction'], settings.maximum_episode_steps,
                {'success_after_steps': task['success_after_steps'], 'suite_id': 'planar-fixture-v1'})
            for task in data['tasks']))

    def select_tasks(self, settings, catalog):
        from ovlab_core.contracts import TaskId
        return catalog.select(None if settings.selected_task_ids is None else tuple(map(TaskId, settings.selected_task_ids)))

    def bundle_references(self, document, execution_profile=None):
        return (ConfigReference(document['settings']['action']['interface_ref'], 'action_interface'),)

    def resolve_deployment(self, document, *, execution_profile, local_execution, devices, environment, deployment):
        self.validate_benchmark(document, 'benchmark')
        if execution_profile:
            raise BenchmarkPluginValidationError('the CPU fixture has no execution profiles')
        self.validate_local_execution(local_execution.get('example-benchmark', {}), 'local execution')
        return BenchmarkDeploymentSpec('benchmark-example', 'ovlab-benchmark-example:local',
                                       'OVLAB_EXAMPLE_BENCHMARK_IMAGE', requires_gpu=False)

    def execution_metadata(self, settings):
        return {'backend': 'cpu'}

    def create_adapter(self, settings):
        from .runtime import PlanarBenchmarkAdapter
        selection = self.select_tasks(settings, self.task_catalog(settings))
        return PlanarBenchmarkAdapter(selection.tasks, self.describe_capabilities(settings))


plugin = PlanarBenchmarkPlugin()
