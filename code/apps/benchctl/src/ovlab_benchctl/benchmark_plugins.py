"""Typed benchmark discovery and control resolution without simulator imports."""
from dataclasses import dataclass
from pathlib import Path

from ovlab_benchmarks import BenchmarkPlugin
from .errors import ConfigCompatibilityError, ConfigReferenceError, ConfigSchemaError
from .extensions import ExtensionCatalog, ProviderRef


def extension_catalog(root):
    from .policy_plugins import registry_path
    from .plugin_projects import read_document
    root = Path(root).resolve()
    frozen = root / 'configs/extensions/registry.json'
    path = frozen if (root / 'configs/.ovlab-bundle.json').is_file() and frozen.is_file() else registry_path(root)
    document = read_document(path)
    if document.get('schema_version') == 'ovlab.plugins/v1':
        document = {'schema_version': 'ovlab.extensions/v1', 'projects': {
            name: {'source': document['sources'][value['source_id']], 'manifest': value['manifest']}
            for name,value in document.get('projects', {}).items()}}
    return ExtensionCatalog(root, document=document)


@dataclass
class LoadedBenchmark:
    ref: ProviderRef
    descriptor: dict
    implementation: BenchmarkPlugin


def benchmark_for_reference(reference, root):
    catalog = extension_catalog(root)
    provider = catalog.resolve_provider(reference, expected_kind="benchmark")
    implementation = catalog.load_provider(reference, expected_kind="benchmark", contract=BenchmarkPlugin)
    return LoadedBenchmark(provider.ref, provider.descriptor, implementation)


def benchmark_for_document(document, root):
    reference = document.get('provider')
    if not isinstance(reference, str):
        raise ConfigSchemaError('benchmark configuration requires an explicit provider binding')
    catalog = extension_catalog(root)
    provider = catalog.resolve_provider(reference, expected_kind='benchmark')
    if provider.descriptor.get('config_type') != document.get('type'):
        raise ConfigCompatibilityError('benchmark type differs from its selected provider')
    implementation = catalog.load_provider(reference, expected_kind='benchmark', contract=BenchmarkPlugin)
    return LoadedBenchmark(provider.ref, provider.descriptor, implementation)


def call_benchmark(owner, method, *args, **kwargs):
    from ovlab_benchmarks import BenchmarkPluginCompatibilityError, BenchmarkPluginReferenceError, BenchmarkPluginValidationError
    try:
        return getattr(owner.implementation, method)(*args, **kwargs)
    except BenchmarkPluginValidationError as exc:
        raise ConfigSchemaError(str(exc)) from exc
    except BenchmarkPluginCompatibilityError as exc:
        raise ConfigCompatibilityError(str(exc)) from exc
    except BenchmarkPluginReferenceError as exc:
        raise ConfigReferenceError(str(exc)) from exc


def peer_context(owner, settings):
    from ovlab_core.contracts import BenchmarkContext
    plugin = owner.implementation
    catalog = call_benchmark(owner, 'task_catalog', settings)
    selection = call_benchmark(owner, 'select_tasks', settings, catalog)
    claims = plugin.method_claims(settings)
    identity = plugin.identity(settings)
    if identity.plugin_id != owner.ref.project:
        raise ConfigCompatibilityError('benchmark identity differs from its registered project')
    return BenchmarkContext(identity, plugin.describe_capabilities(settings), selection.identity, claims)


def create_adapter(owner, settings):
    from ovlab_benchmarks import BenchmarkAdapter, create_benchmark_adapter, validate_runtime_capabilities
    plugin = owner.implementation
    declared = plugin.describe_capabilities(settings)
    catalog = plugin.task_catalog(settings)
    selected = plugin.select_tasks(settings, catalog)
    delegate = create_benchmark_adapter(plugin, settings)

    class CheckedAdapter(BenchmarkAdapter):
        def _initialize(self, context):
            actual = delegate.initialize(context)
            validate_runtime_capabilities(declared, actual, plugin.capability_refinement(settings))
            return actual
        def _list_tasks(self):
            tasks = delegate.list_tasks()
            expected = {task.task_id: task for task in selected.tasks}
            actual = {task.task_id: task for task in tasks}
            if set(expected) != set(actual):
                raise ConfigCompatibilityError('runtime task catalog differs from its locked selection')
            for key, task in expected.items():
                observed = actual[key]
                if (task.task_name, task.natural_language_instruction, task.suite_name,
                    task.task_index, task.maximum_steps) != (
                    observed.task_name, observed.natural_language_instruction, observed.suite_name,
                    observed.task_index, observed.maximum_steps):
                    raise ConfigCompatibilityError(f'runtime task identity differs from catalog: {key}')
            return tasks
        def _reset_episode(self, context): return delegate.reset_episode(context)
        def _step(self, request): return delegate.step(request)
        def _abort_episode(self): return delegate.abort_episode()
        def _close(self): return delegate.close()
        def runtime_metadata(self):
            metadata = getattr(delegate, 'runtime_metadata', None)
            return metadata() if callable(metadata) else {}
    return CheckedAdapter()
