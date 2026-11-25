"""Public benchmark SDK with dependency-free control imports."""
from importlib import import_module

_EXPORTS = {'BenchmarkAdapter': 'adapter', 'BenchmarkActionRequest': 'contracts', 'BenchmarkResetResult': 'contracts', 'BenchmarkStepResult': 'contracts', 'TaskDescriptor': 'contracts', 'BenchmarkAdapterError': 'errors', 'BenchmarkLifecycleError': 'errors', 'BenchmarkStepError': 'errors', 'BenchmarkSettings': 'settings', 'AssetRequirement': 'catalog', 'CapabilityRefinement': 'catalog', 'TaskCatalog': 'catalog', 'TaskSelection': 'catalog', 'create_benchmark_adapter': 'catalog', 'validate_runtime_capabilities': 'catalog', 'BenchmarkDeploymentSpec': 'plugin', 'BenchmarkPlugin': 'plugin', 'BenchmarkPluginSettings': 'plugin', 'BenchmarkPluginCompatibilityError': 'plugin', 'BenchmarkPluginReferenceError': 'plugin', 'BenchmarkPluginUnavailableError': 'plugin', 'BenchmarkPluginValidationError': 'plugin', 'BenchmarkRuntimeMount': 'plugin', 'ConfigReference': 'plugin'}
__all__ = sorted(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(name)
    value = getattr(import_module('.'+_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
