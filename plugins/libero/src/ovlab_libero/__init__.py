"""Opt-in LIBERO integration, with lazy simulator and contract imports."""
from importlib import import_module
_EXPORTS = {'InitialStateSelection': 'settings', 'LiberoActionError': 'errors', 'LiberoAdapterError': 'errors', 'LiberoAdapterSettings': 'settings', 'LiberoBenchmarkAdapter': 'adapter', 'configured_capabilities': 'adapter', 'LiberoConfigurationError': 'errors', 'LiberoDependencyError': 'errors', 'LiberoEnvironmentError': 'errors', 'LiberoObservationError': 'errors', 'LiberoObservationProfile': 'settings', 'LiberoRendererBackend': 'renderer', 'LiberoRendererRuntime': 'renderer', 'LiberoRendererSettings': 'renderer', 'resolve_renderer_settings': 'renderer'}
__all__ = sorted(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(name)
    value = getattr(import_module('.'+_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
