"""Public exports loaded on demand, keeping plugin discovery dependency-light."""
from importlib import import_module

_EXPORTS = {'PolicyPlugin': ('.plugin', 'PolicyPlugin'),
 'PolicyPluginV2': ('.plugin', 'PolicyPluginV2'),
 'PolicyPluginValidationError': ('.plugin', 'PolicyPluginValidationError'),
 'PolicyPluginCompatibilityError': ('.plugin', 'PolicyPluginCompatibilityError'),
 'PolicyPluginReferenceError': ('.plugin', 'PolicyPluginReferenceError'),
 'PolicyCapabilities': ('ovlab_core.contracts', 'PolicyCapabilities'),
 'PolicyAdapter': ('.adapter', 'PolicyAdapter'),
 'PolicyAdapterError': ('.errors', 'PolicyAdapterError'),
 'PolicyInferenceError': ('.errors', 'PolicyInferenceError'),
 'PolicyLifecycleError': ('.errors', 'PolicyLifecycleError')}

__all__ = [
    "PolicyPlugin", "PolicyPluginV2", "PolicyPluginValidationError", "PolicyPluginCompatibilityError", "PolicyPluginReferenceError",
    "PolicyAdapter",
    "PolicyAdapterError",
    "PolicyCapabilities",
    "PolicyInferenceError",
    "PolicyLifecycleError",
]


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
