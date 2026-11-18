"""Public exports loaded on demand, keeping plugin discovery dependency-light."""
from importlib import import_module

_EXPORTS = {'OpenVlaMergedLoraAdapter': ('.adapter', 'OpenVlaMergedLoraAdapter'),
 'method_descriptor_from_registry': ('.settings', 'method_descriptor_from_registry')}

__all__ = ["OpenVlaMergedLoraAdapter", "method_descriptor_from_registry"]


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
