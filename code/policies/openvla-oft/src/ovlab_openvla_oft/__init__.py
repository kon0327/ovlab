"""Public exports loaded on demand, keeping plugin discovery dependency-light."""
from importlib import import_module

_EXPORTS = {'OpenVlaOftAdapter': ('.adapter', 'OpenVlaOftAdapter'),
 'OftFileIdentity': ('.artifact', 'OftFileIdentity'),
 'OpenVlaOftArtifact': ('.artifact', 'OpenVlaOftArtifact'),
 'validate_oft_method': ('.artifact', 'validate_oft_method'),
 'OpenVlaOftRuntime': ('.runtime', 'OpenVlaOftRuntime'),
 'OpenVlaOftSettings': ('.settings', 'OpenVlaOftSettings'),
 'ModelQuantization': ('ovlab_openvla_common', 'ModelQuantization')}

__all__ = [
    "OftFileIdentity", "OpenVlaOftAdapter", "OpenVlaOftArtifact", "OpenVlaOftRuntime",
    "ModelQuantization", "OpenVlaOftSettings", "validate_oft_method",
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
