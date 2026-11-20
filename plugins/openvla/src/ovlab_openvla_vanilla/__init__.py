"""Public exports loaded on demand, keeping plugin discovery dependency-light."""
from importlib import import_module

_EXPORTS = {'OpenVlaVanillaAdapter': ('.adapter', 'OpenVlaVanillaAdapter'),
 'OpenVlaActionDecodeError': ('.errors', 'OpenVlaActionDecodeError'),
 'OpenVlaCheckpointError': ('.errors', 'OpenVlaCheckpointError'),
 'OpenVlaDependencyError': ('.errors', 'OpenVlaDependencyError'),
 'OpenVlaInferenceError': ('.errors', 'OpenVlaInferenceError'),
 'OpenVlaLoadError': ('.errors', 'OpenVlaLoadError'),
 'OpenVlaPreprocessingError': ('.errors', 'OpenVlaPreprocessingError'),
 'OpenVlaVanillaError': ('.errors', 'OpenVlaVanillaError'),
 'HuggingFaceOpenVlaRuntime': ('.runtime', 'HuggingFaceOpenVlaRuntime'),
 'OpenVlaRuntime': ('.runtime', 'OpenVlaRuntime'),
 'RuntimePrediction': ('.runtime', 'RuntimePrediction'),
 'InferenceSynchronization': ('.settings', 'InferenceSynchronization'),
 'ModelDType': ('.settings', 'ModelDType'),
 'ModelQuantization': ('.settings', 'ModelQuantization'),
 'OpenVlaVanillaSettings': ('.settings', 'OpenVlaVanillaSettings')}

__all__ = [
    "HuggingFaceOpenVlaRuntime", "InferenceSynchronization", "ModelDType", "ModelQuantization",
    "OpenVlaActionDecodeError", "OpenVlaCheckpointError", "OpenVlaDependencyError",
    "OpenVlaInferenceError", "OpenVlaLoadError", "OpenVlaPreprocessingError",
    "OpenVlaRuntime", "OpenVlaVanillaAdapter", "OpenVlaVanillaError",
    "OpenVlaVanillaSettings", "RuntimePrediction",
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
