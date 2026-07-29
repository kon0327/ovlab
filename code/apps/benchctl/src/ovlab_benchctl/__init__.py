"""Public OVLAB configuration API with dependency-light lazy exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "ConfigCompatibilityError": (".errors", "ConfigCompatibilityError"),
    "ConfigError": (".errors", "ConfigError"),
    "ConfigReferenceError": (".errors", "ConfigReferenceError"),
    "ConfigResolver": (".resolver", "ConfigResolver"),
    "ConfigSchemaError": (".errors", "ConfigSchemaError"),
    "MetricSetSettings": (".models", "MetricSetSettings"),
    "MockBenchmarkSettings": (".models", "MockBenchmarkSettings"),
    "MockPolicySettings": (".models", "MockPolicySettings"),
    "ProtocolSettings": (".models", "ProtocolSettings"),
    "ResolvedConfigWriteError": (".errors", "ResolvedConfigWriteError"),
    "ResolvedExperimentConfig": (".models", "ResolvedExperimentConfig"),
    "StrictYamlError": (".errors", "StrictYamlError"),
    "dumps": (".strict_yaml", "dumps"),
    "load": (".strict_yaml", "load"),
    "loads": (".strict_yaml", "loads"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *__all__))
