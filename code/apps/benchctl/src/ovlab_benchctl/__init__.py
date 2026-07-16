"""Public strict configuration composition API."""

from .errors import (
    ConfigCompatibilityError, ConfigError, ConfigReferenceError, ConfigSchemaError,
    ResolvedConfigWriteError, StrictYamlError,
)
from .models import MetricSetSettings, ProtocolSettings, ResolvedExperimentConfig
from .resolver import ConfigResolver
from .strict_yaml import dumps, load, loads

__all__ = [
    "ConfigCompatibilityError", "ConfigError", "ConfigReferenceError", "ConfigResolver",
    "ConfigSchemaError", "MetricSetSettings", "ProtocolSettings", "ResolvedConfigWriteError",
    "ResolvedExperimentConfig", "StrictYamlError", "dumps", "load", "loads",
]
