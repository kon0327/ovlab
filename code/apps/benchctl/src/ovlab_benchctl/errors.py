"""Configuration parsing, schema, resolution, and persistence errors."""


class ConfigError(ValueError):
    """Base error for strict OVLAB configuration."""


class StrictYamlError(ConfigError):
    """A document is outside the supported strict YAML subset."""


class ConfigSchemaError(ConfigError):
    """A document violates its kind-specific schema."""


class ConfigReferenceError(ConfigError):
    """A component, inheritance, or resource reference is invalid."""


class ConfigCompatibilityError(ConfigError):
    """Individually valid components cannot be composed safely."""


class ResolvedConfigWriteError(ConfigError):
    """The immutable resolved configuration cannot be written."""
