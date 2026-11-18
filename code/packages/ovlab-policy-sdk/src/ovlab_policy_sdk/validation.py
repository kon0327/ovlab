"""Small strict validation helpers available to plugin authors."""
from typing import Any
from .plugin import PolicyPluginValidationError as ConfigSchemaError

def mapping(value: Any, path: str, *, required=(), optional=()) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigSchemaError(f"{path} must be a mapping")
    required, allowed = set(required), set(required) | set(optional)
    missing, unknown = sorted(required - set(value)), sorted(set(value) - allowed)
    if missing: raise ConfigSchemaError(f"{path} is missing required keys: {', '.join(missing)}")
    if unknown: raise ConfigSchemaError(f"{path} contains unknown keys: {', '.join(unknown)}")
    return value


def exact_type(value: Any, expected: type | tuple[type, ...], path: str) -> None:
    types = expected if isinstance(expected, tuple) else (expected,)
    if bool in types:
        valid = type(value) is bool
    elif int in types:
        valid = type(value) is int
    else:
        valid = isinstance(value, types)
    if not valid:
        names = "/".join(item.__name__ for item in types)
        raise ConfigSchemaError(f"{path} must be {names}, got {type(value).__name__}")


def enum(value: Any, choices: tuple[str, ...], path: str) -> None:
    exact_type(value, str, path)
    if value not in choices:
        raise ConfigSchemaError(f"{path} must be one of: {', '.join(choices)}")


def non_empty_string(value: Any, path: str) -> None:
    exact_type(value, str, path)
    if not value.strip(): raise ConfigSchemaError(f"{path} must not be empty")


def number(value: Any, path: str) -> None:
    if type(value) not in (int, float): raise ConfigSchemaError(f"{path} must be a number")
