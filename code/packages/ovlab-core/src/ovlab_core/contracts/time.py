"""Explicit monotonic and wall-clock timestamp validation."""

from .errors import validation_error


def validate_timestamp_ns(value: int, contract: str, field: str) -> int:
    """Validate a non-negative integer nanosecond timestamp."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise validation_error(contract, field, "must be a non-negative integer nanosecond timestamp")
    if value < 0:
        raise validation_error(contract, field, "must be non-negative")
    return value


def validate_optional_timestamp_ns(value: int | None, contract: str, field: str) -> int | None:
    """Validate an optional non-negative integer nanosecond timestamp."""
    if value is not None:
        validate_timestamp_ns(value, contract, field)
    return value
