"""Immutable JSON-compatible metadata and numerical value helpers."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np

from .errors import validation_error

MetadataScalar: TypeAlias = str | int | float | bool | None
MetadataValue: TypeAlias = MetadataScalar | tuple["MetadataValue", ...] | Mapping[str, "MetadataValue"]
Metadata: TypeAlias = Mapping[str, MetadataValue]


def _normalize_value(value: Any, contract: str, field: str) -> MetadataValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise validation_error(contract, field, "floating-point metadata must be finite")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, MetadataValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise validation_error(contract, field, "metadata keys must be strings")
            normalized[key] = _normalize_value(nested, contract, f"{field}.{key}")
        return MappingProxyType(normalized)
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(item, contract, f"{field}[]") for item in value)
    raise validation_error(contract, field, f"unsupported metadata value type: {type(value).__name__}")


def normalize_metadata(value: Mapping[str, Any] | None, contract: str, field: str = "metadata") -> Metadata:
    """Return a recursively immutable JSON-compatible metadata mapping."""
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise validation_error(contract, field, "must be a mapping with string keys")
    normalized = _normalize_value(value, contract, field)
    if not isinstance(normalized, Mapping):
        raise validation_error(contract, field, "must be a mapping")
    return normalized


def immutable_numeric_array(
    value: Any,
    contract: str,
    field: str,
    *,
    ndim: int | tuple[int, ...] | None = None,
    finite: bool = True,
) -> np.ndarray:
    """Create a bytes-backed read-only copy of a numeric NumPy array."""
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise validation_error(contract, field, "object dtype is not supported")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise validation_error(contract, field, "must contain numeric values")
    if ndim is not None:
        expected = (ndim,) if isinstance(ndim, int) else ndim
        if array.ndim not in expected:
            raise validation_error(contract, field, f"must have dimensionality in {expected}, got {array.ndim}")
    if any(size == 0 for size in array.shape):
        raise validation_error(contract, field, "must have a non-empty shape")
    if finite and not np.all(np.isfinite(array)):
        raise validation_error(contract, field, "must contain only finite values")
    contiguous = np.ascontiguousarray(array)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)
    return immutable


def normalize_contract_value(value: Any, contract: str, field: str) -> Any:
    """Normalize transport-safe raw or signal values without accepting arbitrary objects."""
    if isinstance(value, np.ndarray):
        return immutable_numeric_array(value, contract, field)
    if isinstance(value, np.generic):
        return _normalize_value(value.item(), contract, field)
    return _normalize_value(value, contract, field)
