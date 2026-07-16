"""Stable public API for benchmark adapters."""

from .adapter import BenchmarkAdapter
from .contracts import BenchmarkActionRequest, BenchmarkResetResult, BenchmarkStepResult, TaskDescriptor
from .errors import BenchmarkAdapterError, BenchmarkLifecycleError, BenchmarkStepError

__all__ = [
    "BenchmarkActionRequest",
    "BenchmarkAdapter",
    "BenchmarkAdapterError",
    "BenchmarkLifecycleError",
    "BenchmarkResetResult",
    "BenchmarkStepError",
    "BenchmarkStepResult",
    "TaskDescriptor",
]
