"""Stable public API for benchmark adapters."""

from .adapter import BenchmarkAdapter
from .contracts import BenchmarkActionRequest, BenchmarkResetResult, BenchmarkStepResult, TaskDescriptor
from .errors import BenchmarkAdapterError, BenchmarkLifecycleError, BenchmarkStepError
from .settings import BenchmarkSettings

__all__ = [
    "BenchmarkActionRequest",
    "BenchmarkAdapter",
    "BenchmarkAdapterError",
    "BenchmarkLifecycleError",
    "BenchmarkResetResult",
    "BenchmarkStepError",
    "BenchmarkStepResult",
    "BenchmarkSettings",
    "TaskDescriptor",
]
