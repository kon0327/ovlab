"""Benchmark adapter exceptions."""

from ovlab_core.contracts import AdapterState


class BenchmarkAdapterError(Exception):
    """Base class for benchmark adapter errors."""


class BenchmarkLifecycleError(BenchmarkAdapterError):
    """Raised when an operation is invalid in the current adapter state."""

    def __init__(self, operation: str, state: AdapterState, reason: str | None = None) -> None:
        message = f"benchmark operation '{operation}' is invalid in state '{state.value}'"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message)
        self.operation = operation
        self.state = state


class BenchmarkStepError(BenchmarkAdapterError):
    """Raised when a benchmark cannot execute a valid step request."""
