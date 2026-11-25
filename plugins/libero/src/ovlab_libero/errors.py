"""LIBERO integration errors with preserved native causes."""

from ovlab_benchmarks.errors import BenchmarkAdapterError, BenchmarkStepError


class LiberoAdapterError(BenchmarkAdapterError):
    """Base error for the concrete LIBERO integration."""


class LiberoDependencyError(LiberoAdapterError):
    """The pinned LIBERO runtime is unavailable or incompatible."""


class LiberoConfigurationError(LiberoAdapterError):
    """LIBERO adapter settings or task selection are invalid."""


class LiberoObservationError(LiberoAdapterError):
    """A native observation cannot satisfy declared capabilities."""


class LiberoActionError(BenchmarkStepError, LiberoAdapterError):
    """A requested action is incompatible with the native controller."""


class LiberoEnvironmentError(LiberoAdapterError):
    """A native suite or environment operation failed."""
