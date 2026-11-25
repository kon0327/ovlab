"""Dependency-free dataset lifecycle errors shared by providers and callers."""
class DatasetError(RuntimeError):
    """Base error for immutable dataset operations."""


class DatasetRequestError(DatasetError):
    """The requested source or transformation is invalid."""


class DatasetUnavailableError(DatasetError):
    """Required immutable data is unavailable."""


class DatasetIntegrityError(DatasetError):
    """Dataset bytes or metadata failed verification."""


class DatasetSecurityError(DatasetError):
    """The acquisition input violates the dataset boundary."""


class DatasetInterruptedError(DatasetError):
    """Acquisition stopped before publication."""


