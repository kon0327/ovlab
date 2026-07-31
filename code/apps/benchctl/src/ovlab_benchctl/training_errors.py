"""Typed Gate I dataset, training, and checkpoint lifecycle failures."""


class DatasetError(RuntimeError):
    """Base error for immutable dataset operations."""


class DatasetRequestError(DatasetError):
    """A dataset request is invalid or unsupported."""


class DatasetUnavailableError(DatasetError):
    """A requested immutable dataset is not locally ready."""


class DatasetIntegrityError(DatasetError):
    """Dataset bytes or metadata failed verification."""


class DatasetSecurityError(DatasetError):
    """Acquisition input would violate the dataset security boundary."""


class DatasetInterruptedError(DatasetError):
    """Dataset acquisition was interrupted before publication."""


class TrainingProfileError(ValueError):
    """A training profile violates the versioned scientific schema."""


class TrainingPlanError(RuntimeError):
    """A profile cannot be resolved to an executable immutable plan."""


class TrainingResourceError(TrainingPlanError):
    """A resolved plan exceeds or conflicts with available resources."""


class TrainingRuntimeError(RuntimeError):
    """The isolated trainer failed after planning."""


class TrainingInterruptedError(TrainingRuntimeError):
    """The isolated trainer was interrupted."""


class CheckpointBundleError(RuntimeError):
    """A finalized training checkpoint is missing, incompatible, or corrupt."""
