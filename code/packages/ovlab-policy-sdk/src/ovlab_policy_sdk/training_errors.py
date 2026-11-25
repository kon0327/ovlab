"""Typed Gate I dataset, training, and checkpoint lifecycle failures."""


from ovlab_core.dataset_errors import (
    DatasetError, DatasetRequestError, DatasetUnavailableError, DatasetIntegrityError,
    DatasetSecurityError, DatasetInterruptedError,
)


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


class CheckpointResolutionError(RuntimeError):
    """A checkpoint could not be resolved or failed immutable verification."""
