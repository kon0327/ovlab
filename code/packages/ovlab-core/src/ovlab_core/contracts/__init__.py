"""Stable public API for OVLAB's dependency-light data contracts."""

from .actions import (
    ActionPrediction,
    ActionRepresentation,
    ActionSpec,
    ExecutedAction,
    GripperConvention,
    PredictionValidity,
    RawPolicyOutput,
    RotationRepresentation,
)
from .capabilities import (
    AdapterState,
    BenchmarkCapabilities,
    ImageObservationSpec,
    ObservationRequirements,
    ObservationSpec,
    PolicyCapabilities,
    ProprioceptiveObservationSpec,
)
from .errors import ContractCompatibilityError, ContractError, ContractValidationError
from .identifiers import EpisodeId, InstructionId, PredictionId, RunId, StepId, TaskId
from .instruction import Instruction, InstructionSource
from .lifecycle import EpisodeContext, RunContext, StepContext, TaskContext
from .metadata import Metadata, MetadataScalar, MetadataValue, immutable_numeric_array, normalize_metadata
from .observation import (
    ColorSpace,
    ImageEncoding,
    ImageObservation,
    PolicyObservation,
    ProprioceptiveObservation,
)
from .signals import SignalAccess, SignalRegistry, SignalSpec, SignalValue
from .trace import EpisodeTerminalStatus, EpisodeTrace
from .version import OVLAB_CONTRACT_VERSION

__all__ = [
    "OVLAB_CONTRACT_VERSION",
    "AdapterState",
    "ActionPrediction",
    "ActionRepresentation",
    "ActionSpec",
    "BenchmarkCapabilities",
    "ColorSpace",
    "ContractCompatibilityError",
    "ContractError",
    "ContractValidationError",
    "EpisodeContext",
    "EpisodeId",
    "EpisodeTerminalStatus",
    "EpisodeTrace",
    "ExecutedAction",
    "GripperConvention",
    "ImageEncoding",
    "ImageObservation",
    "ImageObservationSpec",
    "Instruction",
    "InstructionId",
    "InstructionSource",
    "Metadata",
    "MetadataScalar",
    "MetadataValue",
    "ObservationRequirements",
    "ObservationSpec",
    "PolicyObservation",
    "PolicyCapabilities",
    "PredictionId",
    "PredictionValidity",
    "ProprioceptiveObservation",
    "ProprioceptiveObservationSpec",
    "RawPolicyOutput",
    "RotationRepresentation",
    "RunContext",
    "RunId",
    "SignalAccess",
    "SignalRegistry",
    "SignalSpec",
    "SignalValue",
    "StepContext",
    "StepId",
    "TaskContext",
    "TaskId",
    "immutable_numeric_array",
    "normalize_metadata",
]
