"""Stable public API for synchronous in-process OVLAB execution."""

from .artifacts import FilesystemRunArtifactStore, InMemoryRunArtifactStore, RunArtifactStore, TraceCodec
from .connection import ConnectionReport
from .configuration import RunConfigurationSnapshot
from .errors import (
    ArtifactError, ConnectionError, ExperimentExecutionError, RecorderError,
    RunnerError, RunnerLifecycleError,
)
from .lifecycle import DeterministicClock, RecorderState, RunnerState, SystemClock
from .inspection import RunIntegrityError, inspect_run, verify_run
from .offline_metrics import MetricRecomputationError, recompute_run_metrics
from .reporting import regenerate_report
from .plan import (
    ActionExecutionMode, ActionExecutionPolicy, ArtifactStoreSettings, EpisodeErrorPolicy,
    ExperimentPlan, MetricAvailabilityPolicy, TraceRecordingPolicy,
)
from .provenance import ProvenanceSnapshot, StaticProvenanceProvider
from .recorder import EpisodeRecorder
from .runner import ExperimentRunner

__all__ = [
    "ActionExecutionMode", "ActionExecutionPolicy", "ArtifactError", "ArtifactStoreSettings",
    "ConnectionError", "ConnectionReport", "DeterministicClock", "EpisodeErrorPolicy",
    "EpisodeRecorder", "ExperimentExecutionError", "ExperimentPlan", "ExperimentRunner",
    "FilesystemRunArtifactStore", "InMemoryRunArtifactStore", "MetricAvailabilityPolicy",
    "MetricRecomputationError", "ProvenanceSnapshot", "RecorderError", "RecorderState", "RunArtifactStore", "RunnerError",
    "RunConfigurationSnapshot", "RunIntegrityError", "RunnerLifecycleError", "RunnerState", "StaticProvenanceProvider", "SystemClock", "TraceCodec",
    "TraceRecordingPolicy",
    "inspect_run", "recompute_run_metrics", "regenerate_report", "verify_run",
]
