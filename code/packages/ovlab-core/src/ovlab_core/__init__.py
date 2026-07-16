"""Stable top-level API for OpenVLABenchmark core negotiation contracts."""

from .compatibility import (
    CompatibilityIssue,
    CompatibilityReport,
    CompatibilitySeverity,
    negotiate_capabilities,
)
from .contracts import (
    OVLAB_CONTRACT_VERSION,
    AdapterState,
    BenchmarkCapabilities,
    ImageObservationSpec,
    ObservationRequirements,
    ObservationSpec,
    PolicyCapabilities,
    ProprioceptiveObservationSpec,
)

__all__ = [
    "OVLAB_CONTRACT_VERSION",
    "AdapterState",
    "BenchmarkCapabilities",
    "CompatibilityIssue",
    "CompatibilityReport",
    "CompatibilitySeverity",
    "ImageObservationSpec",
    "ObservationRequirements",
    "ObservationSpec",
    "PolicyCapabilities",
    "ProprioceptiveObservationSpec",
    "negotiate_capabilities",
]
