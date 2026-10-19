"""Stable top-level API for OpenVLABenchmark core negotiation contracts.

Import contracts on demand so the host progress renderer stays stdlib-only.
"""

from importlib import import_module


_COMPATIBILITY = {
    "CompatibilityIssue", "CompatibilityReport", "CompatibilitySeverity",
    "negotiate_capabilities",
}
_CONTRACTS = {
    "OVLAB_CONTRACT_VERSION", "OVLAB_VERSION", "AdapterState",
    "BenchmarkCapabilities", "ImageObservationSpec", "ObservationRequirements",
    "ObservationSpec", "PolicyCapabilities", "ProprioceptiveObservationSpec",
}

__all__ = sorted(_COMPATIBILITY | _CONTRACTS)


def __getattr__(name: str):
    if name in _COMPATIBILITY:
        return getattr(import_module(".compatibility", __name__), name)
    if name in _CONTRACTS:
        return getattr(import_module(".contracts", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
