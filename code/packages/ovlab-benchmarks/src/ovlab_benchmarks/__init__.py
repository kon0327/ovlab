"""Stable public API for benchmark adapters."""

from .adapter import BenchmarkAdapter
from .contracts import BenchmarkActionRequest, BenchmarkResetResult, BenchmarkStepResult, TaskDescriptor
from .errors import BenchmarkAdapterError, BenchmarkLifecycleError, BenchmarkStepError
from .settings import BenchmarkSettings
from .plugin import (
    AssetRequirement, BenchmarkDeploymentSpec, BenchmarkPlugin, BenchmarkPluginCompatibilityError,
    BenchmarkPluginReferenceError, BenchmarkPluginSettings, BenchmarkPluginUnavailableError,
    BenchmarkPluginValidationError, BenchmarkRuntimeMount, CapabilityRefinement, ConfigReference,
    TaskCatalog, TaskSelection, create_benchmark_adapter, validate_runtime_capabilities,
)

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
    "AssetRequirement", "BenchmarkDeploymentSpec", "BenchmarkPlugin", "BenchmarkPluginSettings",
    "BenchmarkPluginCompatibilityError", "BenchmarkPluginReferenceError", "BenchmarkPluginUnavailableError",
    "BenchmarkPluginValidationError", "BenchmarkRuntimeMount", "CapabilityRefinement", "ConfigReference",
    "TaskCatalog", "TaskSelection", "create_benchmark_adapter", "validate_runtime_capabilities",
]
