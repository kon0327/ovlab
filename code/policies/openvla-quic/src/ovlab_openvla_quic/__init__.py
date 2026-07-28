"""Contract-only QuIC wrapper API."""

from .adapter import OpenVLAQuICAdapter, OpenVLAQuICPEFTAdapter, OpenVLAQuICWCAdapter
from .config import descriptor_from_document
from .descriptors import (
    EXTERNAL_QUIC_COMMIT, QuICImplementationStatus, QuICMethodDescriptor,
    QuICPEFTAccounting, QuICPlacementEntry, QuICPlacementManifest, QuICProfileDefinition,
    QuICProfileId, QuICProviderSpec, QuICVariant, QuICWCAccounting, skeleton_descriptor,
)
from .errors import (
    QuICDescriptorError, QuICError, QuICImplementationUnavailableError,
    QuICProviderContractError,
)
from .provider import QuICCapabilityDeclaration, QuICExternalProvider, QuICProviderLoader
from .registration import QUIC_ADAPTER_REGISTRY, adapter_class_for
from .service import create_runtime_adapter

__all__ = [
    "EXTERNAL_QUIC_COMMIT", "OpenVLAQuICAdapter", "OpenVLAQuICPEFTAdapter",
    "OpenVLAQuICWCAdapter", "QUIC_ADAPTER_REGISTRY", "QuICCapabilityDeclaration",
    "QuICDescriptorError", "QuICError", "QuICExternalProvider",
    "QuICImplementationStatus", "QuICImplementationUnavailableError", "QuICMethodDescriptor",
    "QuICPEFTAccounting", "QuICPlacementEntry", "QuICPlacementManifest",
    "QuICProfileDefinition", "QuICProfileId", "QuICProviderContractError",
    "QuICProviderLoader", "QuICProviderSpec", "QuICVariant", "QuICWCAccounting",
    "adapter_class_for", "create_runtime_adapter", "descriptor_from_document", "skeleton_descriptor",
]
