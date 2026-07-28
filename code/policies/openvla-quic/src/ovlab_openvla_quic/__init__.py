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
    QuICPEFTIntegrationIncompleteError, QuICProviderContractError,
    QuICWCImplementationIncompleteError,
)
from .provider import QuICCapabilityDeclaration, QuICExternalProvider, QuICProviderLoader
from .registration import QUIC_ADAPTER_REGISTRY, adapter_class_for
from .service import create_runtime_adapter
from .source import (
    COMPOUND_PEFT_ARCHIVE_SHA256, COMPOUND_PEFT_FILE_COUNT,
    COMPOUND_PEFT_MANIFEST_SHA256, COMPOUND_PEFT_PACKAGE_VERSION,
    CompoundLegacyConfig, PAPER_IDENTITY, SOURCE_LIMITATIONS,
    compound_peft_source_identity, statically_compile_compound_peft,
    verify_compound_peft_tree,
)

__all__ = [
    "EXTERNAL_QUIC_COMMIT", "OpenVLAQuICAdapter", "OpenVLAQuICPEFTAdapter",
    "OpenVLAQuICWCAdapter", "QUIC_ADAPTER_REGISTRY", "QuICCapabilityDeclaration",
    "COMPOUND_PEFT_ARCHIVE_SHA256", "COMPOUND_PEFT_FILE_COUNT",
    "COMPOUND_PEFT_MANIFEST_SHA256", "COMPOUND_PEFT_PACKAGE_VERSION",
    "CompoundLegacyConfig", "PAPER_IDENTITY", "SOURCE_LIMITATIONS",
    "QuICDescriptorError", "QuICError", "QuICExternalProvider",
    "QuICImplementationStatus", "QuICImplementationUnavailableError", "QuICMethodDescriptor",
    "QuICPEFTIntegrationIncompleteError",
    "QuICPEFTAccounting", "QuICPlacementEntry", "QuICPlacementManifest",
    "QuICProfileDefinition", "QuICProfileId", "QuICProviderContractError",
    "QuICProviderLoader", "QuICProviderSpec", "QuICVariant", "QuICWCAccounting",
    "QuICWCImplementationIncompleteError", "adapter_class_for", "compound_peft_source_identity",
    "create_runtime_adapter", "descriptor_from_document", "skeleton_descriptor",
    "statically_compile_compound_peft", "verify_compound_peft_tree",
]
