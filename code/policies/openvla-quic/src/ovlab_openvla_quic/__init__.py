"""Public exports loaded on demand, keeping plugin discovery dependency-light."""
from importlib import import_module

_EXPORTS = {'OpenVLAQuICAdapter': ('.adapter', 'OpenVLAQuICAdapter'),
 'OpenVLAQuICPEFTAdapter': ('.adapter', 'OpenVLAQuICPEFTAdapter'),
 'OpenVLAQuICWCAdapter': ('.adapter', 'OpenVLAQuICWCAdapter'),
 'OpenVlaQuICSettings': ('.config', 'OpenVlaQuICSettings'),
 'descriptor_from_document': ('.config', 'descriptor_from_document'),
 'EXTERNAL_QUIC_COMMIT': ('.descriptors', 'EXTERNAL_QUIC_COMMIT'),
 'QuICImplementationStatus': ('.descriptors', 'QuICImplementationStatus'),
 'QuICMethodDescriptor': ('.descriptors', 'QuICMethodDescriptor'),
 'QuICPEFTAccounting': ('.descriptors', 'QuICPEFTAccounting'),
 'QuICPlacementEntry': ('.descriptors', 'QuICPlacementEntry'),
 'QuICPlacementManifest': ('.descriptors', 'QuICPlacementManifest'),
 'QuICProfileDefinition': ('.descriptors', 'QuICProfileDefinition'),
 'QuICProfileId': ('.descriptors', 'QuICProfileId'),
 'QuICProviderSpec': ('.descriptors', 'QuICProviderSpec'),
 'QuICVariant': ('.descriptors', 'QuICVariant'),
 'QuICWCAccounting': ('.descriptors', 'QuICWCAccounting'),
 'skeleton_descriptor': ('.descriptors', 'skeleton_descriptor'),
 'QuICDescriptorError': ('.errors', 'QuICDescriptorError'),
 'QuICError': ('.errors', 'QuICError'),
 'QuICImplementationUnavailableError': ('.errors', 'QuICImplementationUnavailableError'),
 'QuICPEFTIntegrationIncompleteError': ('.errors', 'QuICPEFTIntegrationIncompleteError'),
 'QuICProviderContractError': ('.errors', 'QuICProviderContractError'),
 'QuICWCImplementationIncompleteError': ('.errors', 'QuICWCImplementationIncompleteError'),
 'QuICCapabilityDeclaration': ('.provider', 'QuICCapabilityDeclaration'),
 'QuICExternalProvider': ('.provider', 'QuICExternalProvider'),
 'QuICProviderLoader': ('.provider', 'QuICProviderLoader'),
 'QUIC_ADAPTER_REGISTRY': ('.registration', 'QUIC_ADAPTER_REGISTRY'),
 'adapter_class_for': ('.registration', 'adapter_class_for'),
 'create_policy_adapter': ('.service', 'create_policy_adapter'),
 'create_runtime_adapter': ('.service', 'create_runtime_adapter'),
 'descriptor_from_settings': ('.service', 'descriptor_from_settings'),
 'COMPOUND_PEFT_ARCHIVE_SHA256': ('.source', 'COMPOUND_PEFT_ARCHIVE_SHA256'),
 'COMPOUND_PEFT_FILE_COUNT': ('.source', 'COMPOUND_PEFT_FILE_COUNT'),
 'COMPOUND_PEFT_MANIFEST_SHA256': ('.source', 'COMPOUND_PEFT_MANIFEST_SHA256'),
 'COMPOUND_PEFT_PACKAGE_VERSION': ('.source', 'COMPOUND_PEFT_PACKAGE_VERSION'),
 'CompoundLegacyConfig': ('.source', 'CompoundLegacyConfig'),
 'PAPER_IDENTITY': ('.source', 'PAPER_IDENTITY'),
 'SOURCE_LIMITATIONS': ('.source', 'SOURCE_LIMITATIONS'),
 'compound_peft_source_identity': ('.source', 'compound_peft_source_identity'),
 'statically_compile_compound_peft': ('.source', 'statically_compile_compound_peft'),
 'verify_compound_peft_tree': ('.source', 'verify_compound_peft_tree')}

__all__ = [
    "OpenVlaQuICSettings", "create_policy_adapter", "descriptor_from_settings",
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


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
