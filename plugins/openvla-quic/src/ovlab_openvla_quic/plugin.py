"""QuIC plugin entry point and optional historical descriptor extension."""
from ovlab_openvla_common.plugin import OpenVlaPlugin

class QuicPlugin(OpenVlaPlugin):
    def validate_document(self, doc, path):
        from .plugin_documents import validate_quic_policy_descriptor
        validate_quic_policy_descriptor(doc, path)

    def resolve_document(self, doc, *, runtime=False):
        from .config import descriptor_from_document
        descriptor = descriptor_from_document(doc)
        if runtime: descriptor.require_runtime_ready()
        return {"schema_version": "ovlab-cli/1.0.0", "kind": "resolved_policy_descriptor",
                "descriptor": descriptor.as_metadata(), "scientific_config_hash": descriptor.scientific_hash,
                "execution_config_hash": None, "runtime_ready": runtime}

    def describe_document(self, doc):
        from .config import descriptor_from_document
        descriptor = descriptor_from_document(doc)
        metadata = descriptor.as_metadata()
        return {"method": descriptor.variant.value, "family": descriptor.family,
                "artifact": metadata["artifact_identity"], "capabilities": metadata["capability_identity"],
                "readiness": {"runtime_ready": False, "implementation_status": metadata["implementation_status"], "openvla_integration_status": metadata["openvla_integration_status"]},
                "profile": metadata["profile"], "scientific_config_hash": descriptor.scientific_hash,
                "execution_config_hash": None, "unavailable_fields": metadata["unavailable_fields"]}

plugin = QuicPlugin("openvla_quic", "ovlab-openvla-quic-peft")
