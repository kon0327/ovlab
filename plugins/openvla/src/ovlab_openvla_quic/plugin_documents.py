"""Validation for historical QuIC method descriptors, owned by the plugin."""
from typing import Any
from ovlab_policy_sdk.plugin import PolicyPluginValidationError as ConfigSchemaError
from ovlab_policy_sdk.validation import mapping, exact_type, enum, non_empty_string, number
SCHEMA_VERSION = "0.1.0"

def header(doc: dict[str, Any], path: str, kind: str, *, typed: bool = False, identified: bool = False) -> None:
    expected = {"schema_version", "kind"}
    if typed: expected.add("type")
    if identified: expected.add("id")
    for key in expected:
        if key not in doc: raise ConfigSchemaError(f"{path} is missing required key {key!r}")
    if doc["schema_version"] != SCHEMA_VERSION:
        raise ConfigSchemaError(f"{path}.schema_version must equal {SCHEMA_VERSION!r}")
    if doc["kind"] != kind:
        raise ConfigSchemaError(f"{path}.kind must equal {kind!r}")


def validate_quic_policy_descriptor(doc, path):
    header(doc, path, "quic_policy_descriptor", identified=True)
    common = (
        "schema_version", "kind", "id", "variant", "mode", "family", "descriptor_mode",
        "implementation_status", "source_import_status", "generic_compound_backend_status",
        "openvla_integration_status", "runtime_validated", "training_validated",
        "libero_validated", "compression_verified",
        "published_method_relation", "weight_compression", "profile", "placement_manifest",
        "external_provider", "source", "base_model", "artifact", "provenance", "deployment_state",
        "capabilities", "normalization", "parameterization", "accounting",
        "unavailable_fields", "semantics",
    )
    mapping(doc, path, required=common)
    enum(doc["variant"], ("quic-peft", "quic-wc"), f"{path}.variant")
    expected_mode = "peft" if doc["variant"] == "quic-peft" else "wc"
    expected_relation = "direct" if expected_mode == "peft" else "proposed_extension"
    if doc["id"] != doc["variant"]:
        raise ConfigSchemaError(f"{path}.id must equal its unambiguous variant ID")
    enum(doc["mode"], (expected_mode,), f"{path}.mode")
    enum(doc["family"], ("openvla_quic",), f"{path}.family")
    if doc["descriptor_mode"] is not True:
        raise ConfigSchemaError(f"{path}.descriptor_mode must be true in Gate F")
    enum(doc["implementation_status"], ("skeleton",), f"{path}.implementation_status")
    expected_source = "present" if expected_mode == "peft" else "absent"
    expected_backend = "legacy_reference_available" if expected_mode == "peft" else "not_applicable"
    enum(doc["source_import_status"], (expected_source,), f"{path}.source_import_status")
    enum(doc["generic_compound_backend_status"], (expected_backend,), f"{path}.generic_compound_backend_status")
    enum(doc["openvla_integration_status"], ("skeleton",), f"{path}.openvla_integration_status")
    for name in ("runtime_validated", "training_validated", "libero_validated", "compression_verified"):
        if doc[name] is not False:
            raise ConfigSchemaError(f"{path} skeleton cannot claim {name}")
    enum(doc["published_method_relation"], (expected_relation,), f"{path}.published_method_relation")
    expected_compression = expected_mode == "wc"
    if doc["weight_compression"] is not expected_compression:
        raise ConfigSchemaError(f"{path}.weight_compression misclassifies the variant")
    profile = mapping(doc["profile"], f"{path}.profile", required=(
        "id", "definition_availability", "definition_version", "definition_hash",
    ))
    enum(profile["id"], ("QP0", "QP1", "QP2", "QP3", "QP4"), f"{path}.profile.id")
    expected_profile_availability = "not_applicable" if profile["id"] == "QP0" else "unresolved"
    enum(profile["definition_availability"], (expected_profile_availability,), f"{path}.profile.definition_availability")
    if profile["definition_version"] is not None or profile["definition_hash"] is not None:
        raise ConfigSchemaError(f"{path}.profile must not invent a Gate F definition")
    placement = mapping(doc["placement_manifest"], f"{path}.placement_manifest", required=(
        "availability", "version", "hash", "entries",
    ))
    enum(placement["availability"], ("unresolved",), f"{path}.placement_manifest.availability")
    if placement["version"] is not None or placement["hash"] is not None or placement["entries"] != []:
        raise ConfigSchemaError(f"{path}.placement_manifest must remain unresolved and empty")
    provider = mapping(doc["external_provider"], f"{path}.external_provider", required=(
        "package", "api_name", "api_version", "source_repository", "source_commit",
    ))
    expected_provider = {
        "package": "ovlab_openvla_quic.runtime", "api_name": "ovlab-quic-provider",
        "api_version": "1.0.0", "source_repository": "external/openvla-quic",
        "source_commit": "deab81fbe4035c3de2c2da3d63db966fe3361f82",
    }
    if provider != expected_provider:
        raise ConfigSchemaError(f"{path}.external_provider must identify the pinned public API boundary")
    if expected_mode == "peft":
        source = mapping(doc["source"], f"{path}.source", required=(
            "availability", "path", "source_origin", "source_revision_kind", "archive_sha256",
            "archive_verification", "extracted_manifest_sha256", "file_count",
            "upstream_git_revision", "official_implementation_status", "relation_to_paper",
            "scientific_oracle_status", "package_version", "license",
        ))
        expected_source_identity = {
            "availability": "available", "path": "references/compound-peft",
            "source_origin": "user_supplied_archive", "source_revision_kind": "content_hash",
            "archive_sha256": "b024ba61b852d83beec631b724489b3bc3055c4a883f2df0c05b6c9857103e9a",
            "archive_verification": "user_supplied_archive_not_locally_available",
            "extracted_manifest_sha256": "8084213849149a47f9bf84dd0c9220b319faf7df8dba39cdef3894e85e00f845",
            "file_count": 130, "upstream_git_revision": "unavailable",
            "official_implementation_status": "unverified",
            "relation_to_paper": "claimed_by_readme_unverified",
            "scientific_oracle_status": False, "package_version": "0.12.1.dev0",
            "license": "Apache-2.0",
        }
        if source != expected_source_identity:
            raise ConfigSchemaError(f"{path}.source differs from the immutable source intake")
    else:
        source = mapping(doc["source"], f"{path}.source", required=("availability", "reason"))
        if source["availability"] != "unavailable":
            raise ConfigSchemaError(f"{path}.source must remain unavailable for QuIC-WC")
        non_empty_string(source["reason"], f"{path}.source.reason")
    for name in (
        "base_model", "artifact", "provenance", "deployment_state", "capabilities",
        "normalization", "parameterization", "accounting",
    ):
        item = mapping(doc[name], f"{path}.{name}", required=("availability", "reason"))
        enum(item["availability"], ("unavailable",), f"{path}.{name}.availability")
        non_empty_string(item["reason"], f"{path}.{name}.reason")
    exact_type(doc["unavailable_fields"], list, f"{path}.unavailable_fields")
    if not doc["unavailable_fields"] or len(doc["unavailable_fields"]) != len(set(doc["unavailable_fields"])):
        raise ConfigSchemaError(f"{path}.unavailable_fields must list unique unavailable evidence")
    if any(not isinstance(item, str) or not item for item in doc["unavailable_fields"]):
        raise ConfigSchemaError(f"{path}.unavailable_fields must contain non-empty strings")
    if expected_mode == "peft":
        semantics = mapping(doc["semantics"], f"{path}.semantics", required=(
            "requires_base_model", "deployment_replaces_base_weights", "adaptation_type",
            "published_adapter_efficiency_only", "standalone_weight_compression",
            "dense_adapter_materialization", "complete_base_model_required",
        ))
        if semantics != {
            "requires_base_model": True,
            "deployment_replaces_base_weights": False,
            "adaptation_type": "multiplicative_adapter",
            "published_adapter_efficiency_only": True,
            "standalone_weight_compression": False,
            "dense_adapter_materialization": True,
            "complete_base_model_required": True,
        }:
            raise ConfigSchemaError(f"{path}.semantics misclassifies QuIC-PEFT")
    else:
        semantics = mapping(doc["semantics"], f"{path}.semantics", required=(
            "requires_dense_source_for_conversion", "requires_replaced_dense_weights_at_deployment",
            "deployment_replaces_selected_weights", "dense_runtime_reconstruction_allowed",
            "dense_adapter_materialization_allowed", "factorization_family",
        ))
        if semantics != {
            "requires_dense_source_for_conversion": "configurable",
            "requires_replaced_dense_weights_at_deployment": False,
            "deployment_replaces_selected_weights": True,
            "dense_runtime_reconstruction_allowed": False,
            "dense_adapter_materialization_allowed": False,
            "factorization_family": "unselected",
        }:
            raise ConfigSchemaError(f"{path}.semantics misclassifies QuIC-WC")
