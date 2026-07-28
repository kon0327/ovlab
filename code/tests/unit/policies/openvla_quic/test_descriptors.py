from dataclasses import replace

import pytest

from ovlab_openvla_common import vanilla_base_method_descriptor
from ovlab_openvla_quic import (
    OpenVLAQuICPEFTAdapter,
    OpenVLAQuICWCAdapter,
    QUIC_ADAPTER_REGISTRY,
    QuICDescriptorError,
    QuICImplementationStatus,
    QuICImplementationUnavailableError,
    QuICPEFTAccounting,
    QuICPlacementEntry,
    QuICPlacementManifest,
    QuICProfileDefinition,
    QuICProfileId,
    QuICVariant,
    QuICWCAccounting,
    adapter_class_for,
    skeleton_descriptor,
)


def _available(value: str) -> dict[str, object]:
    return {"availability": "available", "identity": value}


def _implemented(variant: QuICVariant, profile: QuICProfileId = QuICProfileId.QP0):
    descriptor = skeleton_descriptor(variant, profile)
    if variant is QuICVariant.PEFT:
        artifact = {"availability": "available", "form": "multiplicative_adapter", "identity": "adapter@test"}
        deployment = {
            "availability": "available", "active_adapter": True, "merge_state": "unmerged",
            "requires_base_model": True, "deployment_replaces_base_weights": False,
        }
    else:
        artifact = {"availability": "available", "form": "compact_weight_factors", "identity": "factors@test"}
        deployment = {
            "availability": "available", "replacement_state": "replaced",
            "requires_replaced_dense_weights_at_deployment": False,
            "deployment_replaces_selected_weights": True,
            "dense_runtime_reconstruction_allowed": False,
        }
    return replace(
        descriptor,
        implementation_status=QuICImplementationStatus.IMPLEMENTED,
        profile=(
            QuICProfileDefinition(profile)
            if profile is QuICProfileId.QP0
            else QuICProfileDefinition(profile, "test-definition/1", "a" * 64)
        ),
        base_model_identity=_available("base@test"),
        artifact_identity=artifact,
        provenance_identity=_available(f"provenance@{variant.value}"),
        deployment_state=deployment,
        capability_identity=_available("capabilities@test"),
        normalization_identity=_available("normalization@test"),
        parameterization=_available("parameters@test"),
    )


def test_both_variants_are_registered_as_distinct_sibling_adapters():
    assert QUIC_ADAPTER_REGISTRY == {
        "quic-peft": OpenVLAQuICPEFTAdapter,
        "quic-wc": OpenVLAQuICWCAdapter,
    }
    assert adapter_class_for("quic-peft") is not adapter_class_for("quic-wc")
    with pytest.raises(ValueError, match="unknown QuIC variant"):
        adapter_class_for("vanilla")


def test_peft_is_published_adapter_efficiency_not_weight_compression():
    metadata = skeleton_descriptor(QuICVariant.PEFT).as_metadata()
    assert metadata["family"] == "openvla_quic"
    assert metadata["variant_id"] == "quic-peft"
    assert metadata["published_method_relation"] == "direct"
    assert metadata["requires_base_model"] is True
    assert metadata["deployment_replaces_base_weights"] is False
    assert metadata["adaptation_type"] == "multiplicative_adapter"
    assert metadata["weight_compression"] is False
    assert metadata["runtime_validated"] is False
    assert metadata["compression_verified"] is False


def test_wc_is_an_unvalidated_proposed_weight_compression_extension():
    metadata = skeleton_descriptor(QuICVariant.WC).as_metadata()
    assert metadata["variant_id"] == "quic-wc"
    assert metadata["published_method_relation"] == "proposed_extension"
    assert metadata["requires_dense_source_for_conversion"] == "configurable"
    assert metadata["requires_replaced_dense_weights_at_deployment"] is False
    assert metadata["deployment_replaces_selected_weights"] is True
    assert metadata["dense_runtime_reconstruction_allowed"] is False
    assert metadata["weight_compression"] is True
    assert metadata["runtime_validated"] is False
    assert metadata["compression_verified"] is False


def test_variant_accounting_and_compression_claims_cannot_be_confused():
    with pytest.raises(QuICDescriptorError, match="accounting schema"):
        replace(skeleton_descriptor(QuICVariant.PEFT), accounting=QuICWCAccounting())
    with pytest.raises(QuICDescriptorError, match="complete-model weight compression"):
        replace(
            _implemented(QuICVariant.PEFT),
            compression_verified=True,
            accounting=QuICPEFTAccounting(100, 10, 10, 110, 400, 40, 440),
        )
    with pytest.raises(QuICDescriptorError, match="complete accounting"):
        replace(_implemented(QuICVariant.WC), compression_verified=True)


def test_mode_specific_artifact_forms_and_deployment_states_cannot_be_swapped():
    with pytest.raises(QuICDescriptorError, match="artifact form"):
        replace(
            _implemented(QuICVariant.PEFT),
            artifact_identity={
                "availability": "available", "form": "compact_weight_factors",
                "identity": "wrong@test",
            },
        )
    with pytest.raises(QuICDescriptorError, match="deployment state"):
        replace(
            _implemented(QuICVariant.WC),
            deployment_state={
                "availability": "available", "replacement_state": "replaced",
                "requires_replaced_dense_weights_at_deployment": True,
                "deployment_replaces_selected_weights": True,
                "dense_runtime_reconstruction_allowed": False,
            },
        )
    with pytest.raises(QuICDescriptorError, match="deployment state"):
        replace(
            _implemented(QuICVariant.WC),
            deployment_state={
                "availability": "available", "replacement_state": "replaced",
                "requires_replaced_dense_weights_at_deployment": False,
                "deployment_replaces_selected_weights": True,
                "dense_runtime_reconstruction_allowed": True,
            },
        )


def test_orthogonal_and_optimized_fine_tuning_are_explicitly_distinct():
    relation = skeleton_descriptor(QuICVariant.PEFT).as_metadata()[
        "orthogonal_fine_tuning_relation"
    ]
    assert "Orthogonal Fine-Tuning (Qiu et al.)" in relation
    assert "OpenVLA-OFT Optimized Fine-Tuning" in relation


def test_qp0_has_no_active_transformation_or_definition():
    profile = QuICProfileDefinition(QuICProfileId.QP0)
    assert profile.as_metadata() == {
        "id": "QP0",
        "active_transformation": False,
        "definition_availability": "not_applicable",
        "definition_version": None,
        "definition_hash": None,
    }
    with pytest.raises(QuICDescriptorError, match="QP0 has no active"):
        QuICProfileDefinition(QuICProfileId.QP0, "v1", "a" * 64)


@pytest.mark.parametrize("profile", list(QuICProfileId)[1:])
@pytest.mark.parametrize("variant", list(QuICVariant))
def test_qp1_through_qp4_require_versioned_hashable_definitions_for_runtime(variant, profile):
    unresolved = replace(
        _implemented(variant),
        profile=QuICProfileDefinition(profile),
    )
    with pytest.raises(QuICImplementationUnavailableError) as failure:
        unresolved.require_runtime_ready()
    assert failure.value.variant == variant.value
    assert failure.value.next_implementation_gate == variant.next_gate

    resolved = _implemented(variant, profile)
    resolved.require_runtime_ready()
    assert resolved.profile.definition_version == "test-definition/1"
    assert resolved.profile.definition_hash == "a" * 64


def test_same_qp_label_has_separate_peft_and_wc_scientific_identities():
    peft = _implemented(QuICVariant.PEFT, QuICProfileId.QP2)
    wc = _implemented(QuICVariant.WC, QuICProfileId.QP2)
    assert peft.profile.profile_id is wc.profile.profile_id
    assert peft.scientific_hash != wc.scientific_hash


def test_placement_manifest_is_versioned_hashable_and_content_bound():
    entry = QuICPlacementEntry(
        component_family="language_backbone",
        selector="model.layers.*.self_attn.q_proj.weight",
        layer_indices=(0, 2),
        tensor_role="attention_query_weight",
        original_shape=(8, 8),
        protected=False,
        rationale="explicit test-only placement",
    )
    manifest = QuICPlacementManifest("available", "test-placement/1", (entry,))
    assert len(manifest.manifest_hash) == 64
    assert manifest.as_metadata()["entries"][0]["layer_indices"] == [0, 2]
    with pytest.raises(QuICDescriptorError, match="does not match"):
        QuICPlacementManifest("available", "test-placement/1", (entry,), "b" * 64)
    with pytest.raises(QuICDescriptorError, match="must not fabricate"):
        QuICPlacementManifest("unresolved", version="fake")


def test_peft_accounting_keeps_base_adapter_trainable_and_runtime_separate():
    accounting = QuICPEFTAccounting(
        base_parameters=100,
        adapter_parameters=10,
        trainable_parameters=8,
        runtime_total_parameters=110,
        base_bytes=400,
        adapter_bytes=40,
        runtime_artifact_bytes=440,
    )
    assert accounting.as_metadata()["availability"] == "available"
    assert accounting.runtime_total_parameters == accounting.base_parameters + accounting.adapter_parameters
    with pytest.raises(QuICDescriptorError, match="complete or explicitly unavailable"):
        QuICPEFTAccounting(base_parameters=100)
    with pytest.raises(QuICDescriptorError, match="P_runtime_total"):
        QuICPEFTAccounting(100, 10, 8, 109, 400, 40, 440)
    with pytest.raises(QuICDescriptorError, match="positive integer"):
        QuICPEFTAccounting(100, 0, 1, 100, 400, 1, 401)


def test_wc_accounting_keeps_replaced_factors_remainder_and_deployment_separate():
    accounting = QuICWCAccounting(
        dense_replaced_parameters=100,
        compact_factor_parameters=20,
        uncompressed_remainder_parameters=200,
        deployed_total_parameters=220,
        dense_replaced_bytes=400,
        compact_factor_bytes=80,
        uncompressed_remainder_bytes=800,
        deployed_total_bytes=880,
    )
    assert accounting.as_metadata()["availability"] == "available"
    assert accounting.deployed_total_parameters == (
        accounting.compact_factor_parameters + accounting.uncompressed_remainder_parameters
    )
    with pytest.raises(QuICDescriptorError, match="complete or explicitly unavailable"):
        QuICWCAccounting(dense_replaced_parameters=100)
    with pytest.raises(QuICDescriptorError, match="compact factors must be smaller"):
        QuICWCAccounting(100, 100, 200, 300, 400, 400, 800, 1200)
    with pytest.raises(QuICDescriptorError, match="P_deployed_total"):
        QuICWCAccounting(100, 20, 200, 219, 400, 80, 800, 880)


def test_skeleton_and_future_implemented_policy_never_share_scientific_identity():
    skeleton = skeleton_descriptor(QuICVariant.PEFT)
    implemented = _implemented(QuICVariant.PEFT)
    assert skeleton.scientific_hash != implemented.scientific_hash
    assert skeleton.execution_hash(backend="cuda", precision="bf16", kernel_identity=None) != (
        implemented.execution_hash(backend="cuda", precision="bf16", kernel_identity=None)
    )


def test_scientific_identity_covers_resources_profile_placement_capability_and_normalization():
    descriptor = _implemented(QuICVariant.PEFT, QuICProfileId.QP1)
    entry = QuICPlacementEntry(
        "language_backbone", "layers.*.projection", (1,), "weight", (8, 8), False,
        "test-only identity binding",
    )
    placement = QuICPlacementManifest("available", "test/1", (entry,))
    mutations = (
        replace(descriptor, base_model_identity=_available("different-base")),
        replace(
            descriptor,
            artifact_identity={
                "availability": "available", "form": "multiplicative_adapter",
                "identity": "different-artifact",
            },
        ),
        replace(descriptor, provenance_identity=_available("different-provenance")),
        replace(descriptor, capability_identity=_available("different-capabilities")),
        replace(descriptor, normalization_identity=_available("different-normalization")),
        replace(descriptor, parameterization=_available("different-parameters")),
        replace(descriptor, placement=placement),
        replace(
            descriptor,
            profile=QuICProfileDefinition(QuICProfileId.QP1, "test-definition/2", "b" * 64),
        ),
    )
    assert all(item.scientific_hash != descriptor.scientific_hash for item in mutations)


def test_execution_identity_covers_runtime_but_not_machine_paths_or_timings():
    descriptor = skeleton_descriptor(QuICVariant.PEFT)
    first = descriptor.execution_hash(
        backend="cuda", precision="bf16", kernel_identity="kernel@1", device_identity="gpu@test"
    )
    assert first == descriptor.execution_hash(
        backend="cuda", precision="bf16", kernel_identity="kernel@1", device_identity="gpu@test"
    )
    assert first != descriptor.execution_hash(
        backend="cpu", precision="bf16", kernel_identity="kernel@1"
    )
    assert first != descriptor.execution_hash(
        backend="cuda", precision="float32", kernel_identity="kernel@1"
    )
    assert first != descriptor.execution_hash(
        backend="cuda", precision="bf16", kernel_identity="kernel@1", device_identity="other-gpu"
    )


def test_non_quic_openvla_descriptor_contains_no_qp_namespace():
    metadata = vanilla_base_method_descriptor().as_metadata()
    assert not any(key.lower().startswith("qp") for key in metadata)
