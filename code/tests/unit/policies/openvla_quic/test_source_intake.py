from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

import pytest

from ovlab_benchctl.strict_yaml import load
from ovlab_openvla_quic import (
    COMPOUND_PEFT_ARCHIVE_SHA256,
    COMPOUND_PEFT_FILE_COUNT,
    COMPOUND_PEFT_MANIFEST_SHA256,
    COMPOUND_PEFT_PACKAGE_VERSION,
    CompoundLegacyConfig,
    PAPER_IDENTITY,
    QuICDescriptorError,
    QuICPEFTIntegrationIncompleteError,
    QuICVariant,
    QuICWCImplementationIncompleteError,
    compound_peft_source_identity,
    skeleton_descriptor,
    statically_compile_compound_peft,
    verify_compound_peft_tree,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
SOURCE_ROOT = REPOSITORY_ROOT / "external/compound-peft"
EXTERNAL_PROVIDER_ROOT = REPOSITORY_ROOT / "external/openvla-quic"
PROVENANCE_PATH = REPOSITORY_ROOT / "external/compound-peft.provenance.yaml"
MANIFEST_PATH = REPOSITORY_ROOT / "external/compound-peft.manifest.sha256"


def _legacy(**overrides):
    values = {
        "r": 4,
        "compound_pattern": ["comp_1", "comp_2", "comp_3"],
        "compound_type": "comp",
        "block_share": False,
        "use_orthogonal": True,
        "num_adapters": 1,
        "adapter_multiplicative": True,
        "use_scaling": False,
        "use_offset_blocks": False,
    }
    values.update(overrides)
    return CompoundLegacyConfig.from_mapping(values)


def test_archive_record_and_extracted_manifest_have_immutable_content_identity():
    provenance = load(PROVENANCE_PATH)
    assert provenance["archive"]["sha256"] == COMPOUND_PEFT_ARCHIVE_SHA256
    assert provenance["archive"]["verification"] == (
        "user_supplied_identity_archive_not_available_for_local_recomputation"
    )
    assert provenance["archive"]["byte_size"] == "unavailable"
    assert hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() == COMPOUND_PEFT_MANIFEST_SHA256
    verified = verify_compound_peft_tree(SOURCE_ROOT)
    assert verified == {
        "source_path": str(SOURCE_ROOT.resolve()),
        "manifest_sha256": COMPOUND_PEFT_MANIFEST_SHA256,
        "file_count": COMPOUND_PEFT_FILE_COUNT,
        "license": "Apache-2.0",
        "package_version": COMPOUND_PEFT_PACKAGE_VERSION,
        "divergence_classification": "none_manifest_matches",
    }


def test_license_version_registration_and_compound_types_are_present_in_snapshot():
    assert "Version 2.0, January 2004" in (SOURCE_ROOT / "peft/LICENSE").read_text()
    init_source = (SOURCE_ROOT / "peft/src/peft/__init__.py").read_text()
    type_source = (SOURCE_ROOT / "peft/src/peft/utils/peft_types.py").read_text()
    config_source = (SOURCE_ROOT / "peft/src/peft/tuners/compound/config.py").read_text()
    model_source = (SOURCE_ROOT / "peft/src/peft/tuners/compound/model.py").read_text()
    layer_source = (SOURCE_ROOT / "peft/src/peft/tuners/compound/layer.py").read_text()
    mapping_source = (SOURCE_ROOT / "peft/src/peft/mapping.py").read_text()
    save_source = (SOURCE_ROOT / "peft/src/peft/utils/save_and_load.py").read_text()
    assert f'__version__ = "{COMPOUND_PEFT_PACKAGE_VERSION}"' in init_source
    assert 'COMPOUND = "COMPOUND"' in type_source
    assert "class CompoundConfig" in config_source
    assert "class CompoundModel" in model_source
    assert "class Linear(CompoundLayer)" in layer_source
    assert "class Conv2d(CompoundLayer)" in layer_source
    assert '"COMPOUND": CompoundConfig' in mapping_source
    assert '"COMPOUND": CompoundModel' in mapping_source
    assert "PeftType.COMPOUND" in save_source
    assert "def merge(" in layer_source and "def unmerge(" in layer_source
    assert sum(path.read_text().count("\n") for path in (
        SOURCE_ROOT / "peft/src/peft/tuners/compound/__init__.py",
        SOURCE_ROOT / "peft/src/peft/tuners/compound/config.py",
        SOURCE_ROOT / "peft/src/peft/tuners/compound/layer.py",
        SOURCE_ROOT / "peft/src/peft/tuners/compound/model.py",
    )) == 721


def test_extracted_tree_has_no_links_special_files_git_metadata_or_supplied_tests():
    paths = list(SOURCE_ROOT.rglob("*"))
    assert all(not path.is_symlink() for path in paths)
    assert all(path.is_file() or path.is_dir() for path in paths)
    assert not list(SOURCE_ROOT.rglob(".git"))
    assert not list(SOURCE_ROOT.rglob(".gitmodules"))
    assert not [
        path for path in paths
        if path.is_file() and (path.name.startswith("test") or path.name.endswith("_test.py"))
    ]
    assert all(path.stat().st_mode & 0o6000 == 0 for path in paths)


def test_static_compilation_is_syntax_only_and_generates_no_cache():
    before = sorted(SOURCE_ROOT.rglob("*.pyc"))
    compiled = statically_compile_compound_peft(SOURCE_ROOT)
    assert compiled > 0
    assert sorted(SOURCE_ROOT.rglob("*.pyc")) == before == []
    assert not list(SOURCE_ROOT.rglob("__pycache__"))


def test_exact_legacy_to_canonical_mapping_preserves_raw_semantics():
    normalized = _legacy().normalize(target_output_dimension=256)
    assert normalized["legacy"] == {
        "r": 4,
        "compound_pattern": ["comp_1", "comp_2", "comp_3"],
        "compound_type": "comp",
        "block_share": False,
        "use_orthogonal": True,
        "num_adapters": 1,
        "adapter_multiplicative": True,
        "use_scaling": False,
        "use_offset_blocks": False,
    }
    assert normalized["canonical"] == {
        "num_blocks": 4,
        "block_dimension": 64,
        "block_dimension_derivation": "target_output_dimension_divided_by_num_blocks",
        "compound_orders": [1, 2, 3],
        "compound_operation": "determinant",
        "parameter_sharing": "unshared_blocks",
        "orthogonality_enforcement": "cayley",
        "adapter_chain_length": 1,
        "adapter_composition": "multiplicative",
        "learnable_scaling": False,
        "offset_blocks": False,
    }


def test_legacy_r_is_never_interpreted_as_lora_rank_or_direct_paper_b():
    normalized = _legacy().normalize()
    assert "rank" not in normalized["canonical"]
    guards = normalized["semantic_guards"]
    assert guards["r_semantics"] == "number_of_block_diagonal_blocks_not_lora_rank"
    assert guards["r_to_paper_b_translation"] == "unresolved_not_directly_equated"
    with pytest.raises(QuICDescriptorError, match="fields must equal"):
        CompoundLegacyConfig.from_mapping({**normalized["legacy"], "lora_rank": 4})


@pytest.mark.parametrize("pattern", (["comp_4"], ["comp_1", "comp_4"], []))
def test_unsupported_or_arbitrary_compound_orders_are_rejected(pattern):
    with pytest.raises(QuICDescriptorError, match="compound_pattern|supports only"):
        _legacy(compound_pattern=pattern)


def test_implementation_extensions_remain_separate_from_paper_formulation_fields():
    normalized = _legacy(
        compound_type="perm",
        num_adapters=2,
        adapter_multiplicative=False,
        use_scaling=True,
        use_offset_blocks=True,
    ).normalize()
    classification = normalized["field_classification"]
    assert classification["paper_formulation_fields"] == [
        "canonical.compound_orders",
        "canonical.compound_operation:determinant",
        "canonical.orthogonality_enforcement",
    ]
    assert classification["implementation_extensions_active"] == [
        "compound_operation:perm",
        "multi_adapter_chain",
        "additive_adapter_composition",
        "learnable_scaling",
        "offset_blocks",
    ]
    assert normalized["semantic_guards"]["paper_equivalence_validated"] is False


def test_provenance_is_unverified_content_hash_not_a_git_revision_or_oracle():
    identity = compound_peft_source_identity()
    assert identity["source_origin"] == "user_supplied_archive"
    assert identity["source_revision_kind"] == "content_hash"
    assert identity["upstream_git_revision"] == "unavailable"
    assert identity["official_implementation_status"] == "unverified"
    assert identity["relation_to_paper"] == "claimed_by_readme_unverified"
    assert identity["scientific_oracle_status"] is False
    limitations = identity["limitations"]
    assert limitations["paper_pretrained_matrix_shape"] == "square_assumed"
    assert limitations["rectangular_openvla_projection_applicability"] == "unverified"
    assert limitations["target_module_placement"] == "unresolved"
    assert limitations["dense_adapter_materialization"] is True
    assert limitations["determinant_compute_dtype"] == "float32_promoted"
    assert limitations["merge_unmerge_validation"] == "implemented_unvalidated"
    assert limitations["checkpoint_round_trip_validation"] == "implemented_unvalidated"
    assert limitations["paper_forward_merge_numerical_equivalence"] == "unvalidated"


def test_readme_citation_disagrees_with_authoritative_paper_identity():
    readme = (SOURCE_ROOT / "README.md").read_text()
    assert PAPER_IDENTITY == {
        "title": "QuIC: Quantum-Inspired Compound Adapters for Parameter Efficient Fine-Tuning",
        "authors": ["Snehal Raj", "Brian Coyle"],
        "arxiv": "2502.06916",
    }
    assert "Compound Adapters with Dynamical Lie Algebra" in readme
    assert "Music, Luka" in readme and "Kashefi, Elham" in readme
    assert "Brian Coyle" not in readme
    assert compound_peft_source_identity()["readme_citation_agrees_with_paper"] is False


def test_descriptor_construction_imports_neither_torch_peft_nor_external_provider():
    before = set(sys.modules)
    descriptor = skeleton_descriptor(QuICVariant.PEFT)
    introduced = set(sys.modules) - before
    assert descriptor.source_import_status == "present"
    assert descriptor.generic_compound_backend_status == "legacy_reference_available"
    assert descriptor.openvla_integration_status == "skeleton"
    assert not {name for name in introduced if name == "torch" or name.startswith("torch.")}
    assert not {name for name in introduced if name == "peft" or name.startswith("peft.")}
    assert "openvla_quic.ovlab_provider" not in sys.modules


@contextmanager
def _external_provider_modules():
    path = str(EXTERNAL_PROVIDER_ROOT)
    before_modules = set(sys.modules)
    before_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("openvla_quic.ovlab_provider")
    finally:
        sys.path.remove(path)
        sys.dont_write_bytecode = before_bytecode
        for name in set(sys.modules) - before_modules:
            if name == "openvla_quic" or name.startswith("openvla_quic."):
                sys.modules.pop(name, None)


def test_provider_discovery_does_not_shadow_or_import_bundled_peft():
    peft_spec_before = importlib.util.find_spec("peft")
    with _external_provider_modules() as module:
        wc = module.create_provider(variant="quic-wc")
        assert "openvla_quic.compound_peft_bridge" not in sys.modules
        peft = module.create_provider(variant="quic-peft")
        assert peft.describe()["backend"]["source_import_status"] == "present"
        assert wc.describe()["source_import_status"] == "absent"
        assert "peft" not in sys.modules and "torch" not in sys.modules
    assert importlib.util.find_spec("peft") == peft_spec_before is None


def test_external_bridge_has_recorded_uncommitted_content_identity():
    files = sorted((EXTERNAL_PROVIDER_ROOT / "openvla_quic").glob("*.py"))
    manifest = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(EXTERNAL_PROVIDER_ROOT).as_posix()}\n"
        for path in files
    ).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    provenance = load(PROVENANCE_PATH)["ownership"]
    assert digest == provenance["backend_bridge_python_content_sha256"]
    assert provenance["backend_bridge_base_revision"] == (
        "deab81fbe4035c3de2c2da3d63db966fe3361f82"
    )
    assert provenance["backend_bridge_git_state"] == "modified_uncommitted_gate_f_addendum"


def test_peft_and_wc_runtime_fail_with_distinct_typed_errors_before_loading():
    with pytest.raises(QuICPEFTIntegrationIncompleteError) as peft_failure:
        skeleton_descriptor(QuICVariant.PEFT).require_runtime_ready()
    assert peft_failure.value.next_implementation_gate == "I"
    assert "legacy_reference_available" in peft_failure.value.implementation_status

    with pytest.raises(QuICWCImplementationIncompleteError) as wc_failure:
        skeleton_descriptor(QuICVariant.WC).require_runtime_ready()
    assert wc_failure.value.next_implementation_gate == "J"
    assert "source_absent" in wc_failure.value.implementation_status


def test_external_provider_bones_raise_before_model_or_cuda_and_wc_does_not_load_peft_bridge():
    with _external_provider_modules() as module:
        wc = module.create_provider(variant="quic-wc")
        wc_error = importlib.import_module("openvla_quic.wc_provider").QuICWCBackendIncompleteError
        with pytest.raises(wc_error):
            wc.load({})
        assert "openvla_quic.compound_peft_bridge" not in sys.modules

        peft = module.create_provider(variant="quic-peft")
        peft_error = importlib.import_module(
            "openvla_quic.compound_peft_bridge"
        ).CompoundPeftOpenVLAIntegrationIncompleteError
        with pytest.raises(peft_error):
            peft.load({})
        assert peft.load_counts() == {"base_model": 0, "quic_artifact": 0, "processor": 0}
        assert wc.load_counts() == {"base_model": 0, "quic_artifact": 0, "processor": 0}
        assert "peft" not in sys.modules and "torch" not in sys.modules


def test_wc_source_has_no_compound_peft_dense_forward_dependency():
    wc_source = (EXTERNAL_PROVIDER_ROOT / "openvla_quic/wc_provider.py").read_text()
    assert "compound_peft" not in wc_source
    assert "CompoundPeft" not in wc_source
    descriptor = skeleton_descriptor(QuICVariant.WC).as_metadata()
    assert descriptor["dense_adapter_materialization_allowed"] is False
    assert descriptor["dense_runtime_reconstruction_allowed"] is False
    assert descriptor["requires_replaced_dense_weights_at_deployment"] is False


def test_dependency_direction_never_imports_ovlab_from_external_sources():
    compound_source = "\n".join(
        path.read_text(errors="ignore") for path in SOURCE_ROOT.rglob("*.py")
    ).lower()
    provider_source = "\n".join(
        path.read_text() for path in (EXTERNAL_PROVIDER_ROOT / "openvla_quic").glob("*.py")
    ).lower()
    assert "import ovlab" not in compound_source and "from ovlab" not in compound_source
    assert "import ovlab" not in provider_source and "from ovlab" not in provider_source


def test_dense_materialization_and_float32_determinant_are_recorded_source_properties():
    layer_source = (SOURCE_ROOT / "peft/src/peft/tuners/compound/layer.py").read_text()
    assert "weight_i = self._block_diagonal" in layer_source
    assert "delta_weight = self.get_delta_weight" in layer_source
    assert "submatrices_float = submatrices.float()" in layer_source
    assert "torch.linalg.det(submatrices_float)" in layer_source


def test_scientific_hash_is_sensitive_to_source_and_manifest_identity():
    descriptor = skeleton_descriptor(QuICVariant.PEFT)
    changed_archive = replace(
        descriptor,
        source_identity={
            **dict(descriptor.source_identity),
            "archive_sha256": "a" * 64,
        },
    )
    changed_manifest = replace(
        descriptor,
        source_identity={
            **dict(descriptor.source_identity),
            "extracted_manifest_sha256": "b" * 64,
        },
    )
    assert descriptor.scientific_hash != changed_archive.scientific_hash
    assert descriptor.scientific_hash != changed_manifest.scientific_hash
    assert changed_archive.scientific_hash != changed_manifest.scientific_hash


def test_cpu_characterization_is_explicitly_deferred_without_mutating_dependencies():
    provenance = load(PROVENANCE_PATH)
    assert importlib.util.find_spec("torch") is None
    assert importlib.util.find_spec("peft") is None
    assert importlib.util.find_spec("transformers") is None
    assert provenance["validation"]["cpu_characterization"] == (
        "deferred_missing_torch_peft_transformers_in_ovlab_tester"
    )
