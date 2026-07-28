from copy import deepcopy
from pathlib import Path

import pytest

from ovlab_benchctl.errors import ConfigSchemaError
from ovlab_benchctl.resolver import ConfigResolver
from ovlab_benchctl.schema import validate
from ovlab_benchctl.strict_yaml import load
from ovlab_openvla_quic import (
    EXTERNAL_QUIC_COMMIT,
    QuICImplementationUnavailableError,
    QuICVariant,
    descriptor_from_document,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
CONFIG_ROOT = REPOSITORY_ROOT / "configs"
PACKAGE_ROOT = REPOSITORY_ROOT / "code/policies/openvla-quic"
CONFIGS = {
    QuICVariant.PEFT: "policies/openvla-quic/quic-peft-bones.yaml",
    QuICVariant.WC: "policies/openvla-quic/quic-wc-bones.yaml",
}


def _walk(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value


@pytest.mark.parametrize("variant", list(QuICVariant))
def test_portable_descriptor_only_configuration_validates_without_external_import(variant):
    resolver = ConfigResolver(CONFIG_ROOT, repository_root=REPOSITORY_ROOT)
    document = resolver.load_component(CONFIGS[variant], "quic_policy_descriptor")
    descriptor = descriptor_from_document(document)
    assert descriptor.variant is variant
    assert descriptor.provider.source_commit == EXTERNAL_QUIC_COMMIT
    assert document["descriptor_mode"] is True
    assert document["implementation_status"] == "skeleton"
    assert document["runtime_validated"] is False
    assert document["compression_verified"] is False
    assert all(
        not (isinstance(value, str) and Path(value).is_absolute())
        for value in _walk(document)
    )
    with pytest.raises(QuICImplementationUnavailableError):
        descriptor.require_runtime_ready()


@pytest.mark.parametrize("variant", list(QuICVariant))
def test_skeleton_configuration_rejects_runtime_or_compression_claims(variant):
    document = load(CONFIG_ROOT / CONFIGS[variant])
    for field in ("runtime_validated", "compression_verified"):
        mutation = deepcopy(document)
        mutation[field] = True
        with pytest.raises(ConfigSchemaError, match="cannot claim"):
            validate(mutation, "test", "quic_policy_descriptor")


def test_configurations_cannot_alias_or_swap_peft_and_wc_semantics():
    peft = load(CONFIG_ROOT / CONFIGS[QuICVariant.PEFT])
    peft["id"] = "quic-wc"
    with pytest.raises(ConfigSchemaError, match="id must equal"):
        validate(peft, "test", "quic_policy_descriptor")

    wc = load(CONFIG_ROOT / CONFIGS[QuICVariant.WC])
    wc["published_method_relation"] = "direct"
    with pytest.raises(ConfigSchemaError, match="published_method_relation"):
        validate(wc, "test", "quic_policy_descriptor")


@pytest.mark.parametrize("profile", ("QP1", "QP2", "QP3", "QP4"))
def test_gate_f_qp_profiles_remain_unresolved_without_invented_numbers(profile):
    document = load(CONFIG_ROOT / CONFIGS[QuICVariant.PEFT])
    document["profile"] = {
        "id": profile,
        "definition_availability": "unresolved",
        "definition_version": None,
        "definition_hash": None,
    }
    validate(document, "test", "quic_policy_descriptor")
    mutation = deepcopy(document)
    mutation["profile"]["definition_version"] = "guessed-v1"
    mutation["profile"]["definition_hash"] = "a" * 64
    with pytest.raises(ConfigSchemaError, match="must not invent"):
        validate(mutation, "test", "quic_policy_descriptor")


def test_provider_boundary_is_pinned_and_cannot_be_redirected():
    document = load(CONFIG_ROOT / CONFIGS[QuICVariant.PEFT])
    document["external_provider"]["source_commit"] = "0" * 40
    with pytest.raises(ConfigSchemaError, match="pinned public API boundary"):
        validate(document, "test", "quic_policy_descriptor")


def test_quic_package_contains_only_wrapper_contract_layers():
    production_files = {path.name for path in (PACKAGE_ROOT / "src/ovlab_openvla_quic").glob("*.py")}
    assert production_files == {
        "__init__.py", "adapter.py", "config.py", "descriptors.py", "errors.py",
        "provider.py", "registration.py", "service.py",
    }
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PACKAGE_ROOT / "src/ovlab_openvla_quic").glob("*.py"))
    )
    assert "from openvla_quic" not in source
    assert "import openvla_quic" not in source
    assert "NotImplementedError" not in source
    forbidden_modules = {
        "layers.py", "factorization.py", "training.py", "kernels.py",
        "conversion.py", "optimizers.py", "losses.py",
    }
    assert not production_files & forbidden_modules


def test_non_quic_registry_method_metadata_has_no_qp_fields():
    registry = load(CONFIG_ROOT / "resources/registry.yaml")
    for resource_id, entry in registry["checkpoints"].items():
        method = entry.get("method", {})
        assert not any(key.lower().startswith("qp") for key in method), resource_id


def test_no_runnable_quic_experiment_was_added():
    experiment_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (CONFIG_ROOT / "experiments").rglob("*.yaml")
    )
    assert "quic-peft" not in experiment_text
    assert "quic-wc" not in experiment_text

