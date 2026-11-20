"""QuIC defaults are explicit templates; algorithms remain outside OVLAB."""
from pathlib import Path

import pytest

from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.policy_plugins import call_plugin, plugin_for_type
from ovlab_benchctl.schema import validate

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
PACKAGE_ROOT = REPOSITORY_ROOT / "plugins/openvla-quic"


def test_default_adapter_template_validates_only_for_explicit_checkpoint_handoff():
    reference = "plugin://openvla-quic/policy/default.yaml"
    document = ConfigBundleBuilder(REPOSITORY_ROOT).load_component(reference, "policy", validate_document=False)
    owner = plugin_for_type(document["type"], REPOSITORY_ROOT)
    call_plugin(owner, "validate_checkpoint_template", document, reference)
    assert not {"adapter_path", "adapter_manifest_sha256", "adapter_tensor_sha256"} & document["settings"]["quic"].keys()
    with pytest.raises(ValueError, match="adapter"):
        validate(document, reference, "policy")
    document["settings"]["runtime"]["quantization"] = "4bit"
    with pytest.raises(ValueError, match="dense"):
        call_plugin(owner, "validate_checkpoint_template", document, reference)


def test_non_quic_checkpoint_method_metadata_has_no_qp_fields():
    registry = ConfigBundleBuilder(REPOSITORY_ROOT).load_component("plugin://openvla/registry/default.yaml", "resource_registry")
    for resource_id, entry in registry["checkpoints"].items():
        assert not any(key.lower().startswith("qp") for key in entry.get("method", {})), resource_id


def test_quic_package_contains_only_wrapper_contract_layers():
    production_files = {path.name for path in (PACKAGE_ROOT / "src/ovlab_openvla_quic").glob("*.py")}
    assert production_files == {
        "__init__.py", "adapter.py", "config.py", "descriptors.py", "errors.py",
        "provider.py", "registration.py", "service.py", "source.py", "runtime.py",
        "plugin.py", "plugin_documents.py",
    }
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PACKAGE_ROOT / "src/ovlab_openvla_quic").glob("*.py"))
    )
    import ast
    tree = ast.parse((PACKAGE_ROOT / "src/ovlab_openvla_quic/runtime.py").read_text())
    assert not any(isinstance(node, ast.ImportFrom) and (node.module or "").startswith("quic_petf")
                   for node in tree.body)
    assert "NotImplementedError" not in source
    forbidden_modules = {
        "layers.py", "factorization.py", "training.py", "kernels.py",
        "conversion.py", "optimizers.py", "losses.py",
    }
    assert not production_files & forbidden_modules
