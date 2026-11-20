"""Explicit configuration ownership, partial defaults and frozen workers."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest

from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.config_references import ConfigReferences
from ovlab_benchctl.errors import ConfigReferenceError, ConfigSchemaError
from ovlab_benchctl.experiment_workflow import ExperimentWorkflow
from ovlab_benchctl.resolver import ConfigResolver
from ovlab_benchctl.strict_yaml import dumps

ROOT = Path(__file__).resolve().parents[4]


def test_builtin_and_user_origins_are_explicit_and_do_not_shadow_each_other(tmp_path):
    (tmp_path / "config/protocols").mkdir(parents=True)
    (tmp_path / "configs/protocols").mkdir(parents=True)
    builtin = tmp_path / "config/protocols/default.yaml"
    local = tmp_path / "configs/protocols/default.yaml"
    builtin.write_text("builtin")
    local.write_text("user")
    (tmp_path / "configs/ovlab/protocols").mkdir(parents=True)
    (tmp_path / "configs/ovlab/protocols/default.yaml").write_text("shadow")
    refs = ConfigReferences(tmp_path)
    assert refs.source("ovlab://protocols/default.yaml") == builtin
    assert refs.source("protocols/default.yaml") == local
    assert refs.source("configs/protocols/default.yaml") == local
    assert refs.reference(builtin) == "ovlab://protocols/default.yaml"
    assert refs.bundle_path(builtin) == "ovlab/protocols/default.yaml"
    assert refs.parent(local, "ovlab://protocols/default.yaml") == builtin
    with pytest.raises(ConfigReferenceError, match="escapes"):
        refs.parent(local, "../../config/protocols/default.yaml")


@pytest.mark.parametrize("reference", [
    "ovlab://protocols/../secret.yaml", "ovlab://protocols/%2e%2e/secret.yaml",
    "ovlab://protocols/default.yaml?override=1", "ovlab://protocols/default.yaml#other",
    "ovlab://protocols\\secret/default.yaml", "https://example.com/default.yaml",
])
def test_configuration_uris_reject_ambiguous_or_escaping_paths(tmp_path, reference):
    with pytest.raises(ConfigReferenceError):
        ConfigReferences(tmp_path).source(reference)


def test_frozen_builtin_references_do_not_need_the_original_repository(tmp_path):
    target = tmp_path / "configs/ovlab/protocols/default.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("frozen")
    (tmp_path / "configs/.ovlab-bundle.json").write_text("{}")
    refs = ConfigReferences(tmp_path)
    assert refs.source("ovlab://protocols/default.yaml") == target
    assert refs.source("ovlab/protocols/default.yaml") == target
    assert refs.reference(target) == "ovlab://protocols/default.yaml"


def test_partial_model_evaluation_defaults_are_validated_after_user_composition(tmp_path):
    root = tmp_path / "ovlab"
    (root / "plugins").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "plugins/registry.json").write_text((ROOT / "plugins/registry.json").read_text())
    registration = json.loads((root / "plugins/registry.json").read_text())
    for name, entry in registration["plugins"].items():
        entry["source"]["path"] = str(ROOT / "plugins" / name)
    (root / "plugins/registry.json").write_text(json.dumps(registration))
    builder = ConfigBundleBuilder(root)
    partial = builder.load_component("plugin://openvla/evaluate/default.yaml", "experiment", validate_document=False)
    assert "benchmark" not in partial["components"]
    with pytest.raises(ConfigSchemaError, match="missing"):
        builder.load_component("plugin://openvla/evaluate/default.yaml", "experiment")
    local = root / "configs/evaluate.yaml"
    local.write_text(dumps({
        "schema_version": "0.1.0", "kind": "experiment",
        "extends": "plugin://openvla/evaluate/default.yaml",
        "experiment": {"id": "local", "name": "Local experiment", "tags": ["example"]},
        "components": {"benchmark": "ovlab://benchmarks/libero.yaml", "metrics": "ovlab://metrics/default.yaml",
                       "protocol": "ovlab://protocols/smoke.yaml", "artifacts": "ovlab://artifacts/default.yaml"},
    }))
    composed = builder.load_component("evaluate.yaml", "experiment")
    assert composed["components"]["policy"] == "plugin://openvla/policy/default.yaml"
    assert "extends" not in composed


def test_workflow_selects_plugin_defaults_directly_and_requires_training_binding():
    workflow = ExperimentWorkflow(ROOT)
    plan = workflow.plan(ROOT / "examples/experiments/plugin-pipeline-smoke.yaml")
    train = plan["nodes"][0]["configuration"]["train"]
    evaluation = plan["nodes"][1]["configuration"]["evaluate"]
    assert train["profile"] == "plugin://openvla-quic/train/default.yaml"
    assert evaluation["profile"] == "plugin://openvla-quic/evaluate/default.yaml"
    assert evaluation["checkpoint_from"] == ""
    policy = evaluation["component_documents"]["policy"]["document"]
    assert "adapter_path" not in policy["settings"]["quic"]
    with pytest.raises(ConfigSchemaError):
        ConfigResolver(ROOT / "configs", repository_root=ROOT).load_component(
            "plugin://openvla-quic/policy/default.yaml", "policy")


def test_frozen_model_bundle_contains_shared_control_code_and_builtin_closure(tmp_path):
    root = tmp_path / "ovlab"
    (root / "configs").mkdir(parents=True)
    (root / "plugins").mkdir()
    shutil.copytree(ROOT / "config", root / "config")
    registration = json.loads((ROOT / "plugins/registry.json").read_text())
    for name, entry in registration["plugins"].items():
        entry["source"]["path"] = str(ROOT / "plugins" / name)
    (root / "plugins/registry.json").write_text(json.dumps(registration))
    scenario = root / "configs/evaluate.yaml"
    scenario.write_text(dumps({
        "schema_version": "0.1.0", "kind": "experiment",
        "extends": "plugin://openvla/evaluate/default.yaml",
        "experiment": {"id": "frozen", "name": "Frozen integration", "tags": ["smoke"]},
        "components": {"benchmark": "ovlab://benchmarks/libero.yaml", "metrics": "ovlab://metrics/default.yaml",
                       "protocol": "ovlab://protocols/smoke.yaml", "artifacts": "ovlab://artifacts/default.yaml"},
    }))
    builder = ConfigBundleBuilder(root)
    bundle = builder.build("evaluate.yaml", "ovlab://profiles/egl.yaml")
    paths = {row["path"] for row in bundle.files}
    assert "ovlab/benchmarks/libero.yaml" in paths
    assert "plugins/configs/openvla/policy/default.yaml" in paths
    assert "plugins/sources/openvla-support/src/ovlab_openvla_common/plugin.py" in paths
    # Another model supporting the same artifact form is not a dependency.
    assert not any(path.startswith("plugins/projects/openvla-quic/") for path in paths)
    assert not any(path.startswith("plugins/projects/openvla-oft/") for path in paths)
    with builder.materialize(bundle) as frozen:
        worker = tmp_path / "worker"
        shutil.copytree(frozen, worker / "configs")
        code = '''from pathlib import Path
from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.policy_plugins import plugin_for_type
root = Path(__import__('sys').argv[1])
builder = ConfigBundleBuilder(root)
doc = builder.load_component('evaluate.yaml', 'experiment')
assert doc['components']['policy'] == 'plugin://openvla/policy/default.yaml'
builder.load_component('ovlab://benchmarks/libero.yaml', 'benchmark')
builder.load_component(doc['components']['policy'], 'policy')
plugin = plugin_for_type('openvla_vanilla', root)
module = __import__(plugin.implementation.__class__.__module__, fromlist=[''])
assert Path(module.__file__).is_relative_to(root)
print('frozen model control and builtin references resolved')
'''
        environment = dict(os.environ, OVLAB_ROOT=str(worker),
                           OVLAB_PLUGIN_REGISTRY=str(worker / "configs/plugins/registry.json"))
        environment["PYTHONPATH"] = os.pathsep.join(str(p) for p in ROOT.glob("code/**/src"))
        completed = subprocess.run([sys.executable, "-c", code, str(worker)], text=True,
                                   capture_output=True, env=environment)
        assert completed.returncode == 0, completed.stderr
