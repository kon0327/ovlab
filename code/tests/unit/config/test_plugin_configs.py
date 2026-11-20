"""Project-owned config inheritance and frozen worker handoff."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest

from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchctl.config_references import ConfigReferences
from ovlab_benchctl.errors import ConfigReferenceError
from ovlab_benchctl.policy_plugins import registry_scope
from ovlab_benchctl.resolver import ConfigResolver
from ovlab_benchctl.strict_yaml import dumps
from ovlab_benchctl.training_composition import load_training_document

ROOT = Path(__file__).resolve().parents[4]


def _project(tmp_path):
    root = tmp_path / "ovlab"
    project = tmp_path / "standalone"
    (root / "configs/plugins").mkdir(parents=True)
    (project / "configs").mkdir(parents=True)
    (project / "plugin.json").write_text(json.dumps({
        "schema_version": "ovlab.plugin/v1", "api_version": "1", "kind": "policy",
        "id": "demo", "version": "1", "configs": {"root": "configs"},
    }))
    (root / "configs/plugins/registry.json").write_text(json.dumps({
        "schema_version": "ovlab.plugins/v2", "plugins": {
            "demo": {"source": {"kind": "local", "path": str(project)}}},
    }))
    return root, project


def test_local_catalog_can_extend_plugin_catalog_and_override_one_checkpoint(tmp_path):
    root, project = _project(tmp_path)
    (project / "configs/resources.yaml").write_text(dumps({
        "schema_version": "0.1.0", "kind": "resource_registry",
        "checkpoints": {"base": {"repo_id": "owner/base", "revision": "local/main"}},
        "repositories": {},
    }))
    (root / "configs/resources.yaml").write_text(dumps({
        "schema_version": "0.1.0", "kind": "resource_registry",
        "extends": "plugin://demo/resources.yaml",
        "checkpoints": {"base": {"revision": "remote/main"}, "own": {"repo_id": "owner/own"}},
    }))
    result = ConfigResolver(root / "configs", repository_root=root).load_component("resources.yaml", "resource_registry")
    assert result["checkpoints"]["base"] == {"repo_id": "owner/base", "revision": "remote/main"}
    assert result["checkpoints"]["own"] == {"repo_id": "owner/own"}


def test_cross_origin_cycles_and_plugin_traversal_are_rejected(tmp_path):
    root, project = _project(tmp_path)
    base = {"schema_version": "0.1.0", "kind": "resource_registry"}
    for name, parent in (("a", "b.yaml"), ("b", "plugin://demo/a.yaml")):
        (project / f"configs/{name}.yaml").write_text(dumps({**base, "extends": parent}))
    resolver = ConfigResolver(root / "configs", repository_root=root)
    with pytest.raises(ConfigReferenceError, match="cycle"):
        resolver.load_component("plugin://demo/a.yaml", "resource_registry")
    for reference in ("plugin://demo/../secret.yaml", "plugin://demo/%2e%2e/secret.yaml"):
        with pytest.raises(ConfigReferenceError, match="path"):
            resolver.load_component(reference, "resource_registry")
    (project / "configs/escape.yaml").write_text(dumps({**base, "extends": "../plugin.json"}))
    with pytest.raises(ConfigReferenceError, match="escapes"):
        resolver.load_component("plugin://demo/escape.yaml", "resource_registry")


def test_training_profile_can_extend_project_defaults(tmp_path):
    root, project = _project(tmp_path)
    (project / "configs/train.yaml").write_text(dumps({"kind": "training", "schema_version": "1", "training": {"seed": 1, "max_steps": 2}}))
    local = root / "experiments/training/local.yaml"
    local.parent.mkdir(parents=True)
    local.write_text(dumps({"kind": "training", "extends": "plugin://demo/train.yaml", "training": {"max_steps": 5}}))
    result = load_training_document(local, repository_root=root, expected_kind="training")
    assert result["training"] == {"seed": 1, "max_steps": 5}


def test_bundle_contains_project_code_configs_and_resolves_without_original_project(tmp_path):
    root = tmp_path / "host"
    (root / "configs/plugins").mkdir(parents=True)
    (root / "examples").mkdir()
    shutil.copytree(ROOT / "config", root / "config")
    registry = json.loads((ROOT / "plugins/registry.json").read_text())
    for item in registry["plugins"].values():
        item["source"]["path"] = str(ROOT / item["source"]["path"])
    (root / "configs/plugins/registry.json").write_text(json.dumps(registry))
    (root / "examples/eval.yaml").write_text(dumps({
        "schema_version": "0.1.0", "kind": "experiment",
        "extends": "plugin://openvla/evaluate/default.yaml",
        "experiment": {"id": "plugin-eval", "name": "Plugin evaluation", "tags": ["test"]},
        "components": {"benchmark": "ovlab://benchmarks/libero.yaml", "metrics": "ovlab://metrics/default.yaml",
                       "protocol": "ovlab://protocols/smoke.yaml", "artifacts": "ovlab://artifacts/default.yaml"},
    }))
    profile = tmp_path / "machine.yaml"
    profile.write_text(dumps({"schema_version": "0.1.0", "kind": "local_profile", "id": "machine",
        "paths": {"checkpoint_root": "/fixture/checkpoints", "dataset_root": "/fixture/datasets", "runs_root": "/fixture/runs"},
        "devices": {"primary_gpu": "cuda:0"}}))
    expected = ConfigResolver(root / "configs", repository_root=root).resolve(
        "examples/eval.yaml", local_profile=profile, environment={})
    builder = ConfigBundleBuilder(root)
    bundle = builder.build("examples/eval.yaml", "ovlab://profiles/egl.yaml")
    paths = {row["path"] for row in bundle.files}
    assert "plugins/configs/openvla/policy/default.yaml" in paths
    assert "plugins/configs/openvla/registry/default.yaml" in paths
    assert "plugins/projects/openvla/src/ovlab_openvla_vanilla/plugin.py" in paths
    assert "ovlab/benchmarks/libero.yaml" in paths
    assert all(not Path(path).is_absolute() and ".." not in Path(path).parts for path in paths)
    with builder.materialize(bundle) as frozen:
        worker = tmp_path / "worker"
        shutil.copytree(frozen, worker / "configs")
        environment = dict(os.environ, OVLAB_ROOT=str(worker), OVLAB_PLUGIN_REGISTRY=str(worker / "configs/plugins/registry.json"))
        # A fresh worker has SDK/core on PYTHONPATH, with no original model source roots.
        environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in ROOT.glob("code/**/src"))
        script = (
            "from pathlib import Path; from ovlab_benchctl.resolver import ConfigResolver; "
            f"r=ConfigResolver(Path({str(worker / 'configs')!r}), repository_root=Path({str(worker)!r})); "
            "p=r.load_component('plugin://openvla/policy/default.yaml','policy'); "
            "assert p['settings']['checkpoint_id']=='openvla-7b'; "
            f"x=r.resolve('configs/{bundle.experiment}',local_profile=Path({str(profile)!r}),environment={{}}); "
            "print(x.scientific_config_hash)"
        )
        result = subprocess.run([sys.executable, "-c", script], env=environment, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected.scientific_config_hash
