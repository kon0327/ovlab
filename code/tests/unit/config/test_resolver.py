"""Configuration composition and runtime contracts, independent of study presets."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest

from ovlab_libero import LiberoAdapterSettings, LiberoRendererBackend
from ovlab_core.contracts import GripperConvention
from ovlab_metrics import ActionModificationMetricConfig, RepeatedNoOpMetricConfig
from ovlab_openvla_common import OpenVlaArtifactForm, OpenVlaMethodFamily, action_specs_match
from ovlab_openvla_vanilla import ModelDType, ModelQuantization, OpenVlaVanillaSettings
from ovlab_benchctl import (
    ConfigCompatibilityError, ConfigReferenceError, ConfigResolver, ConfigSchemaError,
    MockBenchmarkSettings, MockPolicySettings, ResolvedConfigWriteError, load,
)
from ovlab_benchctl.strict_yaml import dumps

REPOSITORY = Path(__file__).resolve().parents[4]


def profile(path, *, suffix="a", device="cuda:0"):
    root = path / f"machine-{suffix}"
    target = path / f"profile-{suffix}.yaml"
    target.write_text(dumps({"schema_version": "0.1.0", "kind": "local_profile", "id": f"machine-{suffix}",
        "paths": {"checkpoint_root": str(root / "checkpoints"), "dataset_root": str(root / "datasets"),
                  "runs_root": str(root / "runs")}, "devices": {"primary_gpu": device}}))
    return target


@pytest.fixture
def workspace(tmp_path):
    """Real plugin defaults plus isolated user overrides; no archived old recipes."""
    root = tmp_path / "workspace"
    (root / "configs/plugins").mkdir(parents=True)
    (root / "experiments").mkdir()
    shutil.copytree(REPOSITORY / "config", root / "config")
    registry = json.loads((REPOSITORY / "plugins/registry.json").read_text())
    for source in registry.get("infrastructure", {}).get("sources", {}).values():
        if source["kind"] == "local": source["path"] = str(REPOSITORY / source["path"])
    for entry in registry["projects"].values():
        entry["source"]["path"] = str(REPOSITORY / entry["source"]["path"])
    (root / "configs/plugins/registry.json").write_text(json.dumps(registry))
    return root


def resolver(root):
    return ConfigResolver(root / "configs", repository_root=root)


def write(root, path, document):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps(document))
    return target


def scenario(root, *, model="vanilla", quantization="none", suite="libero_10", renderer="egl"):
    project = "openvla-oft" if model == "oft" else "openvla"
    policy = f"plugin://{project}/policy/" + ("lora.yaml" if model == "lora" else "default.yaml")
    write(root, "configs/policy/local.yaml", {"schema_version": "0.1.0", "kind": "policy", "extends": policy,
        "settings": {"runtime": {"quantization": quantization}}})
    benchmark = {"schema_version": "0.1.0", "kind": "benchmark", "extends": "plugin://libero/benchmark/default.yaml",
                 "settings": {"suite": suite, "task_indices": [0]}}
    if model == "oft":
        benchmark["settings"]["observation"] = {"profile": "native_oft", "cameras": {
            "primary": {"native_name": "agentview", "canonical_name": "camera.primary.rgb"},
            "wrist": {"native_name": "robot0_eye_in_hand", "canonical_name": "camera.wrist.rgb"}}}
    write(root, "configs/benchmark/local.yaml", benchmark)
    return write(root, "experiments/local.yaml", {"schema_version": "0.1.0", "kind": "experiment",
        "extends": f"plugin://{project}/evaluate/default.yaml",
        "deployment": {"renderer": renderer},
        "experiment": {"id": "local-eval", "name": "Local evaluation", "tags": ["test"]},
        "components": {"policy": "policy/local.yaml", "benchmark": "benchmark/local.yaml",
            "metrics": "ovlab://metrics/default.yaml", "protocol": "ovlab://protocols/default.yaml",
            "artifacts": "ovlab://artifacts/default.yaml"}})


def update(path, change):
    document = load(path)
    change(document)
    path.write_text(dumps(document))


def resolve(root, experiment, tmp_path, **options):
    return resolver(root).resolve(experiment, local_profile=profile(tmp_path), environment={}, **options)


@pytest.mark.parametrize("suite", ["libero_spatial", "libero_object", "libero_goal", "libero_10"])
def test_local_extension_selects_any_supported_suite(workspace, suite):
    scenario(workspace, suite=suite)
    doc = resolver(workspace).load_component("benchmark/local.yaml", "benchmark")
    assert doc["settings"]["suite"] == suite
    assert doc["settings"]["observation"]["cameras"]["primary"]["canonical_name"] == "camera.primary.rgb"
    assert "extends" not in doc


def test_shipped_complete_components_are_schema_valid(workspace):
    components = {"benchmarks/libero.yaml": "benchmark", "benchmarks/mock.yaml": "benchmark",
        "policies/mock.yaml": "policy", "interfaces/actions/mock-delta-gripper-v1.yaml": "action_interface",
        "interfaces/actions/libero-osc-pose-v1.yaml": "action_interface", "metrics/default.yaml": "metric_set",
        "protocols/default.yaml": "protocol", "protocols/smoke.yaml": "protocol", "profiles/egl.yaml": "execution_profile",
        "profiles/glfw.yaml": "execution_profile", "artifacts/default.yaml": "artifact_store",
        "registry/default.yaml": "resource_registry", "local/example.yaml": "local_profile"}
    for reference, kind in components.items():
        relocated = {"benchmarks/libero.yaml": "benchmark/default.yaml",
                     "interfaces/actions/libero-osc-pose-v1.yaml": "interfaces/osc-pose-v1.yaml",
                     "profiles/egl.yaml": "execution/egl.yaml", "profiles/glfw.yaml": "execution/glfw.yaml"}
        uri = "plugin://libero/" + relocated[reference] if reference in relocated else "ovlab://" + reference
        assert resolver(workspace).load_component(uri, kind)["kind"] == kind
    for project in ("openvla", "openvla-oft"):
        assert resolver(workspace).load_component(f"plugin://{project}/policy/default.yaml", "policy")["kind"] == "policy"
    # QuIC policy is intentionally a template until a real adapter identity is bound.
    for protocol in ("default", "smoke"):
        document = resolver(workspace).load_component(f"ovlab://protocols/{protocol}.yaml", "protocol")
        assert document["recording"]["observations"] and document["recording"]["images"]


@pytest.mark.parametrize("model", ["lora", "oft"])
@pytest.mark.parametrize("quantization, expected", [("none", ModelQuantization.NONE),
    ("8bit", ModelQuantization.BITSANDBYTES_INT8), ("4bit", ModelQuantization.BITSANDBYTES_NF4_4BIT)])
def test_inference_quantization_preserves_method_and_training_identity(workspace, tmp_path, model, quantization, expected):
    resolved = resolve(workspace, scenario(workspace, model=model, quantization=quantization), tmp_path)
    assert resolved.policy_settings.quantization is expected
    if model == "lora":
        assert resolved.policy_settings.method_descriptor.quantization == quantization
        assert resolved.policy_settings.method_descriptor.training_quantization == "none"
        assert resolved.policy_settings.method_descriptor.family is OpenVlaMethodFamily.LORA
        assert resolved.policy_settings.method_descriptor.merge_status.value == "merged"
        assert resolved.policy_settings.runtime_artifact is not None
    else:
        assert resolved.benchmark_settings.observation_profile.value == "native_oft"


def test_runtime_quantization_changes_scientific_hash(workspace, tmp_path):
    experiment = scenario(workspace, model="lora")
    full = resolve(workspace, experiment, tmp_path)
    update(workspace / "configs/policy/local.yaml", lambda d: d["settings"]["runtime"].update(quantization="4bit"))
    quantized = resolve(workspace, experiment, tmp_path)
    assert full.scientific_config_hash != quantized.scientific_config_hash
    assert full.policy_settings.method_descriptor.quantization == "none"
    assert quantized.policy_settings.method_descriptor.quantization == "4bit"


def test_resolver_constructs_owner_settings_and_verified_interfaces(workspace, tmp_path):
    resolved = resolve(workspace, scenario(workspace, suite="libero_spatial"), tmp_path, execution_profile="plugin://libero/execution/egl.yaml")
    assert isinstance(resolved.benchmark_settings, LiberoAdapterSettings)
    assert isinstance(resolved.policy_settings, OpenVlaVanillaSettings)
    assert resolved.benchmark_settings.suite_names == ("LIBERO-Spatial",)
    assert resolved.benchmark_settings.maximum_episode_steps == 300
    assert resolved.benchmark_settings.renderer.resolved_backend is LiberoRendererBackend.EGL
    assert resolved.benchmark_settings.renderer.device_id == 0
    assert resolved.policy_settings.model_dtype is ModelDType.BFLOAT16
    assert resolved.policy_settings.unnorm_key == "bridge_orig"
    assert resolved.policy_settings.model.source == "openvla/openvla-7b"
    assert resolved.policy_settings.method_descriptor.family is OpenVlaMethodFamily.VANILLA
    assert resolved.policy_settings.canonical_camera_name == "camera.primary.rgb"
    assert resolved.action_spec.gripper_convention is GripperConvention.CLOSED_POSITIVE
    assert resolved.action_spec.units == ("normalized_command",) * 7
    assert "action_interface" not in resolved.scientific_config["components"]
    assert resolved.metric_settings.required_metric_ids == ("task.success_rate", "action.smoothness_1", "action.smoothness_2")
    assert isinstance(resolved.metric_settings.configurations["failure.action_modification_rate"], ActionModificationMetricConfig)
    repeated = resolved.metric_settings.configurations["failure.repeated_no_op_rate"]
    assert isinstance(repeated, RepeatedNoOpMetricConfig) and repeated.minimum_consecutive_run_length == 5
    assert resolved.protocol_settings.rollouts_per_task == 50
    assert resolved.protocol_settings.trace_recording_policy.image_sampling_stride == 1
    assert resolved.artifact_settings.root.endswith("machine-a/runs")


def test_mock_smoke_resolves_to_typed_runtime_settings(tmp_path):
    configured = resolver(REPOSITORY)
    experiment = "examples/experiments/mock-smoke.yaml"
    result = configured.resolve(experiment, local_profile=profile(tmp_path), environment={})
    assert isinstance(result.benchmark_settings, MockBenchmarkSettings)
    assert isinstance(result.policy_settings, MockPolicySettings)
    assert result.benchmark_settings.terminal_outcomes == ("success", "time_limit")
    assert result.policy_settings.horizon == 1
    assert result.scientific_config_hash == configured.resolve(experiment, local_profile=profile(tmp_path, suffix="b"), environment={}).scientific_config_hash


def test_merged_checkpoint_preserves_declared_artifact_semantics(workspace, tmp_path):
    result = resolve(workspace, scenario(workspace, model="lora"), tmp_path)
    assert result.policy_settings.model.revision == "80970322773f81baa2e22fe495d0487b93a05cfa"
    method = result.policy_settings.method_descriptor
    assert method.family is OpenVlaMethodFamily.LORA
    assert method.artifact_form is OpenVlaArtifactForm.MERGED_FULL_WEIGHTS
    assert not method.active_peft_adapter and not method.runtime_peft_modules
    assert result.policy_settings.runtime_artifact is not None


def test_merged_lora_rejects_cross_suite_acceptance(workspace, tmp_path):
    with pytest.raises(ConfigCompatibilityError, match="cross-suite"):
        resolve(workspace, scenario(workspace, model="lora", suite="libero_spatial"), tmp_path)


def test_merged_weights_cannot_be_configured_as_vanilla(workspace, tmp_path):
    experiment = scenario(workspace, model="lora")
    policy = resolver(workspace).load_component("plugin://openvla/policy/lora.yaml", "policy")
    policy["type"] = "openvla_vanilla"
    policy["provider"] = "plugin:policy:openvla/vanilla"
    write(workspace, "configs/policy/local.yaml", policy)
    with pytest.raises(ConfigCompatibilityError, match="cannot be classified.*Vanilla"):
        resolve(workspace, experiment, tmp_path)


@pytest.mark.parametrize("false_family", ["oft", "quic"])
def test_registry_rejects_wrong_method_family(workspace, false_family):
    write(workspace, "configs/registry/local.yaml", {"schema_version": "0.1.0", "kind": "resource_registry",
        "extends": "plugin://openvla/registry/default.yaml", "checkpoints": {
            "openvla-7b-finetuned-libero-10": {"method": {"family": false_family}}}})
    with pytest.raises(ConfigSchemaError, match="must be one of: lora"):
        resolver(workspace).load_component("registry/local.yaml", "resource_registry")


def test_registry_rejects_non_digest_identity(workspace):
    write(workspace, "configs/registry/local.yaml", {"schema_version": "0.1.0", "kind": "resource_registry",
        "extends": "plugin://openvla/registry/default.yaml", "checkpoints": {
            "openvla-7b-finetuned-libero-10": {"expected_sha256": "z" * 64}}})
    with pytest.raises(ConfigSchemaError, match="SHA-256"):
        resolver(workspace).load_component("registry/local.yaml", "resource_registry")


def test_method_identity_changes_scientific_hash_without_machine_paths(workspace, tmp_path):
    experiment = scenario(workspace, model="lora")
    baseline = resolve(workspace, experiment, tmp_path)
    write(workspace, "configs/registry/local.yaml", {"schema_version": "0.1.0", "kind": "resource_registry",
        "extends": "plugin://openvla/registry/default.yaml", "checkpoints": {
            "openvla-7b-finetuned-libero-10": {"method": {"version": "1.0.1"}}}})
    update(experiment, lambda d: d.update(resources={"registry": "registry/local.yaml"}))
    changed = resolve(workspace, experiment, tmp_path)
    assert baseline.scientific_config_hash != changed.scientific_config_hash
    scientific = json.dumps(dict(changed.scientific_config), default=dict)
    assert "machine-a" not in scientific and str(tmp_path) not in scientific


def test_scientific_hash_excludes_local_profile_but_execution_hash_includes_it(workspace, tmp_path):
    experiment = scenario(workspace)
    first = resolver(workspace).resolve(experiment, local_profile=profile(tmp_path, suffix="a", device="cuda:0"), environment={})
    second = resolver(workspace).resolve(experiment, local_profile=profile(tmp_path, suffix="b", device="cuda:1"), environment={})
    assert first.scientific_config_hash == second.scientific_config_hash
    assert first.execution_config_hash != second.execution_config_hash
    scientific = json.dumps(dict(first.scientific_config), default=dict)
    execution = json.dumps(dict(first.execution_config), default=dict)
    assert "machine-a" not in scientific and "cuda:0" not in scientific
    assert "machine-a" in execution and "cuda:0" in execution


def test_checkpoint_local_path_is_execution_only(workspace, tmp_path):
    experiment = scenario(workspace, model="lora")
    results = []
    for suffix in ("a", "b"):
        local = profile(tmp_path, suffix=suffix)
        update(local, lambda d: d.update(resources={"checkpoints": {
            "openvla-7b-finetuned-libero-10": {"local_path": f"/checkpoints/{suffix}"}}}))
        results.append(resolver(workspace).resolve(experiment, local_profile=local, environment={}))
    first, second = results
    assert first.policy_settings.model.source == second.policy_settings.model.source
    assert first.policy_settings.model.local_path == "/checkpoints/a"
    assert second.policy_settings.model.local_path == "/checkpoints/b"
    assert first.policy_settings.settings_hash == second.policy_settings.settings_hash
    assert first.scientific_config_hash == second.scientific_config_hash
    assert first.execution_config_hash != second.execution_config_hash


def test_renderer_profiles_are_execution_only(workspace, tmp_path):
    experiment = scenario(workspace)
    egl = resolve(workspace, experiment, tmp_path, execution_profile="plugin://libero/execution/egl.yaml")
    glfw = resolve(workspace, experiment, tmp_path, execution_profile="plugin://libero/execution/glfw.yaml")
    assert egl.benchmark_settings.renderer.resolved_backend is LiberoRendererBackend.EGL
    assert glfw.benchmark_settings.renderer.resolved_backend is LiberoRendererBackend.GLFW
    assert egl.benchmark_settings.renderer.device_id == 0
    assert glfw.benchmark_settings.renderer.device_id is None
    assert egl.scientific_config_hash == glfw.scientific_config_hash
    assert egl.execution_config_hash != glfw.execution_config_hash
    assert action_specs_match(egl.action_spec, glfw.action_spec)
    assert egl.metric_settings == glfw.metric_settings


def test_experiment_renderer_default_is_execution_only(workspace, tmp_path):
    experiment = scenario(workspace)
    egl = resolve(workspace, experiment, tmp_path)
    update(experiment, lambda d: d["deployment"].update(renderer="glfw"))
    glfw = resolve(workspace, experiment, tmp_path)
    assert egl.benchmark_settings.renderer.resolved_backend is LiberoRendererBackend.EGL
    assert glfw.benchmark_settings.renderer.resolved_backend is LiberoRendererBackend.GLFW
    assert egl.scientific_config_hash == glfw.scientific_config_hash
    assert egl.execution_config_hash != glfw.execution_config_hash


def test_deployment_profile_must_match_policy(workspace, tmp_path):
    experiment = scenario(workspace, model="oft")
    update(experiment, lambda d: d["deployment"].update(profile="openvla"))
    with pytest.raises(ConfigCompatibilityError, match="incompatible with policy type"):
        resolve(workspace, experiment, tmp_path)


def test_unsupported_renderer_is_rejected_without_native_imports(workspace, tmp_path):
    experiment = scenario(workspace)
    renderer = write(workspace, "configs/execution/invalid.yaml", load(REPOSITORY / "plugins/libero/configs/execution/egl.yaml"))
    update(renderer, lambda d: d["execution"]["libero"]["renderer"].update(backend="vulkan"))
    before = {name for name in sys.modules if name in {"mujoco", "robosuite"} or name.startswith("libero.libero")}
    with pytest.raises(ConfigSchemaError, match="egl, glfw"):
        resolve(workspace, experiment, tmp_path, execution_profile="execution/invalid.yaml")
    after = {name for name in sys.modules if name in {"mujoco", "robosuite"} or name.startswith("libero.libero")}
    assert before == after


def test_resolved_config_is_deterministic_parseable_and_immutable(workspace, tmp_path):
    resolved = resolve(workspace, scenario(workspace), tmp_path)
    target = resolved.write(tmp_path)
    assert target.name == "resolved_config.yaml"
    document = load(target)
    assert document["scientific_config_hash"] == resolved.scientific_config_hash
    assert document["execution_config_hash"] == resolved.execution_config_hash
    with pytest.raises(ResolvedConfigWriteError, match="already exists"):
        resolved.write(tmp_path)
    with pytest.raises(TypeError):
        resolved.execution_config["new"] = "value"


def test_unknown_key_is_rejected_after_composition(workspace):
    write(workspace, "configs/benchmark/local.yaml", {"schema_version": "0.1.0", "kind": "benchmark",
        "extends": "plugin://libero/benchmark/default.yaml", "unknown_setting": True})
    with pytest.raises(ConfigSchemaError, match="unknown keys"):
        resolver(workspace).load_component("benchmark/local.yaml", "benchmark")


def test_component_reference_traversal_is_rejected(workspace, tmp_path):
    experiment = scenario(workspace)
    update(experiment, lambda d: d["components"].update(benchmark="../outside.yaml"))
    with pytest.raises(ConfigReferenceError, match="escapes"):
        resolve(workspace, experiment, tmp_path)


def test_extends_cycle_is_rejected(workspace):
    for name, parent in (("a", "b.yaml"), ("b", "a.yaml")):
        write(workspace, f"configs/benchmark/{name}.yaml", {"schema_version": "0.1.0", "kind": "benchmark", "extends": parent})
    with pytest.raises(ConfigReferenceError, match="cycle"):
        resolver(workspace).load_component("benchmark/a.yaml", "benchmark")


def test_cross_component_camera_mismatch_is_rejected(workspace, tmp_path):
    experiment = scenario(workspace)
    update(workspace / "configs/policy/local.yaml", lambda d: d["settings"].update(input={"camera": "camera.wrist.rgb"}))
    with pytest.raises(ConfigCompatibilityError, match="camera"):
        resolve(workspace, experiment, tmp_path)


def test_cross_component_action_convention_mismatch_is_rejected(workspace, tmp_path):
    experiment = scenario(workspace)
    action = write(workspace, "configs/interfaces/custom.yaml", load(REPOSITORY / "plugins/libero/configs/interfaces/osc-pose-v1.yaml"))
    update(action, lambda d: d.update(gripper_convention="open_positive"))
    update(workspace / "configs/policy/local.yaml", lambda d: d["settings"].update(action={"action_interface": "interfaces/custom.yaml"}))
    with pytest.raises(ConfigCompatibilityError, match="native action interface"):
        resolve(workspace, experiment, tmp_path)


def test_policy_action_interface_must_match_benchmark(workspace, tmp_path):
    experiment = scenario(workspace)
    update(workspace / "configs/policy/local.yaml", lambda d: d["settings"].update(
        action={"action_interface": "ovlab://interfaces/actions/mock-delta-gripper-v1.yaml"}))
    with pytest.raises(ConfigCompatibilityError, match="native action interface"):
        resolve(workspace, experiment, tmp_path)


def test_reporting_changes_only_derived_identity(workspace, tmp_path):
    source = load(REPOSITORY / "examples/experiments/mock-smoke.yaml")
    experiment = write(workspace, "experiments/mock.yaml", source)
    enabled = resolve(workspace, experiment, tmp_path)
    update(experiment, lambda d: d["reporting"].update(enabled=False))
    disabled = resolve(workspace, experiment, tmp_path)
    assert enabled.scientific_config_hash == disabled.scientific_config_hash
    assert enabled.execution_config_hash == disabled.execution_config_hash
    assert enabled.reporting_settings.enabled and not disabled.reporting_settings.enabled


def test_explicit_local_execution_profile_is_composed_frozen_and_used(workspace, tmp_path):
    from ovlab_benchctl.config_bundle import ConfigBundleBuilder
    experiment = scenario(workspace)
    write(workspace, 'configs/execution/selected.yaml', {
        'schema_version':'0.1.0', 'kind':'execution_profile',
        'extends':'plugin://libero/execution/egl.yaml', 'id':'selected-machine-execution',
        'execution':{'libero':{'renderer':{'device_id':2}}}})
    update(experiment, lambda doc: doc['deployment'].update(execution_profile='execution/selected.yaml'))
    host = resolve(workspace, experiment, tmp_path)
    assert host.benchmark_settings.renderer.device_id == 2
    builder = ConfigBundleBuilder(workspace)
    bundle = builder.build('experiments/local.yaml', None)
    assert bundle.execution_profile == 'execution/selected.yaml'
    assert {'execution/selected.yaml', 'plugins/configs/libero/execution/egl.yaml'} <= {row['path'] for row in bundle.files}
    with builder.materialize(bundle) as frozen:
        worker = tmp_path / 'worker-execution'
        shutil.copytree(frozen, worker / 'configs')
        loaded = ConfigResolver(worker / 'configs', repository_root=worker).resolve(
            worker / 'configs' / bundle.experiment, local_profile=profile(tmp_path, suffix='worker'), environment={})
        assert loaded.benchmark_settings.renderer.device_id == 2
        assert loaded.scientific_config_hash == host.scientific_config_hash
