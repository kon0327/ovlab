from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from ovlab_benchmarks.libero import LiberoAdapterSettings, LiberoRendererBackend
from ovlab_core.contracts import GripperConvention
from ovlab_metrics import ActionModificationMetricConfig, RepeatedNoOpMetricConfig
from ovlab_openvla_common import OpenVlaArtifactForm, OpenVlaMethodFamily, action_specs_match
from ovlab_openvla_vanilla import ModelDType, OpenVlaVanillaSettings
from ovlab_benchctl import (
    ConfigCompatibilityError, ConfigReferenceError, ConfigResolver, ConfigSchemaError,
    MockBenchmarkSettings, MockPolicySettings, ResolvedConfigWriteError, load,
)
from ovlab_benchctl.schema import validate


REPOSITORY = Path(__file__).resolve().parents[4]
CONFIGS = REPOSITORY / "configs"
EXPERIMENT = "configs/experiments/libero-spatial-vanilla.yaml"


def profile(path, *, suffix="a", device="cuda:0"):
    root = path / f"machine-{suffix}"
    document = f'''schema_version: "0.1.0"
kind: local_profile
id: machine-{suffix}

paths:
  checkpoint_root: {root}/checkpoints
  dataset_root: {root}/datasets
  runs_root: {root}/runs

devices:
  primary_gpu: {device}
'''
    target = path / f"profile-{suffix}.yaml"
    target.write_text(document, encoding="utf-8")
    return target


def resolver(root=CONFIGS, repository=REPOSITORY):
    return ConfigResolver(root, repository_root=repository)


def test_extends_resolves_all_libero_variants():
    expected = {
        "spatial.yaml": "libero_spatial", "object.yaml": "libero_object",
        "goal.yaml": "libero_goal", "libero10.yaml": "libero_10",
    }
    for name, suite in expected.items():
        document = resolver().load_component(f"benchmarks/libero/{name}", "benchmark")
        assert document["settings"]["suite"] == suite
        assert document["settings"]["observation"]["cameras"]["primary"]["canonical_name"] == "camera.primary.rgb"
        assert "extends" not in document


def test_every_versioned_component_is_schema_valid():
    components = {
        "benchmarks/mock/base.yaml": "benchmark",
        "benchmarks/libero/spatial-smoke.yaml": "benchmark",
        "benchmarks/libero/libero10-smoke.yaml": "benchmark",
        "policies/mock/base.yaml": "policy",
        "policies/mock/libero-noop.yaml": "policy",
        "policies/openvla-vanilla/base.yaml": "policy",
        "policies/openvla-lora/merged-libero10.yaml": "policy",
        "interfaces/actions/mock-delta-gripper-v1.yaml": "action_interface",
        "interfaces/actions/libero-osc-pose-v1.yaml": "action_interface",
        "metrics/action-safe-v1.yaml": "metric_set",
        "protocols/libero-standard-v1.yaml": "protocol",
        "protocols/libero-smoke-v1.yaml": "protocol",
        "protocols/smoke-v1.yaml": "protocol",
        "profiles/libero-bench-egl.yaml": "execution_profile",
        "profiles/libero-playground-glfw.yaml": "execution_profile",
        "artifacts/filesystem.yaml": "artifact_store",
        "resources/registry.yaml": "resource_registry",
        "experiments/libero-spatial-vanilla.yaml": "experiment",
        "experiments/mock-e2e-smoke.yaml": "experiment",
        "experiments/libero-mock-smoke.yaml": "experiment",
        "experiments/libero-vanilla-smoke.yaml": "experiment",
        "experiments/libero10-vanilla-rpc-smoke.yaml": "experiment",
        "experiments/libero10-lora-merged-rpc-smoke.yaml": "experiment",
    }
    for reference, kind in components.items():
        document = resolver().load_component(reference, kind)
        assert document["kind"] == kind
    local = load(CONFIGS / "local/profile.example.yaml")
    validate(local, "profile.example.yaml", "local_profile")


def test_every_libero_protocol_records_all_episode_video_frames_by_default():
    for protocol in sorted((CONFIGS / "protocols").glob("libero-*.yaml")):
        document = resolver().load_component(
            str(protocol.relative_to(CONFIGS)), "protocol"
        )
        assert document["recording"]["observations"] is True, protocol.name
        assert document["recording"]["images"] is True, protocol.name


def test_resolver_constructs_owner_settings_and_verified_interfaces(tmp_path):
    resolved = resolver().resolve(EXPERIMENT, local_profile=profile(tmp_path))
    assert isinstance(resolved.benchmark_settings, LiberoAdapterSettings)
    assert isinstance(resolved.policy_settings, OpenVlaVanillaSettings)
    assert resolved.benchmark_settings.suite_names == ("LIBERO-Spatial",)
    assert resolved.benchmark_settings.maximum_episode_steps == 300
    assert resolved.benchmark_settings.renderer.requested_backend is LiberoRendererBackend.EGL
    assert resolved.benchmark_settings.renderer.resolved_backend is LiberoRendererBackend.EGL
    assert resolved.benchmark_settings.renderer.device_id == 0
    assert resolved.policy_settings.model_dtype is ModelDType.BFLOAT16
    assert resolved.policy_settings.unnorm_key == "bridge_orig"
    assert resolved.policy_settings.model.source == "openvla/openvla-7b"
    assert resolved.policy_settings.method_descriptor.family is OpenVlaMethodFamily.VANILLA
    assert resolved.policy_settings.canonical_camera_name == "camera.primary.rgb"
    assert resolved.action_spec.gripper_convention is GripperConvention.CLOSED_POSITIVE
    assert resolved.action_spec.units == ("normalized_command",) * 7
    assert resolved.metric_settings.required_metric_ids == (
        "task.success_rate", "action.smoothness_1", "action.smoothness_2")
    assert isinstance(resolved.metric_settings.configurations["failure.action_modification_rate"], ActionModificationMetricConfig)
    repeated = resolved.metric_settings.configurations["failure.repeated_no_op_rate"]
    assert isinstance(repeated, RepeatedNoOpMetricConfig) and repeated.minimum_consecutive_run_length == 5
    assert resolved.protocol_settings.rollouts_per_task == 50
    assert resolved.protocol_settings.trace_recording_policy.record_image_arrays is True
    assert resolved.protocol_settings.trace_recording_policy.image_sampling_stride == 1
    assert resolved.artifact_settings.root.endswith("machine-a/runs")


def test_mock_smoke_resolves_to_typed_runtime_settings(tmp_path):
    resolved = resolver().resolve(
        "configs/experiments/mock-e2e-smoke.yaml", local_profile=profile(tmp_path)
    )
    assert isinstance(resolved.benchmark_settings, MockBenchmarkSettings)
    assert isinstance(resolved.policy_settings, MockPolicySettings)
    assert resolved.benchmark_settings.terminal_outcomes == ("success", "time_limit")
    assert resolved.policy_settings.horizon == 1
    assert resolved.scientific_config_hash == resolver().resolve(
        "configs/experiments/mock-e2e-smoke.yaml", local_profile=profile(tmp_path, suffix="b")
    ).scientific_config_hash


def test_rpc_smoke_is_aligned_to_libero10_merged_lora_method(tmp_path):
    resolved = resolver().resolve(
        "configs/experiments/libero10-lora-merged-rpc-smoke.yaml",
        local_profile=profile(tmp_path),
        execution_profile="profiles/libero-bench-egl.yaml",
        environment={},
    )
    assert resolved.benchmark_settings.suite_names == ("LIBERO-10",)
    assert resolved.benchmark_settings.task_indices == (0,)
    assert resolved.protocol_settings.maximum_episode_steps == 2
    assert resolved.policy_settings.model.source == "openvla/openvla-7b-finetuned-libero-10"
    assert resolved.policy_settings.model.revision == "80970322773f81baa2e22fe495d0487b93a05cfa"
    assert resolved.policy_settings.unnorm_key == "libero_10"
    assert resolved.policy_settings.model_dtype is ModelDType.BFLOAT16
    assert resolved.policy_settings.attention_implementation == "flash_attention_2"
    method = resolved.policy_settings.method_descriptor
    assert method.family is OpenVlaMethodFamily.LORA
    assert method.artifact_form is OpenVlaArtifactForm.MERGED_FULL_WEIGHTS
    assert method.merge_status.value == "merged"
    assert not method.active_peft_adapter and not method.runtime_peft_modules
    assert method.quantization == "none"
    assert not any(key.lower().startswith("qp") for key in method.as_metadata())
    assert resolved.policy_settings.runtime_artifact is not None


def test_merged_lora_policy_rejects_cross_suite_acceptance(tmp_path):
    root = copied_configs(tmp_path)
    experiment = root / "experiments/libero10-lora-merged-rpc-smoke.yaml"
    experiment.write_text(
        experiment.read_text().replace(
            "benchmarks/libero/libero10-smoke.yaml", "benchmarks/libero/spatial-smoke.yaml"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigCompatibilityError, match="cross-suite"):
        ConfigResolver(root, repository_root=tmp_path).resolve(
            experiment, local_profile=profile(tmp_path), environment={}
        )


def test_merged_weights_cannot_be_configured_as_vanilla(tmp_path):
    root = copied_configs(tmp_path)
    policy = root / "policies/openvla-lora/merged-libero10.yaml"
    policy.write_text(policy.read_text().replace("openvla_lora_merged", "openvla_vanilla"), encoding="utf-8")
    with pytest.raises(ConfigCompatibilityError, match="cannot be classified.*Vanilla"):
        ConfigResolver(root, repository_root=tmp_path).resolve(
            root / "experiments/libero10-lora-merged-rpc-smoke.yaml",
            local_profile=profile(tmp_path),
            environment={},
        )


@pytest.mark.parametrize("false_family", ["oft", "quic"])
def test_merged_lora_registry_rejects_other_method_families(tmp_path, false_family):
    root = copied_configs(tmp_path)
    registry = root / "resources/registry.yaml"
    registry.write_text(
        registry.read_text().replace("family: lora", f"family: {false_family}"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigSchemaError, match="must be one of: lora"):
        ConfigResolver(root, repository_root=tmp_path).load_component(
            "resources/registry.yaml", "resource_registry"
        )


def test_merged_lora_registry_rejects_non_digest_checkpoint_identity(tmp_path):
    root = copied_configs(tmp_path)
    registry = root / "resources/registry.yaml"
    registry.write_text(
        registry.read_text().replace(
            "33abee128d94d3bf54660b48c233c525f7969608924c58152602acca1b190eed",
            "z" * 64,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigSchemaError, match="SHA-256"):
        ConfigResolver(root, repository_root=tmp_path).load_component(
            "resources/registry.yaml", "resource_registry"
        )


def test_method_identity_changes_scientific_hash_without_machine_path_leakage(tmp_path):
    baseline = resolver().resolve(
        "configs/experiments/libero10-lora-merged-rpc-smoke.yaml",
        local_profile=profile(tmp_path),
        environment={},
    )
    root = copied_configs(tmp_path)
    registry = root / "resources/registry.yaml"
    registry.write_text(
        registry.read_text().replace('version: "1.0.0"', 'version: "1.0.1"', 1),
        encoding="utf-8",
    )
    changed = ConfigResolver(root, repository_root=tmp_path).resolve(
        root / "experiments/libero10-lora-merged-rpc-smoke.yaml",
        local_profile=profile(tmp_path, suffix="b"),
        environment={},
    )
    assert baseline.scientific_config_hash != changed.scientific_config_hash
    baseline_scientific = json.dumps(
        dict(baseline.scientific_config), default=lambda value: dict(value)
    )
    assert "machine-a" not in baseline_scientific
    assert str(tmp_path) not in baseline_scientific


def test_scientific_hash_excludes_local_profile_but_execution_hash_includes_it(tmp_path):
    first = resolver().resolve(EXPERIMENT, local_profile=profile(tmp_path, suffix="a", device="cuda:0"))
    second = resolver().resolve(EXPERIMENT, local_profile=profile(tmp_path, suffix="b", device="cuda:1"))
    assert first.scientific_config_hash == second.scientific_config_hash
    assert first.execution_config_hash != second.execution_config_hash
    scientific = json.dumps(dict(first.scientific_config), default=lambda value: dict(value))
    assert "machine-a" not in scientific and "cuda:0" not in scientific
    execution = json.dumps(dict(first.execution_config), default=lambda value: dict(value))
    assert "machine-a" in execution and "cuda:0" in execution


def test_checkpoint_local_path_is_execution_only_and_preserves_registry_identity(tmp_path):
    first_profile = profile(tmp_path, suffix="checkpoint-a")
    second_profile = profile(tmp_path, suffix="checkpoint-b")
    resource_id = "openvla-7b-finetuned-libero-10"
    for target, location in (
        (first_profile, "/checkpoints/resolved/first"),
        (second_profile, "/checkpoints/resolved/second"),
    ):
        target.write_text(
            target.read_text(encoding="utf-8")
            + "resources:\n  checkpoints:\n"
            + f"    {resource_id}:\n      local_path: {location}\n",
            encoding="utf-8",
        )
    experiment = "configs/experiments/libero10-lora-merged-rpc-smoke.yaml"

    first = resolver().resolve(experiment, local_profile=first_profile, environment={})
    second = resolver().resolve(experiment, local_profile=second_profile, environment={})

    assert first.policy_settings.model.source == second.policy_settings.model.source
    assert first.policy_settings.model.revision == second.policy_settings.model.revision
    assert first.policy_settings.model.local_path == "/checkpoints/resolved/first"
    assert second.policy_settings.model.local_path == "/checkpoints/resolved/second"
    assert first.policy_settings.settings_hash == second.policy_settings.settings_hash
    assert first.scientific_config_hash == second.scientific_config_hash
    assert first.execution_config_hash != second.execution_config_hash


def test_renderer_profiles_are_execution_only_and_preserve_scientific_contracts(tmp_path):
    egl = resolver().resolve(
        EXPERIMENT,
        local_profile=profile(tmp_path),
        execution_profile="profiles/libero-bench-egl.yaml",
        environment={},
    )
    glfw = resolver().resolve(
        EXPERIMENT,
        local_profile=profile(tmp_path),
        execution_profile="profiles/libero-playground-glfw.yaml",
        environment={},
    )
    assert egl.benchmark_settings.renderer.resolved_backend is LiberoRendererBackend.EGL
    assert egl.benchmark_settings.renderer.device_id == 0
    assert glfw.benchmark_settings.renderer.resolved_backend is LiberoRendererBackend.GLFW
    assert glfw.benchmark_settings.renderer.device_id is None
    assert egl.scientific_config_hash == glfw.scientific_config_hash
    assert egl.execution_config_hash != glfw.execution_config_hash
    assert egl.benchmark_settings.suite_names == glfw.benchmark_settings.suite_names
    assert egl.benchmark_settings.task_indices == glfw.benchmark_settings.task_indices
    assert egl.benchmark_settings.camera_names == glfw.benchmark_settings.camera_names
    assert egl.benchmark_settings.base_seed == glfw.benchmark_settings.base_seed
    assert action_specs_match(egl.action_spec, glfw.action_spec)
    assert egl.metric_settings == glfw.metric_settings


def test_unsupported_renderer_backend_is_rejected_without_native_imports(tmp_path):
    import sys

    root = copied_configs(tmp_path)
    renderer = root / "profiles/libero-bench-egl.yaml"
    renderer.write_text(renderer.read_text().replace("backend: egl", "backend: vulkan"), encoding="utf-8")
    before = {name for name in sys.modules if name == "mujoco" or name == "robosuite" or name.startswith("libero.libero")}
    with pytest.raises(ConfigSchemaError, match="egl, glfw"):
        ConfigResolver(root, repository_root=tmp_path).resolve(
            root / "experiments/libero-spatial-vanilla.yaml",
            local_profile=profile(tmp_path),
            execution_profile="profiles/libero-bench-egl.yaml",
            environment={},
        )
    after = {name for name in sys.modules if name == "mujoco" or name == "robosuite" or name.startswith("libero.libero")}
    assert after == before


def test_resolved_config_is_deterministic_parseable_and_immutable(tmp_path):
    resolved = resolver().resolve(EXPERIMENT, local_profile=profile(tmp_path))
    target = resolved.write(tmp_path)
    assert target.name == "resolved_config.yaml"
    document = load(target)
    assert document["scientific_config_hash"] == resolved.scientific_config_hash
    assert document["execution_config_hash"] == resolved.execution_config_hash
    with pytest.raises(ResolvedConfigWriteError, match="already exists"):
        resolved.write(tmp_path)
    with pytest.raises(TypeError):
        resolved.execution_config["new"] = "value"


def copied_configs(tmp_path):
    destination = tmp_path / "configs"
    shutil.copytree(CONFIGS, destination)
    return destination


def test_unknown_key_is_rejected_after_composition(tmp_path):
    root = copied_configs(tmp_path)
    target = root / "benchmarks/libero/spatial.yaml"
    target.write_text(target.read_text() + "unknown_setting: true\n", encoding="utf-8")
    with pytest.raises(ConfigSchemaError, match="unknown keys"):
        ConfigResolver(root, repository_root=tmp_path).load_component("benchmarks/libero/spatial.yaml", "benchmark")


def test_component_reference_traversal_is_rejected(tmp_path):
    root = copied_configs(tmp_path)
    experiment = root / "experiments/libero-spatial-vanilla.yaml"
    experiment.write_text(experiment.read_text().replace("benchmarks/libero/spatial.yaml", "../outside.yaml"), encoding="utf-8")
    with pytest.raises(ConfigReferenceError, match="escapes"):
        ConfigResolver(root, repository_root=tmp_path).resolve(
            experiment, local_profile=profile(tmp_path))


def test_extends_cycle_is_rejected(tmp_path):
    root = copied_configs(tmp_path)
    base = root / "benchmarks/libero/base.yaml"
    base.write_text(base.read_text().replace("type: libero\n", "type: libero\nextends: spatial.yaml\n"), encoding="utf-8")
    with pytest.raises(ConfigReferenceError, match="cycle"):
        ConfigResolver(root, repository_root=tmp_path).load_component("benchmarks/libero/spatial.yaml", "benchmark")


def test_cross_component_camera_mismatch_is_rejected(tmp_path):
    root = copied_configs(tmp_path)
    policy = root / "policies/openvla-vanilla/base.yaml"
    policy.write_text(policy.read_text().replace("camera.primary.rgb", "camera.wrist.rgb", 1), encoding="utf-8")
    with pytest.raises(ConfigCompatibilityError, match="camera"):
        ConfigResolver(root, repository_root=tmp_path).resolve(
            root / "experiments/libero-spatial-vanilla.yaml", local_profile=profile(tmp_path))


def test_cross_component_action_convention_mismatch_is_rejected(tmp_path):
    root = copied_configs(tmp_path)
    interface = root / "interfaces/actions/libero-osc-pose-v1.yaml"
    interface.write_text(interface.read_text().replace("closed_positive", "open_positive"), encoding="utf-8")
    with pytest.raises(ConfigCompatibilityError, match="ActionSpec"):
        ConfigResolver(root, repository_root=tmp_path).resolve(
            root / "experiments/libero-spatial-vanilla.yaml", local_profile=profile(tmp_path))
