"""Exact kind-specific schemas for OVLAB configuration version 0.1.0."""

from pathlib import Path
from typing import Any

from .errors import ConfigSchemaError

SCHEMA_VERSION = "0.1.0"


def mapping(value: Any, path: str, *, required=(), optional=()) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigSchemaError(f"{path} must be a mapping")
    required, allowed = set(required), set(required) | set(optional)
    missing, unknown = sorted(required - set(value)), sorted(set(value) - allowed)
    if missing: raise ConfigSchemaError(f"{path} is missing required keys: {', '.join(missing)}")
    if unknown: raise ConfigSchemaError(f"{path} contains unknown keys: {', '.join(unknown)}")
    return value


def exact_type(value: Any, expected: type | tuple[type, ...], path: str) -> None:
    types = expected if isinstance(expected, tuple) else (expected,)
    if bool in types:
        valid = type(value) is bool
    elif int in types:
        valid = type(value) is int
    else:
        valid = isinstance(value, types)
    if not valid:
        names = "/".join(item.__name__ for item in types)
        raise ConfigSchemaError(f"{path} must be {names}, got {type(value).__name__}")


def enum(value: Any, choices: tuple[str, ...], path: str) -> None:
    exact_type(value, str, path)
    if value not in choices:
        raise ConfigSchemaError(f"{path} must be one of: {', '.join(choices)}")


def non_empty_string(value: Any, path: str) -> None:
    exact_type(value, str, path)
    if not value.strip(): raise ConfigSchemaError(f"{path} must not be empty")


def number(value: Any, path: str) -> None:
    if type(value) not in (int, float): raise ConfigSchemaError(f"{path} must be a number")


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


def validate_experiment(doc, path):
    header(doc, path, "experiment")
    mapping(
        doc,
        path,
        required=("schema_version", "kind", "experiment", "components", "resources"),
        optional=("deployment", "reporting"),
    )
    experiment = mapping(doc["experiment"], f"{path}.experiment", required=("id", "name", "tags"))
    non_empty_string(experiment["id"], f"{path}.experiment.id")
    non_empty_string(experiment["name"], f"{path}.experiment.name")
    exact_type(experiment["tags"], list, f"{path}.experiment.tags")
    if not experiment["tags"] or any(not isinstance(item, str) or not item for item in experiment["tags"]):
        raise ConfigSchemaError(f"{path}.experiment.tags must contain non-empty strings")
    components = mapping(doc["components"], f"{path}.components", required=(
        "benchmark", "policy", "metrics", "protocol", "action_interface", "artifacts"))
    resources = mapping(doc["resources"], f"{path}.resources", required=("registry",))
    for key, value in components.items(): exact_type(value, str, f"{path}.components.{key}")
    exact_type(resources["registry"], str, f"{path}.resources.registry")
    if "deployment" in doc:
        deployment = mapping(
            doc["deployment"],
            f"{path}.deployment",
            required=("profile", "renderer"),
        )
        enum(deployment["profile"], ("openvla", "oft"), f"{path}.deployment.profile")
        enum(deployment["renderer"], ("egl", "glfw"), f"{path}.deployment.renderer")
    if "reporting" in doc:
        reporting = mapping(
            doc["reporting"], f"{path}.reporting",
            required=("enabled", "profile", "on_task_finalize", "on_run_finalize", "failure_policy"),
        )
        for key in ("enabled", "on_task_finalize", "on_run_finalize"):
            exact_type(reporting[key], bool, f"{path}.reporting.{key}")
        non_empty_string(reporting["profile"], f"{path}.reporting.profile")
        enum(reporting["failure_policy"], ("warn",), f"{path}.reporting.failure_policy")


def validate_benchmark(doc, path):
    header(doc, path, "benchmark", typed=True)
    mapping(doc, path, required=("schema_version", "kind", "type", "settings"), optional=("extends",))
    if doc["type"] == "mock":
        settings = mapping(doc["settings"], f"{path}.settings", required=(
            "task_count", "maximum_episode_steps", "modify_actions", "terminal_outcomes",
            "observation", "action", "privileged_signals"))
        for key in ("task_count", "maximum_episode_steps"):
            exact_type(settings[key], int, f"{path}.settings.{key}")
            if settings[key] <= 0: raise ConfigSchemaError(f"{path}.settings.{key} must be positive")
        exact_type(settings["modify_actions"], bool, f"{path}.settings.modify_actions")
        exact_type(settings["terminal_outcomes"], list, f"{path}.settings.terminal_outcomes")
        if len(settings["terminal_outcomes"]) != settings["task_count"]:
            raise ConfigSchemaError(f"{path}.settings.terminal_outcomes must match task_count")
        for outcome in settings["terminal_outcomes"]:
            enum(outcome, ("success", "failure", "time_limit"), f"{path}.settings.terminal_outcomes[]")
        obs = mapping(settings["observation"], f"{path}.settings.observation", required=(
            "camera", "width", "height", "dtype", "color_space", "proprioception"))
        non_empty_string(obs["camera"], f"{path}.settings.observation.camera")
        non_empty_string(obs["proprioception"], f"{path}.settings.observation.proprioception")
        for key in ("width", "height"):
            exact_type(obs[key], int, f"{path}.settings.observation.{key}")
            if obs[key] <= 0: raise ConfigSchemaError(f"{path}.settings.observation.{key} must be positive")
        enum(obs["dtype"], ("uint8",), f"{path}.settings.observation.dtype")
        enum(obs["color_space"], ("rgb",), f"{path}.settings.observation.color_space")
        action = mapping(settings["action"], f"{path}.settings.action", required=("interface_ref",))
        non_empty_string(action["interface_ref"], f"{path}.settings.action.interface_ref")
        signals = mapping(settings["privileged_signals"], f"{path}.settings.privileged_signals", required=("enabled",))
        exact_type(signals["enabled"], bool, f"{path}.settings.privileged_signals.enabled")
        return
    if doc["type"] != "libero": raise ConfigSchemaError(f"{path}.type supports only 'libero' or 'mock'")
    settings = mapping(doc["settings"], f"{path}.settings", required=(
        "suite", "task_indices", "observation", "initialization", "action", "privileged_signals"))
    enum(settings["suite"], ("libero_spatial", "libero_object", "libero_goal", "libero_10"), f"{path}.settings.suite")
    if settings["task_indices"] != "all":
        exact_type(settings["task_indices"], list, f"{path}.settings.task_indices")
        if any(type(item) is not int or item < 0 for item in settings["task_indices"]):
            raise ConfigSchemaError(f"{path}.settings.task_indices must be 'all' or non-negative integers")
    obs = mapping(settings["observation"], f"{path}.settings.observation", required=(
        "profile", "cameras", "width", "height", "color_space", "dtype"))
    enum(obs["profile"], ("primary_rgb", "native_oft"), f"{path}.settings.observation.profile")
    camera_keys = ("primary", "wrist") if obs["profile"] == "native_oft" else ("primary",)
    cameras = mapping(obs["cameras"], f"{path}.settings.observation.cameras", required=camera_keys)
    primary = mapping(cameras["primary"], f"{path}.settings.observation.cameras.primary", required=("native_name", "canonical_name"))
    for key in primary: exact_type(primary[key], str, f"{path}.settings.observation.cameras.primary.{key}")
    if "wrist" in cameras:
        wrist = mapping(cameras["wrist"], f"{path}.settings.observation.cameras.wrist", required=("native_name", "canonical_name"))
        for key in wrist: exact_type(wrist[key], str, f"{path}.settings.observation.cameras.wrist.{key}")
    for key in ("width", "height"):
        exact_type(obs[key], int, f"{path}.settings.observation.{key}")
        if obs[key] <= 0: raise ConfigSchemaError(f"{path}.settings.observation.{key} must be positive")
    enum(obs["color_space"], ("rgb",), f"{path}.settings.observation.color_space")
    enum(obs["dtype"], ("uint8",), f"{path}.settings.observation.dtype")
    init = mapping(settings["initialization"], f"{path}.settings.initialization", required=("state_selection", "settling_steps"))
    enum(init["state_selection"], ("rollout_index", "seeded"), f"{path}.settings.initialization.state_selection")
    exact_type(init["settling_steps"], int, f"{path}.settings.initialization.settling_steps")
    if init["settling_steps"] < 0: raise ConfigSchemaError(f"{path}.settings.initialization.settling_steps must be non-negative")
    action = mapping(settings["action"], f"{path}.settings.action", required=("interface_ref",))
    exact_type(action["interface_ref"], str, f"{path}.settings.action.interface_ref")
    signals = mapping(settings["privileged_signals"], f"{path}.settings.privileged_signals", required=("enabled",))
    exact_type(signals["enabled"], bool, f"{path}.settings.privileged_signals.enabled")


def validate_policy(doc, path):
    header(doc, path, "policy", typed=True)
    mapping(doc, path, required=("schema_version", "kind", "type", "settings"), optional=("extends",))
    if doc["type"] == "mock":
        settings = mapping(doc["settings"], f"{path}.settings", required=(
            "horizon", "deterministic", "input", "action", "raw_output"))
        exact_type(settings["horizon"], int, f"{path}.settings.horizon")
        if settings["horizon"] <= 0: raise ConfigSchemaError(f"{path}.settings.horizon must be positive")
        exact_type(settings["deterministic"], bool, f"{path}.settings.deterministic")
        input_ = mapping(settings["input"], f"{path}.settings.input", required=("camera", "proprioception"))
        non_empty_string(input_["camera"], f"{path}.settings.input.camera")
        if input_["proprioception"] is not None:
            non_empty_string(input_["proprioception"], f"{path}.settings.input.proprioception")
        action = mapping(settings["action"], f"{path}.settings.action", required=("interface_ref",))
        non_empty_string(action["interface_ref"], f"{path}.settings.action.interface_ref")
        raw = mapping(settings["raw_output"], f"{path}.settings.raw_output", required=("enabled",))
        exact_type(raw["enabled"], bool, f"{path}.settings.raw_output.enabled")
        return
    if doc["type"] not in ("openvla_vanilla", "openvla_lora_merged", "openvla_oft"):
        raise ConfigSchemaError(
            f"{path}.type supports only 'openvla_vanilla', 'openvla_lora_merged', 'openvla_oft', or 'mock'"
        )
    settings = mapping(doc["settings"], f"{path}.settings", required=(
        "checkpoint_id", "processor_id", "unnorm_key", "input", "runtime", "action", "raw_output"))
    for key in ("checkpoint_id", "processor_id", "unnorm_key"):
        exact_type(settings[key], str, f"{path}.settings.{key}")
    if doc["type"] == "openvla_oft":
        input_ = mapping(settings["input"], f"{path}.settings.input", required=("cameras", "proprioception"))
        cameras = mapping(input_["cameras"], f"{path}.settings.input.cameras", required=("primary", "wrist"))
        for key, value in cameras.items(): non_empty_string(value, f"{path}.settings.input.cameras.{key}")
        non_empty_string(input_["proprioception"], f"{path}.settings.input.proprioception")
    else:
        input_ = mapping(settings["input"], f"{path}.settings.input", required=("camera",))
        exact_type(input_["camera"], str, f"{path}.settings.input.camera")
    runtime_keys = (
        "device_resource", "dtype", "attention_implementation", "local_files_only",
        "trust_remote_code", "deterministic", "synchronize_inference",
    )
    if doc["type"] != "openvla_oft":
        runtime_keys += ("quantization",)
    runtime = mapping(
        settings["runtime"], f"{path}.settings.runtime", required=runtime_keys
    )
    exact_type(runtime["device_resource"], str, f"{path}.settings.runtime.device_resource")
    enum(runtime["dtype"], ("bfloat16", "float16", "float32"), f"{path}.settings.runtime.dtype")
    if doc["type"] != "openvla_oft":
        enum(runtime["quantization"], ("none", "4bit"), f"{path}.settings.runtime.quantization")
    if runtime["attention_implementation"] is not None: exact_type(runtime["attention_implementation"], str, f"{path}.settings.runtime.attention_implementation")
    for key in ("local_files_only", "trust_remote_code", "deterministic", "synchronize_inference"):
        exact_type(runtime[key], bool, f"{path}.settings.runtime.{key}")
    action = mapping(settings["action"], f"{path}.settings.action", required=("codec", "interface_ref"))
    enum(action["codec"], ("openvla-to-libero-v1",), f"{path}.settings.action.codec")
    exact_type(action["interface_ref"], str, f"{path}.settings.action.interface_ref")
    raw = mapping(settings["raw_output"], f"{path}.settings.raw_output", required=("enabled",))
    exact_type(raw["enabled"], bool, f"{path}.settings.raw_output.enabled")


def validate_action_interface(doc, path):
    header(doc, path, "action_interface", identified=True)
    keys = ("schema_version", "kind", "id", "dimension", "representation", "translation_indices",
            "rotation_indices", "gripper_indices", "rotation_representation", "gripper_convention", "dtype",
            "units", "control_frequency_hz", "minimum", "maximum")
    mapping(doc, path, required=keys)
    exact_type(doc["dimension"], int, f"{path}.dimension")
    if doc["dimension"] <= 0: raise ConfigSchemaError(f"{path}.dimension must be positive")
    for key in ("translation_indices", "rotation_indices", "gripper_indices", "minimum", "maximum"):
        exact_type(doc[key], list, f"{path}.{key}")
    enum(doc["gripper_convention"], ("closed_positive", "open_positive", "binary_closed_one", "binary_open_one", "none"), f"{path}.gripper_convention")
    enum(doc["representation"], ("delta_pose", "absolute_pose", "joint_position", "joint_delta", "other"), f"{path}.representation")
    enum(doc["rotation_representation"], ("axis_angle", "euler_xyz", "quaternion_xyzw", "quaternion_wxyz", "none"), f"{path}.rotation_representation")
    enum(doc["dtype"], ("float32",), f"{path}.dtype")
    if isinstance(doc["units"], str):
        non_empty_string(doc["units"], f"{path}.units")
    else:
        exact_type(doc["units"], list, f"{path}.units")
        if any(not isinstance(item, str) or not item for item in doc["units"]):
            raise ConfigSchemaError(f"{path}.units must contain non-empty strings")
    number(doc["control_frequency_hz"], f"{path}.control_frequency_hz")


_PLUGIN_KEYS = {
    "task.success": ("enabled",), "task.success_rate": ("enabled",),
    "action.variance": ("enabled", "action_source"), "action.smoothness_1": ("enabled", "action_source"),
    "action.smoothness_2": ("enabled", "action_source"), "failure.invalid_prediction_rate": ("enabled",),
    "failure.action_modification_rate": ("enabled", "absolute_tolerance", "relative_tolerance"),
    "failure.repeated_no_op_rate": ("enabled", "action_source", "norm_threshold", "minimum_consecutive_steps"),
    "failure.gripper_flicker_rate": ("enabled", "action_source", "activation_threshold", "deadband", "flicker_window_steps", "minimum_dwell_steps"),
    "failure.collision_rate": ("enabled", "required"), "system.inference_latency": ("enabled",),
    "episode.length": ("enabled",), "system.control_frequency": ("enabled",),
}


def validate_metric_set(doc, path):
    header(doc, path, "metric_set", identified=True)
    mapping(doc, path, required=("schema_version", "kind", "id", "required", "plugins"))
    exact_type(doc["required"], list, f"{path}.required")
    plugins = doc["plugins"]
    if not isinstance(plugins, dict): raise ConfigSchemaError(f"{path}.plugins must be a mapping")
    unknown = sorted(set(plugins) - set(_PLUGIN_KEYS))
    if unknown: raise ConfigSchemaError(f"{path}.plugins contains unknown metric IDs: {', '.join(unknown)}")
    for metric_id, config in plugins.items():
        allowed = _PLUGIN_KEYS[metric_id]
        mapping(config, f"{path}.plugins.{metric_id}", required=allowed)
        exact_type(config["enabled"], bool, f"{path}.plugins.{metric_id}.enabled")
        if "action_source" in config: enum(config["action_source"], ("applied", "requested"), f"{path}.plugins.{metric_id}.action_source")
        for key in ("absolute_tolerance", "relative_tolerance", "norm_threshold", "activation_threshold", "deadband"):
            if key in config: number(config[key], f"{path}.plugins.{metric_id}.{key}")
        for key in ("minimum_consecutive_steps", "flicker_window_steps", "minimum_dwell_steps"):
            if key in config: exact_type(config[key], int, f"{path}.plugins.{metric_id}.{key}")
        if "required" in config: exact_type(config["required"], bool, f"{path}.plugins.{metric_id}.required")
    enabled = {key for key, value in plugins.items() if value["enabled"]}
    if any(item not in enabled for item in doc["required"]):
        raise ConfigSchemaError(f"{path}.required must reference enabled plugins")


def validate_protocol(doc, path):
    header(doc, path, "protocol", identified=True)
    mapping(doc, path, required=("schema_version", "kind", "id", "execution", "recording", "reproducibility"))
    execution = mapping(doc["execution"], f"{path}.execution", required=(
        "rollouts_per_task", "base_seed", "maximum_episode_steps", "action_chunks", "episode_errors", "unavailable_metrics"))
    for key in ("rollouts_per_task", "base_seed", "maximum_episode_steps"): exact_type(execution[key], int, f"{path}.execution.{key}")
    if execution["rollouts_per_task"] <= 0 or execution["maximum_episode_steps"] <= 0 or execution["base_seed"] < 0:
        raise ConfigSchemaError(f"{path}.execution counts must be positive and base_seed non-negative")
    chunks = mapping(execution["action_chunks"], f"{path}.execution.action_chunks", required=("mode", "replan_interval"))
    enum(chunks["mode"], ("receding_horizon", "open_loop_chunk", "fixed_replan_interval"), f"{path}.execution.action_chunks.mode")
    exact_type(chunks["replan_interval"], int, f"{path}.execution.action_chunks.replan_interval")
    enum(execution["episode_errors"], ("stop_run", "continue_task", "continue_run"), f"{path}.execution.episode_errors")
    enum(execution["unavailable_metrics"], ("allow_unavailable", "require_selected"), f"{path}.execution.unavailable_metrics")
    recording_keys = ("observations", "images", "proprioception", "predictions", "raw_policy_output", "evaluation_signals", "privileged_signals")
    recording = mapping(doc["recording"], f"{path}.recording", required=recording_keys)
    for key in recording_keys: exact_type(recording[key], bool, f"{path}.recording.{key}")
    repro = mapping(doc["reproducibility"], f"{path}.reproducibility", required=("reject_dirty_external_repositories", "require_checkpoint_identity"))
    for key in repro: exact_type(repro[key], bool, f"{path}.reproducibility.{key}")


def validate_registry(doc, path):
    header(doc, path, "resource_registry")
    mapping(doc, path, required=("schema_version", "kind", "checkpoints", "repositories"))
    if not isinstance(doc["checkpoints"], dict) or not doc["checkpoints"]:
        raise ConfigSchemaError(f"{path}.checkpoints must be a non-empty mapping")
    if not isinstance(doc["repositories"], dict) or not doc["repositories"]:
        raise ConfigSchemaError(f"{path}.repositories must be a non-empty mapping")
    for resource_id, entry in doc["checkpoints"].items():
        entry_path = f"{path}.checkpoints.{resource_id}"
        mapping(
            entry,
            entry_path,
            required=("repo_id", "revision", "expected_sha256"),
            optional=("files", "artifact", "method"),
        )
        non_empty_string(entry["repo_id"], f"{entry_path}.repo_id")
        for key in ("revision", "expected_sha256"):
            if entry[key] is not None: exact_type(entry[key], str, f"{entry_path}.{key}")
        if ("artifact" in entry) != ("method" in entry):
            raise ConfigSchemaError(f"{entry_path}.artifact and method must be declared together")
        if "files" in entry:
            for key in ("revision", "expected_sha256"):
                non_empty_string(entry[key], f"{entry_path}.{key}")
            revision = entry["revision"]
            if len(revision) != 40 or any(
                character not in "0123456789abcdef" for character in revision
            ):
                raise ConfigSchemaError(f"{entry_path}.revision must be a full commit digest")
            aggregate = entry["expected_sha256"]
            if len(aggregate) != 64 or any(
                character not in "0123456789abcdef" for character in aggregate
            ):
                raise ConfigSchemaError(f"{entry_path}.expected_sha256 must be a SHA-256 digest")
            files = entry["files"]
            exact_type(files, dict, f"{entry_path}.files")
            if not files:
                raise ConfigSchemaError(f"{entry_path}.files must not be empty")
            for name, item in files.items():
                file_path = f"{entry_path}.files.{name}"
                non_empty_string(name, f"{file_path}.name")
                mapping(item, file_path, required=("size", "sha256"))
                exact_type(item["size"], int, f"{file_path}.size")
                digest = item["sha256"]
                if item["size"] <= 0 or not isinstance(digest, str) or len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    raise ConfigSchemaError(f"{file_path} has invalid identity")
        if "artifact" not in entry:
            continue
        for key in ("revision", "expected_sha256"):
            non_empty_string(entry[key], f"{entry_path}.{key}")
        if len(entry["expected_sha256"]) != 64 or any(
            character not in "0123456789abcdef" for character in entry["expected_sha256"]
        ):
            raise ConfigSchemaError(f"{entry_path}.expected_sha256 must be a SHA-256 digest")
        if entry["method"].get("family") == "openvla_oft":
            artifact = mapping(
                entry["artifact"], f"{entry_path}.artifact",
                required=(
                    "form", "merge_status", "active_peft_adapter", "runtime_peft_modules",
                    "published_unmerged_adapter", "files", "parameter_counts", "byte_counts",
                ),
            )
            enum(artifact["form"], ("merged_backbone_with_auxiliary_components",), f"{entry_path}.artifact.form")
            enum(artifact["merge_status"], ("merged",), f"{entry_path}.artifact.merge_status")
            for key in ("active_peft_adapter", "runtime_peft_modules"):
                if artifact[key] is not False:
                    raise ConfigSchemaError(f"{entry_path}.artifact.{key} must be false")
            if artifact["published_unmerged_adapter"] is not True:
                raise ConfigSchemaError(f"{entry_path}.artifact.published_unmerged_adapter must be true")
            files = artifact["files"]
            exact_type(files, dict, f"{entry_path}.artifact.files")
            if not files:
                raise ConfigSchemaError(f"{entry_path}.artifact.files must not be empty")
            for name, item in files.items():
                non_empty_string(name, f"{entry_path}.artifact.files.name")
                mapping(item, f"{entry_path}.artifact.files.{name}", required=("size", "sha256"))
                exact_type(item["size"], int, f"{entry_path}.artifact.files.{name}.size")
                digest = item["sha256"]
                if item["size"] <= 0 or not isinstance(digest, str) or len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    raise ConfigSchemaError(f"{entry_path}.artifact.files.{name} has invalid identity")
            counts = mapping(artifact["parameter_counts"], f"{entry_path}.artifact.parameter_counts", required=(
                "merged_backbone", "lora_trainable", "lora_file_tensors", "action_head",
                "proprio_projector", "auxiliary", "complete_adaptation", "total_runtime",
            ))
            byte_counts = mapping(artifact["byte_counts"], f"{entry_path}.artifact.byte_counts", required=(
                "merged_backbone_weights", "lora_file", "auxiliary", "complete_adaptation",
            ))
            if any(type(value) is not int or value <= 0 for value in (*counts.values(), *byte_counts.values())):
                raise ConfigSchemaError(f"{entry_path}.artifact parameter and byte counts must be positive integers")
            method = mapping(entry["method"], f"{entry_path}.method", required=(
                "id", "version", "family", "acronym_expansion", "backbone_adaptation",
                "declared_base_model", "declared_base_revision", "artifact_form", "backbone_merge_status",
                "runtime_active_adapter", "parallel_decoding", "action_representation", "objective",
                "action_chunk_size", "action_dimension", "normalization", "image_inputs",
                "proprioception_dimension", "film", "diffusion", "quantization",
                "adaptation_suite", "dataset_identity", "training_step", "lora", "training_provenance",
            ))
            enum(method["acronym_expansion"], ("optimized_fine_tuning",), f"{entry_path}.method.acronym_expansion")
            enum(method["quantization"], ("none",), f"{entry_path}.method.quantization")
            if method["film"] is not False or method["diffusion"] is not False:
                raise ConfigSchemaError(f"{entry_path}.method must not enable FiLM or diffusion")
            exact_type(method["lora"], dict, f"{entry_path}.method.lora")
            exact_type(method["training_provenance"], dict, f"{entry_path}.method.training_provenance")
            continue
        artifact = mapping(
            entry["artifact"],
            f"{entry_path}.artifact",
            required=(
                "form", "merge_status", "active_peft_adapter", "runtime_peft_modules",
                "adapter_config", "adapter_recoverability", "files",
            ),
        )
        enum(artifact["form"], ("merged_full_weights",), f"{entry_path}.artifact.form")
        enum(artifact["merge_status"], ("merged",), f"{entry_path}.artifact.merge_status")
        for key in ("active_peft_adapter", "runtime_peft_modules"):
            exact_type(artifact[key], bool, f"{entry_path}.artifact.{key}")
            if artifact[key]:
                raise ConfigSchemaError(f"{entry_path}.artifact.{key} must be false for merged weights")
        enum(
            artifact["adapter_config"],
            ("not_present_in_published_artifact",),
            f"{entry_path}.artifact.adapter_config",
        )
        enum(
            artifact["adapter_recoverability"],
            ("not_recoverable_from_published_artifact",),
            f"{entry_path}.artifact.adapter_recoverability",
        )
        exact_type(artifact["files"], dict, f"{entry_path}.artifact.files")
        if not artifact["files"]:
            raise ConfigSchemaError(f"{entry_path}.artifact.files must not be empty")
        for name, item in artifact["files"].items():
            file_path = f"{entry_path}.artifact.files.{name}"
            non_empty_string(name, f"{file_path}.name")
            mapping(item, file_path, required=("size", "sha256"))
            exact_type(item["size"], int, f"{file_path}.size")
            if item["size"] <= 0:
                raise ConfigSchemaError(f"{file_path}.size must be positive")
            non_empty_string(item["sha256"], f"{file_path}.sha256")
            if len(item["sha256"]) != 64 or any(
                character not in "0123456789abcdef" for character in item["sha256"]
            ):
                raise ConfigSchemaError(f"{file_path}.sha256 must be a SHA-256 digest")
        method = mapping(
            entry["method"],
            f"{entry_path}.method",
            required=(
                "id", "version", "family", "declared_base_model", "declared_base_revision",
                "adaptation_suite", "quantization", "lora", "training_provenance",
            ),
        )
        for key in ("id", "version", "declared_base_model"):
            non_empty_string(method[key], f"{entry_path}.method.{key}")
        if method["declared_base_revision"] is not None:
            non_empty_string(method["declared_base_revision"], f"{entry_path}.method.declared_base_revision")
        enum(method["family"], ("lora",), f"{entry_path}.method.family")
        enum(method["adaptation_suite"], ("LIBERO-10",), f"{entry_path}.method.adaptation_suite")
        enum(method["quantization"], ("none",), f"{entry_path}.method.quantization")
        lora = mapping(
            method["lora"],
            f"{entry_path}.method.lora",
            required=(
                "rank", "alpha", "scaling", "dropout", "bias", "target_policy",
                "modules_to_save", "merge_procedure",
            ),
        )
        for key in ("rank", "alpha"):
            exact_type(lora[key], int, f"{entry_path}.method.lora.{key}")
        for key in ("scaling", "dropout"):
            number(lora[key], f"{entry_path}.method.lora.{key}")
        enum(lora["bias"], ("none",), f"{entry_path}.method.lora.bias")
        enum(lora["target_policy"], ("all-linear",), f"{entry_path}.method.lora.target_policy")
        if lora["modules_to_save"] is not None:
            raise ConfigSchemaError(f"{entry_path}.method.lora.modules_to_save must be null")
        enum(
            lora["merge_procedure"],
            ("merge_and_unload()+save_pretrained()",),
            f"{entry_path}.method.lora.merge_procedure",
        )
        exact_type(method["training_provenance"], dict, f"{entry_path}.method.training_provenance")
    for resource_id, entry in doc["repositories"].items():
        mapping(entry, f"{path}.repositories.{resource_id}", required=("path",))
        exact_type(entry["path"], str, f"{path}.repositories.{resource_id}.path")


def validate_local_profile(doc, path):
    header(doc, path, "local_profile", identified=True)
    mapping(
        doc,
        path,
        required=("schema_version", "kind", "id", "paths", "devices"),
        optional=("execution", "resources"),
    )
    paths = mapping(doc["paths"], f"{path}.paths", required=("checkpoint_root", "dataset_root", "runs_root"))
    for key, value in paths.items(): exact_type(value, str, f"{path}.paths.{key}")
    if not isinstance(doc["devices"], dict) or not doc["devices"]:
        raise ConfigSchemaError(f"{path}.devices must be a non-empty mapping")
    for key, value in doc["devices"].items(): exact_type(value, str, f"{path}.devices.{key}")
    if "resources" in doc:
        resources = mapping(doc["resources"], f"{path}.resources", required=("checkpoints",))
        checkpoints = resources["checkpoints"]
        exact_type(checkpoints, dict, f"{path}.resources.checkpoints")
        for resource_id, entry in checkpoints.items():
            non_empty_string(resource_id, f"{path}.resources.checkpoints.id")
            override = mapping(
                entry,
                f"{path}.resources.checkpoints.{resource_id}",
                required=("local_path",),
            )
            non_empty_string(
                override["local_path"],
                f"{path}.resources.checkpoints.{resource_id}.local_path",
            )
            if not Path(override["local_path"]).expanduser().is_absolute():
                raise ConfigSchemaError(
                    f"{path}.resources.checkpoints.{resource_id}.local_path must be absolute"
                )
    if "execution" in doc:
        execution = mapping(doc["execution"], f"{path}.execution", required=("libero",))
        libero = mapping(execution["libero"], f"{path}.execution.libero", required=("renderer",))
        renderer = mapping(libero["renderer"], f"{path}.execution.libero.renderer", required=("device_id",))
        exact_type(renderer["device_id"], int, f"{path}.execution.libero.renderer.device_id")
        if renderer["device_id"] < 0:
            raise ConfigSchemaError(f"{path}.execution.libero.renderer.device_id must be non-negative")


def validate_execution_profile(doc, path):
    header(doc, path, "execution_profile", identified=True)
    mapping(doc, path, required=("schema_version", "kind", "id", "execution"))
    execution = mapping(doc["execution"], f"{path}.execution", required=("libero",))
    libero = mapping(execution["libero"], f"{path}.execution.libero", required=("renderer",))
    renderer = mapping(
        libero["renderer"], f"{path}.execution.libero.renderer", required=("backend",), optional=("device_id",)
    )
    enum(renderer["backend"], ("egl", "glfw"), f"{path}.execution.libero.renderer.backend")
    if "device_id" in renderer and renderer["device_id"] is not None:
        exact_type(renderer["device_id"], int, f"{path}.execution.libero.renderer.device_id")
        if renderer["device_id"] < 0:
            raise ConfigSchemaError(f"{path}.execution.libero.renderer.device_id must be non-negative")
    if renderer["backend"] == "glfw" and "device_id" in renderer:
        raise ConfigSchemaError(f"{path}.execution.libero.renderer.device_id is applicable only to EGL")


def validate_artifacts(doc, path):
    header(doc, path, "artifact_store", typed=True)
    mapping(doc, path, required=("schema_version", "kind", "type", "settings"))
    if doc["type"] != "filesystem": raise ConfigSchemaError(f"{path}.type supports only 'filesystem'")
    settings = mapping(doc["settings"], f"{path}.settings", required=("root_resource",))
    exact_type(settings["root_resource"], str, f"{path}.settings.root_resource")


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
        "package": "openvla_quic.ovlab_provider", "api_name": "ovlab-quic-provider",
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
            "availability": "available", "path": "external/compound-peft",
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


VALIDATORS = {
    "experiment": validate_experiment, "benchmark": validate_benchmark, "policy": validate_policy,
    "action_interface": validate_action_interface, "metric_set": validate_metric_set, "protocol": validate_protocol,
    "resource_registry": validate_registry, "local_profile": validate_local_profile,
    "execution_profile": validate_execution_profile, "artifact_store": validate_artifacts,
    "quic_policy_descriptor": validate_quic_policy_descriptor,
}


def validate(doc: dict[str, Any], path: str, expected_kind: str | None = None) -> None:
    kind = doc.get("kind")
    if expected_kind is not None and kind != expected_kind:
        raise ConfigSchemaError(f"{path}.kind must equal {expected_kind!r}, got {kind!r}")
    try: validator = VALIDATORS[kind]
    except (KeyError, TypeError) as exc: raise ConfigSchemaError(f"{path}.kind is unknown: {kind!r}") from exc
    validator(doc, path)
