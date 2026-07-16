"""Exact kind-specific schemas for OVLAB configuration version 0.1.0."""

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
    mapping(doc, path, required=("schema_version", "kind", "experiment", "components", "resources"))
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


def validate_benchmark(doc, path):
    header(doc, path, "benchmark", typed=True)
    mapping(doc, path, required=("schema_version", "kind", "type", "settings"), optional=("extends",))
    if doc["type"] != "libero": raise ConfigSchemaError(f"{path}.type supports only 'libero'")
    settings = mapping(doc["settings"], f"{path}.settings", required=(
        "suite", "task_indices", "observation", "initialization", "rendering", "action", "privileged_signals"))
    enum(settings["suite"], ("libero_spatial", "libero_object", "libero_goal", "libero_10"), f"{path}.settings.suite")
    if settings["task_indices"] != "all":
        exact_type(settings["task_indices"], list, f"{path}.settings.task_indices")
        if any(type(item) is not int or item < 0 for item in settings["task_indices"]):
            raise ConfigSchemaError(f"{path}.settings.task_indices must be 'all' or non-negative integers")
    obs = mapping(settings["observation"], f"{path}.settings.observation", required=(
        "profile", "cameras", "width", "height", "color_space", "dtype"))
    enum(obs["profile"], ("primary_rgb",), f"{path}.settings.observation.profile")
    cameras = mapping(obs["cameras"], f"{path}.settings.observation.cameras", required=("primary",))
    primary = mapping(cameras["primary"], f"{path}.settings.observation.cameras.primary", required=("native_name", "canonical_name"))
    for key in primary: exact_type(primary[key], str, f"{path}.settings.observation.cameras.primary.{key}")
    for key in ("width", "height"):
        exact_type(obs[key], int, f"{path}.settings.observation.{key}")
        if obs[key] <= 0: raise ConfigSchemaError(f"{path}.settings.observation.{key} must be positive")
    enum(obs["color_space"], ("rgb",), f"{path}.settings.observation.color_space")
    enum(obs["dtype"], ("uint8",), f"{path}.settings.observation.dtype")
    init = mapping(settings["initialization"], f"{path}.settings.initialization", required=("state_selection", "settling_steps"))
    enum(init["state_selection"], ("rollout_index", "seeded"), f"{path}.settings.initialization.state_selection")
    exact_type(init["settling_steps"], int, f"{path}.settings.initialization.settling_steps")
    if init["settling_steps"] < 0: raise ConfigSchemaError(f"{path}.settings.initialization.settling_steps must be non-negative")
    rendering = mapping(settings["rendering"], f"{path}.settings.rendering", required=("mode", "gpu_resource"))
    enum(rendering["mode"], ("headless",), f"{path}.settings.rendering.mode")
    exact_type(rendering["gpu_resource"], str, f"{path}.settings.rendering.gpu_resource")
    action = mapping(settings["action"], f"{path}.settings.action", required=("interface_ref",))
    exact_type(action["interface_ref"], str, f"{path}.settings.action.interface_ref")
    signals = mapping(settings["privileged_signals"], f"{path}.settings.privileged_signals", required=("enabled",))
    exact_type(signals["enabled"], bool, f"{path}.settings.privileged_signals.enabled")


def validate_policy(doc, path):
    header(doc, path, "policy", typed=True)
    mapping(doc, path, required=("schema_version", "kind", "type", "settings"), optional=("extends",))
    if doc["type"] != "openvla_vanilla": raise ConfigSchemaError(f"{path}.type supports only 'openvla_vanilla'")
    settings = mapping(doc["settings"], f"{path}.settings", required=(
        "checkpoint_id", "processor_id", "unnorm_key", "input", "runtime", "action", "raw_output"))
    for key in ("checkpoint_id", "processor_id", "unnorm_key"):
        exact_type(settings[key], str, f"{path}.settings.{key}")
    input_ = mapping(settings["input"], f"{path}.settings.input", required=("camera",))
    exact_type(input_["camera"], str, f"{path}.settings.input.camera")
    runtime = mapping(settings["runtime"], f"{path}.settings.runtime", required=(
        "device_resource", "dtype", "attention_implementation", "local_files_only", "trust_remote_code",
        "deterministic", "synchronize_inference"))
    exact_type(runtime["device_resource"], str, f"{path}.settings.runtime.device_resource")
    enum(runtime["dtype"], ("bfloat16", "float16", "float32"), f"{path}.settings.runtime.dtype")
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
    enum(doc["gripper_convention"], ("closed_positive",), f"{path}.gripper_convention")
    enum(doc["representation"], ("delta_pose",), f"{path}.representation")
    enum(doc["rotation_representation"], ("axis_angle",), f"{path}.rotation_representation")
    enum(doc["dtype"], ("float32",), f"{path}.dtype")
    non_empty_string(doc["units"], f"{path}.units")
    number(doc["control_frequency_hz"], f"{path}.control_frequency_hz")


_PLUGIN_KEYS = {
    "task.success": ("enabled",), "task.success_rate": ("enabled",),
    "action.variance": ("enabled", "action_source"), "action.smoothness_1": ("enabled", "action_source"),
    "action.smoothness_2": ("enabled", "action_source"), "failure.invalid_prediction_rate": ("enabled",),
    "failure.action_modification_rate": ("enabled", "absolute_tolerance", "relative_tolerance"),
    "failure.repeated_no_op_rate": ("enabled", "action_source", "norm_threshold", "minimum_consecutive_steps"),
    "failure.gripper_flicker_rate": ("enabled", "action_source", "activation_threshold", "deadband", "flicker_window_steps", "minimum_dwell_steps"),
    "failure.collision_rate": ("enabled", "required"), "system.inference_latency": ("enabled",),
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
        mapping(entry, f"{path}.checkpoints.{resource_id}", required=("relative_path", "expected_revision", "expected_sha256"))
        non_empty_string(entry["relative_path"], f"{path}.checkpoints.{resource_id}.relative_path")
        for key in ("expected_revision", "expected_sha256"):
            if entry[key] is not None: exact_type(entry[key], str, f"{path}.checkpoints.{resource_id}.{key}")
    for resource_id, entry in doc["repositories"].items():
        mapping(entry, f"{path}.repositories.{resource_id}", required=("path",))
        exact_type(entry["path"], str, f"{path}.repositories.{resource_id}.path")


def validate_local_profile(doc, path):
    header(doc, path, "local_profile", identified=True)
    mapping(doc, path, required=("schema_version", "kind", "id", "paths", "devices"))
    paths = mapping(doc["paths"], f"{path}.paths", required=("checkpoint_root", "dataset_root", "runs_root"))
    for key, value in paths.items(): exact_type(value, str, f"{path}.paths.{key}")
    if not isinstance(doc["devices"], dict) or not doc["devices"]:
        raise ConfigSchemaError(f"{path}.devices must be a non-empty mapping")
    for key, value in doc["devices"].items(): exact_type(value, str, f"{path}.devices.{key}")


def validate_artifacts(doc, path):
    header(doc, path, "artifact_store", typed=True)
    mapping(doc, path, required=("schema_version", "kind", "type", "settings"))
    if doc["type"] != "filesystem": raise ConfigSchemaError(f"{path}.type supports only 'filesystem'")
    settings = mapping(doc["settings"], f"{path}.settings", required=("root_resource",))
    exact_type(settings["root_resource"], str, f"{path}.settings.root_resource")


VALIDATORS = {
    "experiment": validate_experiment, "benchmark": validate_benchmark, "policy": validate_policy,
    "action_interface": validate_action_interface, "metric_set": validate_metric_set, "protocol": validate_protocol,
    "resource_registry": validate_registry, "local_profile": validate_local_profile, "artifact_store": validate_artifacts,
}


def validate(doc: dict[str, Any], path: str, expected_kind: str | None = None) -> None:
    kind = doc.get("kind")
    if expected_kind is not None and kind != expected_kind:
        raise ConfigSchemaError(f"{path}.kind must equal {expected_kind!r}, got {kind!r}")
    try: validator = VALIDATORS[kind]
    except (KeyError, TypeError) as exc: raise ConfigSchemaError(f"{path}.kind is unknown: {kind!r}") from exc
    validator(doc, path)
