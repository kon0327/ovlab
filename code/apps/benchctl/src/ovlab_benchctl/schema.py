"""Exact kind-specific schemas for OVLAB configuration version 0.1.0."""

from pathlib import Path
import hashlib
import re
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
        "benchmark", "policy", "metrics", "protocol", "artifacts"))
    resources = mapping(doc["resources"], f"{path}.resources", required=("registry",))
    for key, value in components.items(): exact_type(value, str, f"{path}.components.{key}")
    exact_type(resources["registry"], str, f"{path}.resources.registry")
    if "deployment" in doc:
        deployment = mapping(
            doc["deployment"],
            f"{path}.deployment",
            required=("profile", "renderer"),
            optional=("offline",),
        )
        non_empty_string(deployment["profile"], f"{path}.deployment.profile")
        enum(deployment["renderer"], ("egl", "glfw"), f"{path}.deployment.renderer")
        if "offline" in deployment:
            exact_type(deployment["offline"], bool, f"{path}.deployment.offline")
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
        action = mapping(settings["action"], f"{path}.settings.action", required=("action_interface",))
        non_empty_string(action["action_interface"], f"{path}.settings.action.action_interface")
        raw = mapping(settings["raw_output"], f"{path}.settings.raw_output", required=("enabled",))
        exact_type(raw["enabled"], bool, f"{path}.settings.raw_output.enabled")
        return
    from .policy_plugins import plugin_for_type, call_plugin
    call_plugin(plugin_for_type(doc["type"]), "validate_policy", doc, path)


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
    if not isinstance(doc["repositories"], dict):
        raise ConfigSchemaError(f"{path}.repositories must be a mapping")
    for resource_id, entry in doc["checkpoints"].items():
        entry_path = f"{path}.checkpoints.{resource_id}"
        if not isinstance(resource_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", resource_id) is None:
            raise ConfigSchemaError(f"{entry_path} requires a path-safe checkpoint ID")
        mapping(
            entry,
            entry_path,
            required=("repo_id",),
            optional=("revision", "expected_sha256", "files", "artifact", "method"),
        )
        non_empty_string(entry["repo_id"], f"{entry_path}.repo_id")
        for key in ("revision", "expected_sha256"):
            if key in entry: exact_type(entry[key], str, f"{entry_path}.{key}")
        revision = entry.get("revision", "local/main")
        if revision in {"local/main", "remote/main"}:
            if any(key in entry for key in ("expected_sha256", "files", "artifact", "method")):
                raise ConfigSchemaError(f"{entry_path} moving revision cannot declare pinned hashes or artifact metadata")
            continue
        if not isinstance(revision, str) or len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ConfigSchemaError(f"{entry_path}.revision must be local/main, remote/main, or a full commit digest")
        if "expected_sha256" not in entry:
            raise ConfigSchemaError(f"{entry_path}.expected_sha256 is required for a pinned revision")
        if ("artifact" in entry) != ("method" in entry):
            raise ConfigSchemaError(f"{entry_path}.artifact and method must be declared together")
        aggregate = entry["expected_sha256"]
        if len(aggregate) != 64 or any(character not in "0123456789abcdef" for character in aggregate):
            raise ConfigSchemaError(f"{entry_path}.expected_sha256 must be a SHA-256 digest")
        if "artifact" in entry:
            exact_type(entry["artifact"], dict, f"{entry_path}.artifact")
            exact_type(entry["method"], dict, f"{entry_path}.method")
        files = entry.get("files", entry.get("artifact", {}).get("files"))
        if not isinstance(files, dict) or not files:
            raise ConfigSchemaError(f"{entry_path} requires a non-empty file manifest for a pinned checkpoint")
        manifest = []
        for name, item in files.items():
            file_path = f"{entry_path}.files.{name}"
            non_empty_string(name, f"{file_path}.name")
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ConfigSchemaError(f"{file_path} must be a relative checkpoint file path")
            mapping(item, file_path, required=("size", "sha256"))
            exact_type(item["size"], int, f"{file_path}.size")
            digest = item["sha256"]
            if item["size"] < 0 or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ConfigSchemaError(f"{file_path} has invalid identity")
            manifest.append(f"{name} {item['size']} {digest}\n")
        if hashlib.sha256("".join(manifest).encode()).hexdigest() != aggregate:
            raise ConfigSchemaError(f"{entry_path} aggregate SHA-256 differs from its file manifest")
        if "artifact" not in entry:
            continue
        for key in ("revision", "expected_sha256"):
            non_empty_string(entry[key], f"{entry_path}.{key}")
        if len(entry["expected_sha256"]) != 64 or any(
            character not in "0123456789abcdef" for character in entry["expected_sha256"]
        ):
            raise ConfigSchemaError(f"{entry_path}.expected_sha256 must be a SHA-256 digest")
        from .policy_plugins import registered_plugins, call_plugin
        exact_type(entry["method"], dict, f"{entry_path}.method")
        exact_type(entry["artifact"], dict, f"{entry_path}.artifact")
        family = entry["method"].get("family")
        for plugin in registered_plugins():
            if (entry["artifact"].get("form") in plugin.descriptor.get("artifact_forms", []) or plugin.descriptor["family"] == family) and plugin.descriptor.get("entrypoint"):
                call_plugin(plugin, "validate_checkpoint", entry, entry_path)
                break
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




def validate_deployment(doc, path):
    import re
    mapping(doc, path, required=("schema_version", "kind", "id", "components", "resources"),
            optional=("image", "build", "runtime", "service", "input"))
    enum(doc["schema_version"], ("ovlab.deployment/v1",), f"{path}.schema_version")
    enum(doc["kind"], ("deployment",), f"{path}.kind")
    non_empty_string(doc["id"], f"{path}.id")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", doc["id"]) is None:
        raise ConfigSchemaError(f"{path}.id must be a lowercase container-safe identifier")
    for section, key in (("components", "policy"), ("resources", "registry")):
        values = mapping(doc[section], f"{path}.{section}", required=(key,))
        non_empty_string(values[key], f"{path}.{section}.{key}")
    if "image" in doc:
        non_empty_string(doc["image"], f"{path}.image")
        if any(char.isspace() for char in doc["image"]) or doc["image"].startswith("-"):
            raise ConfigSchemaError(f"{path}.image must be a Docker image reference")
    if "build" in doc:
        exact_type(doc["build"], bool, f"{path}.build")
    runtime = mapping(doc.get("runtime", {}), f"{path}.runtime", optional=("quantization", "gpu"))
    if "quantization" in runtime:
        enum(runtime["quantization"], ("none", "8bit", "4bit"), f"{path}.runtime.quantization")
    if "gpu" in runtime and runtime["gpu"] is not None:
        non_empty_string(runtime["gpu"], f"{path}.runtime.gpu")
    service = mapping(doc.get("service", {}), f"{path}.service",
                      optional=("host", "port", "container_port", "restart", "readiness_timeout_s"))
    if "host" in service:
        import ipaddress
        exact_type(service["host"], str, f"{path}.service.host")
        try:
            if ipaddress.ip_address(service["host"]).version != 4:
                raise ValueError("IPv4 required")
        except (ValueError, TypeError) as exc:
            raise ConfigSchemaError(f"{path}.service.host must be an IPv4 address") from exc
    for key in ("port", "container_port", "readiness_timeout_s"):
        if key not in service or (key == "port" and service[key] is None):
            continue
        exact_type(service[key], int, f"{path}.service.{key}")
        if not 1 <= service[key] <= (65535 if key != "readiness_timeout_s" else 86400):
            raise ConfigSchemaError(f"{path}.service.{key} is out of range")
    if "restart" in service:
        enum(service["restart"], ("no", "on-failure", "always", "unless-stopped"), f"{path}.service.restart")
    inputs = mapping(doc.get("input", {}), f"{path}.input", optional=("height", "width"))
    for key, value in inputs.items():
        exact_type(value, int, f"{path}.input.{key}")
        if value <= 0:
            raise ConfigSchemaError(f"{path}.input.{key} must be positive")


VALIDATORS = {
    "deployment": validate_deployment,
    "experiment": validate_experiment, "benchmark": validate_benchmark, "policy": validate_policy,
    "action_interface": validate_action_interface, "metric_set": validate_metric_set, "protocol": validate_protocol,
    "resource_registry": validate_registry, "local_profile": validate_local_profile,
    "execution_profile": validate_execution_profile, "artifact_store": validate_artifacts,
}


def validate(doc: dict[str, Any], path: str, expected_kind: str | None = None, *, repository_root=None) -> None:
    kind = doc.get("kind")
    if expected_kind is not None and kind != expected_kind:
        raise ConfigSchemaError(f"{path}.kind must equal {expected_kind!r}, got {kind!r}")
    from .policy_plugins import registry_scope, plugin_for_document, call_plugin
    with registry_scope(repository_root):
        validator = VALIDATORS.get(kind) if isinstance(kind, str) else None
        if validator is not None:
            validator(doc, path)
            return
        plugin = plugin_for_document(kind, repository_root)
        if plugin is None: raise ConfigSchemaError(f"{path}.kind is unknown: {kind!r}")
        call_plugin(plugin, "validate_document", doc, path)
