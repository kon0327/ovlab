from typing import Any
from ovlab_benchmarks import BenchmarkPluginValidationError as ConfigSchemaError
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

def validate_benchmark(doc, path):
    header(doc, path, "benchmark", typed=True)
    mapping(doc, path, required=("schema_version", "kind", "type", "settings"), optional=("extends", "provider"))
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

def validate_execution_profile(doc, path):
    header(doc, path, "execution_profile", identified=True)
    mapping(doc, path, required=("schema_version", "kind", "id", "execution"), optional=("provider",))
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
