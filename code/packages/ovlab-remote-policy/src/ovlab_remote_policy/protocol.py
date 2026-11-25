"""Wire protocol, strict schemas, and OVLAB contract serialization."""

from __future__ import annotations

import base64
import json
import math
import socket
import struct
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np

from ovlab_core.contracts import (
    ActionRepresentation,
    ActionSpec,
    ColorSpace,
    GripperConvention,
    ImageObservation,
    ImageEncoding,
    ImageObservationSpec,
    Instruction,
    InstructionId,
    InstructionSource,
    EpisodeContext,
    EpisodeId,
    ObservationRequirements,
    PolicyCapabilities,
    PolicyObservation,
    ProprioceptiveObservation,
    ProprioceptiveObservationSpec,
    PredictionId,
    PredictionValidity,
    RunContext,
    RunId,
    RotationRepresentation,
    StepId,
    TaskId,
)
from ovlab_remote_policy.errors import RemotePolicyProtocolError

PROTOCOL_VERSION = "ovlab-policy-rpc/2.0.0"
HEADER_SIZE = 4
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 1024 * 1024
MAX_PROPRIO_BYTES = 64 * 1024

OPERATIONS = frozenset(
    {"initialize", "health", "reset_episode", "predict", "end_episode", "close"}
)


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RemotePolicyProtocolError(f"{where} must be an object")
    return value


def require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    where: str,
) -> None:
    optional = optional or set()
    present = set(value)
    missing = required - present
    unexpected = present - required - optional
    if missing:
        raise RemotePolicyProtocolError(f"{where} is missing fields: {sorted(missing)}")
    if unexpected:
        raise RemotePolicyProtocolError(f"{where} contains forbidden fields: {sorted(unexpected)}")


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise RemotePolicyProtocolError("non-finite floats are not valid protocol metadata")
        return value
    raise RemotePolicyProtocolError(f"unsupported protocol value type: {type(value).__name__}")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RemotePolicyProtocolError(f"message is not JSON serializable: {exc}") from exc


def encode_frame(value: Mapping[str, Any], *, max_frame_bytes: int = MAX_FRAME_BYTES) -> bytes:
    payload = canonical_json_bytes(value)
    if not payload:
        raise RemotePolicyProtocolError("empty protocol payload")
    if len(payload) > max_frame_bytes:
        raise RemotePolicyProtocolError(
            f"protocol frame is {len(payload)} bytes; limit is {max_frame_bytes}"
        )
    return struct.pack("!I", len(payload)) + payload


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RemotePolicyProtocolError("peer closed the socket during a protocol frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket, *, max_frame_bytes: int = MAX_FRAME_BYTES) -> dict[str, Any]:
    header = recv_exact(sock, HEADER_SIZE)
    (size,) = struct.unpack("!I", header)
    if size == 0:
        raise RemotePolicyProtocolError("zero-length protocol frame")
    if size > max_frame_bytes:
        raise RemotePolicyProtocolError(f"protocol frame declares {size} bytes; limit is {max_frame_bytes}")
    raw = recv_exact(sock, size)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemotePolicyProtocolError(f"invalid JSON protocol frame: {exc}") from exc
    if not isinstance(value, dict):
        raise RemotePolicyProtocolError("protocol frame root must be an object")
    return value


def send_frame(sock: socket.socket, value: Mapping[str, Any]) -> None:
    sock.sendall(encode_frame(value))


def validate_request_envelope(value: Any) -> dict[str, Any]:
    envelope = dict(_require_mapping(value, "request"))
    require_exact_keys(
        envelope,
        required={"protocol_version", "request_id", "operation", "payload"},
        where="request",
    )
    if envelope["protocol_version"] != PROTOCOL_VERSION:
        raise RemotePolicyProtocolError(
            f"unsupported protocol version {envelope['protocol_version']!r}; expected {PROTOCOL_VERSION!r}"
        )
    request_id = envelope["request_id"]
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise RemotePolicyProtocolError("request_id must be a non-empty string of at most 128 characters")
    operation = envelope["operation"]
    if operation not in OPERATIONS:
        raise RemotePolicyProtocolError(f"unsupported operation: {operation!r}")
    envelope["payload"] = dict(_require_mapping(envelope["payload"], "request.payload"))
    return envelope


def validate_response_envelope(value: Any, *, request_id: str) -> dict[str, Any]:
    envelope = dict(_require_mapping(value, "response"))
    require_exact_keys(
        envelope,
        required={"protocol_version", "request_id", "status", "payload"},
        optional={"error"},
        where="response",
    )
    if envelope["protocol_version"] != PROTOCOL_VERSION:
        raise RemotePolicyProtocolError(
            f"response protocol version {envelope['protocol_version']!r} is incompatible"
        )
    if envelope["request_id"] != request_id:
        raise RemotePolicyProtocolError(
            f"stale response for {envelope['request_id']!r}; expected {request_id!r}"
        )
    if envelope["status"] not in {"ok", "error"}:
        raise RemotePolicyProtocolError("response status must be 'ok' or 'error'")
    envelope["payload"] = dict(_require_mapping(envelope["payload"], "response.payload"))
    if envelope["status"] == "ok" and "error" in envelope:
        raise RemotePolicyProtocolError("successful response must not contain an error")
    if envelope["status"] == "error":
        error = _require_mapping(envelope.get("error"), "response.error")
        require_exact_keys(error, required={"code", "message"}, where="response.error")
        if not isinstance(error["code"], str) or not isinstance(error["message"], str):
            raise RemotePolicyProtocolError("response error code and message must be strings")
    return envelope


def make_request(request_id: str, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return validate_request_envelope(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "operation": operation,
            "payload": dict(payload),
        }
    )


def make_success(request_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": "ok",
        "payload": _plain(payload),
    }


def make_error(request_id: str, code: str, message: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": "error",
        "payload": {},
        "error": {"code": code, "message": message},
    }


def action_spec_to_wire(spec: ActionSpec) -> dict[str, Any]:
    return {
        "dimension": spec.dimension,
        "representation": spec.representation.value,
        "translation_indices": list(spec.translation_indices),
        "rotation_indices": list(spec.rotation_indices),
        "gripper_indices": list(spec.gripper_indices),
        "rotation_representation": spec.rotation_representation.value,
        "gripper_convention": spec.gripper_convention.value,
        "units": list(spec.units),
        "minimum": None if spec.minimum is None else spec.minimum.tolist(),
        "maximum": None if spec.maximum is None else spec.maximum.tolist(),
        "dtype": spec.dtype,
        "control_frequency_hz": spec.control_frequency_hz,
        "metadata": _plain(spec.metadata),
    }


def action_spec_from_wire(value: Any) -> ActionSpec:
    data = dict(_require_mapping(value, "action_spec"))
    require_exact_keys(
        data,
        required={
            "dimension", "representation", "translation_indices", "rotation_indices", "gripper_indices",
            "rotation_representation", "gripper_convention", "units", "minimum", "maximum", "dtype",
            "control_frequency_hz", "metadata",
        },
        where="action_spec",
    )
    return ActionSpec(
        dimension=data["dimension"],
        representation=ActionRepresentation(data["representation"]),
        translation_indices=tuple(data["translation_indices"]),
        rotation_indices=tuple(data["rotation_indices"]),
        gripper_indices=tuple(data["gripper_indices"]),
        rotation_representation=RotationRepresentation(data["rotation_representation"]),
        gripper_convention=GripperConvention(data["gripper_convention"]),
        units=tuple(data["units"]),
        minimum=None if data["minimum"] is None else np.asarray(data["minimum"], dtype=_numeric_dtype(data["dtype"])),
        maximum=None if data["maximum"] is None else np.asarray(data["maximum"], dtype=_numeric_dtype(data["dtype"])),
        dtype=data["dtype"],
        control_frequency_hz=data["control_frequency_hz"],
        metadata=data["metadata"],
    )


def image_spec_to_wire(spec: ImageObservationSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "shapes": [list(shape) for shape in spec.shapes],
        "dtype": spec.dtype,
        "encodings": [encoding.value for encoding in spec.encodings],
        "color_spaces": [color.value for color in spec.color_spaces],
        "required": spec.required,
        "minimum_count": spec.minimum_count,
        "maximum_count": spec.maximum_count,
        "metadata": _plain(spec.metadata),
    }


def image_spec_from_wire(value: Any) -> ImageObservationSpec:
    data = dict(_require_mapping(value, "image_spec"))
    require_exact_keys(
        data,
        required={"name", "shapes", "dtype", "encodings", "color_spaces", "required", "minimum_count", "maximum_count", "metadata"},
        where="image_spec",
    )
    return ImageObservationSpec(
        name=data["name"],
        shapes=tuple(tuple(shape) for shape in data["shapes"]),
        dtype=data["dtype"],
        encodings=tuple(ImageEncoding(item) for item in data["encodings"]),
        color_spaces=tuple(ColorSpace(item) for item in data["color_spaces"]),
        required=data["required"],
        minimum_count=data["minimum_count"],
        maximum_count=data["maximum_count"],
        metadata=data["metadata"],
    )


def proprio_spec_to_wire(spec: ProprioceptiveObservationSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "shapes": [list(shape) for shape in spec.shapes],
        "dtype": spec.dtype,
        "units": list(spec.units),
        "required": spec.required,
        "metadata": _plain(spec.metadata),
    }


def proprio_spec_from_wire(value: Any) -> ProprioceptiveObservationSpec:
    data = dict(_require_mapping(value, "proprioception_spec"))
    require_exact_keys(
        data,
        required={"name", "shapes", "dtype", "units", "required", "metadata"},
        where="proprioception_spec",
    )
    return ProprioceptiveObservationSpec(
        name=data["name"],
        shapes=tuple(tuple(shape) for shape in data["shapes"]),
        dtype=data["dtype"],
        units=tuple(data["units"]),
        required=data["required"],
        metadata=data["metadata"],
    )


def capabilities_to_wire(capabilities: PolicyCapabilities) -> dict[str, Any]:
    requirements = capabilities.observation_requirements
    return {
        "component_name": capabilities.component_name,
        "component_version": capabilities.component_version,
        "contract_version": capabilities.contract_version,
        "observation_requirements": {
            "images": [image_spec_to_wire(spec) for spec in requirements.images],
            "proprioception": [proprio_spec_to_wire(spec) for spec in requirements.proprioception],
            "minimum_image_count": requirements.minimum_image_count,
            "maximum_image_count": requirements.maximum_image_count,
            "minimum_proprioception_count": requirements.minimum_proprioception_count,
            "maximum_proprioception_count": requirements.maximum_proprioception_count,
            "metadata": _plain(requirements.metadata),
        },
        "output_action_spec": action_spec_to_wire(capabilities.output_action_spec),
        "supports_single_action": capabilities.supports_single_action,
        "supports_action_chunks": capabilities.supports_action_chunks,
        "minimum_action_horizon": capabilities.minimum_action_horizon,
        "maximum_action_horizon": capabilities.maximum_action_horizon,
        "supports_dynamic_instructions": capabilities.supports_dynamic_instructions,
        "supports_deterministic_reset": capabilities.supports_deterministic_reset,
        "exposes_raw_policy_output": capabilities.exposes_raw_policy_output,
        "metadata": _plain(capabilities.metadata),
    }


def capabilities_from_wire(value: Any) -> PolicyCapabilities:
    data = dict(_require_mapping(value, "capabilities"))
    require_exact_keys(
        data,
        required={
            "component_name", "component_version", "contract_version", "observation_requirements", "output_action_spec",
            "supports_single_action", "supports_action_chunks", "minimum_action_horizon", "maximum_action_horizon",
            "supports_dynamic_instructions", "supports_deterministic_reset", "exposes_raw_policy_output", "metadata",
        },
        where="capabilities",
    )
    requirements = dict(_require_mapping(data["observation_requirements"], "observation_requirements"))
    require_exact_keys(
        requirements,
        required={
            "images", "proprioception", "minimum_image_count", "maximum_image_count",
            "minimum_proprioception_count", "maximum_proprioception_count", "metadata",
        },
        where="observation_requirements",
    )
    return PolicyCapabilities(
        component_name=data["component_name"],
        component_version=data["component_version"],
        contract_version=data["contract_version"],
        observation_requirements=ObservationRequirements(
            images=tuple(image_spec_from_wire(item) for item in requirements["images"]),
            proprioception=tuple(proprio_spec_from_wire(item) for item in requirements["proprioception"]),
            minimum_image_count=requirements["minimum_image_count"],
            maximum_image_count=requirements["maximum_image_count"],
            minimum_proprioception_count=requirements["minimum_proprioception_count"],
            maximum_proprioception_count=requirements["maximum_proprioception_count"],
            metadata=requirements["metadata"],
        ),
        output_action_spec=action_spec_from_wire(data["output_action_spec"]),
        supports_single_action=data["supports_single_action"],
        supports_action_chunks=data["supports_action_chunks"],
        minimum_action_horizon=data["minimum_action_horizon"],
        maximum_action_horizon=data["maximum_action_horizon"],
        supports_dynamic_instructions=data["supports_dynamic_instructions"],
        supports_deterministic_reset=data["supports_deterministic_reset"],
        exposes_raw_policy_output=data["exposes_raw_policy_output"],
        metadata=data["metadata"],
    )


def instruction_to_wire(instruction: Instruction) -> dict[str, Any]:
    if instruction.metadata:
        raise RemotePolicyProtocolError("instruction metadata is forbidden by the remote prediction schema")
    if instruction.source is not InstructionSource.BENCHMARK:
        raise RemotePolicyProtocolError("remote prediction requires the authoritative benchmark instruction")
    return {
        "instruction_id": str(instruction.instruction_id),
        "text": instruction.text,
        "timestamp_ns": instruction.timestamp_ns,
        "source": instruction.source,
        "supersedes": None if instruction.supersedes is None else str(instruction.supersedes),
    }


def instruction_from_wire(value: Any) -> Instruction:
    data = dict(_require_mapping(value, "instruction"))
    require_exact_keys(
        data,
        required={"instruction_id", "text", "timestamp_ns", "source", "supersedes"},
        where="instruction",
    )
    source = InstructionSource(data["source"])
    if source is not InstructionSource.BENCHMARK:
        raise RemotePolicyProtocolError("remote prediction requires the authoritative benchmark instruction")
    return Instruction(
        instruction_id=InstructionId(data["instruction_id"]),
        text=data["text"],
        timestamp_ns=data["timestamp_ns"],
        source=source,
        supersedes=None if data["supersedes"] is None else InstructionId(data["supersedes"]),
    )


def observation_to_predict_payload(observation: PolicyObservation, *, episode_id: str) -> dict[str, Any]:
    """Transfer named policy-visible modalities without model camera conventions."""
    if observation.metadata:
        raise RemotePolicyProtocolError("observation metadata is forbidden by the remote prediction schema")
    names = [item.name for item in observation.images + observation.proprioception]
    if len(names) != len(set(names)):
        raise RemotePolicyProtocolError("remote prediction observation names must be unique")
    return {
        "episode_id": episode_id, "step_id": str(observation.step_id),
        "timestamp_ns": observation.timestamp_ns,
        "instruction": instruction_to_wire(observation.instruction),
        "images": [_image_to_wire(item) for item in observation.images],
        "proprioception": [_proprioception_to_wire(item) for item in observation.proprioception],
    }


def _numeric_dtype(value: Any) -> np.dtype:
    if not isinstance(value, str):
        raise RemotePolicyProtocolError("observation dtype must be a numeric dtype string")
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise RemotePolicyProtocolError("invalid observation dtype") from exc
    if dtype.kind not in "biuf" or dtype.itemsize > 8:
        raise RemotePolicyProtocolError("observation dtype must be a bounded real numeric type")
    return dtype


def _array_to_wire(array: np.ndarray, limit: int, where: str) -> dict[str, Any]:
    dtype = _numeric_dtype(array.dtype.str)
    if array.nbytes > limit:
        raise RemotePolicyProtocolError(f"{where} is {array.nbytes} bytes; limit is {limit}")
    return {"shape": list(array.shape), "dtype": dtype.str,
            "data_b64": base64.b64encode(array.tobytes(order="C")).decode("ascii")}


def _image_to_wire(image: ImageObservation) -> dict[str, Any]:
    return {"name": image.name, "camera_name": image.camera_name, "timestamp_ns": image.timestamp_ns,
            "encoding": image.encoding.value, "color_space": image.color_space.value,
            **_array_to_wire(image.data, MAX_IMAGE_BYTES, "image")}


def _proprioception_to_wire(item: ProprioceptiveObservation) -> dict[str, Any]:
    return {"name": item.name, "timestamp_ns": item.timestamp_ns, "units": list(item.units),
            **_array_to_wire(item.values, MAX_PROPRIO_BYTES, "proprioception")}


def predict_payload_to_observation(value: Any) -> tuple[str, PolicyObservation]:
    data = dict(_require_mapping(value, "predict payload"))
    require_exact_keys(data, required={"episode_id", "step_id", "timestamp_ns", "instruction", "images", "proprioception"}, where="predict payload")
    if not isinstance(data["episode_id"], str) or not data["episode_id"]:
        raise RemotePolicyProtocolError("episode_id must be a non-empty string")
    for key in ("images", "proprioception"):
        if not isinstance(data[key], list):
            raise RemotePolicyProtocolError(f"predict payload {key} must be a list")
    images = tuple(_image_from_wire(item) for item in data["images"])
    proprioception = tuple(_proprioception_from_wire(item) for item in data["proprioception"])
    names = [item.name for item in images + proprioception]
    if len(names) != len(set(names)):
        raise RemotePolicyProtocolError("predict payload observation names must be unique")
    observation = PolicyObservation(StepId(data["step_id"]), data["timestamp_ns"],
                                    instruction_from_wire(data["instruction"]), images, proprioception)
    return data["episode_id"], observation


def _shape(value: Any, dimensions: tuple[int, ...], where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) not in dimensions or any(type(size) is not int or size <= 0 for size in value):
        raise RemotePolicyProtocolError(f"{where} shape must contain positive dimensions")
    return tuple(value)


def _decode_array(data_b64: Any, *, shape: tuple[int, ...], dtype: np.dtype, limit: int, where: str) -> np.ndarray:
    expected_bytes = math.prod(shape) * dtype.itemsize
    if expected_bytes > limit:
        raise RemotePolicyProtocolError(f"{where} declares {expected_bytes} bytes; limit is {limit}")
    if not isinstance(data_b64, str):
        raise RemotePolicyProtocolError(f"{where} data_b64 must be a string")
    if len(data_b64) > 4 * ((limit + 2) // 3):
        raise RemotePolicyProtocolError(f"{where} encoded data exceeds its size limit")
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise RemotePolicyProtocolError(f"{where} data_b64 is not valid base64") from exc
    if len(raw) != expected_bytes:
        raise RemotePolicyProtocolError(f"{where} byte length {len(raw)} does not match declared shape {list(shape)} ({expected_bytes} bytes)")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _image_from_wire(value: Any) -> ImageObservation:
    item = dict(_require_mapping(value, "predict payload image"))
    require_exact_keys(item, required={"name", "camera_name", "timestamp_ns", "encoding", "color_space", "shape", "dtype", "data_b64"}, where="predict payload image")
    array = _decode_array(item["data_b64"], shape=_shape(item["shape"], (2, 3), "image"),
                          dtype=_numeric_dtype(item["dtype"]), limit=MAX_IMAGE_BYTES, where="image")
    try:
        return ImageObservation(item["name"], array, item["timestamp_ns"], ImageEncoding(item["encoding"]),
                                ColorSpace(item["color_space"]), item["camera_name"])
    except (ValueError, TypeError) as exc:
        raise RemotePolicyProtocolError(f"invalid image contract: {exc}") from exc


def _proprioception_from_wire(value: Any) -> ProprioceptiveObservation:
    item = dict(_require_mapping(value, "predict payload proprioception"))
    require_exact_keys(item, required={"name", "timestamp_ns", "shape", "dtype", "units", "data_b64"}, where="predict payload proprioception")
    shape = _shape(item["shape"], (1,), "proprioception")
    if not isinstance(item["units"], list) or len(item["units"]) != shape[0]:
        raise RemotePolicyProtocolError("proprioception units must match its shape")
    array = _decode_array(item["data_b64"], shape=shape, dtype=_numeric_dtype(item["dtype"]),
                          limit=MAX_PROPRIO_BYTES, where="proprioception")
    return ProprioceptiveObservation(item["name"], array, item["timestamp_ns"], tuple(item["units"]))


def run_context_to_wire(context: RunContext) -> dict[str, Any]:
    return {
        "run_id": str(context.run_id),
        "created_wall_time_utc_ns": context.created_wall_time_utc_ns,
        "experiment_name": context.experiment_name,
        "seed": context.seed,
        "contract_version": context.contract_version,
        "metadata": _plain(context.metadata),
    }


def run_context_from_wire(value: Any) -> RunContext:
    data = dict(_require_mapping(value, "run_context"))
    require_exact_keys(
        data,
        required={
            "run_id", "created_wall_time_utc_ns", "experiment_name", "seed", "contract_version", "metadata"
        },
        where="run_context",
    )
    return RunContext(
        run_id=RunId(data["run_id"]),
        created_wall_time_utc_ns=data["created_wall_time_utc_ns"],
        experiment_name=data["experiment_name"],
        seed=data["seed"],
        contract_version=data["contract_version"],
        metadata=data["metadata"],
    )


def episode_context_to_wire(context: EpisodeContext) -> dict[str, Any]:
    return {
        "run_id": str(context.run_id),
        "task_id": str(context.task_id),
        "episode_id": str(context.episode_id),
        "rollout_index": context.rollout_index,
        "seed": context.seed,
        "initial_instruction": instruction_to_wire(context.initial_instruction),
        "metadata": _plain(context.metadata),
    }


def episode_context_from_wire(value: Any) -> EpisodeContext:
    data = dict(_require_mapping(value, "episode_context"))
    require_exact_keys(
        data,
        required={
            "run_id", "task_id", "episode_id", "rollout_index", "seed", "initial_instruction", "metadata"
        },
        where="episode_context",
    )
    return EpisodeContext(
        run_id=RunId(data["run_id"]),
        task_id=TaskId(data["task_id"]),
        episode_id=EpisodeId(data["episode_id"]),
        rollout_index=data["rollout_index"],
        seed=data["seed"],
        initial_instruction=instruction_from_wire(data["initial_instruction"]),
        metadata=data["metadata"],
    )


def prediction_to_wire(prediction: Any) -> dict[str, Any]:
    actions = np.asarray(prediction.actions)
    spec = prediction.action_spec
    dtype = _numeric_dtype(spec.dtype)
    if actions.dtype != dtype or actions.ndim != 2 or actions.shape != (prediction.horizon, spec.dimension):
        raise RemotePolicyProtocolError("prediction action shape/dtype differs from its ActionSpec and horizon")
    if actions.nbytes > MAX_PROPRIO_BYTES or not np.all(np.isfinite(actions)):
        raise RemotePolicyProtocolError("prediction actions exceed the size limit or contain non-finite values")
    common = {
        "prediction_id": str(prediction.prediction_id), "step_id": str(prediction.step_id),
        "dtype": dtype.str, "action_spec": action_spec_to_wire(spec),
        "timestamp_ns": prediction.timestamp_ns, "inference_duration_ns": prediction.inference_duration_ns,
        "validity": prediction.validity.value, "confidence": prediction.confidence,
        "metadata": _plain(prediction.metadata),
    }
    if actions.shape[0] == 1:
        return {**common, "action": actions[0].tolist(), "shape": [spec.dimension]}
    return {**common, "actions": actions.tolist(), "shape": list(actions.shape), "horizon": actions.shape[0]}


def prediction_from_wire(value: Any) -> dict[str, Any]:
    data = dict(_require_mapping(value, "prediction"))
    common = {"prediction_id", "step_id", "dtype", "shape", "action_spec", "timestamp_ns",
              "inference_duration_ns", "validity", "confidence", "metadata"}
    vector = "action" in data
    require_exact_keys(data, required=common | ({"action"} if vector else {"actions", "horizon"}), where="prediction")
    spec = action_spec_from_wire(data["action_spec"])
    dtype = _numeric_dtype(data["dtype"])
    shape = _shape(data["shape"], (1,) if vector else (2,), "prediction")
    horizon = 1 if vector else data["horizon"]
    expected = (spec.dimension,) if vector else (horizon, spec.dimension)
    if type(horizon) is not int or horizon < 1 or shape != expected or dtype != _numeric_dtype(spec.dtype):
        raise RemotePolicyProtocolError("prediction shape/dtype differs from its ActionSpec and horizon")
    if math.prod(shape) * dtype.itemsize > MAX_PROPRIO_BYTES:
        raise RemotePolicyProtocolError("prediction action array exceeds its size limit")
    try:
        actions = np.asarray(data["action" if vector else "actions"], dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise RemotePolicyProtocolError("invalid prediction action array") from exc
    if actions.shape != shape or not np.all(np.isfinite(actions)):
        raise RemotePolicyProtocolError("prediction actions must match the declared shape and contain finite values")
    data["actions"] = actions[np.newaxis, :] if vector else actions
    data["horizon"] = horizon
    data["action_spec"] = spec
    data["prediction_id"] = PredictionId(data["prediction_id"])
    data["step_id"] = StepId(data["step_id"])
    data["validity"] = PredictionValidity(data["validity"])
    return data
