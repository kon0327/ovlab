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
    PredictionId,
    PredictionValidity,
    RunContext,
    RunId,
    RotationRepresentation,
    StepId,
    TaskId,
)
from ovlab_remote_policy.errors import RemotePolicyProtocolError

PROTOCOL_VERSION = "ovlab-policy-rpc/1.0.0"
HEADER_SIZE = 4
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 1024 * 1024
PRIMARY_RGB_NAME = "camera.primary.rgb"

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
        minimum=None if data["minimum"] is None else np.asarray(data["minimum"], dtype=np.float32),
        maximum=None if data["maximum"] is None else np.asarray(data["maximum"], dtype=np.float32),
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


def capabilities_to_wire(capabilities: PolicyCapabilities) -> dict[str, Any]:
    requirements = capabilities.observation_requirements
    return {
        "component_name": capabilities.component_name,
        "component_version": capabilities.component_version,
        "contract_version": capabilities.contract_version,
        "observation_requirements": {
            "images": [image_spec_to_wire(spec) for spec in requirements.images],
            "proprioception": [],
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
    if requirements["proprioception"]:
        raise RemotePolicyProtocolError("remote policy capability cannot require proprioception")
    return PolicyCapabilities(
        component_name=data["component_name"],
        component_version=data["component_version"],
        contract_version=data["contract_version"],
        observation_requirements=ObservationRequirements(
            images=tuple(image_spec_from_wire(item) for item in requirements["images"]),
            proprioception=(),
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


def observation_to_predict_payload(
    observation: PolicyObservation,
    *,
    episode_id: str,
) -> dict[str, Any]:
    if observation.proprioception:
        raise RemotePolicyProtocolError("proprioception is forbidden by the remote prediction schema")
    if observation.metadata:
        raise RemotePolicyProtocolError("observation metadata is forbidden by the remote prediction schema")
    images = {image.name: image for image in observation.images}
    if set(images) != {PRIMARY_RGB_NAME}:
        raise RemotePolicyProtocolError(
            f"remote prediction requires only {PRIMARY_RGB_NAME!r}; received {sorted(images)}"
        )
    image = images[PRIMARY_RGB_NAME]
    array = image.data
    if image.encoding is not ImageEncoding.RAW:
        raise RemotePolicyProtocolError("camera.primary.rgb encoding must be raw")
    if array.dtype != np.uint8:
        raise RemotePolicyProtocolError("camera.primary.rgb dtype must be uint8")
    if array.ndim != 3 or array.shape[2] != 3:
        raise RemotePolicyProtocolError("camera.primary.rgb must have shape [height, width, 3]")
    raw = array.tobytes(order="C")
    if len(raw) > MAX_IMAGE_BYTES:
        raise RemotePolicyProtocolError(f"RGB image is {len(raw)} bytes; limit is {MAX_IMAGE_BYTES}")
    return {
        "episode_id": episode_id,
        "step_id": str(observation.step_id),
        "instruction": instruction_to_wire(observation.instruction),
        "image": {
            "shape": list(array.shape),
            "layout": "HWC",
            "dtype": "uint8",
            "data_b64": base64.b64encode(raw).decode("ascii"),
        },
    }


def predict_payload_to_observation(value: Any) -> tuple[str, PolicyObservation]:
    data = dict(_require_mapping(value, "predict payload"))
    require_exact_keys(
        data,
        required={"episode_id", "step_id", "instruction", "image"},
        where="predict payload",
    )
    if not isinstance(data["episode_id"], str) or not data["episode_id"]:
        raise RemotePolicyProtocolError("episode_id must be a non-empty string")
    image = dict(_require_mapping(data["image"], "predict payload image"))
    require_exact_keys(
        image,
        required={"shape", "layout", "dtype", "data_b64"},
        where="predict payload image",
    )
    if image["layout"] != "HWC" or image["dtype"] != "uint8":
        raise RemotePolicyProtocolError("RGB metadata must specify HWC and uint8")
    shape = image["shape"]
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, str | bytes)
        or len(shape) != 3
        or not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in shape)
        or shape[2] != 3
    ):
        raise RemotePolicyProtocolError("RGB shape must be three positive integers ending in 3")
    expected_bytes = int(np.prod(shape, dtype=np.int64))
    if expected_bytes > MAX_IMAGE_BYTES:
        raise RemotePolicyProtocolError(f"RGB image declares {expected_bytes} bytes; limit is {MAX_IMAGE_BYTES}")
    if not isinstance(image["data_b64"], str):
        raise RemotePolicyProtocolError("RGB data_b64 must be a string")
    try:
        raw = base64.b64decode(image["data_b64"], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise RemotePolicyProtocolError("RGB data_b64 is not valid base64") from exc
    if len(raw) != expected_bytes:
        raise RemotePolicyProtocolError(
            f"RGB byte length {len(raw)} does not match declared shape {list(shape)} ({expected_bytes} bytes)"
        )
    array = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape)).copy()
    instruction = instruction_from_wire(data["instruction"])
    observation = PolicyObservation(
        step_id=StepId(data["step_id"]),
        timestamp_ns=instruction.timestamp_ns,
        instruction=instruction,
        images=(
            ImageObservation(
                name=PRIMARY_RGB_NAME,
                data=array,
                encoding=ImageEncoding.RAW,
                color_space=ColorSpace.RGB,
                camera_name="camera.primary",
                timestamp_ns=instruction.timestamp_ns,
            ),
        ),
    )
    return data["episode_id"], observation


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
    if actions.dtype != np.float32 or actions.shape != (1, 7):
        raise RemotePolicyProtocolError(
            f"remote policy must return canonical float32 action shape (1, 7), got {actions.dtype} {actions.shape}"
        )
    if prediction.action_spec.gripper_convention is not GripperConvention.CLOSED_POSITIVE:
        raise RemotePolicyProtocolError("remote action must use CLOSED_POSITIVE gripper convention")
    return {
        "prediction_id": str(prediction.prediction_id),
        "step_id": str(prediction.step_id),
        "action": actions[0].tolist(),
        "dtype": "float32",
        "shape": [7],
        "action_spec": action_spec_to_wire(prediction.action_spec),
        "timestamp_ns": prediction.timestamp_ns,
        "inference_duration_ns": prediction.inference_duration_ns,
        "validity": prediction.validity.value,
        "confidence": prediction.confidence,
        "metadata": _plain(prediction.metadata),
    }


def prediction_from_wire(value: Any) -> dict[str, Any]:
    data = dict(_require_mapping(value, "prediction"))
    require_exact_keys(
        data,
        required={
            "prediction_id", "step_id", "action", "dtype", "shape", "action_spec", "timestamp_ns",
            "inference_duration_ns", "validity", "confidence", "metadata",
        },
        where="prediction",
    )
    if data["dtype"] != "float32" or data["shape"] != [7]:
        raise RemotePolicyProtocolError("prediction must declare one float32 [7] action")
    action = np.asarray(data["action"], dtype=np.float32)
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise RemotePolicyProtocolError("prediction action must contain seven finite values")
    action_spec = action_spec_from_wire(data["action_spec"])
    if action_spec.dimension != 7 or action_spec.dtype != "float32":
        raise RemotePolicyProtocolError("prediction ActionSpec must describe float32 [7]")
    if action_spec.gripper_convention is not GripperConvention.CLOSED_POSITIVE:
        raise RemotePolicyProtocolError("prediction ActionSpec must use CLOSED_POSITIVE")
    data["action"] = action
    data["action_spec"] = action_spec
    data["prediction_id"] = PredictionId(data["prediction_id"])
    data["step_id"] = StepId(data["step_id"])
    data["validity"] = PredictionValidity(data["validity"])
    return data
