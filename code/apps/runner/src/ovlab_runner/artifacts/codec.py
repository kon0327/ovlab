"""Safe JSON/JSONL/NPY codec for immutable EpisodeTrace artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ovlab_core.contracts import (
    ActionPrediction, ActionRepresentation, ActionSpec, ColorSpace, EpisodeContext, EpisodeId,
    EpisodeTerminalStatus, EpisodeTrace, ExecutedAction, GripperConvention, ImageEncoding,
    ImageObservation, Instruction, InstructionId, InstructionSource, PolicyObservation,
    PredictionId, PredictionValidity, ProprioceptiveObservation, RawPolicyOutput, RotationRepresentation,
    RunId, SignalAccess, SignalValue, StepContext, StepId, TaskId,
)

from ..errors import ArtifactError


class TraceCodec:
    schema_version = "1.0.0"

    def encode(self, trace: EpisodeTrace, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        arrays = directory / "arrays"
        arrays.mkdir()
        counter = [0]

        def array(value, label):
            index = counter[0]
            counter[0] += 1
            relative = f"arrays/{index:06d}-{label}.npy"
            path = directory / relative
            np.save(path, np.asarray(value), allow_pickle=False)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            return {"$array": relative, "sha256": checksum, "dtype": str(value.dtype), "shape": list(value.shape)}

        payload = self._encode_trace(trace, array)
        self._write_json(directory / "trace.json", payload)
        events = []
        collections = (
            ("step_context", trace.step_contexts), ("observation", trace.observations),
            ("instruction", trace.instruction_events), ("prediction", trace.policy_predictions),
            ("executed_action", trace.executed_actions), ("signal", trace.evaluation_signals),
        )
        for kind, values in collections:
            for index, value in enumerate(values):
                events.append((value.timestamp_ns, kind, index))
        with (directory / "events.jsonl").open("x", encoding="utf-8") as stream:
            for timestamp, kind, index in sorted(events):
                stream.write(json.dumps({"timestamp_ns": timestamp, "type": kind, "index": index}, sort_keys=True) + "\n")

    def decode(self, directory: Path) -> EpisodeTrace:
        payload = json.loads((directory / "trace.json").read_text(encoding="utf-8"))

        def array(reference):
            relative = reference["$array"]
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ArtifactError("array reference is not a safe relative path")
            target = directory / path
            if not target.is_file():
                raise ArtifactError(f"referenced array is missing: {relative}")
            if hashlib.sha256(target.read_bytes()).hexdigest() != reference["sha256"]:
                raise ArtifactError(f"array checksum mismatch: {relative}")
            value = np.load(target, allow_pickle=False)
            if str(value.dtype) != reference["dtype"] or list(value.shape) != reference["shape"]:
                raise ArtifactError(f"array dtype or shape mismatch: {relative}")
            return value

        return self._decode_trace(payload, array)

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    def _encode_trace(self, trace, array):
        return {
            "schema_version": self.schema_version,
            "episode_context": _episode(trace.episode_context),
            "step_contexts": [_step(value) for value in trace.step_contexts],
            "observations": [_observation(value, array) for value in trace.observations],
            "instruction_events": [_instruction(value) for value in trace.instruction_events],
            "policy_predictions": [_prediction(value, array) for value in trace.policy_predictions],
            "executed_actions": [_executed(value, array) for value in trace.executed_actions],
            "evaluation_signals": [_signal(value, array) for value in trace.evaluation_signals],
            "terminal_status": trace.terminal_status.value,
            "start_timestamp_ns": trace.start_timestamp_ns,
            "end_timestamp_ns": trace.end_timestamp_ns,
            "metadata": _plain(trace.metadata, array, "metadata"),
        }

    def _decode_trace(self, value, array):
        if value.get("schema_version") != self.schema_version:
            raise ArtifactError("unsupported trace schema version")
        context = _decode_episode(value["episode_context"])
        return EpisodeTrace(
            context,
            tuple(_decode_step(item) for item in value["step_contexts"]),
            tuple(_decode_observation(item, array) for item in value["observations"]),
            tuple(_decode_instruction(item) for item in value["instruction_events"]),
            tuple(_decode_prediction(item, array) for item in value["policy_predictions"]),
            tuple(_decode_executed(item, array) for item in value["executed_actions"]),
            tuple(_decode_signal(item, array) for item in value["evaluation_signals"]),
            EpisodeTerminalStatus(value["terminal_status"]), value["start_timestamp_ns"], value["end_timestamp_ns"],
            _decode_plain(value["metadata"], array),
        )


def _plain(value, array, label):
    if isinstance(value, np.ndarray): return array(value, label)
    if hasattr(value, "items"): return {key: _plain(item, array, label) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(item, array, label) for item in value]
    return value


def _decode_plain(value, array):
    if isinstance(value, dict) and "$array" in value: return array(value)
    if isinstance(value, dict): return {key: _decode_plain(item, array) for key, item in value.items()}
    if isinstance(value, list): return [_decode_plain(item, array) for item in value]
    return value


def _instruction(value):
    return {"instruction_id": str(value.instruction_id), "text": value.text, "timestamp_ns": value.timestamp_ns, "source": value.source.value, "supersedes": None if value.supersedes is None else str(value.supersedes), "metadata": _metadata(value.metadata)}


def _decode_instruction(value):
    return Instruction(InstructionId(value["instruction_id"]), value["text"], value["timestamp_ns"], InstructionSource(value["source"]), None if value["supersedes"] is None else InstructionId(value["supersedes"]), value["metadata"])


def _episode(value):
    return {"run_id": str(value.run_id), "task_id": str(value.task_id), "episode_id": str(value.episode_id), "rollout_index": value.rollout_index, "seed": value.seed, "initial_instruction": _instruction(value.initial_instruction), "metadata": _metadata(value.metadata)}


def _decode_episode(value):
    return EpisodeContext(RunId(value["run_id"]), TaskId(value["task_id"]), EpisodeId(value["episode_id"]), value["rollout_index"], value["seed"], _decode_instruction(value["initial_instruction"]), value["metadata"])


def _step(value):
    return {"run_id": str(value.run_id), "task_id": str(value.task_id), "episode_id": str(value.episode_id), "step_id": str(value.step_id), "step_index": value.step_index, "timestamp_ns": value.timestamp_ns}


def _decode_step(value):
    return StepContext(RunId(value["run_id"]), TaskId(value["task_id"]), EpisodeId(value["episode_id"]), StepId(value["step_id"]), value["step_index"], value["timestamp_ns"])


def _observation(value, array):
    return {"step_id": str(value.step_id), "timestamp_ns": value.timestamp_ns, "instruction": _instruction(value.instruction), "images": [{"name": image.name, "data": array(image.data, "image"), "timestamp_ns": image.timestamp_ns, "encoding": image.encoding.value, "color_space": image.color_space.value, "camera_name": image.camera_name, "metadata": _metadata(image.metadata)} for image in value.images], "proprioception": [{"name": item.name, "values": array(item.values, "proprio"), "timestamp_ns": item.timestamp_ns, "units": list(item.units), "metadata": _metadata(item.metadata)} for item in value.proprioception], "metadata": _metadata(value.metadata)}


def _decode_observation(value, array):
    images = tuple(ImageObservation(item["name"], array(item["data"]), item["timestamp_ns"], ImageEncoding(item["encoding"]), ColorSpace(item["color_space"]), item["camera_name"], item["metadata"]) for item in value["images"])
    proprio = tuple(ProprioceptiveObservation(item["name"], array(item["values"]), item["timestamp_ns"], tuple(item["units"]), item["metadata"]) for item in value["proprioception"])
    return PolicyObservation(StepId(value["step_id"]), value["timestamp_ns"], _decode_instruction(value["instruction"]), images, proprio, value["metadata"])


def _spec(value, array):
    return {"dimension": value.dimension, "representation": value.representation.value, "translation_indices": list(value.translation_indices), "rotation_indices": list(value.rotation_indices), "gripper_indices": list(value.gripper_indices), "rotation_representation": value.rotation_representation.value, "gripper_convention": value.gripper_convention.value, "units": list(value.units), "minimum": None if value.minimum is None else array(value.minimum, "action-min"), "maximum": None if value.maximum is None else array(value.maximum, "action-max"), "dtype": value.dtype, "control_frequency_hz": value.control_frequency_hz, "metadata": _metadata(value.metadata)}


def _decode_spec(value, array):
    return ActionSpec(value["dimension"], ActionRepresentation(value["representation"]), tuple(value["translation_indices"]), tuple(value["rotation_indices"]), tuple(value["gripper_indices"]), RotationRepresentation(value["rotation_representation"]), GripperConvention(value["gripper_convention"]), tuple(value["units"]), None if value["minimum"] is None else array(value["minimum"]), None if value["maximum"] is None else array(value["maximum"]), value["dtype"], value["control_frequency_hz"], value["metadata"])


def _prediction(value, array):
    raw = None if value.raw_output is None else {"prediction_id": str(value.raw_output.prediction_id), "value": _plain(value.raw_output.value, array, "raw"), "timestamp_ns": value.raw_output.timestamp_ns, "metadata": _metadata(value.raw_output.metadata)}
    return {"prediction_id": str(value.prediction_id), "step_id": str(value.step_id), "actions": array(value.actions, "prediction"), "action_spec": _spec(value.action_spec, array), "timestamp_ns": value.timestamp_ns, "inference_duration_ns": value.inference_duration_ns, "horizon": value.horizon, "validity": value.validity.value, "confidence": value.confidence, "raw_output": raw, "metadata": _metadata(value.metadata)}


def _decode_prediction(value, array):
    raw = value["raw_output"]
    decoded_raw = None if raw is None else RawPolicyOutput(PredictionId(raw["prediction_id"]), _decode_plain(raw["value"], array), raw["timestamp_ns"], raw["metadata"])
    return ActionPrediction(PredictionId(value["prediction_id"]), StepId(value["step_id"]), array(value["actions"]), _decode_spec(value["action_spec"], array), value["timestamp_ns"], value["inference_duration_ns"], value["horizon"], PredictionValidity(value["validity"]), value["confidence"], decoded_raw, value["metadata"])


def _executed(value, array):
    return {"prediction_id": str(value.prediction_id), "step_id": str(value.step_id), "selected_chunk_index": value.selected_chunk_index, "requested_action": array(value.requested_action, "requested"), "applied_action": array(value.applied_action, "applied"), "timestamp_ns": value.timestamp_ns, "modification_reason": value.modification_reason, "metadata": _metadata(value.metadata)}


def _decode_executed(value, array):
    return ExecutedAction(PredictionId(value["prediction_id"]), StepId(value["step_id"]), value["selected_chunk_index"], array(value["requested_action"]), array(value["applied_action"]), value["timestamp_ns"], value["modification_reason"], value["metadata"])


def _signal(value, array):
    return {"name": value.name, "value": _plain(value.value, array, "signal"), "timestamp_ns": value.timestamp_ns, "source": value.source, "step_id": None if value.step_id is None else str(value.step_id), "metadata": _metadata(value.metadata), "access": None if value.access is None else value.access.value}


def _decode_signal(value, array):
    return SignalValue(value["name"], _decode_plain(value["value"], array), value["timestamp_ns"], value["source"], None if value["step_id"] is None else StepId(value["step_id"]), value["metadata"], None if value["access"] is None else SignalAccess(value["access"]))


def _metadata(value):
    if hasattr(value, "items"): return {key: _metadata(item) for key, item in value.items()}
    if isinstance(value, tuple): return [_metadata(item) for item in value]
    return value
