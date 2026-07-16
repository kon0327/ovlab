"""Focused tests for the stable ovlab_core.contracts public API."""

from dataclasses import FrozenInstanceError
from inspect import signature

import numpy as np
import pytest

from ovlab_core.contracts import (
    OVLAB_CONTRACT_VERSION,
    ActionPrediction,
    ActionRepresentation,
    ActionSpec,
    ColorSpace,
    ContractCompatibilityError,
    ContractError,
    ContractValidationError,
    EpisodeContext,
    EpisodeId,
    EpisodeTerminalStatus,
    EpisodeTrace,
    ExecutedAction,
    GripperConvention,
    ImageEncoding,
    ImageObservation,
    Instruction,
    InstructionId,
    InstructionSource,
    PolicyObservation,
    PredictionId,
    PredictionValidity,
    ProprioceptiveObservation,
    RawPolicyOutput,
    RotationRepresentation,
    RunContext,
    RunId,
    SignalAccess,
    SignalRegistry,
    SignalSpec,
    SignalValue,
    StepContext,
    StepId,
    TaskContext,
    TaskId,
    normalize_metadata,
)


def instruction(timestamp_ns: int = 10) -> Instruction:
    return Instruction(InstructionId("instruction-1"), "move the object", timestamp_ns, InstructionSource.BENCHMARK)


def action_spec() -> ActionSpec:
    return ActionSpec(
        dimension=7,
        representation=ActionRepresentation.DELTA_POSE,
        translation_indices=(0, 1, 2),
        rotation_indices=(3, 4, 5),
        gripper_indices=(6,),
        rotation_representation=RotationRepresentation.AXIS_ANGLE,
        gripper_convention=GripperConvention.OPEN_POSITIVE,
        units=("m", "m", "m", "rad", "rad", "rad", "unitless"),
        minimum=np.full(7, -1.0),
        maximum=np.full(7, 1.0),
        dtype="float32",
        control_frequency_hz=20.0,
    )


def contexts():
    run_id = RunId("run-1")
    task_id = TaskId("task-1")
    episode_id = EpisodeId("episode-1")
    initial = instruction()
    run = RunContext(run_id, 1_700_000_000_000_000_000, "experiment", 7)
    task = TaskContext(run_id, task_id, "suite", "task name", 0)
    episode = EpisodeContext(run_id, task_id, episode_id, 0, 11, initial)
    step = StepContext(run_id, task_id, episode_id, StepId("step-1"), 0, 20)
    return run, task, episode, step


def trace_members():
    _, _, episode, step = contexts()
    image = ImageObservation(
        "front",
        np.zeros((4, 4, 3), dtype=np.uint8),
        20,
        ImageEncoding.RAW,
        ColorSpace.RGB,
        "front-camera",
    )
    proprio = ProprioceptiveObservation("robot", np.zeros(2), 20, ("rad", "rad"))
    observation = PolicyObservation(step.step_id, 20, episode.initial_instruction, (image,), (proprio,))
    raw = RawPolicyOutput(PredictionId("prediction-1"), (1, 2, 3), 21)
    prediction = ActionPrediction(
        raw.prediction_id,
        step.step_id,
        np.zeros(7),
        action_spec(),
        22,
        2,
        1,
        raw_output=raw,
    )
    executed = ExecutedAction(raw.prediction_id, step.step_id, 0, np.zeros(7), np.zeros(7), 23)
    signal = SignalValue("success", False, 24, "benchmark", step.step_id)
    return episode, step, observation, prediction, executed, signal


def test_valid_construction_of_public_contracts() -> None:
    run, task, episode, step = contexts()
    episode, step, observation, prediction, executed, signal = trace_members()
    signal_spec = SignalSpec(
        "success", "bool", (), "", SignalAccess.EVALUATION_ONLY, "Whether the task succeeded"
    )
    registry = SignalRegistry((signal_spec,))
    trace = EpisodeTrace(
        episode,
        (step,),
        (observation,),
        (episode.initial_instruction,),
        (prediction,),
        (executed,),
        (signal,),
        EpisodeTerminalStatus.SUCCESS,
        10,
        25,
    )

    assert run.contract_version == OVLAB_CONTRACT_VERSION
    assert task.run_id == run.run_id
    assert registry.resolve("success") is signal_spec
    assert trace.terminal_status is EpisodeTerminalStatus.SUCCESS
    assert isinstance(ContractCompatibilityError("version"), ContractError)


@pytest.mark.parametrize("identifier", [RunId, TaskId, EpisodeId, StepId, InstructionId, PredictionId])
@pytest.mark.parametrize("value", ["", "   "])
def test_identifiers_reject_empty_values(identifier, value: str) -> None:
    with pytest.raises(ContractValidationError, match=rf"{identifier.__name__}\.value"):
        identifier(value)


def test_identifier_equality_hash_and_string() -> None:
    first = RunId("run")
    second = RunId("run")
    assert first == second
    assert hash(first) == hash(second)
    assert str(first) == "run"
    assert first != TaskId("run")


def test_instruction_can_supersede_previous_instruction() -> None:
    previous = instruction()
    replacement = Instruction(
        InstructionId("instruction-2"),
        "stop and return",
        30,
        InstructionSource.USER,
        supersedes=previous.instruction_id,
    )
    assert replacement.supersedes == previous.instruction_id


def test_instruction_rejects_blank_text() -> None:
    with pytest.raises(ContractValidationError, match=r"Instruction\.text"):
        Instruction(InstructionId("i"), "  ", 0, InstructionSource.SYSTEM)


def test_nested_metadata_is_effectively_immutable() -> None:
    original = {"nested": {"items": [1, 2]}}
    value = instruction().__class__(
        InstructionId("metadata"), "test", 0, InstructionSource.SYSTEM, metadata=original
    )
    original["nested"]["items"].append(3)
    assert value.metadata["nested"]["items"] == (1, 2)
    with pytest.raises(TypeError):
        value.metadata["new"] = "value"
    with pytest.raises(TypeError):
        value.metadata["nested"]["new"] = "value"


def test_unsupported_metadata_object_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="unsupported metadata value type"):
        normalize_metadata({"bad": object()}, "Example")


def test_numpy_arrays_are_defensively_copied_and_immutable() -> None:
    source = np.ones((2, 2, 3), dtype=np.float32)
    image = ImageObservation("image", source, 0, ImageEncoding.RAW, ColorSpace.RGB, "camera")
    source[0, 0, 0] = 9
    assert image.data[0, 0, 0] == 1
    with pytest.raises(ValueError):
        image.data.flags.writeable = True
    with pytest.raises(ValueError):
        image.data[0, 0, 0] = 2


@pytest.mark.parametrize(
    "data",
    [np.zeros(3), np.zeros((0, 2)), np.array([[np.nan]]), np.array([[object()]], dtype=object)],
)
def test_invalid_image_shapes_and_values(data: np.ndarray) -> None:
    with pytest.raises(ContractValidationError, match=r"ImageObservation\.data"):
        ImageObservation("image", data, 0, ImageEncoding.RAW, ColorSpace.OTHER, "camera")


def test_policy_observation_has_no_signal_channel() -> None:
    assert "signals" not in signature(PolicyObservation).parameters
    assert "evaluation_signals" not in signature(PolicyObservation).parameters


def test_duplicate_signal_registration_is_rejected() -> None:
    first = SignalSpec("state", "float32", (2,), "m", SignalAccess.POLICY_VISIBLE, "State")
    second = SignalSpec("state", "float32", (2,), "m", SignalAccess.PRIVILEGED, "Privileged state")
    with pytest.raises(ContractValidationError, match="duplicate signal names"):
        SignalRegistry((first, second))


def test_signal_access_classification_and_required_lookup() -> None:
    public = SignalSpec("public", "float32", (), "", SignalAccess.POLICY_VISIBLE, "Public")
    private = SignalSpec("private", "float32", (), "", SignalAccess.PRIVILEGED, "Private", optional=True)
    registry = SignalRegistry((private, public))
    assert tuple(spec.name for spec in registry) == ("private", "public")
    assert registry.resolve("private").access is SignalAccess.PRIVILEGED
    assert registry.has_required("public")
    assert not registry.has_required("private")


def test_single_action_is_canonicalized_to_one_by_dimension() -> None:
    prediction = ActionPrediction(
        PredictionId("p"), StepId("s"), np.zeros(7), action_spec(), 0, 0, 1
    )
    assert prediction.actions.shape == (1, 7)


def test_action_chunk_preserves_horizon_by_dimension() -> None:
    prediction = ActionPrediction(
        PredictionId("p"), StepId("s"), np.zeros((3, 7)), action_spec(), 0, 10, 3
    )
    assert prediction.actions.shape == (3, 7)
    assert prediction.horizon == 3


def test_action_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="action dimension"):
        ActionPrediction(PredictionId("p"), StepId("s"), np.zeros(6), action_spec(), 0, 0, 1)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_action_is_rejected(bad_value: float) -> None:
    actions = np.zeros(7)
    actions[0] = bad_value
    with pytest.raises(ContractValidationError, match="finite"):
        ActionPrediction(PredictionId("p"), StepId("s"), actions, action_spec(), 0, 0, 1)


def test_overlapping_action_indices_are_rejected() -> None:
    with pytest.raises(ContractValidationError, match="must not overlap"):
        ActionSpec(
            3,
            ActionRepresentation.OTHER,
            translation_indices=(0, 1),
            rotation_indices=(1, 2),
            rotation_representation=RotationRepresentation.AXIS_ANGLE,
        )


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(np.zeros(6), np.ones(7)), (np.array([0, 0]), np.array([-1, 1]))],
)
def test_inconsistent_action_limits_are_rejected(minimum, maximum) -> None:
    dimension = 7 if minimum.shape[0] == 6 else 2
    with pytest.raises(ContractValidationError, match="minimum|maximum"):
        ActionSpec(dimension, ActionRepresentation.JOINT_POSITION, minimum=minimum, maximum=maximum)


def test_executed_action_preserves_requested_and_applied_values() -> None:
    requested = np.zeros(7)
    applied = requested.copy()
    applied[0] = 0.5
    executed = ExecutedAction(
        PredictionId("p"), StepId("s"), 1, requested, applied, 10, "safety envelope"
    )
    assert not np.array_equal(executed.requested_action, executed.applied_action)
    assert executed.modification_reason == "safety envelope"


def test_modified_executed_action_requires_reason() -> None:
    with pytest.raises(ContractValidationError, match="modification_reason"):
        ExecutedAction(PredictionId("p"), StepId("s"), 0, np.zeros(2), np.ones(2), 0)


def test_lifecycle_rejects_negative_values_and_wrong_contract_version() -> None:
    with pytest.raises(ContractValidationError, match=r"RunContext\.seed"):
        RunContext(RunId("run"), 0, "experiment", -1)
    with pytest.raises(ContractValidationError, match=r"RunContext\.contract_version"):
        RunContext(RunId("run"), 0, "experiment", 0, contract_version="9.0.0")


def test_invalid_episode_time_ordering_is_rejected() -> None:
    episode, step, observation, prediction, executed, signal = trace_members()
    with pytest.raises(ContractValidationError, match="end_timestamp_ns"):
        EpisodeTrace(
            episode,
            (step,),
            (observation,),
            (episode.initial_instruction,),
            (prediction,),
            (executed,),
            (signal,),
            EpisodeTerminalStatus.FAILURE,
            30,
            29,
        )


def test_trace_rejects_content_from_different_episode() -> None:
    episode, step, observation, prediction, executed, signal = trace_members()
    foreign = StepContext(step.run_id, step.task_id, EpisodeId("other"), step.step_id, 0, 20)
    with pytest.raises(ContractValidationError, match="different run/task/episode"):
        EpisodeTrace(
            episode,
            (foreign,),
            (observation,),
            (episode.initial_instruction,),
            (prediction,),
            (executed,),
            (signal,),
            EpisodeTerminalStatus.ABORTED,
            10,
            30,
        )


def test_trace_collections_and_contracts_are_frozen() -> None:
    episode, step, observation, prediction, executed, signal = trace_members()
    trace = EpisodeTrace(
        episode,
        [step],
        [observation],
        [episode.initial_instruction],
        [prediction],
        [executed],
        [signal],
        EpisodeTerminalStatus.SUCCESS,
        10,
        30,
    )
    assert isinstance(trace.observations, tuple)
    with pytest.raises(FrozenInstanceError):
        trace.end_timestamp_ns = 40


def test_public_import_surface_contains_expected_contracts() -> None:
    import ovlab_core.contracts as contracts

    expected = {
        "ActionPrediction",
        "EpisodeTrace",
        "PolicyObservation",
        "SignalRegistry",
        "OVLAB_CONTRACT_VERSION",
    }
    assert expected <= set(contracts.__all__)
