"""Synthetic immutable traces for dependency-free metric tests."""

import numpy as np

from helpers.contexts import make_episode_context, make_run_context, make_step_context
from ovlab_core.contracts import (
    ActionPrediction,
    ActionRepresentation,
    ActionSpec,
    EpisodeTerminalStatus,
    EpisodeTrace,
    ExecutedAction,
    GripperConvention,
    PredictionId,
    PredictionValidity,
    RotationRepresentation,
    SignalAccess,
    SignalValue,
)


def metric_action_spec(dimension=2, *, gripper=False, convention=GripperConvention.NONE):
    gripper_indices = (dimension - 1,) if gripper else ()
    return ActionSpec(
        dimension,
        ActionRepresentation.OTHER,
        gripper_indices=gripper_indices,
        rotation_representation=RotationRepresentation.NONE,
        gripper_convention=convention,
        units=("normalized_command",) * dimension,
        dtype="float32",
    )


def synthetic_trace(
    actions=((0, 0), (1, 0), (3, 0)),
    *,
    requested=None,
    terminal_status=EpisodeTerminalStatus.SUCCESS,
    success_signal=True,
    include_success=True,
    prediction_validities=None,
    inference_durations=(1_000_000, 2_000_000, 4_000_000),
    action_spec=None,
    collision_values=(),
    step_indices=None,
    horizon=1,
):
    actions = tuple(np.asarray(action, dtype=np.float32) for action in actions)
    requested = actions if requested is None else tuple(np.asarray(action, dtype=np.float32) for action in requested)
    spec = action_spec or metric_action_spec(actions[0].shape[0] if actions else 2)
    episode = make_episode_context(episode_id="metric-episode", seed=3)
    indices = tuple(range(len(actions))) if step_indices is None else tuple(step_indices)
    contexts = tuple(make_step_context(episode, index, 10 + offset) for offset, index in enumerate(indices))
    validities = prediction_validities or (PredictionValidity.VALID,) * len(actions)
    predictions, executed = [], []
    for offset, (context, applied, asked) in enumerate(zip(contexts, actions, requested)):
        prediction_id = PredictionId(f"metric-prediction-{offset}")
        chunks = np.repeat(applied[np.newaxis, :], horizon, axis=0)
        if horizon > 1:
            chunks[1:] = 99
        predictions.append(
            ActionPrediction(
                prediction_id,
                context.step_id,
                chunks,
                spec,
                context.timestamp_ns,
                inference_durations[offset] if offset < len(inference_durations) else 1_000_000,
                horizon,
                validities[offset],
            )
        )
        changed = not np.array_equal(asked, applied)
        executed.append(
            ExecutedAction(
                prediction_id,
                context.step_id,
                0,
                asked,
                applied,
                context.timestamp_ns,
                "test modification" if changed else None,
            )
        )
    signals = []
    if include_success:
        signals.append(
            SignalValue(
                "benchmark.task_success",
                bool(success_signal),
                20,
                "synthetic",
                contexts[-1].step_id if contexts else None,
                access=SignalAccess.EVALUATION_ONLY,
            )
        )
    for offset, value in enumerate(collision_values):
        signals.append(
            SignalValue(
                "safety.collision_event",
                bool(value),
                30 + offset,
                "synthetic",
                None,
                access=SignalAccess.PRIVILEGED,
            )
        )
    signals.sort(key=lambda signal: signal.timestamp_ns)
    return EpisodeTrace(
        episode,
        contexts,
        (),
        (episode.initial_instruction,),
        tuple(predictions),
        tuple(executed),
        tuple(signals),
        terminal_status,
        0,
        100,
    )
