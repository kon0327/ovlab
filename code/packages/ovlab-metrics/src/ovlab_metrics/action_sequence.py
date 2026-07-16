"""Shared extraction of contiguous executed command sequences."""

from dataclasses import dataclass

import numpy as np

from ovlab_core.contracts import ActionSpec, EpisodeTrace, immutable_numeric_array

from .config import ActionSource
from .errors import ActionSequenceError


def action_spec_identity(spec: ActionSpec) -> dict:
    return {
        "dimension": spec.dimension,
        "representation": spec.representation.value,
        "translation_indices": spec.translation_indices,
        "rotation_indices": spec.rotation_indices,
        "gripper_indices": spec.gripper_indices,
        "rotation_representation": spec.rotation_representation.value,
        "gripper_convention": spec.gripper_convention.value,
        "units": spec.units,
        "dtype": spec.dtype,
    }


@dataclass(frozen=True, slots=True)
class ActionSequence:
    values: np.ndarray
    action_spec: ActionSpec
    source: ActionSource
    step_indices: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "values", immutable_numeric_array(self.values, type(self).__name__, "values", ndim=2))


def extract_action_sequence(trace: EpisodeTrace, source: ActionSource = ActionSource.APPLIED) -> ActionSequence:
    if not trace.executed_actions:
        raise ActionSequenceError("trace has no executed actions")
    contexts = {context.step_id: context.step_index for context in trace.step_contexts}
    indexed = []
    for action in trace.executed_actions:
        if action.step_id not in contexts:
            raise ActionSequenceError("executed action references an unknown step")
        indexed.append((contexts[action.step_id], action))
    indices = tuple(index for index, _ in indexed)
    if len(indices) != len(set(indices)) or indices != tuple(sorted(indices)):
        raise ActionSequenceError("executed action steps are duplicate or non-monotonic")
    if indices != tuple(range(indices[0], indices[0] + len(indices))):
        raise ActionSequenceError("executed action sequence contains a step gap")
    predictions = {prediction.prediction_id: prediction for prediction in trace.policy_predictions}
    specs = [predictions[action.prediction_id].action_spec for _, action in indexed if action.prediction_id in predictions]
    if len(specs) != len(indexed):
        raise ActionSequenceError("action specification is unavailable for an executed action")
    identities = [action_spec_identity(spec) for spec in specs]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ActionSequenceError("executed actions use inconsistent action specifications")
    field_name = "applied_action" if source is ActionSource.APPLIED else "requested_action"
    arrays = [getattr(action, field_name) for _, action in indexed]
    dimensions = {array.shape for array in arrays}
    if len(dimensions) != 1:
        raise ActionSequenceError("executed actions have inconsistent dimensions")
    values = np.stack(arrays)
    if not np.all(np.isfinite(values)):
        raise ActionSequenceError("executed actions contain non-finite values")
    return ActionSequence(values, specs[0], source, indices)
