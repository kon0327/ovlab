"""Render action semantics from the recorded contract, independently of a model."""
from collections.abc import Mapping


def trace_action_spec(trace):
    if not trace.policy_predictions:
        return None
    spec = trace.policy_predictions[0].action_spec
    return {"dimension": spec.dimension, "representation": spec.representation.value,
            "translation_indices": list(spec.translation_indices), "rotation_indices": list(spec.rotation_indices),
            "gripper_indices": list(spec.gripper_indices), "rotation_representation": spec.rotation_representation.value,
            "gripper_convention": spec.gripper_convention.value, "units": list(spec.units), "dtype": spec.dtype,
            "control_frequency_hz": spec.control_frequency_hz}


def component_names(dimension, spec=None):
    if not isinstance(spec, Mapping):
        return tuple(f"component_{index}" for index in range(dimension))
    joint = spec.get("representation") in {"joint_delta", "joint_position"}
    names = [f"joint_{index}" if joint else f"component_{index}" for index in range(dimension)]
    for key, labels in (("translation_indices", ("tx", "ty", "tz")),
                        ("rotation_indices", ("rx", "ry", "rz") if spec.get("rotation_representation") in {"axis_angle", "euler_xyz"}
                         else ("qx", "qy", "qz", "qw") if spec.get("rotation_representation") == "quaternion_xyzw"
                         else ("qw", "qx", "qy", "qz") if spec.get("rotation_representation") == "quaternion_wxyz"
                         else tuple(f"rotation_{index}" for index in range(len(spec.get("rotation_indices", ())))))):
        for index, label in zip(spec.get(key, ()), labels):
            if 0 <= index < dimension:
                names[index] = label
    indices = spec.get("gripper_indices", ())
    for number, index in enumerate(indices):
        if 0 <= index < dimension:
            names[index] = "gripper" if len(indices) == 1 else f"gripper_{number}"
    return tuple(names)


def shared_action_spec(episodes):
    specs = [episode.get("action_spec") for episode in episodes]
    return specs[0] if specs and all(spec == specs[0] for spec in specs) else None
