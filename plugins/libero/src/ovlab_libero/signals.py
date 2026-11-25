"""Declared evaluation and cross-task privileged LIBERO signals."""

from collections.abc import Mapping

import numpy as np

from ovlab_core.contracts import SignalAccess, SignalRegistry, SignalSpec, SignalValue, StepId

from .errors import LiberoObservationError
from .settings import LiberoObservationProfile


def signal_registry(
    profile: LiberoObservationProfile,
    *,
    simulator_state_dimension: int | None = None,
) -> SignalRegistry:
    specs = [
        SignalSpec("benchmark.task_success", "bool", (), "", SignalAccess.EVALUATION_ONLY, "LIBERO goal predicate"),
        SignalSpec("benchmark.reward", "float64", (), "", SignalAccess.EVALUATION_ONLY, "Native reward"),
        SignalSpec("episode.terminated", "bool", (), "", SignalAccess.EVALUATION_ONLY, "Native/success termination"),
        SignalSpec("episode.truncated", "bool", (), "", SignalAccess.EVALUATION_ONLY, "OVLAB step limit"),
        SignalSpec("episode.native_step_index", "int64", (), "", SignalAccess.EVALUATION_ONLY, "Policy step index"),
        SignalSpec("episode.initial_state_index", "int64", (), "", SignalAccess.EVALUATION_ONLY, "Selected state"),
    ]
    if profile is not LiberoObservationProfile.RGB_PROPRIOCEPTION:
        specs.extend(
            (
                SignalSpec("robot.eef.position", "float32", (3,), "m", SignalAccess.PRIVILEGED, "Ground-truth EEF position"),
                SignalSpec("robot.eef.orientation_xyzw", "float32", (4,), "unitless", SignalAccess.PRIVILEGED, "Ground-truth EEF quaternion"),
                SignalSpec("robot.gripper.joint_position", "float32", (2,), "rad", SignalAccess.PRIVILEGED, "Ground-truth gripper joints"),
            )
        )
    if simulator_state_dimension is not None:
        specs.append(SignalSpec(
            "simulator.state",
            "float64",
            (simulator_state_dimension,),
            "native_state",
            SignalAccess.PRIVILEGED,
            "Flattened MuJoCo state for deterministic replay auditing",
            optional=True,
        ))
    return SignalRegistry(specs)


def map_signals(
    raw: Mapping[str, object],
    profile: LiberoObservationProfile,
    step_id: StepId,
    timestamp_ns: int,
    *,
    reward: float,
    success: bool,
    terminated: bool,
    truncated: bool,
    native_step_index: int,
    initial_state_index: int,
    simulator_state: np.ndarray | None = None,
) -> tuple[SignalValue, ...]:
    values = [
        SignalValue("benchmark.task_success", success, timestamp_ns, "libero", step_id, access=SignalAccess.EVALUATION_ONLY),
        SignalValue("benchmark.reward", float(reward), timestamp_ns, "libero", step_id, access=SignalAccess.EVALUATION_ONLY),
        SignalValue("episode.terminated", terminated, timestamp_ns, "ovlab-libero", step_id, access=SignalAccess.EVALUATION_ONLY),
        SignalValue("episode.truncated", truncated, timestamp_ns, "ovlab-libero", step_id, access=SignalAccess.EVALUATION_ONLY),
        SignalValue("episode.native_step_index", native_step_index, timestamp_ns, "ovlab-libero", step_id, access=SignalAccess.EVALUATION_ONLY),
        SignalValue("episode.initial_state_index", initial_state_index, timestamp_ns, "ovlab-libero", step_id, access=SignalAccess.EVALUATION_ONLY),
    ]
    if profile is not LiberoObservationProfile.RGB_PROPRIOCEPTION:
        for signal_name, key, shape in (
            ("robot.eef.position", "robot0_eef_pos", (3,)),
            ("robot.eef.orientation_xyzw", "robot0_eef_quat", (4,)),
            ("robot.gripper.joint_position", "robot0_gripper_qpos", (2,)),
        ):
            if key not in raw:
                raise LiberoObservationError(f"required privileged native signal {key!r} is missing")
            value = np.asarray(raw[key], dtype=np.float32)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise LiberoObservationError(f"privileged native signal {key!r} is invalid")
            values.append(
                SignalValue(signal_name, value, timestamp_ns, "libero", step_id, access=SignalAccess.PRIVILEGED)
            )
    if simulator_state is not None:
        state = np.asarray(simulator_state, dtype=np.float64).reshape(-1)
        if state.size == 0 or not np.all(np.isfinite(state)):
            raise LiberoObservationError("native simulator state is invalid")
        values.append(SignalValue(
            "simulator.state",
            state,
            timestamp_ns,
            "libero-mujoco",
            step_id,
            metadata={"comparison_tolerance": {"absolute": 1.0e-9, "relative": 1.0e-9}},
            access=SignalAccess.PRIVILEGED,
        ))
    return tuple(sorted(values, key=lambda value: value.name))
