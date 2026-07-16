"""Small fake of the native LIBERO surface consumed by the adapter."""

from types import SimpleNamespace

import numpy as np

from ovlab_benchmarks.libero.tasks import NativeTaskRecord, resolve_native_suite
from ovlab_core.contracts import (
    EpisodeContext,
    EpisodeId,
    Instruction,
    InstructionId,
    InstructionSource,
    RunId,
    TaskId,
)


def fake_raw_observation(height: int = 4, width: int = 5) -> dict[str, np.ndarray]:
    primary = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    return {
        "agentview_image": primary,
        "robot0_eye_in_hand_image": np.full((height, width, 3), 17, dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.01, -0.01], dtype=np.float32),
    }


def fake_libero_episode(*, rollout_index=0, instruction="perform native task 0") -> EpisodeContext:
    return EpisodeContext(
        RunId("test-run"),
        TaskId("libero/spatial/0"),
        EpisodeId("libero-episode"),
        rollout_index,
        11,
        Instruction(InstructionId("instruction"), instruction, 2, InstructionSource.BENCHMARK),
    )


class FakeLiberoEnvironment:
    def __init__(self, *, height=4, width=5, success_after=None) -> None:
        self.action_spec = (np.full(7, -1.0), np.full(7, 1.0))
        self.raw = fake_raw_observation(height, width)
        self.success_after = success_after
        self.steps = 0
        self.actions = []
        self.closed = False
        self.selected_state = None

    def reset(self):
        return self.raw

    def set_init_state(self, state):
        self.selected_state = state
        return self.raw

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.steps += 1
        success = self.check_success()
        return self.raw, float(self.steps) / 10.0, success, {}

    def check_success(self):
        return self.success_after is not None and self.steps >= self.success_after

    def close(self):
        self.closed = True


class FakeLiberoBackend:
    def __init__(self, *, task_count=2, state_count=3, success_after=None) -> None:
        self.task_count = task_count
        self.state_count = state_count
        self.success_after = success_after
        self.environments = []

    def discover_suite(self, suite_name):
        native_suite = resolve_native_suite(suite_name)
        return tuple(
            NativeTaskRecord(
                suite_name,
                native_suite,
                index,
                f"native_task_{index}",
                f"perform native task {index}",
                native_suite,
                f"native_task_{index}.bddl",
                tuple(f"state-{index}-{state}" for state in range(self.state_count)),
                SimpleNamespace(name=f"native_task_{index}"),
            )
            for index in range(self.task_count)
        )

    def create_environment(self, task, settings):
        environment = FakeLiberoEnvironment(
            height=settings.camera_height,
            width=settings.camera_width,
            success_after=self.success_after,
        )
        self.environments.append(environment)
        return environment
