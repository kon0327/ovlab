"""Stable LIBERO suite names, task records, and deterministic state selection."""

from dataclasses import dataclass

import numpy as np

from ovlab_benchmarks import TaskDescriptor
from ovlab_core.contracts import EpisodeContext, TaskId

from .errors import LiberoConfigurationError
from .settings import InitialStateSelection

LIBERO_PINNED_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"

SUITE_IDENTIFIERS = {
    "LIBERO-Spatial": "libero_spatial",
    "LIBERO-Object": "libero_object",
    "LIBERO-Goal": "libero_goal",
    "LIBERO-10": "libero_10",
}


@dataclass(frozen=True, slots=True)
class NativeTaskRecord:
    suite_name: str
    native_suite_name: str
    task_index: int
    name: str
    language: str
    problem_folder: str
    bddl_file: str
    initial_states: tuple[object, ...]
    native_task: object

    @property
    def task_id(self) -> TaskId:
        slug = self.native_suite_name.removeprefix("libero_").replace("_", "-")
        return TaskId(f"libero/{slug}/{self.task_index}")

    def descriptor(self, maximum_steps: int) -> TaskDescriptor:
        return TaskDescriptor(
            self.suite_name,
            self.task_id,
            self.name,
            self.task_index,
            self.language,
            maximum_steps,
            {
                "native_suite": self.native_suite_name,
                "native_task_reference": f"{self.problem_folder}/{self.bddl_file}",
                "available_initial_state_count": len(self.initial_states),
                "libero_commit": LIBERO_PINNED_COMMIT,
            },
        )


def resolve_native_suite(suite_name: str) -> str:
    try:
        return SUITE_IDENTIFIERS[suite_name]
    except KeyError as exc:
        supported = ", ".join(SUITE_IDENTIFIERS)
        raise LiberoConfigurationError(f"unknown LIBERO suite {suite_name!r}; supported: {supported}") from exc


def select_initial_state_index(
    episode: EpisodeContext,
    count: int,
    strategy: InitialStateSelection,
    base_seed: int,
) -> int:
    if count <= 0:
        raise LiberoConfigurationError("the selected task has no initial states")
    if strategy is InitialStateSelection.ROLLOUT_INDEX:
        return episode.rollout_index % count
    rng = np.random.default_rng(np.random.SeedSequence((base_seed, episode.seed, episode.rollout_index)))
    return int(rng.integers(0, count))
