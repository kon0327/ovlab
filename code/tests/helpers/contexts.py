"""Deterministic lifecycle contexts shared by adapter tests."""

from ovlab_core.contracts import (
    EpisodeContext,
    EpisodeId,
    Instruction,
    InstructionId,
    InstructionSource,
    RunContext,
    RunId,
    StepContext,
    StepId,
    TaskId,
)


def make_run_context(*, run_id: str = "test-run", seed: int = 7) -> RunContext:
    return RunContext(RunId(run_id), 1, "adapter-contract-tests", seed)


def make_episode_context(
    *, run_id: str = "test-run", episode_id: str = "episode-0", seed: int = 11
) -> EpisodeContext:
    instruction = Instruction(
        InstructionId(f"{episode_id}-instruction"),
        "move deterministically",
        2,
        InstructionSource.BENCHMARK,
    )
    return EpisodeContext(
        RunId(run_id), TaskId("mock-task-0"), EpisodeId(episode_id), 0, seed, instruction
    )


def make_step_context(episode: EpisodeContext, step_index: int, timestamp_ns: int) -> StepContext:
    return StepContext(
        episode.run_id,
        episode.task_id,
        episode.episode_id,
        StepId(f"{episode.episode_id}-step-{step_index}"),
        step_index,
        timestamp_ns,
    )
