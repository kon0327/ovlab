"""Opt-in real LIBERO benchmark plus deterministic dependency-free policy."""

import os

import numpy as np
import pytest

from ovlab_benchmarks.libero import LiberoAdapterSettings, LiberoBenchmarkAdapter
from ovlab_benchmarks.libero.actions import libero_action_spec
from ovlab_core.contracts import (
    OVLAB_CONTRACT_VERSION, ActionPrediction, EpisodeContext, ImageObservationSpec,
    ObservationRequirements, PolicyCapabilities, PredictionId, RunContext, RunId,
    TaskId,
)
from ovlab_metrics import MetricRegistry
from ovlab_policy_sdk import PolicyAdapter
from ovlab_runner import (
    DeterministicClock, ExperimentPlan, ExperimentRunner, InMemoryRunArtifactStore,
)

pytestmark = [
    pytest.mark.libero,
    pytest.mark.gpu,
    pytest.mark.manual,
    pytest.mark.skipif(os.environ.get("OVLAB_RUN_LIBERO_RUNNER") != "1", reason="explicit real LIBERO runner opt-in required"),
]


class NoOpLiberoPolicy(PolicyAdapter):
    def _initialize(self, run_context):
        from ovlab_core.contracts import ColorSpace, ImageEncoding
        image = ImageObservationSpec("camera.primary.rgb", ((64, 64, 3),), "uint8", (ImageEncoding.RAW,), (ColorSpace.RGB,))
        return PolicyCapabilities(
            "deterministic-noop", "1.0.0", OVLAB_CONTRACT_VERSION,
            ObservationRequirements((image,), (), 1, 1, 0, 0), libero_action_spec(),
            True, False, 1, 1, False, True, False,
        )

    def _reset_episode(self, episode_context): self.index = 0
    def _predict(self, observation):
        action = np.array([[0, 0, 0, 0, 0, 0, -1]], dtype=np.float32)
        result = ActionPrediction(PredictionId(f"noop-{self.index}"), observation.step_id, action, self.capabilities.output_action_spec, observation.timestamp_ns + 1, 1, 1)
        self.index += 1
        return result
    def _close(self): pass


def test_real_libero_runner_produces_trace_and_metrics() -> None:
    benchmark = LiberoBenchmarkAdapter(LiberoAdapterSettings(task_indices=(0,), camera_width=64, camera_height=64, maximum_episode_steps=1, initialization_settling_steps=1))
    plan = ExperimentPlan(
        RunContext(RunId("real-libero-runner"), 1, "real LIBERO runner smoke", 0),
        (TaskId("libero/spatial/0"),), 1, 0, 1,
        enabled_metric_ids=("task.success", "system.inference_latency"),
    )
    store = InMemoryRunArtifactStore()
    runner = ExperimentRunner(plan, benchmark, NoOpLiberoPolicy(), store, clock=DeterministicClock(), metric_registry=MetricRegistry.default())
    runner.connect(); runner.run()
    episode = next(iter(store.runs["real-libero-runner"]["episodes"].values()))
    assert episode["trace"].executed_actions
    assert episode["metrics"]
