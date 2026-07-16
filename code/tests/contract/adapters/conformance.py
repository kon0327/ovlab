"""Black-box conformance assertions reusable by concrete adapter packages."""

import numpy as np

from helpers.contexts import make_episode_context, make_run_context, make_step_context
from ovlab_benchmarks import BenchmarkActionRequest
from ovlab_core import negotiate_capabilities
from ovlab_core.contracts import AdapterState, PredictionId


def assert_benchmark_conformance(adapter, episode=None) -> None:
    run = make_run_context()
    episode = episode or make_episode_context()
    capabilities = adapter.initialize(run)
    assert adapter.state is AdapterState.READY
    assert adapter.list_tasks()
    reset = adapter.reset_episode(episode)
    assert reset.episode_context == episode
    request = BenchmarkActionRequest(
        make_step_context(episode, 0, reset.initial_observation.timestamp_ns),
        PredictionId("conformance-prediction"),
        0,
        np.zeros(capabilities.action_spec.dimension, dtype=capabilities.action_spec.dtype),
        reset.initial_observation.timestamp_ns + 1,
    )
    result = adapter.step(request)
    assert result.step_context == request.step_context
    adapter.close()
    assert adapter.state is AdapterState.CLOSED


def assert_policy_conformance(adapter, benchmark) -> None:
    run = make_run_context()
    episode = make_episode_context()
    benchmark_capabilities = benchmark.initialize(run)
    policy_capabilities = adapter.initialize(run)
    negotiate_capabilities(benchmark_capabilities, policy_capabilities).require_compatible()
    observation = benchmark.reset_episode(episode).initial_observation
    adapter.reset_episode(episode)
    prediction = adapter.predict(observation)
    assert prediction.step_id == observation.step_id
    assert policy_capabilities.minimum_action_horizon <= prediction.horizon
    assert prediction.horizon <= policy_capabilities.maximum_action_horizon
    adapter.end_episode(episode)
    adapter.close()
    benchmark.close()
