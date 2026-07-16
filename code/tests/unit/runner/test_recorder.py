"""Recorder lifecycle, filtering, and immutable trace tests."""

import pytest

from helpers.contexts import make_episode_context, make_run_context
from helpers.mock_benchmark import MockBenchmark
from helpers.mock_policy import MockPolicy
from ovlab_benchmarks import BenchmarkActionRequest
from ovlab_core.contracts import EpisodeTerminalStatus, PredictionId, SignalAccess
from ovlab_runner import DeterministicClock, EpisodeRecorder, RecorderError, TraceRecordingPolicy


def test_recorder_builds_immutable_trace_and_filters_privileged_signals() -> None:
    episode = make_episode_context()
    benchmark, policy = MockBenchmark(maximum_steps=1), MockPolicy()
    benchmark.initialize(make_run_context())
    policy.initialize(make_run_context())
    policy.reset_episode(episode)
    reset = benchmark.reset_episode(episode)
    recorder = EpisodeRecorder(TraceRecordingPolicy(record_privileged_signals=False), DeterministicClock())
    recorder.start(episode)
    recorder.record_observation(reset.initial_observation, 0)
    recorder.record_signals(reset.evaluation_signals)
    prediction = policy.predict(reset.initial_observation)
    recorder.record_prediction(prediction)
    context = recorder.step_contexts[-1]
    result = benchmark.step(BenchmarkActionRequest(context, prediction.prediction_id, 0, prediction.actions[0], 2))
    recorder.record_step(context, result)
    trace = recorder.finalize(EpisodeTerminalStatus.SUCCESS)
    assert len(trace.executed_actions) == 1
    assert all(signal.access is not SignalAccess.PRIVILEGED for signal in trace.evaluation_signals)
    assert not trace.executed_actions[0].applied_action.flags.writeable
    with pytest.raises(RecorderError): recorder.finalize(EpisodeTerminalStatus.SUCCESS)


def test_recorder_rejects_duplicate_steps_and_unrecorded_prediction_relationship() -> None:
    episode = make_episode_context()
    benchmark = MockBenchmark()
    benchmark.initialize(make_run_context())
    reset = benchmark.reset_episode(episode)
    recorder = EpisodeRecorder(TraceRecordingPolicy(), DeterministicClock())
    recorder.start(episode)
    recorder.record_observation(reset.initial_observation, 0)
    with pytest.raises(RecorderError): recorder.record_observation(reset.initial_observation, 0)
