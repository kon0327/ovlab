"""Synchronous deterministic episode execution."""

import hashlib

from ovlab_benchmarks import BenchmarkActionRequest
from ovlab_core.contracts import (
    EpisodeContext, EpisodeId, EpisodeTerminalStatus, Instruction, InstructionId,
    InstructionSource,
)

from .errors import ExperimentExecutionError
from .plan import ActionExecutionMode
from .recorder import EpisodeRecorder


def make_episode_context(plan, task, task_order_index, rollout_index, clock):
    seed = plan.episode_seed(task.task_id, task_order_index, rollout_index)
    digest = hashlib.sha256(f"{plan.run_context.run_id}\0{task.task_id}\0{rollout_index}".encode()).hexdigest()[:16]
    episode_id = EpisodeId(f"episode-{task_order_index:04d}-{rollout_index:04d}-{digest}")
    timestamp = clock.monotonic_ns()
    instruction = Instruction(
        InstructionId(f"instruction-{digest}"), task.natural_language_instruction, timestamp, InstructionSource.BENCHMARK
    )
    return EpisodeContext(plan.run_context.run_id, task.task_id, episode_id, rollout_index, seed, instruction)


def execute_episode(plan, task, task_order_index, rollout_index, benchmark, policy, clock):
    context = make_episode_context(plan, task, task_order_index, rollout_index, clock)
    recorder = EpisodeRecorder(plan.trace_recording_policy, clock)
    recorder.start(context, {"task_maximum_steps": task.maximum_steps, "plan_hash": plan.hash})
    policy_active = False
    failure_domain = "runner"
    try:
        failure_domain = "policy"
        policy.reset_episode(context)
        policy_active = True
        failure_domain = "benchmark"
        reset = benchmark.reset_episode(context)
        observation = reset.initial_observation
        recorder.record_observation(observation, 0)
        recorder.record_signals(reset.evaluation_signals)
        pending = None
        chunk_index = 0
        execution_limit = 0
        step_index = 0
        maximum_steps = task.maximum_steps or plan.default_maximum_episode_steps
        while step_index < maximum_steps:
            needs_prediction = pending is None or chunk_index >= execution_limit
            if needs_prediction:
                failure_domain = "policy"
                pending = policy.predict(observation)
                failure_domain = "runner"
                recorder.record_prediction(pending)
                if pending.horizon < 1 or pending.horizon > policy.capabilities.maximum_action_horizon:
                    raise ExperimentExecutionError("prediction horizon violates policy capabilities")
                if plan.action_execution_policy.mode is ActionExecutionMode.RECEDING_HORIZON:
                    execution_limit = 1
                elif plan.action_execution_policy.mode is ActionExecutionMode.OPEN_LOOP_CHUNK:
                    execution_limit = pending.horizon
                else:
                    interval = plan.action_execution_policy.replan_interval
                    if interval > pending.horizon:
                        raise ExperimentExecutionError("fixed replan interval exceeds returned prediction horizon")
                    execution_limit = interval
                chunk_index = 0
            if chunk_index >= pending.horizon:
                raise ExperimentExecutionError("runner attempted to execute beyond prediction horizon")
            step_context = recorder.step_contexts[-1]
            request_timestamp = clock.monotonic_ns()
            request = BenchmarkActionRequest(
                step_context, pending.prediction_id, chunk_index, pending.actions[chunk_index], request_timestamp,
                {"closed_loop_step_started_ns": request_timestamp, "action_execution_mode": plan.action_execution_policy.mode.value},
            )
            failure_domain = "benchmark"
            result = benchmark.step(request)
            failure_domain = "runner"
            recorder.record_step(step_context, result)
            step_index += 1
            if result.terminated or result.truncated:
                terminal = _terminal_status(result)
                policy.end_episode(context)
                policy_active = False
                return recorder.finalize(terminal, {"executed_step_count": step_index}), None
            if step_index >= maximum_steps:
                raise ExperimentExecutionError("benchmark did not terminate or truncate at its declared task maximum")
            observation = result.next_observation
            recorder.record_observation(observation, step_index)
            chunk_index += 1
            if observation.instruction != pending_step_instruction(pending, recorder):
                pending = None
        raise ExperimentExecutionError("episode loop exited without terminal result")
    except KeyboardInterrupt:
        if policy_active:
            _safe_end(policy, context)
        _safe_abort(benchmark)
        return recorder.finalize(EpisodeTerminalStatus.ABORTED, {"failure_type": "KeyboardInterrupt"}), KeyboardInterrupt()
    except Exception as exc:
        if policy_active:
            _safe_end(policy, context)
        _safe_abort(benchmark)
        status = EpisodeTerminalStatus.POLICY_ERROR if failure_domain in ("policy", "runner") else EpisodeTerminalStatus.BENCHMARK_ERROR
        return recorder.finalize(status, {"failure_type": type(exc).__name__, "failure_message": str(exc)[:240]}), exc


def pending_step_instruction(prediction, recorder):
    for observation in reversed(recorder.observations):
        if observation.step_id == prediction.step_id:
            return observation.instruction
    return recorder.context.initial_instruction


def _terminal_status(result):
    if result.success is True: return EpisodeTerminalStatus.SUCCESS
    if result.truncated: return EpisodeTerminalStatus.TIME_LIMIT
    return EpisodeTerminalStatus.FAILURE


def _safe_end(policy, context):
    try: policy.end_episode(context)
    except Exception: pass


def _safe_abort(benchmark):
    try:
        if benchmark.state.value == "episode_active":
            benchmark.abort_episode()
    except Exception:
        pass
