"""Single-episode recorder that finalizes one immutable EpisodeTrace."""

from dataclasses import replace

from ovlab_core.contracts import (
    ActionPrediction,
    EpisodeContext,
    EpisodeTerminalStatus,
    EpisodeTrace,
    PolicyObservation,
    RawPolicyOutput,
    SignalAccess,
    StepContext,
)

from .errors import RecorderError
from .lifecycle import RecorderState


class EpisodeRecorder:
    def __init__(self, recording_policy, clock) -> None:
        self.policy = recording_policy
        self.clock = clock
        self.state = RecorderState.CREATED
        self.context = None
        self.start_timestamp_ns = None
        self.step_contexts = []
        self.observations = []
        self.instructions = []
        self.predictions = []
        self.executed_actions = []
        self.signals = []
        self.metadata = {}
        self._trace = None

    def start(self, context: EpisodeContext, metadata=None) -> None:
        if self.state is not RecorderState.CREATED:
            raise RecorderError("recorder can start exactly once")
        self.context = context
        self.start_timestamp_ns = self.clock.monotonic_ns()
        self.metadata = dict(metadata or {})
        self.metadata["recording_policy_hash"] = self.policy.hash
        self.metadata["omitted"] = tuple(self._omissions())
        self.instructions.append(context.initial_instruction)
        self.state = RecorderState.RECORDING

    def record_observation(self, observation: PolicyObservation, step_index: int) -> None:
        self._require_recording()
        if observation.instruction != self.context.initial_instruction and observation.instruction not in self.instructions:
            self.instructions.append(observation.instruction)
        expected_ids = (self.context.run_id, self.context.task_id, self.context.episode_id)
        context = StepContext(*expected_ids, observation.step_id, step_index, observation.timestamp_ns)
        if self.step_contexts and step_index <= self.step_contexts[-1].step_index:
            raise RecorderError("step indices must be increasing and unique")
        self.step_contexts.append(context)
        if step_index == 0:
            self.start_timestamp_ns = min(self.start_timestamp_ns, observation.timestamp_ns)
        if self.policy.record_policy_observations:
            self.observations.append(self._filter_observation(observation, step_index))

    def record_signals(self, signals) -> None:
        self._require_recording()
        for signal in signals:
            if signal.access is SignalAccess.PRIVILEGED and not self.policy.record_privileged_signals:
                continue
            if signal.access is not SignalAccess.PRIVILEGED and not self.policy.record_evaluation_signals:
                continue
            self.signals.append(signal)

    def record_prediction(self, prediction: ActionPrediction) -> None:
        self._require_recording()
        if prediction.step_id not in {context.step_id for context in self.step_contexts}:
            raise RecorderError("prediction references an unrecorded step")
        if not self.policy.record_raw_policy_output and prediction.raw_output is not None:
            prediction = replace(prediction, raw_output=None)
        self.predictions.append(prediction)

    def record_step(self, step_context: StepContext, result) -> None:
        self._require_recording()
        if not self.step_contexts or self.step_contexts[-1] != step_context:
            raise RecorderError("step result must match the current recorded observation context")
        if any(action.step_id == step_context.step_id for action in self.executed_actions):
            raise RecorderError("one executed action is allowed per benchmark step")
        if result.executed_action.prediction_id not in {prediction.prediction_id for prediction in self.predictions}:
            raise RecorderError("executed action must reference a recorded prediction")
        self.executed_actions.append(result.executed_action)
        self.record_signals(result.evaluation_signals)

    def finalize(self, terminal_status: EpisodeTerminalStatus, metadata=None) -> EpisodeTrace:
        self._require_recording()
        event_timestamps = [self.start_timestamp_ns]
        for values in (self.step_contexts, self.observations, self.instructions, self.predictions, self.executed_actions, self.signals):
            event_timestamps.extend(value.timestamp_ns for value in values)
        end = max(self.clock.monotonic_ns(), *event_timestamps)
        if end < self.start_timestamp_ns:
            raise RecorderError("clock moved backwards")
        self.metadata.update(metadata or {})
        self._trace = EpisodeTrace(
            self.context,
            tuple(self.step_contexts),
            tuple(self.observations),
            tuple(self.instructions),
            tuple(self.predictions),
            tuple(self.executed_actions),
            tuple(sorted(self.signals, key=lambda signal: signal.timestamp_ns)),
            terminal_status,
            self.start_timestamp_ns,
            end,
            self.metadata,
        )
        self.state = RecorderState.FINALIZED
        return self._trace

    @property
    def trace(self):
        if self.state is not RecorderState.FINALIZED:
            raise RecorderError("trace is unavailable before finalization")
        return self._trace

    def close(self):
        self.state = RecorderState.CLOSED

    def _require_recording(self):
        if self.state is not RecorderState.RECORDING:
            raise RecorderError(f"operation requires RECORDING, current state is {self.state.value}")

    def _filter_observation(self, observation, step_index):
        images = observation.images if self.policy.record_image_arrays and step_index % self.policy.image_sampling_stride == 0 else ()
        proprioception = observation.proprioception if self.policy.record_proprioception else ()
        return replace(observation, images=images, proprioception=proprioception)

    def _omissions(self):
        fields = []
        for name in (
            "record_policy_observations", "record_image_arrays", "record_proprioception",
            "record_raw_policy_output", "record_evaluation_signals", "record_privileged_signals",
        ):
            if not getattr(self.policy, name):
                fields.append(name.removeprefix("record_"))
        if self.policy.image_sampling_stride > 1:
            fields.append(f"images_stride_{self.policy.image_sampling_stride}")
        return fields
