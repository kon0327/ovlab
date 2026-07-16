# ovlab-runner

`ovlab-runner` is the synchronous, single-process orchestration layer connecting
one `BenchmarkAdapter`, one `PolicyAdapter`, immutable traces, offline metric
plug-ins, and an injected artifact store. It contains no concrete LIBERO or
OpenVLA dependency and does not implement RPC, CLI composition, Docker,
parallel execution, reporting, or resume/retry.

## Lifecycle and connection

The runner lifecycle is `CREATED → CONNECTED → RUNNING → COMPLETED|FAILED →
CLOSED`. `connect()` initializes both adapters, negotiates capabilities,
validates selected tasks, resolves metric IDs, checks recording-policy versus
signal requirements, and produces an immutable `ConnectionReport`. No rollout
starts after an incompatibility. `close()` is idempotent; `run()` releases both
adapters in `finally` while preserving its completed or failed state.

`ExperimentPlan` fixes the run context, ordered task IDs, rollout count, base
episode seed, maximum-step fallback, action execution policy, enabled and
required metrics, typed metric configurations, unavailable/error policies,
recording policy, artifact settings, and metadata. Its canonical JSON form and
SHA-256 hash are persisted. Episode seeds hash the base seed, stable task
identity/order, and rollout index without global random state.

## Action chunks

- `RECEDING_HORIZON` predicts every step and executes index 0.
- `OPEN_LOOP_CHUNK` executes indices `0..H-1`, then replans.
- `FIXED_REPLAN_INTERVAL` executes at most the configured first `K` actions.

Prediction ID and selected chunk index are retained in every `ExecutedAction`.
No element beyond `H` is executed. Termination discards pending elements
immediately, and an instruction change invalidates a pending chunk. Step indices
start at zero. The task descriptor's declared maximum is authoritative; the
runner never adds an unrecorded boundary step.

## Recording and failures

One `EpisodeRecorder` owns one episode. It checks IDs, increasing unique step
indices, prediction/action relationships, one executed action per benchmark
step, monotonic final bounds, and single finalization. Predictions may remain
unexecuted and one prediction may supply several executed chunk actions.
Evaluation and privileged signals remain separate from `PolicyObservation`.

The recording policy explicitly controls observations, image arrays,
proprioception, raw policy output, evaluation signals, privileged signals, and
image stride. Mandatory lifecycle, prediction, and executed-action evidence is
always retained; omissions and the policy hash are written into trace metadata.

Policy failures finalize `POLICY_ERROR` without a fake action. Benchmark
failures finalize `BENCHMARK_ERROR`. Interruptions finalize `ABORTED` when safe
and are re-raised. `STOP_RUN` stops immediately, `CONTINUE_TASK` attempts the
next rollout of the same task, and `CONTINUE_RUN` skips its remaining rollouts
and proceeds to the next selected task.
The backward-compatible `BenchmarkAdapter.abort_episode()` hook releases active
benchmark resources so continuation is possible after a policy failure.

## Metrics and success

The raw trace is finalized and persisted before metrics run. Metric errors
cannot delete or rewrite it. Episode results are stored separately, then
homogeneous results are macro-aggregated per task. The benchmark provides
authoritative success, the runner records it, and `task.success` validates it;
the runner never infers success from reward.

Inference duration comes from `ActionPrediction`. Runner call timing is
closed-loop telemetry. Neither is labelled RPC latency.

## Artifacts

The filesystem store derives hashed safe keys rather than using IDs as paths:

```text
runs/<run-key>/
├── manifest.started.json
├── plan.json
├── connection.json
├── tasks/<task-key>/episodes/<episode-key>/
│   ├── trace.json
│   ├── events.jsonl
│   ├── arrays/*.npy
│   ├── trace.finalized.json
│   ├── finalized.json
│   └── metrics.episode.json
├── tasks/<task-key>/metrics.task.json
└── manifest.completed.json | manifest.failed.json
```

Structured fields use deterministic JSON, ordered event references use JSONL,
and numerical arrays use non-pickle `.npy` files with relative references,
dtype/shape declarations, and SHA-256 checksums. Reads reject missing, modified,
or path-traversing arrays. Writes use temporary sibling files and atomic rename;
started manifests, traces, results, and final manifests are never overwritten.
Partial directories remain distinguishable and no cleanup is automatic.

The optional real test uses pinned LIBERO plus a deterministic no-checkpoint
policy:

```bash
OVLAB_RUN_LIBERO_RUNNER=1 MUJOCO_GL=egl \
  conda run -n openvla-oft deploy/scripts/test.sh \
  code/tests/integration/runner/test_libero_runner_manual.py -q -s
```
