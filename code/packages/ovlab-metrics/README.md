# ovlab-metrics

`ovlab-metrics` recomputes deterministic results offline from immutable
`EpisodeTrace` evidence. It does not rerun policies or benchmarks, perform file
I/O, select storage formats, or import LIBERO, OpenVLA, Torch, Pandas, SciPy, or
reporting libraries.

## Plug-in architecture and taxonomy

Episode plug-ins expose an immutable descriptor, inspectable requirements, a
typed default configuration, and a pure `evaluate(trace, config)` operation.
Task plug-ins aggregate homogeneous episode results. `MetricRegistry` is
explicit and deterministic; entry-point discovery is deferred.

Metric levels are exactly `TASK`, `ACTION`, and `SYSTEM`. Failure indicators
remain a cross-cutting `is_failure_indicator` property and are not a fourth
level.

| Metric ID | Level | Failure indicator | Scope | Required data | Unit |
|---|---|---:|---|---|---|
| `task.success` | task | no | episode | authoritative success signal, terminal status | boolean |
| `task.success_rate` | task | no | task | episode success results | ratio |
| `action.variance` | action | no | episode/task | executed actions and action spec | action unit² |
| `action.smoothness_1` | action | no | episode/task | ≥2 executed actions | action unit/control step |
| `action.smoothness_2` | action | no | episode/task | ≥3 executed actions | action unit/control step² |
| `failure.invalid_prediction_rate` | system | yes | episode/task | stored prediction validity | ratio |
| `failure.action_modification_rate` | action | yes | episode/task | requested and applied commands | ratio |
| `failure.repeated_no_op_rate` | action | yes | episode/task | contiguous executed commands | ratio |
| `failure.gripper_flicker_rate` | action | yes | episode/task | action spec and gripper commands | ratio |
| `failure.collision_rate` | action | yes | episode/task | semantic `safety.collision_event` | ratio |
| `system.inference_latency` | system | no | episode/task | prediction inference duration | ms |

## Requirements and unavailable results

Requirements describe trace fields, canonical signals, shapes, dtypes, allowed
access classes, and minimum sample counts without running a metric. Expected
missing data becomes `UNAVAILABLE`; insufficient samples become
`INSUFFICIENT_DATA`; semantic irrelevance becomes `NOT_APPLICABLE`. None of
these states has a numeric value. Unexpected failures become concise `ERROR`
results in normal mode and are raised in strict mode.

`SignalValue.access` is retained in new traces so an offline metric can
explicitly consume evaluation-only or privileged evidence. Older traces with an
unknown access class remain usable but produce a requirement warning.

## Executed action sequence and formulas

Action metrics default to `APPLIED`, the exact command handed to the benchmark.
`REQUESTED` is selectable through typed configuration. Only `ExecutedAction`
entries participate: unselected action-chunk elements never affect a metric.
Sequences must have one consistent action specification, increasing contiguous
step indices, finite values, and a constant dimension.

For `T` commands with dimension `D`, population action variance is:

```text
Var_d = (1/T) Σ_t (a_t,d - mean_d)²
Var_a = (1/D) Σ_d Var_d
```

The discrete command-space smoothness metrics are:

```text
Smooth_1 = (1/(T-1)) Σ_t ||a_t - a_(t-1)||₂
Smooth_2 = (1/(T-2)) Σ_t ||a_t - 2a_(t-1) + a_(t-2)||₂
```

They are not physical velocity or acceleration and do not divide by elapsed
time. Dimensions are not implicitly normalized, so comparisons require the
same action specification.

## Success and task aggregation

The authoritative `benchmark.task_success` signal determines episode success;
terminal status is a consistency check. A contradiction produces `ERROR`.
Success-rate eligibility includes success, failure, time limit, policy error,
and—by configurable default—policy-attributable abort. Benchmark errors are
excluded. Because the current terminal contract cannot encode abort attribution,
the configuration decision is preserved in diagnostics rather than inferred.

Action metrics use macro aggregation: first one value per episode, then mean,
population standard deviation (`ddof=0`), median, minimum, and maximum. The
engine rejects mixed run/task IDs, versions, configuration hashes, action
sources, or action specifications and never silently drops `ERROR` results.

## Failure and system semantics

Repeated no-op thresholds, modification tolerances, and gripper flicker rules
are typed and hashed. Binary gripper deadbands are centered on their activation
threshold; signed conventions use positive/negative activation thresholds.
Collision rate consumes only the explicit semantic boolean
`safety.collision_event`. LIBERO currently does not provide it, so collision
rate is correctly unavailable rather than inferred from intended MuJoCo
contacts.

Inference latency uses only `ActionPrediction.inference_duration_ns`, converts
to milliseconds, and reports mean, median, linear-interpolated p95, extrema,
population standard deviation, and count. RPC round-trip and closed-loop step
latencies are distinct future metrics and are never estimated here.

Configurations have a canonical JSON representation and deterministic SHA-256
hash embedded in every result. This allows results to be recomputed from raw
immutable traces and prevents incompatible aggregation. Reference-run
selection and acceptance envelopes remain explicitly deferred in `reference/`.
