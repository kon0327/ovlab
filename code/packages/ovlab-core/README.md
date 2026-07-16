# ovlab-core

`ovlab-core` provides the dependency-light data contracts shared by OpenVLABenchmark components. It defines stable in-memory boundaries for observations, instructions, evaluation signals, action lifecycles, lifecycle identifiers, and immutable episode traces. The package requires only Python 3.10+ and NumPy.

The Python package version and contract schema version are separate. Schema compatibility is identified centrally by `OVLAB_CONTRACT_VERSION`.

## Information boundary

`PolicyObservation` contains only inputs intentionally visible to a policy: the current instruction, named images, and named proprioceptive observations. Evaluation data uses `SignalSpec`, `SignalValue`, and `SignalRegistry`. Signals explicitly declare whether they are policy-visible, evaluation-only, or privileged. There is no helper that copies arbitrary benchmark state or evaluation signals into a policy observation.

This structural separation prevents simulator-only state, success predicates, object poses, and similar privileged information from being passed to a policy accidentally. Benchmark adapters construct both channels explicitly and capability negotiation rejects attempts to request privileged signals as policy inputs.

## Adapter capabilities and compatibility

`BenchmarkCapabilities` declares produced observations, accepted actions,
evaluation signals, task suites, and benchmark behavior. `PolicyCapabilities`
declares required observations, produced actions, supported action horizons, and
policy behavior. `negotiate_capabilities()` returns a stable, inspectable report;
`require_compatible()` raises when the report contains errors.

## Action lifecycle

OVLAB keeps three action stages distinct:

1. `RawPolicyOutput` stores transport-safe model-native output before decoding. It is evidence, not an executable command.
2. `ActionPrediction` stores a decoded numeric action and an explicit `ActionSpec`. Its canonical shape is always `[H, D]`; a single action is `[1, D]`, while an action chunk has `H > 1`.
3. `ExecutedAction` stores the selected requested action and the action actually applied to the environment. Any difference requires a modification reason.

`ActionSpec` declares dimensions, semantic indices, representations, conventions, units, limits, dtype, and control frequency. Contracts validate these declarations but never normalize, clip, decode, or execute actions.

## Time and lifecycle

Within-run ordering uses non-negative monotonic integer nanoseconds named `timestamp_ns`. Wall-clock values are named explicitly, such as `created_wall_time_utc_ns`; contracts never infer or convert between the two time domains.

Run, task, episode, step, instruction, and prediction identifiers are immutable string value objects. They validate caller-supplied values but never generate identifiers. ID generation belongs to the future runner.

## Effective immutability

Contracts use frozen dataclasses and immutable tuples. Metadata is recursively normalized into read-only mappings and tuples and accepts only JSON-compatible values. Numerical data is copied into bytes-backed read-only NumPy arrays so changes to caller-owned arrays cannot mutate a contract.

NumPy arrays are an in-memory boundary. A future RPC or persistence layer will provide an explicit array codec; serialization logic is intentionally not duplicated across contracts.

## Episode traces

`EpisodeTrace` is the immutable raw evidence for an episode. It contains lifecycle contexts, policy-visible observations, instruction events, decoded predictions, executed actions, evaluation signals, terminal state, and timestamps. It does not contain derived metric results. Later metric packages will recompute results from stored traces without modifying the original trace.

Trace construction validates deterministic event ordering and verifies event step references through `StepContext`, including run, task, and episode membership.

## Deferred work

The adapter-contract phase intentionally does not implement:

- concrete benchmark integrations or benchmark-specific fields,
- concrete policy backends or model decoding,
- metrics and failure analysis,
- RPC, protobuf, or other transport codecs,
- JSONL, Parquet, or database persistence,
- configuration loading,
- runner or experiment execution,
- Docker and deployment behavior.

## Development

Run the focused contract tests without installing the package globally:

```bash
PYTHONPATH=code/packages/ovlab-core/src \
  python -m pytest code/tests/unit/core/contracts -q
```
