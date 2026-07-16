# ovlab-benchmarks

`ovlab-benchmarks` defines the synchronous boundary implemented by simulation
benchmarks. A concrete adapter declares capabilities, lists stable task
descriptors, resets seeded episodes, accepts exactly one selected action per
step, and returns the action actually executed together with the next
policy-visible observation and a separate evaluation-signal channel.

Adapters use the lifecycle `created → ready → episode_active → ready → closed`.
Terminal or truncated steps return the benchmark to `ready`; `close()` is
idempotent. Implementations subclass `BenchmarkAdapter` and implement its
protected hooks. Public methods enforce lifecycle and identifier consistency.

Evaluation-only and privileged state must be declared in `SignalRegistry` and
returned as `SignalValue` objects. It must never be inserted into
`PolicyObservation`. The runner must negotiate the benchmark capabilities with
the selected policy before reset.

The package is transport-neutral and does not import simulator or external
repository code. Concrete integrations belong in later benchmark-specific
packages.

The package dependency rule is:

```text
ovlab-benchmarks ─┐
                  ├──> ovlab-core
ovlab-policy-sdk ─┘
```

There is no dependency between the two adapter packages.
