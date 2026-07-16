# Adapter contracts and capability negotiation

OVLAB keeps benchmark mechanics and policy inference behind two small,
synchronous interfaces. This local boundary is the semantic source of truth for
future RPC protocols: transports may encode it, but must not redefine it.

Before an episode, the runner initializes both adapters and calls
`negotiate_capabilities(benchmark, policy)`. The deterministic report checks the
contract version, named image and proprioceptive inputs, shapes, dtypes,
encodings, color spaces, units, action dimension and conventions, action
horizons, and dynamic-instruction support. Errors block execution; warnings
describe unavailable optional inputs. `report.require_compatible()` is the
standard fail-fast boundary.

The data flow is intentionally split:

```text
benchmark ── PolicyObservation ──> policy
benchmark ── SignalValue ────────> trace / metrics
policy ───── ActionPrediction ───> runner selects one chunk element
runner ───── BenchmarkActionRequest ──> benchmark
benchmark ── ExecutedAction ─────> trace
```

Signals marked `evaluation_only` or `privileged` must never be copied into a
policy observation. Requested and applied actions remain distinct so clipping,
safety changes, or benchmark-side modification is preserved as evidence.

Deterministic mocks live in `code/tests/helpers`. Reusable black-box assertions
live in `code/tests/contract/adapters`; concrete adapters should apply these
checks in addition to their own unit tests. The CPU-only integration rollout
demonstrates capability negotiation, single-action selection from chunks,
episode termination, and preservation of evaluation evidence.
