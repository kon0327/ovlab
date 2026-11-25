# ovlab-benchmarks

`ovlab-benchmarks` is the benchmark SDK. It defines stateless plugin control
hooks, portable task catalogs, capability checks and the stateful benchmark
adapter lifecycle. Concrete simulators, their configuration, task bindings,
renderers and dependency environments belong to independent projects. The
LIBERO implementation is [plugins/libero](../../../plugins/libero/README.md),
installed as `ovlab-libero` and imported as `ovlab_libero`.

## Dependency boundary

```text
benchmark project -> ovlab-benchmarks -> ovlab-core
policy project    -> ovlab-policy-sdk -> ovlab-core
dataset provider  -> ovlab-core.datasets
```

The SDK does not depend on `ovlab-benchctl`, a policy SDK, a concrete benchmark,
or model libraries. Orchestration selects providers and supplies resolved
contexts. Plugins do not import private CLI implementation modules.

`ovlab_benchmarks.plugin` contains the standard-library control API:
`BenchmarkPlugin` API `"1"`, `BenchmarkDeploymentSpec`,
`BenchmarkRuntimeMount` and `ConfigReference`. Importing this module does not
load NumPy, a simulator or a renderer. The package's public exports are lazy.
Typed catalog, task selection and runtime capability utilities live in
`ovlab_benchmarks.catalog`; adapters and canonical core DTOs can use NumPy.
Keep deployment-only hooks such as `resolve_deployment` on the lightweight side
of this boundary.

## Plugin control contract

A project declares a `kind: benchmark` provider in its `ovlab.extension/v1`
manifest. A benchmark configuration selects it explicitly, for example:

```yaml
kind: benchmark
type: libero
provider: plugin:benchmark:libero/libero
```

The `type` names the provider-owned configuration schema; the qualified
`provider` chooses its implementation. The common extension registry handles
source selection and bounded config references without importing that
implementation during declaration-only discovery.

| Hook | Responsibility |
| --- | --- |
| `validate_benchmark(document, path)` | Validate the complete benchmark document without native initialization. |
| `resolve_settings(document, context)` | Create plugin-owned settings from `BenchmarkResolutionContext`. |
| `identity(settings)` | Return a portable `BenchmarkIdentity`. |
| `describe_capabilities(settings)` | Declare observation, action and signal contracts without creating a simulator. |
| `task_catalog(settings)` / `select_tasks(settings, catalog)` | Describe authoritative tasks and the exact selected order. |
| `create_adapter(settings)` | Return a `BenchmarkAdapter` in `CREATED` state. |
| `resolve_deployment(...)` | Describe image, build role, environment, GPU and mounts using lightweight control imports. |

Optional hooks describe execution profiles, additional bounded config
references, asset requirements, execution metadata, methodological claims and
explicitly allowed optional runtime signals. Acquisition is a separate invoked
operation; control hooks never download assets, start native environments or
mutate the process environment.

The resolver builds a canonical `BenchmarkContext` from benchmark identity,
declared capabilities, selected task-set identity and named method claims.
Policy API `"2"` consumes this context rather than another project's YAML.
Standalone policy serving receives no benchmark context. Neither a policy
normalization key nor an arbitrary task-ID prefix establishes benchmark identity.

## Catalog and runtime checks

`TaskCatalog` holds portable `TaskDescriptor` values. It has no simulator
handles, machine-local paths or loaded initial-state arrays. Explicit task IDs
retain their requested order; suite selection retains catalog order. Catalog
identity and selected task identity are recorded independently of execution
locations.

The concrete plugin binds catalog entries to native tasks during initialization
and checks the runtime against the selected description. Capability validation
rejects changed peer contracts and undeclared signals. `CapabilityRefinement`
permits only the exact additional optional, non-policy signals it declares.
Static compatibility and required-metric checks run before model/native
initialization; the initialized adapters must still report their actual
capabilities.

## Adapter lifecycle and signals

Adapters subclass `BenchmarkAdapter` and implement its protected hooks. The
public methods enforce `created → ready → episode_active → ready → closed`,
stable identifiers and one selected action per step. Terminal/truncated steps
return to `ready`; `close()` is idempotent. Each step records the action actually
handed to the benchmark alongside the next observation.

Policy-visible observations and evaluation signals use separate channels.
Privileged state, success, reward and other evaluation-only data belong in
`SignalRegistry`/`SignalValue`, never in `PolicyObservation`. The runner
negotiates both adapters and the metric requirements before reset.

See the [extension architecture and qualification gates](../../../docs/EXTENSION_ARCHITECTURE.md)
for selected locks, frozen workers and the distinction between declaration,
SDK, native runtime and public-pipeline validation. Metric/report module
registration is reserved by the common catalog; their runtime modularization
is subsequent work.
