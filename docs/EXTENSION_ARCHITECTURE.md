# Extension architecture implementation and qualification

This document records the approved staged migration from policy-only project
registration to a common plugin/module registry and a standalone LIBERO project.
It is a development plan and evidence log, not a claim that the new runtime has
already passed qualification. The starting revision is `6196ecb` on
`feat/benchmark-plugin-architecture`.

## Scope and ownership

A project has a category (`plugin` or `module`). Its providers have explicit
kinds, including policy, benchmark and dataset provider. A provider binding names
the category, kind, project and provider; selection must not depend on whichever
matching implementation happened to be imported first. Configuration references
retain their owner: `ovlab://`, `plugin://` or `module://`.

The common registry owns declaration validation, explicit acquisition, source
selection, portable identities and selected locks. SDKs own typed contracts.
Projects own their configuration, control code, runtime implementations,
upstream pins, assets, recipes and dependency locks. Orchestration owns lifecycle,
transport, resource validation and result persistence. Model projects never own
the selected benchmark service or its image, renderer, mounts or GPU needs.

The LIBERO integration project revision and the upstream LIBERO revision are
independent identities. Evaluation assets/BDDL/initial states belong to the
upstream benchmark source; RLDS training data have a separate dataset revision
and immutable manifest. Acquisition is explicit, while runtime consumes verified
read-only resources.

Metric and report runtime extension packages, a new metric SDK and external
publication are subsequent work. This migration uses the existing metric,
signal, canonical result, report and export contracts. `reporting.enabled=false`
continues to suppress derived HTML while preserving the isolated export. It is
not a request to omit the reporting/export execution role.

## Stages and commit boundaries

| Stage | Complete change | Required qualification before commit |
| --- | --- | --- |
| 0 | Read-only plan and independent audit. | Review architecture, acceptance criteria, ownership and available runtime. |
| 1 | Common registry and manifests, typed IDs, explicit bindings and migration of model manifests. Existing policy SDK behavior remains active. | Declarative registry/source isolation; existing model control/config regressions; no unused project acquisition or runtime imports. |
| 2 | Selected lock, transitive config/control/source closure, portable identities and strict frozen resolution. The existing runtime remains usable. | Relocated local sources retain identity; selected content changes invalidate identity; frozen workers resolve without host files, Git or external cache; missing frozen inputs fail closed. |
| 3 | Add neutral `BenchmarkPlugin`, `DatasetProvider`, `BenchmarkContext` and policy API v2 contracts without activating the production switch. Add a CPU toy provider. | SDK-level toy evaluation through current metrics and report/export; static task/action/observation/signal checks before adapter initialization and runtime recheck. This is not CLI/Compose qualification. |
| 4 | Atomically activate API v2 in all three model projects and standalone callers; migrate complete LIBERO distribution/config/renderer/recipe/locks/assets/dataset bridge; activate generic resolver/native factory, paired Compose, both build roles and both image identities; migrate active examples and editor/catalog. | Public CPU toy policy + benchmark pipeline through canonical metrics/report/export; basic clean installed/frozen execution without all-plugin `PYTHONPATH`; checkpoint handoff regression; actual LIBERO evaluation and report smoke. |
| 5 | Independent clean installation/Git fixture regression, expanded checkpoint-from-training pipeline, documentation and removal of agreed API v1 compatibility shims. | Repeat complete isolated CPU and real LIBERO gates, review compatibility caller inventory and preserve historical artifact verification. |

Each stage follows: implementation and focused tests; relevant regression suite;
independent read-only review; fixes and affected checks repeated; a real smoke
appropriate to that stage; scoped cleanup; review of the final diff; commit only
the completed stage. A skipped runtime test is an open gate. Preparing stage 4
in smaller edits must not produce a supposedly complete commit with a broken
production path.

Static compatibility checks are introduced with the contracts in stage 3 and
activated in stage 4. They must reject incompatible action/observation contracts,
unknown tasks and statically missing required metrics before model/native
initialization. Runtime validates actual native capabilities against the declared
contract. Optional signal refinement is allowed only through an explicit rule.

Numerical inference methodology, action transforms, preprocessing, normalization,
task ordering, seeds and initial-state selection remain unchanged. Historical
canonical artifacts are never rewritten. New runs carry explicit benchmark,
suite and renderer metadata; task ID prefixes are not an identity source.

## Check levels and their limits

| Level | What it proves | What it does not prove |
| --- | --- | --- |
| Declaration | Registry/manifest structure, references, templates and available source metadata. No provider import, fetch, model loading or GPU. | SDK behavior, runtime dependencies, resource contents or execution. |
| Control/SDK | Explicit lightweight control API validation and static capabilities/task/metric negotiation. No native/model initialization or network acquisition. | Actual native capabilities, model inference or renderer readiness. |
| Runtime | Actual provider initialization, capability recheck, lifecycle and resource use. | End-to-end publication unless canonical/report/export checks also pass. |
| Public pipeline | User-facing CLI/Compose path, canonical run, metric recomputation and report/export integrity. | GPU qualification when the benchmark is a CPU toy. |

`plugin check` currently includes SDK control-entrypoint loading. During migration
the declaration-only check and explicit control check must remain distinguishable;
the documentation must not promise an import-free operation while executing
Python hooks. Lazy discovery means an unselected unavailable remote project does
not prevent selection of a different local project.

## Isolation regression preparation

`code/tests/integration/test_extension_isolation.py` targets the common
`ExtensionCatalog` API. Its fixtures are independent directories under pytest's
temporary root. They contain a module with configuration and no providers, a
policy provider whose entrypoint raises if imported, and an uncached remote
registration that must remain unused. Tests cover:

- Listing unavailable remote metadata without fetch, provider import or writes.
- Explicit provider selection and selected snapshots excluding unrelated remotes.
- Typed provider identity and wrong-kind rejection.
- `module://` configuration without a provider/runtime implementation.
- Local source identity independent of location but sensitive to content.
- Rejection of ambiguous/traversing module configuration references.

Run the integration tests after the common catalog implementation is available:

```bash
conda run -n ovlab-tester deploy/scripts/test.sh \
  code/tests/integration/test_extension_isolation.py -p no:cacheprovider
```

Strict frozen resolution is a stage 2 gate, not a skipped placeholder test. Once
freeze is implemented, extend this integration file with a fresh subprocess
whose only project inputs are the selected lock and materialized closure:

1. Freeze selected policy/module configuration and transitive control inputs.
2. Copy the closure to a different worker root and make the original fixture
   sources unavailable. Supply no host plugin source directories on `PYTHONPATH`.
3. Set an empty test-owned home/cache and explicit worker registry; forbid Git,
   network acquisition and fallback registry selection.
4. Resolve the frozen references, run the selected lightweight control check and
   verify all project-loaded modules originate inside the worker closure.
5. Remove or tamper with one frozen input while its original host source still
   exists. Resolution must fail; it must never repair itself from that source.
6. Repeat with an unused unavailable registry project. It must not become a
   selected dependency.

This subprocess test complements a clean installed/image gate. The general test
launcher intentionally includes local source roots for development, so it cannot
alone establish package completeness.

`ovlab-tester` supplies the lightweight pytest/NumPy environment; it does not
include the Jinja2/matplotlib dependencies needed by the report/export smoke.
Use an existing reporting-capable environment for that gate (the local
`/home/kony/miniconda3/envs/openvla-quic/bin/python` is available during this
migration), or the isolated reporting image. Do not silently install dependencies
into a user's environment. Test invocations set `PYTHONDONTWRITEBYTECODE=1`, disable
the pytest cache provider and use explicitly owned temporary paths under `/tmp`.

## Smoke verification checklist

The following existing tests provide the migration baseline. Their current mock
or fake adapters do not qualify the new public benchmark runtime or GPU renderer.

| Area | Existing check | Migration acceptance |
| --- | --- | --- |
| All model config/plans | `code/tests/unit/config/test_configuration_origins.py`, `code/tests/unit/config/test_plugin_configs.py`, `code/tests/unit/cli/test_deployment.py`, `code/tests/unit/cli/test_model_deployment.py` | Validate complete, compatible OpenVLA, OFT and QuIC compositions, deployment plans and environment-selected registry. Partial project defaults remain explicitly partial; OFT needs its supported observation profile and QuIC needs its adapter binding. |
| Independent CPU policy | `code/tests/unit/cli/test_plugins.py::test_external_plugin_resolves_predicts_and_plans_deployment_and_evaluation`, `code/tests/unit/cli/test_plugin_projects.py` | Preserve the standalone example policy's config, adapter inference, registration and isolated build planning. Fake Docker runners test planning and cleanup only. |
| Canonical metrics | `code/tests/integration/runner/test_mock_run.py` | Trace finalization precedes metrics; episode/task results recompute from immutable traces; partial/error outcomes remain distinct. |
| Report/export | `code/tests/reporting/test_reports_exports.py` | Verify HTML/export inputs and file integrity, unavailable metric reasons and the HTML-disabled/always-export behavior. |
| Editor/catalog | `code/tests/workflow_builder/test_builder.py` | Exported toy and LIBERO workflow configuration agrees with the real resolver; declaration discovery never loads native runtimes or fetches upstream sources. |
| Native LIBERO | `code/tests/integration/benchmarks/libero/test_real_libero.py`, `code/tests/integration/runner/test_libero_runner_manual.py` | Adapt to project imports after migration; explicitly run reset/step and actual evaluation, then canonical integrity, metric recompute and existing report/export checks. |

The new public CPU smoke must use an independently registered toy benchmark with
non-LIBERO suite/task IDs, camera name and action contract. It must prove a
canonical run, available metrics, optional unavailable metrics with reasons,
required-metric rejection, report generation and isolated export verification.
The old transport smoke ends at `deploy connect` and cannot substitute for this.

For both the CPU and LIBERO public smoke, record the exact invocation, selected
project/upstream identities, frozen config identity, immutable image IDs, run
path, canonical verify result, recomputation result and report/export verify
result. A real GPU test must initialize the real renderer/native benchmark;
mocks, metadata, dry runs and skipped tests are not GPU evidence.

## Cleanup and runtime availability

Smoke fixtures use distinct temporary roots and unique Compose project names.
Cleanup runs after success, failure and ordinary interruption. Remove only paths,
containers, volumes and temporary image references created by that smoke. Never
prune user caches, builds, source checkouts, existing images or datasets. Keep
qualification evidence identified by its run/manifest path. Hard process kills
cannot guarantee cleanup; retain enough scoped identity for a later recovery.

Read-only runtime inspection on 2026-11-24 established Docker client/server
29.6.1 and an NVIDIA GeForce RTX 4090 with 24,564 MiB VRAM (driver 576.52).
No active containers or compute GPU processes were reported during that check.
Existing local benchmark, reporting, dataset and all three model images were
listed. This establishes availability only: the images have not been qualified
against the new contracts. No image build or container start/stop was performed
for this preparation. Recheck current workloads before future runtime smoke.

## Progress and evidence

- Stage 0: plan and independent boundary audit completed; runtime availability
  inspected read-only.
- Stage 1: common catalog implementation and independent isolation tests are in
  progress. API-level test execution and second review are pending.
- Initial focused baseline attempt in `ovlab-tester`: report/export checks could
  not run because Jinja2/matplotlib are absent; the configured-run check reached
  the in-progress registry integration before `extensions.py` was available.
  These are not passing smoke evidence. Repeat after the catalog is complete
  using the reporting-capable environment for report/export checks.
- Stages 2–5: not yet qualified. The frozen subprocess and complete runtime
  acceptance gates above remain open.

## Implementation evidence

Phase 1 adds the shared catalog and migrates the in-tree policy manifests without changing their runtime SDK. The broad CLI/config/deployment regression run passed 617 tests; a provider-aware adjustment to the artifact classification test was verified by the subsequent 19-test review run. A separate 64-test run passed registration/Git cache/origin, configuration and canonical CPU run/report/metric checks. Benchmark SDK files are prepared separately and are not part of this phase.
