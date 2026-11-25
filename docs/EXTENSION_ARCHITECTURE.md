# Extension architecture implementation and qualification

This document records the approved staged migration from policy-only project
registration to a common plugin/module registry and a standalone LIBERO project.
It is a development plan and evidence log; qualification applies to the exact
gates and runs recorded below. The starting revision is `6196ecb` on
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

## Implemented boundaries

The registration format is `ovlab.extensions/v1` with a `projects` mapping.
Each binding selects a local directory or a full immutable Git commit and an
optional manifest path. New projects default to `extension.json`; the three
existing model projects explicitly retain the filename `plugin.json` with the
new `ovlab.extension/v1` contents. There is one manifest per project, not a pair
of competing declarations.

Manifest category belongs to the project; provider kind belongs to each entry
under `providers`. A plugin project can contain several provider kinds. Common
metadata supports policy, benchmark, service, dataset and training providers;
module metadata reserves metric/report kinds without supplying a runtime SDK.
A module containing only configuration and no providers is a supported
registration/discovery fixture, not a qualified metric/report implementation.
Manifest schema, provider API and Python package versions are independent.

Qualified provider bindings include category, kind, project and local provider
ID, for example `plugin:benchmark:libero/libero`. Component `type` remains the
selected provider's configuration type. Manifest source/build names are local
to their project; common catalog identities use `project/local`, including
`project/self`. Physical staging is
`external/extensions/<project>/<source>`, independently of existing user
checkouts under `external/`. Build selectors use `project/build-role`; a local
role alias is accepted only when its owner is unique. The temporary
`extension_compat.py` facade preserves caller shapes, not a second registration
authority or a separate source/build namespace.

`ExtensionCatalog` owns declarative discovery, acquisition primitives, bounded
configuration roots and selected snapshots. It does not import provider code
for listing or selection and does not fetch an unavailable unrelated project.
Acquisition is explicit; the pinned-project cache checks origin, revision and
cleanliness. Selected locks (`ovlab.extension-lock/v1`) describe portable
declarations and source identities, while locations describe materialized
host/worker paths separately. A frozen worker resolves from its selected lock
and bundled locations and fails if an input is absent; it must not recover from
host checkouts, external caches, installed project locators or the network.

The SDK split keeps `ovlab_benchmarks.plugin` standard-library-only for control
contracts and deployment descriptions. `ovlab_benchmarks.catalog` owns typed
task/catalog/capability utilities and can use the canonical NumPy DTOs. Policy
API `"2"` receives `BenchmarkContext`: identity, capabilities, selected task-set
identity and explicit method claims. It never parses the selected benchmark's
YAML. Standalone serving has no benchmark context and must not infer one from
a model normalization key. `BenchmarkResolutionContext` carries already
resolved orchestration inputs into the benchmark's own settings hook.

`plugins/libero` owns the concrete benchmark distribution (`ovlab_libero`),
portable task catalog and native bindings, action/observation/signal mapping,
execution profiles, renderer, runtime configuration, dependency locks and image
recipes. Its benchmark provider declares the benchmark image/GPU/mount needs;
the selected model independently declares its policy image/resources. The
orchestrator composes and verifies that pair. Its RLDS provider implements
`ovlab_core.datasets.DatasetProvider`; source-specific acquisition remains in
the project, while publication, verified inventories and lifecycle remain
generic. SDK/core code has no dependency on `ovlab-benchctl`.

Metrics, report generation and export continue through the existing canonical
pipeline after the model/benchmark boundary. Registration support for future
modules does not change those runtime owners during this migration.

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
- Stages 1–3 are committed as `25c008d`, `e75e659` and `d3e27c4` respectively;
  their recorded evidence is listed below. Those gates do not establish a
  qualified API v2 production deployment or LIBERO image.
- Stage 4 is complete: API v2 model/benchmark composition, public CPU and native
  LIBERO workflows, metrics/report/export and read-only integrity checks passed.
- Stage 5 is complete: installed pinned-Git regression, API compatibility cleanup,
  repeated public CPU/native GPU smoke and scoped cleanup. External publication
  and metric/report runtime modules remain separate future work.

## Implementation evidence

Phase 1 adds the shared catalog and migrates the in-tree policy manifests without changing their runtime SDK. The broad CLI/config/deployment regression run passed 617 tests; a provider-aware adjustment to the artifact classification test was verified by the subsequent 19-test review run. A separate 64-test run passed registration/Git cache/origin, configuration and canonical CPU run/report/metric checks. Benchmark SDK files are prepared separately and are not part of this phase.

Phase 2 freezes the selected neutral lock beside its portable runtime locations, prunes unrelated policy declarations, validates cached origins and closes project/control-source host fallback. Targeted build/config regressions: 125 passed. Second review, relocation and closed-module-bundle checks: 24 passed. Large runtime data remains external; only selected control code/configuration is bundled.

Phase 3 adds BenchmarkPlugin API1, immutable task/catalog/peer contracts, policy API2 alongside unchanged API1, a lower-level dataset provider contract, and pure connection preflight. The SDK toy executes two CPU tasks into canonical artifacts, metrics and report without qualifying the public CLI/Compose route. SDK plus runner review: 39 tests passed. No concrete simulator/runtime imports are introduced in SDK contracts.

Phase 4 integration also removes the remaining model implementation from the
RPC SDK: vanilla service identity lives in the OpenVLA package. RPC v2 carries
named typed observations and arbitrary ActionSpec dimensions/conventions. Model
packages own action-interface defaults; benchmark execution profiles are
selected explicitly and frozen with local extensions. Reports receive explicit
benchmark/suite/renderer identity and recorded action semantics. Historical
artifact fallback is read-only. The generic metrics/report implementation still
belongs to the subsequent module phases.


### Stage 4 public qualification (2026-11-25)

The full unit/contract/selected-integration gate passed 1,146 tests before final
review. Review fixed adapter abort delegation and runtime task-limit/index drift,
optional runtime metadata, missing frozen example/experiment fallback and
unselected-invalid-project isolation. The subsequent broad gate passed 1,155
tests and exposed one old inventory assertion; explicit selected-corrupt-project
rejection and the updated inventory behavior then passed the 27-test registry
review. Deployment review passed 38 tests; report/export regression passed 30.
The remaining report/browser tests are repeated at the final stage.

Independent CPU invocation (generated private example registry/profile):

```bash
OVLAB_EXTENSION_REGISTRY="$PWD/configs/extensions/cpu-example.json" \
OVLAB_LOCAL_PROFILE="$PWD/configs/local/cpu-plugin-example.yaml" \
OVLAB_COMPOSE_ENV_FILE="$PWD/configs/local/cpu-plugin-example.env" \
ovlab eval run configs/evaluation/cpu-plugin-example.yaml --offline \
  --project-name ovlab-phase4-cpu-final --json
```

Run: `cpu-plugin-example_2026-11-25_12-16-09_acecdf64`, under
`/tmp/ovlab-phase4-cpu-data/runs/evaluate/`. It completed the three opaque tasks
with a two-dimensional joint action contract. Policy and benchmark services
exited 0; Compose cleanup completed. Frozen bundle identity:
`e47946e5e57067bddee910097607f2f5f6bdab259beafb3880e8f11a2aef1388`.
Canonical verify, metric recompute (all results agree), report verify and isolated
export verify succeeded. All 58 canonical files retained identical SHA-256.

Native GPU invocation:

```bash
OVLAB_LOCAL_PROFILE="$PWD/configs/local/gate-b-showrack.yaml" \
ovlab run examples/experiments/plugin-pipeline-smoke.yaml --json
```

Workflow: `plugin-pipeline-smoke_2026-11-25_12-14-14_2370e2a427`.
Training: `plugin-pipeline-smoke-train_2026-11-25_13-15-25_4b67e506` (three
optimizer steps), finalized adapter `checkpoint-f7270eb3f519920f36f657aed037ff08`.
Evaluation: `plugin-pipeline-smoke-eval_2026-11-25_12-18-00_f5444601`, one real
LIBERO task with a two-step limit, under that workflow's
`evaluate/evaluate-checkpoint/` directory in `ovlab-data/runs`.
Both eval services exited 0 and cleanup completed. Training and evaluation
reports and isolated exports succeeded. Canonical verify, metric recompute,
report verify and export verify passed; all 43 evaluation canonical files kept
identical SHA-256. Frozen eval bundle:
`31408d7c39c44ae19ac5f2ee70c23cbc8c3875c8caf7f6e0f87be112b33f287d`.
The initial workflow attempt caught missing adapter mounts in static preflight;
mounts now share the same read-only container locations across preflight and
both workers, covered by the deployment regression.

Selected immutable image IDs for this qualification:

| Role | Image ID (sha256) |
| --- | --- |
| LIBERO benchmark | `29613151ccc5cbbd5fdc8af56204a797a74f839ff28a17a11f9d8938b6d9151c` |
| LIBERO dataset | `cf6b5f4f68a53033b0967c041b694888e0936d363512ddf38a595d29785852f6` |
| QuIC policy | `409974411fc76a3af3b1127519a3c2e7ee2cd2bc15d39853b64aea8d5e217c71` |
| QuIC training | `d11ec14cfdc75cb3a076b69909d30d98afa0929099cb688357b2dc147fe313bf` |
| QuIC finalizer | `453ff2d72593e99802e6992b9acbb545c812004d2f0a238a366e88a7d8690b64` |
| Reporting | `4fff3381dcbd71499214f682daaf908d44e6e6e8a25f2d37a70c8e53a9fcaf33` |

This is pipeline/lifecycle qualification, not a model-quality experiment. Dataset,
base checkpoint and native LIBERO assets were reused read-only; no user source
checkout was modified. OpenVLA/OFT images were not used by this native gate.


### Stage 5 final review

Policy API v1 and first-match `accepts_settings`/`accepts_capabilities` discovery
are removed. `PolicyPlugin` is the canonical API `"2"`; standalone context is
`None`, and model hooks receive canonical peer contracts. Internal legacy-shaped
worker data remains a projection of the single neutral registration authority.
Historical artifact readers remain read-only.

The pinned-Git installed-worker regression builds three control SDK/application
wheels offline, installs them into an isolated target and registers two independent
local Git remotes at exact commits. Branch heads then contain unusable newer
code. After freezing, it removes host projects, remotes and caches, and runs
Python with `-I -S`, an empty home and no Git on PATH. Network/subprocess
tripwires remain active. Both providers and their bundled task catalog load
from the selected frozen closure; NumPy and native/model runtimes are absent.

Final regression: **1,168 passed** (unit, contract, selected isolation/CPU/Git
integration and model packaging). Report/export plus editor/resolver parity:
**48 passed, 2 skipped**; both skips are optional Playwright browser modules.
The offline editor's direct Node suite passed **21 tests**, and JavaScript syntax
checks passed. Browser UI interaction tests remain unexecuted because the test
environment does not supply Playwright/Chromium. No browser dependency was
installed into the user's environment. The temporary checksum-verified Node
runtime was isolated under `/tmp`.

Review removed fixed renderer choices from the editor: execution aliases and
defaults come from the selected benchmark descriptor. It also validates policy
provider/config-type agreement. LIBERO owns all benchmark/dataset lock inputs,
including `.in` files. Empty old benchmark/config directories and generated
Python caches were removed.

Historical compatibility was checked against
`plugin-pipeline-smoke-eval_2026-11-20_19-41-46_a5acc158`: canonical verify and
metric recomputation passed, and all 43 canonical file hashes stayed unchanged.
The original OpenVLA, OFT and QuIC working trees remained clean.

The final API2 CPU run is
`cpu-plugin-example_2026-11-25_12-27-18_92e96a4d`, with frozen bundle
`b95c59c4a4df91d9ce4f62cb232fdad50a2233856e263e153c5c05e8f3998799`.
Canonical verify, metric recompute, report verify and isolated export verify all
passed; all 58 canonical files kept identical hashes. Final production-image
and repeated GPU qualification are recorded below.


### Repeated native GPU and final image gate

All ten registered production roles built successfully through the isolated
source path. The three model projects and LIBERO also passed `plugin check`
inside their clean runtime images with `--network none --read-only`, without
host source or configuration mounts. This qualifies installed control imports
for all four projects; numerical native execution is qualified by the QuIC
workflow, not by the other model image builds.

Workflow: `plugin-pipeline-smoke_2026-11-25_12-38-28_2370e27c09`. Training: `plugin-pipeline-smoke-train_2026-11-25_13-39-01_f6e0a2cc`; optimizer steps: **3**; finalized adapter: `checkpoint-8c942c92de1630d747148e02b49c2ca0`.
Evaluation: `plugin-pipeline-smoke-eval_2026-11-25_12-41-57_9978f0c2` (one two-step native LIBERO rollout).
Frozen bundle: `dad1be5a61b7038047badf894ec425b90762139e79e002fb5e7d41a03e4bc108`.
Both services exited 0, Compose cleanup completed, and training/evaluation
reports and exports completed. Canonical verify, metric recomputation (all
results agree), report verify and export verify succeeded. All 43 evaluation
canonical file hashes remained unchanged.

| Image | Qualified image ID |
| --- | --- |
| `ovlab-reporting:local` | `sha256:5c527ac4c1c56e7085aa9e98b1deeeb23977c1815e1fe6598dcdc6222b07b7e2` |
| `ovlab-policy-openvla:local` | `sha256:7b0fcf9fd691db4b06518178cfc3d2309c7e8052fe97fa844eecb06684e5f1c8` |
| `ovlab-training-openvla:local` | `sha256:22bb83b7c53c3b9564cc421c2bed4634f5db7b843232ead4afa0ff8f3112227c` |
| `ovlab-finalizer-openvla:local` | `sha256:16c9c2904395e016177ea9f6093fed7e13a4bd7c61a65e3d594d825651f7858a` |
| `ovlab-policy-openvla-oft:local` | `sha256:60545f6e9acdcd44e8ff91020bb39045d5d83ff1b22c627e1bb049f9a05fc612` |
| `ovlab-policy-openvla-quic:local` | `sha256:d65a71df30e5c42a46a09a8781ce398712e33da38140e8a42071fd2911a5defa` |
| `ovlab-training-openvla-quic:local` | `sha256:d49b1c92d984c587731a8ea7e8aceb06fc17590bfcad5c1377a1c167fdfc63b9` |
| `ovlab-finalizer-openvla-quic:local` | `sha256:6b7d7b94c0e8e18e91746a5757d999b37e458342313cf3c47d859756ed6a714b` |
| `ovlab-benchmark-libero:local` | `sha256:cc13d2427dc0cc0b510754e2e8d6eefce42909cb13f76c92b9c13625456b8da2` |
| `ovlab-dataset:local` | `sha256:aa3f93f69b0f5f5cab929a80e2b755b7bc088886a3dad9dfcd19b2429e2bca4f` |

The third review also rejected unknown/mismatched benchmark providers in the
editor when no renderer override is present, and reserved the neutral registry,
root and experiment/config paths against plugin mount environment overrides.
The direct Node suite passed 21 tests; the final host deployment/model/plugin
and resolver-parity gate passed **98 tests**. The latter guard is host-side
validation and does not change the selected workers' RPC or numerical contract.

Successful native workflow artifacts remain under their named `ovlab-data`
run/export roots. Final CPU evidence is archived under
`/tmp/ovlab-extension-qualification/cpu/`; JSON verification evidence and final
smoke logs are beside it. Generated private CPU registry/profile files,
intermediate failed smoke data, test-only images, temporary test directories,
Node runtime and Python caches are removed. Existing user containers, source
checkouts, datasets, base weights, local profile and Compose environment remain
untouched.
