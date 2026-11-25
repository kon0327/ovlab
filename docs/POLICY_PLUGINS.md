# Model plugins in OVLAB

A policy plugin owns the model integration: configuration validation, settings,
observation compatibility, adapter construction, RPC identity and model-specific
runtime mounts. OVLAB owns configuration composition, checkpoint verification,
containers, RPC transport, evaluation lifecycle and artifacts. Adding a model requires a plugin and a registry entry; it requires no new branch in the
resolver, CLI or deployment code.

The current model contract is `ovlab_policy_sdk.plugin.PolicyPluginV2` (API
`"2"`). OpenVLA, merged LoRA, OFT and QuIC use canonical `BenchmarkContext`
rather than reading another project's benchmark YAML. The API v1 base remains
only for migration compatibility. The standalone CPU example also uses API v2.
Their settings and artifact semantics remain in the integration packages. Model
libraries and weights are imported only during adapter initialization. Registry
listing and deployment planning work with the standard library alone when the
plugin's control hooks also avoid optional runtime imports.

For a complete first integration, follow the [runnable developer
README](../examples/policy-plugin/README.md). It includes all configuration
files and a CPU service using a tiny bundled artifact, so you can validate the
connection without downloading a model. Return here for the full contract.

## Project registration and source selection

A plugin is an independent project containing a manifest, a Python
package, configurations and build recipes. The user registry only binds that
project to a local directory or an immutable Git revision. Policy variants,
upstream model sources, capability declarations, defaults, checkpoint catalogs,
training hooks and dependency locks belong to the project.

The common author-facing registry uses `ovlab.extensions/v1`:

```json
{
  "schema_version": "ovlab.extensions/v1",
  "projects": {
    "my-model": {
      "source": {
        "kind": "git",
        "url": "https://github.com/example/my-model-plugin.git",
        "commit": "0123456789abcdef0123456789abcdef01234567"
      }
    }
  }
}
```

The commit illustrates the required full lowercase 40-character format; replace
it with an actual commit. Local registrations use
`{"kind": "local", "path": "/home/developer/my-model-plugin"}` instead. The manifest's
`id` must match the registry key. To register a complete project, use:

```bash
./ovlab plugin register /home/developer/my-model-plugin --dry-run
./ovlab plugin register /home/developer/my-model-plugin
./ovlab plugin register https://github.com/example/my-model-plugin.git --commit FULL_COMMIT
```

The command reads the project manifest to obtain its ID and validate the complete
registration. It writes `configs/extensions/registry.json` by default; `--registry`
selects another destination. A conflicting existing registration is preserved.
Remote registration acquires the pinned manifest into an ignored project cache.
Offline operations and build dry runs require that cached manifest; they never
fetch an uncached project. `plugin register --dry-run` previews a remote acquisition
without creating a cache or writing the registry.
The checked-in registry registers independent `openvla`, `openvla-oft` and
`openvla-quic` projects. OpenVLA owns Vanilla and LoRA adapter support; OFT and
QuIC own their distinct runtimes. Shared implementation code lives in the
reusable `plugins/_shared/openvla-support` distribution, explicitly declared as
a control/build source by each consuming project. Core packages do not depend
on that distribution. The independent `libero` project owns the benchmark and
RLDS dataset provider. Transitional generic recipes remain under the registry's
`infrastructure: {sources, builds}` table; they are not policy declarations.

Selection starts with an explicit registry argument, then
`OVLAB_EXTENSION_REGISTRY` (the temporary `OVLAB_PLUGIN_REGISTRY` alias is still
accepted), `configs/extensions/registry.json`, the legacy user location
`configs/plugins/registry.json`, the repository's `plugins/registry.json`, an
installed registry and the packaged default. Relative registry and local source paths are
relative to the OVLAB repository. `plugin` commands also accept `--registry` for
that one invocation. Python orchestrators honor their explicit environment
mappings without modifying `os.environ`. A selected registry is a complete
selection; registering into a new file starts from the selected default to retain
existing registrations.

New projects default to `extension.json`. Existing model projects retain the
filename `plugin.json` through the registry's explicit `manifest` field, with
the same new schema. Do not publish two competing manifests. A project manifest
with a policy provider looks like this:

```json
{
  "schema_version": "ovlab.extension/v1",
  "category": "plugin",
  "id": "my-model",
  "version": "1.0.0",
  "sources": {},
  "providers": {
    "default": {
      "kind": "policy",
      "api_version": "2",
      "config_type": "my_policy",
      "id": "my-model-policy",
      "family": "my_family",
      "runtime_status": "implemented",
      "entrypoint": "my_model.ovlab_plugin:plugin",
      "sources": [],
      "capabilities": {
        "observation": {"images": {"roles": ["primary"]}},
        "action": {"dimension": 7},
        "training": {"supported": false}
      },
      "deployment": {
        "profile": "my-model",
        "role": "policy-my-model",
        "image": "ovlab-policy-my-model:local",
        "image_env": "OVLAB_MY_MODEL_IMAGE",
        "policy_service": "policy-my-model",
        "quantization": ["none"],
        "observation_profile": "primary_rgb"
      }
    }
  },
  "builds": {
    "policy-my-model": {
      "dockerfile": "docker/Dockerfile",
      "image": "ovlab-policy-my-model:local",
      "sources": [],
      "locks": ["locks/runtime.requirements.txt"]
    }
  },
  "configs": {"root": "configs", "checkpoints": ["registry/default.yaml"]}
}
```

Dockerfile and lock paths are project-relative and bounded to that project.
Source and build names are local to the manifest; catalog identities are
`<project>/<local-id>`, with `<project>/self` implicitly added to providers and
builds. Source staging uses `external/extensions/<project>/<source>`; the
project itself is `external/extensions/my-model/self`. Docker `COPY` paths use
that materialization, while provider identity remains independent of location.
This namespace preserves existing development checkouts such as
`external/openvla`. Additional
model implementations can remain in separate upstream repositories: declare
these under the manifest's `sources` as pinned Git or local sources, then list
their IDs in the variant and recipe. OVLAB acquires each source independently.
Neither the upstream model nor the core needs an extra dispatch branch.

A model project can expose supported adapter variants and images. Independent
model implementations should be independent projects; LoRA remains an adaptation
choice within OpenVLA. Several variants may share a build role. The deployment image must
match the role's recipe, and the recipe must include all variant sources. Shared
profile definitions must be identical. Set `deployment.gpu: false` for CPU-only
policies. Persistent deployment also accepts `runtime.gpu: null`; benchmark GPU
requirements remain independent. Descriptive `capabilities` mappings advertise
interfaces, artifact loaders, methods, quantization backends and limitations;
plugin validation remains authoritative for each actual combination.

Qualified provider IDs distinguish category, kind, project and local provider,
for example `plugin:policy:my-model/default`. Manifest schema version, provider
API version and package `version` have independent meanings. Selected frozen
state uses `ovlab.extension-lock/v1`, separate portable source identities from
execution locations, and includes only the selected closure. A temporary
`extension_compat.py` projection supplies legacy policy callers; it is not a
second registration authority or the author-facing format.

## Preparing and building sources

Git branches and tags are rejected as source revisions. Managed upstream sources
are cloned into ignored `external/extensions/<project>/<source>` for development sync. Remote
project manifests have a separate ignored project cache below `plugins/.cache/`.
Local projects are used directly for lightweight host hooks and copied into
content-identified snapshots for builds. Source identity includes executable
modes; source symlinks cannot escape their project. Git metadata is excluded
from image contexts.

```bash
./ovlab plugin list --json
./ovlab plugin inspect my-model
./ovlab plugin check my-model
./ovlab plugin sync my-model --dry-run
./ovlab plugin sync my-model
./ovlab plugin sync my-model --offline
./ovlab plugin build my-model --dry-run
./ovlab plugin build my-model
```

These selectors are project IDs from the registry, not variant IDs or policy
types. `list` shows projects; `inspect` includes all their policy variants,
capabilities, configuration templates and source identities. `check` loads only
SDK control entrypoints and checks declared template files without constructing
an adapter or loading weights. It does not prove that runtime dependencies or
checkpoint contents are ready; validate a completed experiment and use
`deploy verify`/`deploy connect` for a running service.

`sync` verifies origin, revision and cleanliness and refuses to replace dirty or
unmanaged destinations. Edited local snapshots are preserved. A changed Git pin
can be checked out only in a clean managed checkout. Ignored files that would be
copied into an image also make it dirty; bytecode, generated package metadata
and MP4 recordings under `rollouts/` are excluded from source snapshots. Tracked
edits remain protected. Offline acquisition uses cached Git objects or local
projects. It does not promise offline package installation inside Docker.

Builds prepare a temporary context from registered sources by default; they do
not need a previous sync or use a mutable development checkout. Each selected
Git source is fetched at its registered commit, and local projects are snapshotted.
The temporary context contains only selected sources at their managed `external/`
paths and is removed after success or failure. `--offline` can reuse synced Git
objects without copying dirty working files. `--workspace-sources` explicitly
selects verified persistent development sources. Source preparation uses locks
and atomic publication and retains rollback data while replacing local snapshots.

`bash deploy/scripts/build-images.sh PROJECT/ROLE ...` builds qualified registered
roles, including policy, benchmark, training and finalizer roles; an unqualified
role is accepted only when it has a unique owner. With no roles it builds all.
The `plugin build PROJECT` command selects all image recipes owned by that project, including policy, training and finalizer images. Add `--config TARGET` when the
image should include an explicit experiment/deployment configuration closure:

```bash
./ovlab plugin build my-model --config configs/deploy/my-service.yaml
```

Plugin defaults and user overrides resolve through the same bounded inheritance
mechanism before the configuration is frozen into the image. Runtime deployment
can supply its own separately identified bundle. Changing weights or inference
settings is independent from changing implementation dependencies.

A recipe used with `--config` must mount the optional `ovlab_config_bundle`
BuildKit secret and extract it through the generic OVLAB installer (as the example
does), or implement the same bounded configuration-copy contract itself. Recipes
without this declared input are rejected for configured builds.

Installed Python distributions can include their complete project assets and
expose an `ovlab.plugin_projects` entrypoint returning their manifest directory.
The OpenVLA distribution installs its manifest, configs, recipes, locks and source
project under `<prefix>/share/ovlab/plugins/openvla`. This locator imports no model
runtime. A project registered from Git or a local source remains the normal
source for image construction.

## Developing the plugin

The entrypoint is `module:object`. Export either a `PolicyPluginV2` instance or a
zero-argument class. The implementation must advertise `api_version = "2"`;
unsupported versions fail before model initialization. Python packages can live
at the source root or under `src/`. The exported object must implement the selected SDK contract. Instances are cached on their
loaded handle. Control hooks should be stateless and avoid modifying the global
process environment. The package must be installed by the image's
build recipe. Registration does not install model dependencies into the host.

Implement the required policy hooks:

| Method | Responsibility |
| --- | --- |
| `validate_policy(doc, path)` | Validate the complete composed policy document; do not load weights or access the network. |
| `resolve_settings(doc, benchmark_context, action_spec, checkpoints, devices, *, repository_root, environment)` | Construct settings from canonical peer capabilities, selected task identity, explicit method claims and resolved resources. Return settings exposing `record_raw_output: bool`; standalone serving passes `None` for the peer. |
| `describe_capabilities(settings)` | Describe input requirements and action output without constructing or loading a model. |
| `create_adapter(settings)` | Return a `PolicyAdapter` in `CREATED` state; defer expensive work to `_initialize`. |
| `identity(capabilities)` | Return RPC identity with `model_identity`, `normalization_identity`, `prompt_template_identity`, `action_codec_identity`, `runtime_versions` and optional `method_descriptor`. Preserve immutable checkpoint identity and truthful method metadata. |

The SDK also provides strict `mapping`, `exact_type`, `enum`, `number` and
`non_empty_string` helpers. Raise `PolicyPluginValidationError`,
`PolicyPluginCompatibilityError` or `PolicyPluginReferenceError`; OVLAB
translates these to its normal configuration errors. Do not import
`ovlab_benchctl` from a plugin.

Optional hooks have safe defaults:

| Hook | Use |
| --- | --- |
| `validate_observation(doc, benchmark_context)` | Check named inputs, shapes, proprioception, instruction and explicit methodological claims through canonical contracts. |
| `deployment_context(policy, deployment)` | Supply standalone serving context without creating a benchmark; model defaults and normalization belong here. |
| `scientific_policy(doc)` | Return a deep copy without machine-local execution paths; keep immutable hashes and scientific parameters. |
| `runtime_resources(doc, repository_root, environment)` | Return execution metadata for additional resources. |
| `runtime_mounts(doc, repository_root, environment)` | Return read-only directory mappings with `source`, `target`, optional `environment` and `required_files`; see the mount contract below. |
| `validate_checkpoint(entry, path)` | Validate method-specific artifact metadata. Registry `artifact_forms` or `family` selects the validator. |
| `validate_document`, `resolve_document`, `describe_document` | Extend descriptor-only inspection; declare kinds using registry `document_kinds`. |

Mount sources must be absolute host directories; targets must be normalized
absolute container paths. Targets may not overlap the configuration bundle,
resolved checkpoint, RPC or temporary directories, or another plugin mount.
Required files must be relative and remain inside the source directory.
Environment-variable names must be unique and may not override OVLAB's core
configuration/checkpoint variables. Both serving and evaluation validate these
contracts before starting containers. Dry runs show the mounts without requiring
the model directories to exist yet.

The `accepts_settings` and `accepts_capabilities` hooks support existing direct
Python callers of the compatibility helpers. Normal service startup selects the
qualified provider declared in the policy configuration; new plugins do not need these
hooks.

`PolicyAdapter` supplies the lifecycle and checks state transitions, prediction
shape, horizon and action contract. Implement `_initialize`, `_reset_episode`,
`_predict`, `_close`, and optionally `_end_episode`. Initialization returns
`PolicyCapabilities` with truthful observation requirements and output
`ActionSpec`. Predictions carry `PredictionId`, source `StepId`, action arrays,
timestamps and inference duration. Keep the runner's episode identity intact.
Never expose benchmark evaluation signals as model observations.

## Default categories and composition

Each model project supplies modular `configs/policy`, `deploy`, `registry`,
`train` (when supported), and `evaluate` defaults, with one commented example per
supported category in `examples/`. Ablations, task subsets and smoke budgets live
in user experiments or the small OVLAB verification set. Model projects do not
own benchmark or metric choices.

Training defaults may omit the dataset selection. Evaluation defaults contain
policy, deployment and checkpoint catalog choices; the experiment must supply
benchmark, metrics, protocol and artifacts. Composition resolves inheritance and
workflow patches first, then validates complete documents. Executing an
incomplete default fails explicitly.

Declare discoverable templates in manifest `configs.templates`, for example
`{"train": ["train/default.yaml"], "evaluate": ["evaluate/default.yaml"]}`.
`ovlab train profiles` distinguishes templates from complete runnable profiles.
A training extension can implement SDK `validate_profile_template`; its default
is strict complete-profile validation. The OpenVLA support package only relaxes
the missing dataset reference for its training templates and validates all other
fields normally.

A policy plugin can implement `validate_checkpoint_template(doc, path)` for a
policy whose artifact identity will come from training. The default calls strict
`validate_policy`. OVLAB invokes the template hook only in a workflow declaring
`checkpoint_from`; standalone serving and evaluation always require a complete
policy. QuIC templates omit adapter paths and hashes entirely. Finalization binds
real identities and normal validation runs before model startup.

Shared control libraries must be explicit source dependencies. Declare their
source IDs in each consuming provider descriptor
(`providers.<local-id>.control_sources`) and in its policy/build `sources`. OVLAB includes their bounded Python packages in frozen configuration
bundles; Docker recipes install the corresponding distributions. Local sources
are useful inside this checkout. Independently published remote projects must
pin external dependencies to Git commits instead of relying on sibling paths.

Benchmark projects use the same registry and explicit configuration origins.
`BenchmarkPlugin` API `"1"` owns their settings, catalog, native adapter and
deployment requirements; [LIBERO](../plugins/libero/README.md) is a concrete
project. Policies receive canonical `BenchmarkContext`, not benchmark-specific
YAML. A normalization key is a model setting and cannot establish peer identity.
Metric/report module metadata is reserved; their runtime modularization remains
subsequent work.

## First model and benchmark integrations

Start with the [CPU policy project](../examples/policy-plugin/README.md) for
persistent serving and the [independent CPU benchmark
project](../examples/benchmark-plugin/README.md) for the paired evaluation
pipeline. Each has its own manifest, Python package, default configuration,
action contract and image recipe. The benchmark example selects both projects
in one private registry and generates an explicit machine profile and Compose
environment file. Neither example changes the repository registry.

For a new benchmark, implement `BenchmarkPlugin` API `"1"`, declare a portable
`TaskCatalog`, and bind native tasks only during adapter initialization. Put
benchmark settings, native action interface, execution profiles and deployment
requirements in that project. Keep standard-library control hooks in the
entrypoint; typed catalog/capability code and native imports can load lazily.
See the [benchmark SDK guide](../code/packages/ovlab-benchmarks/README.md).

For a new model, implement API `"2"`, retain immutable checkpoint/codec identity,
and declare its own input/action requirements. Its default configuration must
remain usable for standalone serving without registering a benchmark just to
resolve an action interface. The resolver checks those declared contracts
against the separately selected benchmark before initialization.

Model `evaluate/default.yaml` files are composition templates, not complete
experiments. Add benchmark, metrics, protocol and artifact choices in an
experiment. For OFT also select the required wrist camera/proprioception profile;
for QuIC bind a real compatible adapter or use the explicit training checkpoint
handoff. The maintained [verification scenarios](../examples/README.md) show
complete compositions.

Keep machine settings separate from portable plugin defaults. A complete local
profile can supply absolute checkpoint paths without changing their registered
revision or expected hashes:

```yaml
schema_version: "0.1.0"
kind: local_profile
id: my-machine
paths:
  checkpoint_root: /absolute/ovlab-data/checkpoints
  dataset_root: /absolute/ovlab-data/datasets
  runs_root: /absolute/ovlab-data/runs
devices:
  primary_gpu: cuda:0
resources:
  checkpoints:
    my-checkpoint:
      local_path: /absolute/path/to/verified-checkpoint
```

Save the profile under ignored `configs/local/` and select it explicitly with
`OVLAB_LOCAL_PROFILE` (`deploy run` also accepts `--profile`). A benchmark
execution profile is a different input: LIBERO supplies
`plugin://libero/execution/egl.yaml` and `plugin://libero/execution/glfw.yaml`.
Pass it with `eval run --execution-profile` or `OVLAB_EXECUTION_PROFILE` when
needed. The CPU benchmark example needs no renderer execution profile. Local
checkpoint overrides choose a location for verification; they do not bypass
the catalog's revision/file identity checks.

## Configuration and checkpoints

A policy configuration retains this common envelope:

```yaml
schema_version: "0.1.0"
kind: policy
type: my_policy
provider: plugin:policy:my-model/default
settings:
  checkpoint_id: my-checkpoint
  input:
    camera: camera.primary.rgb
  runtime:
    quantization: none
  action:
    action_interface: plugin://my-model/interfaces/actions.yaml
  raw_output:
    enabled: false
```

The plugin validates its own settings and can add model-specific keys. OVLAB uses
`settings.action.action_interface` for the shared action contract and
`settings.checkpoint_id` for the primary checkpoint. A processor can use the
same checkpoint; if `processor_id` is declared, persistent deployment currently
requires it to match the primary checkpoint. Additional files/directories can
be exposed through `runtime_mounts`. Quantization overrides use
`settings.runtime.quantization` and must be accepted by the plugin.

Plugin configuration files live under the manifest's `configs.root` and use
`plugin://<project-id>/<relative-path>` references. A repository override stays
explicit, for example `configs/policies/my-openvla.yaml`:

```yaml
schema_version: "0.1.0"
kind: policy
type: openvla_vanilla
extends: plugin://openvla/policy/default.yaml
settings:
  runtime:
    quantization: 4bit
```

Select `policies/my-openvla.yaml` in the experiment's `components.policy`; core
composition merges it over the plugin default. Relative `extends` stays bounded
to its owning configuration tree; only an explicit plugin URI crosses project
boundaries. `ovlab://` selects the generic OVLAB defaults under `config/`.
User `configs/` and `experiments/` contents are ignored by Git and are applied
only when explicitly selected. OpenVLA and OFT accept `none`, `8bit` and `4bit`
as runtime settings; there is one policy default rather than a file per precision.
QuIC currently accepts `none` only.

Checkpoint registration belongs to the plugin project, for example
`plugin://openvla/registry/default.yaml`. Declare catalog paths in
`configs.checkpoints` so generic listing and training can discover their entries.
Local catalogs can extend plugin catalogs and override individual entries.
It binds weights to a repository, revision and verified file hashes. Sources
are implementation code; checkpoints are model artifacts. The generic settings
resolver provides checkpoints as mappings containing `source`, `revision`,
`expected_sha256`, optional verified `local_path`, and artifact/method metadata.
Use these resolved values; model loading inside a container must use its
read-only mounted path and offline settings.

Deployment YAML points to the policy and resource registry. The policy's
metadata selects its image and build role. Evaluation separately selects a
benchmark provider, which owns the benchmark image, build role, GPU needs,
mounts and execution environment. The orchestrator combines and verifies both
through one shared Compose template. New model profiles require no additional
Compose YAML. EGL/GLFW belongs to LIBERO's execution profile and applies only to
its benchmark service.

## Build and provenance contract

Each plugin owns its Docker recipe and dependency environment. Use an immutable
base image, locked/hash-verified dependencies, and `--no-deps --no-build-isolation`
when installing the local plugin against the resolved dependency lock. Copy
source from `external/extensions/<project>/<source>`. Install OVLAB contracts, SDK, RPC and CLI, and
install the plugin package. Model-specific dependencies belong to this image.

Policy images must provide `ovlab deploy serve` and `ovlab _worker health`, run
as UID/GID `10001`, support the mounted `/opt/ovlab/configs` bundle, and declare:

```dockerfile
LABEL io.github.ovlab.deployment.contract="resolved-checkpoint-quantization-config-bundle-v2" \
      io.github.ovlab.service.transport="tcp-persistent-v1"
ENTRYPOINT ["ovlab"]
```

The build runner supplies the existing `OVLAB_REVISION`,
`OVLAB_SOURCE_MANIFEST_SHA256`, `OVLAB_SOURCE_DIRTY`, `OVLAB_LOCK_SHA256`,
`OVLAB_DOCKERFILE_SHA256`, `OVLAB_IMAGE_REFERENCE`, `OVLAB_VERSION` build arguments
and BuildKit secrets `ovlab_source_manifest` and `ovlab_plugin_registry`.
Configured builds also supply `OVLAB_CONFIG_BUNDLE_SHA256` and the
`ovlab_config_bundle` secret. Declare the hash argument immediately before the
installer layer and consume it in the same `RUN` that extracts the configuration:

```dockerfile
ARG OVLAB_CONFIG_BUNDLE_SHA256=none
RUN --mount=type=secret,id=ovlab_plugin_registry \
    --mount=type=secret,id=ovlab_config_bundle \
    test -n "$OVLAB_CONFIG_BUNDLE_SHA256" \
    && bash /tmp/install-ovlab-packages.sh
```

BuildKit does not include secret contents in layer cache keys. Consuming the
bundle hash prevents a changed local override from reusing an older extracted
configuration, while dependency download layers above the argument remain cached.
The generic installer verifies the supplied bundle identity before publication.
Propagate build identity into image environment/labels as the built-in recipes
do, and copy the source manifest to `/usr/share/ovlab/source-manifest.json`.
The runner attaches the plugin-registry hash and a per-role
`io.github.ovlab.plugin-build.sha256` label. That identity binds the role,
plugin registration, required source commits/local content and recipe/locks.
Persistent deployment and evaluation reject a policy image with another role or
source identity, including when `build: false` is selected. Evaluation verifies
the independently selected benchmark image in the same way. Containers are then
started using the inspected immutable image ID, so moving a tag after preflight
cannot replace the verified image. Unrelated registry
entries do not change that per-role identity. After upgrading from images that
lack this label, rebuild the selected policy roles. The registry hash alone is
not used as a compatibility check. The manifest
includes the registry, pinned revisions and actual source-content hashes even
though materialized sources are ignored by Git.

Neutral configuration bundles contain selected `extensions/registry.json` and
`extensions/lock.json`, along with materialized project/control/configuration
inputs. The lock's source identities are portable; its separate locations map
points into the worker bundle. The temporary policy facade also supplies
`plugins/registry.json` and `plugins/runtime-profile.yaml` for existing callers.
OVLAB passes explicit registry/runtime locations into containers, so they do not
depend on an inherited machine-local profile. Bundle identity includes local source content; remote
selection is pinned by commit. Scientific configuration records the plugin
entrypoint, API and source identity. Runtime initialization should report the
same artifact/method identity through RPC.
Workers read remote commit pins and frozen local content identities from this
bundle. Resolving an experiment in a worker does not require Git, checkout
metadata, or access to the original local project. Missing frozen inputs fail
instead of falling back to host/cache/installed sources. Source synchronization and
checkout verification run on the host before building images.

Dry runs do not clone, build images, create containers or write generated Compose
files. Real runs remove temporary configuration directories, generated Compose
files and build-secret files after completion or failure. Managed sources and
built images remain available as intentional caches.

## Working example and validation

[`examples/policy-plugin`](../examples/policy-plugin/) contains a separate Python
project implementing a constant-action policy, a development Docker recipe and
a small registry. It demonstrates the integration contract; its predictions
have no scientific value. The development recipe reuses
`ovlab-reporting:local` and installs this checkout's current OVLAB packages,
then replaces inherited build metadata and the source manifest.
Replace this with your own pinned dependency environment for a real model.

```bash
./ovlab plugin inspect example-policy --registry examples/policy-plugin/registry.json
./ovlab plugin sync example-policy --registry examples/policy-plugin/registry.json
./ovlab plugin build example-policy --registry examples/policy-plugin/registry.json --dry-run
```

For deployment/evaluation with this registry, set `OVLAB_EXTENSION_REGISTRY` as well
as selecting an `example_policy` configuration. Copy the example project to
start a real integration, replace the runtime adapter, and register its Git
commit or local path. Preserve the policy/checkpoint distinction.

Before distributing a plugin, exercise validation failures, initialization,
episode reset, single actions/chunks, end/close, RPC identity and invalid input
contracts. Run acquisition tests against a temporary Git repository and a local
project. Test offline reuse and preservation of dirty destinations. The OVLAB
integration test `code/tests/unit/cli/test_plugins.py` proves that a new policy
resolves, predicts, plans both deployment modes, builds through a fake Docker
runner and cleans temporary files without editing the core dispatch.

## Optional training and checkpoint preparation

Training is an optional project capability with its own public interfaces in
`ovlab_policy_sdk.training`. A manifest `training` declaration identifies its
`TrainingPlugin` entrypoint, implementation IDs, expected checkpoint loaders and
training images. The backend validates its profile, compiles model-specific
scientific settings through injected host services, creates a trainer, validates
staged outputs and verifies loader compatibility. Core orchestration owns stores,
resource acquisition, execution and immutable publication. Plugins depend on the
SDK and public contracts and receive host operations as callbacks; they do not
import private CLI modules.

A `preparation` declaration binds a `CheckpointPreparationPlugin` entrypoint to
supported input loaders and a compatible image capability, such as merging an
adapter into dense inference weights. The plugin owns conversion semantics; core
owns verified inputs, container execution and artifact publication. An optional
`training.finalizer` image keeps plugin finalization dependencies outside the
infrastructure dataset image. These are separate extension contracts from policy
inference and can be omitted by evaluation-only integrations.

The OpenVLA project supports full, LoRA and experimental QuIC training with
`quantization: none`. Its 4-bit NF4 and 8-bit LLM.int8 declarations describe
inference through bitsandbytes; they do not imply QLoRA training support. OFT's
current integration is an evaluation reference, with no registered OFT trainer.


## Migration from submodules

The former five submodule revisions are recorded in the default source registry.
Existing physical checkouts are preserved and verified by `plugin sync`; a fresh
checkout runs `./ovlab plugin sync` to materialize them. A source directory that
still uses a `.git` file from an old submodule remains usable while its original
Git metadata exists. For a standalone checkout, preserve any edits, move that
old source directory aside, then sync a fresh clone. Do not delete the parent
repository's `.git/modules` while an old source still points there.

The immutable compound-PEFT historical snapshot lives in `references/` with its
original content manifest and provenance. It is a tracked reference artifact,
separate from the ignored source workspace.

Both evaluation workers use RPC `ovlab-policy-rpc/2.0.0`. The transport follows
named image/proprioception contracts and ActionSpec dimension, dtype and gripper
semantics; it imposes no OpenVLA camera name or seven-dimensional action layout.
Rebuild both policy and benchmark images together when upgrading RPC.

A model package owns its action-interface configuration as well as its adapter.
Its default policy must not refer to a benchmark project's internal YAML. OVLAB
compares the configured model ActionSpec with the independently declared native
benchmark ActionSpec; compatible fields must agree without implicit conversion.

Model evaluation defaults select the model deployment only. Benchmark execution
profiles belong to the benchmark. A complete experiment can explicitly set
`deployment.execution_profile: execution/local.yaml` (a local `extends` override)
or a `plugin://` URI; `ovlab eval run --execution-profile ...` overrides it for
one run. Builds freeze the same selected profile. Profiles must bind the same
benchmark provider. Benchmark-defined aliases remain available through
`--renderer`, without a global list of supported renderers.
