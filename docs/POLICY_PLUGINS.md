# Model plugins in OVLAB

A policy plugin owns the model integration: configuration validation, settings,
observation compatibility, adapter construction, RPC identity and model-specific
runtime mounts. OVLAB owns configuration composition, checkpoint verification,
containers, RPC transport, evaluation lifecycle and artifacts. Adding a policy
family requires a plugin and a registry entry; it requires no new branch in the
resolver, CLI or deployment code.

The public interface is `ovlab_policy_sdk.plugin.PolicyPlugin` (API `"1"`).
Existing OpenVLA, merged LoRA, OFT and QuIC integrations use this same interface.
Their settings and artifact semantics remain in the integration packages. Model
libraries and weights are imported only during adapter initialization. Registry
listing and deployment planning work with the standard library alone when the
plugin's control hooks also avoid optional runtime imports.

For a complete first integration, follow the [runnable developer
README](../examples/policy-plugin/README.md). It includes all configuration
files and a CPU service using a tiny bundled artifact, so you can validate the
connection without downloading a model. Return here for the full contract.

## Registry and source selection

The default registry is
`code/apps/benchctl/src/ovlab_benchctl/plugins/registry.json`; it ships as package
data. The same selection is honored by explicit `environment` mappings on the Python
orchestrators without modifying `os.environ`. A project can supply a complete replacement at
`configs/plugins/registry.json`. `OVLAB_PLUGIN_REGISTRY` selects an explicit JSON
file instead. Relative registry and local source paths are relative to the OVLAB
repository. The `plugin` commands also accept `--registry PATH`.

A registry contains `schema_version: "ovlab.plugins/v1"`, `sources`, `plugins`
and `builds` mappings. Source IDs, policy configuration types, and image roles
are independent identifiers. Several policy variants can share a source and an
image role. A plugin can use several sources. Each build recipe must include every source
required by the plugins using that role; the deployment image must match the
recipe image. Shared profile definitions must be identical, and infrastructure
service/profile names are reserved. `deployment.gpu` defaults to `true`; set it
to `false` for a CPU-only policy image. Persistent deployment can also specify
`runtime.gpu: null`. The benchmark keeps its own renderer/GPU requirements.

```json
{
  "schema_version": "ovlab.plugins/v1",
  "sources": {
    "my-model": {
      "kind": "git",
      "url": "https://github.com/example/my-model.git",
      "commit": "0123456789abcdef0123456789abcdef01234567"
    }
  },
  "plugins": {
    "my_policy": {
      "id": "my-model",
      "family": "my_family",
      "runtime_status": "implemented",
      "entrypoint": "my_model.ovlab_plugin:plugin",
      "sources": ["my-model"],
      "deployment": {
        "profile": "my-model",
        "role": "policy-my-model",
        "image": "ovlab-policy-my-model:local",
        "image_env": "OVLAB_MY_MODEL_IMAGE",
        "policy_service": "policy-my-model",
        "benchmark_service": "benchmark-my-model",
        "quantization": ["none"],
        "observation_profile": "primary_rgb"
      }
    }
  },
  "builds": {
    "policy-my-model": {
      "dockerfile": "external/my-model/Dockerfile.ovlab",
      "image": "ovlab-policy-my-model:local",
      "sources": ["my-model"],
      "locks": ["external/my-model/requirements.lock"]
    }
  }
}
```

The SHA above illustrates the required format; replace it with a real full
40-character commit. Branch names and tags are not immutable registrations and
are rejected. URLs identify an upstream project, while commits identify its
version. The source is checked out at `external/<source-id>`; optional Git
`path` must equal that location. The whole `external/` directory is ignored by
Git. Git metadata is excluded from Docker build contexts.

For local development, replace the source entry with:

```json
{"kind": "local", "path": "/home/developer/projects/my-model"}
```

Local sources are copied into `external/<source-id>` for builds. Snapshots carry
content hashes and executable modes, so changes are visible in build identity.
Host control hooks can import the original local source; containers use the
installed package. Keep runtime imports in adapter initialization. A local path
is a development source, not a portable substitute for a published Git commit.

`sync` verifies existing Git origin, revision and cleanliness. It preserves dirty
checkouts and refuses to replace an unmanaged directory or an edited local
snapshot. A changed commit can be fetched and checked out when the checkout is
clean. Git-ignored files that would be copied into a build also make a checkout
dirty; Python bytecode, generated `*.egg-info`/`*.dist-info` metadata and MP4
recordings in `rollouts/` are excluded from both source identity and Docker
contexts. Tracked edits remain protected even for excluded paths. Offline sync uses only
existing commits and local projects. `--offline` controls source acquisition and
checkpoint resolution; dependency installation inside a Docker recipe may still
need network access. Initial
clones are published atomically, sync and builds hold locks on their selected sources, local snapshot replacement
retains a rollback copy until the new snapshot is installed, and source symlinks
may not escape the project. Sources with Git submodules should vendor or resolve
their additional sources explicitly in the registry/build recipe.

```bash
./ovlab plugin list --json
./ovlab plugin inspect my-model
./ovlab plugin sync my-model --dry-run
./ovlab plugin sync my-model
./ovlab plugin sync my-model --offline
./ovlab plugin build my-model --dry-run
./ovlab plugin build my-model
```

`sync` without an ID prepares every registered source, including benchmark
sources. `plugin build` without an ID builds registered policy images. The
existing helper `bash deploy/scripts/build-images.sh ROLE ...` builds any
registered role, including dataset, benchmark, reporting and training images;
without roles it builds all recipes. Build input selection is data in the
registry, not a shell case statement. Build automatically syncs the selected
recipe's sources first.

## Developing the plugin

The entrypoint is `module:object`. Export either a `PolicyPlugin` instance or a
zero-argument class. The implementation must advertise `api_version = "1"`;
unsupported versions fail before model initialization. Python packages can live
at the source root or under `src/`. The exported object must inherit `PolicyPlugin`. Instances are cached on their
loaded handle. Control hooks should be stateless and avoid modifying the global
process environment. The package must be installed by the image's
build recipe. Registration does not install model dependencies into the host.

Implement these four required methods:

| Method | Responsibility |
| --- | --- |
| `validate_policy(doc, path)` | Validate the complete composed policy document; do not load weights or access the network. |
| `resolve_settings(doc, benchmark_doc, action_spec, checkpoints, devices, *, repository_root, environment)` | Construct plugin-owned settings using resolved resource identities and the target action contract. Return an object exposing `record_raw_output: bool`. |
| `create_adapter(settings)` | Return a `PolicyAdapter` in `CREATED` state; defer expensive work to `_initialize`. |
| `identity(capabilities)` | Return RPC identity with `model_identity`, `method_identity`, `runtime_versions`. Preserve immutable checkpoint identity and truthful method metadata. |

The SDK also provides strict `mapping`, `exact_type`, `enum`, `number` and
`non_empty_string` helpers. Raise `PolicyPluginValidationError`,
`PolicyPluginCompatibilityError` or `PolicyPluginReferenceError`; OVLAB
translates these to its normal configuration errors. Do not import
`ovlab_benchctl` from a plugin.

Optional hooks have safe defaults:

| Hook | Use |
| --- | --- |
| `validate_observation(doc, benchmark_doc)` | Check cameras, shapes, proprioception and instruction requirements. |
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
plugin directly from the configured policy type; new plugins do not need these
hooks.

`PolicyAdapter` supplies the lifecycle and checks state transitions, prediction
shape, horizon and action contract. Implement `_initialize`, `_reset_episode`,
`_predict`, `_close`, and optionally `_end_episode`. Initialization returns
`PolicyCapabilities` with truthful observation requirements and output
`ActionSpec`. Predictions carry `PredictionId`, source `StepId`, action arrays,
timestamps and inference duration. Keep the runner's episode identity intact.
Never expose benchmark evaluation signals as model observations.

## Configuration and checkpoints

A policy configuration retains this common envelope:

```yaml
schema_version: "0.1.0"
kind: policy
type: my_policy
settings:
  checkpoint_id: my-checkpoint
  input:
    camera: camera.primary.rgb
  runtime:
    quantization: none
  action:
    action_interface: interfaces/actions/libero-osc-pose-v1.yaml
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

Checkpoint registration stays separate in `configs/resources/registry.yaml`.
It binds weights to a repository, revision and verified file hashes. Sources
are implementation code; checkpoints are model artifacts. The generic settings
resolver provides checkpoints as mappings containing `source`, `revision`,
`expected_sha256`, optional verified `local_path`, and artifact/method metadata.
Use these resolved values; model loading inside a container must use its
read-only mounted path and offline settings.

Deployment YAML points to the policy and resource registry. The plugin's
registry metadata selects the image and build role. Evaluation uses the same
registration to generate policy and benchmark services from one shared Compose
template. New model profiles require no additional Compose YAML. EGL/GLFW is a
benchmark execution choice and is applied to the generated benchmark service.

## Build and provenance contract

Each plugin owns its Docker recipe and dependency environment. Use an immutable
base image, locked/hash-verified dependencies, and `--no-deps --no-build-isolation`
when installing the local plugin against the resolved dependency lock. Copy
source from `external/<source-id>`. Install OVLAB contracts, SDK, RPC and CLI, and
install the plugin package. Model-specific dependencies belong to this image.

Policy images must provide `ovlab service serve` and `ovlab service health`, run
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
Propagate build identity into image environment/labels as the built-in recipes
do, and copy the source manifest to `/usr/share/ovlab/source-manifest.json`.
The runner attaches the plugin-registry hash and a per-role
`io.github.ovlab.plugin-build.sha256` label. That identity binds the role,
plugin registration, required source commits/local content and recipe/locks.
Persistent deployment and evaluation reject a policy image with another role or
source identity, including when `build: false` is selected. Containers are then
started using the inspected immutable image ID, so moving a tag after preflight
cannot replace the verified image. Unrelated registry
entries do not change that per-role identity. After upgrading from images that
lack this label, rebuild the selected policy roles. The registry hash alone is
not used as a compatibility check. The manifest
includes the registry, pinned revisions and actual source-content hashes even
though materialized sources are ignored by Git.

Every configuration bundle contains `plugins/registry.json`, its source identity
snapshot, and `plugins/runtime-profile.yaml`. OVLAB passes both registry and
runtime profile paths into the container, so images do not depend on an inherited
machine-local profile. The generated profile uses portable container paths and
selects CPU for a CPU-only policy. Containers receive `OVLAB_PLUGIN_REGISTRY` pointing to that
read-only snapshot. Bundle identity includes local source content; remote
selection is pinned by commit. Scientific configuration records the plugin
entrypoint, API and source identity. Runtime initialization should report the
same artifact/method identity through RPC.

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

For deployment/evaluation with this registry, set `OVLAB_PLUGIN_REGISTRY` as well
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

Model training algorithms remain separate specialized training components.
This API covers policy integration, model sources, image builds, serving and
evaluation; it does not pretend that arbitrary training loops share the policy
inference lifecycle.

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
