# Training and checkpoint extensions

A plugin project can add training independently of policy inference. Its Python
code imports `ovlab_policy_sdk` and its own packages. OVLAB supplies orchestration,
checkpoint acquisition, run publication, container startup and host callbacks.
The plugin supplies model-specific schemas, scientific plan details, parameter
selection, the training runtime and artifact validation.

`PolicyPlugin` remains the inference boundary. Optional training uses a separate
`TrainingPlugin` contract in `ovlab_policy_sdk.training`. A project that provides
only inference omits `training` and `preparation` from its manifest.

## Registration

The project registration binds the complete integration to a local directory or
an immutable Git commit. Add optional extension metadata to its `plugin.json`:

```json
{
  "training": {
    "entrypoint": "my_model.training:training",
    "implementations": ["my-model-finetune"],
    "expected_loaders": ["my-model-dense-v1"],
    "images": {
      "my-model-finetune": {
        "role": "training-my-model",
        "image": "my-model-training:local",
        "image_env": "MY_MODEL_TRAINING_IMAGE"
      }
    },
    "finalizer": {
      "role": "finalizer-my-model",
      "image": "my-model-finalizer:local",
      "image_env": "MY_MODEL_FINALIZER_IMAGE"
    }
  },
  "preparation": {
    "entrypoint": "my_model.preparation:preparation",
    "expected_loaders": ["my-model-dense-v1"],
    "converter_version": "my-model-converter/v1"
  }
}
```

The referenced roles also need ordinary build recipes in the project's `builds`
map. The training image uses the generic OVLAB training-worker command and the
`offline-dataset-ro-base-ro-training-run-rw-v1` deployment contract. A separate
finalizer image installs the selected plugin's control package and uses the
`dataset-write-only-or-finalizer-v1` contract. It needs artifact validation
libraries, rather than the complete training environment.

A training profile selects a declared `trainer.implementation`. A finalized
checkpoint selects its provider through the declared `expected_loader`. Unknown
or ambiguous providers fail before execution. OVLAB never guesses a model family
from tensor names or chooses a default trainer for a modern profile. Old immutable
plans that lack trainer metadata can be inspected using a single registered
provider; multiple providers require an explicit trainer identity.

The optional `training.modules` map exists for legacy helper imports from older
OVLAB integrations. New plugins should call the SDK and their own modules
directly; they do not need to declare compatibility exports.

## Host hooks

Subclass `TrainingPlugin`, set `api_version = "1"`, and export a singleton or a
zero-argument class from the declared entrypoint. The required hooks are:

| Hook | Responsibility |
| --- | --- |
| `validate_profile(document)` | Validate plugin-owned settings and return a normalized mapping. |
| `estimated_vram(profile)` | Return a documented planning estimate in GiB. |
| `plan(services, profile, **options)` | Build scientific and execution plans using supplied resource services. |
| `create_trainer()` | Return a fresh `TrainerAdapter` instance for one worker run. |
| `validate_staging(context)` | Validate staged weights and return adapter metadata plus tensor inventory. |
| `expected_loader(scientific)` | Return the registered loader for the artifact being finalized. |
| `verify_checkpoint(root, checkpoint)` | Check model-specific artifact semantics after generic file hashes. |

`checkpoint_format(scientific)` defaults to `safetensors`; override it when the
plugin publishes another serialization. OVLAB records this value in the immutable
checkpoint manifest.

`container_resources(plan, dataset)` declares plugin-specific read-only dataset
mounts and worker environment variables. OVLAB validates the mounts and reserves
its configuration, checkpoint and run locations. A training image declaration
can set `gpu: false` for CPU execution. Model-dependent formats such as TFDS are
interpreted by this hook, not by the container orchestrator.

`validate_ablation(profile, factors)` is optional. Override it to declare which
configuration paths may vary and which shared data constraints your trainer
requires. Without the hook, ablation suites are explicitly unsupported. Suite
profiles can reference plugin defaults directly through `plugin://` URIs.

Profile validation runs on the host. Importing the extension and validating a
profile must not import GPU libraries, allocate model weights, download code or
write files. Put runtime imports inside `TrainerAdapter.train` or initialize.

The generic profile envelope requires `schema_version: ovlab.training-profile/v1`,
`kind: training`, a portable `id`, and `trainer.implementation`. The plugin
validates its complete scientific schema. Existing training deployment also
consumes `model`, `dataset`, `resources` and `checkpointing` for immutable resource
resolution and container mounts; follow the OpenVLA project as a complete working
example of that envelope.

`plan` receives `TrainingPlanningServices`, whose public callbacks are
`resolve_model(profile, **options)`, `resolve_dataset(profile, **options)` and
`estimate_vram(profile)`, plus `repository_root` and `model_data_root`. Keep host
paths in execution metadata. Scientific identity should include immutable model,
data and method identities, without depending on machine paths.

## Worker lifecycle and canonical artifacts

Implement `TrainerAdapter` and use `TrainingRunContext` from
`ovlab_policy_sdk.training_support`. OVLAB runs:

1. `preflight(plan)`.
2. `initialize(plan, context)`.
3. `train(plan, context)`.
4. `finalize(context)`.
5. `close()` in all execution paths.

An interrupted worker calls `interrupt(context)` before closing. Increment
`context.close_count` exactly once when releasing its resources. The worker
records lifecycle failures and retains staging evidence.

Write final weights under `context.staging_checkpoint / "weights-or-adapter"`.
Optional sibling directories `processor` and `auxiliary` hold processor assets
and immutable normalization or method metadata. The trainer must not publish
final checkpoint IDs itself. The finalizer validates staging through the plugin,
then OVLAB hashes, publishes and verifies the immutable bundle.

Useful SDK utilities include:

- `training_errors`: typed profile, plan, runtime and checkpoint errors.
- `training_identity`: canonical JSON, deterministic identities, SHA-256,
  bounded relative paths, file inventories and atomic JSON publication.
- `training_support`: lifecycle types, JSONL append, UTC timestamps,
  Safetensors header inspection and finite-payload validation.

`inspect_safetensors(path)` validates finite values and byte ranges using bounded
reads without loading a tensor runtime. Plugins can use it in finalization images.

## Evaluation handoff and offline preparation

Inference from a trained artifact is optional. `TrainingPlugin` defaults reject
`validate_checkpoint_binding`, `bind_training_checkpoint` and
`prepare_dense_checkpoint` with a clear unsupported-operation error. Override them
only when the integration implements and verifies the complete handoff.

During a handoff hook, `ovlab_policy_sdk.training_services.host_services()` returns
OVLAB-supplied checkpoint, run-store, reference-validation and document-validation
callbacks. The SDK declares the `TrainingHostServices` protocol; it does not import
OVLAB orchestration. The service scope ends when the hook returns, preventing
persistent process-global host state. Use these public callbacks when generic
publication or resource resolution is needed. Do not import private CLI modules
from a plugin.

A separate `CheckpointPreparationPlugin.prepare(base, checkpoint, output,
dtype=...)` implements offline conversion inside a compatible image. Its loader
must be explicitly declared under `preparation.expected_loaders`. For the generic
worker, `python -m ovlab_benchctl.checkpoint_preparation` receives `--base`,
`--checkpoint`, `--output`, `--dtype` and optional `--loader`; the loader normally
comes from `checkpoint.json`. The worker dispatches through the registry before
calling the preparation hook.

The preparation hook must preserve the source checkpoint and base model. Bind
converter version, source checkpoint identity, base identity, output file hashes
and converter image identity to any reusable prepared artifact. Verify these
identities before reusing conversion output.

## Testing a new implementation

Test validation and plan generation with small local documents. Test the worker
lifecycle and finalizer using a tiny fixture artifact, including corruption and
unknown-loader cases. Tests can register a temporary local project and wrap calls
in an explicit registry scope; no edit to OVLAB's model-specific code is needed.

`code/tests/unit/training/test_training_plugins.py` demonstrates an arbitrary
standalone trainer with its own schema, images, loader and checkpoint format. It
also verifies that these operations import no OpenVLA implementation.
