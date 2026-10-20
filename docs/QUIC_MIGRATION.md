# QuIC migration into OVLAB

The migration uses `/home/kony/dissertation/quic-petf-mpv/MIGRATE.md` and the
`quic_openvla` implementation supplied on 2026-10-15. It integrates an
experimental dense token OpenVLA adapter backend into OVLAB's existing offline
training, immutable checkpoint and policy-service lifecycle.

## Ownership

```text
OVLAB configuration / trainer / policy wrapper
    -> external/openvla-quic (standalone OpenVLA model, loader, neutral provider)
        -> external/quic-peft-vla (portable PyTorch method)

OVLAB benchmark adapter -> simulator and rollout evaluation
```

`quic-peft-vla` has its own repository and depends on Torch and Safetensors. It is
an independent MIT-licensed Git repository pinned as the
`external/quic-peft-vla` submodule. Its published Python distribution and import
names remain compatible. Tests, CI, source provenance and installation
instructions are in its README.
OpenVLA declares the versioned dependency; QuIC images install the local package
explicitly before the fork. Its determinant C1–C3 compounds, Cayley bases, injection,
adapter-only artifacts and in-memory merge know nothing about OVLAB,
Transformers, OpenVLA, TFDS or LIBERO. The OpenVLA fork supplies the host
identity and a model-level adapter API. Named QP profiles and A1–A9 target
resolution now live in `ovlab_benchctl/quic_experiments.py`, beside OVLAB's
experiment hierarchy and ablation workflow. The fork also has its own explicit
standalone CLI, tuner, verified episode-data interface, optional RLDS preparation
and HTTP deployment; none requires OVLAB or a benchmark simulator. The
standalone JSON schema uses exact target names and adapter parameters, without
interpreting OVLAB research profiles.

The model API exposes `enable_quic`, `quic_parameters`, `save_quic_adapter` and
`load_quic_adapter` on `OpenVLAForActionPrediction`; ordinary forward/generation
uses the inserted adapters. `quic_petf/host.py` supplies runtime loading,
checkpoint identity and preprocessing rather than experiment selection.
`quic_petf/runtime.py` composes checkpoint, processor and adapter for standalone
callers. Its Python namespace is now `quic_petf`; the distribution and command
remain `openvla-quic`. The OVLAB protocol provider is owned by
`ovlab_openvla_quic.runtime`. The external PETF package contains no WC skeleton,
legacy compound-PEFT bridge, or experiment-selection layer.

Adding the model API changes the verified model/integration source hashes.
Adapter artifacts bound to the previous source remain reproducible with the
previous source/image. New runs use the new identity; source metadata must not
be rewritten to make an old adapter appear compatible.

The historical compound-PEFT snapshot and QuIC-WC skeleton remain separate.
WC stays unavailable. No runtime path uses the legacy compound-PEFT backend
as the new method implementation.

## Training and ablations

The public training commands retain `ovlab.training-profile/v1`. QuIC uses
`trainer.implementation: openvla-quic`, `training.mode: peft` and
`training.peft.method: quic`. LoRA and full training keep their existing
defaults. Training profiles now accept bounded relative `extends`; mappings
deep-merge and sequences replace their parent, just like experiment components.

```bash
./ovlab train validate --profile configs/experiments/training/openvla-libero10-quic-smoke.yaml
./ovlab train ablate validate --suite configs/ablations/openvla-quic-placement-smoke.yaml
./ovlab train ablate plan --suite configs/ablations/openvla-quic-placement-smoke.yaml
./ovlab train ablate materialize \
  --suite configs/ablations/openvla-quic-placement-smoke.yaml \
  --output /tmp/ovlab-quic-review
```

Planning resolves verified local assets without loading model weights or using
network access. Materialization writes resolved profiles and commands. Use
`train run TARGET` for one variant, or `train ablate run --suite ...
--output NEW_DIRECTORY` to execute sequentially through the existing isolated
trainer, independent finalizer and reporting owner.

The example suite changes only placement and explicit target selectors (A1,
A3, A9); host identity, preprocessing, data partition, seed, optimizer, number
of steps and resource settings remain shared. Undeclared differences and
scientifically duplicate variants fail validation. These profiles are bounded
smoke recipes, not claims about research-quality training budgets.

Training resolves targets and parameter budgets before loading dense weights,
checks exact identity logits, freezes the host, checks all adapter gradients
and updates, and validates on held-out episodes. It saves adapter-only weights
and checks serialized adapter reload logits on the unchanged dense host.
`weights-or-adapter/adapter.json` and `adapter.safetensors` go through the
existing checkpoint finalizer, which checks tensors, hashes, manifest binding,
base checkpoint, method and dataset identity without importing Torch.

## Data and evaluation

Previously, dataset preparation verified and staged TFDS shards while the
reference trainer delegated splitting to the upstream RLDS stream. There was
no OVLAB-owned episode manifest or held-out teacher-forced validation path.
An explicit `dataset.partition` now records SHA256-ranked episode selection,
grouped train/validation assignment, frame sampling and frozen normalization.
Record IDs identify episodes; a shared HDF5 filename does not. Every frame
from an episode stays in the same split.

The data bridge uses OpenVLA's OXE source mapping and full-trajectory action
standardization, TF Lanczos3 resize and the host processor/action tokenizer.
It imports no benchmark or simulator. A new source requires a registered data
mapping, rather than changes to the QuIC method.

QuIC defaults to explicit host action statistics selected by
`training.normalization.unnorm_key`. The example uses the registered,
suite-finetuned LIBERO-10 dense merged host and declares
`model.adaptation_state: merged`. Vanilla OpenVLA does not contain LIBERO-10
statistics; use the new train-only fitting option for a vanilla host. Importing
an arbitrary historical normalization sidecar remains unsupported.

```bash
./ovlab train run configs/experiments/training/openvla-libero10-quic-smoke.yaml \
  --fit-action-normalization
./ovlab train ablate run \
  --suite configs/ablations/openvla-quic-placement-smoke.yaml \
  --fit-action-normalization --output /tmp/my-fitted-quic-ablation
```

The switch also works on `train validate`, `train plan`, and every `train ablate`
operation, so the effective configuration can be reviewed first. It resolves
to `dataset.partition.action_normalization: train-q99`, changes the scientific
identity, and never edits the input YAML. q01/q99 are fitted only on selected
training frames after the episode split; validation actions are excluded.
The resulting `fitted-action-normalization.json` and `normalization.json` are
saved in the run and bundled checkpoint. The adapter binds the exact contract
digest into its host identity; serving restores the same frozen statistics
directly from the verified adapter. Ablation variants must agree on the fitted
statistics and split manifest. The supplied
`openvla-vanilla-libero10-quic-smoke.yaml` declares train-only fitting explicitly.

Reference full/LoRA profiles may opt into the partition workflow with
`action_normalization: train-q99`; these statistics are fitted only on selected
training frames. Existing profiles without a partition retain their original
data path. The normalization choice is scientific configuration.

Each ablation trainer independently derives the same deterministic manifest
from immutable inputs. The suite preserves the first manifest and requires
exact equality for later variants. This gives shared split identity, while
TFDS scanning is repeated; it does not implement a shared physical frame cache.

Teacher-forced action-token CE, accuracy, normalized decoded L1 and invalid
token rate are offline diagnostics. Rollout tasks, initial states, horizons,
seeds, success and action metrics remain in OVLAB benchmark/protocol configs.
The RPC provider returns actions through an explicit configured codec, with
the gripper conversion applied once. Use the supplied adapter experiment
template with hashes from the finalized artifact. Its placeholder hashes
deliberately prevent accidental use as an evaluated result.

## Images, progress and evidence

See [Docker instructions](../deploy/docker/QUIC.md) for separate policy and
training images, pinned requirements and source-patch reproduction. Simulator
dependencies stay in the benchmark image. The original Python environment and
dataset/checkpoint files are never modified by this migration.

`ConsoleProgress` reports optimizer steps, offline validation batches and
rollout episodes on stderr. `OVLAB_PROGRESS=auto|on|off` controls display and
preserves JSON stdout. Training workers tee raw stdout/stderr into run logs;
container commands, plan, provenance, partition, runtime, bounded first-batch
evidence and metrics enter canonical training evidence.

The QuIC BF16 planning estimate is 23 GiB, based on the source's bounded
24-GiB-card runs; FP32 uses 45 GiB. These are heuristics, not measured guarantees
for arbitrary targets or batch sizes. Actual run peaks are recorded and checked
against the profile cap. The LoRA reference estimate remains separate.

## Validation scope and remaining work

Validation on 2026-10-15:

- OVLAB unit/contract/mock-integration regression: 702 passed, 6 skipped in the
  dependency-light `ovlab-tester` environment.
- Portable method/provider, numerical offline-objective and synthetic TFDS
  tests against the migrated fork in the pinned source environment: 27 passed,
  1 CUDA-only test skipped in that CPU invocation.
- Separate actual CUDA BF16 method smoke: PASS on RTX4090, Torch 2.2.0/CUDA12.1,
  exact identity and save/reload, finite adapter gradients, frozen host;
  peak allocated 18,120,704 bytes. Raw report: `/tmp/ovlab-quic-cuda-smoke.json`.
- Real TFDS source CPU smoke: catalogued 379 episodes, sampled 20 bounded frames
  into 10 train/10 validation frames. A separate synthetic TFDS test verifies
  full-trajectory standardization, finite validation and host statistics reuse.
- Actual isolated 7B BF16 training with `--fit-action-normalization`: completed
  3 optimizer steps with 14,880 trainable parameters and all 4 adapter tensors
  updated. The deterministic split contains 10 train/10 validation episodes,
  40 frames per split; fitted statistics exclude validation. Exact initial
  identity and serialized reload logit errors were both zero. Peak CUDA
  allocation was 17,395,496,960 bytes (16.20 GiB), reserved 20,478,689,280 bytes.
  Run: `openvla-libero10-quic-smoke_2026-10-15_16-20-25_3aaa3aad`;
  checkpoint: `checkpoint-49733a83387ef8645259eba34125e390`. The public CLI
  completed finalization/reporting, then host `train verify`, `checkpoint
  verify` and isolated `train report --verify` all passed. Full console log:
  `/tmp/ovlab-quic-training-final.log`.

The actual run also exposed and corrected host umask permissions, external HF
blob links in isolated mounts, loss of the canonical run ID, target-manifest
ordering and hardlinks across separate Docker bind mounts. Finalization copies
files on `EXDEV`, keeps atomic publication, and grants the shared host group
read access to worker-private staging evidence. Regression tests cover these
cases and canonical/legacy report mounts; actual non-root Docker probes cover
shared groups, finalization and reporting without modifying failed runs.

The subsequent normalization switch passed 154 focused profile/ablation/
finalizer/CLI tests, 8 fitted-provider CPU checks and 2 real TFDS runtime
tests. The latter place extreme actions entirely in validation and verify
that train-only q01/q99, gripper convention and frozen inference statistics
remain correct. No 7B weights are loaded by these checks.

Numerical tests and the completed 7B smoke establish training and artifact
wiring, not policy quality. Paired standalone/provider traces and benchmark
rollouts remain unverified. The source's earlier A1/A3/A9 results remain
source-project evidence.

This integration supports dense ordinary Linear targets and adapter-only
artifacts. Quantized training/hosts, active LoRA modules, OpenVLA-OFT hosts,
resume and full merged model export remain unsupported. The layer merge
primitive is available in the portable package but is not a published merged
checkpoint workflow. Original adapters retain their original host/source
identities; the migration does not bypass identity checks to load historical
artifacts into a different fork.
