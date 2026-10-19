# Training ablations and data partitions

OVLAB ablations compose ordinary `ovlab.training-profile/v1` profiles. They use
relative `extends`, strict YAML, the existing training planner, isolated trainer,
checkpoint finalizer, and training reports. The imported project does not become
an alternative experiment runner.

A suite uses `ovlab.training-ablation/v1`, a base profile, declared factor paths,
and a mapping from variant IDs to profiles. See
[`openvla-quic-placement-smoke.yaml`](../configs/ablations/openvla-quic-placement-smoke.yaml)
and its A1/A3/A9 QP2 profiles. The example is a three-step integration smoke,
not a scientific training budget. It starts from the registered **LIBERO-10
suite-finetuned merged dense host**, whose config already contains the required
normalization statistics. A9 is the source project's local placement extension.

```bash
# Pure schema/invariant validation: no dataset/model imports or GPU allocation.
./ovlab train ablate validate \
  --suite configs/ablations/openvla-quic-placement-smoke.yaml

# Resolve and verify the same immutable checkpoint and dataset for all variants.
./ovlab train ablate plan \
  --suite configs/ablations/openvla-quic-placement-smoke.yaml

# Reviewable resolved profiles, a scientific plan and exact child commands.
# OUTPUT must be a new directory; plans and profiles are immutable.
./ovlab train ablate materialize \
  --suite configs/ablations/openvla-quic-placement-smoke.yaml \
  --output /tmp/quic-placement-plan

# Sequential ordinary OVLAB training containers, finalizers and reports.
./ovlab train ablate run \
  --suite configs/ablations/openvla-quic-placement-smoke.yaml \
  --output /tmp/quic-placement-run
```

`materialize` and `run` preflight local resources; fetch datasets explicitly
first. A materialized profile can also be passed directly to `ovlab train run`.
Relative parent paths are confined to `configs/`, cycles fail, mappings merge,
and lists replace. Every final resolved profile is validated normally. The
suite allows only declared changes beneath `training.peft.adapter`,
`training.peft.placement`, `training.peft.targets`, and
`training.peft.max_trainable_parameters`. Model identity, normalization, data,
split policy, seed, optimizer budget, precision, and hardware contract must stay
identical. Duplicate scientific variants fail even if their IDs differ.

The suite identity includes resolved host/data identities and all variants.
Machine paths, image digests and availability remain in individual execution
plans. Optional `evaluation` records one common OVLAB evaluation experiment;
the suite runner trains and performs teacher-forced validation. Rollouts continue
through the ordinary OVLAB benchmark/deployment workflow, with the same task,
initial-state and seed selection for each trained checkpoint.

## Immutable shared split contract

The dataset store's `prepared/` directory contains verified original RLDS TFDS
shards. Its content identity is separate from an experiment's training partition.
An explicit `dataset.partition` specifies:

```yaml
partition:
  seed: 42
  validation_fraction: 0.2
  # Optional exact group names; omit to use every group in the source.
  # groups: [task-a, task-b]
  max_episodes_per_group: 20
  frame_stride: 8
  max_frames_per_episode: null
  action_normalization: host
```

Selection and train/validation membership use independent seeded SHA256 ranks
within each group, matching the source project's episode-split algorithm.
Selection uses immutable TFDS record IDs, never just an HDF5 path: several
episodes may share a source HDF5 file. Whole episodes stay in one split, and every
selected group must have at least two episodes. Validation receives
`min(n-1, max(1, ceil(n * validation_fraction)))` episodes. Frame stride/caps are
applied only after episode assignment. Duplicate IDs and split tampering fail.
Groups come from `episode_metadata.task_name`, a source file's basename when
available, or normalized language instruction. They are source metadata, not
benchmark API objects.

The isolated loader reads source mappings and complete-trajectory action
standardization from the host OpenVLA OXE registry, then decodes/resizes primary
RGB with TensorFlow Lanczos3, antialiasing, round/clip to uint8. The host image
processor, prompt builder and action tokenizer still own model preprocessing.
No LIBERO package, task suite or simulator is imported by the method or data
partition layer. A new RLDS source needs a registered source mapping and
standardizer; its format is not inferred from a benchmark name.

Each run writes immutable `data-partition.json`: selected IDs, groups, source and
selected frame counts, split membership, source content identity, metadata and
standardizer hashes, preprocessing, frozen normalization and manifest identity.
QuIC defaults to explicitly selected host statistics for both training and
validation. Set `dataset.partition.action_normalization: train-q99` or use
`--fit-action-normalization` to fit q01/q99 on selected training frames only.
The CLI flag resolves into the profile before planning and changes its scientific
identity; it does not rewrite source YAML. It requires an explicit episode
partition. Validation actions never fit those bounds. The source bridge's action
standardization precedes fitting, and its gripper mask remains in effect. The
resulting statistics are frozen, saved in the partition/normalization artifacts,
and carried with the QuIC adapter for reload and inference. Host statistics and
fitted statistics are separate scientific conditions. Reference full/LoRA
partition profiles also support `train-q99`.

```bash
./ovlab train validate --profile configs/training/openvla-libero10-quic-smoke.yaml \
  --fit-action-normalization
./ovlab train run configs/training/openvla-libero10-quic-smoke.yaml \
  --fit-action-normalization
./ovlab train ablate run --suite configs/ablations/openvla-quic-placement-smoke.yaml \
  --fit-action-normalization --output /tmp/quic-placement-fitted-run
```

The flag also applies to `train plan` and ablation `validate`, `plan`, and
`materialize`. For a suite, the override applies to its base and every variant;
the shared immutable manifest must contain the same selected episodes and
fitted statistics across all children. Use a new study output directory when
changing normalization. Existing manifests are verified and never overwritten.
Host-provided statistics may themselves originate from prior training; their
provenance is not a claim that the host has never seen validation tasks.

The first suite child publishes `shared-data-partition.json`; later children must
produce exactly the same manifest. A mismatch fails the suite and preserves
completed child evidence. Data is read-only and no large image cache is copied
into the repository. Current runtime recreates the deterministic catalog in each
isolated child and scans selected trajectories again on each dataset pass;
there is no shared decoded-frame cache. Training repeats its finite selected
frames to meet `max_steps`; validation is finite. DataLoader workers are zero to
preserve the single deterministic TensorFlow reader. Source projects' shuffled
map-style NPZ training and OVLAB's deterministic streaming order are different
training schedules and must not be described as bit-for-bit equivalent.

## Audit against the source workflow

Before this migration, OVLAB's LIBERO bridge already pinned/downloaded,
checksummed and prepared RLDS data, but preparation linked complete TFDS shards;
it did not build per-experiment task/frame manifests. Reference training used
OpenVLA's default RLDS stream, without an explicit OVLAB episode split or
teacher-forced validation. Benchmark evaluation independently chose rollout
tasks and initial states; it never reused a training validation split.

The source `quic-petf-mpv` project prepares selected episodes as NPZ files, shares
one manifest between an experiment's variants, accepts epochs or max steps,
optionally fits training-only normalization and combines offline diagnostics
with paired LIBERO rollouts. OVLAB now supports common episode/frame selection,
held-out validation, independent data/training seeds, fixed optimizer-step
budgets and adapter placement/parameterization ablations through its own
lifecycle. It deliberately keeps rollout evaluation with benchmark adapters.
The current training schema uses `max_steps`; epoch-to-step scheduling and a
persistent shared decoded cache are not implemented.

The source's **vanilla** OpenVLA LIBERO experiments additionally load
`configs/normalization/libero_10.json`, whose statistics came from the
suite-finetuned host config. Vanilla `openvla-7b` does not itself contain a
`libero_10` key. The bundled OVLAB suite therefore uses the suite-finetuned dense
host rather than silently applying absent or newly fitted normalization. Exact
reproduction of that vanilla branch needs a registered immutable normalization
sidecar and an explicit serving handoff; do not replace only the checkpoint ID.
Alternatively, `--fit-action-normalization` creates a new train-only normalization
contract for the selected demonstrations and persists it with the adapter. This
supports a vanilla host without claiming numerical equivalence to the source's
suite-derived-statistics condition.

The smoke budget, one seed, source CPU data tests, and checkpoint reload checks
verify integration contracts. They do not establish rollout quality or a CUDA
training result. Scientific comparisons should derive a larger shared base
profile, use multiple planned seeds/studies, and keep rollout horizons/states
fixed. Training validation loss and token accuracy are diagnostic metrics,
not rollout success.
