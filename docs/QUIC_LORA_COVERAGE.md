# QuIC coverage of the LoRA target modules

The profiles in
[`configs/experiments/training/quic/lora-coverage/`](../configs/experiments/training/quic/lora-coverage/)
adapt the active OpenVLA modules selected by the literal LoRA suffix list:

```yaml
target_modules:
  - up_proj
  - fc2
  - q
  - down_proj
  - q_proj
  - k_proj
  - lm_head
  - o_proj
  - kv
  - fc1
  - gate_proj
  - fc3
  - proj
  - v_proj
  - qkv
```

A full-path regular expression reproduces exact leaf-name matching: `proj`
selects `patch_embed.proj`, for example, without selecting `q_proj` by substring.
Selection includes plain dense `torch.nn.Linear` and `torch.nn.Conv2d` layers.
The published `32 x 3 x 14 x 14` tensor is LoRA's rank-32 **A matrix reshaped
as a convolution kernel**. It identifies a convolutional target; it does not
give the frozen host's output channel count. QuIC rotates the actual host
convolution's output channels with the same output-side QP2 construction used
for linear layers. Convolution stride, padding, dilation, groups and bias
semantics stay part of the original frozen operation.

The placement is `custom` with `allow_action_proximal: true`. It includes all
matching active vision attention/MLP and patch projections, the multimodal
projector, all language attention/MLP layers, the final language layers and
`language_model.lm_head`. Every target uses QP2: determinant compounds of orders
1 and 2, eight unshared orthogonal blocks and the largest feasible base size.
The trainable parameter safety cap is 10 million; the resolved manifest records
the actual count. This is a comparison of module coverage; LoRA rank does not
determine an equivalent QuIC parameter budget.

OpenVLA obtains intermediate vision features before the final vision block,
bypassing final normalization and pooling/classifier heads. Unused final-block
and SigLIP attention-pool modules match LoRA's suffix list, but their weights
never affect policy outputs. `policy_active_only: true` explicitly excludes them.
Review `target-selection.json` for matched paths, excluded inactive paths and
coverage counts, and `target-manifest.json` for the adapted paths and shapes.
An unsupported active target must fail selection. Report the literal-match
count, adapted active count, inactive exclusions and active coverage fraction;
do not present the inactive exclusions as adapted modules. Suffix aliases that
do not occur in this checkpoint need no synthetic targets.

Architecture inspection of the pinned vanilla checkpoint on 2026-10-20,
with intermediate-feature reachability checked against training gradients,
resolved **439** literal matches and **426** active targets:
**424 Linear + 2 Conv2d**, with thirteen inactive exclusions and
100% active target coverage. The two convolution kernels have shapes
`1024 x 3 x 14 x 14` and `1152 x 3 x 14 x 14`. The selected regions contain
198 vision, 3 projector, 196 language and 29 action-proximal modules. QP2
resolves to **1,717,344 trainable parameters**. These are architecture and
parameter-budget checks; they do not establish the full model's training memory.

All four suites start independently from vanilla `openvla-7b` at immutable
revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`. Each run fits action q01/q99
only on its selected training frames and freezes those bounds for validation
and saved-adapter inference. It uses partition seed 42, training seed 7,
AdamW at constant learning rate `1e-4`, BF16, nonreentrant language and vision-block checkpointing,
and no weight quantization. The physical batch size is **4** and accumulation
is 4, giving 16 examples per optimizer update on one GPU.

| Suite | Smoke profile | Deep profile |
| --- | --- | --- |
| LIBERO-10 | `libero10-smoke.yaml` | `libero10-deep.yaml` |
| LIBERO-Object | `libero-object-smoke.yaml` | `libero-object-deep.yaml` |
| LIBERO-Spatial | `libero-spatial-smoke.yaml` | `libero-spatial-deep.yaml` |
| LIBERO-Goal | `libero-goal-smoke.yaml` | `libero-goal-deep.yaml` |

Every profile declares `validation_fraction: 0.05`. Whole episodes are assigned
within each task before frame sampling. The deterministic split reserves
`min(n - 1, max(1, ceil(0.05 * n)))` selected episodes for validation; report the
observed episode and frame fractions from `data-partition.json`. The fraction
can exceed 5% when a task's episode count is not a multiple of 20, and episode
lengths can make the validation frame fraction differ from the episode fraction.

The smoke selects up to 20 episodes per task, samples every eighth frame and
keeps at most four frames per episode. With 20 episodes in a task, one whole
episode (5%) is held out. It performs three optimizer updates and four
validation batches before and after training. The smoke exercises all selected
modules, identity initialization, finite gradients, optimizer updates,
adapter serialization and reload parity under the same physical batch as deep
training.

The deep run removes both episode and frame caps and uses every frame
(`frame_stride: 1`). Its budget is **50,000 optimizer updates per suite**,
equivalent to 800,000 training examples at effective batch 16. This is a fixed
budget, with convergence assessed from the resulting loss and held-out metrics.
The validation loader is finite; `validation_batches: 1000000` permits the
complete 5% holdout to be evaluated before and after training rather than just
four batches. The optimizer schedule and checkpoint policy follow the current
trainer: constant learning rate and one finalized adapter at the end. Deep
training starts from the immutable vanilla host after a successful smoke.

The configured VRAM cap is **24 GiB**. Full coverage with physical batch 4 needs a
measured smoke peak; the planner's historical bounded-target estimate is not
a memory guarantee for this recipe. Runtime records the actual peak and fails
if it exceeds the configured cap. The profiles do not reduce the physical
batch or targets automatically.

Checkpointing covers both Llama and individual timm vision blocks, including
the `get_intermediate_layers` route actually used by OpenVLA. timm's ordinary
checkpointing flag only affects its `forward_features` route. Vision dropout
remains disabled during adapter training; checkpoints are enabled by the
autograd context and preserve module names, targets and parameter counts.
`gradient-checkpointing.json` records the applied policy. The earlier language-only
policy reached a 25.17 GiB allocated peak in the first full-coverage optimizer
step; it does not meet this recipe's 24 GiB limit.

The first full-coverage smoke exposed an implementation allocation of 268.73
GiB in the vocabulary head: a broadcasted `matmul` expanded each compound
matrix across all batch/token positions. The corrected implementation uses
blockwise `bmm` and applies C2 through antisymmetric `R A R.T` contractions,
preserving the QP2 blocks, parameterization and targets. The large matrix of
second-order minors is not constructed during forward/backward. C2 application
uses at least the construction precision, then casts back to the activation
dtype; nonidentity BF16 rounding can differ from the old explicit path.
A CUDA BF16 regression with the failing shape `(4, 294, 32064)` measured a
**1.464 GiB** peak allocation increase for that adapter's forward/backward
(blockwise multiplication with explicit C2 alone used 3.814 GiB). This isolated
measurement excludes the host model and other layers. Rebuild the training
image before retrying the smoke; old failed run evidence is preserved.

The completed RTX 4090 smoke on 2026-10-20 used this 24 GiB recipe, BF16,
physical batch 4 and accumulation 4. All three optimizer steps finished in
about 60 seconds; **all 426 adapter tensors updated**. Peak allocated memory
was **16.49 GiB**, and peak reserved memory was **17.23 GiB**. Initial identity
and serialized reload logit errors were both exactly zero. The run and its
finalized adapter passed independent integrity verification:

- Run: `openvla-quic-lora-coverage-libero10-smoke_2026-10-20_15-44-33_f7b09d0c`
- Checkpoint: `checkpoint-a3823c9bd1c45f7e15af3bd7376f92d5`

This measurement covers LIBERO-10 smoke. The launcher still requires each
other selected suite's smoke to pass before starting any deep training.

## Run the two stages

Prepare the four registered RLDS datasets and the verified host checkpoint
before training. Dataset acquisition is explicit and separate from the offline
trainer:

```bash
bash deploy/scripts/build-images.sh dataset training-openvla-quic
./ovlab dataset fetch --source libero --name libero_10 --format openvla-rlds
./ovlab dataset fetch --source libero --name libero_object --format openvla-rlds
./ovlab dataset fetch --source libero --name libero_spatial --format openvla-rlds
./ovlab dataset fetch --source libero --name libero_goal --format openvla-rlds
```

Rebuild the dataset image as well as the training image: the dataset image
owns the strict checkpoint finalizer and must recognize convolutional QuIC
manifests. Rebuild `policy-openvla-quic` before serving these new adapters.

Validate all eight recipes without loading model weights or allocating a GPU:

```bash
bash deploy/scripts/train-quic-lora-coverage.sh --validate-only
```

Inspect resource plans, then execute the staged launcher:

```bash
bash deploy/scripts/train-quic-lora-coverage.sh --dry-run --output /tmp/quic-coverage-plan
bash deploy/scripts/train-quic-lora-coverage.sh --output /tmp/quic-coverage-run
```

The output directory must be new. The launcher uses normal `ovlab train run`
and verifies each run and checkpoint. It requires exact identity/reload logits,
finite held-out action-token loss and complete active-target coverage including
Conv2d. **All four smokes must pass before any deep training starts.** A failed
smoke, coverage check or integrity check stops the pipeline.
Optional repeated `--suite` arguments restrict both stages to selected suites
(for example `--suite libero-object`); deep cannot bypass its smoke gate.
Its JSON evidence identifies each suite and stage. These are four independent
datasets, so they are not variants of the shared-dataset placement ablation
schema. Their train/validation manifest identities and fitted statistics differ
as expected. Use the same corresponding profile partition and host for any
LoRA reference trained for a scientific method comparison.

For one stage, the ordinary CLI remains available:

```bash
./ovlab train validate \
  --profile configs/experiments/training/quic/lora-coverage/libero10-smoke.yaml
./ovlab train run configs/experiments/training/quic/lora-coverage/libero10-smoke.yaml
```

Record offline action-token CE, accuracy, decoded normalized action L1, invalid
action-token rate, parameter count, adapter bytes, GPU peak and optimizer time.
Offline validation alone does not establish LIBERO success rate: evaluate the
finalized adapters with the matching suite's rollout workflow and paired tasks,
initial states and seeds when reporting policy performance.
