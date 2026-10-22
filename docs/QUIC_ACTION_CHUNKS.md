# Optional OpenVLA QuIC action chunks

The existing token policy is the default. Omit `model.action_head` (or set it to
`null`) to keep its forward, token loss and single action inference. No compound
operator, QuIC wrapper or portable adapter format changes are required.

Enable the continuous variant in an OVLAB training profile:

```yaml
model:
  # Keep the profile's existing base_checkpoint/revision/processor settings.
  action_head:
    type: mlp_chunk
    horizon: 8
    hidden_dim: 512
```

The comparable LIBERO10 smoke pair is the token baseline
`configs/experiments/training/quic/action-chunks/libero10-token-smoke.yaml`
(head disabled) and `configs/experiments/training/quic/action-chunks/libero10-h8-smoke.yaml`
(head enabled). An H=1 continuous-head control is also available at
`configs/experiments/training/quic/action-chunks/libero10-h1-smoke.yaml`.
They keep batch 4, accumulation 4, a 5% whole episode validation partition,
train-only fitted normalization, three optimizer updates and the 24 GiB budget.
They are configurations, not evidence of a successful GPU run.

```bash
./ovlab train run configs/experiments/training/quic/action-chunks/libero10-h8-smoke.yaml --json
```

Rebuild the OpenVLA QuIC training, policy and dataset/finalizer images from these sources
before using container entry points:

```bash
bash deploy/scripts/build-images.sh policy-openvla-quic training-openvla-quic dataset
```
 The existing coverage study launcher validates token
cross entropy and is intended for the token profiles; run chunk profiles through
`ovlab train run` directly.

## Model and optimizer

```text
image -> frozen vision + QuIC -> projector + QuIC --+
                                                  v
instruction prompt --------------------------> Llama + QuIC
                                                  |
                                      final valid prompt state
                                                  |
                                  Linear -> GELU -> Linear
                                                  |
                                         [batch, horizon, 7]
```

This is a prompt state readout with a continuous MLP. It adds no query tokens,
attention mask changes, diffusion or proprioception. It is a separate recipe
from OpenVLA-OFT. The decoder is called directly with `use_cache=False` and
`output_hidden_states=False`; the vocabulary head and all-layer hidden state
collection are bypassed. The last valid prompt position is selected using the
attention mask. Labels/teacher-forced action tokens are rejected in this mode.

The new readout is trained in full alongside QuIC adapters. Its weights and
reference optimizer state stay FP32 with a BF16 backbone. At width 4096, hidden
width 512 and horizon 8 it has 2,126,392 parameters. The configured trainable
parameter budget includes both adapters and the head. `quic_parameters()` still
returns adapters only; `adaptation_parameters()` returns the optimizer's complete
set. The base model stays frozen after `enable_quic()`.

`language_model.lm_head` is explicitly recorded as inactive for this variant.
Use `policy_active_only: true` for selectors that also match it. The fully trained
action head is excluded from QuIC inventory and cannot be injected with QuIC.
LoRA comparisons need the same continuous head and the same head training rule;
the token and chunk variants have different active target inventories.

## Targets and loss

Targets are `a[t], ..., a[t+H-1]` in original adjacent control steps. Frame stride
samples observation/start positions after standardizing the complete trajectory;
it never increases the distance between future actions. Episode ends and changes
of instruction truncate the window. Remaining steps contain zeros and a false
validity mask. Windows cannot cross the train/validation episode split.

Training uses mean absolute error over valid steps and all seven coordinates,
computed in FP32. Prompt-only input tokens keep target actions outside the decoder
inputs. Validation reports `normalized_l1`, valid action steps and coordinate
counts; token cross entropy/accuracy do not describe this objective. The existing
`baseline_validation` is the frozen backbone with the initial randomly initialized
readout and disabled QuIC, not a pretrained continuous policy.

The same train-only q01/q99 statistics normalize every timestep. Inference clips
normalized masked coordinates to [-1,1], unnormalizes them, and clips the unmasked
EEF gripper coordinate to [0,1]. The OVLAB codec applies its declared gripper
conversion separately to each row of the chunk.

## Standalone and Python construction

For the standalone JSON workflow, put the same configuration under
`host.action_head`. Keep exact QuIC targets on active backbone modules; omit
`language_model.lm_head`. NPZ episode datasets, `train`, `evaluate`, `predict` and
`serve` support the variant. HTTP `/act` retains its `action` key: token mode
returns a vector and chunk mode returns a list of action vectors.

Standalone `prepare` with this option keeps complete action/instruction trajectories
and sampled frame indices alongside selected images. Thus `data.frame_stride`
samples observations while future targets remain consecutive control steps.
Previously prepared subsampled NPZ manifests without the full trajectories must
be prepared again. Plain adjacent-frame NPZ episodes work directly.

Direct HF construction supports `OpenVLAConfig(action_head=...)`. For an already
loaded dense model, configure the head **before** calling `enable_quic()`:

```python
model.configure_action_head({"type": "mlp_chunk", "horizon": 8, "hidden_dim": 512})
model.enable_quic(targets)
optimizer = torch.optim.AdamW(model.adaptation_parameters(), lr=1e-4)
output = model(input_ids=prompt_ids, attention_mask=prompt_mask,
               pixel_values=images, actions=future_actions, action_mask=valid_steps)
output.loss.backward()
```

`forward().actions` has shape `[B,H,7]` and contains normalized continuous
predictions. `predict_action()` dispatches to `predict_action_chunk()` when the
head is present; both return decoded `[H,7]` in this mode, including `H=1`.
Prediction convenience methods accept one observation. Build a fresh model to
switch between already configured head variants.

## Artifact and execution contracts

Token artifacts retain `adapter.json` plus `adapter.safetensors`. Chunk artifacts
also contain `action_head.safetensors`; the adapter JSON records its configuration
and SHA256, and safetensors metadata binds it to the host identity and head
configuration. Host identity includes the head configuration and integration
sources. Existing strict source fingerprint checks remain in force.

Use `model.save_quic_adapter()` / `model.load_quic_adapter()` or
`quic_petf.runtime.load_policy()` for the complete policy bundle. The lower level
portable `quic_petf_vla.save_adapter()` / `load_adapter()` operate on QuIC alone.
Runtime restores the head configuration from the artifact and rejects a conflicting
explicit setting, missing weights, changed digests, unexpected tensors and shape
mismatches. Load casts floating head weights to the constructed model's readout
dtype. Training reload checks erase the in-memory head before loading its weights.

The provider advertises its actual fixed horizon and returns the whole chunk.
OVLAB's existing runner owns execution: receding horizon runs the first action,
open loop executes the whole chunk, and a fixed replan interval executes its first
K actions. Keep predicted horizon H and executed actions K separate in ablations.
Start with H=1 versus H=8 and compare K=1 versus K=4 for H=8.

No tests, GPU training or rollouts were run as part of this implementation.
