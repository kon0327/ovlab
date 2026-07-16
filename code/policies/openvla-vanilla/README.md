# OVLAB OpenVLA Vanilla

This package is the synchronous, local reference `PolicyAdapter` for OpenVLA
Vanilla. Importing it is Torch-free and does not allocate a model. The tested
heavy runtime is loaded only by `initialize()` in the existing `openvla` Conda
environment. The authoritative source is pinned at OpenVLA commit
`c8f03f48af692657d3060c19588038c7220e9af9`.

## Runtime contract

`OpenVlaVanillaSettings` is immutable and records the model and optional
processor source, revision/checksum metadata, `unnorm_key`, device, explicit
BF16/FP16/FP32 dtype, attention implementation, canonical camera, codec,
synchronization, and raw-output policy. Its canonical JSON representation has
a deterministic SHA-256 hash. `local_files_only=True` is the default and the
production adapter currently rejects disabling it. Repository identifiers are
resolved only from the existing Hugging Face cache before any model loader is
called; OVLAB never downloads or copies checkpoints.

Initialization uses the pinned API:

```python
AutoProcessor.from_pretrained(local_snapshot, trust_remote_code=True, local_files_only=True)
AutoModelForVision2Seq.from_pretrained(
    local_snapshot,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True,
    attn_implementation="flash_attention_2",
)
```

The model is put in evaluation mode and moved to the configured device. Each
prediction calls the processor as `processor(prompt, PIL.Image)` and then calls
`model.predict_action(..., unnorm_key=..., do_sample=False)` under
`torch.inference_mode()`. The configured normalization statistics are checked
during initialization. The checkpoint identity in capabilities records the
configured source, resolved local snapshot, revision metadata, pinned OpenVLA
commit, model/processor identities, selected action-statistics hash, expected
checksum (if supplied), settings hash, and the identity strength. It does not
rehash multi-gigabyte weights or copy them into run artifacts.

An `expected_checksum` is recorded as a user-supplied reference; it is not
misrepresented as a checksum verified by every initialization. For an explicit
offline weight fingerprint, run once against the resolved snapshot and retain
the output with experiment provenance:

```bash
find /path/to/snapshot -maxdepth 1 -type f -name '*.safetensors' -print0 \
  | sort -z | xargs -0 sha256sum
```

Only `camera.primary.rgb` is required by default: one raw HWC `uint8` RGB image
with shape `256×256×3`. LIBERO already rotates its native image by 180 degrees;
the policy does not rotate or flip it again. It copies the canonical array to
protect the input, then delegates resizing and normalization to the official
processor. Proprioception and evaluation/privileged signals are not consumed.

The exact default prompt is:

```text
In: What action should the robot take to {instruction.lower()}?
Out:
```

The current instruction is formatted for every prediction, so instruction
updates take effect immediately.

| Stage | Input | Output | Owner |
|---|---|---|---|
| Benchmark mapping | Native LIBERO image | Canonical RGB | Benchmark adapter |
| Policy preprocessing | Canonical RGB | Model tensor | Vanilla policy |
| Model decoding | Model output | Decoded OpenVLA action | Vanilla policy |
| Target codec | Decoded action | LIBERO-compatible action | OpenVLA common codec |
| Environment step | Decoded canonical action | Applied command | Benchmark adapter |

The policy emits exactly one `[1, 7]` float32 action. Translation and axis-angle
rotation pass through unchanged. The gripper conversion is
`-sign(2*g - 1)`, giving `+1 = closed` and `-1 = open` for LIBERO. No runner or
benchmark layer repeats it.

The lifecycle is `initialize → reset_episode → predict* → end_episode`, followed
by another episode or idempotent `close`. Vanilla is stateless and declares
horizon 1 only. Greedy evaluation is deterministic at the API level, but
bitwise GPU determinism is not claimed and global Torch determinism flags are
not changed.

`inference_duration_ns` covers local processor preprocessing, synchronized
model execution/decoding, and target-codec postprocessing. It excludes runner,
RPC, environment stepping, metrics, and artifact I/O. CUDA synchronization is
explicit and may reduce throughput. Phase timings are recorded as bounded
metadata. Raw output is disabled by default; when enabled, only the decoded
seven-value NumPy action before the target codec is retained—never logits,
hidden states, or Torch tensors.

## Tests

CPU/fake-runtime tests (no Torch, model, LIBERO, or network):

```bash
conda run -n ovlab-tester deploy/scripts/test.sh \
  code/tests/unit/policies/openvla_vanilla \
  code/tests/contract/policies/openvla_vanilla
```

The real tests require the existing local checkpoint and the tested `openvla`
environment:

```bash
export OVLAB_OPENVLA_CHECKPOINT=openvla/openvla-7b-finetuned-libero-10
export OVLAB_OPENVLA_UNNORM_KEY=libero_10
conda run -n openvla deploy/scripts/test.sh \
  code/tests/manual/policies/openvla_vanilla -m 'openvla and gpu and manual'
```

`OVLAB_OPENVLA_IMAGE_NPY` may point to a saved `256×256×3` uint8 input for the
legacy regression; otherwise its deterministic synthetic image is useful only
as a load/inference smoke input. `OVLAB_RUN_LIBERO_INTEGRATION=1` additionally
enables the bounded task-0 runner test and requires existing LIBERO assets.

RPC, Docker, quantization, LoRA, OFT, QuIC, training, and checkpoint acquisition
are deliberately deferred.
