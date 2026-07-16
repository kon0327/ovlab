# OpenVLA Smoke-Test Instructions

This manual smoke test loads an OpenVLA checkpoint, performs one action prediction, and prints the resulting action. It reuses the same model-loading and inference paths as the preserved deployment harnesses.

## Prerequisites

Run the test from the OVLAB repository root or from this directory using the `openvla` Conda environment:

```bash
conda activate openvla
```

The environment must contain the OpenVLA dependencies, including a FlashAttention build compatible with its installed PyTorch version. Model downloads require access to Hugging Face unless the checkpoint is already cached or supplied as a local path.

For reproducible quantized inference, enforce OVLAB's compatibility pins:

```bash
python -m pip install -c deploy/environments/openvla/constraints.txt \
  "accelerate==0.30.1" \
  "bitsandbytes==0.43.1"
```

The OpenVLA package declares `accelerate>=0.25.0` without an upper bound. Do not use `accelerate 1.x` with the pinned `transformers 4.40.1` and `bitsandbytes 0.43.1`: its device dispatcher may either call the unsupported `model.to()` operation or leave a partially dispatched CPU/CUDA model when that call is bypassed.

## Base OpenVLA checkpoint

The base checkpoint contains normalization statistics for multiple training datasets. Select the dataset statistics explicitly with `--unnorm_key`. For a general Bridge-style smoke test:

```bash
python code/tests/manual/policy_services/openvla/smoke_inference.py \
  --checkpoint openvla/openvla-7b \
  --instruction "put the object in the basket" \
  --unnorm_key bridge_orig
```

## LIBERO checkpoint

Use the LIBERO-specific checkpoint and normalization key for a LIBERO-oriented test:

```bash
python code/tests/manual/policy_services/openvla/smoke_inference.py \
  --checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --instruction "put the object in the basket" \
  --unnorm_key libero_10
```

If a checkpoint exposes exactly one normalization key, the script selects it automatically. If it exposes multiple keys, the script reports the valid choices and requires `--unnorm_key`.

## Quantization

Select one of the supported modes with `--quantization`:

```bash
# Full precision
--quantization none

# 8-bit BitsAndBytes
--quantization 8bit

# 4-bit NF4 BitsAndBytes
--quantization 4bit
```

For example:

```bash
python code/tests/manual/policy_services/openvla/smoke_inference.py \
  --checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --quantization 4bit \
  --unnorm_key libero_10 \
  --instruction "pick up the object"
```

Quantized loading follows the upstream OpenVLA inference pattern: BitsAndBytes places the model on the active CUDA device during `from_pretrained()`. Do not add `device_map="auto"` to these harnesses; automatic CPU offload can split Llama rotary-embedding tensors across CPU and CUDA and cause a mixed-device inference failure. The smoke test validates the pinned quantization stack before loading a checkpoint.

The quantized server derives the image-input dtype from the loaded vision backbone. With the pinned stack this is normally `float16`; forcing the full-precision `bfloat16` input dtype causes a convolution error because the vision bias remains `float16`.

## FlashAttention

Enable FlashAttention explicitly:

```bash
--attn_implementation flash_attention_2
```

For the pinned OpenVLA environment, use the FlashAttention version compatible with the installed PyTorch ABI. An `undefined symbol` error from `flash_attn_2_cuda` indicates an incompatible binary build rather than an OVLAB import error.

## Input image

By default, the test creates a deterministic black `256 x 256` RGB image. This validates model loading and the inference path, but it does not assess policy quality.

Supply a real observation with:

```bash
--image_path /absolute/path/to/observation.png
```

The image is converted to RGB automatically. The synthetic image size can be changed with `--synthetic_image_size`.

## CLI reference

Display every available option without loading a checkpoint:

```bash
python code/tests/manual/policy_services/openvla/smoke_inference.py --help
```

## Expected result

A successful run ends with output similar to:

```text
Smoke test passed.
Checkpoint: openvla/openvla-7b-finetuned-libero-10
Quantization: none
Unnormalization key: libero_10
Action: [...]
```

TensorFlow registration messages and Hugging Face deprecation warnings printed during imports are not failures by themselves. Treat the smoke test as successful only when it prints `Smoke test passed.` and an action vector.
