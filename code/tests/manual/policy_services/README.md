# Policy-service smoke tests

These scripts preserve manually verified model-loading and inference paths while OVLAB's policy-service integrations are being designed.

- `openvla/` contains REST server launch variants for full-precision, FlashAttention 2, 8-bit, 4-bit, and LoRA checkpoints.
- `openvla_oft/` contains an OFT action-chunk inference demo.

The scripts are test assets, not production service implementations. They should eventually be replaced by automated smoke tests around the policy SDK and service containers.

Run scripts from any working directory using paths rooted at the OVLAB repository. The OpenVLA scripts require the OpenVLA policy environment plus `fastapi` and `uvicorn`. The OFT demo resolves its imports and sample observation from `external/openvla-oft` and must run in the OpenVLA-OFT environment.

Run one configurable OpenVLA inference with:

```bash
conda run -n openvla python code/tests/manual/policy_services/openvla/smoke_inference.py \
  --checkpoint openvla/openvla-7b \
  --quantization none \
  --instruction "pick up the object"
```

`--quantization` accepts `none`, `4bit`, or `8bit`. Use `--attn_implementation flash_attention_2` to exercise FlashAttention, `--unnorm_key` for a fine-tuned checkpoint's dataset statistics, and `--image_path` to replace the default synthetic black RGB image with a real observation. If the checkpoint exposes exactly one normalization key, the smoke test selects it automatically; otherwise it reports the valid choices and requires `--unnorm_key`.

See [`openvla/SMOKE_INSTRUCTION.md`](openvla/SMOKE_INSTRUCTION.md) for complete commands, checkpoint examples, and troubleshooting guidance.
