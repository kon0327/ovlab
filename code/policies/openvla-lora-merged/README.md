# OVLAB OpenVLA-LoRA Merged

This adapter identifies the official LIBERO-10 checkpoint as a LoRA
methodological reference whose published runtime form is merged full weights.
It reuses the validated OpenVLA full-weight inference mechanics and never loads,
reconstructs, or claims an active PEFT adapter.

Both `quantization: none` and the shared `quantization: 4bit` BitsAndBytes NF4
runtime are supported. Quantizing this already merged artifact is quantized
merged-LoRA inference, not QLoRA training. The method descriptor therefore
records runtime quantization separately from `training_quantization`.
