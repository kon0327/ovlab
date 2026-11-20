"""Bounded, JSON-safe evidence from the actual OpenVLA training forward pass."""

from __future__ import annotations

import math


def tensor_descriptor(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "elements": tensor.numel()}


def tensor_preview(tensor, limit=512):
    # Slice on the source device before copying. Never copy complete logits to CPU.
    values = tensor.detach().reshape(-1)[:limit].cpu().tolist()
    values = [None if isinstance(value, float) and not math.isfinite(value) else value for value in values]
    return {**tensor_descriptor(tensor), "values": values, "truncated": tensor.numel() > limit, "preview_limit": limit}


def action_token_counts(output, labels, action_token_begin_idx):
    # OpenVLA prepends vision tokens. The tail aligns with the text sequence;
    # shifting removes the final logit and the initial text target.
    targets = labels[:, 1:]
    logits = output.logits[:, -labels.shape[1]:-1]
    predicted = logits.detach().argmax(dim=-1)
    mask = (targets != -100) & (targets > action_token_begin_idx)
    return int(((predicted == targets) & mask).sum().item()), int(mask.sum().item())


def capture_chunk_data_flow(batch, output, tokenizer, *, dataset_records, pixel_dtype, optimizer):
    keys = ("input_ids", "attention_mask", "pixel_values", "actions", "action_mask")
    first = {key: tensor_preview(batch[key][0]) for key in keys}
    ids = batch["input_ids"][0][batch["attention_mask"][0].bool()]
    first["decoded_prompt"] = tokenizer.decode(ids[:512].tolist())
    first["pixel_values"] = tensor_preview(batch["pixel_values"][0].to(dtype=pixel_dtype))
    return {
        "schema_version": "ovlab.training-data-flow/v1", "microbatch_index": 0,
        "objective": "masked_l1", "dataset_records": int(dataset_records),
        "dataset_count_semantics": "selected observation start positions; future targets use original control steps",
        "batch_size": int(batch["input_ids"].shape[0]),
        "input_batch": {**{key: tensor_descriptor(batch[key]) for key in keys},
                        "pixel_values": {**tensor_descriptor(batch["pixel_values"]), "dtype": str(pixel_dtype)}},
        "input_record": first,
        "output_batch": {"actions": tensor_descriptor(output.actions), "loss": float(output.loss.detach()),
                         "valid_action_steps": int(batch["action_mask"].sum())},
        "output_record": {"actions": tensor_preview(output.actions[0]),
                          "targets": tensor_preview(batch["actions"][0]),
                          "valid_steps": tensor_preview(batch["action_mask"][0]),
                          "source": "first sample; normalized continuous chunk, padded future steps excluded from loss"},
        "optimizer": {key: value for key, value in optimizer.defaults.items()
                      if value is None or isinstance(value, (bool, str, int, float, tuple))},
    }


def capture_data_flow(batch, output, tokenizer, *, dataset_records, pixel_dtype, optimizer):
    first = {key: tensor_preview(batch[key][0]) for key in ("input_ids", "attention_mask", "labels", "pixel_values")}
    first["pixel_values"] = tensor_preview(batch["pixel_values"][0].to(dtype=pixel_dtype))
    ids = batch["input_ids"][0][batch["attention_mask"][0].bool()]
    first["decoded_prompt_and_targets"] = tokenizer.decode(ids[:512].tolist())
    pixels = batch["pixel_values"][0].detach().to(dtype=pixel_dtype)
    if pixels.ndim == 3:
        preview = pixels[:3, ::max(1, math.ceil(pixels.shape[1] / 32)), ::max(1, math.ceil(pixels.shape[2] / 32))].float().cpu()
        first["processed_image_preview"] = {
            "values": preview.permute(1, 2, 0).tolist(),
            "semantics": "first three transformed tensor channels; display rescales their finite range, not original RGB",
        }
    text_logits = output.logits[0, -batch["labels"].shape[1]:-1]
    targets = batch["labels"][0, 1:]
    selected = text_logits.detach()[targets != -100]
    predicted = selected.argmax(dim=-1)
    token_preview = tensor_preview(predicted, 256)
    return {
        "schema_version": "ovlab.training-data-flow/v1", "microbatch_index": 0,
        "dataset_records": int(dataset_records), "dataset_count_semantics": "RLDSDataset length reported by the prepared training stream",
        "batch_size": int(batch["input_ids"].shape[0]),
        "input_batch": {**{key: tensor_descriptor(batch[key]) for key in ("input_ids", "attention_mask", "labels", "pixel_values")},
            "pixel_values": {**tensor_descriptor(batch["pixel_values"]), "dtype": str(pixel_dtype)},
            "model_pixel_dtype": str(pixel_dtype), "label_ignore_index": -100},
        "input_record": first,
        "output_batch": {"logits": tensor_descriptor(output.logits), "loss": float(output.loss.detach().item()),
            "logits_semantics": "next-token scores; logits retain the vision-token positions"},
        "output_record": {"predicted_tokens": token_preview, "targets": tensor_preview(targets[targets != -100], 256),
            "decoded_predicted_tokens": tokenizer.decode(token_preview["values"]),
            "source": "first sample, supervised next-token positions; includes EOS if supervised"},
        "optimizer": {key: value for key, value in optimizer.defaults.items() if value is None or isinstance(value, (bool, str, int, float, tuple))},
    }
