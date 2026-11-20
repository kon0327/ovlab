"""OpenVLA trainer; runtime/model imports happen only during execution."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
from typing import Mapping
from ovlab_policy_sdk.training_errors import CheckpointBundleError, TrainingRuntimeError
from ovlab_policy_sdk.training_identity import atomic_json, canonical_json, identity, inventory, safe_relative, sha256_file
from ovlab_policy_sdk.training_support import (TrainerAdapter, TrainingRunContext, TRAINER_ADAPTER_VERSION, TRAINING_RESULT_SCHEMA, utc_now as _utc_now, inspect_safetensors, append_jsonl)

class OpenVlaTrainerAdapter(TrainerAdapter):
    """Production OpenVLA reference adapter; heavy imports occur only in train()."""

    def __init__(self) -> None:
        self._context: TrainingRunContext | None = None
        self._closed = False

    def capabilities(self) -> Mapping[str, object]:
        return {
            "id": "openvla-reference",
            "version": TRAINER_ADAPTER_VERSION,
            "modes": ("full", "peft"),
            "peft_methods": ("lora", "quic"),
            "lora_targets": ("all-linear",),
            "quantization": ("none",),
            "precision": ("bf16", "fp32"),
            "output_kinds": ("full", "adapter"),
            "network": "disabled",
        }

    def validate_profile(self, profile: Mapping[str, object]) -> None:
        training = profile.get("training")
        if not isinstance(training, dict) or training.get("mode") not in {"full", "peft"}:
            raise TrainingRuntimeError("OpenVLA trainer received an unresolved training mode")

    def resolve_plan(self, profile: Mapping[str, object], model: Mapping[str, object], dataset: Mapping[str, object]) -> Mapping[str, object]:
        return {"profile": dict(profile), "model": dict(model), "dataset": dict(dataset)}

    def preflight(self, plan: Mapping[str, object]) -> Mapping[str, object]:
        capabilities = plan.get("capabilities")
        if not isinstance(capabilities, dict) or capabilities.get("compatible") is not True:
            raise TrainingRuntimeError("training plan is not capability-compatible")
        return {"compatible": True, "issues": []}

    def initialize(self, plan: Mapping[str, object], run_context: TrainingRunContext) -> None:
        if self._context is not None:
            raise TrainingRuntimeError("OpenVLA trainer may be initialized exactly once")
        if os.environ.get("OVLAB_TRAINING_RUNTIME") != "isolated-container":
            raise TrainingRuntimeError("OpenVLA training may run only in the isolated production training container")
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise TrainingRuntimeError("OpenVLA training requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
        self._context = run_context

    def train(self, plan: Mapping[str, object], run_context: TrainingRunContext) -> Mapping[str, object]:
        if self._context is not run_context or self._closed:
            raise TrainingRuntimeError("OpenVLA trainer is not initialized")
        if plan["scientific"]["training"].get("peft", {}).get("method") == "quic":
            from .training_quic import execute
            return execute(plan, run_context)
        return _execute_openvla_training(plan, run_context)

    def finalize(self, run_context: TrainingRunContext) -> Mapping[str, object]:
        if not run_context.staging_checkpoint.is_dir():
            raise TrainingRuntimeError("trainer produced no final staging checkpoint")
        return {"staging_checkpoint": str(run_context.staging_checkpoint), "status": "checkpointing"}

    def interrupt(self, run_context: TrainingRunContext) -> None:
        atomic_json(run_context.root / "result.json", {
            "schema_version": TRAINING_RESULT_SCHEMA,
            "run_id": run_context.run_id,
            "status": "interrupted",
            "failure": {"type": "KeyboardInterrupt", "message": "training interrupted"},
        }, exclusive=False, mode=0o644)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._context is not None:
            self._context.close_count += 1
        self._context = None


def _save_training_weights(model, weights, auxiliary, training):
    model.save_pretrained(weights, safe_serialization=True)
    if training['mode'] != 'peft' or training['peft']['method'] != 'lora':
        return
    # PEFT expands all-linear into architecture-specific names in memory and
    # serializes that expansion. Retain it as evidence, while publishing the
    # declared portable policy accepted by our checkpoint loader/finalizer.
    path = weights / 'adapter_config.json'
    config = json.loads(path.read_text())
    targets = config.get('target_modules')
    if targets != 'all-linear' and targets != ['all-linear']:
        if not isinstance(targets, list) or not targets or any(not isinstance(x, str) or not x for x in targets):
            raise TrainingRuntimeError('PEFT did not serialize valid all-linear targets')
        atomic_json(auxiliary / 'lora-targets.json', {'target_policy': 'all-linear', 'resolved_targets': sorted(targets)})
    config['target_modules'] = 'all-linear'
    atomic_json(path, config, exclusive=False)


def _validate_lora_staging(weights, training):
    config_path = weights / "adapter_config.json"
    candidates = sorted(weights.glob("*.safetensors"))
    if not config_path.is_file() or len(candidates) != 1:
        raise CheckpointBundleError("LoRA bundle requires adapter_config.json and one safetensors file")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise CheckpointBundleError("checkpoint is not a LoRA PEFT adapter")
    expected = training.get("peft")
    if not isinstance(expected, dict):
        raise CheckpointBundleError("adapter plan lacks PEFT configuration")
    if adapter_config.get("r") != expected.get("rank") or adapter_config.get("lora_alpha") != expected.get("alpha"):
        raise CheckpointBundleError("adapter configuration does not match resolved LoRA plan")
    targets = adapter_config.get("target_modules")
    if targets != ["all-linear"] and targets != "all-linear":
        raise CheckpointBundleError("adapter target modules do not match OpenVLA all-linear plan")
    tensors = inspect_safetensors(candidates[0])
    if any("lora_" not in str(item["name"]).lower() for item in tensors):
        raise CheckpointBundleError("adapter contains tensors outside declared LoRA targets")
    return adapter_config, tensors


def _execute_openvla_training(plan: Mapping[str, object], context: TrainingRunContext) -> Mapping[str, object]:
    """Run the pinned OpenVLA AutoClass training path without merging adapters."""
    # Imports remain below the runtime gate so validate/plan/inspect stay dependency-light.
    import gc
    import hashlib
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
    from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
    from ovlab_openvla_common import (
        cuda_allocator_snapshot, estimated_training_compute, parameter_inventory,
        performance_sample, reset_cuda_peak,
    )

    if not torch.cuda.is_available():
        raise TrainingRuntimeError("OpenVLA production training requires CUDA")
    scientific = plan["scientific"]
    execution = plan["execution"]
    assert isinstance(scientific, dict) and isinstance(execution, dict)
    profile = scientific["profile"]
    training = scientific["training"]
    assert isinstance(profile, dict) and isinstance(training, dict)
    dataset_profile = profile["dataset"]
    assert isinstance(dataset_profile, dict)
    base_path = Path(os.environ.get("OVLAB_TRAINING_BASE_CHECKPOINT", "/checkpoints/base"))
    data_root = Path(os.environ.get("OVLAB_TRAINING_DATA_ROOT", "/datasets/resolved"))
    dataset_name = os.environ.get("OVLAB_TRAINING_DATASET_NAME")
    if not base_path.is_dir() or not data_root.is_dir() or not dataset_name:
        raise TrainingRuntimeError("training container mounts are incomplete")
    torch.manual_seed(int(training["seed"]))
    np.random.seed(int(training["seed"]) % (2**32))
    torch.cuda.manual_seed_all(int(training["seed"]))
    device = torch.device("cuda:0")
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)
    dtype = torch.bfloat16 if training["precision"] == "bf16" else torch.float32
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        base_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, local_files_only=True,
    ).to(device)
    if bool(training["gradient_checkpointing"]):
        model.gradient_checkpointing_enable()
    initial_hashes: dict[str, str] = {}
    if training["mode"] == "peft":
        peft = training["peft"]
        assert isinstance(peft, dict)
        model = get_peft_model(model, LoraConfig(
            r=int(peft["rank"]), lora_alpha=int(peft["alpha"]),
            lora_dropout=float(peft["dropout"]), target_modules="all-linear",
            init_lora_weights="gaussian", bias="none",
        ))
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                initial_hashes[name] = hashlib.sha256(parameter.detach().cpu().float().numpy().tobytes()).hexdigest()
    parameter_counts = parameter_inventory(model.named_parameters())
    trainable = parameter_counts["trainable"]
    frozen = parameter_counts["frozen"]
    if trainable <= 0 or (training["mode"] == "peft" and frozen <= 0):
        raise TrainingRuntimeError("resolved trainable/frozen parameter selection is invalid")
    if training["mode"] == "peft" and parameter_counts["trainable_non_adapter"] != 0:
        raise TrainingRuntimeError("LoRA training selected undeclared non-adapter parameters")
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(training["learning_rate"]))
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    validation_dataset = None
    if "partition" in dataset_profile:
        from .training_data import make_partitioned_datasets
        image_size = model.config.image_sizes[0]
        resize = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        dataset, validation_dataset, _ = make_partitioned_datasets(
            data_root, dataset_name, dataset_profile["partition"], transform,
            resize_resolution=resize, source_identity=scientific["dataset"],
            manifest_path=context.root / "data-partition.json",
        )
    else:
        dataset = RLDSDataset(
            data_root, dataset_name, transform,
            resize_resolution=tuple(model.config.image_sizes),
            shuffle_buffer_size=max(100, int(training["per_device_batch_size"]) * 8),
            image_aug=False,
        )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    loader = DataLoader(dataset, batch_size=int(training["per_device_batch_size"]), num_workers=0, collate_fn=collator)
    from ovlab_core.progress import ConsoleProgress
    progress = ConsoleProgress("OpenVLA training", total=int(training["max_steps"]), unit="step")
    metrics_path = context.root / "metrics.jsonl"
    events_path = context.root / "events.jsonl"
    steps = 0
    accumulation_started = None
    accumulation_tokens = 0
    accumulation_examples = 0
    accumulation_correct = 0
    accumulation_targets = 0
    from .training_evidence import action_token_counts, capture_data_flow
    initial_memory = cuda_allocator_snapshot(torch, device)
    run_peak_allocated = int(initial_memory.get("peak_allocated_bytes", 0))
    run_peak_reserved = int(initial_memory.get("peak_reserved_bytes", 0))
    optimizer.zero_grad(set_to_none=True)
    model.train()
    with progress:
        for batch_index, batch in enumerate(loader):
            if accumulation_started is None:
                torch.cuda.synchronize()
                memory_before = cuda_allocator_snapshot(torch, device)
                reset_cuda_peak(torch, device)
                accumulation_started = __import__("time").perf_counter_ns()
            attention_mask = batch["attention_mask"]
            accumulation_tokens += int(attention_mask.sum().item())
            accumulation_examples += int(attention_mask.shape[0])
            with torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(device, dtype=dtype),
                    labels=batch["labels"].to(device),
                )
                loss = output.loss
            if not torch.isfinite(loss):
                raise TrainingRuntimeError(f"non-finite loss at microbatch {batch_index}")
            if batch_index == 0:
                atomic_json(context.root / "data-flow.json", capture_data_flow(
                    batch, output, processor.tokenizer, dataset_records=len(dataset),
                    pixel_dtype=dtype, optimizer=optimizer,
                ), mode=0o644)
            correct, target_count = action_token_counts(output, batch["labels"].to(device), action_tokenizer.action_token_begin_idx)
            accumulation_correct += correct
            accumulation_targets += target_count
            (loss / int(training["gradient_accumulation_steps"])).backward()
            if (batch_index + 1) % int(training["gradient_accumulation_steps"]) != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), math.inf)
            if not torch.isfinite(grad_norm):
                raise TrainingRuntimeError(f"non-finite gradient norm at optimizer step {steps}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            torch.cuda.synchronize()
            elapsed_ms = (__import__("time").perf_counter_ns() - accumulation_started) / 1_000_000
            memory_after = cuda_allocator_snapshot(torch, device)
            run_peak_allocated = max(run_peak_allocated, int(memory_after.get("peak_allocated_bytes", 0)))
            run_peak_reserved = max(run_peak_reserved, int(memory_after.get("peak_reserved_bytes", 0)))
            compute = estimated_training_compute(
                parameter_counts["total"], token_count=accumulation_tokens,
                trainable_parameter_count=parameter_counts["trainable"],
            )
            metric = {
                "schema_version": "ovlab.training-metric/v1",
                "global_step": steps,
                "optimizer_step": steps,
                "timestamp_utc": _utc_now(),
                "training_loss": float(loss.detach().cpu()),
                "examples_seen": accumulation_examples,
                "non_padding_tokens": accumulation_tokens,
                "microbatches": int(training["gradient_accumulation_steps"]),
                "action_token_correct": accumulation_correct,
                "action_token_count": accumulation_targets,
                "action_token_accuracy": accumulation_correct / accumulation_targets if accumulation_targets else None,
                "learning_rate": float(training["learning_rate"]),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "step_duration_ms": elapsed_ms,
                "examples_per_second": accumulation_examples / (elapsed_ms / 1000.0),
                "gpu_memory_allocated_bytes": memory_after.get("allocated_bytes"),
                "gpu_memory_reserved_bytes": memory_after.get("reserved_bytes"),
                "gpu_memory_peak_bytes": memory_after.get("peak_allocated_bytes"),
                "gpu_memory_peak_reserved_bytes": memory_after.get("peak_reserved_bytes"),
                "estimated_gflops": compute["estimated_gflops"],
                "performance": performance_sample(
                    phase="training_optimizer_step",
                    parameter_counts=parameter_counts,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    compute=compute,
                ),
                "timing_method": "perf_counter_ns with torch.cuda.synchronize before and after",
            }
            append_jsonl(metrics_path, metric)
            progress.advance(detail=f"loss={metric['training_loss']:.4f}")
            accumulation_started = None
            accumulation_tokens = 0
            accumulation_examples = 0
            accumulation_correct = 0
            accumulation_targets = 0
            if steps >= int(training["max_steps"]):
                break
        if steps != int(training["max_steps"]):
            raise TrainingRuntimeError(f"dataset exhausted after {steps} optimizer steps")
    validation = None
    if validation_dataset is not None:
        from .training_quic import _validation
        validation_loader = DataLoader(validation_dataset, batch_size=int(training["per_device_batch_size"]), num_workers=0, collate_fn=collator)
        validation = _validation(model, validation_loader, action_tokenizer, device=device, dtype=dtype,
            max_batches=int(training.get("validation_batches", 1)), description="Held-out validation")
    checkpoint = context.staging_checkpoint
    weights = checkpoint / "weights-or-adapter"
    processor_dir = checkpoint / "processor"
    auxiliary = checkpoint / "auxiliary"
    weights.mkdir(parents=True)
    processor_dir.mkdir()
    auxiliary.mkdir()
    processor.save_pretrained(processor_dir)
    _save_training_weights(model, weights, auxiliary, training)
    save_dataset_statistics(dataset.dataset_statistics, auxiliary)
    changed = 0
    if training["mode"] == "peft":
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                final_hash = hashlib.sha256(parameter.detach().cpu().float().numpy().tobytes()).hexdigest()
                changed += int(initial_hashes.get(name) != final_hash)
        if changed <= 0:
            raise TrainingRuntimeError("LoRA adapter tensors did not change from initialization")
    result = {
        "schema_version": TRAINING_RESULT_SCHEMA,
        "run_id": context.run_id,
        "status": "checkpointing",
        "optimizer_steps": steps,
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": frozen,
        "total_parameter_count": parameter_counts["total"],
        "adapter_parameter_count": parameter_counts["adapter"],
        "trainable_adapter_parameter_count": parameter_counts["trainable_adapter"],
        "changed_adapter_tensor_count": changed if training["mode"] == "peft" else None,
        "peak_vram_bytes": run_peak_allocated,
        "peak_reserved_vram_bytes": run_peak_reserved,
        "performance": {
            "schema_version": "ovlab.performance-summary/v1",
            "parameter_counts": parameter_counts,
            "cuda_allocator": {
                "source": "pytorch-cuda-caching-allocator",
                "peak_allocated_bytes": run_peak_allocated,
                "peak_reserved_bytes": run_peak_reserved,
                "qualification": "process allocator peak across model load and recorded optimizer steps; not whole-device NVML usage",
            },
            "estimated_compute_method": "dense-parameter-token-proxy/training-v1",
        },
        "checkpoint_id": None,
        "failure": None,
        "validation": validation,
    }
    atomic_json(context.root / "result.json", result, exclusive=False, mode=0o644)
    append_jsonl(events_path, {"schema_version": "ovlab.training-event/v1", "state": "checkpointing", "timestamp": _utc_now(), "details": result})
    del optimizer, loader, dataset, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result
