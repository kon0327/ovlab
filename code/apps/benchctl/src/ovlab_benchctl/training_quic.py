"""OVLAB training owner for the independent QuIC-PEFT method.

Schema validation is dependency-light. Model and tensor imports are confined to
the isolated training entry point; benchmark acquisition/evaluation stays OVLAB's.
"""
from __future__ import annotations

import re
import hashlib
import json
from collections.abc import Mapping

from .training_errors import TrainingProfileError


def validate_quic_settings(value: Mapping[str, object]) -> None:
    from .training_profiles import _keys, _mapping, _integer

    _keys(value, {"method", "adapter", "placement", "targets", "max_trainable_parameters"},
          {"method", "adapter", "placement", "targets"}, "training.peft")
    if value["method"] != "quic":
        raise TrainingProfileError("QuIC method must be quic")
    if "max_trainable_parameters" in value:
        _integer(value["max_trainable_parameters"], "max_trainable_parameters", 1, 10**10)
    placement = _mapping(value["placement"], "placement")
    _keys(placement, {"id", "action_proximal_last_n", "allow_action_proximal"}, {"id"}, "placement")
    if placement["id"] not in {"custom", *(f"A{i}" for i in range(1, 10))}:
        raise TrainingProfileError("placement.id must be custom or A1–A9")
    if "action_proximal_last_n" in placement:
        _integer(placement["action_proximal_last_n"], "action_proximal_last_n", 0, 1000)
    if "allow_action_proximal" in placement and type(placement["allow_action_proximal"]) is not bool:
        raise TrainingProfileError("allow_action_proximal must be boolean")

    def adapter(spec):
        spec = _mapping(spec, "adapter")
        _keys(spec, {"profile", "operation", "orders", "block_count", "block_size", "base_size",
                     "share_across_blocks", "orthogonal", "orientation", "construction_dtype"}, set(), "adapter")
        profile = spec.get("profile", "custom")
        if profile not in {"custom", "QP1", "QP2", "QP3", "QP4"}:
            raise TrainingProfileError("adapter.profile must be custom or QP1–QP4")
        if spec.get("operation", "determinant") != "determinant":
            raise TrainingProfileError("QuIC requires determinant compounds")
        orders = spec.get("orders")
        if orders is not None and (not isinstance(orders, list) or not orders
                or any(type(k) is not int or k not in {1, 2, 3} for k in orders)
                or orders != sorted(set(orders))):
            raise TrainingProfileError("adapter.orders must be ascending unique orders 1, 2, 3")
        for key in ("block_count", "block_size"):
            if key in spec:
                _integer(spec[key], f"adapter.{key}", 1, 10**6)
        if spec.get("base_size") not in {None, "largest_feasible"}:
            _integer(spec["base_size"], "adapter.base_size", 2, 10**6)
        for key in ("share_across_blocks", "orthogonal"):
            if key in spec and type(spec[key]) is not bool:
                raise TrainingProfileError(f"adapter.{key} must be boolean")
        if spec.get("orientation", "output") != "output":
            raise TrainingProfileError("QuIC supports output orientation only")
        if spec.get("construction_dtype", "float32") not in {"float32", "float64"}:
            raise TrainingProfileError("adapter.construction_dtype must be float32 or float64")
        fixed_orders = {"QP1": [1], "QP2": [1, 2], "QP3": [1, 2, 3], "QP4": [1, 2, 3]}
        if profile != "custom":
            if orders is not None and orders != fixed_orders[profile]:
                raise TrainingProfileError("named profile fixes compound orders; use custom for ablations")
            if not spec.get("orthogonal", True):
                raise TrainingProfileError("named profiles require orthogonal: true")
            if profile in {"QP1", "QP2"} and spec.get("share_across_blocks", False):
                raise TrainingProfileError(f"{profile} does not share blocks")
            if profile == "QP4" and not spec.get("share_across_blocks", True):
                raise TrainingProfileError("QP4 requires shared blocks")

    adapter(value["adapter"])
    targets = value["targets"]
    if not isinstance(targets, list) or not targets:
        raise TrainingProfileError("targets must be a nonempty list of explicit selectors")
    for target in targets:
        target = _mapping(target, "target")
        _keys(target, {"names", "pattern", "region", "layers", "profile", "adapter", "policy_active_only"}, set(), "target")
        if ("names" in target) == ("pattern" in target):
            raise TrainingProfileError("target requires exactly one of names or pattern")
        if "names" in target:
            names = target["names"]
            if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
                raise TrainingProfileError("target.names must contain nonempty module paths")
            if len(set(names)) != len(names):
                raise TrainingProfileError("target.names contains duplicates")
        if "pattern" in target:
            if not isinstance(target["pattern"], str) or not target["pattern"]:
                raise TrainingProfileError("target.pattern must be nonempty")
            try:
                re.compile(target["pattern"])
            except re.error as exc:
                raise TrainingProfileError(f"invalid target pattern: {exc}") from exc
        if "region" in target and target["region"] not in {"vision", "projector", "language", "action"}:
            raise TrainingProfileError("target.region must identify a supported OpenVLA region")
        if "layers" in target and (not isinstance(target["layers"], list) or not target["layers"]
                or any(type(i) is not int or i < 0 for i in target["layers"])):
            raise TrainingProfileError("target.layers must contain nonnegative layer indices")
        if "policy_active_only" in target and type(target["policy_active_only"]) is not bool:
            raise TrainingProfileError("target.policy_active_only must be boolean")
        if "profile" in target:
            adapter({"profile": target["profile"]})
        if "adapter" in target:
            adapter(target["adapter"])


def _validate_target_geometry(entry, version):
    """Validate Conv2d metadata without importing the tensor implementation."""
    from .training_errors import CheckpointBundleError
    kind = entry.get("module_type", "Linear")
    if kind == "Linear":
        return  # Legacy linear records retain their original v1 schema.
    if kind != "Conv2d" or version != 2:
        raise CheckpointBundleError("Unsupported QuIC target type or artifact version")
    try:
        inputs, outputs, groups = entry["in_channels"], entry["out_channels"], entry["groups"]
        if (any(type(value) is not int or value <= 0 for value in (inputs, outputs, groups))
                or inputs % groups or outputs % groups
                or inputs != entry["in_features"] or outputs != entry["out_features"]
                or entry["config"]["block_count"] % groups):
            raise ValueError("invalid channel/group geometry")
        for key in ("kernel_size", "stride", "dilation"):
            pair = entry[key]
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(value) is not int or value <= 0 for value in pair)):
                raise ValueError(f"invalid {key}")
        padding = entry["padding"]
        if isinstance(padding, str):
            if padding not in {"same", "valid"}:
                raise ValueError("invalid padding")
        elif (not isinstance(padding, list) or len(padding) != 2
              or any(type(value) is not int or value < 0 for value in padding)):
            raise ValueError("invalid padding")
        if entry["padding_mode"] not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError("invalid padding mode")
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointBundleError("Invalid QuIC Conv2d target geometry") from exc


def validate_staged_adapter(weights, scientific, *, run_root=None):
    """Independent finalizer validation; never imports model code or Torch."""
    from .training_errors import CheckpointBundleError
    from .training_identity import sha256_file
    from .training_runs import inspect_safetensors, _safetensors_header

    payload_path, tensors_path = weights / "adapter.json", weights / "adapter.safetensors"
    if not payload_path.is_file() or not tensors_path.is_file() or len(list(weights.iterdir())) != 2:
        raise CheckpointBundleError("QuIC bundle requires adapter.json and adapter.safetensors only")
    payload = json.loads(payload_path.read_text())
    if payload.get("format_version") not in {1, 2} or payload.get("tensor_sha256") != sha256_file(tensors_path):
        raise CheckpointBundleError("QuIC artifact format or tensor digest mismatch")
    metadata = payload.get("metadata", {})
    if metadata.get("base_checkpoint") != scientific["base_checkpoint"]:
        raise CheckpointBundleError("QuIC adapter base checkpoint does not match its plan")
    if metadata.get("peft") != scientific["training"]["peft"]:
        raise CheckpointBundleError("QuIC adapter configuration does not match its plan")
    if metadata.get("dataset") != scientific["dataset"]:
        raise CheckpointBundleError("QuIC adapter dataset identity does not match its plan")
    normalization = metadata.get("normalization_contract")
    if not isinstance(normalization, dict) or normalization.get("unnorm_key") != scientific["training"]["normalization"]["unnorm_key"]:
        raise CheckpointBundleError("QuIC adapter lacks its planned normalization contract")
    norm_values = {k: v for k, v in normalization.items() if k != "sha256"}
    if normalization.get("sha256") != hashlib.sha256(json.dumps(norm_values, sort_keys=True).encode()).hexdigest():
        raise CheckpointBundleError("QuIC normalization contract digest mismatch")
    mode = scientific.get("profile", {}).get("dataset", {}).get("partition", {}).get("action_normalization", "host")
    if normalization.get("mode", "host") != mode:
        raise CheckpointBundleError("QuIC normalization mode differs from the planned data contract")
    if mode == "train-q99" and payload.get("host_identity", {}).get("normalization_contract_sha256") != normalization["sha256"]:
        raise CheckpointBundleError("fitted normalization is not bound to the QuIC host identity")
    if metadata.get("preprocessing") != {"prompt_style": "modern", "center_crop": False}:
        raise CheckpointBundleError("QuIC preprocessing differs from the training contract")
    if run_root is not None:
        from .training_data import verify_episode_manifest
        partition_path, norm_path = run_root / "data-partition.json", run_root / "normalization.json"
        if not partition_path.is_file() or not norm_path.is_file():
            raise CheckpointBundleError("QuIC training run lacks split/normalization evidence")
        partition = json.loads(partition_path.read_text())
        verify_episode_manifest(partition)
        if metadata.get("split_manifest_id") != partition["manifest_id"]:
            raise CheckpointBundleError("QuIC adapter split identity differs from the training run")
        if normalization != json.loads(norm_path.read_text()):
            raise CheckpointBundleError("QuIC adapter normalization differs from the training run")
        if mode == "train-q99" and (partition.get("normalization", {}).get("fit_split") != "train"
                or partition["normalization"].get("statistics") != normalization["action_statistics"]):
            raise CheckpointBundleError("fitted QuIC normalization differs from the train-only partition statistics")
        if any(partition.get("source", {}).get(key) != value for key, value in scientific["dataset"].items()):
            raise CheckpointBundleError("QuIC split source differs from the planned dataset")
        targets_path = run_root / "target-manifest.json"
        if not targets_path.is_file():
            raise CheckpointBundleError("QuIC training run lacks its resolved target manifest")
        recorded = json.loads(targets_path.read_text()).get("targets")
        saved = payload.get("manifest")
        # Injection follows selector order; serialization follows model traversal.
        # Target identity is its full record keyed by module name, not list order.
        if (not isinstance(recorded, list) or not isinstance(saved, list)
                or len(recorded) != len(saved)
                or len({entry["name"] for entry in recorded}) != len(recorded)
                or {entry["name"]: entry for entry in recorded} != {entry["name"]: entry for entry in saved}):
            raise CheckpointBundleError("QuIC target manifest differs from the resolved training targets")
    binding = {key: payload[key] for key in ("format_version", "host_identity", "manifest")}
    header, _ = _safetensors_header(tensors_path)
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if header.get("__metadata__", {}).get("quic_binding_sha256") != digest:
        raise CheckpointBundleError("QuIC tensor/manifest binding mismatch")
    tensors = inspect_safetensors(tensors_path)
    expected = {}
    for entry in payload["manifest"]:
        _validate_target_geometry(entry, payload["format_version"])
        name, cfg = entry["name"], entry["config"]
        n, count = cfg["base_size"], 1 if cfg["share_across_blocks"] else cfg["block_count"]
        shape = [count, n * (n - 1) // 2] if cfg["orthogonal"] else [count, n, n]
        key = f"{name}.adapter.entries"
        if key in expected:
            raise CheckpointBundleError("duplicate QuIC target")
        expected[key] = (shape, "F32" if cfg["construction_dtype"] == "float32" else "F64")
    if not expected or {item["name"] for item in tensors} != set(expected):
        raise CheckpointBundleError("QuIC tensor inventory differs from declared targets")
    for item in tensors:
        if (item["shape"], item["dtype"]) != expected[item["name"]]:
            raise CheckpointBundleError("QuIC tensor shape or dtype differs from manifest")
    return payload, tensors


def _move_batch(batch, device, dtype):
    return {key: batch[key].to(device=device, dtype=dtype if key == "pixel_values" else None)
            for key in ("input_ids", "attention_mask", "pixel_values", "labels")}


def checked_action_transform(transform, tokenizer, action_tokenizer, *, action_dimension=7):
    """Reject malformed or truncated action supervision before the collator."""
    from .training_errors import TrainingRuntimeError
    if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
        raise TrainingRuntimeError("OpenVLA action training requires right padding and a pad token")

    def checked(frame):
        example = transform(frame)
        ids, labels = example["input_ids"], example["labels"]
        if len(ids) > tokenizer.model_max_length:
            raise TrainingRuntimeError("OpenVLA prompt exceeds model_max_length; truncation would remove action labels")
        if len(ids) < action_dimension + 1:
            raise TrainingRuntimeError("OpenVLA training sequence has too few action tokens")
        action_ids = ids[-action_dimension-1:-1]
        valid = (action_ids > action_tokenizer.action_token_begin_idx) & (action_ids < tokenizer.vocab_size)
        if not bool(valid.all()) or len(action_ids) != action_dimension:
            raise TrainingRuntimeError("host tokenizer does not round-trip seven valid OpenVLA action tokens")
        if tokenizer.eos_token_id is not None and int(ids[-1]) != tokenizer.eos_token_id:
            raise TrainingRuntimeError("OpenVLA training sequence must end with the host EOS")
        if len(labels) != len(ids) or bool((labels[:-action_dimension-1] != -100).any()) or int(labels[-1]) != -100:
            raise TrainingRuntimeError("QuIC labels must mask prompt and EOS")
        if not bool((labels[-action_dimension-1:-1] == action_ids).all()):
            raise TrainingRuntimeError("QuIC labels must supervise all seven action tokens")
        return example
    return checked


def _validation(model, loader, action_tokenizer, *, device, dtype, max_batches, description):
    """Teacher-forced, token-weighted diagnostics; never rollout success."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ovlab_core.progress import ConsoleProgress

    model.eval()
    total_loss = correct = tokens = invalid = batches = 0
    absolute_error = 0.0
    with ConsoleProgress(description, total=min(len(loader), max_batches), unit="batch") as progress:
        with torch.inference_mode():
            for index, raw in enumerate(loader):
                batch = _move_batch(raw, device, dtype)
                output = model(**batch, use_cache=False)
                labels = batch["labels"][:, 1:]
                logits = output.logits[:, -batch["labels"].shape[1]:-1]
                mask = (labels != -100) & (labels > action_tokenizer.action_token_begin_idx)
                target, selected = labels[mask], logits[mask].float()
                count = target.numel()
                if count:
                    prediction = selected.argmax(-1)
                    ce = F.cross_entropy(selected, target, reduction="sum")
                    if not torch.isfinite(ce):
                        raise RuntimeError("nonfinite teacher-forced validation loss")
                    total_loss += float(ce.cpu())
                    correct += int((prediction == target).sum())
                    invalid += int(((prediction <= action_tokenizer.action_token_begin_idx)
                                    | (prediction >= action_tokenizer.tokenizer.vocab_size)).sum())
                    predicted_actions = action_tokenizer.decode_token_ids_to_actions(prediction.cpu().numpy())
                    target_actions = action_tokenizer.decode_token_ids_to_actions(target.cpu().numpy())
                    absolute_error += float(np.abs(predicted_actions - target_actions).sum())
                    tokens += count
                batches += 1
                progress.advance(detail=f"action_tokens={tokens}")
                if index + 1 >= max_batches:
                    break
    if not tokens:
        raise RuntimeError("held-out validation contains no supervised action tokens")
    return {"role": "teacher_forced_validation", "batches": batches, "action_tokens": tokens,
            "action_token_cross_entropy": total_loss / tokens, "action_token_accuracy": correct / tokens,
            "decoded_normalized_l1": absolute_error / tokens, "invalid_action_token_rate": invalid / tokens}


def execute(plan, context):
    """Train adapter-only inside the existing offline OVLAB container lifecycle."""
    import gc
    import random
    import time
    import os
    from pathlib import Path
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .quic_experiments import resolve_targets, target_coverage
    from quic_petf.host import checkpoint_identity, layer_inventory, load_host, enable_gradient_checkpointing
    from quic_petf_vla import adapter_parameters, inject, save_adapter, load_adapter, disabled_adapters, QuICLinear, QuICConv2d
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform
    from ovlab_core.progress import ConsoleProgress
    from ovlab_openvla_common import parameter_inventory, cuda_allocator_snapshot, performance_sample, estimated_training_compute
    from .training_data import make_partitioned_datasets
    from .training_evidence import action_token_counts, capture_data_flow
    from .training_identity import atomic_json
    from .training_runs import TrainingRunStore, TRAINING_RESULT_SCHEMA, _utc_now
    from .training_errors import TrainingRuntimeError

    if not torch.cuda.is_available():
        raise TrainingRuntimeError("QuIC production training requires CUDA")
    scientific = plan["scientific"]
    training, profile = scientific["training"], scientific["profile"]
    settings = training["peft"]
    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if training["precision"] == "bf16" else torch.float32
    base = scientific["base_checkpoint"]
    host = {"checkpoint": os.environ.get("OVLAB_TRAINING_BASE_CHECKPOINT", "/checkpoints/base"),
            "checkpoint_id": base["resource_id"], "revision": base["revision"],
            "adaptation_state": profile["model"].get("adaptation_state", "none"), "family": "openvla_token", "device": str(device),
            "dtype": "bfloat16" if dtype == torch.bfloat16 else "float32",
            "unnorm_key": training["normalization"]["unnorm_key"], "prompt_style": "modern", "center_crop": False}
    inventory, num_layers = layer_inventory(host)
    targets, selection = resolve_targets(inventory, settings, num_layers)
    predicted = sum((1 if cfg.share_across_blocks else cfg.block_count)
                    * (cfg.base_size * (cfg.base_size - 1) // 2 if cfg.orthogonal else cfg.base_size**2)
                    for entry in selection for cfg in [targets[entry["name"]].resolve(entry["out_features"])])
    if predicted > settings.get("max_trainable_parameters", float("inf")):
        raise TrainingRuntimeError("QuIC trainable parameter budget exceeded before loading host weights")
    coverage = target_coverage(inventory, settings, targets, num_layers)
    atomic_json(context.root / "target-selection.json", {
        "targets": selection, "trainable_parameters": predicted, "coverage": coverage}, mode=0o644)
    data_root = Path(os.environ.get("OVLAB_TRAINING_DATA_ROOT", "/datasets/resolved"))
    dataset_name = os.environ.get("OVLAB_TRAINING_DATASET_NAME")
    if not dataset_name or not data_root.is_dir():
        raise TrainingRuntimeError("QuIC training data mount/name is missing")
    mode = profile["dataset"]["partition"].get("action_normalization", "host")
    train_ds = validation_ds = partition = None
    transform_holder = {}
    if mode == "train-q99":
        # Fit before loading weights: vanilla hosts need not have this key.
        # The data factory fits on raw selected train actions; it invokes the
        # deferred token transform only later when its datasets are iterated.
        config = json.loads((Path(host["checkpoint"]) / "config.json").read_text())
        image_size = config["image_sizes"][0]
        resize = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        train_ds, validation_ds, partition = make_partitioned_datasets(data_root, dataset_name,
            profile["dataset"]["partition"], lambda frame: transform_holder["transform"](frame),
            resize_resolution=resize, source_identity=scientific["dataset"],
            manifest_path=context.root / "data-partition.json")
        stats = partition["normalization"]["statistics"]
        normalization = {"unnorm_key": host["unnorm_key"], "action_statistics": stats, "mode": mode}
        normalization["sha256"] = hashlib.sha256(json.dumps(normalization, sort_keys=True, allow_nan=False).encode()).hexdigest()
        host["action_normalization"] = {host["unnorm_key"]: {"action": stats}}
        host["normalization_contract_sha256"] = normalization["sha256"]
        atomic_json(context.root / "fitted-action-normalization.json", {
            "schema_version": "ovlab.fitted-action-normalization/v1", "statistics_split": "train_only",
            "split_manifest_id": partition["manifest_id"], "normalization_contract": normalization,
            "norm_stats": host["action_normalization"],
        }, mode=0o644)
    host_identity = checkpoint_identity(host)
    model, processor, runtime = load_host(host, training=True)
    atomic_json(context.root / "quic-runtime.json", {"host_identity": host_identity, "runtime": runtime}, mode=0o644)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform, prompt_builder_fn=PurePromptBuilder,
        predict_stop_token=False)
    transform = checked_action_transform(transform, processor.tokenizer, action_tokenizer)
    transform_holder["transform"] = transform
    if mode == "host":
        stats = model.norm_stats[host["unnorm_key"]]["action"]
        normalization = {"unnorm_key": host["unnorm_key"], "action_statistics": stats, "mode": mode}
        normalization["sha256"] = hashlib.sha256(json.dumps(normalization, sort_keys=True, allow_nan=False).encode()).hexdigest()
        image_size = model.config.image_sizes[0]
        resize = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        train_ds, validation_ds, partition = make_partitioned_datasets(data_root, dataset_name,
            profile["dataset"]["partition"], transform, resize_resolution=resize,
            source_identity=scientific["dataset"], manifest_path=context.root / "data-partition.json", action_statistics=stats)
    atomic_json(context.root / "normalization.json", normalization, mode=0o644)
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
                                                processor.tokenizer.pad_token_id, padding_side="right")
    batch_size = training["per_device_batch_size"]
    loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0, collate_fn=collator)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, num_workers=0, collate_fn=collator)
    reference_batch = _move_batch(next(iter(val_loader)), device, dtype)

    def reference():
        model.eval()
        with torch.inference_mode():
            output = model(**reference_batch, use_cache=False)
            labels = reference_batch["labels"]
            return output.logits[:, -labels.shape[1]:-1][labels[:, 1:] != -100].float().cpu()

    original = reference()
    manifest = inject(model, targets)
    identity_logits = reference()
    identity_error = float((identity_logits - original).abs().max())
    if not torch.equal(identity_logits, original):
        raise TrainingRuntimeError(f"QuIC identity logit parity failed: maximum error {identity_error}")
    del original, identity_logits
    atomic_json(context.root / "target-manifest.json", {"targets": manifest}, mode=0o644)
    params = adapter_parameters(model)
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in params}:
        raise TrainingRuntimeError("QuIC optimizer selection contains non-adapter parameters")
    if training["gradient_checkpointing"]:
        atomic_json(context.root / "gradient-checkpointing.json", enable_gradient_checkpointing(model), mode=0o644)
    counts = parameter_inventory(model.named_parameters(), adapter_markers=(".adapter.",))
    initial = [p.detach().cpu().clone() for p in params]
    validation_args = dict(device=device, dtype=dtype, max_batches=training["validation_batches"])
    with disabled_adapters(model):
        baseline = _validation(model, val_loader, action_tokenizer, **validation_args, description="Host validation")
    optimizer = torch.optim.AdamW(params, lr=training["learning_rate"], weight_decay=0.0)
    iterator = iter(loader)
    peak_allocated = peak_reserved = 0
    with ConsoleProgress("QuIC training", total=training["max_steps"], unit="step") as progress:
        for step in range(1, training["max_steps"] + 1):
            model.train()
            model.vision_backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            before = cuda_allocator_snapshot(torch, device)
            losses, tokens, examples, correct, target_count = [], 0, 0, 0, 0
            for microbatch in range(training["gradient_accumulation_steps"]):
                raw = next(iterator)
                batch = _move_batch(raw, device, dtype)
                output = model(**batch, use_cache=False)
                loss = output.loss
                if loss is None or not torch.isfinite(loss):
                    raise TrainingRuntimeError("QuIC action-token loss is nonfinite or missing")
                if step == 1 and microbatch == 0:
                    atomic_json(context.root / "data-flow.json", capture_data_flow(raw, output,
                        processor.tokenizer, dataset_records=len(train_ds), pixel_dtype=dtype, optimizer=optimizer), mode=0o644)
                ok, n = action_token_counts(output, batch["labels"], action_tokenizer.action_token_begin_idx)
                correct += ok
                target_count += n
                tokens += int(batch["attention_mask"].sum())
                examples += batch["input_ids"].shape[0]
                losses.append(float(loss.detach()))
                (loss / training["gradient_accumulation_steps"]).backward()
                del output, batch, loss
            gradient_issues = [f"{name}: {'missing' if p.grad is None else 'nonfinite'}"
                               for name, p in model.named_parameters() if p.requires_grad
                               and (p.grad is None or not torch.isfinite(p.grad).all())]
            if gradient_issues:
                details = "; ".join(gradient_issues[:12])
                raise TrainingRuntimeError(f"QuIC adapter gradient missing/nonfinite ({len(gradient_issues)}): {details}")
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            after = cuda_allocator_snapshot(torch, device)
            peak_allocated = max(peak_allocated, after["peak_allocated_bytes"])
            peak_reserved = max(peak_reserved, after["peak_reserved_bytes"])
            if peak_allocated > float(profile["resources"]["max_vram_gib"]) * 1024**3:
                raise TrainingRuntimeError("QuIC run exceeded configured peak VRAM budget")
            compute = estimated_training_compute(counts["total"], token_count=tokens, trainable_parameter_count=counts["trainable"])
            metric = {"schema_version": "ovlab.training-metric/v1", "global_step": step, "optimizer_step": step,
                "timestamp_utc": _utc_now(), "training_loss": sum(losses) / len(losses),
                "examples_seen": examples, "non_padding_tokens": tokens,
                "microbatches": training["gradient_accumulation_steps"], "action_token_correct": correct,
                "action_token_count": target_count, "action_token_accuracy": correct / target_count if target_count else None,
                "learning_rate": training["learning_rate"], "gradient_norm": float(grad_norm),
                "step_duration_ms": elapsed_ms, "examples_per_second": examples / (elapsed_ms / 1000),
                "gpu_memory_allocated_bytes": after["allocated_bytes"], "gpu_memory_reserved_bytes": after["reserved_bytes"],
                "gpu_memory_peak_bytes": after["peak_allocated_bytes"], "gpu_memory_peak_reserved_bytes": after["peak_reserved_bytes"],
                "estimated_gflops": compute["estimated_gflops"],
                "performance": performance_sample(phase="training_optimizer_step", parameter_counts=counts,
                    memory_before=before, memory_after=after, compute=compute)}
            TrainingRunStore._append(context.root / "metrics.jsonl", metric)
            progress.advance(detail=f"loss={metric['training_loss']:.4f} VRAM={peak_allocated / 1024**3:.2f}GiB")
    changed = [not torch.equal(a, p.detach().cpu()) for a, p in zip(initial, params)]
    if not all(changed):
        raise TrainingRuntimeError("some targeted QuIC adapters did not update")
    final_validation = _validation(model, val_loader, action_tokenizer, **validation_args, description="Adapter validation")
    learned_reference = reference()
    np.save(context.root / "reload-reference.npy", learned_reference.numpy(), allow_pickle=False)
    weights = context.staging_checkpoint / "weights-or-adapter"
    metadata = {"base_checkpoint": base, "peft": settings, "dataset": scientific["dataset"],
                "normalization_contract": normalization, "split_manifest_id": partition["manifest_id"],
                "preprocessing": {"prompt_style": "modern", "center_crop": False},
                "seed": seed, "max_steps": training["max_steps"]}
    saved = save_adapter(model, weights, host_identity, metadata=metadata)
    processor.save_pretrained(context.staging_checkpoint / "processor")
    atomic_json(context.staging_checkpoint / "auxiliary" / "dataset_statistics.json", model.norm_stats, mode=0o644)
    if mode == "train-q99":
        atomic_json(context.staging_checkpoint / "auxiliary" / "fitted-action-normalization.json",
            json.loads((context.root / "fitted-action-normalization.json").read_text()), mode=0o644)
    # Reconstruct wrappers from the serialized artifact on the unchanged dense
    # host; this checks real safetensors reload, without holding a second 7B model.
    del optimizer, params
    for name, module in list(model.named_modules()):
        if isinstance(module, (QuICLinear, QuICConv2d)):
            parent_name, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_name) if parent_name else model, child, module.base_layer)
    load_adapter(model, weights, host_identity)
    reloaded = reference()
    if not torch.equal(reloaded, learned_reference):
        raise TrainingRuntimeError("QuIC save/reload logit parity failed")
    result = {"schema_version": TRAINING_RESULT_SCHEMA, "run_id": context.run_id, "status": "checkpointing",
        "optimizer_steps": training["max_steps"], "trainable_parameter_count": counts["trainable"],
        "frozen_parameter_count": counts["frozen"], "total_parameter_count": counts["total"],
        "adapter_parameter_count": counts["adapter"], "trainable_adapter_parameter_count": counts["trainable_adapter"],
        "changed_adapter_tensor_count": sum(changed), "peak_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved, "checkpoint_id": None, "failure": None,
        "identity_logit_max_absolute_error": identity_error, "reload_logit_max_absolute_error": 0.0,
        "baseline_validation": baseline, "adapter_validation": final_validation,
        "split_manifest_id": partition["manifest_id"], "tensor_sha256": saved["tensor_sha256"],
        "performance": {"schema_version": "ovlab.performance-summary/v1", "parameter_counts": counts,
            "cuda_allocator": {"source": "pytorch-cuda-caching-allocator", "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved}, "estimated_compute_method": "dense-parameter-token-proxy/training-v1"}}
    atomic_json(context.root / "result.json", result, exclusive=False, mode=0o644)
    TrainingRunStore._append(context.root / "events.jsonl", {"schema_version": "ovlab.training-event/v1",
        "state": "checkpointing", "timestamp": _utc_now(), "details": result})
    del iterator, loader, val_loader, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result
