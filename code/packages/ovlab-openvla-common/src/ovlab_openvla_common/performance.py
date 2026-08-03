"""Dependency-light performance telemetry shared by inference and training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


PERFORMANCE_TELEMETRY_SCHEMA = "ovlab.performance-telemetry/v1"
INFERENCE_COMPUTE_ESTIMATOR = "dense-parameter-token-proxy/inference-v1"
TRAINING_COMPUTE_ESTIMATOR = "dense-parameter-token-proxy/training-v1"
CUDA_ALLOCATOR_SOURCE = "pytorch-cuda-caching-allocator"


def parameter_inventory(
    named_parameters: Iterable[tuple[str, Any]],
    *,
    adapter_markers: tuple[str, ...] = ("lora_",),
) -> dict[str, int]:
    """Count runtime parameters without importing Torch.

    Adapter parameters are identified from explicit PEFT naming markers.  The
    returned counts describe the live runtime model, not checkpoint byte size.
    """
    total = trainable = adapter = trainable_adapter = 0
    lowered = tuple(value.lower() for value in adapter_markers)
    for name, parameter in named_parameters:
        count = int(parameter.numel())
        total += count
        is_trainable = bool(getattr(parameter, "requires_grad", False))
        is_adapter = any(marker in str(name).lower() for marker in lowered)
        trainable += count if is_trainable else 0
        adapter += count if is_adapter else 0
        trainable_adapter += count if is_trainable and is_adapter else 0
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "adapter": adapter,
        "trainable_adapter": trainable_adapter,
        "trainable_non_adapter": trainable - trainable_adapter,
    }


def estimated_inference_compute(
    parameter_count: int,
    *,
    input_token_count: int,
    output_token_count: int = 0,
    forward_pass_count: int = 1,
) -> dict[str, object]:
    """Return a labelled dense-model inference FLOPs proxy in GFLOPs."""
    values = (parameter_count, input_token_count, output_token_count, forward_pass_count)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("compute-estimator inputs must be non-negative integers")
    tokens = input_token_count + output_token_count
    operations = 2 * parameter_count * tokens * forward_pass_count
    return {
        "status": "available" if parameter_count and tokens and forward_pass_count else "unavailable",
        "estimated_gflops": operations / 1_000_000_000 if operations else None,
        "method": INFERENCE_COMPUTE_ESTIMATOR,
        "formula": "2 * runtime_parameter_count * (input_tokens + output_tokens) * forward_passes",
        "runtime_parameter_count": parameter_count,
        "input_token_count": input_token_count,
        "output_token_count": output_token_count,
        "forward_pass_count": forward_pass_count,
        "qualification": (
            "analytical dense-parameter/token proxy; not hardware FLOP measurement; "
            "does not model vision preprocessing, sparsity, quantization kernels, attention quadratic cost, or auxiliary heads"
        ),
    }


def estimated_training_compute(
    parameter_count: int,
    *,
    token_count: int,
    trainable_parameter_count: int | None = None,
    forward_backward_pass_count: int = 1,
) -> dict[str, object]:
    """Return a labelled forward+backward training FLOPs proxy in GFLOPs."""
    if trainable_parameter_count is None:
        trainable_parameter_count = parameter_count
    values = (parameter_count, trainable_parameter_count, token_count, forward_backward_pass_count)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("compute-estimator inputs must be non-negative integers")
    if trainable_parameter_count > parameter_count:
        raise ValueError("trainable_parameter_count cannot exceed parameter_count")
    operations = (
        (2 * parameter_count + 4 * trainable_parameter_count)
        * token_count * forward_backward_pass_count
    )
    return {
        "status": "available" if parameter_count and token_count and forward_backward_pass_count else "unavailable",
        "estimated_gflops": operations / 1_000_000_000 if operations else None,
        "method": TRAINING_COMPUTE_ESTIMATOR,
        "formula": "(2 * runtime_parameters + 4 * trainable_parameters) * non_padding_tokens * forward_backward_passes",
        "runtime_parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "token_count": token_count,
        "forward_backward_pass_count": forward_backward_pass_count,
        "qualification": (
            "analytical dense-parameter/token proxy; not hardware FLOP measurement; "
            "forward proxy covers all runtime parameters; backward proxy covers trainable parameters; "
            "does not model vision preprocessing, optimizer operations, sparsity, checkpoint recomputation, or auxiliary heads"
        ),
    }


def cuda_allocator_snapshot(torch_module: Any, device: Any) -> dict[str, object]:
    """Read PyTorch allocator counters, or report why CUDA telemetry is absent."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {
            "status": "unavailable",
            "source": CUDA_ALLOCATOR_SOURCE,
            "reason": "CUDA is unavailable",
        }
    return {
        "status": "available",
        "source": CUDA_ALLOCATOR_SOURCE,
        "device": str(device),
        "allocated_bytes": int(cuda.memory_allocated(device)),
        "reserved_bytes": int(cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(cuda.max_memory_reserved(device)),
        "qualification": "PyTorch process allocator counters; not whole-device NVML memory usage",
    }


def reset_cuda_peak(torch_module: Any, device: Any) -> None:
    """Begin a scoped allocator peak measurement when CUDA is available."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.reset_peak_memory_stats(device)


def performance_sample(
    *,
    phase: str,
    parameter_counts: dict[str, int],
    memory_before: dict[str, object],
    memory_after: dict[str, object],
    compute: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": PERFORMANCE_TELEMETRY_SCHEMA,
        "phase": phase,
        "parameter_counts": dict(parameter_counts),
        "cuda_memory_before": dict(memory_before),
        "cuda_memory_after": dict(memory_after),
        "estimated_compute": dict(compute),
    }
