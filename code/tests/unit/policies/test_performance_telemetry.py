from types import SimpleNamespace

from ovlab_openvla_common import (
    cuda_allocator_snapshot, estimated_inference_compute, estimated_training_compute,
    parameter_inventory, performance_sample, reset_cuda_peak,
)


class Parameter:
    def __init__(self, count, trainable):
        self._count = count
        self.requires_grad = trainable

    def numel(self):
        return self._count


class Cuda:
    reset_device = None

    @staticmethod
    def is_available(): return True

    @staticmethod
    def memory_allocated(device): return 100

    @staticmethod
    def memory_reserved(device): return 200

    @staticmethod
    def max_memory_allocated(device): return 300

    @staticmethod
    def max_memory_reserved(device): return 400

    @classmethod
    def reset_peak_memory_stats(cls, device): cls.reset_device = device


def test_parameter_inventory_separates_live_adapter_parameters():
    counts = parameter_inventory([
        ("base.weight", Parameter(100, False)),
        ("layer.lora_A.weight", Parameter(10, True)),
        ("layer.lora_B.weight", Parameter(20, True)),
    ])
    assert counts == {
        "total": 130, "trainable": 30, "frozen": 100, "adapter": 30,
        "trainable_adapter": 30, "trainable_non_adapter": 0,
    }


def test_compute_proxies_disclose_formula_inputs_and_are_not_measurements():
    inference = estimated_inference_compute(1_000, input_token_count=10, output_token_count=2)
    training = estimated_training_compute(1_000, token_count=10)
    assert inference["estimated_gflops"] == 0.000024
    assert training["estimated_gflops"] == 0.00006
    lora = estimated_training_compute(
        1_000, trainable_parameter_count=100, token_count=10,
    )
    assert lora["estimated_gflops"] == 0.000024
    assert lora["trainable_parameter_count"] == 100
    assert "not hardware FLOP measurement" in inference["qualification"]
    assert "not hardware FLOP measurement" in training["qualification"]


def test_cuda_snapshot_is_allocator_scoped_and_performance_sample_is_versioned():
    torch = SimpleNamespace(cuda=Cuda)
    snapshot = cuda_allocator_snapshot(torch, "cuda:0")
    reset_cuda_peak(torch, "cuda:0")
    sample = performance_sample(
        phase="inference", parameter_counts={"total": 1},
        memory_before=snapshot, memory_after=snapshot,
        compute=estimated_inference_compute(1, input_token_count=1),
    )
    assert snapshot["peak_reserved_bytes"] == 400
    assert "not whole-device NVML" in snapshot["qualification"]
    assert Cuda.reset_device == "cuda:0"
    assert sample["schema_version"] == "ovlab.performance-telemetry/v1"
