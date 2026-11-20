"""Shared inference-time quantization contract for OpenVLA policy variants."""

from enum import Enum


class ModelQuantization(str, Enum):
    """Supported runtime weight representations.

    These values describe inference only. They do not imply QLoRA or any other
    quantized training method.
    """

    NONE = "none"
    BITSANDBYTES_INT8 = "8bit"
    BITSANDBYTES_NF4_4BIT = "4bit"

    def configuration(self) -> dict[str, object]:
        if self is ModelQuantization.NONE:
            return {"mode": "none"}
        if self is ModelQuantization.BITSANDBYTES_INT8:
            return {
                "mode": self.value,
                "backend": "bitsandbytes",
                "bits": 8,
                "quant_type": "llm_int8",
                "compute_dtype": "bfloat16",
            }
        return {
            "mode": self.value,
            "backend": "bitsandbytes",
            "bits": 4,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "storage_dtype": "float16",
            "double_quantization": True,
        }
