"""Run one configurable OpenVLA inference as a manual smoke test."""

import json
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from deploy import OpenVLAServer as FullPrecisionServer  # noqa: E402
from deploy_4bit import OpenVLAServer as FourBitServer  # noqa: E402
from deploy_8bit import OpenVLAServer as EightBitServer  # noqa: E402


@dataclass
class DeployConfig:
    """Configuration for a single OpenVLA smoke-test inference."""

    checkpoint: Union[str, Path] = "openvla/openvla-7b"
    quantization: str = "none"
    attn_implementation: Optional[str] = None
    instruction: str = "pick up the object"
    unnorm_key: Optional[str] = None
    image_path: Optional[Path] = None
    synthetic_image_size: int = 256


def validate_quantized_stack(quantization: str) -> None:
    """Reject library combinations known to break OpenVLA quantized loading."""
    if quantization.lower() == "none":
        return

    expected_versions = {
        "accelerate": "0.30.1",
        "bitsandbytes": "0.43.1",
        "transformers": "4.40.1",
    }
    mismatches = []
    for package, expected in expected_versions.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not installed"
        if installed != expected:
            mismatches.append(f"{package}=={installed} (expected {expected})")

    if mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(
            "Incompatible OpenVLA quantization stack: "
            f"{details}. Install deploy/environments/openvla/constraints.txt before running 4-bit or 8-bit inference."
        )


def load_test_image(cfg: DeployConfig) -> np.ndarray:
    """Load a configured RGB image or create a deterministic synthetic image."""
    if cfg.image_path is None:
        if cfg.synthetic_image_size <= 0:
            raise ValueError("synthetic_image_size must be positive")
        return np.zeros((cfg.synthetic_image_size, cfg.synthetic_image_size, 3), dtype=np.uint8)

    image_path = cfg.image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Smoke-test image does not exist: {image_path}")
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def create_server(cfg: DeployConfig):
    """Create the deploy server matching the requested quantization mode."""
    server_types = {
        "none": FullPrecisionServer,
        "4bit": FourBitServer,
        "8bit": EightBitServer,
    }
    try:
        server_type = server_types[cfg.quantization.lower()]
    except KeyError as exc:
        supported = ", ".join(server_types)
        raise ValueError(f"Unsupported quantization '{cfg.quantization}'; choose one of: {supported}") from exc

    return server_type(cfg.checkpoint, cfg.attn_implementation)


def resolve_unnorm_key(server, requested_key: Optional[str]) -> str:
    """Validate or infer the dataset-statistics key exposed by a checkpoint."""
    norm_stats = getattr(server.vla, "norm_stats", None)
    if not isinstance(norm_stats, dict) or not norm_stats:
        raise RuntimeError("The loaded checkpoint does not expose any action normalization statistics")

    available_keys = tuple(sorted(norm_stats))
    if requested_key is not None:
        if requested_key not in norm_stats:
            available = ", ".join(available_keys)
            raise ValueError(f"Unknown unnorm_key '{requested_key}' for this checkpoint; choose one of: {available}")
        return requested_key

    if len(available_keys) == 1:
        selected_key = available_keys[0]
        print(f"Using the checkpoint's only available unnorm_key: {selected_key}")
        return selected_key

    available = ", ".join(available_keys)
    raise ValueError(f"This checkpoint provides multiple normalization datasets; set --unnorm_key to one of: {available}")


@draccus.wrap()
def smoke_inference(cfg: DeployConfig) -> None:
    """Load one policy variant, run one action prediction, and print the result."""
    validate_quantized_stack(cfg.quantization)
    image = load_test_image(cfg)
    server = create_server(cfg)
    unnorm_key = resolve_unnorm_key(server, cfg.unnorm_key)
    payload = {
        "image": image,
        "instruction": cfg.instruction,
        "unnorm_key": unnorm_key,
    }

    response = server.predict_action(payload)
    if response == "error":
        raise RuntimeError("OpenVLA action prediction failed; inspect the preceding server traceback")

    action = json.loads(response.body.decode("utf-8"))
    print("Smoke test passed.")
    print(f"Checkpoint: {cfg.checkpoint}")
    print(f"Quantization: {cfg.quantization}")
    print(f"Unnormalization key: {unnorm_key}")
    print(f"Action: {np.asarray(action)}")


if __name__ == "__main__":
    smoke_inference()
