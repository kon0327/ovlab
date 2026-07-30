"""OpenVLA Vanilla policy-service entry point for the isolated local RPC."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
from dataclasses import replace
from collections.abc import Mapping

import numpy as np

from ovlab_openvla_common import OpenVlaModelSource, vanilla_base_method_descriptor
from ovlab_openvla_vanilla import (
    InferenceSynchronization,
    ModelDType,
    ModelQuantization,
    OpenVlaVanillaAdapter,
    OpenVlaVanillaSettings,
)

from ovlab_remote_policy.service import PolicyService


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--attention-implementation", choices=("flash_attention_2",), default="flash_attention_2")
    parser.add_argument("--quantization", choices=("none", "4bit"), default="none")
    return parser.parse_args()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _identity_provider(capabilities) -> Mapping[str, object]:
    metadata = capabilities.metadata
    checkpoint = dict(metadata["checkpoint_identity"])
    runtime = dict(metadata.get("runtime", {}))
    method = dict(metadata["method_descriptor"])
    if "total_parameter_count" in runtime:
        method["total_runtime_parameter_count"] = runtime["total_parameter_count"]
    if "load_counts" in runtime:
        method["load_counts"] = dict(runtime["load_counts"])
    if "inference_parameter_trainability" in runtime:
        method["runtime_parameter_trainability"] = runtime[
            "inference_parameter_trainability"
        ]
    return {
        "model_identity": checkpoint,
        "normalization_identity": {
            "unnorm_key": checkpoint["unnorm_key"],
            "action_statistics_identity": checkpoint["action_statistics_identity"],
        },
        "prompt_template_identity": metadata["prompt_template"],
        "action_codec_identity": {
            "identifier": metadata["action_codec"],
            "conversion_owner": "OpenVlaVanillaAdapter",
            "application_count": 1,
            "output_gripper_convention": capabilities.output_action_spec.gripper_convention.value,
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "flash_attn": _version("flash-attn"),
            "bitsandbytes": _version("bitsandbytes"),
            "ovlab_remote_policy": _version("ovlab-remote-policy"),
            "policy_component": f"{capabilities.component_name}@{capabilities.component_version}",
            "protocol_component": "ovlab-remote-policy@0.1.0",
        },
        "method_descriptor": method,
    }


def main() -> int:
    args = _arguments()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("OpenVLA policy service requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    if args.checkpoint != "openvla/openvla-7b":
        raise RuntimeError(
            "the Vanilla service is reserved for openvla/openvla-7b; use the merged-LoRA "
            "service for openvla/openvla-7b-finetuned-libero-10"
        )
    quantization = ModelQuantization(args.quantization)
    descriptor = replace(
        vanilla_base_method_descriptor(),
        declared_base_revision=args.revision,
        quantization=quantization.value,
    )
    settings = OpenVlaVanillaSettings(
        model=OpenVlaModelSource(args.checkpoint, revision=args.revision),
        processor=OpenVlaModelSource(args.checkpoint, revision=args.revision),
        unnorm_key=args.unnorm_key,
        device=args.device,
        model_dtype=ModelDType.BFLOAT16,
        quantization=quantization,
        attention_implementation=args.attention_implementation,
        local_files_only=True,
        trust_remote_code=True,
        deterministic_inference=True,
        synchronization=InferenceSynchronization.IF_CUDA,
        record_raw_output=False,
        method_descriptor=descriptor,
        metadata={"reference": "unadapted OpenVLA base model"},
    )
    service = PolicyService(
        args.socket,
        OpenVlaVanillaAdapter(settings),
        identity_provider=_identity_provider,
    )
    service.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
