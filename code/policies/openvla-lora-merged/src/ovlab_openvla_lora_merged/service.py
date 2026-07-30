"""Offline service for the registered merged OpenVLA-LoRA reference."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from ovlab_benchctl import ConfigResolver
from ovlab_openvla_common import OpenVlaModelSource, OpenVlaRuntimeArtifact
from ovlab_openvla_lora_merged import OpenVlaMergedLoraAdapter, method_descriptor_from_registry
from ovlab_openvla_vanilla import (
    InferenceSynchronization,
    ModelDType,
    ModelQuantization,
    OpenVlaVanillaSettings,
)
from ovlab_remote_policy.service import PolicyService


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--resource-id", required=True)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument(
        "--attention-implementation",
        choices=("flash_attention_2",),
        default="flash_attention_2",
    )
    parser.add_argument("--quantization", choices=("none", "4bit"), default="none")
    return parser.parse_args()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _identity_provider(capabilities):
    metadata = capabilities.metadata
    checkpoint = dict(metadata["checkpoint_identity"])
    runtime = dict(metadata["runtime"])
    return {
        "model_identity": checkpoint,
        "normalization_identity": {
            "unnorm_key": checkpoint["unnorm_key"],
            "action_statistics_identity": checkpoint["action_statistics_identity"],
        },
        "prompt_template_identity": metadata["prompt_template"],
        "action_codec_identity": {
            "identifier": metadata["action_codec"],
            "conversion_owner": metadata["action_codec_owner"],
            "application_count": 1,
            "output_gripper_convention": capabilities.output_action_spec.gripper_convention.value,
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "flash_attn": _version("flash-attn"),
            "peft": _version("peft"),
            "bitsandbytes": _version("bitsandbytes"),
            "policy_component": f"{capabilities.component_name}@{capabilities.component_version}",
            "protocol_component": "ovlab-remote-policy@0.1.0",
            "openvla_git_commit": checkpoint["openvla_git_commit"],
        },
        "method_descriptor": {
            **dict(metadata["method_descriptor"]),
            "total_runtime_parameter_count": runtime["total_parameter_count"],
            "load_counts": dict(runtime["load_counts"]),
            "runtime_parameter_trainability": runtime["inference_parameter_trainability"],
        },
    }


def main() -> int:
    args = _arguments()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("merged LoRA service requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    registry_path = Path(args.registry).resolve()
    config_root = registry_path.parent.parent
    registry = ConfigResolver(config_root, repository_root=config_root.parent).load_component(
        str(registry_path.relative_to(config_root)), "resource_registry"
    )
    try:
        entry = registry["checkpoints"][args.resource_id]
    except KeyError as exc:
        raise RuntimeError(f"unknown merged LoRA resource: {args.resource_id}") from exc
    artifact = OpenVlaRuntimeArtifact.from_registry_entry(args.resource_id, entry)
    quantization = ModelQuantization(args.quantization)
    method = replace(
        method_descriptor_from_registry(entry), quantization=quantization.value
    )
    source = OpenVlaModelSource(entry["repo_id"], entry["revision"], entry["expected_sha256"])
    settings = OpenVlaVanillaSettings(
        model=source,
        processor=source,
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
        method_descriptor=method,
        runtime_artifact=artifact,
        metadata={"reference": "official merged OpenVLA-LoRA LIBERO-10"},
    )
    PolicyService(
        args.socket,
        OpenVlaMergedLoraAdapter(settings),
        identity_provider=_identity_provider,
    ).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
