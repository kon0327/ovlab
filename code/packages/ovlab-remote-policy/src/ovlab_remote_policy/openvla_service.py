"""OpenVLA Vanilla policy-service entry point for the isolated local RPC."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
from collections.abc import Mapping

import numpy as np

from ovlab_openvla_common import OpenVlaModelSource
from ovlab_openvla_vanilla import (
    InferenceSynchronization,
    ModelDType,
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
    return parser.parse_args()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _identity_provider(capabilities) -> Mapping[str, object]:
    metadata = capabilities.metadata
    checkpoint = dict(metadata["checkpoint_identity"])
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
            "ovlab_remote_policy": _version("ovlab-remote-policy"),
            "policy_component": f"{capabilities.component_name}@{capabilities.component_version}",
            "protocol_component": "ovlab-remote-policy@0.1.0",
        },
    }


def main() -> int:
    args = _arguments()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("OpenVLA policy service requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    settings = OpenVlaVanillaSettings(
        model=OpenVlaModelSource(args.checkpoint, revision=args.revision),
        processor=OpenVlaModelSource(args.checkpoint, revision=args.revision),
        unnorm_key=args.unnorm_key,
        device=args.device,
        model_dtype=ModelDType.BFLOAT16,
        attention_implementation=args.attention_implementation,
        local_files_only=True,
        trust_remote_code=True,
        deterministic_inference=True,
        synchronization=InferenceSynchronization.IF_CUDA,
        record_raw_output=False,
        metadata={"reference": "full-weight suite-finetuned OpenVLA LIBERO-10"},
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
