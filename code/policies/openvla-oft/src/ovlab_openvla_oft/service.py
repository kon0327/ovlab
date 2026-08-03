"""Offline AF_UNIX service for the official OpenVLA-OFT LIBERO-10 resource."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
from pathlib import Path

import numpy as np

from ovlab_benchctl import ConfigResolver
from ovlab_openvla_common import ModelQuantization, OpenVlaModelSource
from ovlab_remote_policy.service import PolicyService

from .adapter import OpenVlaOftAdapter
from .artifact import OpenVlaOftArtifact
from .runtime import OPENVLA_OFT_GIT_COMMIT
from .settings import OpenVlaOftSettings


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _identity(capabilities):
    metadata = capabilities.metadata
    method = dict(metadata["method_descriptor"])
    runtime = dict(metadata["runtime"])
    return {
        "model_identity": runtime["verified_artifact"],
        "normalization_identity": {
            "unnorm_key": "libero_10_no_noops", "type": "bounds_q99",
            "statistics_identity": runtime["action_statistics_identity"],
        },
        "prompt_template_identity": metadata["prompt_template"],
        "action_codec_identity": {
            "identifier": metadata["action_codec"], "conversion_owner": metadata["action_codec_owner"],
            "application_count_per_action": 1, "output_gripper_convention": "closed_positive",
        },
        "runtime_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "torch": _version("torch"),
            "transformers": _version("transformers"), "flash_attn": _version("flash-attn"),
            "peft": _version("peft"), "bitsandbytes": _version("bitsandbytes"),
            "openvla_oft_git_commit": OPENVLA_OFT_GIT_COMMIT,
        },
        "method_descriptor": method,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--resource-id", default="openvla-oft-7b-finetuned-libero-10")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quantization", choices=("none", "8bit", "4bit"), default="none")
    args = parser.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("OpenVLA-OFT service requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    registry_path = Path(args.registry).resolve()
    config_root = registry_path.parent.parent
    registry = ConfigResolver(config_root, repository_root=config_root.parent).load_component(
        str(registry_path.relative_to(config_root)), "resource_registry"
    )
    entry = registry["checkpoints"][args.resource_id]
    artifact = OpenVlaOftArtifact.from_registry_entry(args.resource_id, entry)
    source = OpenVlaModelSource(entry["repo_id"], entry["revision"], entry["expected_sha256"])
    settings = OpenVlaOftSettings(
        source, artifact, device=args.device,
        quantization=ModelQuantization(args.quantization),
    )
    PolicyService(args.socket, OpenVlaOftAdapter(settings), identity_provider=_identity).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
