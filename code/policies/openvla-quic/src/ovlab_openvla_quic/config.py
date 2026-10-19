"""Conversion of validated descriptor-only documents into typed identities."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re

from ovlab_core.contracts import ActionSpec
from ovlab_openvla_common import OpenVlaModelSource

from .descriptors import (
    EXTERNAL_QUIC_COMMIT, QuICProfileId, QuICVariant, skeleton_descriptor,
)
from .errors import QuICDescriptorError


@dataclass(frozen=True, slots=True)
class OpenVlaQuICSettings:
    """Portable OVLAB resource settings; resolving them imports no GPU libraries."""

    model: OpenVlaModelSource
    unnorm_key: str
    adapter_path: str
    target_action_spec: ActionSpec
    adapter_manifest_sha256: str
    adapter_tensor_sha256: str
    canonical_camera_name: str = "camera.primary.rgb"
    input_image_shape: tuple[int, int, int] = (256, 256, 3)
    device: str = "cuda:0"
    model_dtype: str = "bfloat16"
    attention_implementation: str = "flash_attention_2"
    prompt_style: str = "modern"
    center_crop: bool = False
    adaptation_state: str = "none"
    action_codec: str = "openvla-decoded-v1"
    synchronize_inference: bool = True
    record_raw_output: bool = False

    def __post_init__(self):
        if not isinstance(self.model, OpenVlaModelSource):
            raise TypeError("QuIC model must be OpenVlaModelSource")
        if not isinstance(self.adapter_path, str) or not self.adapter_path.strip():
            raise QuICDescriptorError("QuIC requires an explicit adapter artifact path")
        for name in ("adapter_manifest_sha256", "adapter_tensor_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise QuICDescriptorError(f"QuIC {name} must be SHA256")
        if self.model_dtype not in {"float32", "float16", "bfloat16"}:
            raise QuICDescriptorError("QuIC dtype must be float32, float16 or bfloat16")
        if self.prompt_style not in {"modern", "v01"}:
            raise QuICDescriptorError("QuIC prompt_style must be modern or v01")
        if self.adaptation_state not in {"none", "merged"}:
            raise QuICDescriptorError("QuIC requires a dense unadapted or merged host")
        if self.action_codec not in {"openvla-decoded-v1", "openvla-to-libero-v1"}:
            raise QuICDescriptorError("unsupported QuIC action codec")
        if type(self.center_crop) is not bool or type(self.synchronize_inference) is not bool:
            raise QuICDescriptorError("QuIC crop and synchronization flags must be boolean")
        if not self.unnorm_key or not self.canonical_camera_name or not self.device:
            raise QuICDescriptorError("QuIC normalization, image channel and device must be explicit")
        shape = tuple(self.input_image_shape)
        if len(shape) != 3 or shape[-1] != 3 or any(type(n) is not int or n < 1 for n in shape):
            raise QuICDescriptorError("QuIC input_image_shape must be positive HWC RGB")
        object.__setattr__(self, "input_image_shape", shape)

    def host_config(self):
        checkpoint = self.model.local_path
        if checkpoint is None and Path(self.model.source).is_dir():
            checkpoint = self.model.source
        if checkpoint is None:
            raise QuICDescriptorError("QuIC serving requires a resolved local checkpoint resource")
        return {
            "family": "openvla_token", "checkpoint": checkpoint,
            "checkpoint_id": self.model.source, "revision": self.model.revision,
            "adaptation_state": self.adaptation_state, "unnorm_key": self.unnorm_key,
            "device": self.device, "dtype": self.model_dtype,
            "attention": self.attention_implementation, "prompt_style": self.prompt_style,
            "center_crop": self.center_crop,
        }


def descriptor_from_document(document: Mapping[str, object]):
    if document.get("kind") != "quic_policy_descriptor":
        raise QuICDescriptorError("document kind must be quic_policy_descriptor")
    variant = QuICVariant(document["variant"])
    descriptor = skeleton_descriptor(variant, QuICProfileId(document["profile"]["id"]))
    expected = descriptor.canonical_dict()
    checks = {
        "implementation_status": expected["implementation_status"],
        "source_import_status": expected["source_import_status"],
        "generic_compound_backend_status": expected["generic_compound_backend_status"],
        "openvla_integration_status": expected["openvla_integration_status"],
        "published_method_relation": expected["published_method_relation"],
        "weight_compression": expected["weight_compression"],
    }
    for key, value in checks.items():
        if document[key] != value:
            raise QuICDescriptorError(f"configuration misclassifies {key}")
    provider = document["external_provider"]
    if provider["source_commit"] != EXTERNAL_QUIC_COMMIT:
        raise QuICDescriptorError("external provider commit differs from pinned submodule")
    if any(document[name] is not False for name in (
        "runtime_validated", "training_validated", "libero_validated", "compression_verified",
    )):
        raise QuICDescriptorError("skeleton configuration cannot claim validation")
    return descriptor
