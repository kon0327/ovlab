"""Immutable, environment-independent Vanilla policy settings."""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from collections.abc import Mapping

import numpy as np

from ovlab_core.contracts import ActionSpec, Metadata, normalize_metadata
from ovlab_openvla_common import (
    LiberoActionCodecConfig,
    OpenVlaModelSource,
    OpenVlaPromptTemplate,
    action_specs_match,
    libero_target_action_spec,
)


class ModelDType(str, Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class InferenceSynchronization(str, Enum):
    IF_CUDA = "if_cuda"
    ALWAYS = "always"
    NONE = "none"


def _plain_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_metadata(nested) for key, nested in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain_metadata(item) for item in value]
    return value


def _action_spec_dict(spec: ActionSpec) -> dict[str, object]:
    return {
        "dimension": spec.dimension,
        "representation": spec.representation.value,
        "translation_indices": list(spec.translation_indices),
        "rotation_indices": list(spec.rotation_indices),
        "gripper_indices": list(spec.gripper_indices),
        "rotation_representation": spec.rotation_representation.value,
        "gripper_convention": spec.gripper_convention.value,
        "units": list(spec.units),
        "minimum": None if spec.minimum is None else np.asarray(spec.minimum).tolist(),
        "maximum": None if spec.maximum is None else np.asarray(spec.maximum).tolist(),
        "dtype": spec.dtype,
        "control_frequency_hz": spec.control_frequency_hz,
    }


@dataclass(frozen=True, slots=True)
class OpenVlaVanillaSettings:
    model: OpenVlaModelSource
    unnorm_key: str
    processor: OpenVlaModelSource | None = None
    canonical_camera_name: str = "camera.primary.rgb"
    input_image_shape: tuple[int, int, int] = (256, 256, 3)
    device: str = "cuda:0"
    model_dtype: ModelDType = ModelDType.BFLOAT16
    attention_implementation: str | None = "flash_attention_2"
    local_files_only: bool = True
    trust_remote_code: bool = True
    deterministic_inference: bool = True
    prompt_template: OpenVlaPromptTemplate = OpenVlaPromptTemplate.OPENVLA_V1
    target_action_spec: ActionSpec = field(default_factory=libero_target_action_spec)
    action_codec: LiberoActionCodecConfig = field(default_factory=LiberoActionCodecConfig)
    synchronization: InferenceSynchronization = InferenceSynchronization.IF_CUDA
    record_raw_output: bool = False
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, OpenVlaModelSource):
            raise TypeError("model must be an OpenVlaModelSource")
        if self.processor is not None and not isinstance(self.processor, OpenVlaModelSource):
            raise TypeError("processor must be an OpenVlaModelSource or None")
        if not isinstance(self.unnorm_key, str) or not self.unnorm_key.strip():
            raise ValueError("unnorm_key must not be empty")
        if not isinstance(self.canonical_camera_name, str) or not self.canonical_camera_name.strip():
            raise ValueError("canonical_camera_name must not be empty")
        shape = tuple(self.input_image_shape)
        if len(shape) != 3 or shape[2] != 3 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in shape):
            raise ValueError("input_image_shape must be a positive HWC RGB shape")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must not be empty")
        if not isinstance(self.model_dtype, ModelDType):
            raise TypeError("model_dtype must be ModelDType")
        if self.attention_implementation is not None and (
            not isinstance(self.attention_implementation, str) or not self.attention_implementation.strip()
        ):
            raise ValueError("attention_implementation must be a non-empty string or None")
        for name in ("local_files_only", "trust_remote_code", "deterministic_inference", "record_raw_output"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if not isinstance(self.prompt_template, OpenVlaPromptTemplate):
            raise TypeError("prompt_template must be OpenVlaPromptTemplate")
        if not isinstance(self.synchronization, InferenceSynchronization):
            raise TypeError("synchronization must be InferenceSynchronization")
        if not isinstance(self.target_action_spec, ActionSpec):
            raise TypeError("target_action_spec must be an ActionSpec")
        if not isinstance(self.action_codec, LiberoActionCodecConfig):
            raise TypeError("action_codec must be LiberoActionCodecConfig")
        if not action_specs_match(self.target_action_spec, libero_target_action_spec()):
            raise ValueError("target_action_spec is incompatible with the verified LIBERO codec")
        object.__setattr__(self, "input_image_shape", shape)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, type(self).__name__))

    @property
    def processor_source(self) -> OpenVlaModelSource:
        return self.model if self.processor is None else self.processor

    def canonical_dict(self) -> dict[str, object]:
        source = lambda item: {"source": item.source, "revision": item.revision,
                               "expected_checksum": item.expected_checksum}
        return {
            "model": source(self.model), "processor": source(self.processor_source),
            "unnorm_key": self.unnorm_key, "canonical_camera_name": self.canonical_camera_name,
            "input_image_shape": list(self.input_image_shape), "device": self.device,
            "model_dtype": self.model_dtype.value, "attention_implementation": self.attention_implementation,
            "local_files_only": self.local_files_only, "trust_remote_code": self.trust_remote_code,
            "deterministic_inference": self.deterministic_inference, "prompt_template": self.prompt_template.value,
            "target_action_spec": _action_spec_dict(self.target_action_spec),
            "action_codec": {"id": self.action_codec.codec_id, "version": self.action_codec.version,
                             "threshold": self.action_codec.threshold},
            "synchronization": self.synchronization.value, "record_raw_output": self.record_raw_output,
            "metadata": _plain_metadata(self.metadata),
        }

    @property
    def settings_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
