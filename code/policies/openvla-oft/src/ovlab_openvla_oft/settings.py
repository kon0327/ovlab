"""Strict, portable OpenVLA-OFT runtime settings."""

from dataclasses import dataclass, field

from ovlab_core.contracts import ActionSpec, Metadata, normalize_metadata
from ovlab_openvla_common import (
    ModelQuantization, OpenVlaModelSource, action_specs_match, libero_target_action_spec,
)

from .artifact import OpenVlaOftArtifact


@dataclass(frozen=True, slots=True)
class OpenVlaOftSettings:
    model: OpenVlaModelSource
    artifact: OpenVlaOftArtifact
    unnorm_key: str = "libero_10_no_noops"
    primary_camera_name: str = "camera.primary.rgb"
    wrist_camera_name: str = "camera.wrist.rgb"
    proprioception_name: str = "robot.proprioception"
    input_image_shape: tuple[int, int, int] = (256, 256, 3)
    proprioception_dimension: int = 8
    action_chunk_size: int = 8
    device: str = "cuda:0"
    quantization: ModelQuantization = ModelQuantization.NONE
    attention_implementation: str | None = None
    target_action_spec: ActionSpec = field(default_factory=libero_target_action_spec)
    record_raw_output: bool = False
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, OpenVlaModelSource) or not isinstance(self.artifact, OpenVlaOftArtifact):
            raise TypeError("model and artifact must use typed immutable identities")
        if self.model.source != self.artifact.repository or self.model.revision != self.artifact.revision:
            raise ValueError("OFT model source differs from the registered artifact")
        if self.model.expected_checksum != self.artifact.aggregate_sha256:
            raise ValueError("OFT model checksum differs from the registered artifact")
        if self.unnorm_key != "libero_10_no_noops":
            raise ValueError("official LIBERO-10 OFT requires unnorm_key=libero_10_no_noops")
        if self.proprioception_dimension != 8 or self.action_chunk_size != 8:
            raise ValueError("official LIBERO OFT requires proprio dimension 8 and action horizon 8")
        if tuple(self.input_image_shape) != (256, 256, 3):
            raise ValueError("Gate E canonical RGB inputs must be HWC uint8 [256,256,3]")
        if not action_specs_match(self.target_action_spec, libero_target_action_spec()):
            raise ValueError("OFT output ActionSpec differs from canonical LIBERO OSC_POSE")
        if not isinstance(self.quantization, ModelQuantization):
            raise TypeError("quantization must be ModelQuantization")
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, type(self).__name__))
