"""Merged LoRA identity over the shared full-weight OpenVLA inference path."""

from ovlab_openvla_common import OpenVlaMethodFamily
from ovlab_openvla_vanilla import OpenVlaVanillaAdapter


class OpenVlaMergedLoraAdapter(OpenVlaVanillaAdapter):
    """Run merged full weights while preserving historical LoRA provenance."""

    component_name = "ovlab-openvla-lora-merged"
    policy_family = "openvla-lora-merged"
    required_method_family = OpenVlaMethodFamily.LORA

    def _validate_method(self) -> None:
        super()._validate_method()
        artifact = self.settings.runtime_artifact
        if artifact is None:
            raise ValueError("merged LoRA adapter requires an immutable runtime artifact manifest")
        descriptor = self.settings.method_descriptor
        if artifact.artifact_form != descriptor.artifact_form.value:
            raise ValueError("runtime artifact form differs from the LoRA method descriptor")
        if artifact.merge_status != descriptor.merge_status.value:
            raise ValueError("runtime artifact merge status differs from the LoRA method descriptor")
