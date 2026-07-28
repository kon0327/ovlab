"""Methodological provenance separated from OpenVLA runtime mechanics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

from ovlab_core.contracts import Metadata, normalize_metadata


class OpenVlaMethodFamily(str, Enum):
    VANILLA = "vanilla"
    LORA = "lora"


class OpenVlaArtifactForm(str, Enum):
    FULL_WEIGHTS = "full_weights"
    MERGED_FULL_WEIGHTS = "merged_full_weights"


class OpenVlaMergeStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    MERGED = "merged"


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class OpenVlaMethodDescriptor:
    method_id: str
    method_version: str
    family: OpenVlaMethodFamily
    artifact_form: OpenVlaArtifactForm
    merge_status: OpenVlaMergeStatus
    active_peft_adapter: bool
    runtime_peft_modules: bool
    declared_base_model: str
    declared_base_revision: str | None
    adaptation_suite: str | None
    quantization: str
    adapter_recoverability: str
    lora_configuration: Metadata = field(default_factory=dict)
    training_provenance: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "method_id", "method_version", "declared_base_model", "quantization",
            "adapter_recoverability",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.family, OpenVlaMethodFamily):
            raise TypeError("family must be OpenVlaMethodFamily")
        if not isinstance(self.artifact_form, OpenVlaArtifactForm):
            raise TypeError("artifact_form must be OpenVlaArtifactForm")
        if not isinstance(self.merge_status, OpenVlaMergeStatus):
            raise TypeError("merge_status must be OpenVlaMergeStatus")
        if type(self.active_peft_adapter) is not bool or type(self.runtime_peft_modules) is not bool:
            raise TypeError("active_peft_adapter and runtime_peft_modules must be booleans")
        for name in ("declared_base_revision", "adaptation_suite"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        lora = dict(self.lora_configuration)
        if self.family is OpenVlaMethodFamily.LORA:
            if self.artifact_form is not OpenVlaArtifactForm.MERGED_FULL_WEIGHTS:
                raise ValueError("the supported LoRA reference must use merged_full_weights")
            if self.merge_status is not OpenVlaMergeStatus.MERGED:
                raise ValueError("the supported LoRA reference must report merge_status=merged")
            if self.active_peft_adapter or self.runtime_peft_modules:
                raise ValueError("merged LoRA runtime must not report active PEFT modules")
            if self.quantization != "none":
                raise ValueError("Gate D merged LoRA must not use QLoRA or quantization")
            required = {
                "rank", "alpha", "scaling", "dropout", "bias", "target_policy",
                "modules_to_save", "merge_procedure",
            }
            if set(lora) != required:
                raise ValueError(f"LoRA configuration fields must equal {sorted(required)}")
            if lora["rank"] != 32 or lora["alpha"] != 16 or lora["scaling"] != 0.5:
                raise ValueError("official merged LoRA requires rank=32, alpha=16, scaling=0.5")
            if lora["dropout"] != 0.0 or lora["bias"] != "none":
                raise ValueError("official merged LoRA requires dropout=0.0 and bias=none")
            if lora["target_policy"] != "all-linear" or lora["modules_to_save"] is not None:
                raise ValueError("official merged LoRA requires all-linear and modules_to_save=None")
            if lora["merge_procedure"] != "merge_and_unload()+save_pretrained()":
                raise ValueError("unsupported merged LoRA serialization procedure")
            if self.adaptation_suite != "LIBERO-10":
                raise ValueError("official Gate D LoRA reference must target LIBERO-10")
            if self.adapter_recoverability != "not_recoverable_from_published_artifact":
                raise ValueError("merged LoRA adapter recoverability is misclassified")
        else:
            if lora:
                raise ValueError("Vanilla method must not contain LoRA configuration")
            if self.artifact_form is not OpenVlaArtifactForm.FULL_WEIGHTS:
                raise ValueError("Vanilla method must use full_weights")
            if self.merge_status is not OpenVlaMergeStatus.NOT_APPLICABLE:
                raise ValueError("Vanilla method merge status must be not_applicable")
            if self.active_peft_adapter or self.runtime_peft_modules:
                raise ValueError("Vanilla runtime must not contain PEFT modules")
        object.__setattr__(self, "lora_configuration", normalize_metadata(lora, type(self).__name__))
        object.__setattr__(
            self,
            "training_provenance",
            normalize_metadata(self.training_provenance, type(self).__name__),
        )

    def canonical_dict(self) -> dict[str, object]:
        return _plain({
            "method_id": self.method_id,
            "method_version": self.method_version,
            "family": self.family,
            "artifact_form": self.artifact_form,
            "merge_status": self.merge_status,
            "active_peft_adapter": self.active_peft_adapter,
            "runtime_peft_modules": self.runtime_peft_modules,
            "declared_base_model": self.declared_base_model,
            "declared_base_revision": self.declared_base_revision,
            "adaptation_suite": self.adaptation_suite,
            "quantization": self.quantization,
            "adapter_recoverability": self.adapter_recoverability,
            "lora_configuration": self.lora_configuration,
            "training_provenance": self.training_provenance,
        })

    @property
    def identity_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_metadata(self) -> dict[str, object]:
        result = self.canonical_dict()
        result["identity_hash"] = self.identity_hash
        result["qp_profile"] = None
        return result


def vanilla_base_method_descriptor() -> OpenVlaMethodDescriptor:
    return OpenVlaMethodDescriptor(
        method_id="openvla-vanilla-base",
        method_version="1.0.0",
        family=OpenVlaMethodFamily.VANILLA,
        artifact_form=OpenVlaArtifactForm.FULL_WEIGHTS,
        merge_status=OpenVlaMergeStatus.NOT_APPLICABLE,
        active_peft_adapter=False,
        runtime_peft_modules=False,
        declared_base_model="openvla/openvla-7b",
        declared_base_revision=None,
        adaptation_suite=None,
        quantization="none",
        adapter_recoverability="not_applicable",
    )


def method_descriptor_from_registry(entry: dict[str, object]) -> OpenVlaMethodDescriptor:
    """Construct the strict merged-LoRA descriptor from one validated registry entry."""
    method = entry["method"]
    artifact = entry["artifact"]
    lora = method["lora"]
    return OpenVlaMethodDescriptor(
        method_id=method["id"],
        method_version=method["version"],
        family=OpenVlaMethodFamily(method["family"]),
        artifact_form=OpenVlaArtifactForm(artifact["form"]),
        merge_status=OpenVlaMergeStatus(artifact["merge_status"]),
        active_peft_adapter=artifact["active_peft_adapter"],
        runtime_peft_modules=artifact["runtime_peft_modules"],
        declared_base_model=method["declared_base_model"],
        declared_base_revision=method["declared_base_revision"],
        adaptation_suite=method["adaptation_suite"],
        quantization=method["quantization"],
        adapter_recoverability=artifact["adapter_recoverability"],
        lora_configuration={
            "rank": lora["rank"],
            "alpha": lora["alpha"],
            "scaling": lora["scaling"],
            "dropout": lora["dropout"],
            "bias": lora["bias"],
            "target_policy": lora["target_policy"],
            "modules_to_save": lora["modules_to_save"],
            "merge_procedure": lora["merge_procedure"],
        },
        training_provenance=method["training_provenance"],
    )
