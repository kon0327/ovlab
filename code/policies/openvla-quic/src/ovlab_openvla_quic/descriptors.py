"""Strict scientific identities for QuIC-PEFT and the proposed QuIC-WC extension."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re

from ovlab_core.contracts import Metadata, normalize_metadata

from .errors import QuICDescriptorError, QuICImplementationUnavailableError

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
EXTERNAL_QUIC_COMMIT = "deab81fbe4035c3de2c2da3d63db966fe3361f82"


class QuICVariant(str, Enum):
    PEFT = "quic-peft"
    WC = "quic-wc"

    @property
    def mode(self) -> str:
        return "peft" if self is QuICVariant.PEFT else "wc"

    @property
    def next_gate(self) -> str:
        return "I" if self is QuICVariant.PEFT else "J"


class QuICImplementationStatus(str, Enum):
    SKELETON = "skeleton"
    IMPLEMENTED = "implemented"


class QuICProfileId(str, Enum):
    QP0 = "QP0"
    QP1 = "QP1"
    QP2 = "QP2"
    QP3 = "QP3"
    QP4 = "QP4"


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _hash(value: object) -> str:
    payload = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _optional_positive(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise QuICDescriptorError(f"{name} must be a positive integer or unavailable")


@dataclass(frozen=True, slots=True)
class QuICProviderSpec:
    package: str = "openvla_quic.ovlab_provider"
    api_name: str = "ovlab-quic-provider"
    api_version: str = "1.0.0"
    source_repository: str = "external/openvla-quic"
    source_commit: str = EXTERNAL_QUIC_COMMIT

    def __post_init__(self) -> None:
        for name in ("package", "api_name", "api_version", "source_repository", "source_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise QuICDescriptorError(f"provider {name} must be a non-empty string")
        if self.source_repository != "external/openvla-quic":
            raise QuICDescriptorError("external QuIC source must be external/openvla-quic")
        if _GIT_REVISION.fullmatch(self.source_commit) is None:
            raise QuICDescriptorError("external QuIC source commit must be a full Git revision")

    def as_metadata(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class QuICProfileDefinition:
    profile_id: QuICProfileId
    definition_version: str | None = None
    definition_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, QuICProfileId):
            raise QuICDescriptorError("profile_id must be QP0 through QP4")
        supplied = self.definition_version is not None or self.definition_hash is not None
        if supplied and (self.definition_version is None or self.definition_hash is None):
            raise QuICDescriptorError("profile version and hash must be supplied together")
        if self.definition_version is not None and not self.definition_version.strip():
            raise QuICDescriptorError("profile definition version must not be empty")
        if self.definition_hash is not None and _DIGEST.fullmatch(self.definition_hash) is None:
            raise QuICDescriptorError("profile definition hash must be SHA-256")
        if self.profile_id is QuICProfileId.QP0 and supplied:
            raise QuICDescriptorError("QP0 has no active QuIC profile definition")

    @property
    def active(self) -> bool:
        return self.profile_id is not QuICProfileId.QP0

    @property
    def resolved(self) -> bool:
        return self.profile_id is QuICProfileId.QP0 or self.definition_hash is not None

    def require_runtime_ready(self, variant: QuICVariant) -> None:
        if self.active and not self.resolved:
            raise QuICImplementationUnavailableError(
                variant.value,
                "openvla_quic.ovlab_provider",
                f"{self.profile_id.value} profile definition unresolved",
                variant.next_gate,
            )

    def as_metadata(self) -> dict[str, object]:
        return {
            "id": self.profile_id.value,
            "active_transformation": self.active,
            "definition_availability": (
                "not_applicable" if self.profile_id is QuICProfileId.QP0
                else "available" if self.resolved else "unresolved"
            ),
            "definition_version": self.definition_version,
            "definition_hash": self.definition_hash,
        }


@dataclass(frozen=True, slots=True)
class QuICPlacementEntry:
    component_family: str
    selector: str
    layer_indices: tuple[int, ...]
    tensor_role: str
    original_shape: tuple[int, ...] | None
    protected: bool
    rationale: str

    def __post_init__(self) -> None:
        for name in ("component_family", "selector", "tensor_role", "rationale"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise QuICDescriptorError(f"placement {name} must be non-empty")
        layers = tuple(self.layer_indices)
        if any(type(item) is not int or item < 0 for item in layers) or len(layers) != len(set(layers)):
            raise QuICDescriptorError("placement layer indices must be unique non-negative integers")
        shape = None if self.original_shape is None else tuple(self.original_shape)
        if shape is not None and (not shape or any(type(item) is not int or item <= 0 for item in shape)):
            raise QuICDescriptorError("placement original_shape must contain positive integers")
        if type(self.protected) is not bool:
            raise QuICDescriptorError("placement protected flag must be boolean")
        object.__setattr__(self, "layer_indices", layers)
        object.__setattr__(self, "original_shape", shape)

    def as_metadata(self) -> dict[str, object]:
        return {
            "component_family": self.component_family,
            "selector": self.selector,
            "layer_indices": list(self.layer_indices),
            "tensor_role": self.tensor_role,
            "original_shape": None if self.original_shape is None else list(self.original_shape),
            "protected": self.protected,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class QuICPlacementManifest:
    availability: str
    version: str | None = None
    entries: tuple[QuICPlacementEntry, ...] = ()
    declared_hash: str | None = None

    def __post_init__(self) -> None:
        if self.availability not in {"unresolved", "available"}:
            raise QuICDescriptorError("placement availability must be unresolved or available")
        entries = tuple(self.entries)
        if self.availability == "unresolved":
            if self.version is not None or entries or self.declared_hash is not None:
                raise QuICDescriptorError("unresolved placement must not fabricate version, entries, or hash")
        else:
            if not isinstance(self.version, str) or not self.version.strip() or not entries:
                raise QuICDescriptorError("available placement requires a version and entries")
            calculated = _hash({"version": self.version, "entries": [item.as_metadata() for item in entries]})
            if self.declared_hash is not None and self.declared_hash != calculated:
                raise QuICDescriptorError("placement manifest hash does not match its contents")
            object.__setattr__(self, "declared_hash", calculated)
        object.__setattr__(self, "entries", entries)

    @property
    def manifest_hash(self) -> str | None:
        return self.declared_hash

    def as_metadata(self) -> dict[str, object]:
        return {
            "availability": self.availability,
            "version": self.version,
            "hash": self.manifest_hash,
            "entries": [entry.as_metadata() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class QuICPEFTAccounting:
    base_parameters: int | None = None
    adapter_parameters: int | None = None
    trainable_parameters: int | None = None
    runtime_total_parameters: int | None = None
    base_bytes: int | None = None
    adapter_bytes: int | None = None
    runtime_artifact_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _optional_positive(getattr(self, name), name)
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(value is not None for value in values) and any(value is None for value in values):
            raise QuICDescriptorError("PEFT accounting must be complete or explicitly unavailable")
        if self.base_parameters is not None:
            if self.runtime_total_parameters != self.base_parameters + self.adapter_parameters:
                raise QuICDescriptorError("P_runtime_total must equal P_base + P_QuIC_adapter")
            if self.trainable_parameters > self.adapter_parameters:
                raise QuICDescriptorError("P_trainable cannot exceed P_QuIC_adapter")
            if self.runtime_artifact_bytes != self.base_bytes + self.adapter_bytes:
                raise QuICDescriptorError("runtime artifact bytes must separate and sum base plus adapter")

    @property
    def available(self) -> bool:
        return self.base_parameters is not None

    def as_metadata(self) -> dict[str, object]:
        return {"availability": "available" if self.available else "unavailable", **{
            name: getattr(self, name) for name in self.__dataclass_fields__
        }}


@dataclass(frozen=True, slots=True)
class QuICWCAccounting:
    dense_replaced_parameters: int | None = None
    compact_factor_parameters: int | None = None
    uncompressed_remainder_parameters: int | None = None
    deployed_total_parameters: int | None = None
    dense_replaced_bytes: int | None = None
    compact_factor_bytes: int | None = None
    uncompressed_remainder_bytes: int | None = None
    deployed_total_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _optional_positive(getattr(self, name), name)
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(value is not None for value in values) and any(value is None for value in values):
            raise QuICDescriptorError("WC accounting must be complete or explicitly unavailable")
        if self.dense_replaced_parameters is not None:
            if self.deployed_total_parameters != (
                self.compact_factor_parameters + self.uncompressed_remainder_parameters
            ):
                raise QuICDescriptorError("P_deployed_total must equal factors plus uncompressed remainder")
            if self.compact_factor_parameters >= self.dense_replaced_parameters:
                raise QuICDescriptorError("compact factors must be smaller than replaced dense parameters")
            if self.deployed_total_bytes != self.compact_factor_bytes + self.uncompressed_remainder_bytes:
                raise QuICDescriptorError("deployed bytes must equal factors plus uncompressed remainder")

    @property
    def available(self) -> bool:
        return self.dense_replaced_parameters is not None

    def as_metadata(self) -> dict[str, object]:
        return {"availability": "available" if self.available else "unavailable", **{
            name: getattr(self, name) for name in self.__dataclass_fields__
        }}


@dataclass(frozen=True, slots=True)
class QuICMethodDescriptor:
    variant: QuICVariant
    display_name: str
    implementation_status: QuICImplementationStatus
    profile: QuICProfileDefinition
    placement: QuICPlacementManifest
    provider: QuICProviderSpec = field(default_factory=QuICProviderSpec)
    runtime_validated: bool = False
    compression_verified: bool = False
    base_model_identity: Metadata = field(default_factory=lambda: {
        "availability": "unavailable", "reason": "not selected in Gate F"
    })
    artifact_identity: Metadata = field(default_factory=lambda: {
        "availability": "unavailable", "reason": "not created in Gate F"
    })
    provenance_identity: Metadata = field(default_factory=lambda: {
        "availability": "unavailable", "reason": "training or conversion has not occurred"
    })
    deployment_state: Metadata = field(default_factory=lambda: {
        "availability": "unavailable", "reason": "no runtime artifact exists in Gate F"
    })
    capability_identity: Metadata = field(default_factory=lambda: {
        "availability": "unavailable", "reason": "external provider not implemented"
    })
    normalization_identity: Metadata = field(default_factory=lambda: {
        "availability": "unavailable", "reason": "concrete policy not selected"
    })
    parameterization: Metadata = field(default_factory=lambda: {
        "availability": "unavailable", "reason": "QP profile definition unresolved"
    })
    accounting: QuICPEFTAccounting | QuICWCAccounting = field(default_factory=QuICPEFTAccounting)
    unavailable_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.variant, QuICVariant) or not isinstance(
            self.implementation_status, QuICImplementationStatus
        ):
            raise QuICDescriptorError("variant and implementation status must be typed enums")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise QuICDescriptorError("display name must be non-empty")
        if type(self.runtime_validated) is not bool or type(self.compression_verified) is not bool:
            raise QuICDescriptorError("runtime/compression validation flags must be boolean")
        if self.implementation_status is QuICImplementationStatus.SKELETON:
            if self.runtime_validated or self.compression_verified:
                raise QuICDescriptorError("skeleton methods cannot claim runtime or compression validation")
        unavailable = tuple(self.unavailable_fields)
        if not unavailable or len(unavailable) != len(set(unavailable)):
            raise QuICDescriptorError("unavailable_fields must explicitly list unique unavailable evidence")
        expected_accounting = QuICPEFTAccounting if self.variant is QuICVariant.PEFT else QuICWCAccounting
        if not isinstance(self.accounting, expected_accounting):
            raise QuICDescriptorError("accounting schema must match the QuIC variant")
        if self.compression_verified and not self.accounting.available:
            raise QuICDescriptorError("compression cannot be verified without complete accounting")
        if self.variant is QuICVariant.PEFT and self.compression_verified:
            raise QuICDescriptorError("QuIC-PEFT must not claim complete-model weight compression")
        for name in (
            "base_model_identity", "artifact_identity", "provenance_identity",
            "deployment_state", "capability_identity", "normalization_identity", "parameterization",
        ):
            value = dict(getattr(self, name))
            if value.get("availability") not in {"available", "unavailable"}:
                raise QuICDescriptorError(f"{name} must declare availability")
            object.__setattr__(self, name, normalize_metadata(value, type(self).__name__, name))
        if self.artifact_identity["availability"] == "available":
            expected_form = (
                "multiplicative_adapter"
                if self.variant is QuICVariant.PEFT
                else "compact_weight_factors"
            )
            if self.artifact_identity.get("form") != expected_form:
                raise QuICDescriptorError(
                    f"{self.variant.value} artifact form must be {expected_form}"
                )
        if self.deployment_state["availability"] == "available":
            state = dict(self.deployment_state)
            if self.variant is QuICVariant.PEFT:
                required_state = {
                    "availability", "active_adapter", "merge_state",
                    "requires_base_model", "deployment_replaces_base_weights",
                }
                if set(state) != required_state:
                    raise QuICDescriptorError(
                        "QuIC-PEFT deployment state fields are incomplete or ambiguous"
                    )
                if (
                    type(state["active_adapter"]) is not bool
                    or state["merge_state"] not in {"unmerged", "merged"}
                    or state["requires_base_model"] is not True
                    or state["deployment_replaces_base_weights"] is not False
                ):
                    raise QuICDescriptorError("QuIC-PEFT deployment state is inconsistent")
            else:
                required_state = {
                    "availability", "replacement_state",
                    "requires_replaced_dense_weights_at_deployment",
                    "deployment_replaces_selected_weights",
                    "dense_runtime_reconstruction_allowed",
                }
                if set(state) != required_state:
                    raise QuICDescriptorError(
                        "QuIC-WC deployment state fields are incomplete or ambiguous"
                    )
                if (
                    state["replacement_state"] != "replaced"
                    or state["requires_replaced_dense_weights_at_deployment"] is not False
                    or state["deployment_replaces_selected_weights"] is not True
                    or state["dense_runtime_reconstruction_allowed"] is not False
                ):
                    raise QuICDescriptorError("QuIC-WC deployment state is inconsistent")
        object.__setattr__(self, "unavailable_fields", unavailable)

    @property
    def family(self) -> str:
        return "openvla_quic"

    @property
    def published_method_relation(self) -> str:
        return "direct" if self.variant is QuICVariant.PEFT else "proposed_extension"

    def require_runtime_ready(self) -> None:
        if self.implementation_status is not QuICImplementationStatus.IMPLEMENTED:
            raise QuICImplementationUnavailableError(
                self.variant.value, self.provider.package,
                self.implementation_status.value, self.variant.next_gate,
                self.provider.source_repository,
            )
        self.profile.require_runtime_ready(self.variant)
        required = (
            self.base_model_identity, self.artifact_identity, self.provenance_identity,
            self.deployment_state, self.capability_identity, self.normalization_identity,
            self.parameterization,
        )
        if any(value["availability"] != "available" for value in required):
            raise QuICImplementationUnavailableError(
                self.variant.value, self.provider.package, "incomplete resource identity",
                self.variant.next_gate, self.provider.source_repository,
            )

    def canonical_dict(self) -> dict[str, object]:
        common = {
            "family": self.family,
            "variant_id": self.variant.value,
            "mode": self.variant.mode,
            "display_name": self.display_name,
            "implementation_status": self.implementation_status.value,
            "runtime_validated": self.runtime_validated,
            "compression_verified": self.compression_verified,
            "published_method_relation": self.published_method_relation,
            "provider": self.provider.as_metadata(),
            "profile": self.profile.as_metadata(),
            "placement_manifest": self.placement.as_metadata(),
            "base_model_identity": self.base_model_identity,
            "artifact_identity": self.artifact_identity,
            "artifact_form": self.artifact_identity.get("form", "unavailable"),
            "provenance_identity": self.provenance_identity,
            "deployment_state": self.deployment_state,
            "capability_identity": self.capability_identity,
            "normalization_identity": self.normalization_identity,
            "parameterization": self.parameterization,
            "accounting": self.accounting.as_metadata(),
            "quantization": "none",
            "unavailable_fields": list(self.unavailable_fields),
            "orthogonal_fine_tuning_relation": (
                "QuIC C1 recovers Orthogonal Fine-Tuning (Qiu et al.); this is unrelated to "
                "OpenVLA-OFT Optimized Fine-Tuning"
            ),
        }
        if self.variant is QuICVariant.PEFT:
            common.update({
                "requires_base_model": True,
                "deployment_replaces_base_weights": False,
                "adaptation_type": "multiplicative_adapter",
                "weight_compression": False,
            })
        else:
            common.update({
                "requires_dense_source_for_conversion": "configurable",
                "requires_replaced_dense_weights_at_deployment": False,
                "deployment_replaces_selected_weights": True,
                "weight_compression": True,
                "dense_runtime_reconstruction_allowed": False,
            })
        return _plain(common)

    @property
    def scientific_hash(self) -> str:
        return _hash(self.canonical_dict())

    def execution_hash(
        self,
        *,
        backend: str,
        precision: str,
        kernel_identity: str | None,
        device_identity: str | None = None,
    ) -> str:
        for name, value in (("backend", backend), ("precision", precision)):
            if not isinstance(value, str) or not value.strip():
                raise QuICDescriptorError(f"execution {name} must be non-empty")
        for name, value in (("kernel_identity", kernel_identity), ("device_identity", device_identity)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise QuICDescriptorError(f"execution {name} must be a non-empty string or unavailable")
        return _hash({
            "scientific_hash": self.scientific_hash,
            "backend": backend,
            "device_identity": device_identity,
            "precision": precision,
            "kernel_identity": kernel_identity,
        })

    def as_metadata(self) -> dict[str, object]:
        return {**self.canonical_dict(), "scientific_identity_hash": self.scientific_hash}


def skeleton_descriptor(variant: QuICVariant, profile: QuICProfileId = QuICProfileId.QP0) -> QuICMethodDescriptor:
    return QuICMethodDescriptor(
        variant=variant,
        display_name="QuIC-PEFT" if variant is QuICVariant.PEFT else "QuIC-WC",
        implementation_status=QuICImplementationStatus.SKELETON,
        profile=QuICProfileDefinition(profile),
        placement=QuICPlacementManifest("unresolved"),
        accounting=QuICPEFTAccounting() if variant is QuICVariant.PEFT else QuICWCAccounting(),
        unavailable_fields=(
            "base_model_revision", "artifact_revision", "artifact_hash", "provenance",
            "deployment_state", "profile_definition",
            "placement_manifest", "capabilities", "normalization", "parameter_counts",
            "artifact_bytes", "runtime_kernel", "measured_latency", "measured_compression",
        ),
    )
