"""Descriptor-aware service factory that rejects skeletons before socket readiness."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from ovlab_core.contracts import ColorSpace, ImageEncoding, ImageObservationSpec
from ovlab_remote_policy.protocol import action_spec_to_wire, image_spec_to_wire

from .adapter import OpenVLAQuICPEFTAdapter, OpenVLAQuICWCAdapter
from .config import OpenVlaQuICSettings
from .descriptors import (
    QuICImplementationStatus, QuICMethodDescriptor, QuICPlacementEntry,
    QuICPlacementManifest, QuICProfileDefinition, QuICProfileId, QuICVariant,
    QuICProviderSpec,
)
from .errors import QuICDescriptorError


def create_runtime_adapter(descriptor: QuICMethodDescriptor):
    """Future service entry point; Gate F always fails before constructing a service."""
    descriptor.require_runtime_ready()
    cls = (
        OpenVLAQuICPEFTAdapter
        if descriptor.variant is QuICVariant.PEFT
        else OpenVLAQuICWCAdapter
    )
    return cls(descriptor)


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor_from_settings(settings: OpenVlaQuICSettings):
    """Bind an OVLAB adapter artifact to explicit resources and policy interfaces."""
    if not isinstance(settings, OpenVlaQuICSettings):
        raise TypeError("settings must be OpenVlaQuICSettings")
    directory = Path(settings.adapter_path)
    payload_path = directory / "adapter.json"
    if _sha_file(payload_path) != settings.adapter_manifest_sha256:
        raise QuICDescriptorError("QuIC adapter manifest differs from configured SHA256")
    payload = json.loads(payload_path.read_text())
    if payload.get("format_version") != 1 or not payload.get("manifest"):
        raise QuICDescriptorError("unsupported or empty QuIC adapter artifact")
    identity = payload["host_identity"]
    if identity.get("family") != "openvla_token":
        raise QuICDescriptorError("QuIC provider only supports dense token OpenVLA")
    if identity.get("revision") != settings.model.revision:
        raise QuICDescriptorError("configured checkpoint revision differs from adapter host")
    metadata = payload.get("metadata", {})
    normalization = metadata.get("normalization_contract")
    if not isinstance(normalization, dict) or normalization.get("unnorm_key") != settings.unnorm_key:
        raise QuICDescriptorError("adapter normalization contract differs from policy settings")
    normalization_mode = normalization.get("mode", "host")
    if normalization_mode not in {"host", "train-q99"}:
        raise QuICDescriptorError("unsupported adapter normalization mode")
    if normalization_mode == "train-q99":
        if set(normalization) != {"unnorm_key", "action_statistics", "mode", "sha256"}:
            raise QuICDescriptorError("fitted normalization contract fields are incomplete or ambiguous")
        contract = {key: value for key, value in normalization.items() if key != "sha256"}
        calculated = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if normalization.get("sha256") != calculated or identity.get("normalization_contract_sha256") != calculated:
            raise QuICDescriptorError("fitted normalization contract SHA256 or host identity mismatch")
    elif "normalization_contract_sha256" in identity:
        raise QuICDescriptorError("fitted host identity requires train-q99 normalization contract")
    preprocessing = metadata.get("preprocessing", {"prompt_style": "modern", "center_crop": False})
    if preprocessing != {"prompt_style": settings.prompt_style, "center_crop": settings.center_crop}:
        raise QuICDescriptorError("adapter preprocessing contract differs from policy settings")
    tensor_sha = _sha_file(directory / "adapter.safetensors")
    if tensor_sha != payload.get("tensor_sha256") or tensor_sha != settings.adapter_tensor_sha256:
        raise QuICDescriptorError("QuIC adapter tensor SHA256 mismatch")
    image = ImageObservationSpec(
        name=settings.canonical_camera_name, shapes=(settings.input_image_shape,),
        dtype="uint8", encodings=(ImageEncoding.RAW,), color_spaces=(ColorSpace.RGB,),
    )
    declaration = {
        "images": [image_spec_to_wire(image)], "proprioception": [],
        "action_spec": action_spec_to_wire(settings.target_action_spec),
        "minimum_horizon": 1, "maximum_horizon": 1,
        "dynamic_instructions": True, "deterministic_reset": True,
    }
    entries = tuple(QuICPlacementEntry(
        component_family=entry["name"].split(".")[0], selector=entry["name"],
        layer_indices=(), tensor_role="dense_linear_output",
        original_shape=(entry["out_features"], entry["in_features"]), protected=False,
        rationale="explicit adapter artifact target; no automatic target expansion",
    ) for entry in payload["manifest"])
    configs = [entry["config"] for entry in payload["manifest"]]
    config_hash = hashlib.sha256(json.dumps(configs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    profile_name = metadata.get("peft", {}).get("adapter", {}).get("profile", "custom")
    profile = QuICProfileId(profile_name)
    host = settings.host_config()
    # The registry resource name can differ from the source repository name;
    # the artifact's checkpoint identity is verified against bytes at load.
    host["checkpoint_id"] = identity["checkpoint_id"]
    if normalization_mode == "train-q99":
        host["normalization_contract_sha256"] = normalization["sha256"]
        host["action_normalization"] = {settings.unnorm_key: {"action": normalization["action_statistics"]}}
    transform = {"kind": "identity"}
    if settings.action_codec == "openvla-to-libero-v1":
        if settings.target_action_spec.dimension != 7:
            raise QuICDescriptorError("OpenVLA pose codec requires a seven dimensional action interface")
        transform = {"kind": "threshold", "index": 6, "minimum": 0.0, "maximum": 1.0,
                     "threshold": 0.5, "below": 1.0, "equal": 0.0, "above": -1.0}
    return QuICMethodDescriptor(
        variant=QuICVariant.PEFT, display_name="QuIC-PEFT (dense token OpenVLA)",
        provider=QuICProviderSpec(source_commit="533b1a76ea600f50738cb9369ed47192a216e2af"),
        implementation_status=QuICImplementationStatus.IMPLEMENTED,
        profile=QuICProfileDefinition(profile, "quic-peft-vla/0.1.0", config_hash),
        placement=QuICPlacementManifest("available", "quic-adapter-v1", entries),
        source_import_status="present", generic_compound_backend_status="portable_determinant_backend",
        openvla_integration_status="implemented",
        source_identity={"availability": "available", "package": "quic-peft-vla", "version": "0.1.0",
                         "sha256": identity.get("quic_method_sha256"),
                         "integration_source_sha256": identity.get("integration_source_sha256")},
        base_model_identity={"availability": "available", "host_identity": identity, "host": host},
        artifact_identity={"availability": "available", "form": "multiplicative_adapter",
                           "directory": str(directory.resolve()), "manifest_sha256": _sha_file(payload_path),
                           "tensor_sha256": tensor_sha},
        provenance_identity={"availability": "available", "training_metadata": metadata},
        deployment_state={"availability": "available", "active_adapter": True, "merge_state": "unmerged",
                          "requires_base_model": True, "deployment_replaces_base_weights": False},
        capability_identity={"availability": "available", "declaration": declaration,
                             "action_codec": settings.action_codec,
                             "action_transform": transform,
                             "synchronize_inference": settings.synchronize_inference},
        normalization_identity={"availability": "available", "contract": normalization,
                                "preprocessing": preprocessing},
        parameterization={"availability": "available", "manifest": payload["manifest"]},
        unavailable_fields=("measured_latency", "measured_compression", "cuda_validation", "rollout_validation"),
    )


def create_policy_adapter(settings: OpenVlaQuICSettings):
    return create_runtime_adapter(descriptor_from_settings(settings))


def _identity_provider(capabilities):
    """Expose QuIC provenance through the existing OVLAB RPC handshake."""
    metadata = capabilities.metadata
    method = metadata["method_descriptor"]
    contract = method["normalization_identity"]["contract"]
    preprocessing = method["normalization_identity"]["preprocessing"]
    versions = {"python": platform.python_version()}
    for distribution in ("torch", "transformers", "safetensors", "quic-peft-vla", "openvla-quic", "ovlab-openvla-quic"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "source-tree"
    return {
        "model_identity": dict(method["base_model_identity"]["host_identity"]),
        "normalization_identity": {
            "unnorm_key": contract["unnorm_key"],
            "action_statistics_identity": contract.get("sha256", hashlib.sha256(
                json.dumps(dict(contract["action_statistics"]), sort_keys=True).encode()).hexdigest()),
        },
        "prompt_template_identity": f"openvla-{preprocessing['prompt_style']}@1.0.0",
        "action_codec_identity": {
            "identifier": method["capability_identity"]["action_codec"],
            "conversion_owner": "ovlab_openvla_quic.runtime.QuICPETFProvider",
            "application_count": 1,
            "output_gripper_convention": capabilities.output_action_spec.gripper_convention.value,
        },
        "runtime_versions": versions, "method_descriptor": dict(method),
    }
