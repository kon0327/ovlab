"""OVLAB protocol integration for the standalone PETF QuIC runtime.

Checkpoint loading is delegated to quic_petf; protocol descriptors, episode
lifecycle and action-interface conversion belong to OVLAB. Heavy imports stay
inside load/predict so descriptor discovery remains dependency-free.
"""

import hashlib
import time
from pathlib import Path

PROVIDER_API_VERSION = "1.0.0"


def create_provider(*, variant: str, method_descriptor=None):
    if variant != "quic-peft":
        raise ValueError(f"PETF provider does not implement variant: {variant!r}")
    return QuICPETFProvider(method_descriptor=method_descriptor)


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class QuICPETFProvider:
    variant_id = "quic-peft"

    def __init__(self, *, method_descriptor=None):
        self.descriptor = method_descriptor
        self.model = self.processor = self.policy = None
        self._counts = {"base_model": 0, "quic_artifact": 0, "processor": 0}
        self._episode_id = None
        self._require_descriptor()

    def api_version(self):
        return PROVIDER_API_VERSION

    def _require_descriptor(self):
        if self.descriptor is None:
            raise ValueError("QuIC adapter-only serving requires an explicit implemented method descriptor")
        if self.descriptor.get("variant_id") != self.variant_id:
            raise ValueError("provider only implements QuIC-PEFT")
        if self.descriptor.get("implementation_status") != "implemented":
            raise ValueError("provider requires an implemented method descriptor")

    def describe(self):
        self._require_descriptor()
        return {key: self.descriptor[key] for key in (
            "family", "variant_id", "scientific_identity_hash", "base_model_identity",
            "artifact_identity", "normalization_identity", "profile", "placement_manifest",
        )}

    def capability_description(self):
        self._require_descriptor()
        return self.descriptor["capability_identity"]["declaration"]

    def load(self, request):
        self._require_descriptor()
        if self.model is not None:
            raise RuntimeError("QuIC provider is already loaded")
        expected = dict(self.descriptor)
        scientific_hash = expected.pop("scientific_identity_hash")
        if request.get("scientific_identity_hash") != scientific_hash or request.get("method_descriptor") != expected:
            raise ValueError("provider load descriptor or scientific identity mismatch")
        artifact = expected["artifact_identity"]
        directory = Path(artifact["directory"])
        if _sha_file(directory / "adapter.json") != artifact["manifest_sha256"]:
            raise ValueError("adapter manifest changed after capability negotiation")
        if _sha_file(directory / "adapter.safetensors") != artifact["tensor_sha256"]:
            raise ValueError("adapter tensors changed after capability negotiation")
        from quic_petf.host import OpenVLAQuICPolicy, checkpoint_identity, load_host
        from quic_petf_vla import load_adapter

        host = expected["base_model_identity"]["host"]
        identity = checkpoint_identity(host)
        if identity != expected["base_model_identity"]["host_identity"]:
            raise ValueError("host checkpoint, source, normalization or method identity mismatch")
        model, processor, runtime = load_host(host)
        try:
            normalization = expected["normalization_identity"]["contract"]
            if normalization.get("unnorm_key") != host["unnorm_key"]:
                raise ValueError("normalization key differs from adapter contract")
            actual_stats = model.norm_stats[host["unnorm_key"]]["action"]
            if actual_stats != normalization.get("action_statistics"):
                raise ValueError("host action normalization differs from adapter contract")
            actual_manifest = load_adapter(model, directory, identity)
            if actual_manifest != expected["parameterization"]["manifest"]:
                raise ValueError("resolved QuIC targets differ from descriptor")
            model.eval()
            self.model, self.processor = model, processor
            self.policy = OpenVLAQuICPolicy(model, processor, host)
            self._counts = {"base_model": 1, "quic_artifact": 1, "processor": 1}
            return {"host_identity": identity, "adapter_sha256": artifact["tensor_sha256"],
                    "runtime": runtime, "artifact_form": "adapter_only", "quantization": "none"}
        except Exception:
            del model, processor
            raise

    def reset_episode(self, request):
        if self.policy is None:
            raise RuntimeError("QuIC provider is not loaded")
        import torch
        seed = request["seed"]
        if type(seed) is not int or seed < 0:
            raise ValueError("episode seed must be a nonnegative integer")
        torch.manual_seed(seed)
        self._episode_id = request["episode_id"]

    def predict(self, request):
        if self.policy is None or self._episode_id is None:
            raise RuntimeError("load and reset_episode are required before predict")
        if request.get("episode_id") != self._episode_id:
            raise ValueError("prediction episode differs from active episode")
        import numpy as np
        import torch
        capability = self.descriptor["capability_identity"]
        declaration = capability["declaration"]
        image_spec = declaration["images"][0]
        images = request["images"]
        if set(images) != {image_spec["name"]} or request["proprioception"]:
            raise ValueError("token OpenVLA requires exactly its declared RGB channel")
        image = np.asarray(images[image_spec["name"]]["data"])
        if image.dtype != np.uint8 or list(image.shape) not in image_spec["shapes"]:
            raise ValueError("RGB image dtype or shape differs from capabilities")
        instruction = request["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be nonempty")
        sync = capability.get("synchronize_inference", True) and str(self.model.device).startswith("cuda")
        if sync:
            torch.cuda.synchronize(self.model.device)
        start = time.perf_counter_ns()
        action = np.asarray(self.policy.predict(image, instruction), dtype=np.float32).copy()
        if sync:
            torch.cuda.synchronize(self.model.device)
        duration = time.perf_counter_ns() - start
        spec = declaration["action_spec"]
        if action.shape != (spec["dimension"],) or not np.isfinite(action).all():
            raise ValueError("model returned an invalid decoded action")
        transform = capability.get("action_transform", {"kind": "identity"})
        if transform["kind"] == "threshold":
            index = transform["index"]
            value = float(action[index])
            if not transform["minimum"] <= value <= transform["maximum"]:
                raise ValueError("decoded action coordinate violates source range")
            action[index] = (transform["below"] if value < transform["threshold"]
                             else transform["above"] if value > transform["threshold"] else transform["equal"])
        elif transform["kind"] != "identity":
            raise ValueError("unsupported explicit action transformation")
        if spec["minimum"] is not None and (np.any(action < spec["minimum"]) or np.any(action > spec["maximum"])):
            raise ValueError("decoded action violates target interface bounds")
        return {"actions": action[None, :], "inference_duration_ns": duration,
                "metadata": {"artifact_form": "adapter_only", "greedy": True}}

    def load_counts(self):
        return dict(self._counts)

    def close(self):
        had_model = self.model is not None
        self.policy = self.model = self.processor = None
        self._episode_id = None
        if had_model:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
