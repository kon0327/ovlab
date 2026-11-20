"""Thin PolicyAdapter wrappers delegating all future model behavior externally."""

from __future__ import annotations

from collections.abc import Mapping
import time

import numpy as np

from ovlab_core.contracts import (
    ActionPrediction, PredictionId, PredictionValidity,
)
from ovlab_policy_sdk import PolicyAdapter

from .descriptors import QuICMethodDescriptor, QuICVariant
from .errors import QuICProviderContractError
from .provider import QuICCapabilityDeclaration, QuICProviderLoader

_PRIVILEGED_FIELDS = frozenset({
    "reward", "success", "goal_predicate", "collision", "collisions", "contacts",
    "object_pose", "object_poses", "simulator", "simulator_state", "future_observation",
    "future_observations", "termination", "termination_state",
})


def _contains_privileged_field(value) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _PRIVILEGED_FIELDS or _contains_privileged_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_privileged_field(item) for item in value)
    return False


class OpenVLAQuICAdapter(PolicyAdapter):
    """Variant-neutral lifecycle shell with no QuIC mathematics or fallback policy."""

    required_variant: QuICVariant

    def __init__(
        self,
        descriptor: QuICMethodDescriptor,
        *,
        provider_loader: QuICProviderLoader | None = None,
        _test_provider=None,
        clock_ns=time.perf_counter_ns,
        wall_clock_ns=time.time_ns,
    ) -> None:
        super().__init__()
        if not isinstance(descriptor, QuICMethodDescriptor):
            raise TypeError("descriptor must be QuICMethodDescriptor")
        if descriptor.variant is not self.required_variant:
            raise ValueError(
                f"{type(self).__name__} requires variant {self.required_variant.value}"
            )
        self.descriptor = descriptor
        self._loader = provider_loader or QuICProviderLoader()
        self._provider = _test_provider
        self._provider_is_test_double = _test_provider is not None
        self._clock_ns, self._wall_clock_ns = clock_ns, wall_clock_ns
        self._capability_declaration = None
        self._prediction_index = 0

    def _initialize(self, run_context):
        # This guard intentionally precedes provider discovery, CUDA, models, files, and sockets.
        self.descriptor.require_runtime_ready()
        provider = self._provider or self._loader.load(self.descriptor)
        self._provider = provider
        try:
            api_version = provider.api_version()
            if api_version != self.descriptor.provider.api_version:
                raise QuICProviderContractError(
                    f"provider API {api_version!r} differs from required "
                    f"{self.descriptor.provider.api_version!r}"
                )
            description = dict(provider.describe())
            required_description = {
                "family", "variant_id", "scientific_identity_hash", "base_model_identity",
                "artifact_identity", "normalization_identity", "profile", "placement_manifest",
            }
            if set(description) != required_description:
                raise QuICProviderContractError(
                    f"provider description fields must equal {sorted(required_description)}"
                )
            expected = self.descriptor.canonical_dict()
            expected_description = {
                "family": expected["family"],
                "variant_id": expected["variant_id"],
                "scientific_identity_hash": self.descriptor.scientific_hash,
                "base_model_identity": expected["base_model_identity"],
                "artifact_identity": expected["artifact_identity"],
                "normalization_identity": expected["normalization_identity"],
                "profile": expected["profile"],
                "placement_manifest": expected["placement_manifest"],
            }
            if description != expected_description:
                raise QuICProviderContractError(
                    "provider method, artifact, or resource identity differs from the descriptor"
                )
            declaration = QuICCapabilityDeclaration.from_mapping(
                provider.capability_description()
            )
            identity = provider.load({
                "variant_id": self.descriptor.variant.value,
                "run_id": str(run_context.run_id),
                "scientific_identity_hash": self.descriptor.scientific_hash,
                "method_descriptor": self.descriptor.canonical_dict(),
            })
            if not isinstance(identity, Mapping):
                raise QuICProviderContractError("provider load() must return identity metadata")
            counts = dict(provider.load_counts())
            if any(not isinstance(key, str) or type(value) is not int or value < 0
                   for key, value in counts.items()):
                raise QuICProviderContractError("provider load counts must be non-negative integers")
            self._capability_declaration = declaration
            return declaration.to_policy_capabilities(
                self.descriptor,
                {
                    **dict(identity),
                    "api_name": self.descriptor.provider.api_name,
                    "api_version": api_version,
                    "source_commit": self.descriptor.provider.source_commit,
                    "test_double": self._provider_is_test_double,
                },
                counts,
            )
        except Exception:
            try:
                provider.close()
            finally:
                self._provider = None
            raise

    def _reset_episode(self, episode_context) -> None:
        self._prediction_index = 0
        self._provider.reset_episode({
            "episode_id": str(episode_context.episode_id),
            "task_id": str(episode_context.task_id),
            "seed": episode_context.seed,
            "instruction": episode_context.initial_instruction.text,
        })

    def _predict(self, observation):
        declaration = self._capability_declaration
        images = {item.name: item for item in observation.images}
        proprio = {item.name: item for item in observation.proprioception}
        expected_images = {item.name for item in declaration.images}
        expected_proprio = {item.name for item in declaration.proprioception}
        required_images = {item.name for item in declaration.images if item.required}
        required_proprio = {item.name for item in declaration.proprioception if item.required}
        if not required_images <= set(images) or not required_proprio <= set(proprio):
            raise QuICProviderContractError("policy observation lacks provider-negotiated channels")
        for spec in declaration.images:
            if spec.name not in images:
                continue
            value = images[spec.name]
            if (
                tuple(value.data.shape) not in spec.shapes
                or value.data.dtype.name != spec.dtype
                or value.encoding not in spec.encodings
                or value.color_space not in spec.color_spaces
            ):
                raise QuICProviderContractError(
                    f"image {spec.name!r} violates provider-negotiated metadata"
                )
        for spec in declaration.proprioception:
            if spec.name not in proprio:
                continue
            value = proprio[spec.name]
            if tuple(value.values.shape) not in spec.shapes or value.values.dtype.name != spec.dtype:
                raise QuICProviderContractError(
                    f"proprioception {spec.name!r} violates provider-negotiated metadata"
                )
        episode = self._episode_context
        request_id = f"{episode.episode_id}:prediction:{self._prediction_index}"
        request = {
            "request_id": request_id,
            "episode_id": str(episode.episode_id),
            "step_id": str(observation.step_id),
            "instruction": observation.instruction.text,
            "images": {
                name: {
                    "data": np.array(images[name].data, copy=True),
                    "shape": tuple(images[name].data.shape),
                    "dtype": str(images[name].data.dtype),
                    "layout": "HWC",
                }
                for name in sorted(expected_images & set(images))
            },
            "proprioception": {
                name: {
                    "values": np.array(proprio[name].values, copy=True),
                    "shape": tuple(proprio[name].values.shape),
                    "dtype": str(proprio[name].values.dtype),
                }
                for name in sorted(expected_proprio & set(proprio))
            },
        }
        started = self._clock_ns()
        result = self._provider.predict(request)
        finished = self._clock_ns()
        if not isinstance(result, Mapping) or set(result) != {
            "actions", "inference_duration_ns", "metadata"
        }:
            raise QuICProviderContractError(
                "provider prediction must contain only actions, inference_duration_ns, and metadata"
            )
        actions = np.asarray(result["actions"])
        if actions.dtype != np.float32 or actions.ndim != 2 or actions.shape[1] != declaration.action_spec.dimension:
            raise QuICProviderContractError("provider actions violate negotiated float32 [H,D] shape")
        if not np.all(np.isfinite(actions)):
            raise QuICProviderContractError("provider actions must be finite")
        minimum, maximum = declaration.action_spec.minimum, declaration.action_spec.maximum
        if minimum is not None and (
            np.any(actions < minimum[np.newaxis, :])
            or np.any(actions > maximum[np.newaxis, :])
        ):
            raise QuICProviderContractError("provider actions violate negotiated action bounds")
        horizon = actions.shape[0]
        if not declaration.minimum_horizon <= horizon <= declaration.maximum_horizon:
            raise QuICProviderContractError("provider prediction horizon violates negotiated capabilities")
        duration = result["inference_duration_ns"]
        if type(duration) is not int or duration < 0 or duration > finished - started:
            raise QuICProviderContractError("provider inference duration is invalid")
        metadata = result["metadata"]
        if not isinstance(metadata, Mapping):
            raise QuICProviderContractError("provider prediction metadata must be a mapping")
        if _contains_privileged_field(metadata):
            raise QuICProviderContractError(
                "provider prediction metadata must not contain privileged evaluation fields"
            )
        self._prediction_index += 1
        return ActionPrediction(
            prediction_id=PredictionId(request_id),
            step_id=observation.step_id,
            actions=actions,
            action_spec=declaration.action_spec,
            timestamp_ns=self._wall_clock_ns(),
            inference_duration_ns=finished - started,
            horizon=horizon,
            validity=PredictionValidity.VALID,
            metadata={
                "external_provider": dict(metadata),
                "implementation_owner": self.descriptor.provider.source_repository,
                "model_inference_duration_ns": duration,
            },
        )

    def _end_episode(self, episode_context) -> None:
        del episode_context
        self._prediction_index = 0

    def _close(self) -> None:
        if self._provider is not None:
            self._provider.close()
        self._provider = None
        self._capability_declaration = None


class OpenVLAQuICPEFTAdapter(OpenVLAQuICAdapter):
    required_variant = QuICVariant.PEFT


class OpenVLAQuICWCAdapter(OpenVLAQuICAdapter):
    required_variant = QuICVariant.WC
