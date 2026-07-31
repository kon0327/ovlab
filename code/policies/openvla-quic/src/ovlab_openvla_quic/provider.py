"""Neutral, lazy external provider boundary; external code need not import OVLAB."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from typing import Protocol, runtime_checkable

from ovlab_core.contracts import (
    OVLAB_CONTRACT_VERSION,
    OVLAB_VERSION,
    ObservationRequirements,
    PolicyCapabilities,
)
from ovlab_remote_policy.protocol import (
    action_spec_from_wire, image_spec_from_wire, proprio_spec_from_wire,
)

from .descriptors import QuICMethodDescriptor
from .errors import QuICImplementationUnavailableError, QuICProviderContractError


@runtime_checkable
class QuICExternalProvider(Protocol):
    """Duck-typed API to be implemented publicly by external/openvla-quic."""

    def api_version(self) -> str: ...
    def describe(self) -> Mapping[str, object]: ...
    def capability_description(self) -> Mapping[str, object]: ...
    def load(self, request: Mapping[str, object]) -> Mapping[str, object]: ...
    def reset_episode(self, request: Mapping[str, object]) -> None: ...
    def predict(self, request: Mapping[str, object]) -> Mapping[str, object]: ...
    def close(self) -> None: ...
    def load_counts(self) -> Mapping[str, int]: ...


@dataclass(frozen=True, slots=True)
class QuICCapabilityDeclaration:
    images: tuple
    proprioception: tuple
    action_spec: object
    minimum_horizon: int
    maximum_horizon: int
    dynamic_instructions: bool
    deterministic_reset: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "QuICCapabilityDeclaration":
        if not isinstance(value, Mapping):
            raise QuICProviderContractError("provider capability description must be a mapping")
        required = {
            "images", "proprioception", "action_spec", "minimum_horizon", "maximum_horizon",
            "dynamic_instructions", "deterministic_reset",
        }
        if set(value) != required:
            raise QuICProviderContractError(
                f"provider capability fields must equal {sorted(required)}"
            )
        try:
            images = tuple(image_spec_from_wire(item) for item in value["images"])
            proprio = tuple(proprio_spec_from_wire(item) for item in value["proprioception"])
            action_spec = action_spec_from_wire(value["action_spec"])
        except Exception as exc:
            raise QuICProviderContractError(f"invalid provider capability schema: {exc}") from exc
        minimum, maximum = value["minimum_horizon"], value["maximum_horizon"]
        if type(minimum) is not int or type(maximum) is not int or minimum <= 0 or maximum < minimum:
            raise QuICProviderContractError("provider horizons must be positive and ordered")
        for name in ("dynamic_instructions", "deterministic_reset"):
            if type(value[name]) is not bool:
                raise QuICProviderContractError(f"provider {name} must be boolean")
        if not images:
            raise QuICProviderContractError("provider must declare at least one policy-visible image")
        names = [item.name for item in (*images, *proprio)]
        if len(names) != len(set(names)):
            raise QuICProviderContractError("provider observation channel names must be unique")
        return cls(images, proprio, action_spec, minimum, maximum,
                   value["dynamic_instructions"], value["deterministic_reset"])

    def to_policy_capabilities(
        self,
        descriptor: QuICMethodDescriptor,
        provider_identity: Mapping[str, object],
        load_counts: Mapping[str, int],
    ) -> PolicyCapabilities:
        return PolicyCapabilities(
            component_name=f"ovlab-openvla-{descriptor.variant.value}",
            component_version=OVLAB_VERSION,
            contract_version=OVLAB_CONTRACT_VERSION,
            observation_requirements=ObservationRequirements(
                images=self.images,
                proprioception=self.proprioception,
                minimum_image_count=sum(item.required for item in self.images),
                maximum_image_count=len(self.images),
                minimum_proprioception_count=sum(item.required for item in self.proprioception),
                maximum_proprioception_count=len(self.proprioception),
            ),
            output_action_spec=self.action_spec,
            supports_single_action=self.minimum_horizon <= 1 <= self.maximum_horizon,
            supports_action_chunks=self.maximum_horizon > 1,
            minimum_action_horizon=self.minimum_horizon,
            maximum_action_horizon=self.maximum_horizon,
            supports_dynamic_instructions=self.dynamic_instructions,
            supports_deterministic_reset=self.deterministic_reset,
            exposes_raw_policy_output=False,
            metadata={
                "policy_family": "openvla_quic",
                "method_descriptor": descriptor.as_metadata(),
                "provider_identity": dict(provider_identity),
                "provider_load_counts": dict(load_counts),
                "normalization_identity": dict(descriptor.normalization_identity),
                "implementation_owner": descriptor.provider.source_repository,
            },
        )


class QuICProviderLoader:
    """Discover the external provider only after the descriptor is runtime-complete."""

    def load(self, descriptor: QuICMethodDescriptor) -> QuICExternalProvider:
        descriptor.require_runtime_ready()
        try:
            module = importlib.import_module(descriptor.provider.package)
        except (ImportError, ModuleNotFoundError) as exc:
            raise QuICImplementationUnavailableError(
                descriptor.variant.value,
                descriptor.provider.package,
                descriptor.implementation_status.value,
                descriptor.variant.next_gate,
                descriptor.provider.source_repository,
            ) from exc
        factory = getattr(module, "create_provider", None)
        if not callable(factory):
            raise QuICProviderContractError(
                f"{descriptor.provider.package} must export callable create_provider(variant=...)"
            )
        provider = factory(variant=descriptor.variant.value)
        required = (
            "api_version", "describe", "capability_description", "load", "reset_episode",
            "predict", "close", "load_counts",
        )
        if any(not callable(getattr(provider, name, None)) for name in required):
            raise QuICProviderContractError("external QuIC provider API is incomplete")
        return provider
