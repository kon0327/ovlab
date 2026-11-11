"""Dependency-light catalogue of policy implementations and source ownership."""

from copy import deepcopy

from .quic_experiments import PROFILES

POLICY_DESCRIPTORS = (
    {"id": "vanilla", "config_type": "openvla_vanilla", "family": "openvla",
     "runtime_status": "implemented", "source_repositories": ["external/openvla"]},
    {"id": "lora", "config_type": "openvla_lora_merged", "family": "lora",
     "runtime_status": "implemented", "source_repositories": ["external/openvla"]},
    {"id": "openvla-oft", "config_type": "openvla_oft", "family": "openvla_oft",
     "runtime_status": "implemented", "source_repositories": ["external/openvla-oft"]},
    {"id": "quic-peft", "config_type": "openvla_quic", "family": "openvla_quic",
     "runtime_status": "experimental", "supported_quic_profiles": [*PROFILES, "custom"],
     "source_repositories": ["external/openvla-quic", "external/quic-peft-vla"]},
    {"id": "quic-wc", "config_type": "quic-wc", "family": "openvla_quic",
     "runtime_status": "unavailable", "source_repositories": []},
)


def registered_policies() -> list[dict[str, object]]:
    """Return declared capabilities without provider discovery or runtime imports."""
    return deepcopy(list(POLICY_DESCRIPTORS))
