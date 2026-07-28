"""Dependency-free registered policy identity catalogue."""

POLICY_DESCRIPTORS = (
    {"id": "vanilla", "config_type": "openvla_vanilla", "family": "openvla"},
    {"id": "lora", "config_type": "openvla_lora_merged", "family": "lora"},
    {"id": "openvla-oft", "config_type": "openvla_oft", "family": "openvla_oft"},
    {"id": "quic-peft", "config_type": "quic-peft", "family": "openvla_quic", "quic_profile": "QP0"},
    {"id": "quic-wc", "config_type": "quic-wc", "family": "openvla_quic", "quic_profile": "QP0"},
)


def registered_policies() -> list[dict[str, object]]:
    """Return static descriptors without provider discovery or runtime imports."""
    return [dict(item) for item in POLICY_DESCRIPTORS]
