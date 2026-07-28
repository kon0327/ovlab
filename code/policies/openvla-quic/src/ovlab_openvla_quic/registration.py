"""Unambiguous registration without runtime fallback aliases."""

from .adapter import OpenVLAQuICPEFTAdapter, OpenVLAQuICWCAdapter
from .descriptors import QuICVariant

QUIC_ADAPTER_REGISTRY = {
    QuICVariant.PEFT.value: OpenVLAQuICPEFTAdapter,
    QuICVariant.WC.value: OpenVLAQuICWCAdapter,
}


def adapter_class_for(variant: str):
    try:
        return QUIC_ADAPTER_REGISTRY[variant]
    except KeyError as exc:
        raise ValueError(f"unknown QuIC variant: {variant!r}") from exc
