"""Descriptor-aware service factory that rejects skeletons before socket readiness."""

from .adapter import OpenVLAQuICPEFTAdapter, OpenVLAQuICWCAdapter
from .descriptors import QuICMethodDescriptor, QuICVariant


def create_runtime_adapter(descriptor: QuICMethodDescriptor):
    """Future service entry point; Gate F always fails before constructing a service."""
    descriptor.require_runtime_ready()
    cls = (
        OpenVLAQuICPEFTAdapter
        if descriptor.variant is QuICVariant.PEFT
        else OpenVLAQuICWCAdapter
    )
    return cls(descriptor)
