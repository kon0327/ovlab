"""Conversion of validated descriptor-only documents into typed identities."""

from collections.abc import Mapping

from .descriptors import (
    EXTERNAL_QUIC_COMMIT, QuICProfileId, QuICVariant, skeleton_descriptor,
)
from .errors import QuICDescriptorError


def descriptor_from_document(document: Mapping[str, object]):
    if document.get("kind") != "quic_policy_descriptor":
        raise QuICDescriptorError("document kind must be quic_policy_descriptor")
    variant = QuICVariant(document["variant"])
    descriptor = skeleton_descriptor(variant, QuICProfileId(document["profile"]["id"]))
    expected = descriptor.canonical_dict()
    checks = {
        "implementation_status": expected["implementation_status"],
        "published_method_relation": expected["published_method_relation"],
        "weight_compression": expected["weight_compression"],
    }
    for key, value in checks.items():
        if document[key] != value:
            raise QuICDescriptorError(f"configuration misclassifies {key}")
    provider = document["external_provider"]
    if provider["source_commit"] != EXTERNAL_QUIC_COMMIT:
        raise QuICDescriptorError("external provider commit differs from pinned submodule")
    if document["runtime_validated"] is not False or document["compression_verified"] is not False:
        raise QuICDescriptorError("skeleton configuration cannot claim validation")
    return descriptor
