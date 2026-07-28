"""Typed QuIC integration failures."""


class QuICError(RuntimeError):
    """Base error for the contract-only QuIC integration."""


class QuICDescriptorError(QuICError, ValueError):
    """A methodological, artifact, placement, or accounting contract is invalid."""


class QuICProviderContractError(QuICError):
    """An external provider does not implement the negotiated neutral API."""


class QuICImplementationUnavailableError(QuICError):
    """The requested external implementation is intentionally unavailable."""

    def __init__(
        self,
        variant: str,
        package: str,
        status: str,
        next_gate: str,
        source: str = "external/openvla-quic",
    ) -> None:
        self.variant = variant
        self.expected_package = package
        self.expected_source = source
        self.implementation_status = status
        self.next_implementation_gate = next_gate
        super().__init__(
            f"{variant} implementation is unavailable: expected external provider {package!r} "
            f"from {source!r}; "
            f"status={status}; next implementation gate={next_gate}"
        )


class QuICPEFTIntegrationIncompleteError(QuICImplementationUnavailableError):
    """The legacy compound source exists, but no validated OpenVLA bridge exists."""

    def __init__(self) -> None:
        super().__init__(
            "quic-peft",
            "openvla_quic.ovlab_provider",
            "legacy_reference_available_openvla_integration_skeleton",
            "I",
            "external/openvla-quic -> external/compound-peft",
        )


class QuICWCImplementationIncompleteError(QuICImplementationUnavailableError):
    """QuIC-WC has neither a source backend nor a runtime implementation."""

    def __init__(self) -> None:
        super().__init__(
            "quic-wc",
            "openvla_quic.ovlab_provider",
            "source_absent_implementation_skeleton",
            "J",
            "external/openvla-quic",
        )
