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
