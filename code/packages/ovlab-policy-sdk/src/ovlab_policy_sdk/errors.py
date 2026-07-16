"""Policy adapter exceptions."""

from ovlab_core.contracts import AdapterState


class PolicyAdapterError(Exception):
    """Base class for policy adapter errors."""


class PolicyLifecycleError(PolicyAdapterError):
    """Raised when an operation is invalid in the current adapter state."""

    def __init__(self, operation: str, state: AdapterState, reason: str | None = None) -> None:
        message = f"policy operation '{operation}' is invalid in state '{state.value}'"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message)
        self.operation = operation
        self.state = state


class PolicyInferenceError(PolicyAdapterError):
    """Raised when a policy cannot produce a conforming prediction."""
