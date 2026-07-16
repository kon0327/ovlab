"""Focused errors for the Vanilla adapter."""

from ovlab_policy_sdk.errors import PolicyAdapterError, PolicyInferenceError


class OpenVlaVanillaError(PolicyAdapterError):
    """Base error for the OpenVLA Vanilla policy."""


class OpenVlaDependencyError(OpenVlaVanillaError):
    """The tested runtime dependency stack is unavailable."""


class OpenVlaLoadError(OpenVlaVanillaError):
    """The local model or processor could not be loaded."""


class OpenVlaCheckpointError(OpenVlaLoadError):
    """Checkpoint resolution or action statistics are invalid."""


class OpenVlaPreprocessingError(OpenVlaVanillaError):
    """Canonical input could not be processed."""


class OpenVlaInferenceError(PolicyInferenceError, OpenVlaVanillaError):
    """OpenVLA failed to produce a conforming action."""


class OpenVlaActionDecodeError(OpenVlaInferenceError):
    """The decoded model action is invalid."""
