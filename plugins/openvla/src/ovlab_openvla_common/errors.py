"""Dependency-light OpenVLA validation errors."""


class OpenVlaCommonError(ValueError):
    """Base error for shared OpenVLA contracts."""


class OpenVlaObservationError(OpenVlaCommonError):
    """A canonical policy observation is not valid for OpenVLA."""


class OpenVlaActionCodecError(OpenVlaCommonError):
    """An action cannot be converted to the configured target convention."""
