"""Exceptions raised by OVLAB contracts."""


class ContractError(Exception):
    """Base class for contract-related errors."""


class ContractValidationError(ContractError, ValueError):
    """Raised when a contract field violates its schema."""


class ContractCompatibilityError(ContractError):
    """Raised when contract schema versions are incompatible."""


def validation_error(contract: str, field: str, reason: str) -> ContractValidationError:
    """Create a consistently formatted field validation error."""
    return ContractValidationError(f"{contract}.{field}: {reason}")
