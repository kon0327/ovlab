"""Errors raised by the remote policy transport."""


class RemotePolicyError(RuntimeError):
    """Base error for the local policy RPC boundary."""


class RemotePolicyProtocolError(RemotePolicyError):
    """A peer sent an invalid or incompatible protocol message."""


class RemotePolicyServiceError(RemotePolicyError):
    """The remote policy service rejected a validly framed request."""


class RemotePolicyTimeoutError(RemotePolicyError):
    """A bounded startup, request, or shutdown operation timed out."""

