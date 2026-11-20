"""Compatibility access to registered project helpers; new callers use plugin APIs."""

def __getattr__(name):
    from .training_plugins import extension_symbol
    return extension_symbol('training_quic', name)
