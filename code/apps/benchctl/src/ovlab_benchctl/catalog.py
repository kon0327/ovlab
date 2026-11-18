"""Dependency-light catalogue generated from the model plugin registry."""
from copy import deepcopy
from .policy_plugins import registered_plugins


def registered_policies(repository_root=None):
    return [deepcopy(plugin.descriptor) for plugin in registered_plugins(repository_root)]
