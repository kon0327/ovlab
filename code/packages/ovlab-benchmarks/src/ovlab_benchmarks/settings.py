"""Control-plane benchmark description, independent of simulator startup."""
from typing import Protocol

from ovlab_core.contracts import ActionSpec, BenchmarkCapabilities


class BenchmarkSettings(Protocol):
    def describe_capabilities(self, action_spec: ActionSpec) -> BenchmarkCapabilities:
        """Describe configured inputs/actions without acquiring runtime resources."""
        ...
