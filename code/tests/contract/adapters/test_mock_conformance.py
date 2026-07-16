"""Apply reusable adapter conformance checks to deterministic mocks."""

import pytest

from helpers.mock_benchmark import MockBenchmark
from helpers.mock_policy import MockPolicy

from .conformance import assert_benchmark_conformance, assert_policy_conformance


def test_mock_benchmark_conforms() -> None:
    assert_benchmark_conformance(MockBenchmark())


@pytest.mark.parametrize("horizon", [1, 2])
def test_mock_policy_conforms(horizon: int) -> None:
    assert_policy_conformance(MockPolicy(horizon=horizon), MockBenchmark())
