"""Apply generic and compatibility contracts to the concrete LIBERO adapter."""

from dataclasses import replace

from helpers.contexts import make_run_context
from helpers.fake_libero import FakeLiberoBackend, fake_libero_episode
from helpers.mock_policy import MockPolicy
from ovlab_benchmarks.libero import LiberoAdapterSettings, LiberoBenchmarkAdapter
from ovlab_core import negotiate_capabilities

from contract.adapters.conformance import assert_benchmark_conformance


def benchmark():
    return LiberoBenchmarkAdapter(
        LiberoAdapterSettings(
            camera_width=5,
            camera_height=4,
            initialization_settling_steps=0,
            maximum_episode_steps=1,
        ),
        backend=FakeLiberoBackend(),
    )


def test_libero_adapter_passes_generic_conformance() -> None:
    assert_benchmark_conformance(benchmark(), fake_libero_episode())


def test_libero_capabilities_negotiate_with_matching_policy() -> None:
    adapter = benchmark()
    benchmark_capabilities = adapter.initialize(make_run_context())
    mock = MockPolicy().initialize(make_run_context())
    image = replace(
        mock.observation_requirements.images[0],
        name="camera.primary.rgb",
        shapes=((4, 5, 3),),
    )
    requirements = replace(
        mock.observation_requirements,
        images=(image,),
        proprioception=(),
        minimum_proprioception_count=0,
        maximum_proprioception_count=0,
    )
    policy = replace(mock, observation_requirements=requirements, output_action_spec=benchmark_capabilities.action_spec)
    assert negotiate_capabilities(benchmark_capabilities, policy).compatible
