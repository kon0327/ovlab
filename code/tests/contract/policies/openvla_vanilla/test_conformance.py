import numpy as np

from contract.adapters.conformance import assert_policy_conformance
from helpers.contexts import make_run_context
from helpers.fake_openvla import FakeOpenVlaRuntime, SequenceClock
from helpers.mock_benchmark import MockBenchmark
from ovlab_benchmarks.libero.actions import libero_action_spec
from ovlab_core import negotiate_capabilities
from ovlab_core.contracts import (
    BenchmarkCapabilities, ColorSpace, ImageEncoding, ImageObservation, ImageObservationSpec,
    ObservationSpec, OVLAB_CONTRACT_VERSION, PolicyObservation, SignalRegistry,
)
from ovlab_openvla_common import OpenVlaModelSource
from ovlab_openvla_vanilla import OpenVlaVanillaAdapter, OpenVlaVanillaSettings


def libero_like_capabilities(action_spec=None):
    return BenchmarkCapabilities(
        "fake-libero", "1.0.0", OVLAB_CONTRACT_VERSION,
        ObservationSpec((ImageObservationSpec("camera.primary.rgb", ((256, 256, 3),), "uint8",
                                                   (ImageEncoding.RAW,), (ColorSpace.RGB,)),)),
        libero_action_spec() if action_spec is None else action_spec,
        SignalRegistry(()), True, False, False, ("fake-libero",),
    )


class FakeLiberoBenchmark(MockBenchmark):
    def __init__(self):
        super().__init__(capabilities_override=libero_like_capabilities())

    def _make_observation(self, step_id, instruction, step_index):
        timestamp = step_index * 10
        image = ImageObservation("camera.primary.rgb", np.zeros((256, 256, 3), dtype=np.uint8),
                                 timestamp, ImageEncoding.RAW, ColorSpace.RGB, "agentview")
        return PolicyObservation(step_id, timestamp, instruction, (image,))


def make_adapter(tmp_path):
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    settings = OpenVlaVanillaSettings(OpenVlaModelSource(str(checkpoint)), "bridge_orig")
    return OpenVlaVanillaAdapter(settings, FakeOpenVlaRuntime(),
                                 clock_ns=SequenceClock(), wall_clock_ns=lambda: 500)


def test_generic_policy_conformance(tmp_path):
    assert_policy_conformance(make_adapter(tmp_path), FakeLiberoBenchmark())


def test_capability_negotiation_matches_libero(tmp_path):
    adapter = make_adapter(tmp_path)
    policy = adapter.initialize(make_run_context())
    report = negotiate_capabilities(libero_like_capabilities(), policy)
    assert report.compatible


def test_capability_negotiation_rejects_wrong_gripper_convention(tmp_path):
    from dataclasses import replace
    from ovlab_core.contracts import GripperConvention
    adapter = make_adapter(tmp_path)
    policy = adapter.initialize(make_run_context())
    wrong = replace(libero_action_spec(), gripper_convention=GripperConvention.OPEN_POSITIVE)
    report = negotiate_capabilities(libero_like_capabilities(wrong), policy)
    assert not report.compatible
    assert any(issue.code == "GRIPPER_CONVENTION_MISMATCH" for issue in report.issues)
