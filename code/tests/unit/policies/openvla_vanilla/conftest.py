from dataclasses import replace

import numpy as np
import pytest

from helpers.contexts import make_episode_context, make_run_context
from ovlab_core.contracts import (
    ColorSpace, ImageEncoding, ImageObservation, Instruction, InstructionId,
    InstructionSource, PolicyObservation, StepId,
)
from ovlab_openvla_common import OpenVlaModelSource
from ovlab_openvla_vanilla import OpenVlaVanillaAdapter, OpenVlaVanillaSettings
from helpers.fake_openvla import FakeOpenVlaRuntime, SequenceClock


@pytest.fixture
def settings(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    return OpenVlaVanillaSettings(OpenVlaModelSource(str(checkpoint)), "bridge_orig")


@pytest.fixture
def runtime():
    return FakeOpenVlaRuntime()


@pytest.fixture
def observation():
    instruction = Instruction(InstructionId("instruction-0"), "Put The Object In The Basket", 1,
                              InstructionSource.BENCHMARK)
    data = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    image = ImageObservation("camera.primary.rgb", data, 2, ImageEncoding.RAW, ColorSpace.RGB, "agentview")
    return PolicyObservation(StepId("episode-0-step-0"), 2, instruction, (image,))


@pytest.fixture
def active_adapter(settings, runtime):
    adapter = OpenVlaVanillaAdapter(settings, runtime, clock_ns=SequenceClock(), wall_clock_ns=lambda: 500)
    adapter.initialize(make_run_context())
    adapter.reset_episode(make_episode_context())
    return adapter
