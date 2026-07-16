"""Deterministic Torch-free OpenVLA runtime used by policy tests."""

import numpy as np

from ovlab_openvla_common import OpenVlaCheckpointIdentity, OpenVlaDecodedAction
from ovlab_openvla_vanilla import OpenVlaCheckpointError, RuntimePrediction


class FakeOpenVlaRuntime:
    def __init__(self, action=None, *, keys=("bridge_orig",), load_error=None):
        self.action = np.array(action if action is not None else [0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        self.keys = keys
        self.load_error = load_error
        self.loaded = False
        self.closed = False
        self.calls = []
        self.resets = []

    def load(self, settings):
        if self.load_error:
            raise self.load_error
        if settings.unnorm_key not in self.keys:
            raise OpenVlaCheckpointError(f"missing action statistics for {settings.unnorm_key}")
        self.loaded = True
        return OpenVlaCheckpointIdentity(
            settings.model.source, "/fake/snapshot", "c8f03f48af692657d3060c19588038c7220e9af9",
            "fake-model", "fake-processor", settings.unnorm_key, "sha256:fake", settings.model.revision,
            settings.model.expected_checksum, settings.settings_hash,
        )

    def predict(self, image, prompt, unnorm_key):
        self.calls.append((image.copy(), prompt, unnorm_key))
        return RuntimePrediction(OpenVlaDecodedAction(self.action), 20, 30)

    def reset_episode(self, seed):
        self.resets.append(seed)

    def close(self):
        self.closed = True
        self.loaded = False


class SequenceClock:
    def __init__(self, values=(100, 160, 170, 200)):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)
