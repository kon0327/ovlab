"""Foreground service and handshake-only connection orchestration."""

from __future__ import annotations

import os
from pathlib import Path
import socket as socket_module
import subprocess
import sys
import threading
import time

from helpers.contexts import make_run_context
from helpers.mock_policy import MockPolicy
from ovlab_benchctl.application import OvlabApplication
from ovlab_benchmarks.libero import configured_capabilities
from ovlab_core.contracts import ObservationRequirements, PolicyCapabilities
from ovlab_policy_sdk import PolicyAdapter
from ovlab_remote_policy import UnixPolicyClient
from ovlab_remote_policy.service import PolicyService


REPOSITORY = Path(__file__).resolve().parents[4]


def _profile(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(
        'schema_version: "0.1.0"\nkind: local_profile\nid: cli-service-test\n\npaths:\n'
        f'  checkpoint_root: {tmp_path}/checkpoints\n  dataset_root: {tmp_path}/datasets\n'
        f'  runs_root: {tmp_path}/runs\n\ndevices:\n  primary_gpu: cuda:0\n', encoding="utf-8",
    )
    return path


def _wait(socket, thread):
    deadline = time.monotonic() + 2
    while not socket.is_socket() and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert socket.is_socket()


def test_foreground_service_uses_test_only_factory_and_cleans_socket(tmp_path):
    profile = _profile(tmp_path)
    app = OvlabApplication(REPOSITORY, environment={**os.environ, "OVLAB_LOCAL_PROFILE": str(profile)})
    socket = tmp_path / "service.sock"
    thread = threading.Thread(
        target=lambda: app.serve(
            "configs/experiments/mock-e2e-smoke.yaml",
            socket_path=socket,
            adapter_factory=lambda _settings: MockPolicy(),
        )
    )
    thread.start(); _wait(socket, thread)
    client = UnixPolicyClient(socket, request_timeout_s=2)
    client.connect(); capabilities = client.initialize(make_run_context(run_id="cli-service"))
    assert capabilities.component_name == "mock-policy"
    assert client.health()["state"] == "ready"
    client.request_close(); client.close_socket(); thread.join(timeout=2)
    assert not thread.is_alive() and not socket.exists()


def test_health_is_ready_only_after_initialization_and_does_not_consume_service(tmp_path):
    profile = _profile(tmp_path)
    socket = tmp_path / "health.sock"
    initialized_before_socket = []

    class ReadinessPolicy(MockPolicy):
        def _initialize(self, context):
            initialized_before_socket.append(not socket.exists())
            return super()._initialize(context)

        def _predict(self, observation):
            raise AssertionError("health must never predict")

    app = OvlabApplication(REPOSITORY, environment={**os.environ, "OVLAB_LOCAL_PROFILE": str(profile)})
    thread = threading.Thread(
        target=lambda: app.serve(
            "configs/experiments/mock-e2e-smoke.yaml",
            socket_path=socket,
            adapter_factory=lambda _settings: ReadinessPolicy(),
        )
    )
    thread.start(); _wait(socket, thread)
    assert initialized_before_socket == [True]

    first = app.service_health(socket)
    second = app.service_health(socket)
    assert first["ready"] is True and second["ready"] is True
    assert first["protocol_version"] == "ovlab-policy-rpc/1.0.0"
    assert first["prediction_count"] == 0 and first["trace_created"] is False
    assert thread.is_alive()

    client = UnixPolicyClient(socket, request_timeout_s=2)
    client.connect(); client.initialize(make_run_context(run_id="deployment-run"))
    client.request_close(); client.close_socket(); thread.join(timeout=2)
    assert not thread.is_alive() and not socket.exists()


def test_service_factory_failure_occurs_before_socket_readiness(tmp_path):
    profile = _profile(tmp_path)
    app = OvlabApplication(REPOSITORY, environment={**os.environ, "OVLAB_LOCAL_PROFILE": str(profile)})
    socket = tmp_path / "never-ready.sock"
    try:
        app.serve(
            "configs/experiments/mock-e2e-smoke.yaml",
            socket_path=socket,
            adapter_factory=lambda _settings: (_ for _ in ()).throw(RuntimeError("provider init failed")),
        )
    except RuntimeError as exc:
        assert str(exc) == "provider init failed"
    else:
        raise AssertionError("provider failure was not propagated")
    assert not socket.exists()


def test_service_refuses_and_preserves_unrelated_socket(tmp_path):
    profile = _profile(tmp_path)
    app = OvlabApplication(REPOSITORY, environment={**os.environ, "OVLAB_LOCAL_PROFILE": str(profile)})
    socket = tmp_path / "unrelated.sock"
    owner = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    owner.bind(str(socket))
    try:
        try:
            app.serve(
                "configs/experiments/mock-e2e-smoke.yaml",
                socket_path=socket,
                adapter_factory=lambda _settings: MockPolicy(),
            )
        except RuntimeError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("unrelated socket was accepted")
        assert socket.is_socket()
    finally:
        owner.close()
        socket.unlink()


def test_sigterm_closes_adapter_and_removes_owned_socket(tmp_path):
    socket = tmp_path / "signal.sock"
    script = f'''\
from ovlab_benchctl import cli
from ovlab_benchctl.application import OvlabApplication
from helpers.mock_policy import MockPolicy
from ovlab_remote_policy.service import PolicyService
class App:
    def serve(self, config, socket_path=None):
        PolicyService(socket_path, MockPolicy(), identity_provider=OvlabApplication._identity_provider).serve()
cli._application = lambda: App()
raise SystemExit(cli.main(["service", "serve", "ignored.yaml", "--socket", {str(socket)!r}]))
'''
    process = subprocess.Popen(
        [sys.executable, "-c", script], cwd=REPOSITORY, env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    deadline = time.monotonic() + 2
    while not socket.is_socket() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert socket.is_socket()
    process.terminate()
    stdout, stderr = process.communicate(timeout=2)
    assert process.returncode == 130
    assert stdout == ""
    assert "operation interrupted" in stderr
    assert not socket.exists()


class _HandshakeOnlyPolicy(PolicyAdapter):
    def __init__(self, capabilities):
        super().__init__()
        self._configured = capabilities
        self.predict_count = 0

    def _initialize(self, _context):
        return self._configured

    def _reset_episode(self, _context):
        raise AssertionError("connect must not reset an episode")

    def _predict(self, _observation):
        self.predict_count += 1
        raise AssertionError("connect must not predict")

    def _close(self):
        pass


def _policy_capabilities(resolved, *, compatible=True):
    benchmark = configured_capabilities(resolved.benchmark_settings)
    images = benchmark.observation_spec.images
    action = benchmark.action_spec
    if not compatible:
        from helpers.mock_specs import mock_action_spec
        action = mock_action_spec()
    return PolicyCapabilities(
        "handshake-only-policy", "1.0.0", benchmark.contract_version,
        ObservationRequirements(images=(images[0],), minimum_image_count=1, maximum_image_count=1),
        action, True, False, 1, 1, True, True, False,
        {
            "checkpoint_identity": {"unnorm_key": "libero_10", "action_statistics_identity": "test"},
            "prompt_template": "test-prompt@1", "action_codec": "test-codec@1",
            "action_codec_owner": "test-only",
        },
    )


def _connect_case(tmp_path, compatible):
    profile = _profile(tmp_path)
    app = OvlabApplication(REPOSITORY, environment={
        **os.environ,
        "OVLAB_LOCAL_PROFILE": str(profile),
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
    })
    config = "configs/experiments/libero10-lora-merged-rpc-smoke.yaml"
    resolved = app.resolve(config, mode="runtime")
    adapter = _HandshakeOnlyPolicy(_policy_capabilities(resolved, compatible=compatible))
    socket = app.default_socket(config)
    socket.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(socket.parent, 0o700)
    service = PolicyService(socket, adapter, identity_provider=app._identity_provider)
    thread = threading.Thread(target=service.serve)
    thread.start(); _wait(socket, thread)
    result = app.connect(config)
    thread.join(timeout=2)
    return result, adapter, socket, thread


def test_connect_performs_handshake_negotiation_without_prediction_or_trace(tmp_path):
    result, adapter, socket, thread = _connect_case(tmp_path, True)
    assert result["compatible"] is True and result["compatibility_issues"] == []
    assert result["prediction_count"] == 0 and result["trace_created"] is False
    assert adapter.predict_count == 0
    assert not socket.exists() and not thread.is_alive()


def test_connect_reports_capability_incompatibility_structurally(tmp_path):
    result, adapter, socket, thread = _connect_case(tmp_path, False)
    assert result["compatible"] is False
    assert any(item["code"] == "ACTION_DIMENSION_MISMATCH" for item in result["compatibility_issues"])
    assert adapter.predict_count == 0
    assert not socket.exists() and not thread.is_alive()
