from __future__ import annotations

import os
import sys

import pytest

from ovlab_remote_policy import OwnedPolicyServiceProcess, RemotePolicyError, RemotePolicyTimeoutError


def test_startup_timeout_terminates_reaps_and_captures_log(tmp_path):
    socket_path = tmp_path / "never.sock"
    log_path = tmp_path / "startup.log"
    owner = OwnedPolicyServiceProcess(
        [sys.executable, "-c", "import time; print('starting', flush=True); time.sleep(10)"],
        socket_path,
        log_path,
        startup_timeout_s=0.03,
        shutdown_timeout_s=0.03,
    )
    with pytest.raises(RemotePolicyTimeoutError, match="startup timed out"):
        owner.start()
    assert owner.process is None
    assert not socket_path.exists()
    assert "starting" in log_path.read_text()


def test_shutdown_timeout_terminates_reaps_and_removes_socket(tmp_path):
    socket_path = tmp_path / "sleep.sock"
    log_path = tmp_path / "shutdown.log"
    program = (
        "import os,socket,time; "
        f"p={str(socket_path)!r}; "
        "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.bind(p); os.chmod(p,0o600); "
        "s.listen(1); print('ready',flush=True); time.sleep(10)"
    )
    owner = OwnedPolicyServiceProcess(
        [sys.executable, "-c", program],
        socket_path,
        log_path,
        startup_timeout_s=2,
        shutdown_timeout_s=0.03,
    )
    owner.start()
    pid = owner.pid
    assert pid is not None and socket_path.is_socket()
    owner.stop()
    assert owner.process is None
    assert not socket_path.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert "ready" in log_path.read_text()


def test_startup_crash_reports_exit_status_and_log(tmp_path):
    socket_path = tmp_path / "crash.sock"
    log_path = tmp_path / "crash.log"
    owner = OwnedPolicyServiceProcess(
        [sys.executable, "-c", "print('fatal', flush=True); raise SystemExit(4)"],
        socket_path,
        log_path,
        startup_timeout_s=2,
        shutdown_timeout_s=0.1,
    )
    with pytest.raises(RemotePolicyError, match="status 4"):
        owner.start()
    assert owner.process is None
    assert not socket_path.exists()
    assert "fatal" in log_path.read_text()
