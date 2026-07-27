"""Ownership and cleanup for a local policy-service subprocess."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from ovlab_remote_policy.errors import RemotePolicyError, RemotePolicyTimeoutError


class OwnedPolicyServiceProcess:
    """Start, log, terminate, reap, and clean one owned local service."""

    def __init__(
        self,
        command: list[str],
        socket_path: str | Path,
        log_path: str | Path,
        *,
        environment: dict[str, str] | None = None,
        startup_timeout_s: float = 180.0,
        shutdown_timeout_s: float = 10.0,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        if startup_timeout_s <= 0 or shutdown_timeout_s <= 0:
            raise ValueError("process timeouts must be positive")
        self.command = list(command)
        self.socket_path = Path(socket_path)
        self.log_path = Path(log_path)
        self.environment = None if environment is None else dict(environment)
        self.startup_timeout_s = float(startup_timeout_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self.process: subprocess.Popen | None = None
        self._log = None

    def start(self) -> None:
        if self.process is not None:
            raise RemotePolicyError("policy service process was already started")
        self.socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.socket_path.parent, 0o700)
        if self.socket_path.exists():
            raise RemotePolicyError(f"refusing to overwrite existing policy socket: {self.socket_path}")
        self.log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._log = self.log_path.open("xb")
        env = os.environ.copy()
        if self.environment is not None:
            env.update(self.environment)
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout_s
        try:
            while time.monotonic() < deadline:
                return_code = self.process.poll()
                if return_code is not None:
                    raise RemotePolicyError(
                        f"policy service exited during startup with status {return_code}; log: {self.log_path}"
                    )
                if self.socket_path.is_socket():
                    mode = self.socket_path.stat().st_mode & 0o777
                    if mode != 0o600:
                        raise RemotePolicyError(
                            f"policy socket permissions are {mode:o}; expected 600"
                        )
                    return
                time.sleep(0.05)
            raise RemotePolicyTimeoutError(
                f"policy service startup timed out after {self.startup_timeout_s:g}s; log: {self.log_path}"
            )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        process = self.process
        self.process = None
        try:
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=self.shutdown_timeout_s)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=self.shutdown_timeout_s)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=self.shutdown_timeout_s)
            elif process is not None:
                process.wait()
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None
            try:
                self.socket_path.unlink(missing_ok=True)
            except OSError:
                pass

    @property
    def pid(self) -> int | None:
        return None if self.process is None else self.process.pid

