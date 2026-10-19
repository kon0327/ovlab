"""Small stderr progress tracker shared by training and rollout workers."""

from __future__ import annotations

import os
import sys
import time
from typing import TextIO


class ConsoleProgress:
    """Track work without adding a terminal dependency or polluting JSON stdout.

    ``OVLAB_PROGRESS=off`` silences progress. The default ``auto`` draws an
    updating line on terminals and writes periodic plain lines in captured logs.
    ``OVLAB_PROGRESS=on`` also emits every update in captured logs.
    """

    def __init__(
        self, description: str, *, total: int | None = None, unit: str = "item",
        stream: TextIO | None = None, min_interval: float | None = None,
    ) -> None:
        if total is not None and total < 0:
            raise ValueError("progress total must be nonnegative")
        self.description = description
        self.total = total
        self.unit = unit
        self.stream = stream if stream is not None else sys.stderr
        self.completed = 0
        self.detail = ""
        self.mode = os.environ.get("OVLAB_PROGRESS", "auto").lower()
        self.enabled = self.mode not in {"off", "0", "false", "no"}
        try:
            self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        except (OSError, ValueError):
            self.interactive = False
        self.min_interval = min_interval if min_interval is not None else (1.0 if self.interactive else 30.0)
        self._started: float | None = None
        self._last_render = 0.0
        self._closed = False

    def start(self) -> "ConsoleProgress":
        if self._started is None:
            self._started = time.monotonic()
            self._render("running", force=True)
        return self

    def update(self, completed: int, *, detail: str = "") -> None:
        """Set the completed count (optimizer steps, batches, or episodes)."""
        if self._closed:
            return
        if completed < self.completed:
            raise ValueError("progress count cannot decrease")
        self.start()
        self.completed = completed
        self.detail = detail
        self._render("running", force=self.total is not None and completed >= self.total)

    def advance(self, count: int = 1, *, detail: str = "") -> None:
        self.update(self.completed + count, detail=detail)

    def close(self, *, status: str = "completed") -> None:
        if self._closed:
            return
        self.start()
        self._render(status, force=True, final=True)
        self._closed = True

    def _render(self, status: str, *, force: bool = False, final: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and self.mode != "on" and now - self._last_render < self.min_interval:
            return
        self._last_render = now
        elapsed = max(0.0, now - (self._started if self._started is not None else now))
        count = str(self.completed) if self.total is None else f"{self.completed}/{self.total}"
        if self.total is None:
            bar = "[" + ("=" * (self.completed % 20)).ljust(20, ".") + "]"
        else:
            fraction = 1.0 if self.total == 0 else min(1.0, self.completed / self.total)
            filled = int(fraction * 20)
            bar = "[" + ("=" * filled).ljust(20, ".") + "]"
        rate = self.completed / elapsed if elapsed else 0.0
        eta = ""
        if self.total is not None and rate > 0 and self.completed < self.total:
            eta = f" eta={max(0, self.total - self.completed) / rate:.0f}s"
        detail = " ".join(self.detail.split())
        line = f"[{self.description}] {bar} {count} {self.unit} elapsed={elapsed:.0f}s{eta} {status}"
        if detail:
            line += f" {detail}"
        try:
            self.stream.write(("\r" if self.interactive else "") + line + ("\n" if final or not self.interactive else "\x1b[K"))
            self.stream.flush()
        except (OSError, ValueError):
            # A detached console must not alter scientific run state/artifacts.
            self.enabled = False

    def __enter__(self) -> "ConsoleProgress":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        status = "aborted" if exc_type is not None and issubclass(exc_type, KeyboardInterrupt) else "failed" if exc_type else "completed"
        self.close(status=status)
