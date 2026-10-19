"""Small stderr progress tracker shared by training and rollout workers."""

from __future__ import annotations

import os
import shutil
import sys
import time
from typing import TextIO


class ConsoleProgress:
    """Track work without adding a terminal dependency or polluting JSON stdout.

    ``OVLAB_PROGRESS=off`` silences progress. The default ``auto`` updates one
    terminal line and writes periodic plain snapshots to a worker's log.
    ``OVLAB_PROGRESS=on`` also records every update in captured logs.
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
        # A worker may tee stderr to its evidence log. Its isatty() is false so
        # ordinary writes stay readable there; live progress targets the actual
        # console while periodic snapshots go to the log, as in QuIC Progress.
        self.console = getattr(self.stream, "console", self.stream)
        self.log = getattr(self.stream, "log", None)
        self.completed = 0
        self.detail = ""
        self.mode = os.environ.get("OVLAB_PROGRESS", "auto").lower()
        self.enabled = self.mode not in {"off", "0", "false", "no"}
        try:
            self.interactive = bool(getattr(self.console, "isatty", lambda: False)())
        except (OSError, ValueError):
            self.interactive = False
        self.min_interval = min_interval if min_interval is not None else (0.5 if self.interactive else 30.0)
        self._started: float | None = None
        self._last_render = 0.0
        self._last_snapshot = 0.0
        self._closed = False
        self._displayed = False
        self._external_unterminated = False

    def _terminal_columns(self) -> int:
        configured = os.environ.get("OVLAB_TERMINAL_COLUMNS", "")
        try:
            columns = int(configured) if configured else shutil.get_terminal_size((80, 24)).columns
        except ValueError:
            columns = 80
        return columns if columns > 1 else 80

    def prepare_external_write(self, value: str) -> None:
        """Give another console writer a clean line while a bar is active."""
        if not value or not self.interactive or not self.enabled or self._closed:
            return
        try:
            if self._displayed:
                self.console.write("\r\x1b[2K")
                self.console.flush()
                self._displayed = False
            self._external_unterminated = not value.endswith("\n")
        except (OSError, ValueError):
            self.enabled = False

    def start(self) -> "ConsoleProgress":
        if self._started is None:
            self._started = time.monotonic()
            register = getattr(self.stream, "register_progress", None)
            if self.interactive and callable(register):
                register(self)
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
        # close() draws the final count and status. Avoid a redundant full bar
        # marked "running" immediately before the completed line.
        if self.total is None or completed < self.total:
            self._render("running")

    def advance(self, count: int = 1, *, detail: str = "") -> None:
        self.update(self.completed + count, detail=detail)

    def close(self, *, status: str = "completed") -> None:
        if self._closed:
            return
        self.start()
        self._render(status, force=True, final=True)
        self._closed = True
        unregister = getattr(self.stream, "unregister_progress", None)
        if self.interactive and callable(unregister):
            unregister(self)

    def _render(self, status: str, *, force: bool = False, final: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        draw = force or self.mode == "on" or now - self._last_render >= self.min_interval
        snapshot = force or self.mode == "on" or now - self._last_snapshot >= 30.0
        if not draw and (not self.interactive or self.log is None or not snapshot):
            return
        elapsed = max(0.0, now - (self._started if self._started is not None else now))
        count = str(self.completed) if self.total is None else f"{self.completed}/{self.total}"
        fraction = None if self.total is None else (1.0 if self.total == 0 else min(1.0, self.completed / self.total))
        def bar(width: int) -> str:
            filled = self.completed % width if fraction is None else int(fraction * width)
            return "[" + ("=" * filled).ljust(width, ".") + "]"
        rate = self.completed / elapsed if elapsed else 0.0
        eta = ""
        if self.total is not None and rate > 0 and self.completed < self.total:
            eta = f" eta={max(0, self.total - self.completed) / rate:.0f}s"
        detail = " ".join(self.detail.split())
        line = f"[{self.description}] {bar(20)} {count} {self.unit} elapsed={elapsed:.0f}s{eta} {status}"
        if detail:
            line += f" {detail}"
        if self.interactive:
            limit = self._terminal_columns() - 1
            if len(line) > limit:
                # A Docker PTY can report 0 columns. Keep the bar on one
                # physical row so carriage return really replaces it.
                state = " " + status if final else ""
                for width in (20, 12, 8):
                    candidate = f"[{self.description}] {bar(width)} {count} {self.unit}{eta}{state}"
                    if detail:
                        candidate += f" {detail}"
                    line = candidate
                    if len(line) <= limit:
                        break
                if len(line) > limit:
                    line = line[:limit - 1].rstrip() + "…"
        try:
            if self.interactive:
                if draw:
                    if self._external_unterminated:
                        self.console.write("\n")
                        self._external_unterminated = False
                    self.console.write("\r" + line + "\x1b[K" + ("\n" if final else ""))
                    self.console.flush()
                    self._last_render = now
                    self._displayed = not final
                if self.log is not None and snapshot:
                    self.log.write(line + "\n")
                    self.log.flush()
                    self._last_snapshot = now
            elif draw:
                self.stream.write(line + "\n")
                self.stream.flush()
                self._last_render = now
        except (OSError, ValueError):
            # A detached console must not alter scientific run state/artifacts.
            self.enabled = False

    def __enter__(self) -> "ConsoleProgress":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        status = "aborted" if exc_type is not None and issubclass(exc_type, KeyboardInterrupt) else "failed" if exc_type else "completed"
        self.close(status=status)
