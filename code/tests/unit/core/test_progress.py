"""Console output stays separate from machine results and respects partial failures."""

from io import StringIO

import pytest

from ovlab_core.progress import ConsoleProgress


def test_captured_progress_is_throttled_and_always_records_final_count(monkeypatch):
    monkeypatch.delenv("OVLAB_PROGRESS", raising=False)
    stream = StringIO()
    with ConsoleProgress("Train", total=100, unit="step", stream=stream, min_interval=9999) as progress:
        for index in range(10):
            progress.advance(detail=f"loss={10 - index}")
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert "0/100 step" in lines[0]
    assert "10/100 step" in lines[-1] and "loss=1" in lines[-1]
    assert "\r" not in stream.getvalue() and "\x1b" not in stream.getvalue()


def test_failure_remains_partial_and_does_not_hide_exception():
    stream = StringIO()
    with pytest.raises(RuntimeError, match="training failed"):
        with ConsoleProgress("Train", total=3, stream=stream) as progress:
            progress.advance()
            raise RuntimeError("training failed")
    assert "1/3 item" in stream.getvalue().splitlines()[-1]
    assert "failed" in stream.getvalue().splitlines()[-1]


def test_disabled_progress_and_broken_console_do_not_affect_work(monkeypatch):
    monkeypatch.setenv("OVLAB_PROGRESS", "off")
    stream = StringIO()
    with ConsoleProgress("Eval", total=2, stream=stream) as progress:
        progress.advance(2)
    assert stream.getvalue() == ""
    monkeypatch.setenv("OVLAB_PROGRESS", "on")
    stream.close()
    with ConsoleProgress("Eval", total=2, stream=stream) as progress:
        progress.advance(2)
    assert progress.completed == 2


def test_terminal_tee_updates_one_line_and_keeps_audit_snapshots_plain(monkeypatch):
    monkeypatch.delenv("OVLAB_PROGRESS", raising=False)

    class Terminal(StringIO):
        def isatty(self):
            return True

    class Tee:
        def __init__(self):
            self.console, self.log = Terminal(), StringIO()

        def isatty(self):
            return False

    stream = Tee()
    with ConsoleProgress("QuIC training", total=3, unit="step", stream=stream, min_interval=0) as progress:
        for step in range(3):
            progress.advance(detail=f"loss={step}")
    terminal = stream.console.getvalue()
    assert terminal.count("\r") == 4
    assert terminal.count("\n") == 1
    assert "3/3 step" in terminal and "completed" in terminal
    assert terminal.count("3/3 step") == 1
    snapshots = stream.log.getvalue().splitlines()
    assert len(snapshots) == 2
    assert "0/3 step" in snapshots[0]
    assert "3/3 step" in snapshots[1] and "completed" in snapshots[1]
    assert "\r" not in stream.log.getvalue() and "\x1b" not in stream.log.getvalue()
