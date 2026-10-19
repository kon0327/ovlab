"""Container console bars stay on one row when other libraries log."""

from io import StringIO
import re

from ovlab_benchctl.training_worker import _ConsoleTee
from ovlab_core.progress import ConsoleProgress


def test_narrow_terminal_progress_and_external_log_do_not_overlap(monkeypatch):
    monkeypatch.setenv("OVLAB_TERMINAL_COLUMNS", "60")
    monkeypatch.delenv("OVLAB_PROGRESS", raising=False)

    class Terminal(StringIO):
        def isatty(self):
            return True

    terminal = Terminal()
    stderr_log, stdout_log, state = StringIO(), StringIO(), {}
    stderr = _ConsoleTee(terminal, stderr_log, state)
    stdout = _ConsoleTee(terminal, stdout_log, state)
    with ConsoleProgress("QuIC training", total=3, unit="step", stream=stderr, min_interval=0) as progress:
        progress.advance(detail="loss=29.5670 VRAM=16.19GiB")
        stdout.write("Constructing tf.data.Dataset\n")
        progress.advance(detail="loss=28.2302 VRAM=16.19GiB")
        progress.advance(detail="loss=25.7526 VRAM=16.20GiB")

    output = terminal.getvalue()
    bars = re.findall(r"\r(\[QuIC training\].*?)\x1b\[K", output)
    assert len(bars) == 4
    assert all(len(bar) <= 59 for bar in bars)
    assert "\r\x1b[2KConstructing tf.data.Dataset\n\r[QuIC training]" in output
    assert stdout_log.getvalue() == "Constructing tf.data.Dataset\n"
    assert "Constructing" not in stderr_log.getvalue()
    assert "3/3 step" in bars[-1] and "completed" in bars[-1]
