"""Host browser launch behavior without starting a real desktop application."""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ovlab_benchctl import workflow_editor
from ovlab_benchctl.workflow_editor import WorkflowEditorError, open_workflow_editor


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout with spaces"
    directory = root / "code/apps/workflow-builder"
    directory.mkdir(parents=True)
    for name in ("index.html", "styles.css", "catalog.js", "core.js", "dom.js", "app.js"):
        (directory / name).write_text("fixture")
    return root


@pytest.fixture
def fake_openers(tmp_path):
    directory = tmp_path / "fake browser commands"
    directory.mkdir()
    files = {
        "wslpath": '''#!/bin/sh
printf '%s\\n' "$@" > "$OVLAB_TEST_CONVERSION_LOG"
if [ "${OVLAB_TEST_CONVERSION_FAIL:-0}" = 1 ]; then exit 9; fi
printf '%s\\n' "$OVLAB_TEST_WINDOWS_TARGET"
''',
        "powershell.exe": '''#!/bin/sh
printf '%s\\n' "$@" > "$OVLAB_TEST_OPEN_LOG"
exit "${OVLAB_TEST_OPEN_STATUS:-0}"
''',
        "explorer.exe": '''#!/bin/sh
printf '%s\\n' "$@" > "$OVLAB_TEST_OPEN_LOG"
exit "${OVLAB_TEST_OPEN_STATUS:-0}"
''',
        "xdg-open": '''#!/bin/sh
printf '%s\\n' "$@" > "$OVLAB_TEST_OPEN_LOG"
exit "${OVLAB_TEST_OPEN_STATUS:-0}"
''',
    }
    for name, content in files.items():
        script = directory / name
        script.write_text(content)
        script.chmod(0o755)
    return {
        **os.environ,
        "PATH": str(directory) + os.pathsep + os.environ.get("PATH", ""),
        "WSL_DISTRO_NAME": "Ubuntu",
        "OVLAB_TEST_WINDOWS_TARGET": r"\\wsl.localhost\Ubuntu\checkout with spaces\index.html",
        "OVLAB_TEST_CONVERSION_LOG": str(tmp_path / "converted.txt"),
        "OVLAB_TEST_OPEN_LOG": str(tmp_path / "opened.txt"),
    }


def test_wsl_converts_absolute_editor_path_and_opens_windows_browser(checkout, fake_openers):
    result = open_workflow_editor(checkout, environment=fake_openers)
    assert result == {
        "status": "open requested",
        "editor": str(checkout / "code/apps/workflow-builder/index.html"),
        "opener": "Windows default browser",
    }
    assert Path(fake_openers["OVLAB_TEST_CONVERSION_LOG"]).read_text().splitlines() == [
        "-w", str(checkout / "code/apps/workflow-builder/index.html")
    ]
    args = Path(fake_openers["OVLAB_TEST_OPEN_LOG"]).read_text().splitlines()
    assert args[:4] == ["-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand"]
    script = base64.b64decode(args[4]).decode("utf-16le")
    assert "Start-Process -FilePath '" + r"\\wsl.localhost\Ubuntu\checkout with spaces\index.html" + "'" in script


def test_wsl_path_with_powershell_metacharacters_is_a_literal(checkout, fake_openers):
    fake_openers["OVLAB_TEST_WINDOWS_TARGET"] = r"\\wsl.localhost\Ubuntu\O'Brien & $(calc) %PATH%\index.html"
    open_workflow_editor(checkout, environment=fake_openers)
    args = Path(fake_openers["OVLAB_TEST_OPEN_LOG"]).read_text().splitlines()
    script = base64.b64decode(args[4]).decode("utf-16le")
    assert "'" + fake_openers["OVLAB_TEST_WINDOWS_TARGET"].replace("'", "''") + "'" in script


def test_wsl_can_fall_back_to_explorer_if_powershell_is_unavailable(checkout, fake_openers):
    directory = Path(fake_openers["PATH"].split(os.pathsep)[0])
    (directory / "powershell.exe").unlink()
    fake_openers["PATH"] = str(directory)
    assert open_workflow_editor(checkout, environment=fake_openers)["opener"] == "Windows Explorer"
    assert Path(fake_openers["OVLAB_TEST_OPEN_LOG"]).read_text().strip().endswith(r"checkout with spaces\index.html")


def test_explorer_nonzero_exit_is_reported_as_ambiguous(checkout, fake_openers):
    directory = Path(fake_openers["PATH"].split(os.pathsep)[0])
    (directory / "powershell.exe").unlink()
    fake_openers.update(PATH=str(directory), OVLAB_TEST_OPEN_STATUS="1")
    with pytest.raises(WorkflowEditorError, match="may still have opened.*If not, open"):
        open_workflow_editor(checkout, environment=fake_openers)


def test_linux_uses_desktop_opener_without_windows_path_conversion(checkout, fake_openers, monkeypatch):
    monkeypatch.setattr(workflow_editor.platform, "release", lambda: "6.1.0-linux")
    fake_openers["WSL_DISTRO_NAME"] = ""
    result = open_workflow_editor(checkout, environment=fake_openers)
    assert result["opener"] == "xdg-open"
    assert Path(fake_openers["OVLAB_TEST_OPEN_LOG"]).read_text().splitlines() == [result["editor"]]
    assert not Path(fake_openers["OVLAB_TEST_CONVERSION_LOG"]).exists()


def test_launcher_failure_and_missing_files_are_actionable(checkout, fake_openers):
    fake_openers["OVLAB_TEST_OPEN_STATUS"] = "17"
    with pytest.raises(WorkflowEditorError, match="Windows default browser.*exit 17.*open .* in a browser"):
        open_workflow_editor(checkout, environment=fake_openers)
    (checkout / "code/apps/workflow-builder/catalog.js").unlink()
    with pytest.raises(WorkflowEditorError, match="catalog.js.*complete OVLAB checkout"):
        open_workflow_editor(checkout, environment=fake_openers)


def test_failed_path_conversion_does_not_launch_a_browser(checkout, fake_openers):
    fake_openers["OVLAB_TEST_CONVERSION_FAIL"] = "1"
    with pytest.raises(WorkflowEditorError, match="wslpath could not convert"):
        open_workflow_editor(checkout, environment=fake_openers)
    assert not Path(fake_openers["OVLAB_TEST_OPEN_LOG"]).exists()


def test_no_desktop_opener_gives_manual_path(checkout, fake_openers, monkeypatch):
    monkeypatch.setattr(workflow_editor.platform, "release", lambda: "6.1.0-linux")
    monkeypatch.setattr(workflow_editor.shutil, "which", lambda name, path=None: None)
    fake_openers["WSL_DISTRO_NAME"] = ""
    with pytest.raises(WorkflowEditorError, match="no browser opener found") as error:
        open_workflow_editor(checkout, environment=fake_openers)
    assert str(checkout / "code/apps/workflow-builder/index.html") in str(error.value)


def test_long_running_desktop_opener_does_not_keep_cli_open(checkout, fake_openers, monkeypatch):
    class BackgroundOpener:
        def wait(self, *, timeout):
            assert timeout == 1
            raise subprocess.TimeoutExpired("xdg-open", timeout)

    monkeypatch.setattr(workflow_editor.platform, "release", lambda: "6.1.0-linux")
    monkeypatch.setattr(workflow_editor.subprocess, "Popen", lambda *args, **kwargs: BackgroundOpener())
    fake_openers["WSL_DISTRO_NAME"] = ""
    assert open_workflow_editor(checkout, environment=fake_openers)["status"] == "open requested"


def test_installed_command_works_from_another_directory_and_emits_json(tmp_path, checkout, fake_openers):
    bin_dir = tmp_path / "bin"
    env = {**fake_openers, "OVLAB_ROOT": str(checkout), "OVLAB_BIN_DIR": str(bin_dir),
           "OVLAB_PYTHON": sys.executable}
    subprocess.run(["bash", str(ROOT / "install-ovlab-cli.sh")], env=env, check=True, capture_output=True, text=True)
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    result = subprocess.run(["ovlab", "editor", "--json"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "success" and payload["command"] == "editor"
    assert payload["result"]["editor"] == str(checkout / "code/apps/workflow-builder/index.html")
    assert payload["result"]["opener"] == "Windows default browser"
    assert Path(fake_openers["OVLAB_TEST_OPEN_LOG"]).is_file()


def test_help_is_discoverable_and_does_not_start_a_browser(tmp_path, checkout, fake_openers):
    env = {**fake_openers, "OVLAB_ROOT": str(checkout), "OVLAB_PYTHON": sys.executable}
    root = subprocess.run([str(ROOT / "ovlab"), "--help"], cwd=tmp_path, env=env, capture_output=True, text=True)
    sub = subprocess.run([str(ROOT / "ovlab"), "editor", "--help"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert root.returncode == sub.returncode == 0
    assert "editor" in root.stdout and "default browser" in sub.stdout
    assert not Path(fake_openers["OVLAB_TEST_OPEN_LOG"]).exists()


def test_cli_reports_failed_open_in_human_and_json_modes(tmp_path, checkout, fake_openers):
    env = {**fake_openers, "OVLAB_ROOT": str(checkout), "OVLAB_PYTHON": sys.executable,
           "OVLAB_TEST_OPEN_STATUS": "17"}
    for args in (["editor"], ["editor", "--json"]):
        result = subprocess.run([str(ROOT / "ovlab"), *args], cwd=tmp_path, env=env, capture_output=True, text=True)
        assert result.returncode == 6
        if "--json" in args:
            assert json.loads(result.stdout)["errors"][0]["code"] == "editor_unavailable"
        else:
            assert "editor_unavailable" in result.stderr and "Windows default browser" in result.stderr
