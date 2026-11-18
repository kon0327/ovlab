"""Open the local, static workflow editor in the user's browser."""

from __future__ import annotations

import base64
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping


class WorkflowEditorError(RuntimeError):
    """The editor files or the host's browser opener are unavailable."""


def open_workflow_editor(repository_root: str | Path, *, environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Request a browser open without starting a server or requiring Docker."""
    environment = dict(os.environ if environment is None else environment)
    root = Path(repository_root).expanduser().resolve()
    editor = root / "code/apps/workflow-builder/index.html"
    required = ("index.html", "styles.css", "catalog.js", "core.js", "dom.js", "app.js")
    missing = [name for name in required if not (editor.parent / name).is_file()]
    if missing:
        raise WorkflowEditorError(
            f"workflow editor files are unavailable in {editor.parent}: {', '.join(missing)}; "
            "use a complete OVLAB checkout or set OVLAB_ROOT"
        )

    def find(name: str) -> str | None:
        return shutil.which(name, path=environment.get("PATH"))

    is_wsl = bool(environment.get("WSL_DISTRO_NAME")) or "microsoft" in platform.release().lower()
    target = str(editor)
    converter = find("wslpath") if is_wsl else None
    powershell = find("powershell.exe") if is_wsl else None
    explorer = find("explorer.exe") if is_wsl else None
    if converter and (powershell or explorer):
        try:
            converted = subprocess.run(
                [converter, "-w", target], env=environment, capture_output=True,
                text=True, check=False, timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorkflowEditorError(f"cannot convert the editor path for Windows: {exc}") from exc
        target = converted.stdout.strip()
        if converted.returncode or not target or "\n" in target or "\r" in target:
            raise WorkflowEditorError("wslpath could not convert the editor path for Windows")
        if powershell:
            # Encode a single-quoted PowerShell literal: paths can contain shell
            # metacharacters or apostrophes, and WSL does not forward arbitrary
            # Linux environment variables to Windows without WSLENV settings.
            literal = "'" + target.replace("'", "''") + "'"
            script = "$ErrorActionPreference = 'Stop'; Start-Process -FilePath " + literal + " -ErrorAction Stop"
            encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
            command = [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]
            opener = "Windows default browser"
        else:
            command = [explorer, target]
            opener = "Windows Explorer"
    elif sys.platform == "darwin":
        opener_path = find("open")
        if not opener_path:
            raise WorkflowEditorError(f"the macOS 'open' command is unavailable; open {editor} in a browser")
        command = [opener_path, target]
        opener = "macOS open"
    elif sys.platform == "win32":
        try:
            os.startfile(target)  # type: ignore[attr-defined]
        except OSError as exc:
            raise WorkflowEditorError(f"could not open the editor in the default browser: {exc}") from exc
        return {"status": "open requested", "editor": str(editor), "opener": "Windows default browser"}
    else:
        opener_path = find("xdg-open")
        if not opener_path:
            raise WorkflowEditorError(f"no browser opener found; open {editor} in a browser")
        command = [opener_path, target]
        opener = "xdg-open"

    try:
        process = subprocess.Popen(
            command, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            status = process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            # Some desktop openers stay attached to the browser. Do not keep the
            # CLI open for the lifetime of the user's browser session.
            status = None
        if status not in (None, 0):
            if opener == "Windows Explorer":
                raise WorkflowEditorError(
                    f"Windows Explorer returned exit {status}; it may still have opened the editor. "
                    f"If not, open {target} in a browser"
                )
            raise WorkflowEditorError(
                f"{opener} could not open the editor (exit {status}); open {target} in a browser"
            )
    except OSError as exc:
        raise WorkflowEditorError(f"cannot start {opener}: {exc}; open {target} in a browser") from exc
    return {"status": "open requested", "editor": str(editor), "opener": opener}
