"""The reporting container sees canonical runs and historical runs read-only."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("layout", ["canonical", "legacy", "both"])
@pytest.mark.parametrize("verify", [False, True])
def test_report_launcher_mounts_available_training_roots_and_honors_overrides(tmp_path, layout, verify):
    repository = Path(__file__).resolve().parents[4]
    model_data = tmp_path / "model data"
    runs = tmp_path / "custom runs"
    legacy = model_data / "training-runs"
    exports = model_data / "exports"
    (exports / "derived").mkdir(parents=True)
    if layout in {"canonical", "both"}:
        (runs / "train/run-fixture").mkdir(parents=True)
    if layout in {"legacy", "both"}:
        (legacy / "run-fixture").mkdir(parents=True)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(f"#!{sys.executable}\n" + """import json, sys
if sys.argv[1:3] == ['image', 'inspect']:
    sys.exit(0)
print(json.dumps(sys.argv[1:]))
""")
    docker.chmod(0o755)
    environment = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
                   "OVLAB_MODEL_DATA_ROOT": str(model_data), "OVLAB_RUNS_ROOT": str(runs),
                   "OVLAB_DATA_ROOT": str(tmp_path / "other-data"), "OVLAB_REPORTING_RUNTIME": "docker"}
    environment.pop("OVLAB_EXPORTS_ROOT", None)
    completed = subprocess.run(
        [str(repository / "ovlab"), "train", "report", "--run", "run-fixture", "--json", *(["--verify"] if verify else [])],
        env=environment, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    command = json.loads(completed.stdout)
    for root, target, present in (
        (runs, "/var/lib/ovlab/runs", layout in {"canonical", "both"}),
        (legacy, "/var/lib/ovlab/model-data/training-runs", layout in {"legacy", "both"}),
    ):
        assert (f"type=bind,source={root},target={target},readonly" in command) == present
    assert "OVLAB_RUNS_ROOT=/var/lib/ovlab/runs" in command
    assert f"type=bind,source={exports},target=/var/lib/ovlab/exports" + (",readonly" if verify else "") in command
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--gpus" not in command
