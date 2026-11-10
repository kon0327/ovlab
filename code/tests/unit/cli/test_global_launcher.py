import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]


def test_user_launcher_runs_from_other_directory_and_preserves_local_specs(tmp_path):
    bin_dir = tmp_path / "bin"
    env = {**os.environ, "OVLAB_BIN_DIR": str(bin_dir), "OVLAB_PYTHON": sys.executable}
    installer = ["bash", str(ROOT / "install-ovlab-cli.sh")]
    subprocess.run(installer, env=env, check=True, capture_output=True, text=True)
    subprocess.run(installer, env=env, check=True, capture_output=True, text=True)
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    reference = "experiments/workflows/libero-model-tree.json"
    command = ["ovlab", "experiment", "plan", "--spec", reference, "--json"]
    completed = subprocess.run(command, cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)["result"]
    assert result["id"] == "libero-model-tree"
    assert Path(result["spec"]) == ROOT / reference

    local_spec = tmp_path / "local.json"
    local_spec.write_text(json.dumps({"schema_version": "ovlab.experiment/v1", "kind": "workflow", "id": "local", "name": "Local"}))
    completed = subprocess.run(["ovlab", "experiment", "plan", "--spec", "local.json", "--json"],
                               cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)["result"]["spec"] == str(local_spec)


def test_user_launcher_does_not_replace_an_existing_command(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    target = bin_dir / "ovlab"
    target.write_text("existing command\n")
    result = subprocess.run(["bash", str(ROOT / "install-ovlab-cli.sh")],
                            env={**os.environ, "OVLAB_BIN_DIR": str(bin_dir)},
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert target.read_text() == "existing command\n"
    assert not list(bin_dir.glob(".ovlab-launcher.*"))
