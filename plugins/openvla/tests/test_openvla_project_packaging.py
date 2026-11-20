"""The integration is distributable without model code in OVLAB core."""
from email.parser import Parser
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]


def test_plugin_wheel_contains_all_adapters_without_legacy_distribution_dependencies(tmp_path):
    import shutil
    project = tmp_path / "project"
    shutil.copytree(PROJECT, project, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', 'build', 'dist'))
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--disable-pip-version-check", "--wheel-dir", str(tmp_path), str(project)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = tmp_path.glob("ovlab_openvla_plugin-*.whl")
    with ZipFile(wheel) as package:
        members = set(package.namelist())
        for namespace in ("common", "vanilla", "lora_merged", "oft", "quic", "training"):
            assert f"ovlab_openvla_{namespace}/__init__.py" in members
        assert not any(name.startswith(("ovlab_core/", "ovlab_benchctl/")) for name in members)
        project_assets = ".data/data/share/ovlab/plugins/openvla/"
        for required in ("plugin.json", "pyproject.toml", "setup.py", "docker/Dockerfile.openvla",
                         "locks/openvla.requirements.txt", "scripts/install-plugin.sh",
                         "configs/resources/registry.yaml", "src/ovlab_openvla_vanilla/plugin.py"):
            assert any(project_assets + required in member for member in members)
        metadata_name, = (name for name in members if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(package.read(metadata_name).decode())
    assert metadata["Name"] == "ovlab-openvla-plugin"
    assert metadata["Version"] == json.loads((PROJECT / "plugin.json").read_text())["version"]
    requirements = metadata.get_all("Requires-Dist", [])
    assert all(not value.startswith("ovlab-openvla-") for value in requirements)
    assert all(not value.startswith("ovlab-benchctl") for value in requirements)

    target = tmp_path / "installed"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile",
                    "--disable-pip-version-check", "--target", str(target), str(wheel)],
                   check=True, text=True, capture_output=True)
    result = subprocess.run(
        [sys.executable, "-c", "from importlib.metadata import entry_points; "
         "entry = next(e for e in entry_points(group='ovlab.plugin_projects') if e.name == 'openvla'); "
         "print(entry.load()())"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(target)},
        check=True, text=True, capture_output=True,
    )
    installed_project = Path(result.stdout.strip())
    assert (installed_project / "plugin.json").is_file()
    assert (installed_project / "configs/resources/registry.yaml").is_file()
    assert (installed_project / "src/ovlab_openvla_vanilla/plugin.py").is_file()


def test_plugin_owns_model_builds_and_core_installer_stays_generic():
    manifest = json.loads((PROJECT / "plugin.json").read_text())
    assert manifest["schema_version"] == "ovlab.plugin/v1"
    for recipe in manifest["builds"].values():
        dockerfile = PROJECT / recipe["dockerfile"]
        assert dockerfile.is_file()
        text = dockerfile.read_text()
        assert "COPY external/plugin-openvla /opt/ovlab/external/plugin-openvla" in text
        assert "bash /opt/ovlab/external/plugin-openvla/scripts/install-plugin.sh" in text
        assert "ovlab_plugin_registry" in text
        for lock in recipe["locks"]:
            assert (PROJECT / lock).is_file()
    core_installer = (ROOT / "deploy/docker/install-ovlab-packages.sh").read_text()
    assert "code/policies" not in core_installer
    assert "openvla" not in core_installer
    for pyproject in (ROOT / "code").glob("**/pyproject.toml"):
        assert "ovlab-openvla-" not in pyproject.read_text()


def test_plugin_uses_public_contracts_without_importing_core_cli_internals():
    for path in (PROJECT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [
                alias.name for alias in node.names
            ] if isinstance(node, ast.Import) else []
            assert not any(name == "ovlab_benchctl" or name.startswith("ovlab_benchctl.") for name in names), path
