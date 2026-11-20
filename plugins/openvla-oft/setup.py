"""Include the independent project alongside its importable Python packages."""
from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent
assets = [ROOT / name for name in ("plugin.json", "README.md", "pyproject.toml", "setup.py", "MANIFEST.in")]
for directory in ("configs", "examples", "docker", "locks", "scripts", "src"):
    for path in (ROOT / directory).rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_file() and not any(part == "__pycache__" or part.endswith(".egg-info") for part in relative.parts):
            if path.suffix not in {".pyc", ".pyo"}:
                assets.append(path)
groups = {}
for path in sorted(assets):
    target = Path("share/ovlab/plugins/openvla-oft") / path.relative_to(ROOT).parent
    groups.setdefault(target.as_posix(), []).append(path.relative_to(ROOT).as_posix())

setup(data_files=list(groups.items()))
