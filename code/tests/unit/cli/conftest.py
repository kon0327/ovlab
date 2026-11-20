"""Materialize a few test-owned local configurations, then remove them."""
from pathlib import Path
import shutil

import pytest

from helpers.evaluation_fixtures import write_evaluation_fixtures


@pytest.fixture(scope="session", autouse=True)
def evaluation_configurations():
    root = Path(__file__).resolve().parents[4]
    destination = root / "configs/.tests"
    # Refuse to overwrite any pre-existing user data or another test worker.
    if destination.exists():
        raise RuntimeError(f"test configuration destination already exists: {destination}")
    try:
        yield write_evaluation_fixtures(root, destination)
    finally:
        if destination.is_dir():
            shutil.rmtree(destination)
