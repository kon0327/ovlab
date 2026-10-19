import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[5]


def test_reviewed_patch_reconstructs_a_pristine_pinned_submodule_and_preserves_conflicts(tmp_path):
    spec = importlib.util.spec_from_file_location("prepare_openvla_quic", ROOT / "deploy/scripts/prepare_openvla_quic.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    external = tmp_path / "external"
    external.mkdir()
    for name in ("openvla-quic.migration.patch", "openvla-quic.migration.json"):
        shutil.copyfile(ROOT / "external" / name, external / name)
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(ROOT / "external/openvla-quic"),
                    str(external / "openvla-quic")], check=True, capture_output=True)
    with pytest.raises(RuntimeError, match="--apply"):
        helper.prepare(tmp_path)
    assert helper.prepare(tmp_path, apply=True)["status"] == "verified"
    assert helper.prepare(tmp_path, apply=True)["status"] == "verified"
    source = external / "openvla-quic/openvla_quic/peft_provider.py"
    source.write_text(source.read_text() + "\n# local edit must be preserved\n")
    before = source.read_bytes()
    with pytest.raises(RuntimeError, match="does not apply cleanly"):
        helper.prepare(tmp_path, apply=True)
    assert source.read_bytes() == before
