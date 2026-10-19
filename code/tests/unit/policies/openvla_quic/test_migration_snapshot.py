import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[5]


def test_published_submodule_verifies_and_historical_patch_preserves_conflicts(tmp_path):
    spec = importlib.util.spec_from_file_location("prepare_openvla_quic", ROOT / "deploy/scripts/prepare_openvla_quic.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    external = tmp_path / "external"
    external.mkdir()
    for name in ("openvla-quic.migration.patch", "openvla-quic.migration.json"):
        shutil.copyfile(ROOT / "external" / name, external / name)
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(ROOT / "external/openvla-quic"),
                    str(external / "openvla-quic")], check=True, capture_output=True)
    assert helper.prepare(tmp_path)["source"] == "published-submodule"
    assert helper.prepare(tmp_path, apply=True)["status"] == "verified"
    removed = external / "openvla-quic/openvla_quic/wc_provider.py"
    removed.parent.mkdir()
    removed.write_text("# obsolete backend restored by mistake\n")
    with pytest.raises(RuntimeError, match="missing or changed"):
        helper.prepare(tmp_path)
    removed.unlink()
    removed.parent.rmdir()
    source = external / "openvla-quic/quic_petf/runtime.py"
    source.write_text(source.read_text() + "\n# local edit must be preserved\n")
    before = source.read_bytes()
    with pytest.raises(RuntimeError, match="missing or changed"):
        helper.prepare(tmp_path, apply=True)
    assert source.read_bytes() == before
    subprocess.run(["git", "-C", str(external / "openvla-quic"), "restore", "quic_petf/runtime.py"], check=True)
    subprocess.run(["git", "-C", str(external / "openvla-quic"), "checkout", "--quiet",
                    "533b1a76ea600f50738cb9369ed47192a216e2af"], check=True)
    with pytest.raises(RuntimeError, match="--apply"):
        helper.prepare(tmp_path)
    assert helper.prepare(tmp_path, apply=True)["source"] == "historical-patch"
