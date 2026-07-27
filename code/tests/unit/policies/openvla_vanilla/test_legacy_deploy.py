"""Regressions for the preserved Vanilla deployment harness."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


REPOSITORY = Path(__file__).resolve().parents[5]
HELPER = REPOSITORY / "code/tests/manual/policy_services/openvla/legacy_statistics.py"
SPEC = importlib.util.spec_from_file_location("ovlab_legacy_openvla_statistics", HELPER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_local_hf_snapshot_without_sidecar_preserves_embedded_statistics(tmp_path):
    embedded = {"libero_10": {"action": {"mean": [0.0] * 7}}}
    model = SimpleNamespace(norm_stats=embedded)

    assert MODULE.load_local_dataset_statistics(model, tmp_path) is False
    assert model.norm_stats is embedded


def test_local_training_sidecar_explicitly_overrides_embedded_statistics(tmp_path):
    (tmp_path / "dataset_statistics.json").write_text(
        '{"bridge_orig":{"action":{"mean":[1,1,1,1,1,1,1]}}}',
        encoding="utf-8",
    )
    model = SimpleNamespace(norm_stats={"libero_10": {}})

    assert MODULE.load_local_dataset_statistics(model, tmp_path) is True
    assert tuple(model.norm_stats) == ("bridge_orig",)
