"""Read-only registry listing and CLI selection contracts."""

import io
import json
from urllib.error import URLError

import pytest

from ovlab_benchctl import checkpoint_catalog
from ovlab_benchctl.checkpoint_catalog import registry_checkpoints
from ovlab_benchctl.cli import _parser


REVISION = "a" * 40


def test_offline_inventory_checks_cache_without_network_or_mutation(tmp_path, monkeypatch):
    registry = {"checkpoints": {"model": {"repo_id": "org/model", "revision": REVISION,
        "files": {"weights": {"size": 3}}}, "moving": {"repo_id": "org/other", "revision": "remote/main"}}}
    cache = tmp_path / "cache"
    snapshot = cache / "models--org--model" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "weights").write_bytes(b"abc")
    monkeypatch.setattr(checkpoint_catalog, "urlopen", lambda *a, **k: pytest.fail("offline request"))
    env = {"OVLAB_GLOBAL_HF_CACHE": str(cache)}
    result = registry_checkpoints(registry, environment=env, model_data_root=tmp_path / "data")
    assert result["checkpoints"][0]["local_status"] == "present_unverified"
    assert result["checkpoints"][1]["local_status"] == "revision_unknown"
    assert not (tmp_path / "data").exists()
    (snapshot / "weights").write_bytes(b"x")
    assert registry_checkpoints(registry, environment=env, model_data_root=tmp_path)["checkpoints"][0]["local_status"] == "incomplete"


def test_online_resolves_remote_main_and_reports_network_failure(tmp_path, monkeypatch):
    registry = {"checkpoints": {"first": {"repo_id": "org/first", "revision": "remote/main"},
                                "second": {"repo_id": "org/second", "revision": REVISION}}}
    class Response(io.BytesIO):
        pass
    def request(req, timeout):
        if "second" in req.full_url:
            raise URLError("offline")
        assert req.get_header("Authorization") == "Bearer secret"
        return Response(json.dumps({"sha": REVISION}).encode())
    monkeypatch.setattr(checkpoint_catalog, "urlopen", request)
    result = registry_checkpoints(registry, environment={"HF_TOKEN": "secret", "OVLAB_GLOBAL_HF_CACHE": str(tmp_path)},
                                  model_data_root=tmp_path, online=True)
    first, second = result["checkpoints"]
    assert first["remote_status"] == "available" and first["resolved_revision"] == REVISION
    assert second["remote_status"] == "unknown"


def test_checkpoint_list_cli_defaults_and_conflicting_modes():
    parser = _parser()
    assert parser.parse_args(["checkpoint", "list"]).source == "local"
    assert parser.parse_args(["checkpoint", "list", "--source", "registry", "--online"]).online
    with pytest.raises(SystemExit):
        parser.parse_args(["checkpoint", "list", "--online", "--offline"])
