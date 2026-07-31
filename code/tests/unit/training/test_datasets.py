from __future__ import annotations

import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import threading
import zipfile

import pytest

from ovlab_benchctl.datasets import (
    DatasetBridgeRegistry,
    DatasetRequest,
    DatasetStore,
    LiberoDatasetBridge,
    LocalDatasetBridge,
    UrlDatasetBridge,
)
from ovlab_benchctl.training_errors import (
    DatasetIntegrityError,
    DatasetInterruptedError,
    DatasetRequestError,
    DatasetSecurityError,
    DatasetUnavailableError,
)


def test_libero_resolution_is_pinned_deterministic_and_openvla_compatible(tmp_path):
    store = DatasetStore(tmp_path)
    first = store.resolve(DatasetRequest(source="libero", name="libero_10"))
    second = store.resolve(DatasetRequest(source="libero", name="libero_10"))
    assert first == second
    assert first.resolution_id == second.resolution_id
    assert first.source_revision == "a7c9ae18499b6eea8a32f78a9302327b752b1b5f"
    assert first.source_metadata["dataset_directory"] == "libero_10_no_noops/1.0.0"
    assert first.preparation_format == "openvla-rlds"
    assert store._storage_version(first) == "1.0.0"


def test_libero_resolution_rejects_unknown_suite_with_supported_list():
    with pytest.raises(DatasetRequestError, match="supported suites: libero_spatial"):
        LiberoDatasetBridge().resolve(DatasetRequest(source="libero", name="libero_unknown"))


def test_libero_retry_reuses_verified_failed_staging_without_redownload(monkeypatch, tmp_path):
    bridge = LiberoDatasetBridge()
    request = DatasetRequest(source="libero", name="libero_10")
    resolution = bridge.resolve(request)
    prefix = str(resolution.source_metadata["dataset_directory"])
    payloads = {
        "dataset_info.json": b'{"splits":[{"shardLengths":[2]}]}',
        "features.json": b'{"features":{}}',
        "libero_10_no_noops-train.tfrecord-00000-of-00001": b"two-samples",
    }
    entries = [
        {
            "type": "file",
            "path": f"{prefix}/{name}",
            "size": len(content),
            "lfs": {"oid": hashlib.sha256(content).hexdigest()},
        }
        for name, content in payloads.items()
    ]
    monkeypatch.setattr(bridge, "_tree", lambda _resolution: entries)
    monkeypatch.setattr(
        bridge, "_download",
        lambda *_args, **_kwargs: pytest.fail("verified staged bytes must not be downloaded again"),
    )
    store = DatasetStore(tmp_path / "model-data", DatasetBridgeRegistry((bridge,)))
    staging = store.root / ".staging" / f"{resolution.resolution_id}-failed"
    raw = staging / "raw"
    raw.mkdir(parents=True)
    for name, content in payloads.items():
        (raw / name).write_bytes(content)
    (staging / "state.json").write_text(
        '{"state":"failed","resolution_id":"' + resolution.resolution_id + '"}\n',
        encoding="utf-8",
    )
    messages: list[str] = []

    result = store.fetch(request, allow_download=True, progress=messages.append)

    assert result["state"] == "ready"
    assert result["sample_count"] == 2
    assert Path(result["host_path"]).relative_to(store.root).parts == ("libero", "libero_10", "1.0.0")
    assert any(message.startswith("resuming verified staging") for message in messages)
    assert sum(message.startswith("reused ") for message in messages) == len(payloads)
    assert not staging.exists()
    assert store.verify(str(result["dataset_id"]))["status"] == "verified"


def test_libero_prepare_rejects_incomplete_shard_set(tmp_path):
    bridge = LiberoDatasetBridge()
    acquired = tmp_path / "raw"
    acquired.mkdir()
    (acquired / "dataset_info.json").write_text('{"splits":[]}', encoding="utf-8")
    (acquired / "features.json").write_text("{}", encoding="utf-8")
    (acquired / "data-train.tfrecord-00000-of-00002").write_bytes(b"only-one")

    with pytest.raises(DatasetIntegrityError, match="incomplete TFRecord shard set"):
        bridge.prepare(
            bridge.resolve(DatasetRequest(source="libero", name="libero_10")),
            acquired,
            tmp_path / "prepared",
        )


@pytest.mark.parametrize("url", [
    "http://example.org/data.zip",
    "https://user:secret@example.org/data.zip",
    "file:///tmp/data.zip",
])
def test_url_bridge_rejects_insecure_or_credentialed_sources(url):
    digest = "a" * 64
    with pytest.raises(DatasetSecurityError):
        UrlDatasetBridge().resolve(DatasetRequest(source="url", name="data", url=url, sha256=digest))


def test_url_bridge_requires_digest():
    with pytest.raises(DatasetRequestError, match="--url and --sha256"):
        UrlDatasetBridge().resolve(DatasetRequest(source="url", name="data", url="https://example.org/data.zip"))


def test_archive_traversal_and_symlink_are_rejected(tmp_path):
    bridge = UrlDatasetBridge()
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escaped", b"bad")
    destination = tmp_path / "output"
    destination.mkdir()
    with pytest.raises((DatasetSecurityError, ValueError)):
        bridge._extract(archive, "zip", destination)
    assert not (tmp_path / "escaped").exists()


def _local_source(root: Path):
    root.mkdir()
    (root / "features.json").write_text('{"action":"float32[7]"}', encoding="utf-8")
    (root / "part-0000.bin").write_bytes(b"immutable-samples")


def test_local_import_is_immutable_path_independent_and_detects_tampering(tmp_path):
    source = tmp_path / "source"
    _local_source(source)
    first = DatasetStore(tmp_path / "first").import_local(DatasetRequest(
        source="local", name="fixture", version="1", local_path=source,
    ))
    second = DatasetStore(tmp_path / "second").import_local(DatasetRequest(
        source="local", name="fixture", version="1", local_path=source,
    ))
    assert first["dataset_id"] == second["dataset_id"]
    assert first["raw_content_digest"] == second["raw_content_digest"]
    verified = DatasetStore(tmp_path / "first").verify(first["dataset_id"])
    assert verified["status"] == "verified"
    copied = Path(first["host_path"]) / "prepared" / "part-0000.bin"
    copied.chmod(0o644)
    copied.write_bytes(b"tampered")
    with pytest.raises(DatasetIntegrityError, match="integrity verification"):
        DatasetStore(tmp_path / "first").verify(first["dataset_id"])


def test_local_import_rejects_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "outside"
    target.write_text("outside", encoding="utf-8")
    (source / "link").symlink_to(target)
    with pytest.raises(DatasetSecurityError, match="symlink"):
        DatasetStore(tmp_path / "data").import_local(DatasetRequest(
            source="local", name="fixture", version="1", local_path=source,
        ))


def test_download_cannot_be_triggered_without_explicit_authorization(tmp_path):
    request = DatasetRequest(
        source="url", name="fixture", url="https://example.org/data.bin", sha256="a" * 64,
    )
    with pytest.raises(DatasetUnavailableError, match="--allow-dataset-download"):
        DatasetStore(tmp_path).fetch(request, allow_download=False)


def test_registry_unknown_provider_lists_extension_surface():
    with pytest.raises(DatasetRequestError, match="libero, local, url"):
        DatasetBridgeRegistry().get("guess")


def test_bounded_local_http_fixture_acquires_verifies_reuses_and_sanitizes(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    payload = served / "samples.bin"
    payload.write_bytes(b"bounded-http-dataset")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()

    class QuietHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(served), **kwargs)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = DatasetRequest(
            source="url", name="fixture", version="1",
            url=f"http://127.0.0.1:{server.server_port}/samples.bin?temporary=secret",
            sha256=digest, archive="none", allow_local_http=True,
        )
        store = DatasetStore(tmp_path / "model-data")
        first = store.fetch(request, allow_download=True)
        second = store.fetch(request, allow_download=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert first["dataset_id"] == second["dataset_id"]
    assert second["reused"] is True
    assert store.verify(first["dataset_id"])["status"] == "verified"
    serialized = (Path(first["host_path"]) / "manifest.json").read_text(encoding="utf-8")
    assert "temporary=secret" not in serialized
    assert first["acquisition"]["final_origin"].endswith("/samples.bin")


def test_checksum_mismatch_is_rejected(tmp_path):
    bridge = UrlDatasetBridge()
    request = DatasetRequest(
        source="url", name="fixture", url="https://example.org/samples.bin",
        sha256="a" * 64, archive="none",
    )
    acquired = tmp_path / "acquired"
    acquired.mkdir()
    (acquired / "samples.bin").write_bytes(b"wrong-content")
    with pytest.raises(DatasetIntegrityError, match="checksum mismatch"):
        bridge.verify(bridge.resolve(request), acquired)


def test_interrupted_local_import_preserves_evidence_releases_lock_and_never_publishes(monkeypatch, tmp_path):
    source = tmp_path / "source"
    _local_source(source)
    bridge = LocalDatasetBridge()
    original = bridge.import_path

    def interrupt(_source, destination):
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "partial.bin").write_bytes(b"partial")
        raise KeyboardInterrupt()

    monkeypatch.setattr(bridge, "import_path", interrupt)
    store = DatasetStore(tmp_path / "data", DatasetBridgeRegistry((bridge,)))
    request = DatasetRequest(source="local", name="fixture", version="1", local_path=source)
    with pytest.raises(DatasetInterruptedError, match="interrupted"):
        store.import_local(request)
    assert store.list() == []
    states = list((tmp_path / "data/datasets/.staging").glob("*/state.json"))
    assert len(states) == 1 and '"state":"interrupted"' in states[0].read_text(encoding="utf-8")

    monkeypatch.setattr(bridge, "import_path", original)
    completed = store.import_local(request)
    assert store.verify(completed["dataset_id"])["status"] == "verified"


def test_changing_local_dataset_bytes_changes_immutable_identity(tmp_path):
    first_source = tmp_path / "source-one"
    second_source = tmp_path / "source-two"
    _local_source(first_source)
    _local_source(second_source)
    (second_source / "part-0000.bin").write_bytes(b"different-samples")
    first = DatasetStore(tmp_path / "one").import_local(DatasetRequest(
        source="local", name="fixture", version="1", local_path=first_source,
    ))
    second = DatasetStore(tmp_path / "two").import_local(DatasetRequest(
        source="local", name="fixture", version="1", local_path=second_source,
    ))
    assert first["dataset_id"] != second["dataset_id"]


def test_dataset_version_path_rejects_a_different_immutable_identity(tmp_path):
    first_source = tmp_path / "source-one"
    second_source = tmp_path / "source-two"
    _local_source(first_source)
    _local_source(second_source)
    (second_source / "part-0000.bin").write_bytes(b"different-samples")
    store = DatasetStore(tmp_path / "data")

    first = store.import_local(DatasetRequest(
        source="local", name="fixture", version="1.0.0", local_path=first_source,
    ))
    assert Path(first["host_path"]).relative_to(store.root).parts == ("local", "fixture", "1.0.0")

    with pytest.raises(DatasetIntegrityError, match="already bound to a different immutable identity"):
        store.import_local(DatasetRequest(
            source="local", name="fixture", version="1.0.0", local_path=second_source,
        ))


def test_legacy_revision_build_layout_remains_discoverable(tmp_path):
    source = tmp_path / "source"
    _local_source(source)
    store = DatasetStore(tmp_path / "data")
    result = store.import_local(DatasetRequest(
        source="local", name="fixture", version="1", local_path=source,
    ))
    canonical = Path(result["host_path"])
    legacy = store.root / "local" / "fixture" / "1" / result["dataset_id"].split("-", 1)[1][:16]
    canonical.parent.chmod(0o755)
    temporary = canonical.parent / ".legacy-publication"
    os.replace(canonical, temporary)
    legacy.parent.mkdir(parents=True)
    shutil.copytree(temporary, legacy)
    temporary.chmod(0o755)
    (temporary / "manifest.json").unlink()

    inspected = store.inspect(result["dataset_id"])
    assert inspected["host_path"] == str(legacy)
    assert inspected["dataset_version"] == "1"
    assert store.list()[0]["dataset_version"] == "1"
    assert store.verify(result["dataset_id"])["status"] == "verified"
