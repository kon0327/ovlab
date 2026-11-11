"""Export verified, immutable OVLAB checkpoint bundles."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

from .training_errors import CheckpointBundleError
from .training_identity import sha256_file, safe_relative
from .training_runs import CheckpointBundleStore


class CheckpointExportError(ValueError):
    """A requested checkpoint export cannot be safely produced."""


def export_checkpoint(model_data_root: Path, checkpoint_id: str, *, format: str, output: Path):
    if format == "onnx":
        raise CheckpointExportError(
            "ONNX export is not yet implemented for complete OpenVLA, OFT or QuIC policies"
        )
    if format not in {"bundle", "tar"}:
        raise CheckpointExportError(f"unsupported checkpoint export format: {format}")
    store = CheckpointBundleStore(model_data_root)
    store.verify(checkpoint_id)
    inspected = store.inspect(checkpoint_id)
    source = Path(inspected["host_path"]).resolve()
    target = Path(output).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise CheckpointExportError(f"export destination already exists: {target}")
    if target.resolve().is_relative_to(source):
        raise CheckpointExportError("export destination must be outside the source checkpoint")
    if not target.parent.is_dir():
        raise CheckpointExportError(f"export parent directory does not exist: {target.parent}")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise CheckpointExportError("checkpoint bundle contains a symbolic link")
    manifest = inspected["manifest"]
    expected_files = {safe_relative(str(item["path"])) for item in manifest["files"]}
    expected_files.add(Path("manifest.json"))
    actual_files = {path.relative_to(source) for path in source.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise CheckpointExportError("checkpoint bundle contains unexpected or missing files")
    try:
        if format == "bundle":
            with tempfile.TemporaryDirectory(prefix=".ovlab-checkpoint-export-", dir=target.parent) as temporary:
                stage = Path(temporary) / checkpoint_id
                shutil.copytree(source, stage)
                for entry in manifest["files"]:
                    path = stage / safe_relative(str(entry["path"]))
                    if not path.is_file() or path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
                        raise CheckpointExportError(f"export verification failed: {entry['path']}")
                if (stage / "manifest.json").read_bytes() != (source / "manifest.json").read_bytes():
                    raise CheckpointExportError("export manifest changed during copying")
                store.verify(checkpoint_id)
                if target.exists() or target.is_symlink():
                    raise CheckpointExportError(f"export destination already exists: {target}")
                original_mode = source.stat().st_mode & 0o7777
                stage.chmod(original_mode | 0o200)
                stage.rename(target)
                target.chmod(original_mode)
        else:
            descriptor, temp_name = tempfile.mkstemp(prefix=".ovlab-checkpoint-export-", dir=target.parent)
            os.close(descriptor)
            stage = Path(temp_name)
            try:
                with tarfile.open(stage, "w") as archive:
                    archive.add(source, arcname=checkpoint_id, recursive=True)
                with tarfile.open(stage, "r") as archive:
                    for entry in manifest["files"]:
                        name = f"{checkpoint_id}/{safe_relative(str(entry['path'])).as_posix()}"
                        member = archive.getmember(name)
                        stream = archive.extractfile(member)
                        if stream is None or not member.isfile() or member.size != entry["size"]:
                            raise CheckpointExportError(f"export verification failed: {name}")
                        digest = hashlib.sha256()
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                        if digest.hexdigest() != entry["sha256"]:
                            raise CheckpointExportError(f"export verification failed: {name}")
                    stream = archive.extractfile(f"{checkpoint_id}/manifest.json")
                    if stream is None or stream.read() != (source / "manifest.json").read_bytes():
                        raise CheckpointExportError("export manifest changed during archiving")
                store.verify(checkpoint_id)
                try:
                    os.link(stage, target)
                except FileExistsError as exc:
                    raise CheckpointExportError(f"export destination already exists: {target}") from exc
            finally:
                stage.unlink(missing_ok=True)
    except CheckpointBundleError:
        raise
    return {
        "schema_version": "ovlab.checkpoint-export/v1", "checkpoint_id": checkpoint_id,
        "format": format, "output": str(target), "source": str(source),
        "kind": inspected["checkpoint"].get("kind"), "integrity": "verified",
    }
