"""Pinned LIBERO RLDS source provider; immutable publication belongs to OVLAB."""
import json, os, re
from pathlib import Path
from typing import Mapping
from urllib.parse import quote, urlencode
from ovlab_core.datasets import DatasetProvider as DatasetBridge, DatasetRequest, DatasetResolution, DatasetProgress as Progress, DatasetRequestError, DatasetIntegrityError, DatasetSecurityError, DatasetUnavailableError
from ovlab_core.dataset_io import _HttpMixin
from ovlab_core.artifact_identity import SHA256_RE, atomic_json, inventory, safe_relative, sha256_file
LIBERO_BRIDGE_VERSION = "1.0.0"
DATASET_SOURCE_SCHEMA = "ovlab.dataset-source/v1"

class LiberoDatasetBridge(_HttpMixin, DatasetBridge):
    supports_resume = True
    bridge_id = "libero"
    bridge_version = LIBERO_BRIDGE_VERSION
    repository = "openvla/modified_libero_rlds"
    revision = "a7c9ae18499b6eea8a32f78a9302327b752b1b5f"
    suites = {
        "libero_spatial": "libero_spatial_no_noops",
        "libero_object": "libero_object_no_noops",
        "libero_goal": "libero_goal_no_noops",
        "libero_10": "libero_10_no_noops",
    }
    _shard_name = re.compile(r"^.+\.tfrecord-(\d{5})-of-(\d{5})$")

    def publication_version(self, resolution):
        return safe_relative(resolution.source_metadata['dataset_directory']).parts[-1]

    def capabilities(self) -> Mapping[str, object]:
        return {"source": "libero", "suites": tuple(self.suites), "preparation_formats": ("openvla-rlds",)}

    def resolve(self, request: DatasetRequest) -> DatasetResolution:
        if request.name not in self.suites:
            raise DatasetRequestError(
                f"unknown LIBERO suite {request.name!r}; supported suites: {', '.join(self.suites)}"
            )
        folder = self.suites[request.name]
        return DatasetResolution(
            provider="libero",
            logical_name=request.name,
            source_revision=self.revision,
            source_type="huggingface-dataset-snapshot",
            canonical_locator=f"https://huggingface.co/datasets/{self.repository}/tree/{self.revision}/{folder}/1.0.0",
            bridge_id=self.bridge_id,
            bridge_version=self.bridge_version,
            expected_digest=None,
            preparation_format="openvla-rlds",
            source_metadata={
                "repository": self.repository,
                "snapshot_revision": self.revision,
                "dataset_directory": f"{folder}/1.0.0",
                "tfds_name": folder,
                "libero_revision": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
                "license": "MIT",
                "citation": "OpenVLA (arXiv:2406.09246) and LIBERO",
            },
        )

    def _tree(self, resolution: DatasetResolution) -> list[dict[str, object]]:
        prefix = str(resolution.source_metadata["dataset_directory"])
        endpoint = (
            f"https://huggingface.co/api/datasets/{self.repository}/tree/"
            f"{self.revision}/{prefix}?" + urlencode({"recursive": "true", "expand": "true"})
        )
        entries: list[dict[str, object]] = []
        next_url: str | None = endpoint
        while next_url:
            with self._open(next_url) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, list):
                    raise DatasetUnavailableError("Hugging Face dataset tree response is not a list")
                entries.extend(item for item in payload if isinstance(item, dict) and item.get("type") == "file")
                link = response.headers.get("Link", "")
                next_url = None
                for part in link.split(","):
                    if 'rel="next"' in part and "<" in part and ">" in part:
                        next_url = part.split("<", 1)[1].split(">", 1)[0]
                        break
        if not entries:
            raise DatasetUnavailableError("pinned LIBERO dataset snapshot contains no files")
        return entries

    def acquire(self, resolution: DatasetResolution, destination: Path, progress: Progress | None = None) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or not destination.is_dir():
            raise DatasetSecurityError("LIBERO staging destination must be a real directory")
        prefix = str(resolution.source_metadata["dataset_directory"]).rstrip("/") + "/"
        expected_paths: set[Path] = set()
        for entry in self._tree(resolution):
            remote_path = str(entry.get("path", ""))
            if not remote_path.startswith(prefix):
                raise DatasetSecurityError(f"provider returned a file outside requested LIBERO suite: {remote_path}")
            relative = safe_relative(remote_path[len(prefix):])
            if relative in expected_paths:
                raise DatasetIntegrityError(f"provider returned duplicate LIBERO path: {relative}")
            expected_paths.add(relative)
            lfs = entry.get("lfs", {}) if isinstance(entry.get("lfs"), dict) else {}
            expected = lfs.get("oid") if isinstance(lfs.get("oid"), str) else None
            if isinstance(expected, str) and expected.startswith("sha256:"):
                expected = expected[7:]
            if expected is not None and SHA256_RE.fullmatch(expected) is None:
                expected = None
            target = destination / relative
            partial = target.with_name(target.name + ".partial")
            if partial.is_symlink() or (partial.exists() and not partial.is_file()):
                raise DatasetSecurityError(f"unsafe partial LIBERO staging path: {relative}")
            partial.unlink(missing_ok=True)
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise DatasetSecurityError(f"unsafe existing LIBERO staging path: {relative}")
            expected_size = entry.get("size")
            size_matches = not isinstance(expected_size, int) or (
                target.is_file() and target.stat().st_size == expected_size
            )
            if target.is_file() and expected is not None and size_matches and sha256_file(target) == expected:
                if progress is not None:
                    progress(f"reused {relative.as_posix()} ({target.stat().st_size} bytes)")
                continue
            encoded = "/".join(quote(part, safe="") for part in remote_path.split("/"))
            url = f"https://huggingface.co/datasets/{self.repository}/resolve/{self.revision}/{encoded}?download=true"
            self._download(url, target, expected_sha256=expected, progress=progress)
        actual_paths = {
            path.relative_to(destination)
            for path in destination.rglob("*")
            if path.is_file() and path.name != "source.json"
        }
        unexpected = actual_paths - expected_paths
        if unexpected:
            names = ", ".join(sorted(path.as_posix() for path in unexpected))
            raise DatasetIntegrityError(f"LIBERO staging contains files outside the pinned snapshot: {names}")
        atomic_json(destination / "source.json", {
            "schema_version": DATASET_SOURCE_SCHEMA,
            "repository": self.repository,
            "revision": self.revision,
            "dataset_directory": resolution.source_metadata["dataset_directory"],
        }, exclusive=False, mode=0o644)

    def prepare(self, resolution: DatasetResolution, acquired: Path, destination: Path) -> dict[str, object]:
        destination.mkdir(parents=True, exist_ok=False)
        required = {"dataset_info.json", "features.json"}
        names = {path.name for path in acquired.iterdir() if path.is_file()}
        shards = [self._shard_name.fullmatch(name) for name in names if ".tfrecord-" in name]
        if not required <= names or not shards or any(match is None for match in shards):
            raise DatasetIntegrityError("LIBERO OpenVLA RLDS snapshot lacks TFDS metadata or TFRecord shards")
        totals = {int(match.group(2)) for match in shards if match is not None}
        indices = {int(match.group(1)) for match in shards if match is not None}
        if len(totals) != 1 or indices != set(range(next(iter(totals)))):
            raise DatasetIntegrityError("LIBERO OpenVLA RLDS snapshot has an incomplete TFRecord shard set")
        for source in sorted(acquired.iterdir()):
            if source.name == "source.json" or not source.is_file():
                continue
            os.link(source, destination / source.name)
        files, digest, size = inventory(destination)
        info = json.loads((destination / "dataset_info.json").read_text(encoding="utf-8"))
        sample_count = sum(int(value) for split in info.get("splits", ()) for value in split.get("shardLengths", ()))
        return {
            "files": files,
            "content_digest": digest,
            "total_size": size,
            "format": "openvla-rlds",
            "sample_count": sample_count,
            "schema": {
                "observation": "steps.observation.image:uint8[256,256,3]",
                "instruction": "steps.language_instruction:string",
                "action": "steps.action:float32[7]",
            },
        }

