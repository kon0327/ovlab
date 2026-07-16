"""Small OpenVLA-specific runtime boundary and lazy production implementation."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Protocol

import numpy as np

from ovlab_openvla_common import OpenVlaCheckpointIdentity, OpenVlaDecodedAction
from ovlab_core.contracts import Metadata, normalize_metadata

from .errors import (
    OpenVlaActionDecodeError,
    OpenVlaCheckpointError,
    OpenVlaDependencyError,
    OpenVlaInferenceError,
    OpenVlaLoadError,
    OpenVlaPreprocessingError,
)
from .settings import InferenceSynchronization, ModelDType, OpenVlaVanillaSettings

OPENVLA_GIT_COMMIT = "c8f03f48af692657d3060c19588038c7220e9af9"


@dataclass(frozen=True, slots=True)
class RuntimePrediction:
    decoded_action: OpenVlaDecodedAction
    preprocessing_duration_ns: int
    model_duration_ns: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decoded_action, OpenVlaDecodedAction):
            raise TypeError("decoded_action must be OpenVlaDecodedAction")
        for name in ("preprocessing_duration_ns", "model_duration_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, type(self).__name__))


class OpenVlaRuntime(Protocol):
    def load(self, settings: OpenVlaVanillaSettings) -> OpenVlaCheckpointIdentity: ...
    def predict(self, image: np.ndarray, prompt: str, unnorm_key: str) -> RuntimePrediction: ...
    def reset_episode(self, seed: int) -> None: ...
    def close(self) -> None: ...


class HuggingFaceOpenVlaRuntime:
    """Production runtime. Heavy dependencies are imported only by load()."""

    def __init__(self, clock_ns=time.perf_counter_ns) -> None:
        self._clock_ns = clock_ns
        self._settings: OpenVlaVanillaSettings | None = None
        self._model = None
        self._processor = None
        self._torch = None

    @staticmethod
    def _runtime_imports():
        try:
            import torch
            from huggingface_hub import snapshot_download
            from PIL import Image
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except (ImportError, OSError) as exc:
            raise OpenVlaDependencyError(
                "the tested Torch/Transformers/Hugging Face/Pillow runtime is unavailable"
            ) from exc
        return torch, snapshot_download, Image, AutoModelForVision2Seq, AutoProcessor

    @staticmethod
    def _resolve(source, snapshot_download, local_files_only: bool) -> Path:
        candidate = Path(source.source).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
        if not local_files_only:
            raise OpenVlaCheckpointError("OVLAB Vanilla currently requires local_files_only=True")
        try:
            resolved = snapshot_download(
                repo_id=source.source, revision=source.revision, local_files_only=True
            )
        except Exception as exc:
            raise OpenVlaCheckpointError(
                f"checkpoint {source.source!r} is not available in the local Hugging Face cache"
            ) from exc
        path = Path(resolved)
        if not path.is_dir():
            raise OpenVlaCheckpointError(f"resolved checkpoint path is unavailable: {path}")
        return path.resolve()

    def load(self, settings: OpenVlaVanillaSettings) -> OpenVlaCheckpointIdentity:
        torch, snapshot_download, Image, AutoModel, AutoProcessor = self._runtime_imports()
        model_path = self._resolve(settings.model, snapshot_download, settings.local_files_only)
        processor_path = self._resolve(settings.processor_source, snapshot_download, settings.local_files_only)
        dtype = {
            ModelDType.BFLOAT16: torch.bfloat16,
            ModelDType.FLOAT16: torch.float16,
            ModelDType.FLOAT32: torch.float32,
        }[settings.model_dtype]
        try:
            processor = AutoProcessor.from_pretrained(
                str(processor_path), trust_remote_code=settings.trust_remote_code, local_files_only=True
            )
            kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": settings.trust_remote_code,
                "local_files_only": True,
            }
            if settings.attention_implementation is not None:
                kwargs["attn_implementation"] = settings.attention_implementation
            model = AutoModel.from_pretrained(str(model_path), **kwargs)
            stats_file = model_path / "dataset_statistics.json"
            if stats_file.is_file():
                model.norm_stats = json.loads(stats_file.read_text(encoding="utf-8"))
            norm_stats = getattr(model, "norm_stats", None)
            if not isinstance(norm_stats, Mapping) or settings.unnorm_key not in norm_stats:
                available = () if not isinstance(norm_stats, Mapping) else tuple(sorted(norm_stats))
                raise OpenVlaCheckpointError(
                    f"unnorm_key {settings.unnorm_key!r} is unavailable; available keys: {available}"
                )
            model.eval()
            model.to(settings.device)
        except OpenVlaCheckpointError:
            raise
        except Exception as exc:
            self.close()
            raise OpenVlaLoadError(f"failed to load local OpenVLA checkpoint {model_path}") from exc
        self._settings, self._model, self._processor, self._torch = settings, model, processor, torch
        self._Image = Image
        selected_stats = norm_stats[settings.unnorm_key]
        stats_hash = hashlib.sha256(
            json.dumps(selected_stats, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        config = getattr(model, "config", None)
        config_data = config.to_dict() if config is not None and hasattr(config, "to_dict") else {}
        config_hash = hashlib.sha256(
            json.dumps(config_data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        config_identity = f"{type(config).__name__}/sha256:{config_hash}"
        processor_identity = f"{type(processor).__name__}@{settings.processor_source.source}"
        if settings.model.expected_checksum:
            strength = "expected-checksum-reference"
        elif settings.model.revision or (len(model_path.name) == 40 and all(c in "0123456789abcdef" for c in model_path.name.lower())):
            strength = "revision-metadata"
        else:
            strength = "local-path-metadata"
        return OpenVlaCheckpointIdentity(
            configured_source=settings.model.source,
            resolved_local_path=str(model_path),
            openvla_git_commit=OPENVLA_GIT_COMMIT,
            model_identity=str(config_identity),
            processor_identity=processor_identity,
            unnorm_key=settings.unnorm_key,
            action_statistics_identity=f"sha256:{stats_hash}",
            snapshot_revision=(
                settings.model.revision
                or (model_path.name if model_path.parent.name == "snapshots" else None)
            ),
            expected_checksum=settings.model.expected_checksum,
            settings_hash=settings.settings_hash,
            identity_strength=strength,
        )

    def _synchronize(self) -> None:
        assert self._settings is not None and self._torch is not None
        policy = self._settings.synchronization
        is_cuda = self._settings.device.startswith("cuda")
        if policy is InferenceSynchronization.ALWAYS or (policy is InferenceSynchronization.IF_CUDA and is_cuda):
            if is_cuda and self._torch.cuda.is_available():
                self._torch.cuda.synchronize(self._settings.device)

    def predict(self, image: np.ndarray, prompt: str, unnorm_key: str) -> RuntimePrediction:
        if self._model is None or self._processor is None or self._settings is None or self._torch is None:
            raise OpenVlaInferenceError("runtime is not loaded")
        preprocess_start = self._clock_ns()
        try:
            pil_image = self._Image.fromarray(image).convert("RGB")
            inputs = self._processor(prompt, pil_image)
            if "input_ids" not in inputs or "pixel_values" not in inputs:
                raise OpenVlaPreprocessingError("processor output lacks input_ids or pixel_values")
            dtype = {
                ModelDType.BFLOAT16: self._torch.bfloat16,
                ModelDType.FLOAT16: self._torch.float16,
                ModelDType.FLOAT32: self._torch.float32,
            }[self._settings.model_dtype]
            inputs = inputs.to(self._settings.device, dtype=dtype)
        except OpenVlaPreprocessingError:
            raise
        except Exception as exc:
            raise OpenVlaPreprocessingError("OpenVLA processor failed") from exc
        preprocess_end = self._clock_ns()
        try:
            self._synchronize()
            model_start = self._clock_ns()
            with self._torch.inference_mode():
                action = self._model.predict_action(
                    **inputs, unnorm_key=unnorm_key, do_sample=not self._settings.deterministic_inference
                )
            self._synchronize()
            model_end = self._clock_ns()
            decoded = OpenVlaDecodedAction(np.asarray(action))
        except OpenVlaActionDecodeError:
            raise
        except Exception as exc:
            raise OpenVlaInferenceError("OpenVLA predict_action failed") from exc
        shapes = {
            str(key): tuple(int(size) for size in value.shape)
            for key, value in inputs.items()
            if hasattr(value, "shape")
        }
        return RuntimePrediction(
            decoded, preprocess_end - preprocess_start, model_end - model_start,
            {"processor_input_shapes": shapes},
        )

    def reset_episode(self, seed: int) -> None:
        # Vanilla is stateless across predictions; do not mutate global RNG state.
        del seed

    def close(self) -> None:
        self._model = self._processor = self._settings = self._torch = None
