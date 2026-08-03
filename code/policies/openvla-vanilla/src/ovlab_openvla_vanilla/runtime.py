"""Small OpenVLA-specific runtime boundary and lazy production implementation."""

from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
from typing import Mapping, Protocol

import numpy as np

from ovlab_openvla_common import (
    OpenVlaCheckpointIdentity, OpenVlaDecodedAction, cuda_allocator_snapshot,
    estimated_inference_compute, parameter_inventory, performance_sample,
    reset_cuda_peak,
)
from ovlab_core.contracts import Metadata, normalize_metadata

from .errors import (
    OpenVlaActionDecodeError,
    OpenVlaCheckpointError,
    OpenVlaDependencyError,
    OpenVlaInferenceError,
    OpenVlaLoadError,
    OpenVlaPreprocessingError,
)
from .settings import (
    InferenceSynchronization, ModelDType, ModelQuantization, OpenVlaVanillaSettings,
)

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
        self.load_count = 0
        self.processor_load_count = 0
        self.peft_adapter_load_count = 0
        self._runtime_metadata = {}
        self._parameter_counts: dict[str, int] = {}

    @staticmethod
    def _install_lightweight_prismatic_namespace() -> None:
        """Expose prismatic.extern without importing the optional training stack."""
        if "prismatic" in sys.modules:
            return
        spec = importlib.util.find_spec("prismatic")
        locations = None if spec is None else spec.submodule_search_locations
        if not locations:
            raise ImportError("the pinned OpenVLA prismatic package is unavailable")
        package = types.ModuleType("prismatic")
        package.__package__ = "prismatic"
        package.__path__ = list(locations)
        package.__spec__ = spec
        sys.modules["prismatic"] = package

    @staticmethod
    def _runtime_imports():
        try:
            import torch
            from huggingface_hub import snapshot_download
            from PIL import Image
            from transformers import (
                AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor,
                BitsAndBytesConfig,
            )
            HuggingFaceOpenVlaRuntime._install_lightweight_prismatic_namespace()
            from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
            from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
            from prismatic.extern.hf.processing_prismatic import (
                PrismaticImageProcessor, PrismaticProcessor,
            )
        except (ImportError, OSError) as exc:
            raise OpenVlaDependencyError(
                "the tested Torch/Transformers/Hugging Face/Pillow runtime is unavailable"
            ) from exc
        # Fine-tuned OpenVLA snapshots refer their AutoClass implementation back
        # to openvla/openvla-7b. Register the exact implementation shipped in the
        # pinned OpenVLA source so an offline container never needs that second
        # Hub repository merely to load Python code.
        AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
        AutoImageProcessor.register(
            OpenVLAConfig, PrismaticImageProcessor, exist_ok=True
        )
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
        AutoModelForVision2Seq.register(
            OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True
        )
        return (
            torch, snapshot_download, Image, AutoConfig, AutoModelForVision2Seq,
            AutoProcessor, BitsAndBytesConfig,
        )

    @staticmethod
    def _model_load_kwargs(settings, torch, bits_and_bytes_config):
        dtype = {
            ModelDType.BFLOAT16: torch.bfloat16,
            ModelDType.FLOAT16: torch.float16,
            ModelDType.FLOAT32: torch.float32,
        }[settings.model_dtype]
        kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            # AutoClasses are bound to the pinned local OpenVLA implementation
            # in _runtime_imports(); dynamic Hub code is neither needed nor
            # permitted inside the offline policy service.
            "trust_remote_code": False,
            "local_files_only": True,
        }
        if settings.attention_implementation is not None:
            kwargs["attn_implementation"] = settings.attention_implementation
        if settings.quantization is ModelQuantization.BITSANDBYTES_NF4_4BIT:
            kwargs["torch_dtype"] = torch.float16
            kwargs["quantization_config"] = bits_and_bytes_config(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        return kwargs

    @staticmethod
    def _prepare_model_for_inference(model, settings) -> None:
        model.eval()
        # BitsAndBytes owns placement during from_pretrained(). Calling .to()
        # afterwards is unsupported for the pinned Transformers stack.
        if settings.quantization is ModelQuantization.NONE:
            model.to(settings.device)

    @staticmethod
    def _input_dtype(settings, model, torch):
        if settings.quantization is ModelQuantization.BITSANDBYTES_NF4_4BIT:
            try:
                return next(model.vision_backbone.parameters()).dtype
            except (AttributeError, StopIteration) as exc:
                raise OpenVlaPreprocessingError(
                    "4bit runtime cannot determine the vision-backbone input dtype"
                ) from exc
        return {
            ModelDType.BFLOAT16: torch.bfloat16,
            ModelDType.FLOAT16: torch.float16,
            ModelDType.FLOAT32: torch.float32,
        }[settings.model_dtype]

    @staticmethod
    def _resolve(source, snapshot_download, local_files_only: bool) -> Path:
        candidate = Path(source.local_path or source.source).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
        if source.local_path is not None:
            raise OpenVlaCheckpointError(f"resolved checkpoint path is unavailable: {candidate}")
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
        if self._model is not None or self._processor is not None:
            raise OpenVlaLoadError("OpenVLA runtime may load model and processor only once")
        (
            torch, snapshot_download, Image, AutoConfig, AutoModel, AutoProcessor,
            BitsAndBytesConfig,
        ) = self._runtime_imports()
        model_path = self._resolve(settings.model, snapshot_download, settings.local_files_only)
        processor_path = self._resolve(settings.processor_source, snapshot_download, settings.local_files_only)
        artifact_metadata = None
        if settings.runtime_artifact is not None:
            try:
                artifact_metadata = settings.runtime_artifact.verify(model_path)
            except ValueError as exc:
                raise OpenVlaCheckpointError(str(exc)) from exc
        try:
            config = AutoConfig.from_pretrained(
                str(model_path), trust_remote_code=False, local_files_only=True
            )
            processor = AutoProcessor.from_pretrained(
                str(processor_path), config=config, trust_remote_code=False,
                local_files_only=True,
            )
            self.processor_load_count += 1
            kwargs = self._model_load_kwargs(settings, torch, BitsAndBytesConfig)
            model = AutoModel.from_pretrained(str(model_path), config=config, **kwargs)
            self.load_count += 1
            stats_file = model_path / "dataset_statistics.json"
            if stats_file.is_file():
                model.norm_stats = json.loads(stats_file.read_text(encoding="utf-8"))
            norm_stats = getattr(model, "norm_stats", None)
            if not isinstance(norm_stats, Mapping) or settings.unnorm_key not in norm_stats:
                available = () if not isinstance(norm_stats, Mapping) else tuple(sorted(norm_stats))
                raise OpenVlaCheckpointError(
                    f"unnorm_key {settings.unnorm_key!r} is unavailable; available keys: {available}"
                )
            self._prepare_model_for_inference(model, settings)
            peft_names = tuple(name for name, _ in model.named_parameters() if "lora_" in name.lower())
            if peft_names:
                raise OpenVlaCheckpointError("full-weight runtime unexpectedly contains active LoRA parameters")
            loaded_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
            loaded_in_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
            if settings.quantization is ModelQuantization.NONE and (loaded_in_4bit or loaded_in_8bit):
                raise OpenVlaCheckpointError("unquantized runtime unexpectedly loaded quantized weights")
            if settings.quantization is ModelQuantization.BITSANDBYTES_NF4_4BIT and not loaded_in_4bit:
                raise OpenVlaCheckpointError("4bit runtime did not load BitsAndBytes 4-bit weights")
            if loaded_in_8bit:
                raise OpenVlaCheckpointError("8-bit loading is not supported by this runtime contract")
            self._parameter_counts = parameter_inventory(model.named_parameters())
            total_parameter_count = self._parameter_counts["total"]
        except OpenVlaCheckpointError:
            raise
        except Exception as exc:
            self.close()
            raise OpenVlaLoadError(f"failed to load local OpenVLA checkpoint {model_path}") from exc
        self._settings, self._model, self._processor, self._torch = settings, model, processor, torch
        self._Image = Image
        load_memory = cuda_allocator_snapshot(torch, settings.device)
        self._runtime_metadata = {
            "load_counts": {
                "model": self.load_count,
                "processor": self.processor_load_count,
                "peft_adapter": self.peft_adapter_load_count,
            },
            "total_parameter_count": total_parameter_count,
            "parameter_counts": dict(self._parameter_counts),
            "cuda_memory_after_load": load_memory,
            "active_peft_adapter": False,
            "runtime_peft_modules": False,
            "code_loading": "pinned-local-openvla-autoclass-registration",
            "quantized": settings.quantization is not ModelQuantization.NONE,
            "quantization": settings.quantization.configuration(),
            "inference_parameter_trainability": "irrelevant",
            "timing_method": (
                "time.perf_counter_ns around predict_action with torch.cuda.synchronize "
                "before and after the model call when using CUDA"
            ),
        }
        if artifact_metadata is not None:
            self._runtime_metadata["runtime_artifact"] = artifact_metadata
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
            metadata=self._runtime_metadata,
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
            input_shapes = {
                str(key): tuple(int(size) for size in value.shape)
                for key, value in inputs.items()
                if hasattr(value, "shape")
            }
            input_dtypes = {
                str(key): str(value.dtype)
                for key, value in inputs.items()
                if hasattr(value, "dtype")
            }
            input_fingerprints = {
                str(key): hashlib.sha256(
                    value.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()
                for key, value in inputs.items()
                if hasattr(value, "detach")
            }
            input_dtype = self._input_dtype(self._settings, self._model, self._torch)
            inputs = inputs.to(self._settings.device, dtype=input_dtype)
        except OpenVlaPreprocessingError:
            raise
        except Exception as exc:
            raise OpenVlaPreprocessingError("OpenVLA processor failed") from exc
        preprocess_end = self._clock_ns()
        try:
            self._synchronize()
            memory_before = cuda_allocator_snapshot(self._torch, self._settings.device)
            reset_cuda_peak(self._torch, self._settings.device)
            model_start = self._clock_ns()
            with self._torch.inference_mode():
                action = self._model.predict_action(
                    **inputs, unnorm_key=unnorm_key, do_sample=not self._settings.deterministic_inference
                )
            self._synchronize()
            model_end = self._clock_ns()
            memory_after = cuda_allocator_snapshot(self._torch, self._settings.device)
            decoded = OpenVlaDecodedAction(np.asarray(action))
        except OpenVlaActionDecodeError:
            raise
        except Exception as exc:
            raise OpenVlaInferenceError("OpenVLA predict_action failed") from exc
        input_shape = input_shapes.get("input_ids", ())
        input_tokens = int(np.prod(input_shape)) if input_shape else 0
        performance = performance_sample(
            phase="inference",
            parameter_counts=self._parameter_counts,
            memory_before=memory_before,
            memory_after=memory_after,
            compute=estimated_inference_compute(
                self._parameter_counts.get("total", 0),
                input_token_count=input_tokens,
                output_token_count=decoded.value.size,
            ),
        )
        return RuntimePrediction(
            decoded, preprocess_end - preprocess_start, model_end - model_start,
            {
                "processor_input_shapes": input_shapes,
                "processor_input_dtypes": input_dtypes,
                "processor_input_sha256": input_fingerprints,
                "performance": performance,
            },
        )

    def reset_episode(self, seed: int) -> None:
        # Vanilla is stateless across predictions; do not mutate global RNG state.
        del seed

    def close(self) -> None:
        self._model = self._processor = self._settings = self._torch = None
        self._Image = None
        self._parameter_counts = {}

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)
