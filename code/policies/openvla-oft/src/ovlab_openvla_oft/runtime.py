"""Lazy bridge to the pinned official OpenVLA-OFT inference implementation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from ovlab_openvla_common import OpenVlaDecodedActionChunk

from .settings import OpenVlaOftSettings

OPENVLA_OFT_GIT_COMMIT = "e4287e94541f459edc4feabc4e181f537cd569a8"


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class OftRuntimePrediction:
    normalized_actions: np.ndarray
    decoded_actions: OpenVlaDecodedActionChunk
    normalized_proprioception: np.ndarray
    preprocessing_duration_ns: int
    model_duration_ns: int
    metadata: dict[str, object]


class _RecordingProcessor:
    def __init__(self, processor) -> None:
        self.processor = processor
        self.calls: list[dict[str, object]] = []

    def __call__(self, *args, **kwargs):
        result = self.processor(*args, **kwargs)
        self.calls.append({
            "keys": sorted(result.keys()),
            "shapes": {key: list(value.shape) for key, value in result.items() if hasattr(value, "shape")},
            "dtypes": {key: str(value.dtype) for key, value in result.items() if hasattr(value, "dtype")},
            "sha256": {
                key: _sha(value.detach().cpu().contiguous().numpy())
                for key, value in result.items() if hasattr(value, "detach")
            },
        })
        return result


class OpenVlaOftRuntime:
    def __init__(self, clock_ns=time.perf_counter_ns) -> None:
        self._clock_ns = clock_ns
        self._settings = self._model = self._processor = self._action_head = self._proprio_projector = None
        self._torch = self._get_vla_action = None
        self.load_counts = {"backbone": 0, "processor": 0, "published_peft_adapter": 0,
                            "action_head": 0, "proprio_projector": 0}
        self.warmup_duration_ns = 0
        self.prediction_count = 0
        self._runtime_metadata: dict[str, object] = {}

    @staticmethod
    def _imports():
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from experiments.robot.openvla_utils import get_vla_action, load_component_state_dict
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        return (torch, snapshot_download, AutoModelForVision2Seq, AutoProcessor, get_vla_action,
                load_component_state_dict, L1RegressionActionHead, ProprioProjector)

    def _sync(self) -> None:
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize(self._settings.device)

    def load(self, settings: OpenVlaOftSettings) -> dict[str, object]:
        if self._model is not None:
            raise RuntimeError("OFT runtime components may be loaded only once")
        (torch, snapshot_download, AutoModel, AutoProcessor, get_vla_action,
         load_state, L1Head, ProprioProjector) = self._imports()
        if settings.model.local_path is not None:
            snapshot = Path(settings.model.local_path).resolve()
            if not snapshot.is_dir():
                raise RuntimeError(f"resolved OFT checkpoint path is unavailable: {snapshot}")
        else:
            snapshot = Path(snapshot_download(
                repo_id=settings.model.source, revision=settings.model.revision, local_files_only=True,
            )).resolve()
        verified = settings.artifact.verify(snapshot)
        kwargs = {
            "torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True,
            "trust_remote_code": True, "local_files_only": True,
        }
        if settings.attention_implementation is not None:
            kwargs["attn_implementation"] = settings.attention_implementation
        load_started = self._clock_ns()
        processor = AutoProcessor.from_pretrained(str(snapshot), trust_remote_code=True, local_files_only=True)
        self.load_counts["processor"] += 1
        model = AutoModel.from_pretrained(str(snapshot), **kwargs)
        self.load_counts["backbone"] += 1
        model.vision_backbone.set_num_images_in_input(2)
        model.norm_stats = json.loads((snapshot / "dataset_statistics.json").read_text(encoding="utf-8"))
        if settings.unnorm_key not in model.norm_stats:
            raise RuntimeError(f"OFT normalization key is absent: {settings.unnorm_key}")
        model.eval().to(settings.device)
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            raise RuntimeError("Gate E OFT runtime must not be quantized")
        if any("lora_" in name.lower() for name, _ in model.named_parameters()):
            raise RuntimeError("published OFT runtime must use merged backbone weights, not active PEFT")
        action_head = L1Head(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=7)
        action_head.load_state_dict(load_state(snapshot / "action_head--150000_checkpoint.pt"))
        action_head.to(torch.bfloat16).to(settings.device).eval()
        self.load_counts["action_head"] += 1
        projector = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8)
        projector.load_state_dict(load_state(snapshot / "proprio_projector--150000_checkpoint.pt"))
        projector.to(torch.bfloat16).to(settings.device).eval()
        self.load_counts["proprio_projector"] += 1
        self._settings, self._model, self._processor = settings, model, processor
        self._action_head, self._proprio_projector = action_head, projector
        self._torch, self._get_vla_action = torch, get_vla_action
        load_finished = self._clock_ns()
        warm = self._call(
            np.zeros(settings.input_image_shape, dtype=np.uint8),
            np.zeros(settings.input_image_shape, dtype=np.uint8),
            np.zeros(8, dtype=np.float32),
            "do nothing",
            count_prediction=False,
        )
        self.warmup_duration_ns = warm.model_duration_ns
        total_runtime = sum(parameter.numel() for parameter in model.parameters())
        self._runtime_metadata = {
            "load_counts": dict(self.load_counts),
            "cold_component_loading_duration_ns": load_finished - load_started,
            "warmup_duration_ns": self.warmup_duration_ns,
            "timing_method": "perf_counter_ns with torch.cuda.synchronize before and after official get_vla_action",
            "total_runtime_parameter_count": total_runtime,
            "runtime_active_adapter": False,
            "backbone_merge_status": "merged",
            "quantization": "none",
            "verified_artifact": verified,
            "action_statistics_identity": "sha256:" + hashlib.sha256(
                json.dumps(
                    model.norm_stats[settings.unnorm_key], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }
        return dict(self._runtime_metadata)

    def _call(self, primary, wrist, proprio, instruction, *, count_prediction: bool) -> OftRuntimePrediction:
        recorder = _RecordingProcessor(self._processor)
        normalized_capture: dict[str, np.ndarray] = {}
        original_unnormalize = self._model._unnormalize_actions

        def capture(values, unnorm_key=None):
            normalized_capture["actions"] = np.asarray(values).copy()
            return original_unnormalize(values, unnorm_key)

        self._model._unnormalize_actions = capture
        observation = {
            "full_image": np.array(primary, copy=True), "wrist_image": np.array(wrist, copy=True),
            "state": np.asarray(proprio, dtype=np.float32).copy(),
        }
        cfg = SimpleNamespace(
            pretrained_checkpoint="immutable-local-snapshot", use_l1_regression=True,
            use_diffusion=False, use_film=False, num_images_in_input=2, use_proprio=True,
            load_in_8bit=False, load_in_4bit=False, center_crop=True,
            num_open_loop_steps=8, unnorm_key=self._settings.unnorm_key,
        )
        try:
            self._sync()
            started = self._clock_ns()
            actions = self._get_vla_action(
                cfg, self._model, recorder, observation, instruction,
                self._action_head, self._proprio_projector,
            )
            self._sync()
            finished = self._clock_ns()
        finally:
            self._model._unnormalize_actions = original_unnormalize
        decoded = np.asarray(actions)
        normalized = np.asarray(normalized_capture["actions"])
        normalized_proprio = np.asarray(observation["state"])
        if decoded.shape != (8, 7) or normalized.shape != (8, 7) or normalized_proprio.shape != (8,):
            raise RuntimeError("official OFT output or normalized proprioception has an unexpected shape")
        if count_prediction:
            self.prediction_count += 1
        return OftRuntimePrediction(
            normalized, OpenVlaDecodedActionChunk(decoded), normalized_proprio,
            0, finished - started,
            {"processor_calls": recorder.calls, "prompt": f"In: What action should the robot take to {instruction.lower()}?\nOut:"},
        )

    def predict(self, primary, wrist, proprio, instruction) -> OftRuntimePrediction:
        if self._model is None:
            raise RuntimeError("OFT runtime is not loaded")
        return self._call(primary, wrist, proprio, instruction, count_prediction=True)

    def reset_episode(self, seed: int) -> None:
        del seed

    def runtime_metadata(self) -> dict[str, object]:
        return {**self._runtime_metadata, "prediction_count": self.prediction_count}

    def close(self) -> None:
        self._model = self._processor = self._action_head = self._proprio_projector = None
        self._torch = self._get_vla_action = self._settings = None
