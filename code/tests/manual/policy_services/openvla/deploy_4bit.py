"""
deploy.py

Provides a lightweight server/client implementation for deploying OpenVLA models 
(through the HF AutoClass API) over a REST API. 

This version includes crucial memory optimizations (4-bit quantization via BitsAndBytes)
to prevent CUDA Out-Of-Memory errors.
"""

import os.path
import json_numpy

# Patch json module to support numpy arrays natively
json_numpy.patch()

import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def load_4bit_vla(
    openvla_path: Union[str, Path],
    attn_implementation: Optional[str],
) -> torch.nn.Module:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    return AutoModelForVision2Seq.from_pretrained(
        openvla_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
    )

def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in str(openvla_path):
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

# === Server Interface ===
class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: Optional[str] = None):
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
        => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
        => Returns {"action": np.ndarray}
        """
        self.openvla_path = openvla_path
        self.attn_implementation = attn_implementation
        
        # We define this as a fallback for tensor inputs; quantized models handle
        # their own placement during loading.
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        print(f"Loading processor from {self.openvla_path}...")
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)

        print(f"Loading model from {self.openvla_path} (This may take a moment)...")
        self.vla = load_4bit_vla(self.openvla_path, attn_implementation)
        self.device = next(self.vla.parameters()).device
        self.input_dtype = next(self.vla.vision_backbone.parameters()).dtype
        print("Model successfully loaded into VRAM!")

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            stats_path = Path(self.openvla_path) / "dataset_statistics.json"
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    self.vla.norm_stats = json.load(f)

    def predict_action(self, payload: Dict[str, Any]) -> Any:
        try:
            # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
            if double_encode := "encoded" in payload:
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            
            # Send inputs to the same device where our tensors should compute
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(
                self.device, dtype=self.input_dtype
            )
            
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action.tolist())

        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally pass an `unnorm_key: str` to specify the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI(title="OpenVLA Local Server")
        self.app.post("/act")(self.predict_action)
        print(f"Starting server on {host}:{port}...")
        uvicorn.run(self.app, host=host, port=port)

@dataclass
class DeployConfig:
    # HF Hub Path (or path to local run directory)
    openvla_path: Union[str, Path] = "openvla/openvla-7b" 

    # Keep disabled for initial 4-bit loader debugging; set to "flash_attention_2" after the model loads cleanly.
    attn_implementation: Optional[str] = None
    
    # Server Configuration
    host: str = "0.0.0.0"  
    port: int = 8000       

@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg.openvla_path, cfg.attn_implementation)
    server.run(cfg.host, port=cfg.port)

if __name__ == "__main__":
    deploy()
