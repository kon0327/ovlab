import os.path
import json_numpy
json_numpy.patch()
import json
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

class OpenVLAOFTServer:
    def __init__(self, model_id: str = "openvla/openvla-7b"):
        self.device = torch.device("cuda:0")
        
        print(f"Loading OFT-optimized model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # We load in bfloat16 for speed and stability. 
        # Device_map="cuda:0" ensures it goes straight to GPU without calling .to() later.
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": 0} 
        )
        print("Model loaded successfully!")

    def predict_action(self, payload: dict):
        try:
            if "encoded" in payload:
                payload = json.loads(payload["encoded"])

            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", "bridge_orig")

            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
            
            # Fast inference
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            
            with torch.inference_mode():
                action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

            return JSONResponse(action.tolist())
        except Exception as e:
            print(f"Prediction error: {e}")
            return "error"

    def run(self):
        app = FastAPI()
        app.post("/act")(self.predict_action)
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # You can try the official OFT checkpoints if you have them, 
    # otherwise the base model with bfloat16 and flash-attn is the next best thing.
    server = OpenVLAOFTServer("openvla/openvla-7b")
    server.run()