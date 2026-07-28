import pickle
import os
import sys
from pathlib import Path

import numpy as np

OVLAB_ROOT = Path(__file__).resolve().parents[5]
OPENVLA_OFT_ROOT = OVLAB_ROOT / "external" / "openvla-oft"
sys.path.insert(0, str(OPENVLA_OFT_ROOT))

from experiments.robot.libero.run_libero_eval import GenerateConfig
from huggingface_hub import snapshot_download
from experiments.robot import openvla_utils
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
REVISION = "6d0231af0e48c5985f1ff86908f4674b84bc049b"

if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
    raise RuntimeError("This frozen smoke test must run with Hugging Face offline mode enabled")
snapshot = Path(snapshot_download(
    repo_id=CHECKPOINT, revision=REVISION, local_files_only=True,
)).resolve()
if snapshot.name != REVISION:
    raise RuntimeError(f"unexpected cached checkpoint revision: {snapshot}")

# Upstream's model_is_on_hf_hub() probes the network. In offline mode that probe
# misclassifies a cached repo ID as a local path, so constrain it to this pinned
# and already verified test resource. No model or checkpoint code is modified.
openvla_utils.model_is_on_hf_hub = lambda value: value == CHECKPOINT

# Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
cfg = GenerateConfig(
    pretrained_checkpoint = CHECKPOINT,
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 2,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "libero_spatial_no_noops",
)

# Load OpenVLA-OFT policy and inputs processor
vla = openvla_utils.get_vla(cfg)
processor = openvla_utils.get_processor(cfg)

# Load MLP action head to generate continuous actions (via L1 regression)
action_head = openvla_utils.get_action_head(cfg, llm_dim=vla.llm_dim)

# Load proprio projector to map proprio to language embedding space
proprio_projector = openvla_utils.get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load sample observation:
#   observation (dict): {
#     "full_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "state": robot proprioceptive state,
#     "task_description": task description,
#   }
sample_observation = OPENVLA_OFT_ROOT / "experiments" / "robot" / "libero" / "sample_libero_spatial_observation.pkl"
with sample_observation.open("rb") as file:
    observation = pickle.load(file)

# Generate robot action chunk (sequence of future actions)
actions = np.asarray(openvla_utils.get_vla_action(
    cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector,
), dtype=np.float32)
if actions.shape != (NUM_ACTIONS_CHUNK, 7):
    raise RuntimeError(f"unexpected action chunk shape: {actions.shape}")
print("Generated action chunk:")
for act in actions:
    print(act)
