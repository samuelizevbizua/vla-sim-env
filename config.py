import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# -- WEBOTS AND ENVIRONMENT CONFIG ---
CAM_FRAME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), './tmp/camera_frame.png'))

# create camera frame path if it doesn't exist
os.makedirs(os.path.dirname(CAM_FRAME_PATH), exist_ok=True)

WORLD_TABLE_HEIGHT = 0.68

# Mathematically derived via Ground Truth Calibration mapping pixels to world coordinates.
# These values represent the exact Webots World (X, Y) location of the camera's center pixel (750, 500).
CAM_X_OFFSET = 0.014 # calculated
CAM_Y_OFFSET = 0.841 # calculated

# -- HUGGING FACE AND PROMPT CONFIG ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
