import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

WORLD_TABLE_HEIGHT = 0.68

# Mathematically derived via Ground Truth Calibration mapping pixels to world coordinates.
# These values represent the exact Webots World (X, Y) location of the camera's center pixel (750, 500).
CAM_X_OFFSET = 0.014 # calculated
CAM_Y_OFFSET = 0.841 # calculated
CAM_WIDTH = 640
CAM_HEIGHT = 450

# -- HUGGING FACE AND PROMPT CONFIG ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "gemini-2.5-flash" # "gemini-1.5-flash" --- IGNORE ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")