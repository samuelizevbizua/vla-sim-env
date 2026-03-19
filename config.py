import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# -- WEBOTS AND ENVIRONMENT CONFIG ---
CAM_FRAME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), './tmp/camera_frame.png'))

# create camera frame path if it doesn't exist
os.makedirs(os.path.dirname(CAM_FRAME_PATH), exist_ok=True)
WORLD_TABLE_HEIGHT = 0.68

# -- HUGGING FACE AND PROMPT CONFIG ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPTS = {
    "get_vlm_target": '''
        You are a robotic vision system. Look at the provided image.
        1. Identify one of the small red cubes on the right side of the table.
        2. Identify the large blue bin on the left side of the table.
        Respond ONLY with a valid JSON dictionary containing the center pixel coordinates of both items.
        Use this exact format: {\"red_cube\": [x, y], \"blue_bin\": [x, y]}.
        Do not include any other text, markdown, or explanations.
    '''
}
