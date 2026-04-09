import os, base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# world parameters
OVERHEAD_CAM_NAME, OVERHEAD_CAM_DEF= "overhead_camera", "OVERHEAD_CAMERA" # overhead camera node
PERSPECTIVE_CAM_NAME, PERSPECTIVE_CAM_DEF = "perspective_camera", "PERSPECTIVE_CAMERA" # angled camera node

OVERHEAD_EMITTER = "overhead_emitter"
OVERHEAD_RECEIVER = "overhead_receiver"

PERSPECTIVE_EMITTER = "perspective_emitter"
PERSPECTIVE_RECEIVER = "perspective_receiver"

SIM_TABLE_DEF = "SIM_TABLE"

ARM_MOTOR_VELOCITY = 1.0
GRIPPER_MOTOR_VELOCITY = 0.6
GRIPPER_TARGET_HOVER_HEIGHT = 0.035
GRIPPER_LIFT_HEIGHT = 0.2
GRIPPER_TARGET_ANGLE_OPEN = 0.06
GRIPPER_TARGET_ANGLE_CLOSED = 0.85

# -- HUGGING FACE AND PROMPT CONFIG ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_BASE_URL = os.getenv("HF_BASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# VLM CONFIG
VLM_PATHS = {
    1: {"path": "Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud"},
    2: {"path": "gemini-2.5-flash"},
}
SELECTED_VLM = 1
VLM_USER_PROMPTS = {
    "baseline": "Identify ONE blue cube and the YELLOW sorting bin. Return a JSON object with these exact keys: 'cube_x_norm', 'cube_y_norm', 'bin_x_norm', 'bin_y_norm'.",
    "nudge_obj": "Identify ONE blue cube. Return JSON: {'x': float, 'y': float, 'found': bool}",
    "nudge_bin": "Identify the YELLOW bin. Return JSON: {'x': float, 'y': float, 'found': bool}",
    "ood_test": "Identify the jam jar and the YELLOW sorting bin. Return a JSON object with these exact keys: 'target_x_norm', 'target_y_norm', 'bin_x_norm', 'bin_y_norm'."
}

# VLA CONFIG
VLA_PATH = "openvla/openvla-7b"
VLA_API_URL = "https://perinephral-catechisable-mirta.ngrok-free.dev"
DELTA_SCALE = 0.5 # How many physics steps to wait between taking pictures (simulates real-world Hz)
INFERENCE_EVERY_N = 5 
VLA_USER_PROMPTS = {
    "baseline": "pick up the blue cube and put it in the yellow bin",
    "nudge": "pick up the blue cube and put it in the yellow bin",
    "ood": "pick up the jam jar and put it in the yellow bin"
}

# TESTING / DEBUGGING PARAMETERS
TARGET_NUDGES = 1 # Number of nudges to perform during the Dynamic Nudge Tracking test
NUDGE_RANGE = (0.10, 0.20) # Range (in meters) for random nudge shifts on the X and Y axes
NUDGE_TOLERANCE_NORM = min(NUDGE_RANGE)

def GET_RECEIVER_BASE64_IMAGE(receiver, preview_img=False):
    last_packet = None
    while receiver.getQueueLength() > 0:
        last_packet = receiver.getBytes()
        receiver.nextPacket()
    
    if last_packet:
        nparr = np.frombuffer(last_packet, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preview for debugging
        if preview_img:
            cv2.imshow("AI Feed", frame)
            cv2.waitKey(1)

        #Direct memory encoding to Base64 (No disk I/O)
        _, buffer = cv2.imencode('.jpg', frame)
        base64_img = base64.b64encode(buffer).decode('utf-8')
        return base64_img
    return None

def RESIZE_IMAGE(base64_img):
    img_bytes = base64.b64decode(base64_img)
    image     = Image.open(BytesIO(img_bytes)).convert("RGB")
    image     = image.resize((224, 224), Image.LANCZOS)
    return image

def PIXEL_TO_WORLD(px, py, _height, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y):
    _ground_w = 2.0 * _height * np.tan(CAM_FOV / 2.0)
    _ground_h = _ground_w * (CAM_HEIGHT / CAM_WIDTH)
    MPP_X =  _ground_w / CAM_WIDTH
    MPP_Y = -_ground_h / CAM_HEIGHT

    # Math Mapping
    cx, cy = CAM_WIDTH / 2.0, CAM_HEIGHT / 2.0
    world_x = float(round(CAM_X + ((px * CAM_WIDTH) - cx) * MPP_X, 3))
    world_y = float(round(CAM_Y + ((py * CAM_HEIGHT) - cy) * MPP_Y, 3))
    return world_x, world_y