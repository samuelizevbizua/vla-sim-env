import os, base64
import cv2
import numpy as np
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# world parameters
CAM1_NAME, CAM1_DEF= "cam1", "CAM1" # overhead camera node
CAM2_NAME, CAM2_DEF = "cam2", "CAM2" # angled side camera node -- provides depth data
CAM3_NAME, CAM3_DEF = "cam3", "CAM3" # wrist camera node

CAM1_EMITTER = "cam1_emitter"
CAM1_RECEIVER = "cam1_receiver"

CAM2_EMITTER = "cam2_emitter"
CAM2_RECEIVER = "cam2_receiver"

CAM3_EMITTER = "cam3_emitter"
CAM3_RECEIVER = "cam3_receiver"

SIM_TABLE_DEF = "SIM_TABLE"

ARM_MOTOR_VELOCITY = 0.3
GRIPPER_MOTOR_VELOCITY = 0.5
GRIPPER_TARGET_HOVER_HEIGHT = -0.00155
GRIPPER_LIFT_HEIGHT = 0.2
GRIPPER_TARGET_ANGLE_OPEN = 0.0
GRIPPER_TARGET_ANGLE_CLOSED = 0.65

# -- HUGGING FACE AND PROMPT CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN")
HF_BASE_URL = os.getenv("HF_BASE_URL")

VLM_PATH = "Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud"
VLM_USER_PROMPTS = {
    "baseline": "Identify the red cube and the green bin. Return a JSON object with these exact keys: 'cube_x_norm', 'cube_y_norm', 'bin_x_norm', 'bin_y_norm'.",
    "nudge_obj": "Identify the red cube. Return JSON: {'x': float, 'y': float, 'found': bool}",
    "nudge_bin": "Identify the green bin. Return JSON: {'x': float, 'y': float, 'found': bool}",
    "ood_test": "Identify the duck and the green bin. Return a JSON object with these exact keys: 'target_x_norm', 'target_y_norm', 'bin_x_norm', 'bin_y_norm'."
}

VLA_PATH = "lerobot/smolvla_base"
VLA_API_URL = "https://perinephral-catechisable-mirta.ngrok-free.dev"
INFERENCE_EVERY_N = 5
VLA_USER_PROMPTS = {
    "baseline": "pick up the red cube and put it in the green bin",
    "nudge": "pick up the red cube and put it in the green bin",
    "ood": "pick up the duck and put it in the green bin"
}

TARGET_NUDGES = 1
NUDGE_RANGE = (0.10, 0.20)
NUDGE_TOLERANCE_NORM = min(NUDGE_RANGE)

def GET_RECEIVER_BASE64_IMAGE(receiver, preview_img=False, format="base64"):
    last_packet = None
    while receiver.getQueueLength() > 0:
        last_packet = receiver.getBytes()
        receiver.nextPacket()
    if last_packet:
        nparr = np.frombuffer(last_packet, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if preview_img:
            cv2.imshow("AI Feed", frame)
            cv2.waitKey(1)
        if format == "base64":
            _, buffer = cv2.imencode('.jpg', frame)
            base64_img = base64.b64encode(buffer).decode('utf-8')
            return base64_img
        elif format == "numpy":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def PIXEL_TO_WORLD(px, py, TABLE_HEIGHT, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y, CAM_Z):
    _height = CAM_Z - TABLE_HEIGHT
    _ground_w = 2.0 * _height * np.tan(CAM_FOV / 2.0)
    _ground_h = _ground_w * (CAM_HEIGHT / CAM_WIDTH)
    MPP_X = _ground_w / CAM_WIDTH
    MPP_Y = -_ground_h / CAM_HEIGHT
    cx, cy = CAM_WIDTH / 2.0, CAM_HEIGHT / 2.0
    world_x = float(round(CAM_X + ((px * CAM_WIDTH) - cx) * MPP_X, 3))
    world_y = float(round(CAM_Y + ((py * CAM_HEIGHT) - cy) * MPP_Y, 3))
    return world_x, world_y

ACTION_STD = [
    0.38376863449487764,
    0.27165019367589216,
    0.4432032971941338,
    0.24501470104936277,
    6.703042191700911e-08,
    0.38361808193022623,
    0.495171386915027
]

ACTION_MEAN = [
    1.4142366058399456,
    -1.065433056151644,
    1.3003412158654786,
    1.3358856972185498,
    1.5708003423550607,
    -1.7273641064678864,
    0.4306797588350734
]