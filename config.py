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

# -- HUGGING FACE AND PROMPT CONFIG ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "gemini-2.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- VLA CLOUD CONFIG ---
VLA_API_URL = "https://perinephral-catechisable-mirta.ngrok-free.dev"

def GET_RECEIVER_BASE64_IMAGE(receiver):
    last_packet = None
    while receiver.getQueueLength() > 0:
        last_packet = receiver.getBytes()
        receiver.nextPacket()
    
    if last_packet:
        nparr = np.frombuffer(last_packet, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preview for debugging
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