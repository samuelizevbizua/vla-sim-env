import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Robot
import cv2
import numpy as np
from config import (
    CAM2_EMITTER, CAM2_NAME,
    CAM2_DEPTH_EMITTER, CAM2_DEPTH_NAME
)

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# RGB camera
camera = robot.getDevice(CAM2_NAME)
camera.enable(timestep)
rgb_emitter = robot.getDevice(CAM2_EMITTER)

# Depth sensor
depth_sensor = robot.getDevice(CAM2_DEPTH_NAME)
depth_sensor.enable(timestep)
depth_emitter = robot.getDevice(CAM2_DEPTH_EMITTER)

print(f"[Cam2] RGB emitter channel: {rgb_emitter.getChannel()}")
print(f"[Cam2] Depth emitter channel: {depth_emitter.getChannel()}")

while robot.step(timestep) != -1:
    # --- Send RGB image (JPEG compressed) ---
    raw_image = camera.getImage()
    if raw_image:
        # Convert to numpy (BGRA)
        img = np.frombuffer(raw_image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        # Compress to JPEG (quality 80)
        success, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            rgb_emitter.send(encoded_img.tobytes())
        else:
            print("[Cam2] Failed to encode RGB image")

    # --- Send depth data (raw float32, uncompressed) ---
    depth_array = depth_sensor.getRangeImage()
    if depth_array is not None:
        # Convert to bytes (float32)
        depth_bytes = np.array(depth_array, dtype=np.float32).tobytes()
        depth_emitter.send(depth_bytes)