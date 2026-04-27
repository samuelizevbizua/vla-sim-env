import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Robot
import numpy as np
from config import CAM2_EMITTER, CAM2_NAME

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Depth RGB camera -- grayscale depth
camera = robot.getDevice(CAM2_NAME)
camera.enable(timestep)
emitter = robot.getDevice(CAM2_EMITTER)

print(f"[Camera] RGB DEPTH emitter channel: {emitter.getChannel()}")

while robot.step(timestep) != -1:
    # --- Send depth data (raw float32, uncompressed) ---
    depth_array = camera.getRangeImage()
    if depth_array is not None:
        # Convert to bytes (float32)
        bytes = np.array(depth_array, dtype=np.float32).tobytes()
        emitter.send(bytes)