import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Robot
import cv2
import numpy as np
from config import  CAM3_EMITTER, CAM3_NAME, CAM_EMITTER_QUALITY

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice(CAM3_NAME)
camera.enable(timestep)
emitter = robot.getDevice(CAM3_EMITTER)

print(f"[Camera] Emitter initialized with channel: {emitter.getChannel()}")

while robot.step(timestep) != -1:
    # Get raw bytes (BGRA format)
    raw_image = camera.getImage()
    
    if raw_image:
        # Convert bytes to a NumPy array for OpenCV
        img = np.frombuffer(raw_image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        
        success, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, CAM_EMITTER_QUALITY])
        
        if success:
            # Send the compressed bytes
            emitter.send(encoded_img.tobytes())
        else:
            print("[Camera] Failed to encode image")