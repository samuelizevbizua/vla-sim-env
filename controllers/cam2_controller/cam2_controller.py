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

print(f"[Camera] Emitter initialized with channel: {emitter.getChannel()}")

while robot.step(timestep) != -1:
    # Get raw bytes (BGRA format)
    raw_image = camera.getImage()
    
    if raw_image:
        # Convert bytes to a NumPy array for OpenCV
        img = np.frombuffer(raw_image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        
        # Compress to JPEG (Quality 80 is a good balance for AI)
        success, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if success:
            # Send the compressed bytes
            emitter.send(encoded_img.tobytes())
        else:
            print("[Camera] Failed to encode image")