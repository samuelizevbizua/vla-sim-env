import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Robot

CAMERA_FRAME_PATH = os.path.join(os.path.dirname(__file__), '../../tmp/camera_frame.png')

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
camera.enable(timestep)

# Ensure data directory exists
os.makedirs(os.path.dirname(CAMERA_FRAME_PATH), exist_ok=True)

print(f"Camera controller started. Saving frames to: {CAMERA_FRAME_PATH}")

while robot.step(timestep) != -1:
    # Saves a perfect, ready-to-use PNG image!
    camera.saveImage(CAMERA_FRAME_PATH, 100)