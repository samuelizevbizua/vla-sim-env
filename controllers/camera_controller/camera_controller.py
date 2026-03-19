import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Robot
from config import CAM_FRAME_PATH

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
camera.enable(timestep)

print(f"Camera controller started. Saving frames to: {CAM_FRAME_PATH}")

step_count = 0
while robot.step(timestep) != -1:
    # Give the camera 10 steps to "warm up" visually
    # Then only save the image once every 30 steps (approx 1 second)
    # This prevents the file from being constantly locked/wiped!
    if step_count > 10 and step_count % 30 == 0:
        camera.saveImage(CAM_FRAME_PATH, 100)
        
    step_count += 1