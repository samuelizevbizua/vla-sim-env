"""
VLM Main Controller (PLAN-THEN-ACT) for Baseline Test
This controller tests a single, open-loop pick-and-place sequence. 
It takes ONE picture, asks the VLM for coordinates, and blindly executes 
the entire sequence. This serves as the non-reactive baseline for comparison.
"""

import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from utils.VLM import ASK
from utils.TEST import LOG_BASELINE

from config import (
    OVERHEAD_CAM_DEF, 
    OVERHEAD_RECEIVER, 
    SIM_TABLE_DEF, 
    GET_RECEIVER_BASE64_IMAGE,
    VLM_USER_PROMPTS,
    GRIPPER_TARGET_HOVER_HEIGHT, 
    GRIPPER_LIFT_HEIGHT,
    PIXEL_TO_WORLD
)

# Initialize Webots
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = UR10e_Robot(supervisor)

receiver = supervisor.getDevice(OVERHEAD_RECEIVER)
receiver.enable(timestep)

cam_node = supervisor.getFromDef(OVERHEAD_CAM_DEF)
CAM_WIDTH = cam_node.getField("width").getSFInt32()
CAM_HEIGHT = cam_node.getField("height").getSFInt32()
CAM_FOV = cam_node.getField("fieldOfView").getSFFloat()

cam_rig = cam_node.getParentNode()
CAM_X, CAM_Y, CAM_Z = cam_rig.getPosition()

TABLE_HEIGHT = supervisor.getFromDef(SIM_TABLE_DEF).getField("size").getSFVec3f()[2]
_height = CAM_Z - TABLE_HEIGHT

print("\n[Brain] Initializing Single-Pass VLM Test")

# Boot-up pause
for _ in range(int(1000 / timestep)): supervisor.step(timestep)

start_time = supervisor.getTime()
initial_cube_pos = supervisor.getFromDef("OBJ1").getPosition()

img = GET_RECEIVER_BASE64_IMAGE(receiver)

if img is not None:
    content = [
        {"type": "text", "text": VLM_USER_PROMPTS["baseline"]},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    ]

    try:
        print("\n[Vision] Snapping baseline picture and asking VLM...")

        args = ASK(content)
        print(f" -> [Brain] Output: {args}")

        pick_x, pick_y = PIXEL_TO_WORLD(args['cube_x_norm'], args['cube_y_norm'], _height, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y)
        drop_x, drop_y = PIXEL_TO_WORLD(args['bin_x_norm'], args['bin_y_norm'], _height, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y)

        print(f"\n[Muscle] Executing Blind Pick-and-Place sequence...")
        
        robot.operate_gripper(close=False) 
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT]) 
        
        print(" -> Plunging to grab...")
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_TARGET_HOVER_HEIGHT]) 
        robot.operate_gripper(close=True)                                       
        
        print(" -> Lifting to bin...")
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT]) 
        robot.move_to_world_target([drop_x, drop_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
        
        print(" -> Releasing...")
        robot.operate_gripper(close=False)

        # --- END REFEREE LOGGING ---
        end_time = supervisor.getTime()
        
        LOG_BASELINE(
            supervisor=supervisor, 
            architecture="VLM", 
            start_time=start_time, 
            end_time=end_time, 
            predicted_pick=(pick_x, pick_y), 
            predicted_drop=(drop_x, drop_y), 
            initial_target_pos=initial_cube_pos, 
            target_def="OBJ1", 
            bin_def="BIN"
        )
        print("\n[System] Sequence Complete!")

    except Exception as e:
        print(f"[Error] Agent failed: {str(e)}")
else:
    print("[Error] Failed to capture image.")

print("\n[System] Entering Terminal Idle State.")
while supervisor.step(timestep) != -1: pass