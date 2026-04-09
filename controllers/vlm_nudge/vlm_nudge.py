"""
VLM Nudge Controller (SENSE-ACT-VERIFY)
This controller implements a reactive waypoint system.
It uses the VLM to identify objects, then verifies their position 
after moving to ensure a "nudge" didn't occur during transit.
"""

import os, sys, math
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from utils.VLM import ASK, SELECT_VLM

from config import (
    OVERHEAD_CAM_DEF, 
    OVERHEAD_RECEIVER, 
    SIM_TABLE_DEF, 
    GET_RECEIVER_BASE64_IMAGE,
    VLM_USER_PROMPTS,
    GRIPPER_TARGET_HOVER_HEIGHT, 
    GRIPPER_LIFT_HEIGHT,
    PIXEL_TO_WORLD,
    NUDGE_TOLERANCE_NORM
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

llm = SELECT_VLM()

# --- MAIN REACTIVE LOOP ---
task_state = "PICKING" #PICKING, DROPPING, DONE
prev_coords = None

print("\n[Brain] Starting Reactive Nudge Test Controller...")

while task_state != "DONE" and supervisor.step(timestep) != -1:
    img = GET_RECEIVER_BASE64_IMAGE(receiver)
    if not img: continue

    content = [
        {"type": "text", "text": VLM_USER_PROMPTS["nudge"]},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    ]

    if task_state == "PICKING":
        print(f"\n[Sense] Looking for Cube...")
        
        data = ASK(llm, content)
        
        if data.get("found") is False or "x" not in data or "y" not in data:
            print(" -> Cube not found or coordinates missing. Retrying...")
            continue

        curr_coords = (data['x'], data['y'])
        wx, wy = PIXEL_TO_WORLD(curr_coords[0], curr_coords[1], _height, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y)
        
        if prev_coords is None:
            print(f" -> Cube sensed at norm {curr_coords}. Moving to Hover...")
            robot.operate_gripper(close=False)
            robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
            
            prev_coords = curr_coords
            continue 
            
        else:
            dist = math.dist(prev_coords, curr_coords)
            if dist > NUDGE_TOLERANCE_NORM:
                print(f" -> [!] NUDGE DETECTED (Moved {dist:.3f}). Re-tracking...")
                prev_coords = None 
                continue
            else:
                print(" -> Target verified stable. Executing Grab...")
                robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_TARGET_HOVER_HEIGHT])

                robot.operate_gripper(close=True)
                robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
                
                task_state = "DROPPING"
                prev_coords = None 

    elif task_state == "DROPPING":
        print(f"\n[Sense] Looking for Bin...")
        
        data = ASK(llm, content)
        
        if data.get("found") is False or "x" not in data or "y" not in data:
            print(" -> Bin not found or coordinates missing. Retrying...")
            continue

        curr_coords = (data['x'], data['y'])
        wx, wy = PIXEL_TO_WORLD(curr_coords[0], curr_coords[1], _height, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y)
        
        print(f" -> Bin sensed at {curr_coords}. Moving to Drop Zone...")
        robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
        robot.operate_gripper(close=False)

print("\n[System] Task Complete.")
while supervisor.step(timestep) != -1: pass