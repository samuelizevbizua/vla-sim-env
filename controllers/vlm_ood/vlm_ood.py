"""
VLM OOD Controller (Out-Of-Distribution Test)
Tests the model's zero-shot semantic reasoning by asking it to interact 
with a novel object using a single-pass, static environment sequence.
"""

import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from utils.VLM import ASK
from utils.TEST import LOG_OOD

from config import (
    OVERHEAD_CAM_DEF, OVERHEAD_RECEIVER, SIM_TABLE_DEF, GET_RECEIVER_BASE64_IMAGE,
    VLM_USER_PROMPTS, GRIPPER_TARGET_HOVER_HEIGHT, GRIPPER_LIFT_HEIGHT, PIXEL_TO_WORLD
)

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = UR10e_Robot(supervisor)

receiver = supervisor.getDevice(OVERHEAD_RECEIVER)
receiver.enable(timestep)

cam_node = supervisor.getFromDef(OVERHEAD_CAM_DEF)
CAM_WIDTH, CAM_HEIGHT = cam_node.getField("width").getSFInt32(), cam_node.getField("height").getSFInt32()
CAM_X, CAM_Y, CAM_Z = cam_node.getParentNode().getPosition()
CAM_FOV = cam_node.getField("fieldOfView").getSFFloat()

TABLE_HEIGHT = supervisor.getFromDef(SIM_TABLE_DEF).getField("size").getSFVec3f()[2]
_height = CAM_Z - TABLE_HEIGHT

print("\n[Brain] Initializing OOD Single-Pass VLM Test")

for _ in range(int(1000 / timestep)): supervisor.step(timestep)

start_time = supervisor.getTime()
initial_target_pos = supervisor.getFromDef("OBJ2").getPosition()

img = GET_RECEIVER_BASE64_IMAGE(receiver)

if img is not None:
    content = [
        {"type": "text", "text": VLM_USER_PROMPTS["ood_test"]},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    ]

    try:
        print("\n[Vision] Snapping OOD picture and asking VLM...")
        args = ASK(content)
        print(f" -> [Brain] Output: {args}")

        # Safety Check for the NEW JSON keys
        if args.get("found") is False or "target_x_norm" not in args:
            print(" -> [Error] VLM failed to identify the OOD target. Aborting.")
            raise ValueError("Incomplete JSON.")

        # Using target_x_norm instead of cube_x_norm
        pick_x, pick_y = PIXEL_TO_WORLD(args['target_x_norm'], args['target_y_norm'], _height, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y)
        drop_x, drop_y = PIXEL_TO_WORLD(args['bin_x_norm'], args['bin_y_norm'], _height, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y)

        print(f"\n[Muscle] Executing OOD Pick-and-Place sequence...")
        
        robot.operate_gripper(close=False)
        
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT]) 
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_TARGET_HOVER_HEIGHT]) 
        robot.operate_gripper(close=True)   
        
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT]) 
        robot.move_to_world_target([drop_x, drop_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
        robot.operate_gripper(close=False)

        end_time = supervisor.getTime()
        
        LOG_OOD(
            supervisor=supervisor, architecture="VLM", start_time=start_time, end_time=end_time,
            predicted_pick=(pick_x, pick_y), predicted_drop=(drop_x, drop_y),
            initial_target_pos=initial_target_pos, target_def="OBJ2", bin_def="BIN"
        )

        print("\n[System] OOD Sequence Complete!")

    except Exception as e:
        print(f"[Error] Agent failed: {str(e)}")

print("\n[System] Entering Terminal Idle State.")
while supervisor.step(timestep) != -1: pass