"""
VLA Controller using fine‑tuned SmolVLA model.
Sends three camera images + instruction, receives joint‑space actions.
"""

import os
import sys
import base64
import cv2

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import URe_Robot
from utils.VLA import ACTION
from config import (
    CAM1_RECEIVER, CAM2_RECEIVER, CAM2_NAME,
    CAM3_RECEIVER, SIM_TABLE_DEF, GET_RECEIVER_BASE64_IMAGE,
    GET_RECEIVER_DEPTH_BASE64, INFERENCE_EVERY_N, GRIPPER_LIFT_HEIGHT
)

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = URe_Robot(supervisor)

# Enable cameras
receiver_top = supervisor.getDevice(CAM1_RECEIVER)
receiver_top.enable(timestep)

receiver_hand = supervisor.getDevice(CAM3_RECEIVER)
receiver_hand.enable(timestep)

# Depth receiver
depth_receiver = supervisor.getDevice(CAM2_RECEIVER)
depth_receiver.enable(timestep)
depth_node = supervisor.getFromDef(CAM2_NAME)
MIN_DEPTH = depth_node.getField("minRange").getSFFloat()
MAX_DEPTH = depth_node.getField("maxRange").getSFFloat()
DEPTH_HEIGHT = depth_node.getField("height").getSFInt32()
DEPTH_WIDTH = depth_node.getField("width").getSFInt32()

table_node = supervisor.getFromDef(SIM_TABLE_DEF)
table_pos = table_node.getPosition()
TABLE_HEIGHT = table_node.getField("size").getSFVec3f()[2]

# Move to home
robot.move_to_world_target([table_pos[0], table_pos[1], TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT], settle_time_ms=500)
robot.operate_gripper(close=False)

instruction = "pick up the red cube and put it in the green bin"
has_picked = False
step_counter = 0

print("[Brain] VLA Continuous Loop starting...")

while supervisor.step(timestep) != -1:
    step_counter += 1
    if step_counter % INFERENCE_EVERY_N != 0:
        continue

    # Capture images
    top_img_b64 = GET_RECEIVER_BASE64_IMAGE(receiver_top, format="base64")
    hand_img_b64 = GET_RECEIVER_BASE64_IMAGE(receiver_hand, format="base64")
    depth_img_b64 = GET_RECEIVER_DEPTH_BASE64(depth_receiver, DEPTH_HEIGHT, DEPTH_WIDTH, MIN_DEPTH, MAX_DEPTH)

    if None in (top_img_b64, hand_img_b64, depth_img_b64): continue

    # Get action from model
    action = ACTION(top_img_b64, depth_img_b64, hand_img_b64, instruction)
    if action is None:
        print("[System] Waiting for Brain...")
        continue

    joint_targets = action[:6]
    gripper_cmd = action[6]

    # Execute directly
    robot.command_joint_positions(joint_targets, gripper_cmd)

    # Optional state machine
    if gripper_cmd > 0.5 and not has_picked:
        print(" -> [State] Gripper closed – assuming pick.")
        has_picked = True
        instruction = "place the cube in the green bin"