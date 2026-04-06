"""
VLA Nudge Controller (Continuous Tracking)
Tests if the end-to-end continuous policy can dynamically adjust 
to mid-flight target perturbations without a discrete state machine.
"""

import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from utils.VLA import ACTION

from config import (
    PERSPECTIVE_RECEIVER, SIM_TABLE_DEF, VLA_USER_PROMPTS,
    GET_RECEIVER_BASE64_IMAGE, DELTA_SCALE, INFERENCE_EVERY_N
)

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = UR10e_Robot(supervisor)

receiver = supervisor.getDevice(PERSPECTIVE_RECEIVER)
receiver.enable(timestep)

table_node = supervisor.getFromDef(SIM_TABLE_DEF)
table_pos  = table_node.getPosition()
TABLE_HEIGHT = table_node.getField("size").getSFVec3f()[2]

current_x = float(table_pos[0])
current_y = float(table_pos[1])
current_z = TABLE_HEIGHT + 0.2

print("[System] Moving to start pose...")
robot.move_to_world_target([current_x, current_y, current_z])
robot.operate_gripper(close=False)

for _ in range(int(1000 / timestep)): supervisor.step(timestep)

print("\n[Brain] VLA Dynamic Nudge Loop starting...")
print(" -> [Instruction] Please perturb the blue cube in the simulator now.\n")

step_counter = 0

while supervisor.step(timestep) != -1:
    step_counter += 1

    if step_counter % INFERENCE_EVERY_N != 0: continue

    img = GET_RECEIVER_BASE64_IMAGE(receiver)
    if not img: continue

    action = ACTION(img, VLA_USER_PROMPTS["nudge"])
    if action is None: continue 

    dx, dy, dz = action[0], action[1], action[2]
    gripper_state = action[6]

    print(f"[VLA Tracking] dx={dx:+.4f} | dy={dy:+.4f} | dz={dz:+.4f} | grip={gripper_state:.2f}")

    current_x += dx * DELTA_SCALE
    current_y += dy * DELTA_SCALE
    current_z += dz * DELTA_SCALE

    robot.move_to_world_target([current_x, current_y, current_z])
    robot.operate_gripper(close=(gripper_state > 0.5))