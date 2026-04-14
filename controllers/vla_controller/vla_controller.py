"""
VLA Continuous Baseline Controller
Uses VLA model via a server endpoint to continuously predict delta movements.
Includes automated state-tracking for Baseline Logging.
"""

import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from utils.VLA import ACTION
# from utils.TEST import LOG_BASELINE 

from config import (
    PERSPECTIVE_RECEIVER, SIM_TABLE_DEF,
    GET_RECEIVER_BASE64_IMAGE, INFERENCE_EVERY_N, VLA_USER_PROMPTS
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
robot.move_to_world_target([current_x, current_y, current_z], settle_time_ms=0)
robot.operate_gripper(close=False)

# Tracking for baseline
has_picked = False
step_counter = 0

print("[Brain] VLA Continuous Loop starting...")

while supervisor.step(timestep) != -1:
    step_counter += 1
    if step_counter % INFERENCE_EVERY_N != 0: continue

    img = GET_RECEIVER_BASE64_IMAGE(receiver)
    if not img: continue

    # Call the VLA Brain
    action = ACTION(img, VLA_USER_PROMPTS["baseline"])
    
    if action is None:
        print("[System] Waiting for Brain...")
        continue 

    # --- THE FIX: Print the raw output and safely extract ---
    print("-" * 40)
    print(f"[Debug] Brain Output Array ({len(action)} dimensions):")
    print([round(a, 4) for a in action])
    
    dx, dy, dz, gripper_state = action[0], action[1], action[2], action[-1]

    print(f"[VLA] Raw Target -> X={dx:+.4f} | Y={dy:+.4f} | Z={dz:+.4f} | grip={gripper_state:.2f}")

    # ---------------------------------------------------------
    # CALIBRATION OFFSETS (Sim-to-AI Translation)
    # ---------------------------------------------------------
    # AI_ORIGIN is where the AI "thinks" its 0,0,0 coordinate is in Webots.
    # You will likely need to tweak these numbers slightly!
    AI_ORIGIN_X = float(table_pos[0])
    AI_ORIGIN_Y = float(table_pos[1])
    
    # If dz is around -0.45 when it should be touching the table,
    # then the AI's origin is about 45cm ABOVE the table.
    AI_ORIGIN_Z = TABLE_HEIGHT + 0.45 

    # Translate local AI target to global Webots target
    target_x = AI_ORIGIN_X + dx
    target_y = AI_ORIGIN_Y + dy
    target_z = AI_ORIGIN_Z + dz

    # Command the physical body
    robot.move_to_world_target([target_x, target_y, target_z], settle_time_ms=0)
    
    # --- REFEREE STATE MACHINE ---
    if gripper_state > 0.5:
        robot.operate_gripper(close=True)
        if not has_picked:
            print(" -> [State] VLA initiated Pick!")
            has_picked = True
    else:
        robot.operate_gripper(close=False)
        if has_picked:
            print(" -> [State] VLA initiated Drop! Ending sequence.")
            break 

# Final Baseline Log
# LOG_BASELINE(
#     supervisor=supervisor,
#     architecture="VLA",
#     task="Pick and Place"
# )