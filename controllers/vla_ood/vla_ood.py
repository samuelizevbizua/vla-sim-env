"""
VLA OOD Controller (Out-Of-Distribution Test)
Tests the end-to-end model's zero-shot semantic reasoning by asking it 
to interact with a novel, visually complex object (Jam Jar).
"""

import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import URe_Robot
from utils.VLA import ACTION
from utils.TEST import LOG_OOD

from config import (
    CAM2_RECEIVER, SIM_TABLE_DEF, VLA_USER_PROMPTS,
    GET_RECEIVER_BASE64_IMAGE, INFERENCE_EVERY_N
)

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = URe_Robot(supervisor)

receiver = supervisor.getDevice(CAM2_RECEIVER)
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

for _ in range(int(1000 / timestep)): supervisor.step(timestep)

print("\n[Brain] VLA OOD Loop starting...")

# --- NEW: Referee Tracking Variables ---
start_time = supervisor.getTime()
initial_target_pos = supervisor.getFromDef("OBJ2").getPosition()
predicted_pick = None
predicted_drop = None
has_picked = False
MAX_TRIAL_TIME_SEC = 120.0
# ---------------------------------------

step_counter = 0

while supervisor.step(timestep) != -1:
    step_counter += 1
    current_time = supervisor.getTime()

    if (current_time - start_time) > MAX_TRIAL_TIME_SEC:
        print("\n[Referee] Trial timed out!")
        break

    if step_counter % INFERENCE_EVERY_N != 0: continue

    img = GET_RECEIVER_BASE64_IMAGE(receiver)
    if not img: continue

    action = ACTION(img, VLA_USER_PROMPTS["ood"])
    if action is None: continue 

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
    current_x = AI_ORIGIN_X + dx
    current_y = AI_ORIGIN_Y + dy
    current_z = AI_ORIGIN_Z + dz

    robot.move_to_world_target([current_x, current_y, current_z], settle_time_ms=0)

    # --- NEW: State Machine to Track Pick/Drop for the Logger ---
    if gripper_state > 0.5:
        robot.operate_gripper(close=True)
        if not has_picked:
            print(" -> [State] VLA initiated Pick!")
            predicted_pick = (current_x, current_y)
            has_picked = True
    else:
        robot.operate_gripper(close=False)
        if has_picked:
            print(" -> [State] VLA initiated Drop! Ending sequence.")
            predicted_drop = (current_x, current_y)
            break
    # ------------------------------------------------------------

end_time = supervisor.getTime()

# Fallbacks in case it times out before tracking a coordinate
if predicted_pick is None: predicted_pick = (current_x, current_y)
if predicted_drop is None: predicted_drop = (current_x, current_y)

LOG_OOD(
    supervisor=supervisor,
    architecture="VLA",
    start_time=start_time,
    end_time=end_time,
    predicted_pick=predicted_pick,
    predicted_drop=predicted_drop,
    initial_target_pos=initial_target_pos,
    target_def="OBJ2",
    bin_def="BIN"
)

print("\n[System] Entering Terminal Idle State.")
while supervisor.step(timestep) != -1: pass