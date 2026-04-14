"""
VLA Nudge Controller (Continuous Reflex Test)
Tests if the end-to-end continuous policy can dynamically adjust 
to mid-flight target perturbations without a discrete state machine.
"""

import os, sys, math
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from utils.VLA import ACTION
from utils.TEST import AUTOMATE_NUDGE, LOG_NUDGE_METRICS

from config import (
    PERSPECTIVE_RECEIVER, SIM_TABLE_DEF, VLA_USER_PROMPTS,
    GET_RECEIVER_BASE64_IMAGE, INFERENCE_EVERY_N, TARGET_NUDGES
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

for _ in range(int(1000 / timestep)): supervisor.step(timestep)

# --- NUDGE TEST TRACKING VARIABLES ---
nudges_performed = 0
nudge_history = []
current_nudge_truth = None
# ------------------------------------------

start_time = supervisor.getTime()
predicted_drop = None
final_pick_error = 999.0
has_picked = False
step_counter = 0

print(f"\n[Brain] Starting Automated VLA Nudge Test (Target Nudges: {TARGET_NUDGES})...")

while supervisor.step(timestep) != -1:
    step_counter += 1

    if step_counter % INFERENCE_EVERY_N != 0: continue

    img = GET_RECEIVER_BASE64_IMAGE(receiver)
    if not img: continue

    action = ACTION(img, VLA_USER_PROMPTS["nudge"])
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
    
    # --- AUTOMATED PERTURBATION LOGIC ---
    if not has_picked and nudges_performed < TARGET_NUDGES:
        target_pos = supervisor.getFromDef("OBJ1").getPosition()
        dist_to_target = math.dist((current_x, current_y), (target_pos[0], target_pos[1]))
        
        # If the arm gets within 10cm of the cube, teleport it away!
        if dist_to_target < 0.10:
            print(f" -> [Referee] Arm got close! Triggering Nudge {nudges_performed + 1}/{TARGET_NUDGES}...")
            
            # Log the accuracy of the VLA's tracking right before we teleport it again
            if current_nudge_truth is not None:
                error = math.dist((current_x, current_y), current_nudge_truth)
                print(f" -> [Referee] VLA Tracked Nudge! Error: {error:.4f}m")
                nudge_history.append({'truth': current_nudge_truth, 'pred': (current_x, current_y), 'error': error})
            
            current_nudge_truth = AUTOMATE_NUDGE(supervisor)
            nudges_performed += 1
            continue
    # ------------------------------------

    # --- PICK & DROP LOGIC ---
    if gripper_state > 0.5:
        robot.operate_gripper(close=True)
        if not has_picked:
            print(" -> [State] VLA initiated Pick!")
            target_pos = supervisor.getFromDef("OBJ1").getPosition()
            final_pick_error = math.dist((current_x, current_y), (target_pos[0], target_pos[1]))
            
            # Catch the final nudge tracking error right as it commits to grabbing
            if current_nudge_truth is not None:
                error = math.dist((current_x, current_y), current_nudge_truth)
                print(f" -> [Referee] VLA Tracked Final Nudge! Error: {error:.4f}m")
                nudge_history.append({'truth': current_nudge_truth, 'pred': (current_x, current_y), 'error': error})
                current_nudge_truth = None

            has_picked = True
            
    else:
        robot.operate_gripper(close=False)
        if has_picked:
            print(" -> [State] VLA initiated Drop!")
            predicted_drop = (current_x, current_y)
            break 

end_time = supervisor.getTime()

LOG_NUDGE_METRICS(
    supervisor=supervisor, 
    architecture="VLA", 
    start_time=start_time, 
    end_time=end_time,
    nudge_history=nudge_history, 
    final_pick_error=final_pick_error, 
    predicted_drop=predicted_drop
)

print("\n[System] Test Logged Successfully.")
while supervisor.step(timestep) != -1: pass