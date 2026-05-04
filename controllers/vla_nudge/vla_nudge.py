"""
VLA Nudge Controller (Continuous Reflex Test)
Tests if the end-to-end continuous policy can dynamically adjust 
to mid-flight target perturbations without a discrete state machine.
Integrated with standard testing metrics: Task Time, Nudge Error, and Success Rate.
"""

import os, sys, math
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import URe_Robot
from utils.VLA import ACTION
from utils.TEST import AUTOMATE_NUDGE, LOG_NUDGE_METRICS

from config import (
    CAM2_RECEIVER, CAM3_RECEIVER, SIM_TABLE_DEF, VLA_USER_PROMPTS,
    GET_RECEIVER_BASE64_IMAGE, INFERENCE_EVERY_N, TARGET_NUDGES,
    GRIPPER_LIFT_HEIGHT, MAX_TASK_TIME, TARGET_DEF
)
import numpy as np

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = URe_Robot(supervisor)

receiver_agent = supervisor.getDevice(CAM2_RECEIVER)
receiver_agent.enable(timestep)

receiver_wrist = supervisor.getDevice(CAM3_RECEIVER)
receiver_wrist.enable(timestep)

table_node = supervisor.getFromDef(SIM_TABLE_DEF)
table_pos  = table_node.getPosition()
TABLE_HEIGHT = table_node.getField("size").getSFVec3f()[2]

current_x = float(table_pos[0])
current_y = float(table_pos[1])
current_z = TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT

print("[System] Moving to start pose...")
robot.move_to_world_target([current_x, current_y, current_z], settle_time_ms=0)
robot.operate_gripper(close=False)

print(f"\n[Brain] Initializing VLA Nudge Test (Target Nudges: {TARGET_NUDGES})")
for _ in range(int(1000 / timestep)): supervisor.step(timestep)

# --- NUDGE TEST TRACKING VARIABLES ---
nudges_performed = 0
nudge_history = []
current_nudge_truth = None

start_time = supervisor.getTime()
predicted_pick = None
predicted_drop = None
final_pick_error = 999.0
task_complete = False
has_picked = False
step_counter = 0

current_prompt = VLA_USER_PROMPTS.get("nudge", "move the red object into the green container")

print("[Brain] VLA Continuous Loop starting...")

while supervisor.step(timestep) != -1:
    
    if supervisor.getTime() - start_time > MAX_TASK_TIME:
        print("\n[Error] VLA sequence timed out (Exceeded MAX_TASK_TIME).")
        break

    step_counter += 1
    if step_counter % INFERENCE_EVERY_N != 0: continue

    img_agent = GET_RECEIVER_BASE64_IMAGE(receiver_agent)
    img_wrist = GET_RECEIVER_BASE64_IMAGE(receiver_wrist)
    if not img_agent or not img_wrist: continue

    current_pose = robot.get_end_effector_pose()
    gripper_cmd = 1.0 if has_picked else 0.0
    current_state = current_pose + [gripper_cmd, gripper_cmd]

    action = ACTION(img_agent, img_wrist, current_state, current_prompt)    
    if action is None: continue

    current_action = action[0] if isinstance(action[0], list) else action
    dx, dy, dz, droll, dpitch, dyaw, delta_grip = current_action[:7]
    
    # TRANSLATE NEURAL SIGNAL TO PHYSICAL METERS
    ACTION_SCALE = 0.02
    dx *= ACTION_SCALE
    dy *= ACTION_SCALE
    dz *= ACTION_SCALE

    target_x = current_pose[0] + dx
    target_y = current_pose[1] + dy
    target_z = current_pose[2] + dz

    robot.move_to_world_target_async([target_x, target_y, target_z])
    
    # --- AUTOMATED PERTURBATION LOGIC ---
    if not has_picked and nudges_performed < TARGET_NUDGES:
        target_pos = supervisor.getFromDef(TARGET_DEF).getPosition()
        # Measure 3D spatial distance to see if end-effector is approaching the target
        dist_to_target = math.dist((target_x, target_y, target_z), (target_pos[0], target_pos[1], target_pos[2]))
        
        if dist_to_target < 0.10:
            print(f" -> [Referee] Arm got close! Triggering Nudge {nudges_performed + 1}/{TARGET_NUDGES}...")
            
            # Log the accuracy of the VLA's tracking right before we teleport it again
            if current_nudge_truth is not None:
                error = math.dist((target_x, target_y), current_nudge_truth)
                print(f" -> [Referee] VLA Tracked Nudge! Error: {error:.4f}m")
                nudge_history.append({'truth': current_nudge_truth, 'pred': (target_x, target_y), 'error': error})
            
            current_nudge_truth = AUTOMATE_NUDGE(supervisor)
            nudges_performed += 1
            continue
    # ------------------------------------

    # --- PICK & DROP LOGIC ---
    if delta_grip > 0.3:
        robot.operate_gripper(close=True)
        if not has_picked:
            print(" -> [State] VLA initiated Pick!")
            has_picked = True
            predicted_pick = (target_x, target_y)
            
            target_pos = supervisor.getFromDef(TARGET_DEF).getPosition()
            final_pick_error = math.dist((target_x, target_y), (target_pos[0], target_pos[1]))
            
            # Catch the final nudge tracking error right as it commits to grabbing
            if current_nudge_truth is not None:
                error = math.dist((target_x, target_y), current_nudge_truth)
                print(f" -> [Referee] VLA Tracked Final Nudge! Error: {error:.4f}m")
                nudge_history.append({'truth': current_nudge_truth, 'pred': (target_x, target_y), 'error': error})
                current_nudge_truth = None
            
    elif delta_grip < -0.3:
        robot.operate_gripper(close=False)
        if has_picked:
            print(" -> [State] VLA initiated Drop!")
            predicted_drop = (target_x, target_y)
            task_complete = True
            break 

# --- POST-TASK LOGGING ---
end_time = supervisor.getTime()

if task_complete and predicted_pick and predicted_drop:
    LOG_NUDGE_METRICS(
        supervisor=supervisor, architecture="VLA", 
        start_time=start_time, end_time=end_time,
        nudge_history=nudge_history, final_pick_error=final_pick_error, 
        predicted_drop=predicted_drop
    )
    print("\n[System] Sequence complete. Data logged.")
else:
    LOG_NUDGE_METRICS(
        supervisor=supervisor, architecture="VLA", 
        start_time=start_time, end_time=end_time,
        nudge_history=nudge_history, final_pick_error=final_pick_error, 
        predicted_drop=predicted_drop or (0.0, 0.0)
    )
    print("\n[System] Sequence failed or timed out. Logged as failure.")

print("\n[System] Entering idle mode.")
while supervisor.step(timestep) != -1: pass