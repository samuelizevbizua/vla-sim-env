"""
VLA Continuous Baseline Controller
Uses fine-tuned SmolVLA model via a server endpoint to continuously predict delta movements.
Integrated with standard testing metrics: Task Time, Pick Error, and Success Rate.
"""

import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import URe_Robot
from utils.VLA import ACTION
from utils.TEST import LOG_BASELINE  # Brought in the new testing framework

from config import (
    CAM2_RECEIVER, CAM3_RECEIVER, SIM_TABLE_DEF, 
    GET_RECEIVER_BASE64_IMAGE, INFERENCE_EVERY_N,
    GRIPPER_LIFT_HEIGHT, MAX_TASK_TIME, TARGET_DEF
)
import numpy as np

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = URe_Robot(supervisor)

# Get both cameras used in training (Agent view and Wrist view)
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

# --- TEST INITIALIZATION ---
print("\n[Brain] Initializing VLA Baseline (Continuous-Loop)")
for _ in range(int(1000 / timestep)):
    supervisor.step(timestep)

# Track the initial ground truth to measure Pick Error later
initial_cube_pos = supervisor.getFromDef(TARGET_DEF).getPosition()
start_time = supervisor.getTime()

predicted_pick = None
predicted_drop = None
task_complete = False
has_picked = False
step_counter = 0

# Must match one of the LANGUAGE_INSTRUCTIONS used in training
current_prompt = "move the red object into the green container" 

print("[Brain] VLA Continuous Loop starting...")

while supervisor.step(timestep) != -1:
    
    # 1. Enforce Timeout (Prevents infinite drift)
    if supervisor.getTime() - start_time > MAX_TASK_TIME:
        print("\n[Error] VLA sequence timed out (Exceeded MAX_TASK_TIME).")
        break

    step_counter += 1
    if step_counter % INFERENCE_EVERY_N != 0: continue

    # Grab both images
    img_agent = GET_RECEIVER_BASE64_IMAGE(receiver_agent)
    img_wrist = GET_RECEIVER_BASE64_IMAGE(receiver_wrist)
    
    if not img_agent or not img_wrist: continue

    # Grab the fresh, real-time 6D pose from the physical robot
    current_pose = robot.get_end_effector_pose()
    
    # Match the gripper command format used during training
    gripper_cmd = 1.0 if has_picked else 0.0
    
    # Build the exact 8-dim state array [x, y, z, roll, pitch, yaw, grip, grip]
    current_state = current_pose + [gripper_cmd, gripper_cmd]

    print(f"Executing prompt: {current_prompt}")

    # Pass the real-time state array to the ACTION function
    action = ACTION(img_agent, img_wrist, current_state, current_prompt)    
    if action is None:
        print("[System] Waiting for Brain...")
        continue
    else:
        print(f"[Brain] Action received: {action}")

    # Action output from model is 7 dims: [dx, dy, dz, droll, dpitch, dyaw, delta_grip]
    if isinstance(action[0], list):
        current_action = action[0]
    else:
        current_action = action
        
    dx, dy, dz, droll, dpitch, dyaw, delta_grip = current_action[:7]
    
    # TRANSLATE NEURAL SIGNAL [-1, 1] TO PHYSICAL METERS
    ACTION_SCALE = 0.02
    dx = dx * ACTION_SCALE
    dy = dy * ACTION_SCALE
    dz = dz * ACTION_SCALE
    
    print(f"[VLA] Deltas -> dx={dx:+.4f} | dy={dy:+.4f} | dz={dz:+.4f} | grip={delta_grip:+.2f}")

    # --- APPLY DELTA MOVEMENT ---
    current_pose = robot.get_end_effector_pose() # [x, y, z, roll, pitch, yaw]
    
    # Calculate new target pose by adding the SCALED deltas
    target_x = current_pose[0] + dx
    target_y = current_pose[1] + dy
    target_z = current_pose[2] + dz
    
    robot.move_to_world_target_async([target_x, target_y, target_z])
    
    # --- REFEREE STATE MACHINE & METRIC TRACKING ---
    if delta_grip > 0.3:
        robot.operate_gripper(close=True)
        if not has_picked:
            print(" -> [State] VLA initiated Pick!")
            has_picked = True
            # RECORD METRIC: The exact coordinate the VLA committed to the grasp
            predicted_pick = (target_x, target_y) 
            
    elif delta_grip < -0.3:
        robot.operate_gripper(close=False)
        if has_picked:
            print(" -> [State] VLA initiated Drop!")
            # RECORD METRIC: The exact coordinate the VLA released the object
            predicted_drop = (target_x, target_y) 
            task_complete = True
            break  # The sequence is over! Exit the loop to log metrics.

# --- POST-TASK LOGGING ---
end_time = supervisor.getTime()

if task_complete and predicted_pick and predicted_drop:
    LOG_BASELINE(
        supervisor=supervisor, architecture="VLA",
        start_time=start_time, end_time=end_time,
        predicted_pick=predicted_pick, predicted_drop=predicted_drop,
        initial_target_pos=initial_cube_pos
    )
    print("\n[System] Sequence complete. Data logged.")
else:
    # If it timed out or crashed, log it as an explicit failure
    LOG_BASELINE(
        supervisor=supervisor, architecture="VLA",
        start_time=start_time, end_time=end_time,
        predicted_pick=predicted_pick or (0.0, 0.0), 
        predicted_drop=predicted_drop or (0.0, 0.0),
        initial_target_pos=initial_cube_pos, 
        success_override=False
    )
    print("\n[System] Sequence failed or timed out. Logged as failure.")

print("\n[System] Entering idle mode.")
while supervisor.step(timestep) != -1:
    pass