"""
VLA Continuous Baseline Controller
Uses VLA model via a server endpoint to continuously predict delta movements.
Includes automated state-tracking for Baseline Logging.
"""

import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import URe_Robot
from utils.VLA import ACTION
# from utils.TEST import LOG_BASELINE 

from config import (
    CAM2_RECEIVER, SIM_TABLE_DEF, CAM1_RECEIVER,
    GET_RECEIVER_BASE64_IMAGE, INFERENCE_EVERY_N, VLA_USER_PROMPTS,
    GRIPPER_TARGET_HOVER_HEIGHT, GRIPPER_LIFT_HEIGHT
)

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = URe_Robot(supervisor)

# receiver = supervisor.getDevice(CAM2_RECEIVER)
receiver = supervisor.getDevice(CAM1_RECEIVER)
receiver.enable(timestep)

table_node = supervisor.getFromDef(SIM_TABLE_DEF)
table_pos  = table_node.getPosition()
TABLE_HEIGHT = table_node.getField("size").getSFVec3f()[2]

current_x = float(table_pos[0])
current_y = float(table_pos[1])
current_z = TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT

print("[System] Moving to start pose...")
robot.move_to_world_target([current_x, current_y, current_z], settle_time_ms=0)
robot.operate_gripper(close=False)

# Tracking for baseline
has_picked = False
step_counter = 0
# current_prompt = "pick up the red cube"
current_prompt = "hover over the orange ball"

print("[Brain] VLA Continuous Loop starting...")

while supervisor.step(timestep) != -1:
    step_counter += 1
    if step_counter % INFERENCE_EVERY_N != 0: continue

    img = GET_RECEIVER_BASE64_IMAGE(receiver)
    if not img: continue

    # action = ACTION(img, VLA_USER_PROMPTS["baseline"])
    print(f"current prompt: {current_prompt}")
    action = ACTION(img, current_prompt)
    
    if action is None:
        print("[System] Waiting for Brain...")
        continue

    # [x,y,z p,y g]

    # 1. Extract based on SO-100 Joint Kinematics
    base_pan = action[0]
    shoulder = action[1]
    elbow    = action[2]
    gripper_state = action[5]

    print(f"[VLA] Joints -> Base={base_pan:+.2f} | Shoulder={shoulder:+.2f} | Elbow={elbow:+.2f}")

    # ---------------------------------------------------------
    # CALIBRATION OFFSETS & SCALING
    # ---------------------------------------------------------
    START_X = float(table_pos[0]) 
    START_Y = float(table_pos[1])
    START_Z = TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT

    # AI +Base = Right. Webots +X = Right.
    target_x = START_X + (base_pan * 0.5)    
    
    # AI -Elbow = Forward. Webots +Y = Forward.
    target_y = START_Y - (elbow * 0.4) 
    
    # AI -Shoulder = Down. 
    # If shoulder is -1.0, 0.35 + (-1.0 * 0.35) = 0.0 (It perfectly touches the table!)
    # target_z = START_Z + (shoulder * 0.35)
    target_z = START_Z

    # Command the physical body
    robot.move_to_world_target([target_x, target_y, target_z], settle_time_ms=0)
    
    # --- REFEREE STATE MACHINE ---
    if gripper_state > 0.5:
        robot.operate_gripper(close=True)
        if not has_picked:
            print(" -> [State] VLA initiated Pick! Changing instruction to Place.")
            has_picked = True
            print(f"picked cube, robot position at {supervisor.getSelf().getPosition()}")
            # Change the AI's goal now that the cube is grasped!
            # current_prompt = "place the pink cube in the green crate"
            
    else:
        robot.operate_gripper(close=False)
        if has_picked:
            # If it opens its gripper again AFTER having picked it up, it means it dropped it!
            print(" -> [State] VLA initiated Drop! Ending sequence.")
            break 

# Final Baseline Log
# LOG_BASELINE(
#     supervisor=supervisor,
#     architecture="VLA",
#     task="Pick and Place"
# )