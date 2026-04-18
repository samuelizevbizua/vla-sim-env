"""
Record a pick-and-place episode (blue cube → green bin) using the UR5e IK controller,
and save the data in LeRobot v3.0 format.
"""

import os
import sys
import time
import numpy as np

# Add project root to path (so that utils.UR can be found)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from config import (
    CAM1_DEF, CAM3_DEF, GRIPPER_TARGET_HOVER_HEIGHT, GRIPPER_LIFT_HEIGHT,
    CAM1_RECEIVER, CAM3_RECEIVER, GET_RECEIVER_BASE64_IMAGE
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ========== Configuration ==========
FPS = 5
LANGUAGE_INSTRUCTION = "pick up the blue cube and put it in the green bin"
EPISODE_INDEX = 0               # Change for each episode
DATASET_ROOT = "./recorded_dataset"
REPO_ID = "local/ur5e_pickplace"
TASK_NAME = "pick_place_cube"

# ========== Initialise Webots and robot ==========
supervisor = Supervisor()
robot = UR10e_Robot(supervisor)
timestep = int(supervisor.getBasicTimeStep())

# Camera receiver (top‑down camera)
cam1_node = supervisor.getFromDef(CAM1_DEF)
receiver = supervisor.getDevice(CAM1_RECEIVER)
receiver.enable(timestep)

cam3_node = supervisor.getFromDef(CAM3_DEF)
receiver3 = supervisor.getDevice(CAM3_RECEIVER)
receiver3.enable(timestep)

CAM1_WIDTH = cam1_node.getField("width").getSFInt32()
CAM1_HEIGHT = cam1_node.getField("height").getSFInt32()

CAM3_WIDTH = cam3_node.getField("width").getSFInt32()
CAM3_HEIGHT = cam3_node.getField("height").getSFInt32()

CHANNELS = 3  # RGB

# Object and bin references
target = supervisor.getFromDef("OBJ1")
dest = supervisor.getFromDef("BIN")

# ========== Create LeRobot dataset ==========
dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    root=DATASET_ROOT,
    fps=FPS,
    features={
        "observation.images.top": {
            "dtype": "image",
            "shape": (CAM1_HEIGHT, CAM1_WIDTH, CHANNELS),
            "names": ["height", "width", "channels"]
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (CAM3_HEIGHT, CAM3_WIDTH, CHANNELS),
            "names": ["height", "width", "channels"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["state"]
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["action"]
        },
        "language_instruction": {
            "dtype": "string",
            "shape": (1,),
            "names": None
        }
    },
    use_videos=True,
)

print(f"[Record] Episode {EPISODE_INDEX} -> {DATASET_ROOT}")

# ========== Main pick-and-place loop ==========
task_state = "PICKING"
frame_idx = 0
start_time = time.time()
gripper_cmd = 0.0          # 0 = open, 1 = closed

while task_state != "DONE" and supervisor.step(timestep) != -1:
    top_img = GET_RECEIVER_BASE64_IMAGE(receiver, format="numpy")
    wrist_img = GET_RECEIVER_BASE64_IMAGE(receiver3, format="numpy")

    if top_img is None or wrist_img is None:
        continue   # wait for next packet

    # --- Robot state (6 joint angles + gripper command) ---
    joints = np.array([sensor.getValue() for sensor in robot.sensors])
    state = np.concatenate([joints, [gripper_cmd]])

    # --- Execute IK logic (same as ik_controller.py) ---
    if task_state == "PICKING":
        pos = target.getPosition()

        # Open gripper
        robot.operate_gripper(close=False)
        gripper_cmd = 0.0
        # Move to hover
        robot.move_to_world_target([pos[0], pos[1], pos[2] + GRIPPER_LIFT_HEIGHT])
        # Move down to grasp
        robot.move_to_world_target([pos[0], pos[1], pos[2] + GRIPPER_TARGET_HOVER_HEIGHT])
        # Close gripper
        robot.operate_gripper(close=True)
        gripper_cmd = 1.0
        # Lift
        robot.move_to_world_target([pos[0], pos[1], pos[2] + GRIPPER_LIFT_HEIGHT])

        task_state = "DROPPING"

    elif task_state == "DROPPING":
        dest_pos = dest.getPosition()
        # Move above bin
        robot.move_to_world_target([dest_pos[0], dest_pos[1], dest_pos[2] + GRIPPER_LIFT_HEIGHT])
        # Open gripper to drop
        robot.operate_gripper(close=False)
        gripper_cmd = 0.0
        task_state = "DONE"

    # --- Action = target joint positions (the command we just sent) ---
    target_joints = np.array([motor.getTargetPosition() for motor in robot.motors])
    action = np.concatenate([target_joints, [gripper_cmd]])

    # --- Add frame to dataset ---
    dataset.add_frame({
        "observation.images.top": top_img,
        "observation.images.wrist": wrist_img,
        "observation.state": state.astype(np.float32),
        "action": action.astype(np.float32),
        "language_instruction": LANGUAGE_INSTRUCTION,
        "task": TASK_NAME,
    })
    frame_idx += 1

    # Maintain FPS
    time.sleep(1.0 / FPS)

# --- Mark episode as finished ---
dataset.save_episode()
dataset.finalize()

print(f"[Record] Episode {EPISODE_INDEX} saved. Dataset root: {DATASET_ROOT}")
print("Visualise with: lerobot_dataset_viz --repo_id local/ur5e_pickplace --root ./recorded_dataset")