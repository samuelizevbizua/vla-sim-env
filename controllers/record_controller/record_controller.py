"""
Record multiple pick-and-place episodes (blue cube → green bin) using the UR5e IK controller,
with random cube poses per episode, and save data in LeRobot v3.0 format.
All cameras store PNG images (no video encoding). Depth is converted to 3‑channel RGB (uint8).
"""

import os
import sys
import time
import random
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from config import (
    CAM1_DEF, CAM2_DEPTH_DEF, CAM3_DEF,
    GRIPPER_TARGET_HOVER_HEIGHT, GRIPPER_LIFT_HEIGHT,
    CAM1_RECEIVER, CAM3_RECEIVER, CAM2_DEPTH_RECEIVER,
    GET_RECEIVER_BASE64_IMAGE, SIM_TABLE_DEF
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ========== Configuration ==========
FPS = 5
LANGUAGE_INSTRUCTION = "pick up the blue cube and put it in the green bin"
NUM_EPISODES = 50
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "recorded_dataset")
REPO_ID = "local/ur5e_pickplace"
TASK_NAME = "pick_place_cube"

# ========== Initialise Webots and robot ==========
supervisor = Supervisor()
robot = UR10e_Robot(supervisor)
timestep = int(supervisor.getBasicTimeStep())

table_node = supervisor.getFromDef(SIM_TABLE_DEF)
table_pos = table_node.getPosition()
TABLE_HEIGHT = table_node.getField("size").getSFVec3f()[2]
HOME_POS = [float(table_pos[0]), float(table_pos[1]), TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT]

# ========== Cameras ==========
CHANNELS = 3

# CAM1 (top RGB)
cam1_node = supervisor.getFromDef(CAM1_DEF)
CAM1_WIDTH = cam1_node.getField("width").getSFInt32()
CAM1_HEIGHT = cam1_node.getField("height").getSFInt32()
receiver1 = supervisor.getDevice(CAM1_RECEIVER)
receiver1.enable(timestep)

# CAM3 (wrist RGB)
cam3_node = supervisor.getFromDef(CAM3_DEF)
CAM3_WIDTH = cam3_node.getField("width").getSFInt32()
CAM3_HEIGHT = cam3_node.getField("height").getSFInt32()
receiver3 = supervisor.getDevice(CAM3_RECEIVER)
receiver3.enable(timestep)

# ========== Depth sensor ==========
depth_node = supervisor.getFromDef(CAM2_DEPTH_DEF)
MIN_DEPTH = depth_node.getField("minRange").getSFFloat()
MAX_DEPTH = depth_node.getField("maxRange").getSFFloat()
DEPTH_WIDTH = depth_node.getField("width").getSFInt32()
DEPTH_HEIGHT = depth_node.getField("height").getSFInt32()

depth_receiver = supervisor.getDevice(CAM2_DEPTH_RECEIVER)
depth_receiver.enable(timestep)

# ========== Object references ==========
target_node = supervisor.getFromDef("OBJ1")
dest_node = supervisor.getFromDef("BIN")
DEFAULT_CUBE_POS = target_node.getPosition()
DEFAULT_CUBE_ROT = target_node.getField("rotation").getSFRotation()

settle_time_ms = 600

# ========== Helper functions ==========
def reset_cube_to_default():
    target_node.getField("translation").setSFVec3f(DEFAULT_CUBE_POS)
    target_node.getField("rotation").setSFRotation(DEFAULT_CUBE_ROT)
    for _ in range(10):
        supervisor.step(timestep)
    print(f"[Failsafe] Cube reset to default position {DEFAULT_CUBE_POS[:2]}")

def is_cube_safe(z_threshold=0.69):
    pos = target_node.getPosition()
    if pos[2] < z_threshold:
        return False
    if not (-0.5 < pos[0] < 0.5) or not (0.15 < pos[1] < 0.9):
        return False
    return True

def randomize_cube_pose(avoid_bin=True):
    x_min, x_max = -0.35, 0.35
    y_min, y_max = 0.25, 0.75
    if avoid_bin:
        bin_pos = dest_node.getPosition()
        bin_radius = 0.15
        for _ in range(50):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            if ((x - bin_pos[0])**2 + (y - bin_pos[1])**2) > bin_radius**2:
                break
    else:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

    z = DEFAULT_CUBE_POS[2]
    target_node.getField("translation").setSFVec3f([x, y, z])
    yaw = random.uniform(-3.14159, 3.14159)
    target_node.getField("rotation").setSFRotation([0, 0, 1, yaw])

    for _ in range(10):
        supervisor.step(timestep)

    if not is_cube_safe():
        print("[Warning] Cube glitched after randomisation! Resetting to default.")
        reset_cube_to_default()
        x, y = DEFAULT_CUBE_POS[0], DEFAULT_CUBE_POS[1]
        yaw = DEFAULT_CUBE_ROT[3]
    else:
        print(f"[Randomize] Cube moved to ({x:.3f}, {y:.3f}) with yaw {yaw:.3f} rad")
    return (x, y, yaw)

# ========== Create dataset (images, no videos) ==========
dataset_path = Path(DATASET_ROOT)
dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    root=DATASET_ROOT,
    fps=FPS,
    robot_type="unknown",
    features={
        "observation.images.image": {
            "dtype": "image",
            "shape": (CAM1_HEIGHT, CAM1_WIDTH, CHANNELS),
            "names": ["height", "width", "channels"]
        },
        "observation.images.image_with_depth": {
            "dtype": "image",
            "shape": (DEPTH_HEIGHT, DEPTH_WIDTH, CHANNELS),
            "names": ["height", "width", "channels"]
        },
        "observation.images.hand_image": {
            "dtype": "image",
            "shape": (CAM3_HEIGHT, CAM3_WIDTH, CHANNELS),
            "names": ["height", "width", "channels"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
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
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None
        }
    },
    use_videos=False,
)

print(f"[Dataset] Created new image‑based dataset at {DATASET_ROOT}")

# ========== Main recording loop ==========
for episode_idx in range(NUM_EPISODES):
    print(f"\n{'='*50}")
    print(f"[Episode {episode_idx}/{NUM_EPISODES-1}] Randomizing cube pose...")
    randomize_cube_pose()

    robot.move_to_world_target(HOME_POS, wait=True, settle_time_ms=settle_time_ms)
    robot.operate_gripper(close=False)
    gripper_cmd = 0.0

    task_state = "PICKING"
    frame_idx = 0
    start_time = time.time()
    success = False

    while task_state != "DONE" and supervisor.step(timestep) != -1:
        # Capture RGB images
        top_img = GET_RECEIVER_BASE64_IMAGE(receiver1, format="numpy")
        wrist_img = GET_RECEIVER_BASE64_IMAGE(receiver3, format="numpy")
        if top_img is None or wrist_img is None:
            continue

        # Receive depth data
        depth_bytes = None
        while depth_receiver.getQueueLength() > 0:
            depth_bytes = depth_receiver.getBytes()
            depth_receiver.nextPacket()
        if depth_bytes is None:
            continue

        # Convert depth to 3‑channel RGB (uint8)
        depth_img = np.frombuffer(depth_bytes, dtype=np.float32).reshape((DEPTH_HEIGHT, DEPTH_WIDTH))
        depth_norm = (depth_img - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
        depth_norm = np.clip(depth_norm, 0, 1)
        depth_rgb = np.stack([depth_norm, depth_norm, depth_norm], axis=-1)  # float32 in [0,1]
        depth_rgb = (depth_rgb * 255).astype(np.uint8)  # convert to uint8 for image storage

        # Robot state (6 joints + gripper + dummy zero)
        joints = np.array([sensor.getValue() for sensor in robot.sensors])
        state = np.concatenate([joints, [gripper_cmd, 0.0]])

        # IK logic
        if task_state == "PICKING":
            pos = target_node.getPosition()
            robot.operate_gripper(close=False)
            gripper_cmd = 0.0
            robot.move_to_world_target([pos[0], pos[1], pos[2] + GRIPPER_LIFT_HEIGHT], settle_time_ms=settle_time_ms)
            robot.move_to_world_target([pos[0], pos[1], pos[2] + GRIPPER_TARGET_HOVER_HEIGHT], settle_time_ms=settle_time_ms)
            robot.operate_gripper(close=True)
            gripper_cmd = 1.0
            robot.move_to_world_target([pos[0], pos[1], pos[2] + GRIPPER_LIFT_HEIGHT], settle_time_ms=settle_time_ms)
            task_state = "DROPPING"

        elif task_state == "DROPPING":
            dest_pos = dest_node.getPosition()
            robot.move_to_world_target([dest_pos[0], dest_pos[1], dest_pos[2] + GRIPPER_LIFT_HEIGHT], settle_time_ms=settle_time_ms)
            robot.operate_gripper(close=False)
            gripper_cmd = 0.0
            task_state = "DONE"
            # Check success
            cube_final_pos = target_node.getPosition()
            if ((cube_final_pos[0] - dest_pos[0])**2 + (cube_final_pos[1] - dest_pos[1])**2) < 0.15**2:
                success = True

        # Action = commanded joint positions
        target_joints = np.array([motor.getTargetPosition() for motor in robot.motors])
        action = np.concatenate([target_joints, [gripper_cmd]])

        # Add frame (all images are uint8)
        dataset.add_frame({
            "observation.images.image": top_img,
            "observation.images.image_with_depth": depth_rgb,
            "observation.images.hand_image": wrist_img,
            "observation.state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "language_instruction": LANGUAGE_INSTRUCTION,
            "task": TASK_NAME,
            "next.reward": np.array([1.0 if success and task_state == "DONE" else 0.0], dtype=np.float32),
            "next.done": np.array([task_state == "DONE"], dtype=bool),
        })
        frame_idx += 1
        time.sleep(1.0 / FPS)

    dataset.save_episode()
    elapsed = time.time() - start_time
    print(f"[Episode {episode_idx}] Saved. Duration: {elapsed:.2f}s, Frames: {frame_idx}")

    if not is_cube_safe():
        print(f"[Warning] Cube glitched after episode {episode_idx}. Resetting for next episode.")
        reset_cube_to_default()

dataset.finalize()
print(f"\n{'='*50}")
print(f"Recording complete. {NUM_EPISODES} episodes saved to {DATASET_ROOT}")
print(f"Visualise with: lerobot-dataset-viz --repo-id {REPO_ID} --root {DATASET_ROOT} --episode-index 0")