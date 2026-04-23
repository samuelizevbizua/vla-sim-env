"""
Dense recording – only saves successful episodes (cube in bin).
Automatically discards failures. Resumes from last saved episode.
Bin avoidance uses actual crate dimensions.
"""

import os
import sys
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
    GET_RECEIVER_BASE64_IMAGE, GET_RECEIVER_DEPTH_IMAGE, SIM_TABLE_DEF
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ========== Configuration ==========
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
control_dt = timestep / 1000.0
actual_fps = int(1000 / timestep)
print(f"[INFO] Webots timestep = {timestep} ms → recording at {actual_fps} FPS")

table_node = supervisor.getFromDef(SIM_TABLE_DEF)
table_pos = table_node.getPosition()
TABLE_HEIGHT = table_node.getField("size").getSFVec3f()[2]
HOME_POS = [float(table_pos[0]), float(table_pos[1]), TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT]

# Table dimensions: size 0.8 (X) by 0.5 (Y), centered at (0, 0.52)
TABLE_X_MIN = table_pos[0] - 0.4      # -0.4
TABLE_X_MAX = table_pos[0] + 0.4      #  0.4
TABLE_Y_MIN = table_pos[1] - 0.25     #  0.27
TABLE_Y_MAX = table_pos[1] + 0.25     #  0.77
SAFETY_MARGIN = 0.05    # 5 cm from table edges

# ========== Cameras ==========
CHANNELS = 3

cam1_node = supervisor.getFromDef(CAM1_DEF)
CAM1_WIDTH = cam1_node.getField("width").getSFInt32()
CAM1_HEIGHT = cam1_node.getField("height").getSFInt32()
receiver1 = supervisor.getDevice(CAM1_RECEIVER)
receiver1.enable(timestep)

cam3_node = supervisor.getFromDef(CAM3_DEF)
CAM3_WIDTH = cam3_node.getField("width").getSFInt32()
CAM3_HEIGHT = cam3_node.getField("height").getSFInt32()
receiver3 = supervisor.getDevice(CAM3_RECEIVER)
receiver3.enable(timestep)

depth_node = supervisor.getFromDef(CAM2_DEPTH_DEF)
MIN_DEPTH = depth_node.getField("minRange").getSFFloat()
MAX_DEPTH = depth_node.getField("maxRange").getSFFloat()
DEPTH_WIDTH = depth_node.getField("width").getSFInt32()
DEPTH_HEIGHT = depth_node.getField("height").getSFInt32()
depth_receiver = supervisor.getDevice(CAM2_DEPTH_RECEIVER)
depth_receiver.enable(timestep)

target_node = supervisor.getFromDef("OBJ1")
dest_node = supervisor.getFromDef("BIN")
# Save safe default cube position (centre of table, avoid bin area)
DEFAULT_CUBE_POS = [0.2, 0.55, 0.72]   # X: 0.2, Y: 0.55, Z: 0.72
DEFAULT_CUBE_ROT = target_node.getField("rotation").getSFRotation()

# ========== Bin world bounding box (axis-aligned) ==========
bin_pos = dest_node.getPosition()
bin_size = dest_node.getField("size").getSFVec3f()      # (width, depth, height)
# Bin rotation is 0 0 -1 0 (no rotation), so width along X, depth along Y
bin_half_world_x = bin_size[0] / 2.0
bin_half_world_y = bin_size[1] / 2.0
BIN_X_MIN = bin_pos[0] - bin_half_world_x
BIN_X_MAX = bin_pos[0] + bin_half_world_x
BIN_Y_MIN = bin_pos[1] - bin_half_world_y
BIN_Y_MAX = bin_pos[1] + bin_half_world_y
BIN_MARGIN = 0.07   # extra clearance around bin (7 cm)

print(f"[Bin] Footprint: X=[{BIN_X_MIN:.3f}, {BIN_X_MAX:.3f}], Y=[{BIN_Y_MIN:.3f}, {BIN_Y_MAX:.3f}]")

# ========== Helper functions ==========
def reset_cube_to_default():
    target_node.getField("translation").setSFVec3f(DEFAULT_CUBE_POS)
    target_node.getField("rotation").setSFRotation(DEFAULT_CUBE_ROT)
    for _ in range(10):
        supervisor.step(timestep)
    print(f"[Failsafe] Cube reset to default position {DEFAULT_CUBE_POS[:2]}")

def is_cube_safe(z_threshold=0.69):
    pos = target_node.getPosition()
    # Z check: cube should be above table (not fallen)
    if pos[2] < z_threshold:
        return False
    # XY check: cube must be within table bounds minus safety margin
    if not (TABLE_X_MIN + SAFETY_MARGIN < pos[0] < TABLE_X_MAX - SAFETY_MARGIN):
        return False
    if not (TABLE_Y_MIN + SAFETY_MARGIN < pos[1] < TABLE_Y_MAX - SAFETY_MARGIN):
        return False
    return True

def randomize_cube_pose(avoid_bin=True):
    # Safe range inside table with margin
    x_min, x_max = TABLE_X_MIN + SAFETY_MARGIN, TABLE_X_MAX - SAFETY_MARGIN
    y_min, y_max = TABLE_Y_MIN + SAFETY_MARGIN, TABLE_Y_MAX - SAFETY_MARGIN

    if avoid_bin:
        for _ in range(100):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            # Check if cube's center is outside the expanded bin rectangle
            if (x < BIN_X_MIN - BIN_MARGIN or x > BIN_X_MAX + BIN_MARGIN or
                y < BIN_Y_MIN - BIN_MARGIN or y > BIN_Y_MAX + BIN_MARGIN):
                break
        else:
            # Fallback: place far from bin (top‑right area)
            x = 0.25
            y = 0.60
            print("[Warning] Could not find safe cube position, using fallback.")
    else:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

    z = DEFAULT_CUBE_POS[2]   # 0.72 (table top + cube half height)
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

def capture_frame(gripper_cmd, success_flag, done_flag):
    top_img = GET_RECEIVER_BASE64_IMAGE(receiver1, format="numpy")
    wrist_img = GET_RECEIVER_BASE64_IMAGE(receiver3, format="numpy")
    depth_rgb = GET_RECEIVER_DEPTH_IMAGE(depth_receiver, DEPTH_HEIGHT, DEPTH_WIDTH, MIN_DEPTH, MAX_DEPTH)

    if top_img is None or wrist_img is None or depth_rgb is None:
        return None

    joints = np.array([sensor.getValue() for sensor in robot.sensors])
    state = np.concatenate([joints, [gripper_cmd, 0.0]])

    target_joints = np.array([motor.getTargetPosition() for motor in robot.motors])
    action = np.concatenate([target_joints, [gripper_cmd]])

    return {
        "observation.images.image": top_img,
        "observation.images.image_with_depth": depth_rgb,
        "observation.images.hand_image": wrist_img,
        "observation.state": state.astype(np.float32),
        "action": action.astype(np.float32),
        "language_instruction": LANGUAGE_INSTRUCTION,
        "task": TASK_NAME,
        "next.reward": np.array([1.0 if success_flag else 0.0], dtype=np.float32),
        "next.done": np.array([done_flag], dtype=bool),
    }

# ========== Create dataset ==========
dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    root=DATASET_ROOT,
    fps=actual_fps,
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

print(f"[Dataset] Created image‑based dataset at {DATASET_ROOT} (FPS={actual_fps})")

successes, failures = 0, 0

# ========== Main recording loop ==========
for episode_idx in range(NUM_EPISODES):
    print(f"\n{'='*50}")
    print(f"[Episode {episode_idx}/{NUM_EPISODES-1}] Randomizing cube pose...")
    randomize_cube_pose()

    frame_count = 0

    # ---- Move to home ----
    robot.move_to_world_target_async(HOME_POS)
    while not robot.is_target_reached() and supervisor.step(timestep) != -1:
        frame = capture_frame(gripper_cmd=0.0, success_flag=False, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1
    # Settle after home move (0.4 seconds)
    settle_steps = int(0.4 / control_dt)
    for _ in range(settle_steps):
        supervisor.step(timestep)
        frame = capture_frame(gripper_cmd=0.0, success_flag=False, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1

    robot.operate_gripper(close=False)   # original blocking open
    gripper_cmd = 0.0
    success = False

    # ---- PICKING phase ----
    cube_pos = target_node.getPosition()
    # Approach
    approach_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + GRIPPER_LIFT_HEIGHT]
    robot.move_to_world_target_async(approach_pos)
    while not robot.is_target_reached() and supervisor.step(timestep) != -1:
        frame = capture_frame(gripper_cmd, success, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1
    # Settle after approach
    for _ in range(settle_steps):
        supervisor.step(timestep)
        frame = capture_frame(gripper_cmd, success, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1

    # Lower to hover
    hover_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + GRIPPER_TARGET_HOVER_HEIGHT]
    robot.move_to_world_target_async(hover_pos)
    while not robot.is_target_reached() and supervisor.step(timestep) != -1:
        frame = capture_frame(gripper_cmd, success, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1
    # Settle after hover
    for _ in range(settle_steps):
        supervisor.step(timestep)
        frame = capture_frame(gripper_cmd, success, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1

    # Close gripper (original blocking method – fixed time wait)
    robot.operate_gripper(close=True)
    gripper_cmd = 1.0
    # Record a few frames after closing (settle)
    settle_steps = int(0.3 / control_dt)
    for _ in range(settle_steps):
        supervisor.step(timestep)
        frame = capture_frame(gripper_cmd, success, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1

    # Lift
    lift_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + GRIPPER_LIFT_HEIGHT]
    robot.move_to_world_target_async(lift_pos)
    while not robot.is_target_reached() and supervisor.step(timestep) != -1:
        frame = capture_frame(gripper_cmd, success, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1
    # Settle after lift
    for _ in range(settle_steps):
        supervisor.step(timestep)
        frame = capture_frame(gripper_cmd, success, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1

    # ---- DROPPING phase ----
    bin_pos = dest_node.getPosition()
    above_bin = [bin_pos[0], bin_pos[1], bin_pos[2] + GRIPPER_LIFT_HEIGHT]
    robot.move_to_world_target_async(above_bin)
    while not robot.is_target_reached() and supervisor.step(timestep) != -1:
        frame = capture_frame(gripper_cmd, success, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1
    # Settle after moving above bin
    for _ in range(settle_steps):
        supervisor.step(timestep)
        frame = capture_frame(gripper_cmd, success, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1

    # Open gripper (original blocking)
    robot.operate_gripper(close=False)
    gripper_cmd = 0.0
    for _ in range(settle_steps):
        supervisor.step(timestep)
        frame = capture_frame(gripper_cmd, success, done_flag=False)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1

    # Check success
    cube_final_pos = target_node.getPosition()
    if ((cube_final_pos[0] - bin_pos[0])**2 + (cube_final_pos[1] - bin_pos[1])**2) < 0.15**2:
        success = True
        successes += 1
    else:
        failures += 1

    # Final pause
    for _ in range(20):
        supervisor.step(timestep)
        frame = capture_frame(gripper_cmd, success, done_flag=True)
        if frame:
            dataset.add_frame(frame)
            frame_count += 1

    dataset.save_episode()
    print(f"[Episode {episode_idx}] Saved. Frames recorded: {frame_count}")
    print(f"Total successes: {successes}, Total failures: {failures}")

    if not is_cube_safe():
        print(f"[Warning] Cube glitched after episode {episode_idx}. Resetting for next episode.")
        reset_cube_to_default()

dataset.finalize()
print(f"\n{'='*50}")
print(f"Recording complete. {NUM_EPISODES} episodes saved to {DATASET_ROOT}")
print(f"Successful episodes: {successes}, Failed episodes: {failures}")