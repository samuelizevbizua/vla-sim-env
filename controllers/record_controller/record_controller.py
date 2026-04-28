"""
Record episodes in LIBERO format for SmolVLA fine-tuning.
- Two cameras: agent view (image) and wrist view (image2)
- State: [x, y, z, roll, pitch, yaw, grip, grip]  (8 dims)
- Action: [dx, dy, dz, droll, dpitch, dyaw, delta_grip] (7 dims)
- No recording of home reset. One initial frame.
- Home pose initialised automatically by UR.py.
- Extra safe: frames buffered in memory; episode saved only if all IK steps succeed.
- Includes dummy 'task' field to satisfy LeRobot validation.
"""

import os
import sys
import random
import numpy as np
import shutil
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import URe_Robot
from config import (
    CAM2_DEF, CAM3_DEF,
    GRIPPER_TARGET_HOVER_HEIGHT, GRIPPER_LIFT_HEIGHT,
    CAM2_RECEIVER, CAM3_RECEIVER,
    GET_RECEIVER_BASE64_IMAGE, SIM_TABLE_DEF
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ========== Configuration ==========
LANGUAGE_INSTRUCTIONS = [
    "pick up the red cube and put it in the green bin",
    "grab the red block and drop it in the green crate",
    "move the red object into the green container",
    "place the red cube inside the green bin",
    "put the red cube in the crate",
    "put the block in the crate",
    "put the block in the bin",
    "place the block in the crate",
    "place the block in the bin",
    "move the block inside the crate",
]
NUM_EPISODES = 100
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Use a completely new dataset name to avoid any leftover metadata
DATASET_ROOT = os.path.join(PROJECT_ROOT, "recorded_dataset_v3")
REPO_ID = "local/ur5e_pickplace_v3"

# ========== Initialise Webots and robot ==========
supervisor = Supervisor()
robot = URe_Robot(supervisor)
timestep = int(supervisor.getBasicTimeStep())
control_dt = timestep / 1000.0
actual_fps = int(1000 / timestep)
print(f"[INFO] Webots timestep = {timestep} ms → recording at {actual_fps} FPS")

table_node = supervisor.getFromDef(SIM_TABLE_DEF)
table_pos = table_node.getPosition()
table_size = table_node.getField("size").getSFVec3f()
TABLE_HEIGHT = table_size[2]
TABLE_X_MIN = table_pos[0] - table_size[0]/2
TABLE_X_MAX = table_pos[0] + table_size[0]/2
TABLE_Y_MIN = table_pos[1] - table_size[1]/2
TABLE_Y_MAX = table_pos[1] + table_size[1]/2
SAFETY_MARGIN = 0.07

# Cameras: only two – agent view (CAM2) and wrist view (CAM3)
CHANNELS = 3

# Angled agent view (use CAM2)
cam2_node = supervisor.getFromDef(CAM2_DEF)
CAM2_WIDTH = cam2_node.getField("width").getSFInt32()
CAM2_HEIGHT = cam2_node.getField("height").getSFInt32()
receiver2 = supervisor.getDevice(CAM2_RECEIVER)
receiver2.enable(timestep)

# Wrist camera (CAM3)
cam3_node = supervisor.getFromDef(CAM3_DEF)
CAM3_WIDTH = cam3_node.getField("width").getSFInt32()
CAM3_HEIGHT = cam3_node.getField("height").getSFInt32()
receiver3 = supervisor.getDevice(CAM3_RECEIVER)
receiver3.enable(timestep)

# Cube and bin nodes
target_node = supervisor.getFromDef("OBJ1")
dest_node = supervisor.getFromDef("BIN")
DEFAULT_CUBE_POS = target_node.getPosition()
DEFAULT_CUBE_ROT = target_node.getField("rotation").getSFRotation()
DEFAULT_CUBE_POS = [0.2, 0.55, DEFAULT_CUBE_POS[2]]

bin_pos = dest_node.getPosition()
bin_size = dest_node.getField("size").getSFVec3f()
bin_half_world_x = bin_size[0] / 2.0
bin_half_world_y = bin_size[1] / 2.0
BIN_X_MIN = bin_pos[0] - bin_half_world_x
BIN_X_MAX = bin_pos[0] + bin_half_world_x
BIN_Y_MIN = bin_pos[1] - bin_half_world_y
BIN_Y_MAX = bin_pos[1] + bin_half_world_y
BIN_MARGIN = 0.085

# Helper functions for cube
def reset_cube_to_default():
    target_node.getField("translation").setSFVec3f(DEFAULT_CUBE_POS)
    target_node.getField("rotation").setSFRotation(DEFAULT_CUBE_ROT)
    for _ in range(10):
        supervisor.step(timestep)

def is_cube_safe(z_threshold=0.69):
    pos = target_node.getPosition()
    if pos[2] < z_threshold: return False
    if not (TABLE_X_MIN + SAFETY_MARGIN < pos[0] < TABLE_X_MAX - SAFETY_MARGIN): return False
    if not (TABLE_Y_MIN + SAFETY_MARGIN < pos[1] < TABLE_Y_MAX - SAFETY_MARGIN): return False
    return True

def randomize_cube_pose():
    x_min, x_max = TABLE_X_MIN + SAFETY_MARGIN, TABLE_X_MAX - SAFETY_MARGIN
    y_min, y_max = TABLE_Y_MIN + SAFETY_MARGIN, TABLE_Y_MAX - SAFETY_MARGIN
    for _ in range(100):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if (x < BIN_X_MIN - BIN_MARGIN or x > BIN_X_MAX + BIN_MARGIN or
            y < BIN_Y_MIN - BIN_MARGIN or y > BIN_Y_MAX + BIN_MARGIN):
            break
    else:
        x, y = 0.25, 0.60
    z = DEFAULT_CUBE_POS[2]
    target_node.getField("translation").setSFVec3f([x, y, z])
    yaw = random.uniform(-3.14159, 3.14159)
    target_node.getField("rotation").setSFRotation([0, 0, 1, yaw])
    for _ in range(10):
        supervisor.step(timestep)
    if not is_cube_safe():
        reset_cube_to_default()
    return (x, y, yaw)

# ---------- Helper: create a single frame (delta action) ----------
def create_frame(current_pose, gripper_cmd, success, done_flag, previous_pose, previous_gripper_cmd):
    agent_img = GET_RECEIVER_BASE64_IMAGE(receiver2, format="numpy")
    wrist_img = GET_RECEIVER_BASE64_IMAGE(receiver3, format="numpy")
    if agent_img is None or wrist_img is None:
        return None

    # State: [x,y,z,roll,pitch,yaw, grip, grip] (8 dims)
    state = np.array(current_pose + [gripper_cmd, gripper_cmd], dtype=np.float32)

    # Action: delta (7 dims)
    if previous_pose is None:
        action = np.zeros(7, dtype=np.float32)   # first frame
    else:
        delta_pos = np.array(current_pose[:3]) - np.array(previous_pose[:3])
        delta_angle = []
        for i in range(3, 6):
            diff = current_pose[i] - previous_pose[i]
            # wrap to [-π, π]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            delta_angle.append(diff)
        delta_grip = gripper_cmd - previous_gripper_cmd
        action = np.array(list(delta_pos) + delta_angle + [delta_grip], dtype=np.float32)

    frame = {
        "observation.images.image": agent_img,
        "observation.images.image2": wrist_img,
        "observation.state": state,
        "action": action,
        "task": random.choice(LANGUAGE_INSTRUCTIONS),
        "next.reward": np.array([1.0 if success else 0.0], dtype=np.float32),
        "next.done": np.array([done_flag], dtype=bool),
    }
    return frame

# ---------- Helper class to buffer an episode ----------
class EpisodeRecorder:
    def __init__(self):
        self.frames = []
        self.prev_pose = None
        self.prev_grip = None

    def capture(self, current_pose, gripper_cmd, success, done_flag):
        frame = create_frame(current_pose, gripper_cmd, success, done_flag,
                             self.prev_pose, self.prev_grip)
        if frame is not None:
            self.frames.append(frame)
            self.prev_pose = current_pose
            self.prev_grip = gripper_cmd
        return frame is not None

# ========== Create dataset ==========
if os.path.exists(DATASET_ROOT):
    print(f"⚠️ Deleting existing dataset at {DATASET_ROOT} for clean start")
    shutil.rmtree(DATASET_ROOT)

dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    root=DATASET_ROOT,
    fps=actual_fps,
    features={
        "observation.images.image": {
            "dtype": "image",
            "shape": (CAM2_HEIGHT, CAM2_WIDTH, CHANNELS),
            "names": ["height", "width", "channels"]
        },
        "observation.images.image2": {
            "dtype": "image",
            "shape": (CAM3_HEIGHT, CAM3_WIDTH, CHANNELS),
            "names": ["height", "width", "channels"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "grip_l", "grip_r"]
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]
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
    use_videos=True,
)
print(f"[Dataset] LIBERO‑compatible dataset created at {DATASET_ROOT} (FPS={actual_fps})")

# Main recording loop
successes, failures = 0, 0
episode_idx = 0
settle_steps = int(0.4 / control_dt)
grip_settle_steps = int(0.5 / control_dt)

# 10 seconds max per movement to prevent hanging if the robot gets stuck
MAX_PHASE_STEPS = int(10.0 / control_dt) 

# Run until we have exactly NUM_EPISODES (100) successes
while successes < NUM_EPISODES:
    print(f"\n{'='*50}")
    print(f"[Attempt {episode_idx}] Saved Successes: {successes}/{NUM_EPISODES}. Randomising cube...")
    randomize_cube_pose()

    # ---- RESET TO HOME (NO RECORDING) ----
    noisy_home = [angle + random.uniform(-0.05, 0.05) for angle in robot.home_joints]
    robot.set_joint_positions_async(noisy_home)
    print("      -> Moving to home (not recording)")
    wait_steps = int(3 / control_dt)
    for _ in range(wait_steps):
        supervisor.step(timestep)

    # Open gripper
    robot.operate_gripper(close=False)
    gripper_cmd = 0.0

    # ----- Buffer for frames -----
    recorder = EpisodeRecorder()

    # Capture initial frame after stabilising
    for _ in range(5):
        supervisor.step(timestep)
    current_pose = robot.get_end_effector_pose()
    if not recorder.capture(current_pose, gripper_cmd, success=False, done_flag=False):
        print(f"      -> [Warning] Failed to capture initial frame. Skipping attempt {episode_idx}.")
        episode_idx += 1
        continue

    # ---- PICK phase ----
    cube_pos = target_node.getPosition()
    approach_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + GRIPPER_LIFT_HEIGHT]
    robot.move_to_world_target_async(approach_pos)
    if robot._target_joints is None:
        print(f"      -> [Abort] IK failed for approach_pos. Skipping attempt {episode_idx}.")
        episode_idx += 1
        continue
        
    phase_steps = 0
    while not robot.is_target_reached() and supervisor.step(timestep) != -1:
        if phase_steps > MAX_PHASE_STEPS:
            print(f"      -> [Timeout] Arm got stuck approaching. Aborting attempt {episode_idx}.")
            robot._target_joints = None
            break
        current_pose = robot.get_end_effector_pose()
        if not recorder.capture(current_pose, gripper_cmd, success=False, done_flag=False):
            print(f"      -> [Warning] Frame capture failed. Skipping attempt {episode_idx}.")
            break
        phase_steps += 1
    else:
        for _ in range(settle_steps):
            supervisor.step(timestep)
            current_pose = robot.get_end_effector_pose()
            if not recorder.capture(current_pose, gripper_cmd, success=False, done_flag=False):
                print(f"      -> [Warning] Frame capture failed. Skipping attempt {episode_idx}.")
                break
    if robot._target_joints is None or len(recorder.frames) == 0:
        episode_idx += 1
        continue

    hover_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + GRIPPER_TARGET_HOVER_HEIGHT]
    robot.move_to_world_target_async(hover_pos)
    if robot._target_joints is None:
        print(f"      -> [Abort] IK failed for hover_pos. Skipping attempt {episode_idx}.")
        episode_idx += 1
        continue
        
    phase_steps = 0
    while not robot.is_target_reached() and supervisor.step(timestep) != -1:
        if phase_steps > MAX_PHASE_STEPS:
            print(f"      -> [Timeout] Arm got stuck hovering. Aborting attempt {episode_idx}.")
            robot._target_joints = None
            break
        current_pose = robot.get_end_effector_pose()
        if not recorder.capture(current_pose, gripper_cmd, success=False, done_flag=False):
            print(f"      -> [Warning] Frame capture failed. Skipping attempt {episode_idx}.")
            break
        phase_steps += 1
    else:
        for _ in range(settle_steps):
            supervisor.step(timestep)
            current_pose = robot.get_end_effector_pose()
            if not recorder.capture(current_pose, gripper_cmd, success=False, done_flag=False):
                print(f"      -> [Warning] Frame capture failed. Skipping attempt {episode_idx}.")
                break
    if robot._target_joints is None or len(recorder.frames) == 0:
        episode_idx += 1
        continue

    # Close gripper
    robot.operate_gripper(close=True)
    gripper_cmd = 1.0
    for _ in range(grip_settle_steps):
        supervisor.step(timestep)
        current_pose = robot.get_end_effector_pose()
        if not recorder.capture(current_pose, gripper_cmd, success=False, done_flag=False):
            print(f"      -> [Warning] Frame capture failed. Skipping attempt {episode_idx}.")
            break
    if len(recorder.frames) == 0:
        episode_idx += 1
        continue

    lift_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + GRIPPER_LIFT_HEIGHT]
    robot.move_to_world_target_async(lift_pos)
    if robot._target_joints is None:
        print(f"      -> [Abort] IK failed for lift_pos. Skipping attempt {episode_idx}.")
        episode_idx += 1
        continue
        
    phase_steps = 0
    while not robot.is_target_reached() and supervisor.step(timestep) != -1:
        if phase_steps > MAX_PHASE_STEPS:
            print(f"      -> [Timeout] Arm got stuck lifting. Aborting attempt {episode_idx}.")
            robot._target_joints = None
            break
        current_pose = robot.get_end_effector_pose()
        if not recorder.capture(current_pose, gripper_cmd, success=False, done_flag=False):
            print(f"      -> [Warning] Frame capture failed. Skipping attempt {episode_idx}.")
            break
        phase_steps += 1
    else:
        for _ in range(settle_steps):
            supervisor.step(timestep)
            current_pose = robot.get_end_effector_pose()
            if not recorder.capture(current_pose, gripper_cmd, success=False, done_flag=False):
                print(f"      -> [Warning] Frame capture failed. Skipping attempt {episode_idx}.")
                break
    if robot._target_joints is None or len(recorder.frames) == 0:
        episode_idx += 1
        continue

    # ---- DROP phase ----
    bin_pos = dest_node.getPosition()
    above_bin = [bin_pos[0], bin_pos[1], bin_pos[2] + GRIPPER_LIFT_HEIGHT]
    robot.move_to_world_target_async(above_bin)
    if robot._target_joints is None:
        print(f"      -> [Abort] IK failed for above_bin. Skipping attempt {episode_idx}.")
        episode_idx += 1
        continue
        
    phase_steps = 0
    while not robot.is_target_reached() and supervisor.step(timestep) != -1:
        if phase_steps > MAX_PHASE_STEPS:
            print(f"      -> [Timeout] Arm got stuck moving to bin. Aborting attempt {episode_idx}.")
            robot._target_joints = None
            break
        current_pose = robot.get_end_effector_pose()
        if not recorder.capture(current_pose, gripper_cmd, success=False, done_flag=False):
            print(f"      -> [Warning] Frame capture failed. Skipping attempt {episode_idx}.")
            break
        phase_steps += 1
    else:
        for _ in range(settle_steps):
            supervisor.step(timestep)
            current_pose = robot.get_end_effector_pose()
            if not recorder.capture(current_pose, gripper_cmd, success=False, done_flag=False):
                print(f"      -> [Warning] Frame capture failed. Skipping attempt {episode_idx}.")
                break
    if robot._target_joints is None or len(recorder.frames) == 0:
        episode_idx += 1
        continue

    # Open gripper
    robot.operate_gripper(close=False)
    gripper_cmd = 0.0
    for _ in range(grip_settle_steps):
        supervisor.step(timestep)
        current_pose = robot.get_end_effector_pose()
        if not recorder.capture(current_pose, gripper_cmd, success=False, done_flag=False):
            print(f"      -> [Warning] Frame capture failed. Skipping attempt {episode_idx}.")
            break
    if len(recorder.frames) == 0:
        episode_idx += 1
        continue

    # Check success
    cube_final_pos = target_node.getPosition()
    success = ((cube_final_pos[0] - bin_pos[0])**2 + (cube_final_pos[1] - bin_pos[1])**2) < 0.1**2
    if success:
        successes += 1
    else:
        failures += 1

    # Final frames with done=True
    for _ in range(20):
        supervisor.step(timestep)
        current_pose = robot.get_end_effector_pose()
        if not recorder.capture(current_pose, gripper_cmd, success=success, done_flag=True):
            print(f"      -> [Warning] Final frame capture failed. Skipping attempt {episode_idx}.")
            break
    if len(recorder.frames) == 0:
        episode_idx += 1
        continue

    # ---- Episode successful – commit all frames to dataset ----
    if success:
        for frame in recorder.frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"[Attempt {episode_idx}] SUCCESS & SAVED. Frames: {len(recorder.frames)} | Total Successes: {successes}/{NUM_EPISODES} | Total Failures: {failures}")
    else:
        # If the task failed, we just ignore recorder.frames and don't save the episode!
        print(f"[Attempt {episode_idx}] FAILED & DISCARDED. The robot missed the bin. | Total Successes: {successes}/{NUM_EPISODES} | Total Failures: {failures}")
        
    # ---> FORCE MEMORY CLEANUP TO PREVENT MAC CRASHES <---
    recorder.frames.clear()
    del recorder
    gc.collect()
    
    # Increment the loop counter for attempts
    episode_idx += 1

dataset.finalize()
print(f"\nRecording complete. {NUM_EPISODES} successful episodes safely recorded to {DATASET_ROOT}")