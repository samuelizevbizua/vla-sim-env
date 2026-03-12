import sys
import os
import numpy as np
from controller import Supervisor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path: sys.path.append(project_root)

from ik_solver import IK_Solver

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

camera = supervisor.getDevice('camera')
camera.enable(timestep)

solver = IK_Solver()

arm_motors = []
arm_sensors = []
for i in range(1, 8):
    motor = supervisor.getDevice(f'panda_joint{i}')
    arm_motors.append(motor)
    sensor = supervisor.getDevice(f'panda_joint{i}_sensor')
    sensor.enable(timestep)
    arm_sensors.append(sensor)

left_finger  = supervisor.getDevice('panda_finger::left')
right_finger = supervisor.getDevice('panda_finger::right')
left_finger.setPosition(0.04)
right_finger.setPosition(0.04)

target_block = supervisor.getFromDef('BLUE_CUBE_1')

HOVER_CLEARANCE = 0.15  # how high above the cube to hover before descending

# Stage timing (seconds)
STAGE_TIMES = {
    'hover':   3.0,
    'descend': 6.0,
    'grasp':   8.0,
    'lift':   11.0,
}

stage_done = set()

def move_arm(target_pos_local):
    try:
        joints = solver.get_link_pos(target_pos_local)
        for i, motor in enumerate(arm_motors):
            motor.setPosition(joints[i])
        print(f"Moving to local pos: {[round(v, 3) for v in target_pos_local]}")
    except RuntimeError as e:
        print(f"IK ERROR: {e}")

# --- MAIN LOOP ---
while supervisor.step(timestep) != -1:
    t = supervisor.getTime()

    robot_pos        = np.array(supervisor.getSelf().getPosition())
    target_pos_world = np.array(target_block.getPosition())
    target_pos_local = target_pos_world - robot_pos

    # Stage 1: hover above cube
    if t >= STAGE_TIMES['hover'] and 'hover' not in stage_done:
        hover_target = target_pos_local.copy()
        hover_target[2] = target_pos_local[2] + HOVER_CLEARANCE
        move_arm(hover_target.tolist())
        stage_done.add('hover')

    # Stage 2: descend - ik_solver handles finger offset internally
    elif t >= STAGE_TIMES['descend'] and 'descend' not in stage_done:
        move_arm(target_pos_local.tolist())
        stage_done.add('descend')

    # Stage 3: close gripper
    elif t >= STAGE_TIMES['grasp'] and 'grasp' not in stage_done:
        print("Closing gripper...")
        left_finger.setPosition(0.0)
        right_finger.setPosition(0.0)
        stage_done.add('grasp')

    # Stage 4: lift
    elif t >= STAGE_TIMES['lift'] and 'lift' not in stage_done:
        lift_target = target_pos_local.copy()
        lift_target[2] = target_pos_local[2] + HOVER_CLEARANCE
        move_arm(lift_target.tolist())
        stage_done.add('lift')
        print("Grasp sequence complete.")
        break