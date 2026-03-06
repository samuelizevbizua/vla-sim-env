import sys
import os
from controller import Supervisor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path: sys.path.append(project_root)

from ik_solver import IK_Solver

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# Camera Setup
camera = supervisor.getDevice('camera')
camera.enable(timestep)

# IK Setup
solver = IK_Solver(os.path.join(project_root, 'franka_panda.urdf'))

# Motor Setup
arm_motors = []
for i in range(1, 8):
    motor = supervisor.getDevice(f'panda_joint{i}')
    arm_motors.append(motor)

left_gripper = supervisor.getDevice('panda_finger::left')
right_gripper = supervisor.getDevice('panda_finger::right')

# World Objects - tmp
target_block = supervisor.getFromDef('RED_CUBE_1')

# --- MAIN LOOP ---
while supervisor.step(timestep) != -1:
    # 1. SENSE
    target_pos = target_block.getPosition()
    
    # 2. THINK (Logging for now)
    # We will eventually calculate the IK solution here
    # print(f"Target: {target_pos} | Current Joints: {current_joint_positions[0]:.2f}...")

    # 3. ACT
    # Example: Continuous movement test (Joint 1)
    # arm_motors[0].setPosition(1.57) 
    
    pass