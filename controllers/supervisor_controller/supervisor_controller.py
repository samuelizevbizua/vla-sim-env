import sys
import os
import numpy as np
from controller import Supervisor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path: sys.path.append(project_root)

from utils import IK_Solver

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# --- SETUP MOTORS & SENSORS ---
motor_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]
motors = []
sensors = []

for name in motor_names:
    motor = supervisor.getDevice(name)
    if motor:
        motors.append(motor)
        
        # Webots requires you to explicitly turn on the position sensors!
        sensor = motor.getPositionSensor()
        sensor.enable(timestep)
        sensors.append(sensor)
    else:
        print(f"Warning: Could not find motor '{name}'")

# Initialize solver
solver = IK_Solver(gripper_length=0.16)

# temporary hardcoded target to test IK solver - will replace with camera-based target acquisition later
target_block = supervisor.getFromDef('BLUE_BIN')

# Constants for frame alignment (UR10e vs Webots world)
OFFSET_ANGLE = np.pi 
ROTATION_MATRIX = np.array([
    [np.cos(OFFSET_ANGLE), -np.sin(OFFSET_ANGLE), 0],
    [np.sin(OFFSET_ANGLE),  np.cos(OFFSET_ANGLE), 0],
    [0,                     0,                    1]
])

# --- MAIN LOOP ---
step_count = 0
has_printed = False

while supervisor.step(timestep) != -1:
    step_count += 1
    
    # 1. Let the sensors "warm up" for 5 physics steps
    if step_count < 5:
        continue

    # 2. Get World Positions
    robot_node = supervisor.getSelf()
    robot_pos_world = np.array(robot_node.getPosition())
    robot_rot_world = np.array(robot_node.getOrientation()).reshape(3, 3) 
    
    target_pos_world = np.array(target_block.getPosition())

    # 3. Apply Pre-Grasp Hover (Lift target 10cm above the block)
    target_pos_world[2] += 0.2

    # 4. Convert World -> Local Robot Frame
    target_vector_world = target_pos_world - robot_pos_world
    target_pos_local = robot_rot_world.T.dot(target_vector_world)

    # 5. Apply Webots-to-UR Frame Alignment Fix
    ik_target_local = ROTATION_MATRIX.dot(target_pos_local)

    # 6. Read Sensors & Solve IK
    current_joints = [sensor.getValue() for sensor in sensors]
    joint_angles = solver.get_link_pos(ik_target_local.tolist(), current_joints)

    # 7. Actuate Motors or Handle Failure
    if joint_angles is not None and len(motors) == 6:
        for i in range(6):
            motors[i].setPosition(joint_angles[i])
            
        # Print success only once to keep the terminal clean
        if not has_printed:
            print("IK Solved! Smoothly tracking target in real-time...")
            has_printed = True
    else:
        # Gracefully handle unreachable targets
        distance = np.linalg.norm(target_vector_world)
        print(f"Failed. Block is {distance:.2f} meters away (UR10e max reach is 1.30m).")
        break