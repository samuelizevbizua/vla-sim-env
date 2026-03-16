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

# --- 1. SETUP ARM MOTORS & SENSORS ---
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
        sensor = motor.getPositionSensor()
        sensor.enable(timestep)
        sensors.append(sensor)
    else:
        print(f"Warning: Could not find motor '{name}'")

# --- 2. SETUP GRIPPER MOTORS ---
# Standard joint names for the Robotiq 3F in Webots
gripper_names = ['finger_1_joint_1', 'finger_2_joint_1', 'finger_middle_joint_1']
scissor_names = ['palm_finger_1_joint', 'palm_finger_2_joint'] # Add these!

gripper_motors = []
for name in gripper_names:
    g_motor = supervisor.getDevice(name)
    if g_motor:
        gripper_motors.append(g_motor)
    else:
        print(f"Warning: Could not find gripper motor '{name}'")

scissor_motors = []
for name in scissor_names:
    s_motor = supervisor.getDevice(name)
    if s_motor:
        scissor_motors.append(s_motor)
    else:
        print(f"Warning: Could not find scissor motor '{name}'")

solver = IK_Solver(gripper_length=0.16)

# Frame Alignment Constants
OFFSET_ANGLE = np.pi 
ROTATION_MATRIX = np.array([
    [np.cos(OFFSET_ANGLE), -np.sin(OFFSET_ANGLE), 0],
    [np.sin(OFFSET_ANGLE),  np.cos(OFFSET_ANGLE), 0],
    [0,                     0,                    1]
])

# Let sensors warm up before doing any math
for _ in range(5):
    supervisor.step(timestep)

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def wait_to_arrive(target_joints, tolerance=0.02, timeout_seconds=3.0):
    """
    Steps the Webots simulation forward. Breaks if the arm reaches the target,
    or forces the sequence to continue if the timeout is reached.
    """
    max_steps = int((timeout_seconds * 1000) / timestep)
    steps = 0
    
    while supervisor.step(timestep) != -1:
        current_joints = [s.getValue() for s in sensors]
        
        # Calculate the maximum difference across all 6 joints
        error = np.max(np.abs(np.array(target_joints) - np.array(current_joints)))
        
        # Print a live update every ~0.5 seconds so we know it isn't frozen
        if steps % 15 == 0:
            print(f"      -> [Tracking] Max Joint Error: {error:.4f} rad")
        
        if error < tolerance:
            print("      -> [Success] Arm reached target perfectly!")
            break
            
        steps += 1
        if steps > max_steps:
            print(f"      -> [Timeout] Arm settled with {error:.4f} rad error. Forcing next step.")
            break

def move_to_world_target(target_pos_world):
    """
    Handles the math translation and commands the arm to move.
    """
    robot_node = supervisor.getSelf()
    robot_pos_world = np.array(robot_node.getPosition())
    robot_rot_world = np.array(robot_node.getOrientation()).reshape(3, 3) 
    
    # World -> Local -> Rotated alignment
    target_vector_world = target_pos_world - robot_pos_world
    target_pos_local = robot_rot_world.T.dot(target_vector_world)
    ik_target_local = ROTATION_MATRIX.dot(target_pos_local)

    current_joints = [s.getValue() for s in sensors]
    joint_angles = solver.get_link_pos(ik_target_local.tolist(), current_joints)

    if joint_angles is not None:
        for i in range(6):
            motors[i].setPosition(joint_angles[i])
        
        # ACTUALLY WAIT FOR THE ARM TO GET THERE (Now with a safe timeout!)
        wait_to_arrive(joint_angles)
    else:
        print("      -> [Error] IK Failed: Target out of physical reach.")

def operate_gripper(close=True):
    """
    Opens or closes the Robotiq 3F fingers and palm scissor joints.
    """
    # 0.95 rad clamps the fingers, 0.05 opens them wide
    target_angle = 0.95 if close else 0.05
    
    # 0.17 rad swings palms inward (tripod), 0.0 keeps them straight
    scissor_angle = 0.17 if close else 0.0

    # 1. Apply the scissor rotation
    for s_motor in scissor_motors:
        s_motor.setPosition(scissor_angle)
    
    # 2. Apply the finger clamping
    for g_motor in gripper_motors:
        g_motor.setPosition(target_angle)
        
    # Step the simulation forward for about 1 second to let the fingers 
    # physically travel and clamp onto the block via friction.
    for _ in range(int(500 / timestep)):
        supervisor.step(timestep)

# =========================================================================
# THE STATE MACHINE (MAIN SCRIPT)
# =========================================================================

def pick_and_place(pick_pos: list[float], drop_pos: list[float]):

    pick_pos = np.array(pick_pos)
    drop_pos = np.array(drop_pos)

    print("1. Hovering over target...")
    hover_pos = np.copy(pick_pos)
    hover_pos[2] += 0.15 
    operate_gripper(close=False) # Ensure gripper is open
    move_to_world_target(hover_pos)

    print("2. Descending to grab...")
    grab_pos = np.copy(pick_pos)
    grab_pos[2] += 0.00
    # grab_pos[2] += 0.01 # Slightly above absolute center to avoid table collision
    move_to_world_target(grab_pos)

    print("3. Closing gripper...")
    operate_gripper(close=True)

    print("4. Lifting block...")
    move_to_world_target(hover_pos)

    print("5. Moving to drop bin...")
    bin_hover_pos = np.copy(drop_pos)
    bin_hover_pos[2] += 0.20 # Stay high above the bin
    move_to_world_target(bin_hover_pos)

    print("6. Releasing block...")
    operate_gripper(close=False)
    
    print("Sequence Complete!")

# Execute the sequence
pick_pos = supervisor.getFromDef('RED_CUBE_3').getPosition()
drop_pos = supervisor.getFromDef('BLUE_BIN').getPosition()
pick_and_place(pick_pos, drop_pos)

# Keep simulation running after the sequence is done
while supervisor.step(timestep) != -1:
    pass