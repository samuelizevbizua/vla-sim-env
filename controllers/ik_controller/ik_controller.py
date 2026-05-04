"""
This controller implements a standard inverse kinematics system for the UR10e robot in Webots.
It's main function is to test inverse kinematics and the robot's functions via the URe_Robot class, without any AI or high-level decision making. 
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import URe_Robot
from config import GRIPPER_TARGET_HOVER_HEIGHT, GRIPPER_LIFT_HEIGHT, TARGET_DEF, DEST_DEF

supervisor = Supervisor()
robot = URe_Robot(supervisor)

target = supervisor.getFromDef(TARGET_DEF)
dest = supervisor.getFromDef(DEST_DEF) 

timestep = int(supervisor.getBasicTimeStep())
task_state = "PICKING" 

print("\n[IK System] Initializing IK Pick-and-Place Controller...")

while task_state != "DONE" and supervisor.step(timestep) != -1:
    if task_state == "PICKING":
        pos = target.getPosition()
        
        robot.operate_gripper(close=False) 
        robot.move_to_world_target([pos[0], pos[1], pos[2] + GRIPPER_LIFT_HEIGHT])

        robot.move_to_world_target([pos[0], pos[1], pos[2] + GRIPPER_TARGET_HOVER_HEIGHT])
        robot.operate_gripper(close=True)
        
        robot.move_to_world_target([pos[0], pos[1], pos[2] + GRIPPER_LIFT_HEIGHT])
        
        task_state = "DROPPING"

    elif task_state == "DROPPING":
        dest_pos = dest.getPosition()
        
        robot.move_to_world_target([dest_pos[0], dest_pos[1], dest_pos[2] + GRIPPER_LIFT_HEIGHT])        
        robot.operate_gripper(close=False)

        task_state = "DONE"

print("\n[IK System] Sequence Complete. Entering Idle Mode.")
while supervisor.step(timestep) != -1: pass