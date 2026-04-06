"""
This controller implements a reactive continuous-tracking IK system for the UR10e robot in Webots.
It's main function is to test inverse kinematics and the robot's functions via the UR10e_Robot class, without any AI or high-level decision making. 
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from config import GRIPPER_TARGET_HOVER_HEIGHT, GRIPPER_LIFT_HEIGHT

supervisor = Supervisor()
robot = UR10e_Robot(supervisor)

# target = supervisor.getFromDef("OBJ1")
target = supervisor.getFromDef("OBJ2")
dest = supervisor.getFromDef("BIN") 

timestep = int(supervisor.getBasicTimeStep())
task_state = "PICKING" 

print("\n[IK System] Initializing Reactive Continuous-Tracking Controller...")

while task_state != "DONE" and supervisor.step(timestep) != -1:
    if task_state == "PICKING":
        target_pos = target.getPosition()
        
        robot.operate_gripper(close=False)
        
        robot.move_to_world_target([target_pos[0], target_pos[1], target_pos[2] + GRIPPER_TARGET_HOVER_HEIGHT])
        robot.operate_gripper(close=True)
        
        robot.move_to_world_target([target_pos[0], target_pos[1], target_pos[2] + GRIPPER_LIFT_HEIGHT])
        task_state = "DROPPING"
    
    elif task_state == "DROPPING":
        dest = dest.getPosition()
        robot.move_to_world_target([dest[0], dest[1], dest[2] + GRIPPER_LIFT_HEIGHT])
        robot.operate_gripper(close=False)
            
        task_state = "DONE"

while supervisor.step(timestep) != -1:
    pass