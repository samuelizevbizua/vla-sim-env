"""
This controller implements a reactive continuous-tracking IK system for the UR10e robot in Webots.
It's main function is to test inverse kinematics and the robot's functions via the UR10e_Robot class, without any AI or high-level decision making. 
It continuously tracks the target cube and destination bin, and reacts to any movement by re-planning the robot's trajectory in real-time.
"""
import os, sys, math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from config import GRIPPER_TARGET_HOVER_HEIGHT, GRIPPER_LIFT_HEIGHT

supervisor = Supervisor()
robot = UR10e_Robot(supervisor)

target = supervisor.getFromDef("OBJ1")
# target = supervisor.getFromDef("OBJ2")

dest = supervisor.getFromDef("BIN") 

timestep = int(supervisor.getBasicTimeStep())
task_state = "PICKING" 

NUDGE_TOLERANCE = 0.01 

print("\n[IK System] Initializing Reactive Continuous-Tracking Controller...")

while task_state != "DONE" and supervisor.step(timestep) != -1:
    if task_state == "PICKING":
        prev_pos = target.getPosition()
        
        robot.operate_gripper(close=False) 
        robot.move_to_world_target([prev_pos[0], prev_pos[1], prev_pos[2] + GRIPPER_LIFT_HEIGHT])

        for _ in range(int(500 / timestep)): supervisor.step(timestep)

        current_pos = target.getPosition()
        dist = math.dist(prev_pos, current_pos)
        
        if dist > NUDGE_TOLERANCE: continue 
            
        else:
            robot.move_to_world_target([current_pos[0], current_pos[1], current_pos[2] + GRIPPER_TARGET_HOVER_HEIGHT])
            
            for _ in range(int(300 / timestep)): supervisor.step(timestep)

            robot.operate_gripper(close=True)
            
            robot.move_to_world_target([current_pos[0], current_pos[1], current_pos[2] + GRIPPER_LIFT_HEIGHT])
            task_state = "DROPPING"
    
    elif task_state == "DROPPING":
        dest = dest.getPosition()
        robot.move_to_world_target([dest[0], dest[1], dest[2] + GRIPPER_LIFT_HEIGHT])
        robot.operate_gripper(close=False)
            
        task_state = "DONE"

while supervisor.step(timestep) != -1:
    pass