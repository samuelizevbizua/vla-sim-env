import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
from controller import Supervisor
from UR import UR10e_Robot

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# Initialize our new clean Robot class!
robot = UR10e_Robot(supervisor)

def pick_and_place(pick_pos: list[float], drop_pos: list[float]):
    pick_pos = np.array(pick_pos)
    drop_pos = np.array(drop_pos)

    print("\n1. Hovering over target...")
    hover_pos = np.copy(pick_pos)
    hover_pos[2] += 0.15 
    robot.operate_gripper(close=False) 
    robot.move_to_world_target(hover_pos)

    print("\n2. Descending to grab...")
    grab_pos = np.copy(pick_pos)
    grab_pos[2] += 0.02
    robot.move_to_world_target(grab_pos)

    print("\n3. Closing gripper...")
    robot.operate_gripper(close=True)

    print("\n4. Lifting block...")
    robot.move_to_world_target(hover_pos)

    print("\n5. Moving to drop bin...")
    bin_hover_pos = np.copy(drop_pos)
    bin_hover_pos[2] += 0.20
    robot.move_to_world_target(bin_hover_pos)

    print("\n6. Releasing block...")
    robot.operate_gripper(close=False)
    
    print("\nSequence Complete!")

# --- EXECUTION ---
pick_node = supervisor.getFromDef('BLUE_CUBE_1')
drop_node = supervisor.getFromDef('BLUE_BIN')

if pick_node and drop_node:
    pick_and_place(pick_node.getPosition(), drop_node.getPosition())
else:
    print("Error: Could not find objects in scene tree.")

# Keep simulation running
while supervisor.step(timestep) != -1:
    pass