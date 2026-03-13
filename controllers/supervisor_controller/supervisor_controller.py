import sys
import os
# import numpy as np
from controller import Supervisor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path: sys.path.append(project_root)

# from ik_solver import IK_Solver

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

camera = supervisor.getDevice('camera')
camera.enable(timestep)

# solver = IK_Solver("path/to/urdf")

# temporary hardcoded target to test IK solver - will replace with camera-based target acquisition later
target_block = supervisor.getFromDef('BLUE_CUBE_1')

# --- MAIN LOOP ---
while supervisor.step(timestep) != -1:
    t = supervisor.getTime()

    # robot_pos = np.array(supervisor.getSelf().getPosition())
    # target_pos_world = np.array(target_block.getPosition())
    # target_pos_local = target_pos_world - robot_pos

    # ee_pos = solver.get_link_pos(target_pos_local.tolist())
    # print(ee_pos)
    break