import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from UR import UR10e_Robot

supervisor = Supervisor()
robot = UR10e_Robot(supervisor)

# "Cheat" by asking the physics engine exactly where the objects are
target = supervisor.getFromDef("RED_CUBE_1")
dest = supervisor.getFromDef("BLUE_BIN")

target_pos = target.getPosition()
dest_pos = dest.getPosition()

print(f"[Oracle] Ground Truth Positions - Target: ({target_pos}), Destination: ({dest_pos})")

target_x, target_y, target_z = target_pos[0], target_pos[1], target_pos[2]
dest_x, dest_y, dest_z = dest_pos[0], dest_pos[1], dest_pos[2]

print(f"[Oracle] Exact World Coords - Target: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f}), Destination: ({dest_x:.3f}, {dest_y:.3f}, {dest_z:.3f})")

# Pass the negative coordinates to cancel out the UR.py 180-degree flip
print(" -> Hovering...")
robot.operate_gripper(close=False) 
robot.move_to_world_target([target_x, target_y, target_z + 0.15])

print(" -> Grabbing...")
robot.move_to_world_target([target_x, target_y, target_z + 0.02])
robot.operate_gripper(close=True)

print(" -> Lifting...")
robot.move_to_world_target([target_x, target_y, target_z + 0.15])

print(" -> Moving to Bin...")
robot.move_to_world_target([dest_x, dest_y, dest_z + 0.20])

print(" -> Releasing...")
robot.operate_gripper(close=False)
print("\n[Oracle] Task Complete!")

# Keep simulation running after task
timestep = int(supervisor.getBasicTimeStep())
while supervisor.step(timestep) != -1:
    pass