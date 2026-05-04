"""
VLM Baseline Controller (PLAN-THEN-ACT)
Single-pass open-loop pick-and-place. Takes one image, asks the VLM
for coordinates, and executes the full sequence without re-sensing.
This is the non-reactive baseline against which the nudge test is compared.
"""

import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import URe_Robot
from utils.VLM import ASK
from utils.TEST import LOG_BASELINE

from config import (
    CAM1_DEF, CAM1_RECEIVER, SIM_TABLE_DEF,
    GET_RECEIVER_BASE64_IMAGE, VLM_USER_PROMPTS,
    GRIPPER_TARGET_HOVER_HEIGHT, GRIPPER_LIFT_HEIGHT, PIXEL_TO_WORLD,
    TARGET_DEF, MAX_TASK_TIME
)

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = URe_Robot(supervisor)

receiver = supervisor.getDevice(CAM1_RECEIVER)
receiver.enable(timestep)

cam_node = supervisor.getFromDef(CAM1_DEF)
CAM_WIDTH = cam_node.getField("width").getSFInt32()
CAM_HEIGHT = cam_node.getField("height").getSFInt32()
CAM_FOV = cam_node.getField("fieldOfView").getSFFloat()
CAM_X, CAM_Y, CAM_Z = cam_node.getParentNode().getPosition()
TABLE_HEIGHT = supervisor.getFromDef(SIM_TABLE_DEF).getField("size").getSFVec3f()[2]

print("\n[Brain] Initializing VLM Baseline (Single-Pass)")
for _ in range(int(1000 / timestep)):
    supervisor.step(timestep)

start_time = supervisor.getTime()
initial_cube_pos = supervisor.getFromDef(TARGET_DEF).getPosition()
predicted_pick = None
predicted_drop = None
task_complete = False

img = GET_RECEIVER_BASE64_IMAGE(receiver)

if img is not None:
    content = [
        {"type": "text", "text": VLM_USER_PROMPTS["baseline"]},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    ]
    try:
        print("\n[Vision] Querying VLM for object and bin coordinates...")
        args = ASK(content)

        if "cube_x_norm" not in args or "bin_x_norm" not in args:
            raise ValueError(f"Incomplete VLM response: {args}")

        pick_x, pick_y = PIXEL_TO_WORLD(args['cube_x_norm'], args['cube_y_norm'],
                                        TABLE_HEIGHT, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y, CAM_Z)
        drop_x, drop_y = PIXEL_TO_WORLD(args['bin_x_norm'], args['bin_y_norm'],
                                        TABLE_HEIGHT, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y, CAM_Z)

        print(f"\n[Muscle] Executing pick-and-place...")
        robot.operate_gripper(close=False)
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_TARGET_HOVER_HEIGHT])
        robot.operate_gripper(close=True)
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
        robot.move_to_world_target([drop_x, drop_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
        robot.operate_gripper(close=False)

        predicted_pick = (pick_x, pick_y)
        predicted_drop = (drop_x, drop_y)
        task_complete = True
        print("\n[System] Sequence complete.")

    except Exception as e:
        print(f"[Error] Controller failed: {e}")
else:
    print("[Error] Failed to capture image.")

end_time = supervisor.getTime()
timed_out = (end_time - start_time) > MAX_TASK_TIME

if timed_out:
    print("[Error] VLM sequence timed out (Exceeded MAX_TASK_TIME).")

if task_complete and not timed_out and predicted_pick and predicted_drop:
    LOG_BASELINE(
        supervisor=supervisor, architecture="VLM",
        start_time=start_time, end_time=end_time,
        predicted_pick=predicted_pick, predicted_drop=predicted_drop,
        initial_target_pos=initial_cube_pos
    )
else:
    LOG_BASELINE(
        supervisor=supervisor, architecture="VLM",
        start_time=start_time, end_time=end_time,
        predicted_pick=predicted_pick or (0, 0),
        predicted_drop=predicted_drop or (0, 0),
        initial_target_pos=initial_cube_pos,
        success_override=False
    )

print("\n[System] Entering idle mode.")
while supervisor.step(timestep) != -1:
    pass