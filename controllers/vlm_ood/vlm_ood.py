"""
VLM OOD Controller (Out-Of-Distribution Test)
Tests zero-shot semantic reasoning on a novel object (rubber duck)
not seen during any training or configuration phase.
Uses a single-pass plan-then-act sequence identical to the baseline.
"""

import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import URe_Robot
from utils.VLM import ASK
from utils.TEST import LOG_OOD

from config import (
    CAM1_DEF, CAM1_RECEIVER, SIM_TABLE_DEF,
    GET_RECEIVER_BASE64_IMAGE, VLM_USER_PROMPTS,
    GRIPPER_TARGET_HOVER_HEIGHT, GRIPPER_LIFT_HEIGHT, PIXEL_TO_WORLD,
    TARGET2_DEF, MAX_TASK_TIME, DEST_BIN_HEIGHT
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

print("\n[Brain] Initializing VLM OOD Test")
for _ in range(int(1000 / timestep)):
    supervisor.step(timestep)

start_time = supervisor.getTime()
initial_target_pos = supervisor.getFromDef(TARGET2_DEF).getPosition()
predicted_pick = None
predicted_drop = None
task_complete = False
task_timed_out = False

img = GET_RECEIVER_BASE64_IMAGE(receiver)

if img is not None:
    content = [
        {"type": "text", "text": VLM_USER_PROMPTS["ood_test"]},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    ]
    try:
        print("\n[Vision] Querying VLM for OOD target and bin coordinates...")
        args = ASK(content)

        if args.get("found") is False or "target_x_norm" not in args:
            raise ValueError(f"VLM failed to identify OOD target. Response: {args}")

        pick_x, pick_y = PIXEL_TO_WORLD(args['target_x_norm'], args['target_y_norm'],
                                        TABLE_HEIGHT, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y, CAM_Z)
        drop_x, drop_y = PIXEL_TO_WORLD(args['bin_x_norm'], args['bin_y_norm'],
                                        TABLE_HEIGHT + DEST_BIN_HEIGHT, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y, CAM_Z)

        print(f"\n[Muscle] Executing OOD pick-and-place...")
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
        print("\n[System] OOD sequence complete.")

    except Exception as e:
        print(f"[Error] Controller failed: {e}")
else:
    print("[Error] Failed to capture image.")

end_time = supervisor.getTime()
if supervisor.getTime() - start_time > MAX_TASK_TIME:
    task_timed_out = True
    print("[Error] VLM OOD sequence timed out (Exceeded MAX_TASK_TIME).")

if task_complete and not task_timed_out and predicted_pick and predicted_drop:
    LOG_OOD(
        supervisor=supervisor, architecture="VLM",
        start_time=start_time, end_time=end_time,
        predicted_pick=predicted_pick, predicted_drop=predicted_drop,
        initial_target_pos=initial_target_pos
    )
else:
    end_time = supervisor.getTime()
    LOG_OOD(
        supervisor=supervisor, architecture="VLM",
        start_time=start_time, end_time=end_time,
        predicted_pick=(0, 0), predicted_drop=(0, 0),
        initial_target_pos=initial_target_pos,
        success_override=False
    )

print("\n[System] Entering idle mode.")
while supervisor.step(timestep) != -1:
    pass