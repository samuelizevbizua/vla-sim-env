"""
VLM Nudge Controller (SENSE-ACT-VERIFY)
Implements a reactive replanning loop. After each approach, the VLM
re-queries the scene to verify the target position before committing
to a grasp. A programmatic nudge is triggered mid-approach to test
whether the modular pipeline can detect and replan around displacement.
"""

import os, sys, math
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import URe_Robot
from utils.VLM import ASK
from utils.TEST import AUTOMATE_NUDGE, LOG_NUDGE_METRICS

from config import (
    CAM1_DEF, CAM1_RECEIVER, SIM_TABLE_DEF,
    GET_RECEIVER_BASE64_IMAGE, VLM_USER_PROMPTS,
    GRIPPER_TARGET_HOVER_HEIGHT, GRIPPER_LIFT_HEIGHT,
    PIXEL_TO_WORLD, NUDGE_TOLERANCE_NORM, TARGET_NUDGES,
    TARGET_DEF, MAX_TASK_TIME, DEST_BIN_HEIGHT
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

nudges_performed = 0
nudge_history = []
current_nudge_truth = None

start_time = supervisor.getTime()
predicted_pick = None
predicted_drop = None
final_pick_error = 999.0
task_state = "SENSING"
prev_coords = None
retry_count = 0
MAX_RETRIES = 5

print("\n[Brain] Starting VLM Nudge Test...")

while task_state != "DONE" and supervisor.step(timestep) != -1:
    if supervisor.getTime() - start_time > MAX_TASK_TIME:
        print("\n[Error] VLM nudge sequence timed out (Exceeded MAX_TASK_TIME).")
        break

    if task_state == "SENSING":
        img = GET_RECEIVER_BASE64_IMAGE(receiver)
        if not img:
            continue

        content = [
            {"type": "text", "text": VLM_USER_PROMPTS["nudge_obj"]},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
        ]
        data = ASK(content)

        if data.get("found") is False or "x" not in data or "y" not in data:
            retry_count += 1
            print(f" -> Target not found. Retrying... ({retry_count}/{MAX_RETRIES})")
            if retry_count >= MAX_RETRIES:
                print("\n[Fatal] Max retries reached. Aborting.")
                break
            for _ in range(int(2000 / timestep)):
                if supervisor.getTime() - start_time > MAX_TASK_TIME:
                    print("\n[Error] VLM nudge sequence timed out during retry wait.")
                    task_state = "DONE"
                    break
                supervisor.step(timestep)
            if task_state == "DONE":
                break
            continue

        retry_count = 0
        curr_coords = (data['x'], data['y'])
        wx, wy = PIXEL_TO_WORLD(curr_coords[0], curr_coords[1],
                                TABLE_HEIGHT, CAM_FOV, CAM_HEIGHT, CAM_WIDTH,
                                CAM_X, CAM_Y, CAM_Z)

        # Log post-nudge tracking error
        if current_nudge_truth is not None:
            error = math.dist((wx, wy), current_nudge_truth)
            print(f" -> [Referee] Re-planned. Tracking error: {error:.4f}m")
            nudge_history.append({'truth': current_nudge_truth, 'pred': (wx, wy), 'error': error})
            current_nudge_truth = None

        if prev_coords is None:
            print(" -> Target acquired. Moving to hover...")
            prev_coords = curr_coords
            robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])

            # Trigger nudge after first hover
            if nudges_performed < TARGET_NUDGES:
                print(f" -> [Referee] Triggering nudge {nudges_performed + 1}/{TARGET_NUDGES}...")
                current_nudge_truth = AUTOMATE_NUDGE(supervisor)
                nudges_performed += 1
                prev_coords = None
            continue

        else:
            dist = math.dist(prev_coords, curr_coords)
            if dist > NUDGE_TOLERANCE_NORM:
                print(f" -> [!] Displacement detected ({dist:.3f}). Re-tracking...")
                prev_coords = None
                for _ in range(int(2000 / timestep)):
                    if supervisor.getTime() - start_time > MAX_TASK_TIME:
                        print("\n[Error] VLM nudge sequence timed out during re-track wait.")
                        task_state = "DONE"
                        break
                    supervisor.step(timestep)
                if task_state == "DONE":
                    break
                continue
            else:
                print(" -> Target stable. Executing grasp...")
                predicted_pick = (wx, wy)
                final_truth_pos = supervisor.getFromDef(TARGET_DEF).getPosition()
                final_pick_error = math.dist((wx, wy), (final_truth_pos[0], final_truth_pos[1]))

                robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_TARGET_HOVER_HEIGHT])
                robot.operate_gripper(close=True)
                robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
                task_state = "DROPPING"
                prev_coords = None

    elif task_state == "DROPPING":
        img = GET_RECEIVER_BASE64_IMAGE(receiver)
        if not img:
            continue

        content = [
            {"type": "text", "text": VLM_USER_PROMPTS["nudge_bin"]},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
        ]
        data = ASK(content)

        if data.get("found") is False or "x" not in data or "y" not in data:
            retry_count += 1
            print(f" -> Bin not found. Retrying... ({retry_count}/{MAX_RETRIES})")
            if retry_count >= MAX_RETRIES:
                break
            for _ in range(int(2000 / timestep)):
                if supervisor.getTime() - start_time > MAX_TASK_TIME:
                    print("\n[Error] VLM nudge sequence timed out during bin retry wait.")
                    task_state = "DONE"
                    break
                supervisor.step(timestep)
            if task_state == "DONE":
                break
            continue

        retry_count = 0
        curr_coords = (data['x'], data['y'])
        wx, wy = PIXEL_TO_WORLD(curr_coords[0], curr_coords[1],
                                TABLE_HEIGHT + DEST_BIN_HEIGHT, CAM_FOV, CAM_HEIGHT, CAM_WIDTH,
                                CAM_X, CAM_Y, CAM_Z)

        print(f" -> Bin located at ({wx:.3f}, {wy:.3f}). Dropping...")
        predicted_drop = (wx, wy)
        robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
        robot.operate_gripper(close=False)
        task_state = "DONE"

end_time = supervisor.getTime()
timed_out = (end_time - start_time) > MAX_TASK_TIME

if predicted_pick is not None and not timed_out:
    LOG_NUDGE_METRICS(
        supervisor=supervisor, architecture="VLM",
        start_time=start_time, end_time=end_time,
        nudge_history=nudge_history,
        final_pick_error=final_pick_error,
        predicted_drop=predicted_drop
    )
    print("\n[System] Sequence complete. Data logged.")
else:
    LOG_NUDGE_METRICS(
        supervisor=supervisor, architecture="VLM",
        start_time=start_time, end_time=end_time,
        nudge_history=nudge_history,
        final_pick_error=final_pick_error,
        predicted_drop=predicted_drop,
        success_override=False if timed_out else None
    )
    print("\n[System] Sequence aborted or timed out. Data logged.")

print("[System] Entering idle mode.")
while supervisor.step(timestep) != -1:
    pass