"""
VLM Nudge Controller (SENSE-ACT-VERIFY)
This controller implements a reactive waypoint system.
It uses the VLM to identify objects, then verifies their position 
after moving to ensure a "nudge" didn't occur during transit.
"""

import os, sys, math
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from utils.VLM import ASK
from utils.TEST import AUTOMATE_NUDGE, LOG_NUDGE_METRICS

from config import (
    CAM1_DEF, 
    CAM1_RECEIVER, 
    SIM_TABLE_DEF, 
    GET_RECEIVER_BASE64_IMAGE,
    VLM_USER_PROMPTS,
    GRIPPER_TARGET_HOVER_HEIGHT, 
    GRIPPER_LIFT_HEIGHT,
    PIXEL_TO_WORLD,
    NUDGE_TOLERANCE_NORM,
    TARGET_NUDGES
)

# Initialize Webots
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = UR10e_Robot(supervisor)

receiver = supervisor.getDevice(CAM1_RECEIVER)
receiver.enable(timestep)

cam_node = supervisor.getFromDef(CAM1_DEF)
CAM_WIDTH = cam_node.getField("width").getSFInt32()
CAM_HEIGHT = cam_node.getField("height").getSFInt32()
CAM_FOV = cam_node.getField("fieldOfView").getSFFloat()

cam_rig = cam_node.getParentNode()
CAM_X, CAM_Y, CAM_Z = cam_rig.getPosition()
TABLE_HEIGHT = supervisor.getFromDef(SIM_TABLE_DEF).getField("size").getSFVec3f()[2]

nudges_performed = 0
nudge_history = []
current_nudge_truth = None
# ------------------------------------------

start_time = supervisor.getTime()
predicted_pick = None
predicted_drop = None
final_pick_error = 999.0

task_state = "SENSING"
prev_coords = None

retry_count = 0
MAX_RETRIES = 5

print("\n[Brain] Starting Automated VLM Nudge Test...")

while task_state != "DONE" and supervisor.step(timestep) != -1:
    
    if task_state == "SENSING":
        img = GET_RECEIVER_BASE64_IMAGE(receiver)
        if not img: continue
        
        content = [
            {"type": "text", "text": VLM_USER_PROMPTS["nudge_obj"]},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
        ]
        
        data = ASK(content)
        
        if data.get("found") is False or "x" not in data or "y" not in data:
            retry_count += 1
            print(f" -> Target not found or API error. Retrying... ({retry_count}/{MAX_RETRIES})")
            if retry_count >= MAX_RETRIES:
                print("\n[Fatal] Max retries reached! The target is occluded. Aborting.")
                break
                
            # Wait 2 seconds before hitting the API again
            for _ in range(int(2000 / timestep)): supervisor.step(timestep)
            continue
            
        retry_count = 0 
        
        curr_coords = (data['x'], data['y'])
        wx, wy = PIXEL_TO_WORLD(curr_coords[0], curr_coords[1], TABLE_HEIGHT, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y, CAM_Z)

        # Log tracking error if we are recovering from a nudge
        if current_nudge_truth is not None:
            error = math.dist((wx, wy), current_nudge_truth)
            print(f" -> [Referee] VLM Re-planned! Tracking Error: {error:.4f}m")
            nudge_history.append({'truth': current_nudge_truth, 'pred': (wx, wy), 'error': error})
            current_nudge_truth = None

        if prev_coords is None:
            print(" -> Initial target acquired. Moving to hover...")
            prev_coords = curr_coords
            robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
            
            # Programmatic Nudge Trigger
            if nudges_performed < TARGET_NUDGES:
                print(f" -> [Referee] Triggering Nudge {nudges_performed + 1}/{TARGET_NUDGES}...")
                current_nudge_truth = AUTOMATE_NUDGE(supervisor)
                nudges_performed += 1
                prev_coords = None
            continue 
            
        else:
            dist = math.dist(prev_coords, curr_coords)
            if dist > NUDGE_TOLERANCE_NORM:
                print(f" -> [!] NUDGE DETECTED (Moved {dist:.3f}). Re-tracking...")
                prev_coords = None 

                # Wait 2 seconds before snapping the next photo
                for _ in range(int(2000 / timestep)): supervisor.step(timestep)
                continue
            else:
                print(" -> Target verified stable. Executing Grab...")
                predicted_pick = (wx, wy)
                final_truth_pos = supervisor.getFromDef("OBJ1").getPosition()
                final_pick_error = math.dist((wx, wy), (final_truth_pos[0], final_truth_pos[1]))
                
                robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_TARGET_HOVER_HEIGHT])
                robot.operate_gripper(close=True)
                robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
                
                task_state = "DROPPING"
                prev_coords = None 

    elif task_state == "DROPPING":
        img = GET_RECEIVER_BASE64_IMAGE(receiver)
        if not img: continue
        
        content = [
            {"type": "text", "text": VLM_USER_PROMPTS["nudge_bin"]}, 
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
        ]
        
        data = ASK(content)
        
        if data.get("found") is False or "x" not in data or "y" not in data:
            retry_count += 1
            print(f" -> Bin not found or API error. Retrying... ({retry_count}/{MAX_RETRIES})")
            if retry_count >= MAX_RETRIES:
                break
            
            # Wait 2 seconds before hitting the API again
            for _ in range(int(2000 / timestep)): supervisor.step(timestep)
            continue
            
        retry_count = 0

        curr_coords = (data['x'], data['y'])
        wx, wy = PIXEL_TO_WORLD(curr_coords[0], curr_coords[1], TABLE_HEIGHT, CAM_FOV, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y, CAM_Z)
        
        print(f" -> Bin sensed at {wx:.3f}, {wy:.3f}. Executing Drop...")
        predicted_drop = (wx, wy)
        
        robot.move_to_world_target([wx, wy, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
        robot.operate_gripper(close=False)

        task_state = "DONE"

end_time = supervisor.getTime()

if predicted_pick is not None:
    LOG_NUDGE_METRICS(
        supervisor=supervisor,
        architecture="VLM",
        start_time=start_time,
        end_time=end_time,
        nudge_history=nudge_history,
        final_pick_error=final_pick_error,
        predicted_drop=predicted_drop
    )
    print("\n[System] Sequence Complete! Data Logged.")
else:
    print("\n[System] Sequence aborted early. No data logged.")

print("[System] Entering Idle Mode.")
while supervisor.step(timestep) != -1: pass