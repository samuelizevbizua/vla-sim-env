"""
VLM Main Controller (PLAN-THEN-ACT) for Baseline Test
This controller tests a single, open-loop pick-and-place sequence. 
It takes ONE picture, asks the VLM for coordinates, and blindly executes 
the entire sequence. This serves as the non-reactive baseline for comparison.
"""

import os, sys, json
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from utils.VLM import ASK

from config import (
    OVERHEAD_CAM_DEF, 
    OVERHEAD_RECEIVER, 
    SIM_TABLE_DEF, 
    GET_RECEIVER_BASE64_IMAGE, 
    HF_API_TOKEN, 
    GOOGLE_API_KEY,
    HF_BASE_URL,
    VLM_MODEL,
    VLM_MODEL2,
    MODEL_CONFIG,
    VLM_USER_PROMPTS,
    GRIPPER_TARGET_HOVER_HEIGHT, 
    GRIPPER_LIFT_HEIGHT,
    PIXEL_TO_WORLD
)

# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Webots
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = UR10e_Robot(supervisor)

receiver = supervisor.getDevice(OVERHEAD_RECEIVER)
receiver.enable(timestep)

cam_node = supervisor.getFromDef(OVERHEAD_CAM_DEF)
CAM_WIDTH = cam_node.getField("width").getSFInt32()
CAM_HEIGHT = cam_node.getField("height").getSFInt32()

cam_rig = cam_node.getParentNode()
CAM_X, CAM_Y, CAM_Z = cam_rig.getPosition()

TABLE_HEIGHT = supervisor.getFromDef(SIM_TABLE_DEF).getField("size").getSFVec3f()[2]
_height = CAM_Z - TABLE_HEIGHT

print("\n[Brain] Initializing Single-Pass VLM")

# llm = ChatOpenAI(
#     model=VLM_MODEL,
#     api_key=HF_API_TOKEN,
#     base_url=HF_BASE_URL,
#     temperature=MODEL_CONFIG["temperature"],      
#     max_tokens=MODEL_CONFIG["max_tokens"],
#     max_retries=MODEL_CONFIG["max_retries"],
# )

llm = ChatGoogleGenerativeAI(
    model=VLM_MODEL2,
    api_key=GOOGLE_API_KEY,
    temperature=MODEL_CONFIG["temperature"],
    max_retries=MODEL_CONFIG["max_retries"],
)

# Settle the simulation physics & camera buffer
for _ in range(int(1000 / timestep)): 
    supervisor.step(timestep)

img = GET_RECEIVER_BASE64_IMAGE(receiver)

if img is not None:
    content = [
        {"type": "text", "text": VLM_USER_PROMPTS["baseline_all"]},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    ]

    try:
        print("\n[Vision] Snapping baseline picture and asking VLM...")

        args = ASK(llm, content)
        print(f" -> [Brain] Output: {args}")

        pick_x, pick_y = PIXEL_TO_WORLD(args['cube_x_norm'], args['cube_y_norm'], _height, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y)
        drop_x, drop_y = PIXEL_TO_WORLD(args['bin_x_norm'], args['bin_y_norm'], _height, CAM_HEIGHT, CAM_WIDTH, CAM_X, CAM_Y)

        print(f"\n[Muscle] Executing Blind Pick-and-Place sequence...")
        
        robot.operate_gripper(close=False) 
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT]) 
        
        print(" -> Plunging...")
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_TARGET_HOVER_HEIGHT]) 
        
        # # Physics pause to bleed off momentum
        # for _ in range(int(300 / timestep)): supervisor.step(timestep) 
        
        print(" -> Grabbing...")
        robot.operate_gripper(close=True)                                       
        
        print(" -> Lifting...")
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT]) 
        
        print(" -> Moving to Bin...")
        robot.move_to_world_target([drop_x, drop_y, TABLE_HEIGHT + GRIPPER_LIFT_HEIGHT])
        
        print(" -> Releasing...")
        robot.operate_gripper(close=False)

        print("\n[System] Sequence Complete!")

    except Exception as e:
        print(f"[Error] Agent failed: {str(e)}")
else:
    print("[Error] Failed to capture image.")

# Step 3: Terminal Idle
print("\n[System] Entering Terminal Idle State.")
while supervisor.step(timestep) != -1:
    pass