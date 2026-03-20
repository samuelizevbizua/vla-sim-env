import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import base64
from controller import Supervisor
from UR import UR10e_Robot
from config import CAM_FRAME_PATH, HF_API_TOKEN, HF_MODEL_ID, WORLD_TABLE_HEIGHT, CAM_X_OFFSET, CAM_Y_OFFSET
from typing import List

# --- LANGCHAIN IMPORTS ---
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser

# --- 1. THE PYDANTIC BLUEPRINT ---
class TargetCoordinates(BaseModel):
    cube_pixels: List[float] = Field(description="The [X, Y] pixel coordinates of the small red cube.")
    bin_pixels: List[float] = Field(description="The [X, Y] pixel coordinates of the large, hollow blue bin.")

# --- 2. THE ROBOT MUSCLE ---
def execute_pick_and_place(robot, coords: TargetCoordinates):
    # Unpack the lists from Pydantic!
    cube_x, cube_y = coords.cube_pixels[0], coords.cube_pixels[1]
    bin_x, bin_y = coords.bin_pixels[0], coords.bin_pixels[1]
    
    print(f"\n[Muscle] Received Pixels - Cube: ({cube_x}, {cube_y}), Bin: ({bin_x}, {bin_y})")
    
    center_x, center_y = 1500 / 2.0, 1000 / 2.0 
    
    # Mathematically derived from Ground Truth Calibration
    SCALE_X = 0.00098 
    SCALE_Y = -0.00097 
    
    pick_x = round(((cube_x - center_x) * SCALE_X) + CAM_X_OFFSET, 3)
    pick_y = round(((cube_y - center_y) * SCALE_Y) + CAM_Y_OFFSET, 3)
    
    drop_x = round(((bin_x - center_x) * SCALE_X) + CAM_X_OFFSET, 3)
    drop_y = round(((bin_y - center_y) * SCALE_Y) + CAM_Y_OFFSET, 3)
    
    print(f"[Muscle] World Coords - Pick: ({pick_x}, {pick_y}), Drop: ({drop_x}, {drop_y})")

    print(" -> Hovering...")
    robot.operate_gripper(close=False) 
    robot.move_to_world_target([pick_x, pick_y, WORLD_TABLE_HEIGHT + 0.15])
    
    print(" -> Grabbing...")
    robot.move_to_world_target([pick_x, pick_y, WORLD_TABLE_HEIGHT + 0.02])
    robot.operate_gripper(close=True)
    
    print(" -> Lifting...")
    robot.move_to_world_target([pick_x, pick_y, WORLD_TABLE_HEIGHT + 0.15])
    
    print(" -> Moving to Bin...")
    robot.move_to_world_target([drop_x, drop_y, WORLD_TABLE_HEIGHT + 0.20])
    
    print(" -> Releasing...")
    robot.operate_gripper(close=False)
    print("\n[System] Task Complete!")

# --- 3. THE LANGCHAIN BRAIN (LCEL Chain) ---
def run_langchain_vision_chain(robot):
    print("\n[Brain] Initializing LangChain LCEL Pipeline...")
    
    llm = HuggingFaceEndpoint(
        repo_id=HF_MODEL_ID,
        huggingfacehub_api_token=HF_API_TOKEN,
        max_new_tokens=200,
    )
    chat_model = ChatHuggingFace(llm=llm)
    
    # Setup the Pydantic Parser
    parser = PydanticOutputParser(pydantic_object=TargetCoordinates)
    format_instructions = parser.get_format_instructions()

    # The Prompt Template (Using your descriptive approach!)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert robotic vision system. {format_instructions}"),
        ("user", [
            {"type": "text", "text": "Look at the image. The target is a small, solid red cube. The destination is a large, hollow blue bin. Find one red cube and the blue bin."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image_data}"}}
        ])
    ])

    # Construct the LangChain Expression Language (LCEL) Pipeline
    chain = prompt | chat_model | parser

    with open(CAM_FRAME_PATH, "rb") as image_file:
        base64_img = base64.b64encode(image_file.read()).decode('utf-8')

    print("[Brain] Sending image and schema to VLM...")
    try:
        # Invoke the chain! It handles the prompt injection, API call, and JSON parsing all at once.
        result: TargetCoordinates = chain.invoke({
            "image_data": base64_img,
            "format_instructions": format_instructions
        })
        
        print(f"[Brain] LangChain successfully parsed: {result}")
        
        # Pass the structured Pydantic object to the robot
        execute_pick_and_place(robot, result)
        
    except Exception as e:
        print(f"[Error] LangChain pipeline failed: {e}")

# --- WEBOTS EXECUTION ---
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = UR10e_Robot(supervisor)

print("\nWaking up Webots and waiting for a valid camera frame...")
image_ready = False

for _ in range(100):
    supervisor.step(timestep)
    if os.path.exists(CAM_FRAME_PATH) and os.path.getsize(CAM_FRAME_PATH) > 10000:
        supervisor.step(timestep) 
        image_ready = True
        break

if image_ready:
    run_langchain_vision_chain(robot)
else:
    print("[Error] Camera frame is missing.")

while supervisor.step(timestep) != -1:
    pass