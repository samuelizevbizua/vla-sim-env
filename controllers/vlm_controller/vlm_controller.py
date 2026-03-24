import os, sys, base64
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np
from controller import Supervisor
from UR import UR10e_Robot
from config import WORLD_TABLE_HEIGHT, CAM_X_OFFSET, CAM_Y_OFFSET, CAM_HEIGHT, CAM_WIDTH, HF_MODEL

# --- THE ACTUAL MODERN LANGCHAIN IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool

# Initialize Webots
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = UR10e_Robot(supervisor)

receiver = supervisor.getDevice('receiver')
receiver.enable(timestep)

def get_latest_image():
    last_packet = None
    while receiver.getQueueLength() > 0:
        last_packet = receiver.getBytes()
        receiver.nextPacket()
    
    if last_packet:
        nparr = np.frombuffer(last_packet, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    return None

task_active = True
current_phase = 0

@tool
def execute_robot_phase(cube_x_norm: float, cube_y_norm: float, bin_x_norm: float, bin_y_norm: float, phase: int) -> str:
    """
    Executes a phase of the robot task.
    cube_x_norm, cube_y_norm: Object coordinates (FLOAT BETWEEN 0.0 AND 1.0)
    bin_x_norm, bin_y_norm: Destination coordinates (FLOAT BETWEEN 0.0 AND 1.0)
    phase 0: The "Pick" Sequence (Hover, Grab, Lift)
    phase 1: The "Place" Sequence (Move to Bin, Drop)
    phase 99: Task Complete - Stop the robot and end the loop
    """
    global task_active, current_phase
    
    if phase == 99:
        print("\n[Brain] Gemini confirms the table is clear of target cubes!")
        print("[System] Tidy Task Successfully Completed.")
        task_active = False
        return "Task complete. Shutting down."

    # 1. Map normalized 0.0-1.0 to ACTUAL camera resolution
    cube_x, cube_y = cube_x_norm * CAM_WIDTH, cube_y_norm * CAM_HEIGHT
    bin_x, bin_y = bin_x_norm * CAM_WIDTH, bin_y_norm * CAM_HEIGHT

    print(f"\n[Muscle] True Pixels Calculated -> Cube: ({cube_x}, {cube_y}), Bin: ({bin_x}, {bin_y})")

    center_x, center_y = CAM_WIDTH / 2.0, CAM_HEIGHT / 2.0 

    # 2. Updated Scale math for 640x450 resolution
    SCALE_X, SCALE_Y = 0.00229, -0.00215 
    
    pick_x = round(((cube_x - center_x) * SCALE_X) + CAM_X_OFFSET, 3)
    pick_y = round(((cube_y - center_y) * SCALE_Y) + CAM_Y_OFFSET, 3)
    drop_x = round(((bin_x - center_x) * SCALE_X) + CAM_X_OFFSET, 3)
    drop_y = round(((bin_y - center_y) * SCALE_Y) + CAM_Y_OFFSET, 3)

    if phase == 0:
        print(" -> [Phase 0] Blind Pick Sequence Initiated...")
        robot.operate_gripper(close=False) 
        robot.move_to_world_target([pick_x, pick_y, WORLD_TABLE_HEIGHT + 0.15]) 
        robot.move_to_world_target([pick_x, pick_y, WORLD_TABLE_HEIGHT + 0.02]) 
        robot.operate_gripper(close=True)                                       
        robot.move_to_world_target([pick_x, pick_y, WORLD_TABLE_HEIGHT + 0.15]) 
        return "Pick sequence complete. Ready for phase 1."
        
    elif phase == 1:
        print(" -> [Phase 1] Place Sequence Initiated...")
        robot.move_to_world_target([drop_x, drop_y, WORLD_TABLE_HEIGHT + 0.20])
        robot.operate_gripper(close=False)
        return "Drop complete. Ready for the next cube."
        
    return "Unknown phase."

# --- THE CONTINUOUS AGENT LOOP ---
print("\n[Brain] Initializing Naked LLM Agent...")

llm = ChatGoogleGenerativeAI(
    model=HF_MODEL, 
    temperature=0.0,      
    max_output_tokens=512,
    max_retries=3,
)

# 1. BIND THE TOOL DIRECTLY TO THE LLM
llm_with_tools = llm.bind_tools([execute_robot_phase])

system_instruction = (
    "You are an expert robotic vision agent. "
    "When finding objects, you MUST return their center coordinates as a normalized float between 0.0 and 1.0 (e.g., the exact center is [0.5, 0.5]). "
    "Use your tools to move the robot arm."
)

print("Vision-action loop starting...\n")

while task_active and supervisor.step(timestep) != -1:
    # Get the real-time frame from the Receiver
    frame = get_latest_image()
    
    if frame is not None:
        # Preview for debugging
        cv2.imshow("AI Feed", frame)
        cv2.waitKey(1)

        # 3. Direct memory encoding to Base64 (No disk I/O)
        _, buffer = cv2.imencode('.jpg', frame)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        # 4. Construct multimodal prompt
        prompt_text = (
            "If you see red cubes, identify ONE red cube "
            "and call execute_robot_phase with its normalized coordinates (0.0-1.0) "
            "and the GREEN bin's coordinates. "
            "If NO red cubes are visible anywhere, call phase 99. Do not ask for clarification."
        )

        try:
            print("\n[Vision] Snapping picture and asking Gemini...")
            
            response = llm_with_tools.invoke([
                SystemMessage(content=system_instruction),
                HumanMessage(content=[
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ])
            ])
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    args = tool_call["args"]
                    print(f" -> [Brain] decided to call tool with: {args}")
                    
                    # This now takes 8-10 physical seconds, naturally acting as our API cooldown!
                    execute_robot_phase.invoke(args)
            else:
                print(f" -> [Gemini's Voice]: {response.content}")
                print(" -> [System] Short 2s penalty to prevent API spam...")
                for _ in range(int(2000 / timestep)):
                    if supervisor.step(timestep) == -1: break

        except Exception as e:
            error_msg = str(e)
            print(f"[Error] Agent failed: {error_msg}")
            
            # --- THE 429 KILL SWITCH ---
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                print("\n[CRITICAL ERROR] 429 Rate Limit Hit!")
                print("[System] Pausing Webots simulation to protect API quota...")
                
                # 1. Flag the loop to stop
                task_active = False 
                
                # 2. Physically pause the Webots physics engine!
                supervisor.simulationSetMode(supervisor.SIMULATION_MODE_PAUSE)
                
                # 3. Break the while loop immediately
                break 
                
            # If it's a normal error (like a transient network glitch), just do the 2s penalty
            for _ in range(int(2000 / timestep)):
                if supervisor.step(timestep) == -1: break
    else:
        # Spin the simulation forward so the receiver has time to grab the first frame
        supervisor.step(timestep)

# Terminal Idle State
print("\n[System] Entering Terminal Idle State.")
while supervisor.step(timestep) != -1:
    pass