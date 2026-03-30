import os, sys
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
from controller import Supervisor
from UR import UR10e_Robot
from config import OVERHEAD_CAM_DEF, OVERHEAD_RECEIVER, PERSPECTIVE_RECEIVER, SIM_TABLE_DEF, HF_MODEL, GET_RECEIVER_BASE64_IMAGE

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool

# Initialize Webots
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = UR10e_Robot(supervisor)

receiver = supervisor.getDevice(OVERHEAD_RECEIVER)
receiver.enable(timestep)

perspective_receiver = supervisor.getDevice(PERSPECTIVE_RECEIVER)
perspective_receiver.enable(timestep)

cam_node = supervisor.getFromDef(OVERHEAD_CAM_DEF)
CAM_WIDTH = cam_node.getField("width").getSFInt32()
CAM_HEIGHT = cam_node.getField("height").getSFInt32()

cam_rig = cam_node.getParentNode()
CAM_X, CAM_Y, CAM_Z = cam_rig.getPosition()

TABLE_HEIGHT = supervisor.getFromDef(SIM_TABLE_DEF).getField("size").getSFVec3f()[2]

_height = CAM_Z - TABLE_HEIGHT
_ground_w = 2.0 * _height * np.tan(0.67 / 2.0)
_ground_h = _ground_w * (CAM_HEIGHT / CAM_WIDTH)
MPP_X =  _ground_w / CAM_WIDTH
MPP_Y = -_ground_h / CAM_HEIGHT

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

    px_cube = cube_x_norm * CAM_WIDTH
    py_cube = cube_y_norm * CAM_HEIGHT
    px_bin  = bin_x_norm  * CAM_WIDTH
    py_bin  = bin_y_norm  * CAM_HEIGHT

    cx, cy = CAM_WIDTH / 2.0, CAM_HEIGHT / 2.0

    pick_x = round(CAM_X + (px_cube - cx) * MPP_X, 3)
    pick_y = round(CAM_Y + (py_cube - cy) * MPP_Y, 3)
    drop_x = round(CAM_X + (px_bin  - cx) * MPP_X, 3)
    drop_y = round(CAM_Y + (py_bin  - cy) * MPP_Y, 3)

    print(f"\n[Muscle] Cube world: ({pick_x}, {pick_y}) | Bin world: ({drop_x}, {drop_y})")

    if phase == 0:
        print(" -> [Phase 0] Blind Pick Sequence Initiated...")
        robot.operate_gripper(close=False) 
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + 0.15]) 
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + 0.02]) 
        robot.operate_gripper(close=True)                                       
        robot.move_to_world_target([pick_x, pick_y, TABLE_HEIGHT + 0.15]) 
        return "Pick sequence complete. Ready for phase 1."
        
    elif phase == 1:
        print(" -> [Phase 1] Place Sequence Initiated...")
        robot.move_to_world_target([drop_x, drop_y, TABLE_HEIGHT + 0.20])
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

llm_with_tools = llm.bind_tools([execute_robot_phase])

system_instruction = (
    "You are an expert robotic vision agent. "
    "You will receive TWO images: the first is a top-down overhead view used for precise coordinate mapping, "
    "the second is a perspective view for depth context. "
    "Return object coordinates from the FIRST (overhead) image as normalized floats between 0.0 and 1.0. "
    "Use your tools to move the robot arm."
)

print("Vision-action loop starting...\n")

while task_active and supervisor.step(timestep) != -1:
    base64_img = GET_RECEIVER_BASE64_IMAGE(receiver)
    perspective_img = GET_RECEIVER_BASE64_IMAGE(perspective_receiver)
    
    if base64_img is not None:
        prompt_text = (
            "If you see blue cubes, identify ONE blue cube "
            "and call execute_robot_phase with its normalized coordinates (0.0-1.0) "
            "and the GREEN bin's coordinates. "
            "If NO blue cubes are visible anywhere, call phase 99. Do not ask for clarification."
        )

        content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
        ]

        if perspective_img is not None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{perspective_img}"}})

        try:
            print("\n[Vision] Snapping picture and asking Gemini...")
            
            response = llm_with_tools.invoke([
                SystemMessage(content=system_instruction),
                HumanMessage(content=content)
            ])
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    args = tool_call["args"]
                    print(f" -> [Brain] decided to call tool with: {args}")
                    execute_robot_phase.invoke(args)
            else:
                print(f" -> [Gemini's Voice]: {response.content}")
                print(" -> [System] Short 2s penalty to prevent API spam...")
                for _ in range(int(2000 / timestep)):
                    if supervisor.step(timestep) == -1: break

        except Exception as e:
            error_msg = str(e)
            print(f"[Error] Agent failed: {error_msg}")
            
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                print("\n[CRITICAL ERROR] 429 Rate Limit Hit!")
                print("[System] Pausing Webots simulation to protect API quota...")
                task_active = False 
                supervisor.simulationSetMode(supervisor.SIMULATION_MODE_PAUSE)
                break 
                
            for _ in range(int(2000 / timestep)):
                if supervisor.step(timestep) == -1: break
    else:
        supervisor.step(timestep)

print("\n[System] Entering Terminal Idle State.")
while supervisor.step(timestep) != -1:
    pass