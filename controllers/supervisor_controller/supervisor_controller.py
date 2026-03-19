import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import base64
import json
from controller import Supervisor
from huggingface_hub import InferenceClient

from config import HF_API_TOKEN, HF_MODEL_ID, CAM_FRAME_PATH, PROMPTS, WORLD_TABLE_HEIGHT
from UR import UR10e_Robot

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# Initialize our new clean Robot class!
robot = UR10e_Robot(supervisor)

# --- 1. VLM VISION PIPELINE ---
def get_vlm_target():
    print("\n[Brain] Waking up VLM and analyzing camera frame...")
    client = InferenceClient(model=HF_MODEL_ID, token=HF_API_TOKEN)
    
    with open(CAM_FRAME_PATH, "rb") as image_file:
        base64_img = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
        {"type": "text", "text": PROMPTS["get_vlm_target"]}
    ]}]

    try:
        response = client.chat.completions.create(messages=messages, max_tokens=50)
        vlm_output = response.choices[0].message.content
        print(f"[Brain] VLM Output: {vlm_output}")
        return vlm_output
    except Exception as e:
        print(f"[Brain] API Error: {e}")
        return None

# --- 2. MATH & PARSING ---
def parse_and_convert(vlm_text):
    try:
        # Strip out any markdown formatting the VLM might accidentally add (like ```json)
        clean_text = vlm_text.replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON dictionary
        data = json.loads(clean_text)
        
        # Extract the pixels
        cube_pix = data.get("red_cube")
        bin_pix = data.get("blue_bin")
        
        print(f"[Translator] Parsed Image Pixels - Cube: {cube_pix}, Bin: {bin_pix}")
        
        # Pixels to Meters conversion math (1500x1000 scale)
        center_x, center_y = 1500 / 2.0, 1000 / 2.0
        
        # Convert Cube
        cube_world_x = round((cube_pix[0] - center_x) * 0.001, 3)
        cube_world_y = round((cube_pix[1] - center_y) * 0.001, 3)
        
        # Convert Bin
        bin_world_x = round((bin_pix[0] - center_x) * 0.001, 3)
        bin_world_y = round((bin_pix[1] - center_y) * 0.001, 3)
        
        print(f"[Translator] World Coords - Target: ({cube_world_x}, {cube_world_y}) -> Drop: ({bin_world_x}, {bin_world_y})")
        
        return cube_world_x, cube_world_y, bin_world_x, bin_world_y
        
    except Exception as e:
        print(f"[Translator] Failed to parse VLM JSON output. Error: {e}\nRaw text: {vlm_text}")
        return None, None, None, None

# --- 3. ROBOT EXECUTION ---
def execute_pick_and_place(robot, pick_x, pick_y, drop_x, drop_y):
    print("\n[Muscle] Executing physical pick and place sequence...")
    pick_pos = [pick_x, pick_y, WORLD_TABLE_HEIGHT + 0.02]
    drop_pos = [drop_x, drop_y, WORLD_TABLE_HEIGHT + 0.20]

    robot.operate_gripper(close=False) 
    
    print(" -> Hovering...")
    robot.move_to_world_target([pick_x, pick_y, WORLD_TABLE_HEIGHT + 0.15])
    
    print(" -> Grabbing...")
    robot.move_to_world_target(pick_pos)
    robot.operate_gripper(close=True)
    
    print(" -> Lifting...")
    robot.move_to_world_target([pick_x, pick_y, WORLD_TABLE_HEIGHT + 0.15])
    
    print(" -> Moving to Bin...")
    robot.move_to_world_target(drop_pos)
    
    print(" -> Releasing...")
    robot.operate_gripper(close=False)
    print("\n[System] Task Complete!")

print("\nWaking up Webots and waiting for a valid camera frame...")
    
image_ready = False
# Give the simulator up to 100 steps to produce a valid image
for _ in range(100):
    supervisor.step(timestep)
    
    # Check if file exists AND is larger than 10KB (ensuring it's fully written)
    if os.path.exists(CAM_FRAME_PATH) and os.path.getsize(CAM_FRAME_PATH) > 10000:
        # Take one extra step just to ensure the OS has closed the file lock
        supervisor.step(timestep) 
        image_ready = True
        break

if not image_ready:
    print("[Error] Camera frame is missing or corrupted. Is the camera_controller running?")
else:
    print(f"[System] Valid image detected ({os.path.getsize(CAM_FRAME_PATH)} bytes). Proceeding...")
    raw_text = get_vlm_target()
    
    if raw_text:
        # 2. Translate the Output
        target_x, target_y, drop_x, drop_y = parse_and_convert(raw_text)
            
        if target_x is not None:
            execute_pick_and_place(robot, target_x, target_y, drop_x, drop_y)
        else:
            print("Sequence aborted: Could not understand VLM coordinates.")
    else:
        print("Sequence aborted: VLM failed to respond.")

# Keep simulation running
while supervisor.step(timestep) != -1:
    pass