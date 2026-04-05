import os, sys, requests
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from controller import Supervisor
from utils.UR import UR10e_Robot
from config import PERSPECTIVE_RECEIVER, SIM_TABLE_DEF, GET_RECEIVER_BASE64_IMAGE, VLA_API_URL

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
robot = UR10e_Robot(supervisor)

receiver = supervisor.getDevice(PERSPECTIVE_RECEIVER)
receiver.enable(timestep)


table_node = supervisor.getFromDef(SIM_TABLE_DEF)
table_pos  = table_node.getPosition()
TABLE_HEIGHT = table_node.getField("size").getSFVec3f()[2]

DELTA_SCALE = 1.0
INFERENCE_EVERY_N = 3

current_x = float(table_pos[0])
current_y = float(table_pos[1])
current_z = TABLE_HEIGHT + 0.2

print("[System] Moving to start pose...")
robot.move_to_world_target([current_x, current_y, current_z])
robot.operate_gripper(close=False)

print("\n[Brain] VLA loop starting...\n")

step_counter = 0

while supervisor.step(timestep) != -1:
    step_counter += 1

    base64_img = GET_RECEIVER_BASE64_IMAGE(receiver)
    if base64_img is None:
        continue

    if step_counter % INFERENCE_EVERY_N != 0:
        continue

    payload = {
        "image":       base64_img,
        "instruction": "pick up the red cube",
        # "instruction": "pick up the red cube and place it in the green bin",
    }

    try:
        response = requests.post(
            f"{VLA_API_URL}/predict",
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            print(f"[Network Error] {response.status_code}")
            continue

        data = response.json()
        if data.get("status") != "success":
            print(f"[Cloud Error] {data.get('message')}")
            continue

        action        = data["action"]
        dx, dy, dz    = action[0], action[1], action[2]
        gripper_state = action[6]

        print(f"[VLA] dx={dx:+.4f}  dy={dy:+.4f}  dz={dz:+.4f}  grip={gripper_state:.2f}")

        current_x += dx * DELTA_SCALE
        current_y += dy * DELTA_SCALE
        current_z += dz * DELTA_SCALE

        robot.move_to_world_target([current_x, current_y, current_z])
        robot.operate_gripper(close=(gripper_state > 0.5))

    except requests.exceptions.ConnectionError:
        print("[CRITICAL] Ngrok tunnel dead — pausing simulation.")
        supervisor.simulationSetMode(supervisor.SIMULATION_MODE_PAUSE)
        break
    except requests.exceptions.Timeout:
        print("[Warning] Inference timed out — skipping frame.")