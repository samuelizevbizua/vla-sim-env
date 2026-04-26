import time
import requests
from config import VLA_API_URL

def ACTION(image_top_b64, image_depth_b64, image_hand_b64, instruction):
    """
    Sends top, depth, and wrist images to the VLA server.
    Returns the first action (list of 7 floats) or None.
    """
    payload = {
        "image_top": image_top_b64,
        "image_depth": image_depth_b64,
        "image_hand": image_hand_b64,
        "instruction": instruction
    }

    start_time = time.time()
    try:
        response = requests.post(f"{VLA_API_URL}/predict", json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "error":
                print(f" -> [Server Error] {data.get('message')}")
                return None
            actions_chunk = data.get("actions")
            if actions_chunk and len(actions_chunk) > 0:
                first_action = actions_chunk[0]
                duration = (time.time() - start_time) * 1000
                print(f" -> [Network] Round-trip: {duration:.0f}ms")
                return first_action
            else:
                print(" -> [Server] No actions in response")
                return None
        else:
            print(f" -> [Network Error] Status: {response.status_code}")
            return None
    except Exception as e:
        print(f" -> [API Error] {e}")
        return None