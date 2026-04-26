import time
import requests
from config import VLA_API_URL

def ACTION(base64_img_agent, base64_img_wrist, state, instruction):
    """
    Sends images and instruction to the VLA server.
    """
    payload = {
        "image_agent": base64_img_agent,
        "image_wrist": base64_img_wrist,
        "state": state,
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
            
            duration = (time.time() - start_time) * 1000
            print(f" -> [Network] Round-trip: {duration:.0f}ms")
            return data.get("actions")
        else:
            print(f" -> [Network Error] Status: {response.status_code}")
            return None
    except Exception as e:
        print(f" -> [API Error] {e}")
        return None