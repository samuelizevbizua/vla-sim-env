import time
import requests
from config import VLA_API_URL

def ACTION(base64_img, instruction):
    """
    Sends an image and instruction to the VLA server.
    Returns the action array [dx, dy, dz, rx, ry, ... gripper] or None if it fails.
    """
    payload = {
        "image": base64_img,
        "instruction": instruction
    }

    start_time = time.time()
    try:
        response = requests.post(f"{VLA_API_URL}/predict", json=payload, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            # Catch internal server errors hidden inside a 200 OK
            if data.get("status") == "error":
                print(f" -> [Server Error] {data.get('message')}")
                return None
            
            duration = (time.time() - start_time) * 1000
            print(f" -> [Network] Round-trip: {duration:.0f}ms")
            return data.get("action")
        else:
            print(f" -> [Network Error] Status: {response.status_code}")
            return None
    except Exception as e:
        print(f" -> [API Error] {e}")
        return None