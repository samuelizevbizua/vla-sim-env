import requests
from config import VLA_API_URL

def ACTION(base64_img, instruction):
    """
    Sends an image and instruction to the OpenVLA server via ngrok.
    Returns the action array [dx, dy, dz, rx, ry, rz, gripper] or None if it fails.
    """
    payload = {
        "image": base64_img,
        "instruction": instruction
    }

    try:
        response = requests.post(
            f"{VLA_API_URL}/predict",
            json=payload,
            timeout=10, # 10 seconds max wait before assuming network dropout
        )

        if response.status_code != 200:
            print(f" -> [Network Error] Server returned {response.status_code}")
            return None

        data = response.json()
        if data.get("status") != "success":
            print(f" -> [Cloud Error] {data.get('message')}")
            return None

        return data["action"]

    except requests.exceptions.Timeout:
        print(" -> [Network Error] Request timed out. Is Colab still running?")
        return None
    except Exception as e:
        print(f" -> [System Error] {e}")
        return None