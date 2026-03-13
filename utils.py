import os
import sys
import numpy as np
from ur_analytic_ik import ur5e

def get_project_root() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.append(root)
    return root

CAMERA_FRAME_PATH = os.path.join(get_project_root(), 'tmp', 'camera_frame.bin')

class IK_Solver:
    def __init__(self, urdf_path: str) -> None:
        print("Initializing IK_Solver with URDF path:", urdf_path)
        pass


    def get_link_pos(self, target_pos: list[float]) -> list[float]:
        
        pass