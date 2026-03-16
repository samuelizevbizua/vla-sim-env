import os
import sys
import numpy as np
import xml.etree.cElementTree as ET
from ur_analytic_ik import ur10e

def get_project_root() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.append(root)
    return root

CAMERA_FRAME_PATH = os.path.join(get_project_root(), 'tmp', 'camera_frame.bin')

class IK_Solver:
    def __init__(self, gripper_length: float = 0.16) -> None:
        """
        gripper_length: The estimated distance in meters from the wrist to the fingertips.
        Tune this number up or down in Webots if the robot grabs too high or too low!
        """
        # Create a 4x4 matrix for the Tool Center Point offset
        self.tcp_transform = np.eye(4)

        # Shift the math forward along the Z-axis by the length of the gripper
        self.tcp_transform[2, 3] = gripper_length

    def get_link_pos(self, target_pos: list[float], current_joints: list[float]) -> list[float]:
        """
        Takes a 3D target position [x, y, z] and returns 6 joint angles.
        """
        # Setup the target position matrix
        target_pose = np.eye(4)
        target_pose[:3, 3] = target_pos

        # Base Top-Down Orientation
        base_orientation = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0, -1.0]
        ])

        # --- THE WRIST TWIST ---
        # Twist the wrist by 45 degrees (np.pi / 4). 
        # You can change this to np.pi / 6 for a 30-degree twist!
        twist_angle = np.pi / 4 
        
        twist_matrix = np.array([
            [np.cos(twist_angle), -np.sin(twist_angle), 0],
            [np.sin(twist_angle),  np.cos(twist_angle), 0],
            [0,                    0,                   1]
        ])
        
        # Apply the twist to the base orientation
        target_pose[:3, :3] = base_orientation.dot(twist_matrix)

        # Solve using the closest joint configuration AND the gripper length offset
        solution = ur10e.inverse_kinematics_closest_with_tcp(
            target_pose,
            self.tcp_transform,
            *current_joints # Unpacks the 6 current angles into the function
        )

        # Return the single best solution
        if solution is not None and len(solution) > 0:
            return np.array(solution).flatten().tolist()
        else:
            print("Target is physically out of reach!")
            return None