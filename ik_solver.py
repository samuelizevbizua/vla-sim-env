import warnings
from ikpy import chain

warnings.filterwarnings("ignore", category=UserWarning, module='ikpy')

class IK_Solver:
    def __init__(self, urdf_path: str) -> None:
        self.chain = chain.Chain.from_urdf_file(urdf_path)
                
        # Check if joint_type exists to avoid 'OriginLink' error
        self.joints_mask = []
        for link in self.chain.links:
            # If it's the virtual origin or a fixed/prismatic joint, mask it out (False)
            if not hasattr(link, 'joint_type') or link.joint_type == 'fixed' or link.joint_type == 'prismatic':
                self.joints_mask.append(False)
            else:
                self.joints_mask.append(True)

        # Optional: Print to verify it caught exactly 7 joints
        active_count = sum(self.joints_mask)
        print(f"IK Solver initialized with {active_count} active degrees of freedom.")

    def get_link_pos(self, target_pos: list[float], initial_pos: list[float]) -> list[float]:
        padded_initial_pos = [0.0] * len(self.chain.links)
        padded_initial_pos[1:len(initial_pos) + 1] = initial_pos

        link_pos = self.chain.inverse_kinematics(
            target_position=target_pos,
            initial_position=padded_initial_pos,
            active_links_mask=self.joints_mask
        )

        return list(link_pos[1:len(initial_pos) + 1])