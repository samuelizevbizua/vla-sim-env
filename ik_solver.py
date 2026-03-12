import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

# Derived from URDF: link7→hand: 0.049 + hand→finger: 0.0584 + finger tip: 0.045
FINGER_LENGTH = 0.049 + 0.0584 + 0.045  # = 0.1524m

class IK_Solver:
    def __init__(self) -> None:
        self.robot = rtb.models.Panda()
        self.home_q = self.robot.qr

    def get_link_pos(self, target_pos: list[float]) -> list[float]:
        x, y, z = target_pos

        # Offset Z up by finger length so wrist sits above target
        # and fingers land exactly at the target position
        T_target = SE3(x, y, z + FINGER_LENGTH) * SE3.OA([0, 1, 0], [0, 0, -1])

        sol = self.robot.ikine_LM(T_target, q0=self.home_q)

        if not sol.success:
            raise RuntimeError(f"IK failed to converge for target {target_pos}")

        return sol.q.tolist()