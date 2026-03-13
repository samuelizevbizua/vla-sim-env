import os
import sys
import numpy as np
import ikpy.chain as ch

def get_project_root() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.append(root)
    return root

CAMERA_FRAME_PATH = os.path.join(get_project_root(), 'tmp', 'camera_frame.bin')

class IK_Solver:
    def __init__(self, urdf_path: str) -> None:
        chain = ch.Chain.from_urdf_file(urdf_path)
        print("Chain links:")
        for i, link in enumerate(chain.links):
            print(f"  {i}: {link.name}")

        print(f"\nTotal links: {len(chain.links)}")
        # self.robot = rtb.ERobot.URDF(urdf_path)
        # print("Available links:", [link.name for link in self.robot.links])
        # print("EE links:", self.robot.ee_links)

        # ets = self.robot.ets(end='panda_finger::left')
        # self.n = ets.n
        # print(f"DOF to panda_finger::left: {self.n}")
        # self.home_q = np.zeros(self.n)
        pass


    def get_link_pos(self, target_pos: list[float]) -> list[float]:
        # x, y, z = target_pos
        # T_target = SE3(x, y, z) * SE3.OA([0, 1, 0], [0, 0, -1])

        # sol = self.robot.ikine_LM(
        #     T_target,
        #     q0=self.home_q,
        #     end='panda_finger::left',
        #     mask=[1, 1, 1, 0.1, 0.1, 0.1],
        #     slimit=50,
        #     ilimit=500,
        #     tol=1e-6
        # )

        # if not sol.success:
        #     raise RuntimeError(f"IK failed to converge for target {target_pos}")

        # fk = self.robot.fkine(sol.q, end='panda_finger::left')
        # print(f"RTB FK result:  {fk.t}")
        # print(f"RTB target was: {[x, y, z]}")

        # return sol.q[:7].tolist()
        pass