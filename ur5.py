"""
    UR5e controller for Webots — stripped from Allan Souza Almeida's original.
    Vision/CNN components removed. Adapted to accept external Supervisor instance.

    Usage:
        from ur5 import UR5
        ur5 = UR5(supervisor)
        ur5.move_to_pose([x, y, z], [rot_x, rot_y, rot_z])
        ur5.actuate_gripper(1)  # 1 = close, 0 = open
"""

import numpy as np
from math import pi, cos, sin
import math
from functools import reduce
from scipy.spatial.transform import Rotation

PI = pi


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rot_x(theta):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta),  np.cos(theta), 0],
        [0, 0, 0, 1],
    ])


def rot_y(theta):
    return np.array([
        [ np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1],
    ])


def rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


def limit_angle(angle):
    """Wraps angle to [-pi, pi]."""
    angle_mod = angle % (2 * np.pi)
    if angle_mod > np.pi:
        return angle_mod - 2 * np.pi
    return angle_mod


def build_matrix(pos: list, rot: list, euler: str = "XYZ") -> np.ndarray:
    """
    Build a 4x4 homogeneous transformation matrix from position and Euler angles.

    Parameters:
        pos: [x, y, z]
        rot: [rx, ry, rz] in radians
        euler: 'XYZ' or 'ZYX'
    """
    Rx = rot_x(rot[0])
    Ry = rot_y(rot[1])
    Rz = rot_z(rot[2])
    if euler == "XYZ":
        R = reduce(np.dot, [Rx, Ry, Rz])
    else:
        R = reduce(np.dot, [Rz, Ry, Rx])
    R[0][3] = pos[0]
    R[1][3] = pos[1]
    R[2][3] = pos[2]
    return R


def matrix_error(T1: np.ndarray, T2: np.ndarray):
    """
    Returns (angle_error_deg, position_error_mm) between two 4x4 transforms.
    """
    R_diff = np.dot(T1[:3, :3], T2[:3, :3].T)
    angle = np.linalg.norm(Rotation.from_matrix(R_diff).as_rotvec()) * 180 / PI
    dist = np.linalg.norm(T1[:3, 3] - T2[:3, 3]) * 1000
    return angle, dist


# ---------------------------------------------------------------------------
# UR5e DH kinematics (module-level functions)
# ---------------------------------------------------------------------------

# UR5e DH parameters
_d1  = 0.1625
_a2  = 0.425
_a3  = 0.3922
_d4  = 0.1333
_d5  = 0.0997
_d6  = 0.0996 + 0.1237

_DH = np.array([
    [0,   PI / 2,  _d1,  0],
    [_a2, 0,       0,    PI / 2],
    [_a3, 0,       0,    0],
    [0,  -PI / 2,  _d4, -PI / 2],
    [0,   PI / 2,  _d5,  0],
    [0,   0,       _d6,  0],
])


def _dh_matrix(theta, i):
    """Single DH transform for joint i at angle theta."""
    a, alpha, d, offset = _DH[i]
    t = theta + offset
    return np.array([
        [ cos(t), -sin(t)*cos(alpha),  sin(t)*sin(alpha), a*cos(t)],
        [ sin(t),  cos(t)*cos(alpha), -cos(t)*sin(alpha), a*sin(t)],
        [0,        sin(alpha),         cos(alpha),         d       ],
        [0,        0,                  0,                  1       ],
    ])


def forward_kinematics(theta: list) -> tuple:
    """
    UR5e forward kinematics using DH parameters.

    Parameters:
        theta: list of 6 joint angles in radians

    Returns:
        (T, A): T is the 4x4 total transform, A is array of individual transforms
    """
    A = np.array([_dh_matrix(theta[i], i) for i in range(6)])
    T = reduce(np.dot, A)
    return T, A


def inverse_kinematics(
    th: np.ndarray,
    shoulder: str = "left",
    wrist: str = "down",
    elbow: str = "up"
) -> list:
    """
    UR5e analytical inverse kinematics.

    Parameters:
        th: 4x4 target transformation matrix in robot base frame
        shoulder: 'left' or 'right'
        wrist: 'up' or 'down'
        elbow: 'up' or 'down'

    Returns:
        [theta1..theta6] in radians

    Raises:
        ValueError if target is unreachable
    """
    try:
        o5 = th.dot(np.array([[0, 0, -_d6, 1]]).T)
        xc, yc = o5[0][0], o5[1][0]

        # theta1
        psi = math.atan2(yc, xc)
        phi = math.acos(_d4 / np.sqrt(xc**2 + yc**2))
        T1 = np.array([
            limit_angle(psi - phi + PI / 2),
            limit_angle(psi + phi + PI / 2),
        ])
        theta1 = T1[0] if shoulder == "left" else T1[1]

        # theta5
        P60 = np.dot(th, np.array([[0, 0, 0, 1]]).T)
        z61 = P60[0][0] * np.sin(T1) - P60[1][0] * np.cos(T1)
        T5 = np.array([
            np.arccos((z61 - _d4) / _d6),
           -np.arccos((z61 - _d4) / _d6),
        ]).T
        if shoulder == "left":
            theta5 = T5[0][0] if wrist == "up" else T5[0][1]
        else:
            theta5 = T5[1][0] if wrist == "down" else T5[1][1]

        # theta6
        th01 = np.linalg.inv(_dh_matrix(theta1, 0))
        th16 = np.linalg.inv(np.dot(th01, th))
        theta6 = limit_angle(
            math.atan2(-th16[1][2] / np.sin(theta5),
                        th16[0][2] / np.sin(theta5)) + PI
        )

        # theta3
        th61  = np.dot(th01, th)
        th54  = _dh_matrix(theta5, 4)
        th65  = _dh_matrix(theta6, 5)
        th41  = np.dot(th61, np.linalg.inv(np.dot(th54, th65)))
        p31   = np.dot(th41, np.array([[0, _d4, 0, 1]]).T) - np.array([[0, 0, 0, 1]]).T
        p31_x, p31_y = p31[0][0], p31[1][0]
        D = (p31_x**2 + p31_y**2 - _a2**2 - _a3**2) / (2 * _a2 * _a3)
        T3 = np.array([
            math.atan2(-np.sqrt(1 - D**2), D),
            math.atan2( np.sqrt(1 - D**2), D),
        ])
        if shoulder == "left":
            theta3 = T3[0] if elbow == "up" else T3[1]
        else:
            theta3 = T3[1] if elbow == "up" else T3[0]

        # theta2
        epsilon = math.acos(
            ((_a2**2 + p31_x**2 + p31_y**2 - _a3**2)
             / (2 * _a2 * np.sqrt(p31_x**2 + p31_y**2)))
        )
        delta = math.atan2(p31_x, p31_y)
        T2 = np.array([-delta + epsilon, -delta - epsilon])
        theta2 = T2[0] if shoulder == "left" else T2[1]

        # theta4
        th21 = _dh_matrix(theta2, 1)
        th32 = _dh_matrix(theta3, 2)
        th43 = np.dot(np.linalg.inv(np.dot(th21, th32)), th41)
        theta4 = math.atan2(th43[0][0], -th43[1][0])

        return [theta1, theta2, theta3, theta4, theta5, theta6]

    except ValueError:
        raise ValueError(f"Target position unreachable for the UR5e arm")


# ---------------------------------------------------------------------------
# UR5 class
# ---------------------------------------------------------------------------

class UR5:
    """
    UR5e controller. Pass in an existing Webots Supervisor instance.

    Example:
        supervisor = Supervisor()
        ur5 = UR5(supervisor)
        ur5.move_to_pose([0.4, 0.2, 0.3], [PI, 0, 0])
        ur5.actuate_gripper(1)
    """

    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(supervisor.getBasicTimeStep())
        self.joints = None
        self.finger_joints = None
        self.finger_joint_limits = None
        self._init_handles()
        print("UR5 ready.")

    def _init_handles(self):
        joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        finger_names = [
            "finger_1_joint_1",
            "finger_1_joint_2",
            "finger_1_joint_3",
            "finger_2_joint_1",
            "finger_2_joint_2",
            "finger_2_joint_3",
            "finger_middle_joint_1",
            "finger_middle_joint_2",
            "finger_middle_joint_3",
        ]

        self.joints = [self.supervisor.getDevice(n) for n in joint_names]
        self.finger_joints = [self.supervisor.getDevice(n) for n in finger_names]

        self.finger_joint_limits = [
            [0.0695, 0.8],
            [0.01,   1.0],
            [-0.8,  -0.0723],
            [0.0695, 0.8],
            [0.01,   1.0],
            [-0.8,  -0.0723],
            [0.0695, 0.8],
            [0.01,   1.0],
            [-0.8,  -0.0723],
        ]
        self._setup_control_mode()

    def _setup_control_mode(self):
        """Set arm joints to velocity control, enable all sensors."""
        for joint in self.joints:
            joint.setPosition(float("inf"))
            joint.getPositionSensor().enable(self.timestep)
        for joint in self.finger_joints:
            joint.setVelocity(100)
            joint.getPositionSensor().enable(self.timestep)

    def get_joint_angles(self) -> np.ndarray:
        """Returns current arm joint angles as numpy array."""
        return np.array([j.getPositionSensor().getValue() for j in self.joints])

    def get_finger_angles(self) -> np.ndarray:
        """Returns current finger joint angles as numpy array."""
        return np.array([j.getPositionSensor().getValue() for j in self.finger_joints])

    def get_object_frame(self, node) -> np.ndarray:
        """
        Returns the 4x4 homogeneous transform of a Webots node
        in the robot base frame.

        Parameters:
            node: Webots node object (from supervisor.getFromDef())

        Returns:
            4x4 numpy array
        """
        R_obj  = np.array(node.getOrientation()).reshape(3, 3)
        T_obj  = np.array(node.getPosition()).reshape(3, 1)
        th_obj_world = np.hstack((
            np.vstack((R_obj, np.zeros((1, 3)))),
            np.vstack((T_obj, [[1]]))
        ))
        R0     = np.array(self.supervisor.getSelf().getOrientation()).reshape(3, 3)
        T0     = np.array(self.supervisor.getSelf().getPosition()).reshape(3, 1)
        th0_world = np.hstack((
            np.vstack((R0, np.zeros((1, 3)))),
            np.vstack((T0, [[1]]))
        ))
        return np.dot(np.linalg.inv(th0_world), th_obj_world)

    def move_to_config(self, target: list, duration: float = None):
        """
        Move arm to joint configuration using quintic polynomial trajectory.

        Parameters:
            target: list of 6 target joint angles in radians
            duration: motion duration in seconds (auto-calculated if None)
        """
        self.supervisor.step(self.timestep)
        t0  = self.supervisor.getTime()
        q0  = self.get_joint_angles()
        qf  = np.array(target)
        v0  = np.zeros(6)
        vf  = np.zeros(6)
        a0  = np.zeros(6)
        af  = np.zeros(6)

        if duration is None:
            duration = max(np.max(np.abs(qf - q0)) * (4 / (0.5 * PI)), 1.5)

        tf = t0 + duration
        A  = np.array([
            [1, t0, t0**2,    t0**3,    t0**4,    t0**5],
            [0, 1,  2*t0,     3*t0**2,  4*t0**3,  5*t0**4],
            [0, 0,  2,        6*t0,     12*t0**2, 20*t0**3],
            [1, tf, tf**2,    tf**3,    tf**4,    tf**5],
            [0, 1,  2*tf,     3*tf**2,  4*tf**3,  5*tf**4],
            [0, 0,  2,        6*tf,     12*tf**2, 20*tf**3],
        ])
        b = np.array([q0, v0, a0, qf, vf, af])
        x = [np.linalg.solve(A, b[:, i]) for i in range(6)]

        while self.supervisor.getTime() <= tf:
            t = self.supervisor.getTime()
            for idx, joint in enumerate(self.joints):
                joint.setVelocity(
                    x[idx][1]
                    + 2  * x[idx][2] * t
                    + 3  * x[idx][3] * t**2
                    + 4  * x[idx][4] * t**3
                    + 5  * x[idx][5] * t**4
                )
            self.supervisor.step(self.timestep)

        for joint in self.joints:
            joint.setVelocity(0)

    def move_to_pose(
        self,
        pos: list,
        rot: list,
        euler: str = "XYZ",
        shoulder: str = "left",
        wrist: str = "down",
        duration: float = None,
    ):
        """
        Move arm to Cartesian pose using analytical IK.

        Parameters:
            pos: [x, y, z] in robot base frame (metres)
            rot: [rx, ry, rz] Euler angles (radians)
            euler: Euler convention, 'XYZ' or 'ZYX'
            shoulder: 'left' or 'right'
            wrist: 'up' or 'down'
            duration: motion duration in seconds (auto if None)
        """
        T = build_matrix(pos, rot, euler=euler)
        joints = inverse_kinematics(T, shoulder=shoulder, wrist=wrist, elbow="up")
        self.move_to_config(joints, duration=duration)

    def actuate_gripper(self, close: int = 0, duration: float = 2.0):
        """
        Open or close the Robotiq 3f gripper using quintic trajectory.

        Parameters:
            close: 0 = open, 1 = close
            duration: time in seconds
        """
        t0 = self.supervisor.getTime()
        q0 = self.get_finger_angles()
        qf = np.array([lim[close] for lim in self.finger_joint_limits])
        v0 = np.zeros(9)
        vf = np.zeros(9)
        a0 = np.zeros(9)
        af = np.zeros(9)
        tf = t0 + duration

        A = np.array([
            [1, t0, t0**2,    t0**3,    t0**4,    t0**5],
            [0, 1,  2*t0,     3*t0**2,  4*t0**3,  5*t0**4],
            [0, 0,  2,        6*t0,     12*t0**2, 20*t0**3],
            [1, tf, tf**2,    tf**3,    tf**4,    tf**5],
            [0, 1,  2*tf,     3*tf**2,  4*tf**3,  5*tf**4],
            [0, 0,  2,        6*tf,     12*tf**2, 20*tf**3],
        ])
        b = np.array([q0, v0, a0, qf, vf, af])
        x = [np.linalg.solve(A, b[:, i]) for i in range(9)]

        while self.supervisor.getTime() <= tf:
            t = self.supervisor.getTime()
            for idx, joint in enumerate(self.finger_joints):
                joint.setPosition(
                    x[idx][0]
                    + x[idx][1] * t
                    + x[idx][2] * t**2
                    + x[idx][3] * t**3
                    + x[idx][4] * t**4
                    + x[idx][5] * t**5
                )
            self.supervisor.step(self.timestep)
