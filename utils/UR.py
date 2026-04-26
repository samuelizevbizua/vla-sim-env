import numpy as np
from ur_analytic_ik import ur5e
from config import GRIPPER_TARGET_ANGLE_OPEN, GRIPPER_TARGET_ANGLE_CLOSED,  ARM_MOTOR_VELOCITY, GRIPPER_MOTOR_VELOCITY

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
        # Twist the wrist around the Z-axis. 
        twist_angle = np.radians(90) 
        twist_matrix = np.array([
            [np.cos(twist_angle), -np.sin(twist_angle), 0],
            [np.sin(twist_angle),  np.cos(twist_angle), 0],
            [0,                    0,                   1]
        ])
        
        # Apply the twist to the base orientation
        target_pose[:3, :3] = base_orientation.dot(twist_matrix)

        # Solve using the closest joint configuration AND the gripper length offset
        solution = ur5e.inverse_kinematics_closest_with_tcp(
            target_pose,
            self.tcp_transform,
            *current_joints
        )

        # Return the single best solution
        if solution is not None and len(solution) > 0:
            return np.array(solution).flatten().tolist()
        else:
            print("Target is physically out of reach!")
            return None

class URe_Robot:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getSelf()
        
        # 1. Setup Math Solver
        self.solver = IK_Solver(gripper_length=0.16)
        
        # 2. Setup Hardware
        self._init_arm_hardware()
        self._init_gripper_hardware()
        self._init_matrices()
        
        # 3. Warm up sensors
        for _ in range(5):
            self.supervisor.step(self.timestep)

    def _init_arm_hardware(self):
        motor_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.motors = []
        self.sensors = []
        for name in motor_names:
            motor = self.supervisor.getDevice(name)
            if motor:
                motor.setVelocity(ARM_MOTOR_VELOCITY)
                self.motors.append(motor)
                sensor = motor.getPositionSensor()
                sensor.enable(self.timestep)
                self.sensors.append(sensor)
            else:
                print(f"Warning: Could not find motor '{name}'")

    def _init_gripper_hardware(self):
        gripper_names = [
            'ROBOTIQ 2F-85 Gripper::left finger joint',
            'ROBOTIQ 2F-85 Gripper::right finger joint'
        ]

        self.gripper_motors = []
        for name in gripper_names:
            motor = self.supervisor.getDevice(name)
            if motor:
                motor.setVelocity(GRIPPER_MOTOR_VELOCITY)
                self.gripper_motors.append(motor)
            else:
                print(f"Warning: Could not find gripper motor '{name}'")

    def _init_matrices(self):
        # Webots vs UR Coordinate Alignment
        # -90 degrees aligns the Webots Y-forward world to the UR X-forward solver!
        offset_angle = np.pi
        
        self.rotation_matrix = np.array([
            [np.cos(offset_angle), -np.sin(offset_angle), 0],
            [np.sin(offset_angle),  np.cos(offset_angle), 0],
            [0,                     0,                    1]
        ])

    def _wait_to_arrive(self, target_joints, is_gripper=False):
        tolerance, timeout_seconds = 0.04, 7.0
        max_steps = int((timeout_seconds * 1000) / self.timestep)
        steps = 0

        if is_gripper:
            gripper_stall_steps = int(2000 / self.timestep)
            for _ in range(gripper_stall_steps):
                self.supervisor.step(self.timestep)
            return

        while self.supervisor.step(self.timestep) != -1:
            current_joints = [s.getValue() for s in self.sensors]
            error = np.max(np.abs(np.array(target_joints) - np.array(current_joints)))
            
            if steps % 15 == 0:
                print(f"      -> [Tracking] Max Joint Error: {error:.4f} rad")
            
            if error < tolerance:
                print("      -> [Success] Arm reached target!")
                break
                
            steps += 1
            if steps > max_steps:
                print(f"      -> [Timeout] Forcing next step.")
                break

    def move_to_world_target(self, target_pos_world, wait=True, settle_time_ms=300):
        robot_pos_world = np.array(self.robot_node.getPosition())
        robot_rot_world = np.array(self.robot_node.getOrientation()).reshape(3, 3)

        target_vector_world = target_pos_world - robot_pos_world
        target_pos_local = robot_rot_world.T.dot(target_vector_world)
        ik_target_local = self.rotation_matrix.dot(target_pos_local)

        current_joints = [s.getValue() for s in self.sensors]
        joint_angles = self.solver.get_link_pos(ik_target_local.tolist(), current_joints)

        if joint_angles is not None:
            for i in range(6):
                self.motors[i].setPosition(joint_angles[i])
            if wait:
                self._wait_to_arrive(joint_angles)
                
            if settle_time_ms > 0:
                steps = int(settle_time_ms / self.timestep)
                for _ in range(steps):
                    self.supervisor.step(self.timestep)
        else:
            print("      -> [Error] Target out of physical reach.")

    def operate_gripper(self, close=True):
        target_angle = GRIPPER_TARGET_ANGLE_CLOSED if close else GRIPPER_TARGET_ANGLE_OPEN
        for motor in self.gripper_motors:
            motor.setPosition(target_angle)
        if close:
            self._wait_to_arrive([target_angle]*6, is_gripper=True)
        else:
            self._wait_to_arrive([-GRIPPER_TARGET_ANGLE_OPEN]*6, is_gripper=True)
    
    def command_joint_positions(self, joint_angles, gripper_cmd):
        """
        Directly set joint positions (6 joints) and gripper command.
        Used when the policy outputs joint angles directly.
        """
        for i in range(6):
            self.motors[i].setPosition(joint_angles[i])
        # Gripper: angle 0 = open, 1 = closed (scale to target angle)
        target_gripper_angle = GRIPPER_TARGET_ANGLE_CLOSED if gripper_cmd > 0.5 else GRIPPER_TARGET_ANGLE_OPEN
        for motor in self.gripper_motors:
            motor.setPosition(target_gripper_angle)