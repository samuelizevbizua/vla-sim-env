import numpy as np
from ur_analytic_ik import ur5e
from config import (
    GRIPPER_TARGET_ANGLE_OPEN, GRIPPER_TARGET_ANGLE_CLOSED,
    ARM_MOTOR_VELOCITY, GRIPPER_MOTOR_VELOCITY, SIM_TABLE_DEF,
    GRIPPER_LIFT_HEIGHT
)

class IK_Solver:
    def __init__(self, gripper_length: float = 0.16) -> None:
        """
        gripper_length: The estimated distance in meters from the wrist to the fingertips.
        """
        self.tcp_transform = np.eye(4)
        self.tcp_transform[2, 3] = gripper_length

    def get_link_pos(self, target_pos: list[float], current_joints: list[float]) -> list[float]:
        """
        Returns joint angles for the target position, guaranteed to avoid wrist wind‑up
        and 360‑degree spins by unwrapping angles relative to current physical joints.
        """
        target_pose = np.eye(4)
        target_pose[:3, 3] = target_pos

        # FIX: Flipped X and Y to match the 180-degree rotation applied in move_to_world_target
        base_orientation = np.array([
            [ 1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0]
        ])
        target_pose[:3, :3] = base_orientation

        # Get all 8 mathematical solutions
        solutions = ur5e.inverse_kinematics_with_tcp(target_pose, self.tcp_transform)

        best_sol = None
        best_score = float('inf')
        current = np.array(current_joints).flatten()

        for sol in solutions:
            if sol is None:
                continue

            flat_sol = np.array(sol).flatten()
            unwrapped_sol = []

            # 1. UNWRAP ANGLES – find the modulo 2π equivalent nearest to current joint angle
            for i in range(6):
                diff = flat_sol[i] - current[i]
                # shortest angular difference in [-π, π]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                unwrapped_sol.append(current[i] + diff)

            unwrapped_sol = np.array(unwrapped_sol)

            # 2. JOINT CHANGES
            joint_changes = np.abs(unwrapped_sol - current)

            # 3. SCORE THE SOLUTION
            max_change = np.max(joint_changes)          # largest single joint jump
            wrist_change = np.sum(joint_changes[3:])    # wrist joints (3,4,5)
            total_change = np.sum(joint_changes)

            # Penalise configuration flips (max_change) and wrist movement heavily
            score = (max_change * 10.0) + (wrist_change * 5.0) + total_change

            if score < best_score:
                best_score = score
                best_sol = unwrapped_sol

        if best_sol is not None:
            # Return the raw unwrapped angles (do not normalise)
            return best_sol.tolist()
        else:
            print("      -> [Error] No valid IK solution found (target unreachable).")
            return None


class URe_Robot:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getSelf()
        
        # 1. Setup Math Solver
        self.solver = IK_Solver(gripper_length=0.16)
        self._target_joints = None
        self.home_joints = None
        
        self._init_arm_hardware()
        self._init_gripper_hardware()
        self._init_matrices()
        
        # 3. Warm up sensors
        for _ in range(5):
            self.supervisor.step(self.timestep)
            
        # Now this will successfully calculate the IK and block the 
        # controller until the arm has physically arrived at home.
        self._init_home_pose()

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

    def _init_matrices(self):
        # Webots → UR coordinate alignment (no extra twist)
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
            if error < tolerance:
                break
            steps += 1
            if steps > max_steps:
                print("      -> [Timeout] Forcing next step.")
                break

    def move_to_world_target(self, target_pos_world, wait=True, settle_time_ms=500):
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
            return True
        else:
            print("      -> [Error] Target out of physical reach.")
            return False

    def operate_gripper(self, close=True):
        target_angle = GRIPPER_TARGET_ANGLE_CLOSED if close else GRIPPER_TARGET_ANGLE_OPEN
        for motor in self.gripper_motors:
            motor.setPosition(target_angle)
        if close:
            self._wait_to_arrive([target_angle]*6, is_gripper=True)
        else:
            self._wait_to_arrive([-GRIPPER_TARGET_ANGLE_OPEN]*6, is_gripper=True)

    def move_to_world_target_async(self, target_pos_world):
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
            self._target_joints = joint_angles
        else:
            print("      -> [Error] Target out of physical reach.")
            self._target_joints = None

    def is_target_reached(self, tolerance=0.04):
        if self._target_joints is None:
            # If there was an IK error, treat the target as 'reached' to avoid waiting.
            return True
        current = [s.getValue() for s in self.sensors]
        error = max(abs(np.array(current) - np.array(self._target_joints)))
        return error < tolerance

    def set_joint_positions_async(self, joint_angles):
        if len(joint_angles) == 6:
            for i in range(6):
                self.motors[i].setPosition(joint_angles[i])
            self._target_joints = joint_angles
        else:
            print("      -> [Error] Must provide exactly 6 joint angles.")

    def get_end_effector_pose(self) -> list[float]:
        current_joints = [s.getValue() for s in self.sensors]
        fk_mat = ur5e.forward_kinematics(*current_joints)
        tcp_mat = np.eye(4)
        tcp_mat[2, 3] = 0.16
        wrist_pose = fk_mat @ tcp_mat
        x_ur, y_ur, z_ur = wrist_pose[:3, 3]
        world_x = -y_ur
        world_y = x_ur
        world_z = z_ur
        R = wrist_pose[:3, :3]
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = np.arctan2(R[1,0], R[0,0])
        else:
            roll = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = 0.0
        return [float(world_x), float(world_y), float(world_z), float(roll), float(pitch), float(yaw)]

    def _init_home_pose(self):
        """
        Finds a reachable home pose above the table centre.
        Uses the same lift height as in config (GRIPPER_LIFT_HEIGHT) – works because normalisation is applied.
        """
        print("      -> Moving to home pose...")
        table_node = self.supervisor.getFromDef(SIM_TABLE_DEF)
        table_pos = table_node.getPosition()
        table_height = table_node.getField("size").getSFVec3f()[2]
        centre_x = table_pos[0]
        centre_y = table_pos[1]
        centre_z = table_height

        target_z = centre_z + GRIPPER_LIFT_HEIGHT
        target_pos = [centre_x, centre_y, target_z]
        self.move_to_world_target(target_pos, wait=True, settle_time_ms=3500)
        self.home_joints = [float(s.getValue()) for s in self.sensors]
        print("🦾     -> ARM at home pose, continuing rest of controller...")