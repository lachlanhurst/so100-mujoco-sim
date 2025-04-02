import mujoco
from dataclasses import dataclass
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from configs.so100 import So100Config

@dataclass
class Joint:
    """ class for representing a joint"""
    name: str
    # radians, min and max
    range: tuple[float, float]

    def __repr__(self):
        return f"Joint({self.name}, {self.range})"


def joints_from_model(model: mujoco.MjModel) -> list[Joint]:
    """
    Extracts joint details from a mujoco model
    """
    # get the number of joints
    num_joints = model.njnt

    # get joint names
    joint_names = []
    for i in range(num_joints):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_names.append(name)

    # get joint ranges
    joint_ranges = model.jnt_range.reshape(-1, 2)

    joints: list[Joint] = []
    for i in range(num_joints):
        j = Joint(joint_names[i], tuple(joint_ranges[i]))
        joints.append(j)
    return joints


class ArmController:
    """
    Base class for controlling a robotic arm
    """
    def __init__(self, joints: list[Joint]):
        self.joints = joints
        self.joint_set_positions = [0.0] * len(self.joints)
        self.joint_actual_positions = [0.0] * len(self.joints)

    def set_joint_set_position(self, joint_name: str, position: float):
        for i, joint in enumerate(self.joints):
            if joint.name == joint_name:
                # Clamp the position within the joint's range
                clamped_position = max(joint.range[0], min(position, joint.range[1]))
                self.joint_set_positions[i] = clamped_position
                break

    def set_joint_actual_position(self, joint_name: str, position: float):
        for i, joint in enumerate(self.joints):
            if joint.name == joint_name:
                # Clamp the position within the joint's range
                clamped_position = max(joint.range[0], min(position, joint.range[1]))
                self.joint_actual_positions[i] = clamped_position
                break

    def get_joint_set_position(self, joint_name: str) -> float:
        for i, joint in enumerate(self.joints):
            if joint.name == joint_name:
                return self.joint_set_positions[i]
        raise ValueError(f"Joint {joint_name} not found")

    def get_joint_set_positions(self) -> list[float]:
        return self.joint_set_positions

    def set_joint_set_positions(self, positions: list[float]):
        if len(positions) != len(self.joints):
            raise ValueError(f"Expected {len(self.joints)} joint positions, got {len(positions)}")
        # Clamp each position within the corresponding joint's range
        self.joint_set_positions = [
            max(joint.range[0], min(position, joint.range[1]))
            for joint, position in zip(self.joints, positions)
        ]

    def reset(self):
        self.joint_set_positions = [0.0] * len(self.joints)
        self.joint_actual_positions = [0.0] * len(self.joints)
    
    def update(self):
        pass


class MujocoArmController(ArmController):
    """
    Class for controlling the mujoco model of a robotic arm in mujoco
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        joints = joints_from_model(model)
        super().__init__(joints)
        self.model = model
        self.data = data

    def update(self):
        super().update()

        for joint in self.joints:
            # get the actual position of the joint from the mujoco model
            # and update the arm controller values
            joint_actual_pos = self.data.joint(joint.name).qpos[0]
            self.set_joint_actual_position(joint.name, joint_actual_pos)

    def set_positions(self):
        """
        Applies the set joint permissions to the mujoco model
        """
        for i, joint in enumerate(self.joints):
            self.data.actuator(joint.name).ctrl = self.joint_set_positions[i]


class So100ArmController(ArmController):
    """
    Class for controlling the So100 robotic arm
    """
    def __init__(self, port: str, calibration_dir: str):
        # Create the So100 robot from the configuration
        self.robot = make_robot_from_config(
            So100Config(calibration_dir=calibration_dir, port=port)
        )

        print("self.robot.config.follower_arms")
        print(self.robot.config.follower_arms)
        self.robot.connect()
        print(f"{self.robot.capture_observation()}")

        # Define the joints of the So100 arm
        joints = [
            Joint("shoulder_pan", (-1.57, 1.57)),
            Joint("shoulder_lift", (-1.57, 1.57)),
            Joint("elbow_flex", (-1.57, 1.57)),
            Joint("wrist_flex", (-1.57, 1.57)),
            Joint("wrist_roll", (-1.57, 1.57)),
            Joint("gripper", (0, 0.04)),
        ]
        super().__init__(joints)

    def update(self):
        super().update()
        # Update the actual positions of the joints by reading from the robot
        # This is where you would read the actual positions of the joints from the robot
        # and update the joint_actual_positions attribute
        pass

    def set_positions(self):
        """
        Applies the set joint permissions to the So100 robot
        """
        # This is where you would send the set joint positions to the robot
        # for example, using a serial connection or ROS
        pass