import mujoco
import time
import torch
from dataclasses import dataclass
from enum import Enum
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, TorqueMode
from typing import Callable
import csv

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
        j.range = (j.range[0], j.range[1])
        joints.append(j)
    return joints


class ArmController:
    """
    Base class for controlling a robotic arm
    """
    def __init__(self, joints: list[Joint]):
        self.joints = joints
        # the position that this controller has been told to move to
        self.joint_set_positions = [0.0] * len(self.joints)
        # the position this controller is in right now
        self.joint_actual_positions = [0.0] * len(self.joints)
        # the position this controller provides to other controllers
        # when it is primary
        self.joint_output_positions = [0.0] * len(self.joints)

        self._primary = False
        self._name = "Base"
        self._controllable = False

    @property
    def primary(self) -> bool:
        return self._primary

    @primary.setter
    def primary(self, value: bool) -> None:
        self._primary = value
        self._primary_set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def controllable(self) -> bool:
        """
        Returns True if the controller is in a state where
        if can be used to drive the other controllers.
        eg; real robot controller would be false until connected
        """
        return self._controllable

    def set_joint_actual_position(self, joint_name: str, position: float):
        for i, joint in enumerate(self.joints):
            if joint.name == joint_name:
                # Clamp the position within the joint's range
                clamped_position = max(joint.range[0], min(position, joint.range[1]))
                self.joint_actual_positions[i] = clamped_position
                break

    def get_joint_actual_position(self, joint_name: str) -> float:
        for i, joint in enumerate(self.joints):
            if joint.name == joint_name:
                return self.joint_actual_positions[i]
        raise ValueError(f"Joint {joint_name} not found")

    def get_joint_actual_positions(self) -> list[float]:
        return self.joint_actual_positions

    def set_joint_set_position(self, joint_name: str, position: float):
        for i, joint in enumerate(self.joints):
            if joint.name == joint_name:
                # Clamp the position within the joint's range
                clamped_position = max(joint.range[0], min(position, joint.range[1]))
                self.joint_set_positions[i] = clamped_position
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
        self.joint_output_positions = [0.0] * len(self.joints)

    def update(self):
        self.joint_actual_positions = list(self.joint_set_positions)
        self.joint_output_positions = list(self.joint_set_positions)

    def set_positions(self):
        pass

    def _primary_set(self):
        """ override this function if the controller needs to do something when
        its state as primary is changed.
        """
        pass


class UiArmController(ArmController):

    def __init__(self, joints: list[Joint], set_positions_callback: Callable | None = None):
        super().__init__(joints)

        self._name = "User Interface"
        # will always be controllable
        self._controllable = True
        self.set_positions_callback = set_positions_callback

    def set_positions(self):
        if self.set_positions_callback is not None:
            self.set_positions_callback()


class PlaybackRecordState(Enum):
    PLAYING = 1
    RECORDING = 2
    STOPPED = 3


class PlaybackRecordController(ArmController):
    """
    Controls robot joint positions based on existing recording, or records the
    joint positions for later playback. This is determined by the state
    of the controller (set_state).
    """

    def __init__(self, joints: list[Joint], recorded_steps_callback: Callable | None = None):
        super().__init__(joints)

        self._name = "Playback/Record"
        # will always be controllable
        self._controllable = True
        self._recorded_steps_callback = recorded_steps_callback
        self.playback_index = 0
        # last position of the robot arm, it may not be the last recorded position
        # so track this separately
        self.last_position: list[float] = []
        self.recorded_joint_positions: list[list[float]] = []
        self.extra_joint_positions: list[list[float]] = []
        self.state = PlaybackRecordState.STOPPED

    def set_state(self, state: PlaybackRecordState) -> None:
        if self.state == PlaybackRecordState.STOPPED and state == PlaybackRecordState.PLAYING:
            # then we've started playing
            # the current position of the robot may be a fair bit different to the start
            # of the recording, so use the extra positions to get there slowly (instead of jumping)
            back_to_start = calculate_transition_joint_positions(
                self.last_position,
                self.recorded_joint_positions[0],
                0.03
            )
            self.extra_joint_positions = back_to_start
        elif self.state == PlaybackRecordState.PLAYING and state == PlaybackRecordState.STOPPED:
            # if we stop playing, then we need to reset the index
            self.playback_index = 0
        elif self.state == PlaybackRecordState.STOPPED and state == PlaybackRecordState.RECORDING:
            self.playback_index = 0
            self.recorded_joint_positions = []
            self.extra_joint_positions = []

        self.state = state

    @property
    def controllable(self) -> bool:
        # can't playback unless we have something recorded
        return len(self.recorded_joint_positions) > 0

    def update(self):
        super().update()
        if self.state == PlaybackRecordState.PLAYING:
            if self._recorded_steps_callback is not None:
                self._recorded_steps_callback()
            if len(self.extra_joint_positions) > 0:
                joint_positions = self.extra_joint_positions.pop(0)
                self.joint_output_positions = list(joint_positions)
                self.joint_set_positions = list(joint_positions)
                self.last_position = list(self.joint_set_positions)
                return

            if self.playback_index >= len(self.recorded_joint_positions):
                back_to_beginning_steps = calculate_transition_joint_positions(
                    self.last_position,
                    self.recorded_joint_positions[0],
                    0.03
                )
                self.extra_joint_positions = back_to_beginning_steps
                self.playback_index = 0
            joint_positions = self.recorded_joint_positions[self.playback_index]
            self.joint_output_positions = list(joint_positions)
            self.joint_set_positions = list(joint_positions)
            self.last_position = list(self.joint_set_positions)
            self.playback_index += 1

    def set_positions(self):
        self.last_position = list(self.joint_set_positions)
        if self.state == PlaybackRecordState.RECORDING:
            # record the current joint positions
            self.recorded_joint_positions.append(list(self.joint_set_positions))
            if self._recorded_steps_callback is not None:
                self._recorded_steps_callback()

    def save_playback_file(self, file_path: str) -> None:
        """
        Saves the recorded_joint_positions to a CSV file.

        :param file_path: Path to the CSV file where the positions will be saved.
        """
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.recorded_joint_positions)

    def load_playback_file(self, file_path: str) -> None:
        """
        Loads recorded_joint_positions from a CSV file.

        :param file_path: Path to the CSV file to load the positions from.
        """
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            self.recorded_joint_positions = [
                [float(value) for value in row] for row in reader
            ]

        # reset the playback index to the start
        self.playback_index = 0
        self.extra_joint_positions = []

class MujocoArmController(ArmController):
    """
    Class for controlling the mujoco model of a robotic arm in mujoco
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        joints = joints_from_model(model)
        super().__init__(joints)
        self.model = model
        self.data = data

        self._name = "Simulation"
        self._controllable = False

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
    def __init__(self, ):
        self.robot = None

        # we get a JointOutOfRangeError if any of the angle joints exceed +/- 270 deg (4.69 rad)
        # or -10 to 110 for the gripper
        joints = [
            Joint("shoulder_pan", (-4.69, 4.69)),
            Joint("shoulder_lift", (-4.69, 4.69)),
            Joint("elbow_flex", (-4.69, 4.69)),
            Joint("wrist_flex", (-4.69, 4.69)),
            Joint("wrist_roll", (-4.69, 4.69)),
            Joint("gripper", (-0.17, 1.9)),
        ]
        super().__init__(joints)

        self._name = "Robot"

    def connect(self, port: str, calibration_dir: str) -> None:
        # Create the So100 robot from the configuration
        robot = make_robot_from_config(
            So100Config(calibration_dir=calibration_dir, port=port)
        )
        robot.connect()
        # small delay to allow the robot to connect
        time.sleep(0.2)
        self.robot = robot

    def is_connected(self) -> bool:
        """
        Checks if the robot is connected
        """
        if self.robot is None:
            return False
        return self.robot.is_connected

    @property
    def controllable(self) -> bool:
        return self.is_connected()

    def update(self):
        super().update()
        if self.robot is None:
            return
        # Update the actual positions of the joints by reading from the robot
        # This is where you would read the actual positions of the joints from the robot
        # and update the joint_actual_positions attribute
        obs: torch.Tensor = self.robot.capture_observation()['observation.state']

        # print("obs")
        # print(obs)
        obs = torch.deg2rad(obs).tolist()
        # TODO: which motors should be flipped is available in the calibration config
        # kind of that is, the first one isn't reversed in the calibration so not sure
        # what's up
        obs[0] *= -1.0
        obs[1] *= -1.0
        obs[4] *= -1.0

        for i, joint in enumerate(self.joints):
            joint_actual_pos = obs[i]
            self.set_joint_actual_position(joint.name, joint_actual_pos)
        
        # set the output positions to be the actual robot positions
        self.joint_output_positions = list(self.joint_actual_positions)

    def set_positions(self):
        """
        Applies the set joint permissions to the So100 robot
        """
        if self.robot is None:
            return

        position_floats = list(self.joint_set_positions)
        position_floats[0] *= -1.0
        position_floats[1] *= -1.0
        position_floats[4] *= -1.0

        position_tensor = torch.FloatTensor(position_floats)
        position_tensor = torch.rad2deg(position_tensor)

        self.robot.send_action(position_tensor)

        # print("position_tensor")
        # print(position_tensor)

    def _primary_set(self):
        """ override this function if the controller needs to do something when
        its state as primary is changed.
        """
        if self.robot is None:
            return

        for name in self.robot.follower_arms:
            mb: FeetechMotorsBus =  self.robot.follower_arms[name]

            if self.primary:
                mb.write("Torque_Enable", TorqueMode.DISABLED.value)
                mb.write("Lock", 0)
            else:
                mb.write("Torque_Enable", TorqueMode.ENABLED.value)
                mb.write("Lock", 1)


def update_from_controller(source: ArmController, target: ArmController):
    """
    Updates the target controller arm set positions with the actual positions
    from the source arm controller
    """
    # use the arrays directly as while the order of joints is consistent
    # between the real robot config and the mujoco model, the names are not
    target.set_joint_set_positions(source.joint_output_positions)


def positions_aligned(a: list[float], b: list[float], tolerance_rad: float = 0.1) -> bool:
    """
    Checks if each value in list `a` is within `tolerance_rad` of the corresponding value in list `b`.

    :param a: List of float values representing the first set of positions.
    :param b: List of float values representing the second set of positions.
    :param tolerance_rad: The tolerance within which the positions are considered aligned.
    :return: True if all values in `a` are within `tolerance_rad` of the corresponding values in `b`, False otherwise.
    """
    if len(a) != len(b):
        raise ValueError("Lists `a` and `b` must have the same length.")

    return all(abs(a[i] - b[i]) <= tolerance_rad for i in range(len(a)))


def calculate_transition_joint_positions(start_positions: list[float], end_positions: list[float], max_joint_change: float) -> list[list[float]]:
    """
    Calculate the joint positions for a transition between start and end positions.

    :param start_positions: List of float values representing the starting joint positions.
    :param end_positions: List of float values representing the ending joint positions.
    :param max_joint_change: The maximum change allowed for each joint in radians.
    :return: A list of lists, where each inner list represents a joint position at a step in the transition.
    """
    if len(start_positions) != len(end_positions):
        raise ValueError("Start and end positions must have the same length.")

    num_steps = max(abs(end - start) // max_joint_change for start, end in zip(start_positions, end_positions))
    joint_positions = []
    if num_steps == 0:
        # then somehow we are already at the end position
        return []

    for step in range(int(num_steps) + 1):
        interpolated_positions = [
            start + (end - start) * (step / num_steps)
            for start, end in zip(start_positions, end_positions)
        ]
        joint_positions.append(interpolated_positions)

    return joint_positions
