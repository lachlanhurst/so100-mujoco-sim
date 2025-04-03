from lerobot.common.robot_devices.robots.configs import *
from dataclasses import field, dataclass

@RobotConfig.register_subclass("so100ms")
@dataclass
class So100Config(ManipulatorRobotConfig):
    calibration_dir: str = None
    port: str = None

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda self=None: {
            "main": FeetechMotorsBusConfig(
                port=self.port if self else "",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {}
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {}
    )

    mock: bool = False

    @property
    def type(self) -> str:
        return "so100"

    def __post_init__(self):
        # Ensure default values for follower_arms are updated with the provided port
        self.follower_arms = {
            "main": FeetechMotorsBusConfig(
                port=self.port,
                motors={
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            )
        }
