import mujoco
from dataclasses import dataclass


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


