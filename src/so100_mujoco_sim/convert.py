from urdf2mjcf import run
from urdf2mjcf.model import ConversionMetadata, JointParam
import pathlib
import yaml

"""
Script will convert the downloaded so100 urdf to a mujoco xml file, or do the best it can.
A modified working version of the so100.xml file has been included in this repo, so running
this step isn't strictly necessary.

Before running this script you will need to manually edit the downloaded urdf
file to replace the `package://..../file.STL` references with relative references to each of
the stl files (urdf2mjcf seems to not handle this).

There are also some post processing steps required in the output so100.xml file. Specifically:
- including missing material names
- updating body references that have an empty material name
- removing the freejoint created for the base object so it is fixed in world space
"""


urdf_folder = pathlib.Path(__file__).parent.joinpath('urdf')
urdf_file = str(urdf_folder.joinpath('so100/SO_5DOF_ARM100_8j_URDF.SLDASM/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf'))
config_yaml_file = str(urdf_folder.joinpath('so100/SO_5DOF_ARM100_8j_URDF.SLDASM/config/joint_names_SO_5DOF_ARM100_8j_URDF.SLDASM.yaml'))
mjcf_file = str(urdf_folder.joinpath('so100.xml'))


def get_joints():
    joints: list[JointParam] = []
    with open(config_yaml_file, 'r') as file:
        y = yaml.load(file, Loader=yaml.FullLoader)
        joint_names = y['controller_joint_names']
        # first joint name is empty, so filter it out
        joint_names = filter(lambda x: len(x) > 0, joint_names)
        for joint_name in joint_names:
            print(joint_name)
            j = JointParam(name=joint_name, suffixes=[joint_name])
            joints.append(j)
    return joints


def convert():
    joints = get_joints()
    md = ConversionMetadata(joint_params=joints)

    run(
        urdf_path=urdf_file,
        mjcf_path=mjcf_file,
        copy_meshes=True,
        metadata=md
    )


if __name__ == "__main__":
    convert()
