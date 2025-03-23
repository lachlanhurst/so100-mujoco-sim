import os
import pathlib
import tempfile
import zipfile
import requests
import shutil

# the url of the repo zip file
REPO_ZIP_URL = 'https://github.com/TheRobotStudio/SO-ARM100/archive/refs/heads/main.zip'
# path within the zip file to the urdf that we are interested in
REPO_ZIP_URDF_PATH = 'URDF/SO_5DOF_ARM100_8j_URDF.SLDASM/'

"""
We don't want to copy/paste someone elses urdf model here, so download it from their repo.
There are several urdfs in this download, so the following function pulls out the one we
are interested in.
"""

def download():
    print("Downloading so100 urdf...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the repo zip file to temp folder
        zip_path = os.path.join(temp_dir, "repo.zip")
        response = requests.get(REPO_ZIP_URL)
        with open(zip_path, "wb") as zip_file:
            zip_file.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Locate the desired folder
        extracted_folder = os.path.join(temp_dir, "SO-ARM100-main", REPO_ZIP_URDF_PATH)
        if not os.path.exists(extracted_folder):
            raise FileNotFoundError(f"Path {extracted_folder} not found in the extracted zip.")

        # Create the destination folder
        destination_folder = os.path.join(pathlib.Path(__file__).parent, "urdf/so100/SO_5DOF_ARM100_8j_URDF.SLDASM")
        os.makedirs(destination_folder, exist_ok=True)

        # The stuff we need to the destination folder in this repo
        shutil.copytree(extracted_folder, destination_folder, dirs_exist_ok=True)

    print(f"so100 urdf downloaded into {destination_folder}")


if __name__ == "__main__":
    download()
