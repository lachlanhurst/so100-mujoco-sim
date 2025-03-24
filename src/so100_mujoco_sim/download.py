import os
import pathlib
import tempfile
import zipfile
import requests
import shutil

# the url of the repo zip that includes the mujoco so100 model
REPO_ZIP_URL = 'https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip'
# path within the zip file to the model that we are interested in (repo contains many mujoco models)
REPO_ZIP_XML_PATH = 'trs_so_arm100'

"""
We don't want to copy/paste someone elses mujoco model here, so download it from their repo.
There are several models in this download, so the following function pulls out the one we
are interested in.
"""

def download():
    print("Downloading mujoco_menagerie repo")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the repo zip file to temp folder
        zip_path = os.path.join(temp_dir, "repo.zip")
        response = requests.get(REPO_ZIP_URL)
        print("Extracting mujoco_menagerie repo")
        with open(zip_path, "wb") as zip_file:
            zip_file.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        print("Copying so100 mujoco xml into src")
        # Locate the desired folder
        extracted_folder = os.path.join(temp_dir, "mujoco_menagerie-main", REPO_ZIP_XML_PATH)
        if not os.path.exists(extracted_folder):
            raise FileNotFoundError(f"Path {extracted_folder} not found in the extracted zip.")

        # Create the destination folder
        destination_folder = os.path.join(pathlib.Path(__file__).parent, "xml")
        os.makedirs(destination_folder, exist_ok=True)

        # The stuff we need to the destination folder in this repo
        shutil.copytree(extracted_folder, destination_folder, dirs_exist_ok=True)

    print(f"so100 xml downloaded into {destination_folder}")


if __name__ == "__main__":
    download()
