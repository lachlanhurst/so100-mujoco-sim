import os
import pathlib
import tempfile
import zipfile
import requests
import shutil

def download(repo_zip_url: str, repo_zip_name: str, repo_zip_path: str, destination_folder: str):
    """
    Downloads a specific folder from a zip file hosted at a given URL and copies it to a destination folder.

    :param repo_zip_url: URL of the zip file to download.
    :param repo_zip_xml_path: Path within the zip file to the folder to extract.
    :param destination_folder: Path to the destination folder where the extracted files will be copied.
    """
    print(f"Downloading {repo_zip_url} repo")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the repo zip file to temp folder
        zip_path = os.path.join(temp_dir, "repo.zip")
        response = requests.get(repo_zip_url)
        print("Extracting repo zip")
        with open(zip_path, "wb") as zip_file:
            zip_file.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        print("Copying from tmp to destination")
        # Locate the desired folder
        extracted_folder = os.path.join(temp_dir, repo_zip_name, repo_zip_path)
        if not os.path.exists(extracted_folder):
            raise FileNotFoundError(f"Path {extracted_folder} not found in the extracted zip.")

        # Create the destination folder
        os.makedirs(destination_folder, exist_ok=True)

        # Copy the required files to the destination folder
        shutil.copytree(extracted_folder, destination_folder, dirs_exist_ok=True)

    print(f"{repo_zip_url}/{repo_zip_path} downloaded into {destination_folder}")


if __name__ == "__main__":

    # download the mujoco xml model
    download(
        'https://github.com/google-deepmind/mujoco_menagerie/archive/68ff0ee1198e993bd824084b9ccd1826d835ea9f.zip',
        'mujoco_menagerie-68ff0ee1198e993bd824084b9ccd1826d835ea9f',
        'trs_so_arm100',
        os.path.join(pathlib.Path(__file__).parent.parent, "src/so100_mujoco_sim/xml")
    )

    # download the lerobot code
    download(
        'https://github.com/huggingface/lerobot/archive/5322417c0302b517b94d938e12b0e10405e6b649.zip',
        'lerobot-5322417c0302b517b94d938e12b0e10405e6b649',
        'lerobot',
        os.path.join(pathlib.Path(__file__).parent.parent, "src/lerobot")
    )
