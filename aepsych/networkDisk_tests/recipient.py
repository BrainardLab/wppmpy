"""
Created on Tue Jan  7 23:45:21 2025

@author: brainardlab-adm
"""
# Machine-specific paths are read from local_config.json at the aepsych/ root.
# That file is gitignored. Copy local_config.json.template to local_config.json
# and fill in paths for your machine before running.

import jax

jax.config.update("jax_enable_x64", True)
import os  # noqa: E402
import time  # noqa: E402

from aepsych_dconfig import MachineConfig  # noqa: E402

machine = MachineConfig.from_json()

from analysis.utils_communication import (  # noqa: E402
    CommunicateViaTextFile,
    ExperimentFileManager,
)

subject_id = int(input("Subject ID: "))
subject_init = input("Subject initials: ").strip()
session_today = int(input("Session number today: "))

# %%
networkDisk_path = os.path.join(machine.network_disk_path, f"sub{subject_id}")
expt_info = f"sub{subject_id}_{subject_init}_expt_record.pkl"

# Construct the full path to the pickle file
file_path = os.path.join(networkDisk_path, expt_info)

# Load the experiment file manager state from the pickle file
expt_file_manager = ExperimentFileManager.load_state(file_path)

# Retrieve the list of past session numbers
past_session_keys = list(expt_file_manager.session_data.keys())
past_session_num = [num for num in past_session_keys if isinstance(num, int)]

# Find the most recent session number
session_num = max(past_session_num)

# Retrieve the file name of the most recent session
file_name = expt_file_manager.session_data[session_num]["file_name"]

# Validate the subject's initials and session number against the metadata
if (expt_file_manager.session_data[session_num]["sub_initial"] != subject_init) or (
    expt_file_manager.session_data[session_num]["session_number"] != session_today
):
    expected_init = expt_file_manager.session_data[session_num]["sub_initial"]
    expected_sess = expt_file_manager.session_data[session_num]["session_number"]
    raise ValueError(
        f"Mismatch detected in metadata:\n"
        f"- Expected Subject Initials: {expected_init}, "
        f"but received: {subject_init}.\n"
        f"- Expected Session Number: {expected_sess}, "
        f"but received: {session_today}."
    )

# Initialize communication class
communicator = CommunicateViaTextFile(
    networkDisk_path,
    retry_delay=3 / 60,  # 1 frame
    timeout=1200,
)  # 1200s = 20 mins
communicator.check_and_handle_file(file_name)

# Step 1: Wait for Initialization
print("Waiting for initialization command...")
communicator.confirm_communication()
print("Initialization confirmed.")

# Step 2: Wait for and confirm RGB values
trial_counter = 0
while True:
    if communicator.terminate:
        break
    print(f"Trial #{trial_counter}...")
    communicator.confirm_RGBvals(response_delay=0.1)
    trial_counter += 1
    time.sleep(0.01)
    print("RGB values confirmed.")
