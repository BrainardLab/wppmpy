"""
Created on Tue Jan  7 23:45:21 2025

@author: brainardlab-adm
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.utils_communication import (
    CommunicateViaTextFile,
    ExperimentFileManager,
    get_experiment_info_custom,
)
from analysis.utils_load import get_path

# Prompt the user to enter experiment information using a custom Tkinter popup
# Collects subject ID, initials, and today's session number
subject_id, subject_init, session_today = get_experiment_info_custom()

# %%
is_practice = True
if is_practice:
    # Define the shared network path specific to the subject
    networkDisk_path = (
        f"{get_path('network_disk_windows_mapped_lower')}sub{subject_id}\\practice"
    )

    # Metadata file name (pkl) tracking the subject's experiment data
    expt_info = f"sub{subject_id}_{subject_init}_expt_record_practice.pkl"
else:
    # Define the shared network path specific to the subject
    networkDisk_path = f"{get_path('network_disk_windows_mapped_lower')}sub{subject_id}"

    # Metadata file name (pkl) tracking the subject's experiment data
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
expected_init = expt_file_manager.session_data[session_num]["sub_initial"]
expected_session = expt_file_manager.session_data[session_num]["session_number"]
if (expected_init != subject_init) or (expected_session != session_today):
    raise ValueError(
        f"Mismatch detected in metadata:\n"
        f"- Expected Subject Initials: {expected_init}, "
        f"but received: {subject_init}.\n"
        f"- Expected Session Number: {expected_session}, "
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
