"""
Created on Tue Jan  7 21:57:13 2025

@author: brainardlab-adm
"""
# Local configuration is read from local_config.json in this directory.
# That file is gitignored. Copy local_config.json.template to local_config.json
# and fill in the paths and flags for your machine before running.

import json
import os
import pickle
import random
import sys

import numpy as np

_config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "local_config.json"
)
if not os.path.exists(_config_path):
    raise FileNotFoundError(
        f"Local config file not found: {_config_path}\n"
        "Copy local_config.json.template to local_config.json and fill in your "
        "machine-specific paths and settings."
    )
with open(_config_path) as _f:
    _config = json.load(_f)

sys.path.append(_config["ellipsoids_repo_path"])
from analysis.utils_communication import (  # noqa: E402
    CommunicateViaTextFile,
    ExperimentFileManager,
    get_comment_after_session,
    get_experiment_info_custom,
)

# %% Prompt the user for experiment information
# Use a custom Tkinter-based popup to collect subject ID, initials, and session number
subject_id, subject_init, session_today = get_experiment_info_custom()

# Define the main shared network disk path where files will be stored
networkDisk_path = _config["network_disk_path"]
# Create the path for the subject's directory
path_sub = os.path.join(networkDisk_path, f"sub{subject_id}")

# Attempt to load the experiment manager state from a pickle file
try:
    # Define the metadata file name and full path
    expt_info = f"sub{subject_id}_{subject_init}_expt_record.pkl"
    path_metadata = os.path.join(path_sub, expt_info)
    # Load the existing state of the experiment file manager
    expt_file_manager = ExperimentFileManager.load_state(path_metadata)
except Exception:
    # If loading fails (e.g., file not found), initialize a new ExperimentFileManager
    expt_file_manager = ExperimentFileManager(
        subject_id, subject_init, networkDisk_path
    )
# Create a new session file for the current session
file_path, file_name = expt_file_manager.create_session_file(session_today)
# List all files created for this subject
expt_file_manager.list_files()

# %% Initialize communication class
communicator = CommunicateViaTextFile(expt_file_manager.path_sub)
communicator.check_and_handle_file(file_name)

# Step 1: Initialize
print("Initializing communication...")
communicator.initialize_communication()
print("Initialization complete.")
# update the communication status
expt_file_manager.status_updates("Confirmed")

# generate random RGB values or load
flag_load_rgb = _config["flag_load_rgb"]
if not flag_load_rgb:
    # Step 2: Send 10 sets of RGB values
    # Generate MOCS and AEPsych trial types
    MOCS_trial_type = [f"MOCS_{i}" for i in range(1, 6)]
    AEPsych_trial_type = [f"AEPsych_{i}" for i in range(1, 6)]
    trial_type_both = MOCS_trial_type + AEPsych_trial_type
    random.shuffle(trial_type_both)
    trial_type_final = [
        f"Trial_{i + 1}_{item}" for i, item in enumerate(trial_type_both)
    ]
    print(trial_type_final)

    ref_rgb_values = np.random.rand(10, 3)  # Generate 10 random RGB values
    comp_rgb_values = np.random.rand(10, 3)
else:
    stim_at_thres_path = _config["stim_at_thres_path"]
    # Load the dictionary from the pickle file
    with open(stim_at_thres_path, "rb") as f:
        stim_at_thres_dict = pickle.load(f)
    ref_rgb_values = stim_at_thres_dict["MOCS_trials_RGB"]["MOCS_xref_shuffled"]
    comp_rgb_values = stim_at_thres_dict["MOCS_trials_RGB"]["MOCS_x1_shuffled"]

    trial_type_final = [f"MOCS_{i}" for i in range(1, ref_rgb_values.shape[0] + 1)]

# run it
for i, (trial, ref_rgb, comp_rgb) in enumerate(
    zip(trial_type_final, ref_rgb_values, comp_rgb_values, strict=False), start=1
):
    print(f"Sending reference and comparison pair {i}...")
    communicator.send_RGBvals(trial, ref_rgb.tolist(), comp_rgb.tolist())
    print(f"RGB values {i} confirmed.")

# Step 3: Finalize
print("Finalizing communication...")
communicator.finalize()
print("Communication finalized.")
# update the communication status
expt_file_manager.status_updates("Done")

# %%add a comment at the end of an experiment
expt_file_manager.add_comments(get_comment_after_session())
