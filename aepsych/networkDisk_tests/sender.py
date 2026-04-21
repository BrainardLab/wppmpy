"""
Created on Tue Jan  7 21:57:13 2025

@author: brainardlab-adm
"""
# Machine-specific paths are read from local_config.json at the aepsych/ root.
# That file is gitignored. Copy local_config.json.template to local_config.json
# and fill in paths for your machine before running.

import os
import pickle
import random

import numpy as np
from aepsych_dconfig import ExptConfig, MachineConfig

machine = MachineConfig.from_json()
expt = ExptConfig.isoluminant_4d()

from analysis.utils_communication import (  # noqa: E402
    CommunicateViaTextFile,
    ExperimentFileManager,
    get_comment_after_session,
    get_experiment_info_custom,
)

# %% Prompt the user for experiment information
subject_id, subject_init, session_today = get_experiment_info_custom()

# Define the main shared network disk path where files will be stored
networkDisk_path = machine.network_disk_path
path_sub = os.path.join(networkDisk_path, f"sub{subject_id}")

# Attempt to load the experiment manager state from a pickle file
try:
    expt_info = f"sub{subject_id}_{subject_init}_expt_record.pkl"
    path_metadata = os.path.join(path_sub, expt_info)
    expt_file_manager = ExperimentFileManager.load_state(path_metadata)
except Exception:
    expt_file_manager = ExperimentFileManager(
        subject_id, subject_init, networkDisk_path
    )
file_path, file_name = expt_file_manager.create_session_file(session_today)
expt_file_manager.list_files()

# %% Initialize communication class
communicator = CommunicateViaTextFile(expt_file_manager.path_sub)
communicator.check_and_handle_file(file_name)

print("Initializing communication...")
communicator.initialize_communication()
print("Initialization complete.")
expt_file_manager.status_updates("Confirmed")

# %% Generate or load RGB values
if not machine.flag_load_rgb:
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
    with open(machine.stim_at_thres_path, "rb") as f:
        stim_at_thres_dict = pickle.load(f)
    ref_rgb_values = stim_at_thres_dict["MOCS_trials_RGB"]["MOCS_xref_shuffled"]
    comp_rgb_values = stim_at_thres_dict["MOCS_trials_RGB"]["MOCS_x1_shuffled"]

    trial_type_final = [f"MOCS_{i}" for i in range(1, ref_rgb_values.shape[0] + 1)]

# %% Send RGB values
for i, (trial, ref_rgb, comp_rgb) in enumerate(
    zip(trial_type_final, ref_rgb_values, comp_rgb_values, strict=False), start=1
):
    print(f"Sending reference and comparison pair {i}...")
    communicator.send_RGBvals(trial, ref_rgb.tolist(), comp_rgb.tolist())
    print(f"RGB values {i} confirmed.")

print("Finalizing communication...")
communicator.finalize()
print("Communication finalized.")
expt_file_manager.status_updates("Done")

expt_file_manager.add_comments(get_comment_after_session())
