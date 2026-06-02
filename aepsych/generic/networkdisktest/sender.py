import datetime
import os
import random
from pathlib import Path

import numpy as np
from aepsych_dconfig import MachineConfig

machine = MachineConfig.from_json(Path(__file__))

from analysis.utils_communication import (  # noqa: E402
    CommunicateViaTextFile,
)

# %% Prompt the user for experiment information
subject_id = int(input("Subject ID: "))
subject_init = input("Subject initials: ").strip()
session_today = int(input("Session number today: "))

# Create session directory and file
path_sub = os.path.join(machine.network_disk_path, f"sub{subject_id}")
os.makedirs(path_sub, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"sub{subject_id}_{subject_init}_session{session_today}_{timestamp}.txt"
open(os.path.join(path_sub, file_name), "w").close()

# %% Initialize communication class
communicator = CommunicateViaTextFile(path_sub)
communicator.check_and_handle_file(file_name)

print("Initializing communication...")
communicator.initialize_communication()
print("Initialization complete.")

# %% Generate random RGB values for 10 trials
MOCS_trial_type = [f"MOCS_{i}" for i in range(1, 6)]
AEPsych_trial_type = [f"AEPsych_{i}" for i in range(1, 6)]
trial_type_both = MOCS_trial_type + AEPsych_trial_type
random.shuffle(trial_type_both)
trial_type_final = [f"Trial_{i + 1}_{item}" for i, item in enumerate(trial_type_both)]

ref_rgb_values = np.random.rand(10, 3)
comp_rgb_values = np.random.rand(10, 3)

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
