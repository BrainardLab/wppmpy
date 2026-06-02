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
from pathlib import Path  # noqa: E402

from aepsych_dconfig import MachineConfig  # noqa: E402

machine = MachineConfig.from_json(Path(__file__))

import glob  # noqa: E402

from analysis.utils_communication import (  # noqa: E402
    CommunicateViaTextFile,
)

subject_id = int(input("Subject ID: "))
subject_init = input("Subject initials: ").strip()
session_today = int(input("Session number today: "))

# Wait for sender to create the session file
path_sub = os.path.join(machine.network_disk_path, f"sub{subject_id}")
pattern = os.path.join(
    path_sub, f"sub{subject_id}_{subject_init}*session{session_today}*.txt"
)

wait_timeout = 120
print(f"Waiting for session file matching {pattern} ...")
t_start = time.time()
while True:
    matches = glob.glob(pattern)
    if matches:
        break
    if time.time() - t_start > wait_timeout:
        raise TimeoutError(f"Timed out waiting for session file matching {pattern}")
    time.sleep(0.5)

file_name = os.path.basename(max(matches, key=os.path.getmtime))
print(f"Found session file: {file_name}")

# Initialize communication class
communicator = CommunicateViaTextFile(
    path_sub,
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
