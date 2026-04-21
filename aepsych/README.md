# wppmpy aepsych environment

Python environment for running AEPsych-based color discrimination experiments.
`networkDisk_tests/` contains test scripts (`sender.py`, `recipient.m`) that
exercise the sender/recipient communication protocol without a live AEPsych
server.

---

## 1. Install

From the `aepsych/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs `wppmpy-aepsych` and all dependencies, including `aepsych==0.7.3`
and `ellipsoids-elife2025` (fetched from the `dev` branch of the
`fh862/ellipsoids_eLife2025` GitHub repo).

To activate the environment in future sessions:

```bash
source /path/to/wppmpy/aepsych/.venv/bin/activate
```

---

## 2. Local configuration

Machine-specific paths are stored in `aepsych/local_config.json`, which is
gitignored. Create it by copying the template:

```bash
cp local_config.json.template local_config.json
```

Then edit `local_config.json`:

```json
{
    "ellipsoids_repo_path": "/path/to/ellipsoids_eLife2025/ellipsoids",
    "network_disk_path":    "/path/to/shared/network/disk",
    "stim_at_thres_path":   "/path/to/Stim_at_thres_for_image_generation_subN.pkl",
    "color_thres_base_dir": "/path/to/color/threshold/data",
    "flag_load_rgb":        false
}
```

| Field | Description |
|---|---|
| `ellipsoids_repo_path` | Absolute path to the `ellipsoids/` subdirectory inside your local clone of `ellipsoids_eLife2025` |
| `network_disk_path` | Root of the shared network disk where session files are written |
| `stim_at_thres_path` | Path to the subject-specific `Stim_at_thres_for_image_generation_subN.pkl` file |
| `color_thres_base_dir` | Directory containing color threshold data and calibration files |
| `flag_load_rgb` | `false` to generate random RGB values (testing); `true` to load from the pkl file |

---

## 3. Running the networkDisk test (sender.py + recipient.m)

`sender.py` (Python) and `recipient.m` (MATLAB) communicate by appending lines
to a shared text file on the network disk. Start the recipient before the sender.

### Step 1 — Configure MATLAB preferences

In MATLAB, run once per machine (values persist across sessions):

```matlab
setpref('wppm', 'color_thres_base_dir', '/path/to/color/threshold/data');
setpref('wppm', 'network_disk_path',    '/path/to/shared/network/disk');
```

Both paths must match the corresponding entries in `local_config.json`.

### Step 2 — Start recipient.m

Open `networkDisk_tests/recipient.m` in MATLAB and run it. It will show a
dialog asking for subject ID, initials, and session number, then wait for the
sender to initialize.

### Step 3 — Run sender.py

In a terminal with the aepsych venv active:

```bash
cd /path/to/wppmpy/aepsych
.venv/bin/python networkDisk_tests/sender.py
```

A dialog will appear asking for the same subject ID, initials, and session
number entered in MATLAB. The two scripts must agree on all three values.

### What happens

1. `sender.py` creates a session file on the network disk and writes
   `Set_Up_to_Communicate`.
2. `recipient.m` detects this and responds `Ready_To_Communicate`.
3. `sender.py` sends RGB trial values one at a time (`Image_Display` lines).
4. `recipient.m` reads each line, simulates a response, and writes
   `Image_Confirmed`.
5. After all trials, `sender.py` writes `Done` and both scripts exit.
