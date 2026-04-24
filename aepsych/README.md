# wppmpy aepsych environment

Python environment for running AEPsych-based color discrimination experiments.
`networkDisk_tests/` contains test scripts (`sender.py`, `recipient.m`) that
exercise the sender/recipient communication protocol without a live AEPsych
server.

---

## 1. Install

### macOS — Python via pyenv (required for tkinter support)

The experiment environment uses tkinter for GUI dialogs. On macOS, pyenv-compiled
Python lacks `_tkinter` by default, and Homebrew's tcl-tk 9.x is incompatible
with Python 3.12. Install `tcl-tk@8` first, then recompile Python:

```bash
brew install tcl-tk@8

pyenv uninstall 3.12.2   # skip if not yet installed
LDFLAGS="-L$(brew --prefix tcl-tk@8)/lib" \
CPPFLAGS="-I$(brew --prefix tcl-tk@8)/include" \
PKG_CONFIG_PATH="$(brew --prefix tcl-tk@8)/lib/pkgconfig" \
pyenv install 3.12.2
```

Then create the venv using the full pyenv path (pyenv shims may not be active):

```bash
cd /path/to/wppmpy/aepsych
/Users/<you>/.pyenv/versions/3.12.2/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Windows

Windows ships with Tcl/Tk support built into the standard Python installer —
no extra steps needed. Use the standard Python 3.12 installer from python.org,
then:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### Verifying tkinter works

```bash
.venv/bin/python -c "import tkinter; tkinter._test()"
```

A small test window should appear.

---

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
    "network_disk_path": "/path/to/shared/network/disk"
}
```

| Field | Description |
|---|---|
| `network_disk_path` | Root of the shared network disk where session files are written |

---

## 3. Running the networkDisk test (sender.py + recipient.m)

`sender.py` (Python) and `recipient.m` (MATLAB) communicate by appending lines
to a shared text file on the network disk. Start the recipient before the sender.

### Step 1 — Configure MATLAB preferences

In MATLAB, run once per machine (value persists across sessions):

```matlab
setpref('wppm', 'network_disk_path', '/path/to/shared/network/disk');
```

This must match `network_disk_path` in `local_config.json`.

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

Enter the same subject ID, initials, and session number at the terminal
prompts (unlike `recipient.m`, which uses a dialog). The two scripts must
agree on all three values.

### What happens

1. `sender.py` creates a session file on the network disk and writes
   `Set_Up_to_Communicate`.
2. `recipient.m` detects this and responds `Ready_To_Communicate`.
3. `sender.py` sends RGB trial values one at a time (`Image_Display` lines).
4. `recipient.m` reads each line, simulates a response, and writes
   `Image_Confirmed`.
5. After all trials, `sender.py` writes `Done` and both scripts exit.
