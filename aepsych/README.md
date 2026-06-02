# wppmpy aepsych environment

Python environment for running AEPsych-based color discrimination experiments.
`generic/networkdisktest/` contains `sender.py` / `recipient.py` / `recipient.m` for
testing the sender/recipient communication protocol without a live AEPsych server.

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

Then create the venv (from inside `aepsych/`):

```bash
cd /path/to/wppmpy_public/aepsych
~/.pyenv/versions/3.12.2/bin/python3 -m venv aepsych.venv
source aepsych.venv/bin/activate
pip install -e .
```

### Windows

```bash
python -m venv aepsych.venv
aepsych.venv\Scripts\activate
pip install -e .
```

### Verifying tkinter works

```bash
aepsych.venv/bin/python -c "import tkinter; tkinter._test()"
```

---

## 2. Local configuration

Machine-specific paths live in `local_config.json` files that are gitignored.
`MachineConfig.from_json(Path(__file__))` finds the right file automatically by
walking up from the calling script's directory until it hits `pyproject.toml`
(the package root), checking each directory for `local_config.json`.  This means
each experiment sub-directory can have its own config, or share the one at the
`aepsych/` root.

### Setup

Copy the template and fill in your machine's paths:

```bash
cp generic/networkdisktest/local_config.json.template \
   generic/networkdisktest/local_config.json
# edit local_config.json
```

Current fields used by scripts in this repo:

| Field | Used by | Description |
|---|---|---|
| `network_disk_path` | sender.py, recipient.py | Root of the shared network disk |

Fields that a script doesn't reference are silently ignored, so it is safe to
add experiment-specific fields and share one file across scripts.

### WPPM_CONFIG_DIR — for scripts that live outside the experiment tree

Scripts in `generic/networkdisktest/` walk up from their own `__file__` and find
`aepsych/generic/networkdisktest/local_config.json` automatically when run with this venv.

If you want to run these scripts from a *different* repo's venv (e.g.
`wppmpy_private`) and have them use *that* repo's config, set the env var before
running:

```bash
export WPPM_CONFIG_DIR=/path/to/wppmpy_private/aepsych/wppmopl
source /path/to/wppmpy_private/aepsych/aepsych.venv/bin/activate
python generic/networkdisktest/sender.py
```

`WPPM_CONFIG_DIR` is checked first, before the `__file__`-based walk.  It is the
Python equivalent of MATLAB's `setpref` — set it once per shell session for the
experiment you are working on.

### MATLAB side

`recipient.m` reads `network_disk_path` directly from
`generic/networkdisktest/local_config.json` using `jsondecode` — no
`setpref` required.  The same config file is used by `sender.py` and
`recipient.py`, so only one file needs to be configured per machine.

---

## 3. Running the networkDisk test (sender.py + recipient.m)

Start the recipient before the sender.

**Step 1** — Open `generic/networkdisktest/recipient.m` in MATLAB and run it.

**Step 2** — In a terminal with the aepsych venv active:

```bash
source /path/to/wppmpy_public/aepsych/aepsych.venv/bin/activate
python generic/networkdisktest/sender.py
```

Enter the same subject ID, initials, and session number as the MATLAB dialog.

**What happens:**
1. `sender.py` creates a session file and writes `Set_Up_to_Communicate`.
2. `recipient.m` responds `Ready_To_Communicate`.
3. `sender.py` sends RGB trial values (`Image_Display` lines).
4. `recipient.m` confirms each trial (`Image_Confirmed`).
5. `sender.py` writes `Done`; both scripts exit.
