# wppmpy_public — Claude Code context

## Repo layout (actual)

```
wppmpy_public/
  toolbox/          # installed Python package (pip install -e .)
    basis_posterior/
  tutorial/         # introductory notebooks
    example_finitebasis_gaussian/
  analysis/         # analysis notebooks and scripts
    upenn/
      hong_etal_2025/
        download_data.py
        ellipses_from_tables.ipynb
        ellipses_from_fits.ipynb
  data/
    hong_etal_2025/ # OSF data downloaded by download_data.py
  tests/            # pytest suite for the toolbox
  aepsych/          # AEPsych experiment environment (separate venv)
  analysis.venv/    # root venv: pip install -e ".[notebooks]"
```

## Two environments

| Environment | Location | Activate |
|---|---|---|
| Analysis (notebooks, toolbox) | `analysis.venv/` | `source analysis.venv/bin/activate` |
| Experiment / simulation | `aepsych/aepsych.venv/` | `source aepsych/aepsych.venv/bin/activate` |

## aepsych/ subdirectory

Self-contained environment for running AEPsych-based color discrimination
experiments. Has its own venv and `pyproject.toml`.

```
aepsych/
  pyproject.toml          # package: wppmpy-aepsych
  README.md               # install + usage instructions
  aepsych_dconfig/        # MachineConfig, ExptConfig, PregenSobolConfig
  matlab/                 # MATLAB communication class
    WPPMCommunicator.m    # mirrors Python CommunicateViaTextFile
  generic/
    networkdisktest/      # sender.py / recipient.py / recipient.m test scripts
      local_config.json   # gitignored — copy from .template and fill in
```

### Install

```bash
cd aepsych
python3 -m venv aepsych.venv
source aepsych.venv/bin/activate
pip install -e .
```

`ellipsoids-elife2025` is fetched automatically from the `dev` branch of
`fh862/ellipsoids_public` on GitHub — no manual clone needed.

### Run scripts

```bash
aepsych.venv/bin/python generic/networkdisktest/sender.py
```

### Key design decisions

- **`aepsych_dconfig` (not `dconfig`):** Our config package must not be named
  `dconfig` — `ellipsoids-elife2025` ships a namespace `dconfig` (no
  `__init__.py`) that would win at import time because setuptools editable
  finders are appended last to `sys.meta_path`.

- **No `ellipsoids_repo_path`:** `analysis.*` modules are importable directly
  from the installed `ellipsoids-elife2025` package; no `sys.path` hacks needed.

- **`aepsych_dconfig/config_pregenSobol.py`** is a cleaned-up copy of
  `PregenSobolConfig` from `ellipsoids_public/ellipsoids/dconfig/`. Kept
  local to avoid the namespace package conflict at import time.

- **`local_config.json`** (gitignored) holds machine-specific paths:
  `network_disk_path`, `stim_at_thres_path`, `flag_load_rgb`.
  (`color_thres_base_dir` was removed — not needed by sender/recipient test scripts.)

## generic/networkdisktest/ — communication test scripts

Tests the sender/recipient protocol without a live AEPsych server.

- **`sender.py`** — Python sender. Prompts for subject ID/initials/session via
  `input()`, creates a session file via `ExperimentFileManager`, then sends 10
  random RGB trial pairs and finalises. Uses `network_disk_path` and
  `flag_load_rgb` from `local_config.json` (`flag_load_rgb = false` for random
  RGB; `stim_at_thres_path` only needed if `true`). Run as:
  `aepsych.venv/bin/python generic/networkdisktest/sender.py`

- **`recipient.py`** — Python recipient stand-in. Prompts via `input()`, then
  polls for the session file by glob pattern (includes session number to avoid
  picking up stale files from earlier sessions). Uses `CommunicateViaTextFile`
  from `ellipsoids-elife2025`.

- **`recipient.m`** — MATLAB recipient stand-in. Uses `WPPMCommunicator` from
  `aepsych/matlab/`. Added to path via `addpath` relative to the script
  location. Requires the `wppm.network_disk_path` MATLAB preference.

### WPPMCommunicator (aepsych/matlab/WPPMCommunicator.m)

MATLAB class mirroring Python's `CommunicateViaTextFile`. Key interface:

```matlab
% Wait for session file to appear (static — call before constructing)
fullPath = WPPMCommunicator.waitForSessionFile(pathSub, pattern);

% Construct and run
comm = WPPMCommunicator(fullPath);         % optional: retryDelay=, timeout=
comm.confirmCommunication();               % handshake
while ~comm.terminate
    comm.confirmRGBvals(responseDelay);    % one trial; sets terminate on Done
end
```

`appendMessage` and `parseTrialLine` are also public for use in experimental scripts.

### Workflow

1. Start `recipient.m` or `recipient.py` — they wait for the session file.
2. Start `sender.py` — creates the session file, performs handshake, sends trials.
3. Recipient responds to each trial with a random 0/1, confirms, and exits on `Done`.

Both scripts look in `network_disk_path/sub{id}/` (no practice subdirectory).
Session file glob includes session number (`*session{N}*`) to avoid matching
stale files from prior sessions.

## Notebooks (analysis/upenn/hong_etal_2025/)

- **`ellipses_from_tables.ipynb`** — reads `Sigmas_thres` directly from CSV;
  no pkl or JAX needed.
- **`ellipses_from_fits.ipynb`** — loads pkl fit parameters, runs
  `model.compute_U(W_est, grid)` → `compute_Sigmas(U)` for noise covariances;
  also uses pre-stored `Sigmas_thres_grid` from the pkl.

### Grid indexing convention (critical)

`Sigmas_thres_grid` and `Sigmas_noise_grid` in the pkl files have
**W-dim-1 (x) varying along the column axis (j)** and W-dim-2 (y) along
the row axis (i). Correct construction:

```python
g_row, g_col = np.meshgrid(pts, pts, indexing="ij")
grid = np.stack([g_col, g_row], axis=-1)  # grid[i,j] = [x=pts[j], y=pts[i]]
```

Using `np.stack([g_row, g_col])` (natural ij order) causes figures to appear
reflected across the positive diagonal.

### pkl loading

The pkl files were saved with older package versions. Three mechanisms handle
compatibility:

1. `_TargetedStubFinder` — stubs `torch` and `jaxlib.xla_extension` via
   `sys.meta_path` (absent/renamed in current env).
2. `_RobustUnpickler(dill.Unpickler)` — overrides `find_class` to return a
   stub type for any `AttributeError`/`ModuleNotFoundError` (handles e.g.
   `jax.errors.SimplifiedTraceback` removed in jax ≥ 0.5).
3. `_TkStub` — stubs `_tkinter`/`tkinter` C extension; needs `__path__=[]`,
   `__call__`, `__enter__`/`__exit__`, `__iter__`, `__bool__`.

## JAX notes

- **Always** configure `jax_enable_x64 = True` — float64 required throughout.
- **Apple Silicon GPU:** not supported. `jax-metal` doesn't support float64.
  CPU-only on Apple Silicon is fine.
- **NVIDIA GPU:** `pip install "jax[cuda12]"` after main install.

## Pre-commit hooks (wppmpy root)

ruff, ruff-format, nbstripout, nbqa-mypy, pytest, mypy. All must pass before
committing. Run `ruff check --fix aepsych/` and stage all files (no `AM`/`RM`
in `git status`) before committing to avoid stash conflicts with hook auto-fixes.
