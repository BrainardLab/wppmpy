# wppmpy — Claude Code context

## Repo layout (actual)

```
wppmpy/
  toolbox/          # installed Python package (pip install -e .)
    basis_posterior/
  src/              # notebooks and scripts
    example_finitebasis_gaussian/
    hong_etal_2025/
      download_data.py
      ellipses_from_tables.ipynb
      ellipses_from_fits.ipynb
  data/
    hong_etal_2025/ # OSF data downloaded by download_data.py
  tests/            # pytest suite for the toolbox
  aepsych/          # AEPsych experiment environment (separate venv)
  .venv/            # root venv: pip install -e ".[notebooks]"
```

## Two environments

| Environment | Location | Install |
|---|---|---|
| Analysis (notebooks, toolbox) | repo root `.venv/` | `pip install -e ".[notebooks]"` |
| Experiment / simulation | `aepsych/.venv/` | `cd aepsych && pip install -e .` |

## aepsych/ subdirectory

Self-contained environment for running AEPsych-based color discrimination
experiments. Has its own venv and `pyproject.toml`.

```
aepsych/
  pyproject.toml          # package: wppmpy-aepsych
  local_config.json       # gitignored — copy from .template and fill in
  local_config.json.template
  README.md               # install + usage instructions
  aepsych_dconfig/        # MachineConfig, ExptConfig, PregenSobolConfig
  expt/                   # placeholder for future experiment scripts
  sim/                    # placeholder for future simulation scripts
  networkDisk_tests/      # sender.py / recipient.py / recipient.m test scripts
```

### Install

```bash
cd aepsych
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`ellipsoids-elife2025` is fetched automatically from the `dev` branch of
`fh862/ellipsoids_eLife2025` on GitHub — no manual clone needed.

### Run scripts

```bash
.venv/bin/python networkDisk_tests/sender.py
```

### Key design decisions

- **`aepsych_dconfig` (not `dconfig`):** Our config package must not be named
  `dconfig` — `ellipsoids-elife2025` ships a namespace `dconfig` (no
  `__init__.py`) that would win at import time because setuptools editable
  finders are appended last to `sys.meta_path`.

- **No `ellipsoids_repo_path`:** `analysis.*` modules are importable directly
  from the installed `ellipsoids-elife2025` package; no `sys.path` hacks needed.

- **`aepsych_dconfig/config_pregenSobol.py`** is a cleaned-up copy of
  `PregenSobolConfig` from `ellipsoids_eLife2025/ellipsoids/dconfig/`. Kept
  local to avoid the namespace package conflict at import time.

- **`local_config.json`** (gitignored) holds machine-specific paths:
  `network_disk_path`, `stim_at_thres_path`, `color_thres_base_dir`,
  `flag_load_rgb`.

## Notebooks (src/hong_etal_2025/)

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
