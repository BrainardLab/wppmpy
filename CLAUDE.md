# wppmpy — Claude Code context

## Repo layout

```
wppmpy/
  src/wppmpy/         # main Python toolbox (public)
  aepsych/            # AEPsych experiment environment (see below)
  notebooks/          # Jupyter notebooks
  .venv/              # root venv: pip install -e ".[notebooks]"
```

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

### Pre-commit hooks (wppmpy root)

ruff, ruff-format, nbstripout, nbqa-mypy, pytest, mypy. All must pass before
committing. Run `ruff check --fix aepsych/` and stage all files (no `AM`/`RM`
in `git status`) before committing to avoid stash conflicts with hook auto-fixes.
