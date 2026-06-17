# wppmpy

Python code accompanying the **Wishart Process Psychophysical Model (WPPM)** — a Bayesian semi-parametric model that characterises how internal perceptual noise varies continuously across color space.  This builds upon the public respository for the paper, providing additonal code.  See installation instructions below.

Background on the model and related experimental data are described in:

> **Hong et al.** *Comprehensive characterization of human color discrimination using a Wishart process psychophysical model.*
> eLife Reviewed Preprint (2025). https://elifesciences.org/reviewed-preprints/108943v1

This repository is being set up to provide:
- **Illustrative examples** of the statistical ideas underlying the WPPM
- **Additional analysis code** beyond that provided in the repository accompanying the paper
- **Tools** to help readers understand and use our results

These will be developed and added over time.

---

## Two environments

This repository contains two independent Python environments:

| Environment | Purpose | Location |
|---|---|---|
| **Analysis** | Notebooks, toolbox, data exploration | repo root (`analysis.venv/`) |
| **Experiment / simulation** | AEPsych-based color discrimination experiments | `aepsych/` (`aepsych/aepsych.venv/`) |

Most users will only need the analysis environment.  The experiment environment
is for running or simulating psychophysical experiments using the AEPsych
adaptive sampling framework.

---

## Installation — analysis environment

### Basic install (toolbox only — no JAX required)

```bash
git clone https://github.com/BrainardLab/wppmpy_public.git
cd wppmpy_public
python -m venv analysis.venv
source analysis.venv/bin/activate      # macOS / Linux
# analysis.venv\Scripts\activate       # Windows
pip install -e .
```

### With notebooks

The notebooks in this repo require the paper repository and its dependencies (JAX, pandas, scipy, etc.).  Run this instead of (or in place of) `pip install -e .` above:

```bash
pip install -e ".[notebooks]"
```

JAX (CPU build) is pulled in automatically.  For GPU acceleration see the note below.

Then download the required data subset from OSF once (destination set by `data_dir`
in `analysis/upenn/hong_etal_2025/local_config.json` — defaults to `local/data/upenn/hong_etal_2025/`):

```bash
python analysis/upenn/hong_etal_2025/download_data.py
```

### GPU acceleration (optional)

**NVIDIA GPU (CUDA 12):**
```bash
pip install "jax[cuda12]"
```
Run this after `pip install -e ".[notebooks]"` to replace the CPU JAX build.

**Apple Silicon (M1/M2/M3/M4):** GPU acceleration is not available for this
codebase.  The code requires 64-bit floating point
(`jax_enable_x64 = True`), which the Apple Metal JAX plugin (`jax-metal`)
does not support.  CPU-only performance on Apple Silicon is still good.

### Future sessions

Each time you open a new terminal, activate the environment before running code:

```bash
source analysis.venv/bin/activate      # macOS / Linux
# analysis.venv\Scripts\activate       # Windows
```

### Using a local clone of ellipsoids_public

If you have a local clone and want to use it instead of the GitHub copy:

```bash
pip install -e /path/to/ellipsoids_public/ellipsoids
```

---

## Installation — experiment / simulation environment

See **[aepsych/README.md](aepsych/README.md)** for full instructions.

In brief:

```bash
cd aepsych
python3 -m venv aepsych.venv
source aepsych.venv/bin/activate
pip install -e .
```

This installs the `wppmpy-aepsych` package, including `aepsych==0.7.3` and the
`ellipsoids-elife2025` paper code (fetched automatically from GitHub).
Machine-specific paths (network disk, stimulus files) are configured via
`aepsych/generic/networkdisktest/local_config.json` (gitignored; copy from `.template`).

---

## Examples

### Bayesian inference with a finite sinusoidal basis

A key idea in the WPPM is that a smoothness prior is used to leverage data collected across the stimulus space.  This notebook illustrates the idea of using such a prior together with Bayesian inference for a simple example. A finite sinusoidal basis is used to represent functions on $[0, 2\pi)$, an exponentially-decaying (with spatial frequency) Gaussian prior over weights on the basis function encodes a preference for smooth functions, and the closed-form Gaussian posterior is computed from a small number of noisy measurements.  The example also illustrates the improvement you can get if you use a simple heuristic to drive adaptive measurement sampling.

| | |
|---|---|
| **Run interactively in your browser** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BrainardLab/wppmpy_public/blob/main/tutorial/example_finitebasis_gaussian/example_finitebasis_gaussian.ipynb) |
| **View as a static page** | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/BrainardLab/wppmpy_public/blob/main/tutorial/example_finitebasis_gaussian/example_finitebasis_gaussian.ipynb) |

### Threshold ellipses from pre-computed tables (Hong et al. 2025, Figure 2C)

The WPPM was fit to color discrimination data from eight participants and used to read out threshold ellipses on a 7 × 7 grid of reference stimuli in the isoluminant plane of a 2-D model colour space.  This notebook reproduces Figure 2C of Hong et al. (2025) by reading those pre-computed covariance matrices directly from the paper's OSF dataset — no model fitting or JAX computation required.  It also shows how to construct the 95 % bootstrap confidence regions reported in the paper: for each of 120 bootstrap model fits, it ranks datasets by their summed Normalized Bures Similarity to the original fit, retains the top 95 % (114/120), and plots the resulting inner/outer radial envelopes as a coloured band around each black ellipse.

**Data:** download the required OSF data subset once after installation by running `python analysis/upenn/hong_etal_2025/download_data.py`.

Run locally: `analysis/upenn/hong_etal_2025/ellipses_from_tables.ipynb`

---

## Repository layout

```
toolbox/                          # reusable Python modules (analysis environment)
  basis_posterior/                # Bayesian posterior for finite basis models

tutorial/
  example_finitebasis_gaussian/   # introductory Bayesian inference notebook
    example_finitebasis_gaussian.ipynb

analysis/
  upenn/
    hong_etal_2025/               # notebooks and data download for Hong et al. (2025)
      download_data.py            # fetch required OSF data subset
      ellipses_from_tables.ipynb  # reproduce Figure 2C from pre-computed CSV tables
      ellipses_from_fits.ipynb    # reproduce Figure 2C from pkl fit parameters

local/                            # machine-local data and outputs (git-ignored)
  data/upenn/hong_etal_2025/      # OSF data (default location from local_config.json)
  analysis/upenn/hong_etal_2025/  # executed notebooks and figures

tests/                            # pytest test suite for the toolbox

aepsych/                          # experiment / simulation environment (separate venv)
  README.md                       # full install and usage instructions
  pyproject.toml                  # installs wppmpy-aepsych package
  aepsych_dconfig/                # config package: MachineConfig, ExptConfig, AepsychIniConfig
  generic/
    networkdisktest/              # sender/recipient test scripts (Python + MATLAB)
      local_config.json           # machine paths (git-ignored; copy from .template)
  matlab/                         # MATLAB communication class (WPPMCommunicator.m)
```

---

## AEPsych provenance and local_config.json

`local_config.json` files carry an `aepsych_config_file` field:

- **Non-empty** (e.g. `"single_3d_colorDiscrimination_EAVC_4strats.ini"`) — the
  AEPsych experiment was run from scripts in this repo; the named `.ini` lives in
  `aepsych_config/` and is the single source of truth for search bounds and EAVC
  criterion.  `AepsychIniConfig.from_expt_dir()` reads it at runtime.

- **Empty string `""`** — data collection and WPPM fitting happened outside this
  repo (in the `ellipsoids` / `ellipsoids_public` repos).  These experiments enter
  wppmpy at the WPPM-fit stage; there is no AEPsych runner or ini file for them here.
  Code that reads `aepsych_config_file` should treat `""` as "no ini available."

For `hong_etal_2025` the field is `""` — that experiment's AEPsych session was
managed externally and is not reproduced in this repo.

---

## License

See [LICENSE](LICENSE).
