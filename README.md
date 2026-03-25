# wppmpy

Python code accompanying the **Wishart Process Psychophysical Model (WPPM)** — a Bayesian semi-parametric model that characterises how internal perceptual noise varies continuously across color space.

Background on the model and related experimental data are described in:

> **Hong et al.** *Comprehensive characterization of human color discrimination using a Wishart process psychophysical model.*
> eLife Reviewed Preprint (2025). https://elifesciences.org/reviewed-preprints/108943v1

This repository is being set up to provide:
- **Illustrative examples** of the statistical ideas underlying the WPPM
- **Additional analysis code** beyond that provided in the repository accompanying the paper
- **Tools** to help readers understand and use our results

These will be developed and added over time.

---

## Installation

### Basic install (toolbox only — no JAX required)

```bash
git clone https://github.com/BrainardLab/wppmpy.git
cd wppmpy
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install -e .
```

### With Hong et al. (2025) notebooks

The notebooks require the paper repository and its dependencies (JAX, pandas, scipy, etc.).  Run this instead of (or in place of) `pip install -e .` above:

```bash
pip install -e ".[notebooks]"
```

JAX (CPU build) is pulled in automatically.  For GPU acceleration see the note below.

Then download the required data subset from OSF once (saved to `data/hong_etal_2025/`):

```bash
python src/hong_etal_2025/download_data.py
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
does not support.  CPU-only performance on Apple Silicon is still very good.

### Future sessions

Each time you open a new terminal, activate the environment before running code:

```bash
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### Using a local clone of ellipsoids_eLife2025

If you have a local clone and want to use it instead of the GitHub copy:

```bash
pip install -e /path/to/ellipsoids_eLife2025/ellipsoids
```

---

## Examples

### Bayesian inference with a finite sinusoidal basis

A key idea in the WPPM is that a smoothness prior is used to leverage data collected across the stimulus space.  This notebook illustrates the idea of using such a prior together with Bayesian inference for a simple example. A finite sinusoidal basis is used to represent functions on $[0, 2\pi)$, an exponentially-decaying (with spatial frequency) Gaussian prior over weights on the basis function encodes a preference for smooth functions, and the closed-form Gaussian posterior is computed from a small number of noisy measurements.  The example also illustrates the improvement you can get if you use a simple heuristic to drive adaptive measurement sampling.

| | |
|---|---|
| **Run interactively in your browser** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BrainardLab/wppmpy/blob/main/src/example_finitebasis_gaussian/example_finitebasis_gaussian.ipynb) |
| **View as a static page** | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/BrainardLab/wppmpy/blob/main/src/example_finitebasis_gaussian/example_finitebasis_gaussian.ipynb) |

### Threshold ellipses from pre-computed tables (Hong et al. 2025, Figure 2C)

The WPPM was fit to color discrimination data from eight participants and used to read out threshold ellipses on a 7 × 7 grid of reference stimuli in the isoluminant plane of a 2-D model colour space.  This notebook reproduces Figure 2C of Hong et al. (2025) by reading those pre-computed covariance matrices directly from the paper's OSF dataset — no model fitting or JAX computation required.  It also shows how to construct the 95 % bootstrap confidence regions reported in the paper: for each of 120 bootstrap model fits, it ranks datasets by their summed Normalized Bures Similarity to the original fit, retains the top 95 % (114/120), and plots the resulting inner/outer radial envelopes as a coloured band around each black ellipse.

**Data:** download the required OSF data subset once after installation by running `python src/hong_etal_2025/download_data.py`.

| | |
|---|---|
| **Run interactively in your browser** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BrainardLab/wppmpy/blob/main/src/hong_etal_2025/ellipses_from_tables.ipynb) |
| **View as a static page** | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/BrainardLab/wppmpy/blob/main/src/hong_etal_2025/ellipses_from_tables.ipynb) |

---

### More Hong et al. (2025) notebook examples

Additional notebooks — including ones best run locally from the cloned repository — are listed on the [**Hong et al. (2025) notebooks wiki page**](https://github.com/BrainardLab/wppmpy/wiki/Hong-et-al-2025-Notebooks), along with static previews of each.

---

## Repository layout

```
toolbox/                          # reusable Python modules
  basis_posterior/                # Bayesian posterior for finite basis models

src/
  example_finitebasis_gaussian/   # introductory Bayesian inference notebook
    example_finitebasis_gaussian.ipynb
  hong_etal_2025/                 # notebooks and data download for Hong et al. (2025)
    download_data.py              # fetch required OSF data subset
    ellipses_from_tables.ipynb    # reproduce Figure 2C from pre-computed CSV tables
    ellipses_from_fits.ipynb      # reproduce Figure 2C from pkl fit parameters
```

---

## License

See [LICENSE](LICENSE).
