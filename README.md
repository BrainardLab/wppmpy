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

```bash
git clone https://github.com/BrainardLab/wppmpy.git
cd wppmpy
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install -e .
```

To also run the Hong et al. (2025) notebooks, install the notebook dependencies (this pulls in the paper repository and all its requirements, including JAX, automatically):

```bash
pip install -e ".[notebooks]"
```

Then download the required data subset from OSF once (saved to `data/hong_etal_2025/`):

```bash
python src/hong_etal_2025/download_data.py
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

> **More Hong et al. (2025) notebook examples** — including notebooks that are best run locally from the cloned repository — are listed on the [**Hong et al. (2025) notebooks wiki page**](https://github.com/BrainardLab/wppmpy/wiki/Hong-et-al-2025-Notebooks), along with static previews of each.

---

## Repository layout

```
src/
  example_finitebasis_gaussian/   # introductory Bayesian inference notebook
  hong_etal_2025/                 # notebooks and data download for Hong et al. (2025)
    download_data.py              # fetch required OSF data subset
    ellipses_from_tables.ipynb    # reproduce Figure 2C from pre-computed CSV tables
    ellipses_from_fits.ipynb      # reproduce Figure 2C from pkl fit parameters
```

---

## License

See [LICENSE](LICENSE).
