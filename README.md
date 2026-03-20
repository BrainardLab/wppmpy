# wppmpy

Python code accompanying the **Wishart Process Psychophysical Model (WPPM)** — a Bayesian semi-parametric model that characterises how internal perceptual noise varies continuously across color space.

Background on the model and related experimental data iare described in:

> **Hong et al.** *Comprehensive characterization of human color discrimination using a Wishart process psychophysical model.*
> eLife Reviewed Preprint (2025). https://elifesciences.org/reviewed-preprints/108943v1

This repository is being set up to provide:
- **Illustrative examples** of the statistical ideas underlying the WPPM
- **Additional analysis code** beyond that provided in the repository accompanying the paper
- **Tools** to help readers understand and use our results
These will be developed and added over time.

---

## Examples

### Bayesian inference with a finite sinusoidal basis

A key idea in the WPPM is that a smoothness prior is used to leverage data collected across the stimulus space.  This notebook illustrates the idea of using such a prior together with Bayesian inference for a simple example. A finite  inusoidal basis is used to represent functions on $[0, 2\pi)$, an exponentially-decaying (with spatial frequency) Gaussian prior over weights on the basis function encodes a preference for smooth functions, and the closed-form Gaussian posterior is computed from a small number of noisy measurements.

| | |
|---|---|
| **Run interactively in your browser** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BrainardLab/wppmpy/blob/main/src/example_finitebasis_gaussian/example_finitebasis_gaussian.ipynb) |
| **View as a static page** | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/BrainardLab/wppmpy/blob/main/src/example_finitebasis_gaussian/example_finitebasis_gaussian.ipynb) |

---

## Installation

```bash
git clone https://github.com/BrainardLab/wppmpy.git
cd wppmpy
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

---

## Repository layout

```
src/
  example_finitebasis_gaussian/   # introductory Bayesian inference notebook
```

---

## License

See [LICENSE](LICENSE).
