"""Sanity checks for the finite sinusoidal basis and Gaussian posterior.

These tests verify mathematical properties that must hold regardless of the
specific parameter values chosen — useful regression guards as the codebase grows.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

# ── Typed fixture containers ───────────────────────────────────────────────────


@dataclass
class Basis:
    N: int
    n_freqs: int
    M: int
    x: npt.NDArray[np.float64]
    B: npt.NDArray[np.float64]


@dataclass
class Prior:
    sigma: npt.NDArray[np.float64]
    Lambda_inv: npt.NDArray[np.float64]


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def basis() -> Basis:
    N = 11
    n_freqs = (N - 1) // 2
    M = 256
    x = np.linspace(0, 2 * np.pi, M, endpoint=False)

    B = np.zeros((M, N))
    B[:, 0] = 1.0
    for k in range(1, n_freqs + 1):
        B[:, 2 * k - 1] = np.cos(k * x)
        B[:, 2 * k] = np.sin(k * x)

    return Basis(N=N, n_freqs=n_freqs, M=M, x=x, B=B)


@pytest.fixture
def prior(basis: Basis) -> Prior:
    gamma, epsilon = 2.0, 0.5
    orders = (np.arange(basis.N) + 1) // 2
    sigma: npt.NDArray[np.float64] = gamma * np.exp(-epsilon * orders)
    return Prior(sigma=sigma, Lambda_inv=np.diag(1.0 / sigma**2))


# ── Basis matrix tests ─────────────────────────────────────────────────────────


def test_basis_shape(basis: Basis) -> None:
    assert basis.B.shape == (basis.M, basis.N)


def test_dc_column_is_constant(basis: Basis) -> None:
    """First column must be identically 1 (the DC component)."""
    assert np.allclose(basis.B[:, 0], 1.0)


def test_cos_sin_unit_amplitude(basis: Basis) -> None:
    """Each cos/sin column should have amplitude exactly 1."""
    for k in range(1, basis.n_freqs + 1):
        assert np.allclose(np.max(np.abs(basis.B[:, 2 * k - 1])), 1.0, atol=1e-10)
        assert np.allclose(np.max(np.abs(basis.B[:, 2 * k])), 1.0, atol=1e-10)


def test_prior_variance_uniform(basis: Basis, prior: Prior) -> None:
    """Prior predictive variance must be constant across x (symmetry of the basis)."""
    var: npt.NDArray[np.float64] = (basis.B**2) @ (prior.sigma**2)
    assert np.allclose(var, var[0], rtol=1e-10)


# ── Posterior tests ────────────────────────────────────────────────────────────


def test_posterior_covariance_positive_definite(basis: Basis, prior: Prior) -> None:
    """Posterior covariance must be symmetric positive definite."""
    noise_std = 0.2
    rng = np.random.default_rng(0)
    idx = rng.choice(basis.M, size=6, replace=False)
    Phi = basis.B[idx, :]

    Sigma_post = np.linalg.inv(prior.Lambda_inv + Phi.T @ Phi / noise_std**2)

    assert np.allclose(Sigma_post, Sigma_post.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(Sigma_post)
    assert np.all(eigvals > 0)


def test_posterior_tighter_than_prior(basis: Basis, prior: Prior) -> None:
    """Posterior std must be <= prior std everywhere after observing data."""
    noise_std = 0.2
    rng = np.random.default_rng(1)
    idx = rng.choice(basis.M, size=10, replace=False)
    Phi = basis.B[idx, :]

    Sigma_post = np.linalg.inv(prior.Lambda_inv + Phi.T @ Phi / noise_std**2)
    BSig = basis.B @ Sigma_post
    std_post: npt.NDArray[np.float64] = np.sqrt(
        np.maximum(np.sum(BSig * basis.B, axis=1), 0.0)
    )

    # Prior std is uniform (proven by test_prior_variance_uniform)
    std_prior = float(np.sqrt(float(np.sum(prior.sigma**2 * basis.B[0, :] ** 2))))
    assert np.all(std_post <= std_prior + 1e-10)


def test_posterior_mean_interpolates_noiseless(basis: Basis, prior: Prior) -> None:
    """With near-zero noise the posterior mean should pass through the observations."""
    noise_std = 1e-6
    rng = np.random.default_rng(2)
    w_true: npt.NDArray[np.float64] = rng.normal(0.0, prior.sigma)
    f_true = basis.B @ w_true

    idx = rng.choice(basis.M, size=20, replace=False)
    Phi = basis.B[idx, :]
    y = f_true[idx]

    Sigma_post = np.linalg.inv(prior.Lambda_inv + Phi.T @ Phi / noise_std**2)
    mu_post = Sigma_post @ (Phi.T @ y / noise_std**2)
    f_est = basis.B @ mu_post

    assert np.allclose(f_est[idx], y, atol=1e-3)
