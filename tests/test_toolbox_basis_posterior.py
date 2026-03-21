"""Tests for toolbox.basis_posterior.basis_posterior functions.

Each test targets a specific contract or mathematical invariant:

  build_basis_matrix   – shape, domain, DC=1, unit amplitude, rejects even N
  make_prior           – shape, DC=gamma, Lambda_inv diagonal, cos/sin pair symmetry
  draw_from_prior      – shape, single sample, reproducibility, marginal statistics
  compute_posterior    – shape, Σ_post symmetric PD, posterior ≤ prior,
                         noiseless interpolation
  posterior_predictive_std – shape, non-negative, matches direct diag(B Σ Bᵀ) formula
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from toolbox.basis_posterior.basis_posterior import (
    build_basis_matrix,
    compute_posterior,
    draw_from_prior,
    make_prior,
    posterior_predictive_std,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────


@dataclass
class Setup:
    N: int
    M: int
    gamma: float
    epsilon: float
    x: npt.NDArray[np.float64]
    B: npt.NDArray[np.float64]
    sigma: npt.NDArray[np.float64]
    Lambda_inv: npt.NDArray[np.float64]


@pytest.fixture
def std_setup() -> Setup:
    N, M, gamma, epsilon = 11, 256, 2.0, 0.5
    x, B = build_basis_matrix(N=N, M=M)
    sigma, Lambda_inv = make_prior(N=N, gamma=gamma, epsilon=epsilon)
    return Setup(
        N=N,
        M=M,
        gamma=gamma,
        epsilon=epsilon,
        x=x,
        B=B,
        sigma=sigma,
        Lambda_inv=Lambda_inv,
    )


# ── build_basis_matrix ─────────────────────────────────────────────────────────


def test_build_basis_matrix_output_shapes(std_setup: Setup) -> None:
    assert std_setup.x.shape == (std_setup.M,)
    assert std_setup.B.shape == (std_setup.M, std_setup.N)


def test_build_basis_matrix_domain_endpoints(std_setup: Setup) -> None:
    assert std_setup.x[0] == pytest.approx(0.0)
    assert std_setup.x[-1] < 2 * np.pi
    # Spacing should be uniform
    gaps = np.diff(std_setup.x)
    assert np.allclose(gaps, gaps[0])


def test_build_basis_matrix_dc_column(std_setup: Setup) -> None:
    assert np.allclose(std_setup.B[:, 0], 1.0)


def test_build_basis_matrix_cos_sin_unit_amplitude(std_setup: Setup) -> None:
    n_freqs = (std_setup.N - 1) // 2
    for k in range(1, n_freqs + 1):
        assert np.allclose(np.max(np.abs(std_setup.B[:, 2 * k - 1])), 1.0, atol=1e-10)
        assert np.allclose(np.max(np.abs(std_setup.B[:, 2 * k])), 1.0, atol=1e-10)


def test_build_basis_matrix_rejects_even_n() -> None:
    with pytest.raises(ValueError):
        build_basis_matrix(N=10, M=256)


# ── make_prior ─────────────────────────────────────────────────────────────────


def test_make_prior_output_shapes(std_setup: Setup) -> None:
    assert std_setup.sigma.shape == (std_setup.N,)
    assert std_setup.Lambda_inv.shape == (std_setup.N, std_setup.N)


def test_make_prior_dc_term_equals_gamma(std_setup: Setup) -> None:
    """DC term has sinusoidal order 0, so sigma[0] = gamma * exp(0) = gamma."""
    assert std_setup.sigma[0] == pytest.approx(std_setup.gamma)


def test_make_prior_lambda_inv_is_diagonal(std_setup: Setup) -> None:
    off_diag = std_setup.Lambda_inv - np.diag(np.diag(std_setup.Lambda_inv))
    assert np.allclose(off_diag, 0.0)


def test_make_prior_lambda_inv_diagonal_values(std_setup: Setup) -> None:
    assert np.allclose(np.diag(std_setup.Lambda_inv), 1.0 / std_setup.sigma**2)


def test_make_prior_cos_sin_pairs_share_sigma(std_setup: Setup) -> None:
    """cos(kx) and sin(kx) have the same sinusoidal order, so same prior std."""
    n_freqs = (std_setup.N - 1) // 2
    for k in range(1, n_freqs + 1):
        assert std_setup.sigma[2 * k - 1] == pytest.approx(std_setup.sigma[2 * k])


def test_make_prior_sigma_decreases_with_frequency(std_setup: Setup) -> None:
    """Higher frequency components should have smaller prior std (epsilon > 0)."""
    n_freqs = (std_setup.N - 1) // 2
    for k in range(1, n_freqs):
        assert std_setup.sigma[2 * k - 1] > std_setup.sigma[2 * k + 1]


# ── draw_from_prior ────────────────────────────────────────────────────────────


def test_draw_from_prior_shape(std_setup: Setup) -> None:
    W = draw_from_prior(std_setup.sigma, np.random.default_rng(0), n_samples=5)
    assert W.shape == (std_setup.N, 5)


def test_draw_from_prior_single_sample(std_setup: Setup) -> None:
    W = draw_from_prior(std_setup.sigma, np.random.default_rng(0), n_samples=1)
    assert W.shape == (std_setup.N, 1)


def test_draw_from_prior_reproducible(std_setup: Setup) -> None:
    W1 = draw_from_prior(std_setup.sigma, np.random.default_rng(42), n_samples=20)
    W2 = draw_from_prior(std_setup.sigma, np.random.default_rng(42), n_samples=20)
    assert np.allclose(W1, W2)


def test_draw_from_prior_marginal_statistics(std_setup: Setup) -> None:
    """With many samples the per-weight mean should be ≈0 and std ≈ sigma."""
    W = draw_from_prior(std_setup.sigma, np.random.default_rng(0), n_samples=20_000)
    assert np.allclose(W.mean(axis=1), 0.0, atol=0.05)
    assert np.allclose(W.std(axis=1), std_setup.sigma, rtol=0.03)


# ── compute_posterior ──────────────────────────────────────────────────────────


def test_compute_posterior_output_shapes(std_setup: Setup) -> None:
    rng = np.random.default_rng(0)
    x_meas = rng.uniform(0, 2 * np.pi, size=6)
    y_meas = rng.normal(0, 1, size=6)
    Sigma_post, mu_post = compute_posterior(
        std_setup.B, std_setup.Lambda_inv, x_meas, y_meas, noise_std=0.2
    )
    assert Sigma_post.shape == (std_setup.N, std_setup.N)
    assert mu_post.shape == (std_setup.N,)


def test_compute_posterior_sigma_symmetric_positive_definite(std_setup: Setup) -> None:
    rng = np.random.default_rng(1)
    x_meas = rng.uniform(0, 2 * np.pi, size=8)
    y_meas = rng.normal(0, 1, size=8)
    Sigma_post, _ = compute_posterior(
        std_setup.B, std_setup.Lambda_inv, x_meas, y_meas, noise_std=0.2
    )
    assert np.allclose(Sigma_post, Sigma_post.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(Sigma_post) > 0)


def test_compute_posterior_tighter_than_prior(std_setup: Setup) -> None:
    """Posterior predictive std must be ≤ prior std everywhere after observing data."""
    rng = np.random.default_rng(2)
    x_meas = rng.uniform(0, 2 * np.pi, size=10)
    y_meas = rng.normal(0, 1, size=10)
    Sigma_post, _ = compute_posterior(
        std_setup.B, std_setup.Lambda_inv, x_meas, y_meas, noise_std=0.2
    )
    std_post = posterior_predictive_std(std_setup.B, Sigma_post)
    # Prior predictive std is uniform (tested separately in test_basis_posterior.py)
    std_prior = float(np.sqrt(np.sum(std_setup.sigma**2 * std_setup.B[0, :] ** 2)))
    assert np.all(std_post <= std_prior + 1e-10)


def test_compute_posterior_noiseless_interpolation(std_setup: Setup) -> None:
    """With near-zero noise the posterior mean should pass through the observations."""
    rng = np.random.default_rng(3)
    w_true: npt.NDArray[np.float64] = rng.normal(0.0, std_setup.sigma)
    f_true = std_setup.B @ w_true
    idx = rng.choice(std_setup.M, size=20, replace=False)
    x_meas = std_setup.x[idx]
    y_meas = f_true[idx]
    _, mu_post = compute_posterior(
        std_setup.B, std_setup.Lambda_inv, x_meas, y_meas, noise_std=1e-6
    )
    f_est = std_setup.B @ mu_post
    assert np.allclose(f_est[idx], y_meas, atol=1e-3)


# ── posterior_predictive_std ───────────────────────────────────────────────────


def test_posterior_predictive_std_shape(std_setup: Setup) -> None:
    rng = np.random.default_rng(0)
    x_meas = rng.uniform(0, 2 * np.pi, size=6)
    y_meas = rng.normal(0, 1, size=6)
    Sigma_post, _ = compute_posterior(
        std_setup.B, std_setup.Lambda_inv, x_meas, y_meas, noise_std=0.2
    )
    std = posterior_predictive_std(std_setup.B, Sigma_post)
    assert std.shape == (std_setup.M,)


def test_posterior_predictive_std_nonnegative(std_setup: Setup) -> None:
    rng = np.random.default_rng(0)
    x_meas = rng.uniform(0, 2 * np.pi, size=6)
    y_meas = rng.normal(0, 1, size=6)
    Sigma_post, _ = compute_posterior(
        std_setup.B, std_setup.Lambda_inv, x_meas, y_meas, noise_std=0.2
    )
    std = posterior_predictive_std(std_setup.B, Sigma_post)
    assert np.all(std >= 0.0)


def test_posterior_predictive_std_matches_direct_diagonal(std_setup: Setup) -> None:
    """Efficient row-wise formula must agree with the explicit diag(B Σ Bᵀ)."""
    rng = np.random.default_rng(0)
    x_meas = rng.uniform(0, 2 * np.pi, size=6)
    y_meas = rng.normal(0, 1, size=6)
    Sigma_post, _ = compute_posterior(
        std_setup.B, std_setup.Lambda_inv, x_meas, y_meas, noise_std=0.2
    )
    std_fast = posterior_predictive_std(std_setup.B, Sigma_post)
    std_direct: npt.NDArray[np.float64] = np.sqrt(
        np.diag(std_setup.B @ Sigma_post @ std_setup.B.T)
    )
    assert np.allclose(std_fast, std_direct, atol=1e-10)
