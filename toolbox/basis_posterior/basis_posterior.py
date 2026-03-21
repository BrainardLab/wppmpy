"""Finite sinusoidal basis and Gaussian posterior utilities.

Functions
---------
build_basis_matrix   Build the (M × N) sinusoidal basis matrix and domain vector.
make_prior           Construct the exponentially-decaying Gaussian prior.
draw_from_prior      Sample weight vectors from the prior.
compute_posterior    Gaussian-Gaussian conjugate posterior update.
posterior_predictive_std  Pointwise posterior standard deviation over the domain.
"""

import numpy as np
import numpy.typing as npt


def build_basis_matrix(
    N: int,
    M: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the domain vector and the sinusoidal basis matrix.

    Parameters
    ----------
    N : int
        Number of basis functions.  Must be odd: N = 1 + 2*n_freqs.
    M : int
        Number of evenly-spaced sample points on [0, 2π).

    Returns
    -------
    x : ndarray, shape (M,)
        Sample points in [0, 2π).
    B : ndarray, shape (M, N)
        Basis matrix.  Column 0 is the DC component (1s).
        Columns 2k-1 and 2k are cos(k·x) and sin(k·x) for k = 1 … n_freqs.
    """
    if N % 2 == 0:
        raise ValueError(f"N must be odd; got {N}")

    n_freqs = (N - 1) // 2
    x: npt.NDArray[np.float64] = np.linspace(0, 2 * np.pi, M, endpoint=False)

    B: npt.NDArray[np.float64] = np.zeros((M, N))
    B[:, 0] = 1.0
    for k in range(1, n_freqs + 1):
        B[:, 2 * k - 1] = np.cos(k * x)
        B[:, 2 * k] = np.sin(k * x)

    return x, B


def make_prior(
    N: int,
    gamma: float,
    epsilon: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Construct the exponentially-decaying Gaussian prior on weights.

    The prior std for basis function j is:
        sigma_j = gamma * exp(-epsilon * k_j)
    where k_j is the sinusoidal order (0 for DC, k for cos(kx) and sin(kx)).

    Parameters
    ----------
    N : int
        Number of basis functions.
    gamma : float
        Base standard deviation (scale of functions at DC).
    epsilon : float
        Exponential decay rate with sinusoidal order.

    Returns
    -------
    sigma : ndarray, shape (N,)
        Prior standard deviation for each weight.
    Lambda_inv : ndarray, shape (N, N)
        Inverse prior covariance diag(1/sigma²).
    """
    orders = (np.arange(N) + 1) // 2  # sinusoidal order per weight, shape (N,)
    sigma: npt.NDArray[np.float64] = gamma * np.exp(-epsilon * orders)
    Lambda_inv: npt.NDArray[np.float64] = np.diag(1.0 / sigma**2)
    return sigma, Lambda_inv


def draw_from_prior(
    sigma: npt.NDArray[np.float64],
    rng: np.random.Generator,
    n_samples: int = 1,
) -> npt.NDArray[np.float64]:
    """Draw weight vectors from the Gaussian prior.

    Parameters
    ----------
    sigma : ndarray, shape (N,)
        Prior standard deviation for each weight.
    rng : np.random.Generator
        Random number generator (e.g. np.random.default_rng(seed)).
    n_samples : int
        Number of independent weight vectors to draw.

    Returns
    -------
    W : ndarray, shape (N, n_samples)
        Sampled weight vectors.  W[:, s] is the s-th sample.
    """
    N = sigma.shape[0]
    W: npt.NDArray[np.float64] = rng.normal(
        loc=0.0, scale=sigma[:, np.newaxis], size=(N, n_samples)
    )
    return W


def compute_posterior(
    B: npt.NDArray[np.float64],
    Lambda_inv: npt.NDArray[np.float64],
    x_meas: npt.NDArray[np.float64],
    y_meas: npt.NDArray[np.float64],
    noise_std: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Gaussian-Gaussian conjugate posterior update.

    Given a prior  w ~ N(0, Λ)  and noisy measurements
        y_i = Φ[i, :] @ w + ε_i,   ε_i ~ N(0, noise_std²)
    the posterior is  w | y ~ N(mu_post, Sigma_post)  where:

        Sigma_post = (Λ⁻¹ + ΦᵀΦ / noise_std²)⁻¹
        mu_post    = Sigma_post @ (Φᵀ y / noise_std²)

    The design matrix Φ is built internally from x_meas and the structure of B.

    Parameters
    ----------
    B : ndarray, shape (M, N)
        Full basis matrix over the domain (used to infer N and n_freqs).
    Lambda_inv : ndarray, shape (N, N)
        Inverse prior covariance diag(1/sigma²).
    x_meas : ndarray, shape (n_meas,)
        Measurement locations in [0, 2π).
    y_meas : ndarray, shape (n_meas,)
        Noisy observations at x_meas.
    noise_std : float
        Known measurement noise standard deviation.

    Returns
    -------
    Sigma_post : ndarray, shape (N, N)
        Posterior covariance matrix.
    mu_post : ndarray, shape (N,)
        Posterior mean weight vector.
    """
    N = B.shape[1]
    n_freqs = (N - 1) // 2
    n_meas = x_meas.shape[0]

    # Build design matrix Φ, shape (n_meas, N)
    Phi: npt.NDArray[np.float64] = np.zeros((n_meas, N))
    Phi[:, 0] = 1.0
    for k in range(1, n_freqs + 1):
        Phi[:, 2 * k - 1] = np.cos(k * x_meas)
        Phi[:, 2 * k] = np.sin(k * x_meas)

    Sigma_post: npt.NDArray[np.float64] = np.linalg.inv(
        Lambda_inv + Phi.T @ Phi / noise_std**2
    ).astype(np.float64)
    mu_post: npt.NDArray[np.float64] = (
        Sigma_post @ (Phi.T @ y_meas / noise_std**2)
    ).astype(np.float64)
    return Sigma_post, mu_post


def posterior_predictive_std(
    B: npt.NDArray[np.float64],
    Sigma_post: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Pointwise posterior predictive standard deviation over the full domain.

    Computes diag(B @ Sigma_post @ Bᵀ) efficiently without forming the
    full (M × M) matrix:
        var[i] = sum_j (B @ Sigma_post)[i, j] * B[i, j]

    Parameters
    ----------
    B : ndarray, shape (M, N)
        Basis matrix evaluated at all domain sample points.
    Sigma_post : ndarray, shape (N, N)
        Posterior covariance matrix.

    Returns
    -------
    std : ndarray, shape (M,)
        Posterior standard deviation at each sample point.
    """
    BSig: npt.NDArray[np.float64] = B @ Sigma_post  # (M, N)
    var: npt.NDArray[np.float64] = np.sum(BSig * B, axis=1)  # (M,)
    return np.sqrt(np.maximum(var, 0.0))
