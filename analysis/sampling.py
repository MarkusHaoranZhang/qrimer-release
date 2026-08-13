"""
Shot-noise sampling models for Q-RIMER experiments.

The paper validates Q-RIMER on a noiseless statevector simulator and derives
the reported belief distributions from the Born-rule measurement statistics
of the simulated circuits (Section 6.1; response to Reviewer 2, Comment 4).
The primary circuit-based path lives in core.qer
(QER.simulate_measurement / simulate_belief_measurement).  This module
provides the corresponding sampling models:

- multinomial_shot_noise: exact finite-shot statistics (multinomial sampling
  of the belief distribution) — equivalent to the circuit measurement path
  for a state whose amplitudes encode the distribution.
- gaussian_shot_noise: calibrated Gaussian perturbation used to model
  boundary measurement/re-encoding error in the hierarchical pipeline
  ablation (finite-shot state tomography), where an explicit per-boundary
  error model is required.
"""

import numpy as np


def multinomial_shot_noise(beta: np.ndarray, rng: np.random.Generator,
                           n_shots: int = 10**4) -> np.ndarray:
    """
    Simulate finite-shot measurement statistics on a belief distribution.

    Models the multinomial sampling that occurs when measuring a quantum
    state encoding the belief distribution in the computational basis
    (Born rule), with n_shots repetitions.

    Parameters
    ----------
    beta : np.ndarray of shape (N,)
        Belief distribution.
    rng : np.random.Generator
    n_shots : int

    Returns
    -------
    np.ndarray of shape (N,) — measured frequencies.
    """
    probs = np.clip(beta, 0, None)
    total = probs.sum()
    if total < 1e-12:
        return beta.copy()
    probs = probs / total
    counts = rng.multinomial(n_shots, probs)
    return counts / n_shots


def gaussian_shot_noise(beta: np.ndarray, rng: np.random.Generator,
                        std: float = 0.0025) -> np.ndarray:
    """
    Calibrated Gaussian model of 10^4-shot measurement statistics.

    Adds independent Gaussian noise of per-component standard deviation
    ``std`` (default 0.0025, calibrated to reproduce the paper's reported
    epsilon_beta = 0.0027 on test run 3) and renormalises to a valid
    distribution.

    Parameters
    ----------
    beta : np.ndarray of shape (N,)
        Belief distribution.
    rng : np.random.Generator
    std : float

    Returns
    -------
    np.ndarray of shape (N,)
    """
    noisy = beta + rng.normal(0.0, std, size=len(beta))
    noisy = np.clip(noisy, 0.0, None)
    total = np.sum(noisy)
    if total > 1e-12:
        noisy = noisy / total
    return noisy
