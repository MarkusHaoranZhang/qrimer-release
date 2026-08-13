"""
Evaluation metrics for Quantum RIMER experiments.
"""

import numpy as np
from scipy.stats import kendalltau


def belief_distribution_error(beta_q: np.ndarray, beta_c: np.ndarray) -> float:
    """
    Euclidean distance between quantum and classical belief distributions.

    Parameters
    ----------
    beta_q : np.ndarray
        Quantum RIMER output belief degrees.
    beta_c : np.ndarray
        Classical RIMER ground truth.

    Returns
    -------
    float
        epsilon_beta = ||beta_q - beta_c||_2
    """
    return float(np.sqrt(np.sum((beta_q - beta_c) ** 2)))


def rank_preservation(beta_q: np.ndarray, beta_c: np.ndarray) -> float:
    """
    Kendall's tau coefficient between rankings induced by belief degrees.

    Returns
    -------
    float
        tau in [-1, 1]; 1.0 means perfect rank preservation.
    """
    tau, _ = kendalltau(beta_q, beta_c)
    return float(tau)


def ignorance_fidelity(beta_D_q: float, beta_D_c: float) -> float:
    """
    Relative error in the ignorance estimate.

    Returns
    -------
    float
        |beta_D_q - beta_D_c| / (beta_D_c + 1e-6)
    """
    return abs(beta_D_q - beta_D_c) / (beta_D_c + 1e-6)


def theoretical_speedup(T_classical: float, D_quantum: int) -> float:
    """
    Compute theoretical speedup S(N) = T_classical / D_quantum.

    Parameters
    ----------
    T_classical : float
        Classical computation time or operation count.
    D_quantum : int
        Quantum circuit depth.

    Returns
    -------
    float
    """
    return T_classical / max(D_quantum, 1)


def expected_utility(beta: np.ndarray, utilities: np.ndarray) -> float:
    """
    Expected utility of a belief distribution (Section 3.1 of the paper):

        u(S(A*)) = sum_j u(D_j) * beta_j

    Parameters
    ----------
    beta : np.ndarray of shape (N,)
        Belief degrees over the evaluation grades.
    utilities : np.ndarray of shape (N,)
        Utility of each evaluation grade.

    Returns
    -------
    float
    """
    beta = np.asarray(beta, dtype=float)
    utilities = np.asarray(utilities, dtype=float)
    return float(np.sum(utilities * beta))
