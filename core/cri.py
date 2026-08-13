"""
CRI-based fuzzy inference baseline.

Implements a conventional Mamdani-style compositional rule of inference
(CRI) as the second baseline of the paper (Section 6.1).

RIMER originally replaced CRI with the ER approach precisely because CRI
operates on singletons and cannot represent epistemic ignorance.  This
baseline illustrates that limitation: its output is always a normalised
distribution over the evaluation grades, with no residual ignorance
component.

Reference: Zadeh 1973; Mamdani 1975; Yang et al. 2006 (baseline comparison).
"""

import numpy as np


def cri_max_min_infer(match_degrees: np.ndarray,
                      betas: np.ndarray) -> np.ndarray:
    """
    Mamdani max-min compositional rule of inference.

    Each rule contributes a truncated consequent
    mu_k(j) = min(match_k, beta_jk); the output fuzzy set over the N
    evaluation grades is the pointwise maximum over all rules
    mu_D(j) = max_k min(match_k, beta_jk).

    The output is normalised to a probability-like distribution.  Note that
    CRI provides no ignorance component: even when all rules are weakly
    matched or incomplete, the normalised output sums to exactly 1.

    Parameters
    ----------
    match_degrees : np.ndarray of shape (L,)
        Matching degree alpha_k of each rule (Eq. 8 of the paper).
    betas : np.ndarray of shape (L, N)
        Consequent belief degrees (used as consequent fuzzy sets).

    Returns
    -------
    np.ndarray of shape (N,)
        Normalised output distribution over evaluation grades.
    """
    L, N = betas.shape
    mu_D = np.zeros(N)
    for k in range(L):
        mu_D = np.maximum(mu_D, np.minimum(match_degrees[k],
                                           np.maximum(betas[k], 0.0)))
    total = float(mu_D.sum())
    if total > 1e-12:
        mu_D = mu_D / total
    return mu_D


def cri_infer(match_degrees: np.ndarray,
              betas: np.ndarray) -> tuple[np.ndarray, float]:
    """
    CRI inference returning a belief-like distribution plus ignorance.

    The ignorance component is structurally zero: CRI cannot represent
    epistemic ignorance (the defining advantage of BRB reasoning).

    Returns
    -------
    beta   : np.ndarray of shape (N,)
    beta_D : float  — always 0.0 for CRI
    """
    return cri_max_min_infer(match_degrees, betas), 0.0
