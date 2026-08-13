"""
BF-QC basic combination baseline.

Implements the elementary quantum conjunctive combination rule from BF-QC
(Yan et al., Expert Systems with Applications, 2023) as a classical
reference baseline.

BF-QC encodes basic belief assignments as superposition states
|m> = sum_F sqrt(m(F)) |j_F> and implements the conjunctive combination
rule m_conj(A) = sum_{B ∩ C = A} m1(B) * m2(C) on quantum circuits.

As a baseline for Q-RIMER (Section 6.1), this module applies the BF-QC
conjunctive rule directly to the belief degrees WITHOUT the ER-specific
treatments:
  - no two-part decomposition of unassigned mass into m_bar_D / m_hat_D
    (the residual mass m(Theta) is lumped),
  - no ER weighting structure (activation weights are not folded into
    the mass vectors).

This is exactly the baseline used in the ablation study (Section 6.3):
replacing the ER synthesis matrix M_ER with the conjunctive matrix M_cap
increases the belief distribution error by a factor of 15 and
systematically underestimates ignorance by 22%.
"""

import numpy as np


def conjunctive_combine(w: np.ndarray | None,
                        betas: np.ndarray) -> tuple[np.ndarray, float]:
    """
    BF-QC conjunctive combination of evidence (M_cap baseline).

    For singletons-only BBAs (the RIMER mass layout), the conjunctive rule
    m_cap(A) = sum_{B ∩ C = A} m1(B) * m2(C) reduces to, for each grade j::

        m_cap(j)   = m1(j) * m2(j) + m1(j) * m2(Theta) + m1(Theta) * m2(j)
        m_cap(Theta) = m1(Theta) * m2(Theta)

    where the residual mass m(Theta) = 1 - sum_j beta_jk is a single lumped
    component (no m_bar_D / m_hat_D decomposition).

    Parameters
    ----------
    w : np.ndarray of shape (L,) or None
        Activation weights.  The BF-QC basic baseline combines the belief
        degrees directly (w=None); passing w folds the weights into the
        mass vectors (weighted variant, used for controlled comparisons).
    betas : np.ndarray of shape (L, N)
        Belief degrees.

    Returns
    -------
    beta   : np.ndarray of shape (N,)
        Final belief degrees (normalised).
    beta_D : float
        Residual (lumped) ignorance mass.
    """
    L, N = betas.shape
    betas = np.asarray(betas, dtype=float)
    if w is None:
        w = np.ones(L)
    w = np.asarray(w, dtype=float)

    def _mass(k):
        m = np.zeros(N + 1)
        m[:N] = w[k] * np.maximum(betas[k], 0.0)
        m[N] = max(0.0, 1.0 - w[k] * float(np.sum(np.maximum(betas[k], 0.0))))
        return m

    m = _mass(0)
    for k in range(1, L):
        mn = _mass(k)
        m_new = np.zeros(N + 1)
        m_new[:N] = (m[:N] * mn[:N]            # m1(j) * m2(j)
                     + m[:N] * mn[N]           # m1(j) * m2(Theta)
                     + m[N] * mn[:N])          # m1(Theta) * m2(j)
        m_new[N] = m[N] * mn[N]                # m1(Theta) * m2(Theta)
        m = m_new

    denom = float(m.sum()) + 1e-15
    beta = m[:N] / denom
    beta_D = m[N] / denom
    return beta, beta_D
