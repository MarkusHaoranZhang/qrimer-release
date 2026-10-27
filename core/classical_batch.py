"""
Batched vectorised classical ER baseline.

The ER recursion is sequential in the rule index, but independent problem
instances can be advanced together.  This baseline batches the conflict
coefficient and mass-vector updates over a batch of BRB instances, giving a
stronger classical reference for throughput comparisons.
"""

import numpy as np


def er_combine_batch(w: np.ndarray, betas: np.ndarray):
    """
    Batched ER combination.

    Parameters
    ----------
    w : np.ndarray of shape (batch, L)
    betas : np.ndarray of shape (batch, L, N)

    Returns
    -------
    beta : np.ndarray of shape (batch, N)
    beta_D : np.ndarray of shape (batch,)
    """
    w = np.asarray(w, dtype=float)
    betas = np.asarray(betas, dtype=float)
    B, L, N = betas.shape
    if w.shape != (B, L):
        raise ValueError("w must have shape (batch, L)")

    m = np.zeros((B, L, N + 2))
    active = w > 1e-12
    m[..., :N] = w[..., None] * betas
    m[..., N] = 1.0 - w
    m[..., N + 1] = w * np.maximum(0.0, 1.0 - betas.sum(axis=2))
    m[~active, N] = 1.0

    m_c = m[:, 0, :].copy()
    for k in range(1, L):
        m_n = m[:, k, :]
        mc_N = m_c[:, :N]
        mn_N = m_n[:, :N]
        conflict = mc_N.sum(axis=1) * mn_N.sum(axis=1) - np.sum(
            mc_N * mn_N, axis=1
        )
        K = 1.0 / (1.0 - conflict + 1e-15)
        mD_c = m_c[:, N] + m_c[:, N + 1]
        mD_n = m_n[:, N] + m_n[:, N + 1]

        m_new = np.empty_like(m_c)
        m_new[:, :N] = K[:, None] * (
            mc_N * mn_N + mc_N * mD_n[:, None] + mD_c[:, None] * mn_N
        )
        m_new[:, N + 1] = K * (
            m_c[:, N + 1] * m_n[:, N + 1]
            + m_c[:, N + 1] * m_n[:, N]
            + m_c[:, N] * m_n[:, N + 1]
        )
        m_new[:, N] = K * m_c[:, N] * m_n[:, N]
        m_c = m_new

    denom = 1.0 - m_c[:, N] + 1e-15
    beta = m_c[:, :N] / denom[:, None]
    beta_D = m_c[:, N + 1] / denom
    return beta, beta_D
