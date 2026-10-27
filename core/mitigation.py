"""
High-conflict mitigation helpers.

The block-encoding success probability and the effective amplification
factor degrade under strong evidence conflict.  Two standard remedies are
implemented here: discounting the conflicting belief mass toward ignorance,
and a threshold rule that delegates a step to the classical ER recursion.
"""

import numpy as np


def conflict_coefficient(m_I: np.ndarray, m_next: np.ndarray) -> float:
    """Pairwise conflict coefficient between two mass vectors."""
    N = len(m_I) - 2
    a = np.asarray(m_I[:N], dtype=float)
    b = np.asarray(m_next[:N], dtype=float)
    return float(a.sum() * b.sum() - np.dot(a, b))


def discount_beliefs(betas: np.ndarray, discount: float) -> np.ndarray:
    """
    Discount belief degrees toward ignorance.

    ``discount=0`` leaves the beliefs unchanged; ``discount=1`` removes all
    singleton mass.  The reduced mass remains as residual ignorance in the
    rule consequent.
    """
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0, 1]")
    betas = np.asarray(betas, dtype=float)
    return betas * (1.0 - discount)


def should_delegate(conflict: float, threshold: float = 0.9) -> bool:
    """Return True when a step should be delegated to classical ER."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    return conflict > threshold
