"""
ER synthesis matrices and spectral helpers.

The ER combination is a bilinear map from the (N+2)^2-dimensional tensor
product of two mass vectors to the (N+2)-dimensional combined mass vector.
The fixed 0/1 matrix B encodes that map; the data-dependent conflict factor
K is applied as a scalar rescaling after the block-encoded step.
"""

import numpy as np


def build_b_matrix(N: int) -> np.ndarray:
    """
    Build the unnormalised ER synthesis matrix B of shape (N+2, (N+2)^2).

    The layout is (m_1, ..., m_N, m_bar_D, m_hat_D); B has a constant
    0/1 sparsity pattern and does not depend on the data.
    """
    d = N + 2
    B = np.zeros((d, d * d), dtype=float)

    def col(p: int, q: int) -> int:
        return p * d + q

    for j in range(N):
        B[j, col(j, j)] = 1.0
        B[j, col(j, N)] = 1.0
        B[j, col(j, N + 1)] = 1.0
        B[j, col(N, j)] = 1.0
        B[j, col(N + 1, j)] = 1.0

    B[N, col(N, N)] = 1.0
    B[N + 1, col(N + 1, N + 1)] = 1.0
    B[N + 1, col(N + 1, N)] = 1.0
    B[N + 1, col(N, N + 1)] = 1.0
    return B


def hermitian_dilation(B: np.ndarray) -> np.ndarray:
    """Return [[0, B], [B^H, 0]] for a rectangular B."""
    m, n = B.shape
    M = np.zeros((m + n, m + n), dtype=complex)
    M[:m, m:] = B
    M[m:, :m] = B.conj().T
    return M


def spectral_report(N: int) -> dict:
    """
    Numerical spectral data for the fixed ER synthesis operator.

    The singular values of B are invariant to the data-dependent scalar K;
    K only rescales the output.  This distinguishes the spectral condition
    number of the block-encoded operator from the conflict-dependent
    amplification factor used in the sensitivity analysis.
    """
    B = build_b_matrix(N)
    singular = np.linalg.svd(B, compute_uv=False)
    alpha = float(singular[0])
    smallest = float(singular[-1]) if singular[-1] > 0 else 0.0
    cond = float(singular[0] / smallest) if smallest > 0 else float('inf')

    M = hermitian_dilation(B)
    eig = np.linalg.eigvalsh(M)
    nonzero = np.sort(np.abs(eig[np.abs(eig) > 1e-12]))
    return {
        'N': N,
        'shape': B.shape,
        'dilation_shape': M.shape,
        'alpha': alpha,
        'singular_values': singular.tolist(),
        'condition_number': cond,
        'dilation_nonzero_eigenvalues': nonzero.tolist(),
        'dilation_condition_number': (
            float(nonzero[-1] / nonzero[0]) if len(nonzero) else float('inf')
        ),
    }
