"""
Row-side LCU decomposition of the ER synthesis matrix B.

B is decomposed into six partial isometries, each with distinct rows and
columns.  The decomposition is the row-side component required to combine
the sparse column oracle into a complete block encoding.
"""

import numpy as np

from .er_matrices import build_b_matrix


def row_lcu_terms(N: int) -> list[tuple[tuple[int, int], ...]]:
    """
    Return six lists of (column, row) pairs whose sum is B.

    The terms are:
      1. (j,j)->j for j<N, plus (N,N)->N
      2. (j,N)->j for j<N, plus (N+1,N)->N+1
      3. (j,N+1)->j for j<N, plus (N,N+1)->N+1
      4. (N+1,N+1)->N+1
      5. (N,j)->j for j<N
      6. (N+1,j)->j for j<N
    """
    d = N + 2
    terms = [
        tuple([(j * d + j, j) for j in range(N)] + [(N * d + N, N)]),
        tuple([(j * d + N, j) for j in range(N)] + [((N + 1) * d + N, N + 1)]),
        tuple([(j * d + N + 1, j) for j in range(N)] + [(N * d + N + 1, N + 1)]),
        (((N + 1) * d + (N + 1), N + 1),),
        tuple([(N * d + j, j) for j in range(N)]),
        tuple([((N + 1) * d + j, j) for j in range(N)]),
    ]
    return terms


def row_lcu_matrix(term: tuple[tuple[int, int], ...], N: int) -> np.ndarray:
    """Embed one partial-isometry term into a d x d^2 matrix."""
    d = N + 2
    A = np.zeros((d, d * d))
    for c, r in term:
        A[r, c] = 1.0
    return A


def row_lcu_report(N: int) -> dict:
    B = build_b_matrix(N)
    terms = row_lcu_terms(N)
    total = sum(row_lcu_matrix(term, N) for term in terms)
    max_rows = max(len({r for _, r in term}) for term in terms)
    max_cols = max(len({c for c, _ in term}) for term in terms)
    return {
        'N': N,
        'terms': len(terms),
        'ancilla_qubits': int(np.ceil(np.log2(len(terms)))),
        'normalization': len(terms),
        'nonzeros': int(B.astype(bool).sum()),
        'decomposition_error': float(np.max(np.abs(total - B))),
        'max_rows_per_term': max_rows,
        'max_columns_per_term': max_cols,
    }
