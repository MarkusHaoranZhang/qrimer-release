"""Tests for the row-side LCU decomposition."""

import numpy as np

from core.er_matrices import build_b_matrix
from core.row_lcu import row_lcu_matrix, row_lcu_report, row_lcu_terms


def test_row_lcu_sums_to_b():
    for N in (2, 3, 5, 10):
        B = build_b_matrix(N)
        terms = row_lcu_terms(N)
        total = sum(row_lcu_matrix(t, N) for t in terms)
        assert np.max(np.abs(total - B)) < 1e-12
        report = row_lcu_report(N)
        assert report['decomposition_error'] < 1e-12
        assert report['max_rows_per_term'] == report['max_columns_per_term']


def test_row_lcu_term_shapes():
    N = 3
    report = row_lcu_report(N)
    assert report['terms'] == 6
    assert report['ancilla_qubits'] == 3
    assert report['nonzeros'] == 5 * N + 4
