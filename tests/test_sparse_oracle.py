"""Tests for the sparse column-to-row ER oracle."""

import numpy as np

from core.sparse_oracle import SparseEROracle


def test_column_mapping_matches_b_matrix():
    oracle = SparseEROracle(3)
    B = oracle.B
    for c, row in enumerate(oracle.col_to_row):
        col = B[:, c]
        if row < oracle.d:
            assert col[row] == 1.0
            assert col.sum() == 1.0
        else:
            assert col.sum() == 0.0
    report = oracle.resource_report()
    assert report['max_column_degree'] == 1
    assert report['max_row_degree'] == 5


def test_oracle_is_unitary_and_maps_columns():
    oracle = SparseEROracle(2)
    U = oracle.oracle_unitary()
    assert np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])) < 1e-10
    row_dim = 1 << oracle.n_row
    for c, row in enumerate(oracle.col_to_row[:8]):
        vec = np.zeros(U.shape[0], dtype=complex)
        vec[c * row_dim] = 1.0
        out = U @ vec
        expected = c * row_dim + row
        assert abs(out[expected] - 1.0) < 1e-12
