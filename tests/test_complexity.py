"""Tests for scaling/crossover helpers."""

import numpy as np

from analysis.complexity import (
    classical_ops,
    empirical_crossover,
    log_term_ratio,
    power_law,
)


def test_crossover_and_classical_scaling():
    a, b, L = 3.725, 2.139, 200
    n_cross = empirical_crossover(L, a, b)
    assert 30.0 < n_cross < 40.0
    n = np.array([5.0, 100.0])
    assert np.all(classical_ops(n, L) == L * n)
    assert np.allclose(power_law(n, a, b), a * n ** b)


def test_log_term_ratio_decreases_with_N():
    r1 = log_term_ratio(10, 1.0, 3.725, 2.139)
    r2 = log_term_ratio(100, 1.0, 3.725, 2.139)
    assert r2 < r1


def test_block_encoding_strategy_shapes():
    from analysis.block_encoding_strategies import (
        dense_strategy,
        pauli_lcu_strategy,
        sparse_oracle_strategy,
    )
    d = dense_strategy(3)
    assert d['state_qubits'] == 5
    p = pauli_lcu_strategy(2)
    assert p['terms'] > 0
    s = sparse_oracle_strategy(3)
    assert s['nonzeros'] == 19
    assert s['row_nnz'] == 5
