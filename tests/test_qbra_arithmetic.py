"""Tests for the first executable QBRA arithmetic stage."""

import numpy as np

from core.qbra_arithmetic import (
    CombinedT4Product,
    DivisionLookup,
    ProductLookup,
    QBRAArithmetic,
    ShiftAddMultiplier,
    T4ProductShiftAdd,
)
from core.qbra_div_oracle import DivisionOracle


def test_weighted_value_matches_fixed_point_reference():
    q = QBRAArithmetic(bits=3, scale=8)
    got = q.weighted_value(0.5, 0.8)
    expected = (q.quantize(0.5) / (2 ** q.bits - 1)) * 0.8
    assert abs(got - expected) < 0.06


def test_weighted_sum_matches_fixed_point_reference():
    q = QBRAArithmetic(bits=3, scale=8)
    got = q.weighted_sum([0.5, 0.3], [0.8, 0.6])
    expected = (
        (q.quantize(0.5) / (2 ** q.bits - 1)) * 0.8
        + (q.quantize(0.3) / (2 ** q.bits - 1)) * 0.6
    )
    assert abs(got - expected) < 0.08
    assert q.weighted_sum_circuit([0.5, 0.3], [0.8, 0.6]).num_qubits > 0


def test_division_lookup_circuit():
    d = DivisionLookup(bits=2)
    for v, z in [(1, 2), (3, 1), (1, 3), (3, 2)]:
        assert d.divide(v, z) == d.quotient(v, z)
    assert d.circuit().num_qubits == 6


def test_product_lookup_circuit():
    p = ProductLookup(T=2, bits=2)
    assert p.evaluate([2, 3]) == p.product([2, 3]) == 6
    p4 = ProductLookup(T=4, bits=1)
    assert p4.evaluate([1, 1, 0, 1]) == p4.product([1, 1, 0, 1]) == 0


def test_qbra_t1_activation_path():
    q = QBRAArithmetic(bits=3, scale=8)
    w = q.activation_weights([0.5, 0.3], [0.8, 0.6])
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1]


def test_t2_activation_feeds_qer():
    from core.qer import QER
    q = QBRAArithmetic(bits=3, scale=8)
    w = q.activation_weights_t2([(0.8, 0.7), (0.4, 0.3)], [0.8, 0.6])
    assert np.isclose(w.sum(), 1.0)
    betas = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.1]])
    qer = QER(N=3, L=2, kappa_max=50.0)
    beta_q, _, _ = qer.quantum_simulate(w, betas)
    beta_c, _ = qer.classical_simulate(w, betas)
    assert np.linalg.norm(beta_q - beta_c) < 1e-9


def test_shift_add_multiplier_matches_classical():
    for n in (2, 4):
        m = ShiftAddMultiplier(n)
        for a, b in [(1, 1), (3, 2), (7, 3), (10, 2)]:
            if a < 2 ** n:
                assert m.evaluate(a, b) == a * b


def test_t4_product_tree():
    q = T4ProductShiftAdd()
    assert q.evaluate([3, 2, 1, 3]) == 18
    assert q.evaluate([3, 3, 3, 3]) == 81


def test_t4_activation_feeds_qer():
    from core.qer import QER
    q = QBRAArithmetic(bits=3, scale=8)
    w = q.activation_weights_t4(
        [(0.8, 0.7, 0.6, 0.5), (0.4, 0.3, 0.2, 0.1)],
        [0.8, 0.6],
    )
    assert np.isclose(w.sum(), 1.0)
    betas = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.1]])
    qer = QER(N=3, L=2, kappa_max=50.0)
    beta_q, _, _ = qer.quantum_simulate(w, betas)
    beta_c, _ = qer.classical_simulate(w, betas)
    assert np.linalg.norm(beta_q - beta_c) < 1e-9


def test_adaptive_division_resolves_small_quotients():
    d = DivisionLookup(bits=4)
    assert abs(d.divide_adaptive(1, 100) - 0.01) < 0.01
    assert abs(d.divide_adaptive(8, 15) - 8 / 15) < 1e-12


def test_t4_batch_product():
    q = T4ProductShiftAdd(bits=3)
    out = q.batch_evaluate([[3, 2, 1, 3], [3, 3, 3, 3]])
    assert out == [18, 81]


def test_combined_t4_product():
    c = CombinedT4Product(bits=2)
    assert c.evaluate([3, 2, 1, 3]) == 18
    c3 = CombinedT4Product(bits=3)
    assert c3.evaluate([7, 7, 7, 7]) == 2401


def test_shared_combined_t4_product():
    from core.qbra_combined import CombinedT4ProductShared
    c = CombinedT4ProductShared(bits=4)
    assert c.evaluate([15, 15, 15, 15]) == 50625


def test_high_precision_division_oracle():
    d = DivisionOracle(bits=6)
    assert abs(d.divide_adaptive(1, 100) - 0.01) < 0.001
    assert abs(d.divide_adaptive(8, 15) - 8 / 15) < 0.01
