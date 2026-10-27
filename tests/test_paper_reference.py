"""Lock the final manuscript's reference table values."""

from analysis.paper_closure import table3, table10
from analysis.paper_reference import (
    TABLE3_DEPTH,
    TABLE3_QUBITS,
    TABLE10_NOISE,
)


def test_table3_reference_values():
    assert TABLE3_DEPTH[(5, 50)] == 214
    assert TABLE3_DEPTH[(10, 100)] == 687
    assert TABLE3_DEPTH[(15, 200)] == 1243
    assert TABLE3_DEPTH[(20, 200)] == 2156
    assert TABLE3_DEPTH[(30, 200)] == 4621
    assert TABLE3_QUBITS[(20, 200)] == 30


def test_table10_reference_values():
    assert TABLE10_NOISE['N3'][1.0] == 0.0136
    assert TABLE10_NOISE['N10'][1.0] == 0.023
    assert TABLE10_NOISE['N20'][1.0] == 0.065
    assert TABLE10_NOISE['N20'][10.0] == 0.485


def test_closure_model_reads_paper_tables():
    assert table3(20, 200)['depth'] == 2156
    assert table3(20, 200)['qubits'] == 30
    assert table10(20, 1.0) == 0.065
    assert abs(table10(20, 3.0) - 0.161) < 1e-12
