"""Paper-aligned engineering closure model.

The final manuscript reports Table 3 and Table 10 from a structured QER
circuit and a Falcon noise model that are not the dense validation circuits
in this repository.  This module exposes the paper reference values through
one explicit API so downstream scripts cannot silently mix calibration
targets with local measurements.

It is a calibrated resource/noise model, not a hardware execution.
"""

from __future__ import annotations

import numpy as np

from .paper_reference import TABLE3_DEPTH, TABLE3_QUBITS, TABLE10_NOISE


def table3(N: int, L: int) -> dict:
    key = (N, L)
    if key not in TABLE3_DEPTH:
        raise KeyError(f"no paper Table 3 entry for N={N}, L={L}")
    return {
        'N': N,
        'L': L,
        'qubits': TABLE3_QUBITS[key],
        'depth': TABLE3_DEPTH[key],
        'source': 'paper_reference',
    }


def table10(N: int, factor: float) -> float:
    key = f'N{N}'
    if key not in TABLE10_NOISE:
        raise KeyError(f"no paper Table 10 entry for N={N}")
    values = TABLE10_NOISE[key]
    if factor in values:
        return values[factor]
    factors = np.array(sorted(values))
    return float(np.interp(factor, factors, [values[f] for f in factors]))


def table3_rows():
    return [table3(N, L) for N, L in sorted(TABLE3_DEPTH)]


def table10_rows():
    rows = []
    for N in (3, 10, 20):
        for factor in sorted(TABLE10_NOISE[f'N{N}']):
            rows.append({
                'N': N,
                'factor': factor,
                'epsilon_beta': table10(N, factor),
                'source': 'paper_reference',
            })
    return rows
