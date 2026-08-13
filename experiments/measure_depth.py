"""
Measure actual quantum circuit depths via Qiskit transpilation.

This script builds QER circuits for small problem sizes and reports
the transpiled depth, providing the empirical data points that
correspond to Table 3 in the paper.

For N <= 5, the full circuit can be transpiled and its depth measured.
For larger N, only the depth estimate formula is used.
"""

import os
import sys

import numpy as np
from qiskit import transpile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import RANDOM_SEED
from core.qbrb import QBRB
from core.qer import QER
from experiments.datasets import SyntheticDataGenerator


def measure_qer_transpiled_depth(N: int, L: int, T: int = 4) -> dict:
    """
    Build and transpile a QER circuit, reporting actual gate count and depth.

    Parameters
    ----------
    N : int — number of evaluation grades
    L : int — number of rules
    T : int — number of antecedent attributes

    Returns
    -------
    dict with keys: n_qubits, depth_raw, depth_transpiled, gate_count
    """
    gen = SyntheticDataGenerator(N, L, T, seed=RANDOM_SEED)
    betas, rule_weights, _ = gen.generate_rules()

    # Use uniform activation weights for measurement
    w = np.ones(L) / L

    qer = QER(N=N, L=L, kappa_max=50.0, epsilon=0.01)
    circ = qer.build_circuit(w, betas)

    # Raw depth (before transpilation)
    depth_raw = circ.depth()

    # Transpile to basis gates
    transpiled = transpile(circ, basis_gates=['cx', 'u3'], optimization_level=1)
    depth_transpiled = transpiled.depth()
    gate_count = transpiled.size()

    return {
        'N': N,
        'L': L,
        'n_qubits': circ.num_qubits,
        'depth_raw': depth_raw,
        'depth_transpiled': depth_transpiled,
        'gate_count': gate_count,
        'estimate': qer.estimate_depth(),
    }


def measure_qbrb_depth(N: int, L: int, T: int = 4) -> dict:
    """Build and transpile a Q-BRB circuit."""
    gen = SyntheticDataGenerator(N, L, T, seed=RANDOM_SEED)
    betas, _, _ = gen.generate_rules()
    ant_masks = np.ones((L, T), dtype=int)

    qbrb = QBRB(N=N, L=L, T=T)
    circ = qbrb.prepare_full_brb(betas, ant_masks)

    transpiled = transpile(circ, basis_gates=['cx', 'u3'], optimization_level=1)

    return {
        'N': N,
        'L': L,
        'n_qubits': circ.num_qubits,
        'depth_raw': circ.depth(),
        'depth_transpiled': transpiled.depth(),
        'gate_count': transpiled.size(),
        'estimate': qbrb.estimate_depth(),
    }


def main():
    print("=" * 70)
    print("Circuit depth measurement (actual Qiskit transpilation)")
    print("=" * 70)

    # QER circuits — small sizes that can be transpiled
    print("\n[QER circuit depths]")
    print(f"  {'N':>3}  {'L':>3}  {'qubits':>6}  {'raw':>6}  {'transpiled':>10}  "
          f"{'gates':>6}  {'estimate':>8}")
    for N, L in [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4)]:
        try:
            result = measure_qer_transpiled_depth(N, L, T=2)
            print(f"  {result['N']:>3}  {result['L']:>3}  "
                  f"{result['n_qubits']:>6}  {result['depth_raw']:>6}  "
                  f"{result['depth_transpiled']:>10}  {result['gate_count']:>6}  "
                  f"{result['estimate']:>8}")
        except Exception as e:
            print(f"  N={N}, L={L}: failed ({e})")

    # Q-BRB circuits
    print("\n[Q-BRB circuit depths]")
    print(f"  {'N':>3}  {'L':>3}  {'qubits':>6}  {'raw':>6}  {'transpiled':>10}  "
          f"{'gates':>6}  {'estimate':>8}")
    for N, L in [(3, 2), (3, 4), (5, 4)]:
        try:
            result = measure_qbrb_depth(N, L, T=2)
            print(f"  {result['N']:>3}  {result['L']:>3}  "
                  f"{result['n_qubits']:>6}  {result['depth_raw']:>6}  "
                  f"{result['depth_transpiled']:>10}  {result['gate_count']:>6}  "
                  f"{result['estimate']:>8}")
        except Exception as e:
            print(f"  N={N}, L={L}: failed ({e})")

    print("\nNote: 'estimate' is the theoretical upper bound from estimate_depth().")
    print("'transpiled' is the actual depth after Qiskit optimization level 1.")


if __name__ == '__main__':
    main()
