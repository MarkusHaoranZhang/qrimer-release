"""
Explore alternative block-encoding constructions.

Compares the dense Hermitian dilation, Pauli/LCU decomposition, and the
fixed-sparsity oracle specification.  The goal is to choose the most viable
route for the scalable implementation rather than assume one construction.
"""

from analysis.block_encoding_strategies import (
    dense_strategy,
    pauli_lcu_strategy,
    sparse_oracle_strategy,
)


def main():
    print("=" * 86)
    print("Block-encoding strategy exploration")
    print("=" * 86)
    print(f"  {'N':>3}  {'strategy':>15}  {'qubits':>8}  {'terms':>7}  "
          f"{'queries':>8}  {'gates(est.)':>12}")
    for N in (3, 4, 5, 6):
        for fn in (dense_strategy, pauli_lcu_strategy, sparse_oracle_strategy):
            r = fn(N)
            qubits = r.get('state_qubits', 0) + r.get('ancilla_qubits', 0)
            print(f"  {N:>3}  {r['name']:>15}  {qubits:>8}  "
                  f"{r.get('terms', '-'):>7}  {r.get('queries', '-'):>8}  "
                  f"{r.get('gate_estimate', r.get('unitary_gates', '-')):>12}")


if __name__ == '__main__':
    main()
