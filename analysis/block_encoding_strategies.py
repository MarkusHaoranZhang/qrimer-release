"""
Resource comparison of alternative block-encoding strategies.

Three candidate constructions are compared:

1. dense Hermitian-dilation unitary (current executable validation path);
2. Pauli/LCU decomposition of the Hermitian dilation;
3. sparse-oracle construction using the fixed 5-nonzero-per-row pattern.

The first is executable for N <= 10 in this repository; the other two are
resource specifications used to explore the scalable implementation route.
"""

import math

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from core.er_matrices import build_b_matrix, hermitian_dilation


def dense_strategy(N: int) -> dict:
    dim = (N + 2) * (N + 3)
    state_qubits = int(math.ceil(math.log2(dim)))
    return {
        'name': 'dense-Hermitian',
        'N': N,
        'state_qubits': state_qubits,
        'ancilla_qubits': 1,
        'unitary_dim': 1 << (state_qubits + 1),
        'unitary_gates': 1,
        'queries': 1,
    }


def pauli_lcu_strategy(N: int) -> dict:
    B = build_b_matrix(N)
    M = hermitian_dilation(B)
    alpha = float(np.linalg.norm(M, 2))
    D = M.shape[0]
    n_state = int(math.ceil(math.log2(D)))
    dim = 1 << n_state
    Mp = np.zeros((dim, dim), dtype=complex)
    Mp[:D, :D] = M / alpha
    op = SparsePauliOp.from_operator(Operator(Mp))
    terms = len(op)
    coeffs = np.real_if_close(op.coeffs).real
    C = float(np.sum(np.abs(coeffs)))
    n_anc = max(1, int(math.ceil(math.log2(max(terms, 2)))))
    return {
        'name': 'pauli-LCU',
        'N': N,
        'state_qubits': n_state,
        'ancilla_qubits': n_anc,
        'terms': terms,
        'normalization_C': C,
        'queries': terms,
        'gate_estimate': terms * n_state,
    }


def sparse_oracle_strategy(N: int) -> dict:
    d = N + 2
    nonzeros = 5 * N + 4
    row_nnz = 5
    return {
        'name': 'sparse-oracle',
        'N': N,
        'state_qubits': int(math.ceil(math.log2(d * d))),
        'row_nnz': row_nnz,
        'nonzeros': nonzeros,
        'queries': 1,
        'gate_estimate': nonzeros * max(1, int(math.ceil(math.log2(d)))),
        'ancilla_qubits': max(1, int(math.ceil(math.log2(max(row_nnz, 2))))),
    }
