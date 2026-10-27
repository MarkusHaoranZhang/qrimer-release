"""
Sparse column-to-row oracle for the ER synthesis matrix B.

Every column of B has at most one nonzero entry.  This module builds the
associated reversible oracle for small N and reports the sparsity resources
used by the sparse block-encoding route.  The oracle is an explicit
permutation on column and row registers; the full block encoding composes it
with the row-side LCU state preparation.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from .er_matrices import build_b_matrix


class SparseEROracle:
    def __init__(self, N: int):
        self.N = N
        self.d = N + 2
        self.B = build_b_matrix(N)
        self.n_col = int(np.ceil(np.log2(self.d * self.d)))
        self.n_row = int(np.ceil(np.log2(self.d + 1)))
        self.col_to_row = []
        for c in range(self.d * self.d):
            rows = np.flatnonzero(self.B[:, c] > 0.5)
            if len(rows) > 1:
                raise ValueError("column has more than one nonzero")
            self.col_to_row.append(int(rows[0]) if len(rows) else self.d)
        self.nonzero_columns = sum(r < self.d for r in self.col_to_row)

    def oracle_unitary(self) -> np.ndarray:
        col_dim = 1 << self.n_col
        row_dim = 1 << self.n_row
        dim = col_dim * row_dim
        U = np.eye(dim, dtype=complex)
        for c, row in enumerate(self.col_to_row):
            if row == 0:
                continue
            i = c * row_dim
            U[i, i] = 0.0
            U[i + row, i] = 1.0
            U[i, i + row] = 1.0
            U[i + row, i + row] = 0.0
        return U

    def oracle_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_col + self.n_row)
        qc.append(UnitaryGate(self.oracle_unitary(), label='O_B'),
                  range(self.n_col + self.n_row))
        return qc

    def resource_report(self) -> dict:
        col_degree = self.B.astype(bool).sum(axis=0)
        row_degree = self.B.astype(bool).sum(axis=1)
        return {
            'N': self.N,
            'column_qubits': self.n_col,
            'row_qubits': self.n_row,
            'nonzero_columns': int(self.nonzero_columns),
            'max_column_degree': int(col_degree.max()),
            'max_row_degree': int(row_degree.max()),
            'nonzeros': int(self.B.astype(bool).sum()),
        }
