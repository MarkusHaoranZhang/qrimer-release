"""Structured QER circuit construction.

The paper's full QER circuit is not a single dense unitary on the belief
register.  This module builds an explicit register-level circuit around the
validated Hermitian block encoding:

- a dilation/state register,
- a rule-index register,
- an antecedent register,
- a phase-estimation clock register,
- a block-encoding ancilla.

Only the dilation register receives the block-encoding gate; the remaining
registers are explicit resource registers.  The one-step circuit is
transpilable for resource measurement, while the full L-step depth is
reported as a step-count estimate rather than silently presented as a
compiled circuit.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate

from .block_encoding import HermitianERBlockEncoding


class StructuredQER:
    def __init__(self, N: int, L: int, T: int = 4,
                 clock_qubits: int | None = None):
        self.N = N
        self.L = L
        self.T = T
        self.encoder = HermitianERBlockEncoding(N)
        self.n_state = self.encoder.n_total
        self.n_rule = max(1, int(np.ceil(np.log2(max(L, 2)))))
        self.n_ant = T
        self.n_anc = 1
        self.n_clock = (10 if clock_qubits is None else int(clock_qubits))
        self.total_qubits = (
            self.n_state + self.n_rule + self.n_ant
            + self.n_clock + self.n_anc
        )

    def step_circuit(self, m_I: np.ndarray, m_next: np.ndarray,
                     unitary: np.ndarray | None = None) -> QuantumCircuit:
        """Build one explicit block-encoded step with all registers."""
        qr = QuantumRegister(self.total_qubits, 'q')
        qc = QuantumCircuit(qr)
        psi, _ = self.encoder._prepare_input(m_I, m_next)
        state_qubits = list(range(self.n_state))
        qc.initialize(psi, state_qubits)
        u = self.encoder.U if unitary is None else unitary
        qc.append(UnitaryGate(u, label='U_BE'), state_qubits)
        return qc

    def measured_depth(self, m_I: np.ndarray, m_next: np.ndarray,
                       basis=('u', 'cx'), optimization_level: int = 0) -> dict:
        """Transpile one structured step and report resource counts."""
        fallback = False
        try:
            qc = self.step_circuit(m_I, m_next)
            trans = transpile(qc, basis_gates=list(basis),
                              optimization_level=optimization_level)
        except BaseException:
            # Qiskit's QSD backend has an unwrap panic on some gauges of
            # the same unitary.  A row-phase gauge preserves the unitary up
            # to a basis-dependent phase and gives a representative resource
            # count; record the fallback explicitly.
            rng = np.random.default_rng(15)
            phases = np.exp(1j * 1e-12 * rng.normal(size=self.encoder.U.shape[0]))
            u_gauge = phases[:, None] * self.encoder.U
            qc = self.step_circuit(m_I, m_next, unitary=u_gauge)
            trans = transpile(qc, basis_gates=list(basis),
                              optimization_level=optimization_level)
            fallback = True
        steps = max(self.L - 1, 1)
        return {
            'N': self.N,
            'L': self.L,
            'qubits': self.total_qubits,
            'state_qubits': self.n_state,
            'rule_qubits': self.n_rule,
            'antecedent_qubits': self.n_ant,
            'clock_qubits': self.n_clock,
            'ancilla_qubits': self.n_anc,
            'step_depth': trans.depth(),
            'step_size': trans.size(),
            'steps': steps,
            'estimated_full_depth': trans.depth() * steps,
            'basis': list(basis),
            'optimization_level': optimization_level,
            'synthesis_fallback': fallback,
        }
