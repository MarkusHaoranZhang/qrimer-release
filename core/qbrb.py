"""
Quantum Belief Rule Base (Q-BRB) Encoding.
Implements Section 5.1 of the paper.

Prepares |R_k> and the full |BRB> superposition state.
Compatible with Qiskit >= 1.0 (uses StatePreparation instead of initialize).
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation


class QBRB:
    """
    Quantum Belief Rule Base encoder.

    Parameters
    ----------
    N : int  — number of evaluation grades
    L : int  — number of belief rules
    T : int  — number of antecedent attributes
    """

    def __init__(self, N: int, L: int, T: int):
        self.N = N
        self.L = L
        self.T = T
        self.num_cons_qubits = int(np.ceil(np.log2(self.N + 1)))
        self.num_rule_qubits = max(1, int(np.ceil(np.log2(self.L))))
        self.num_ant_qubits = self.T

    def _state_prep_gate(self, amplitudes: np.ndarray, label: str = 'SP'):
        """
        Build a StatePreparation gate for the given amplitude vector.
        Pads to the next power of 2 and normalises.
        """
        n_qubits = int(np.ceil(np.log2(len(amplitudes)))) if len(amplitudes) > 1 else 1
        dim = 2 ** n_qubits
        padded = np.zeros(dim)
        padded[:len(amplitudes)] = amplitudes
        norm = np.linalg.norm(padded)
        if norm > 1e-12:
            padded = padded / norm
        return StatePreparation(padded.tolist(), label=label)

    def prepare_single_rule(self, rule_idx: int, betas: np.ndarray,
                            antecedent_mask: np.ndarray) -> QuantumCircuit:
        """
        Prepare the quantum state for a single rule |R_k>.

        Consequent register encodes sqrt(beta_j) as amplitudes, so that
        measuring qubit j yields probability beta_j (Born rule).
        Antecedent register encodes which attributes are active (|1> = active).
        """
        qr_cons = QuantumRegister(self.num_cons_qubits, f'cons{rule_idx}')
        qr_ant  = QuantumRegister(self.num_ant_qubits,  f'ant{rule_idx}')
        qc = QuantumCircuit(qr_cons, qr_ant, name=f'R{rule_idx}')

        # Consequent: encode sqrt(beta_j) as amplitudes so that measuring
        # the consequent register yields outcome j with probability beta_j.
        p = np.zeros(self.N + 1)
        p[:self.N] = np.sqrt(np.maximum(betas, 0.0))
        p[self.N]  = np.sqrt(max(0.0, 1.0 - np.sum(np.maximum(betas, 0.0))))
        sp = self._state_prep_gate(p, label=f'SP_cons{rule_idx}')
        qc.append(sp, qr_cons)

        # Antecedent: |1> if attribute is used in this rule
        for i in range(self.T):
            if antecedent_mask[i] == 1:
                qc.x(qr_ant[i])

        return qc

    def prepare_full_brb(self, betas_list: np.ndarray,
                         ant_masks: np.ndarray) -> QuantumCircuit:
        """
        Prepare the full |BRB> = (1/sqrt(L)) sum_k |k>|R_k> state.

        Parameters
        ----------
        betas_list : np.ndarray of shape (L, N)
        ant_masks  : np.ndarray of shape (L, T)
        """
        qr_rule = QuantumRegister(self.num_rule_qubits, 'rule')
        qr_cons = QuantumRegister(self.num_cons_qubits, 'cons')
        qr_ant  = QuantumRegister(self.num_ant_qubits,  'ant')
        qc = QuantumCircuit(qr_rule, qr_cons, qr_ant, name='BRB')

        # Uniform superposition over rule index register
        qc.h(qr_rule)

        for k in range(self.L):
            rule_circ = self.prepare_single_rule(k, betas_list[k], ant_masks[k])
            # Convert to gate (StatePreparation is gate-compatible in Qiskit 2.x)
            rule_gate = rule_circ.to_gate(label=f'R{k}')
            ctrl_gate = rule_gate.control(self.num_rule_qubits, label=f'cR{k}')

            # Flip 0-bits so the control fires on |k>
            binary = format(k, f'0{self.num_rule_qubits}b')
            for i, bit in enumerate(reversed(binary)):
                if bit == '0':
                    qc.x(qr_rule[i])

            qc.append(ctrl_gate, list(qr_rule) + list(qr_cons) + list(qr_ant))

            # Unflip
            for i, bit in enumerate(reversed(binary)):
                if bit == '0':
                    qc.x(qr_rule[i])

        return qc

    def estimate_depth(self) -> int:
        """
        Estimated circuit depth: O(L * T) per paper Section 5.1.
        +1 for the initial Hadamard layer.
        """
        return 1 + self.L * self.T
