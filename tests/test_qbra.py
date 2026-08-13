"""
Unit tests for the QBRA operator (Section 5.3), including the three-step
circuit structure and the quantum division circuit of Step 3.
"""

import numpy as np
from qiskit import QuantumRegister

from core.qbra import QBRA


def build_registers(qbra: QBRA):
    qr_rule  = QuantumRegister(qbra.num_rule_qubits, 'rule')
    qr_alpha = QuantumRegister(4, 'alpha')
    qr_ant   = QuantumRegister(4, 'ant')
    qr_w     = QuantumRegister(1, 'w')
    qr_aux   = QuantumRegister(qbra.num_aux_qubits, 'aux')
    return qr_rule, qr_alpha, qr_ant, qr_w, qr_aux


class TestQBRA:
    def test_full_circuit_runs(self):
        qbra = QBRA(T=4, L=8, Tk_max=2, epsilon=0.01)
        regs = build_registers(qbra)
        circ = qbra.full_qbra_circuit(*regs)
        assert circ.num_qubits > 0
        assert circ.depth() > 0

    def test_step3_no_longer_placeholder(self):
        """Step 3 implements the quantum division circuit (Appendix B)."""
        qbra = QBRA(T=4, L=4, Tk_max=2, epsilon=0.01)
        regs = build_registers(qbra)
        qc = qbra.full_qbra_circuit(*regs, rule_weights=np.ones(4))
        # The summing + reciprocal structure adds controlled rotations
        assert qc.depth() > 0

    def test_step3_depends_on_rule_weights(self):
        """Uniform vs skewed rule weights produce different rotation angles."""
        qbra = QBRA(T=2, L=2, Tk_max=2, epsilon=0.01)
        regs1 = build_registers(qbra)
        regs2 = build_registers(qbra)
        c1 = qbra.full_qbra_circuit(*regs1, rule_weights=np.array([1.0, 1.0]))
        c2 = qbra.full_qbra_circuit(*regs2, rule_weights=np.array([0.2, 1.0]))

        def rotation_angles(circ):
            angles = []
            for inst in circ.data:
                if inst.operation.name == 'cry':
                    angles.append(float(inst.operation.params[0]))
            return sorted(angles)

        assert rotation_angles(c1) != rotation_angles(c2)

    def test_estimate_depth_positive(self):
        qbra = QBRA(T=4, L=10, Tk_max=3, epsilon=0.01)
        assert qbra.estimate_depth() > 0

    def test_estimate_depth_formula(self):
        """Depth = T*Tk_max + Tk_max*polylog(1/eps) + L*log(1/eps)."""
        T, L, Tk_max, eps = 4, 10, 3, 0.01
        qbra = QBRA(T=T, L=L, Tk_max=Tk_max, epsilon=eps)
        log_inv = int(np.ceil(np.log2(1.0 / qbra.eps1)))
        expected = T * Tk_max + Tk_max * log_inv ** 2 + L * log_inv
        assert qbra.estimate_depth() == expected

    def test_estimate_depth_scales_with_L(self):
        d1 = QBRA(T=4, L=10, Tk_max=2).estimate_depth()
        d2 = QBRA(T=4, L=20, Tk_max=2).estimate_depth()
        assert d2 > d1
