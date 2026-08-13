"""
Unit tests for QBRB: circuit construction and depth estimation.
"""

import numpy as np

from core.qbrb import QBRB


class TestQBRB:
    def test_prepare_single_rule_runs(self):
        q = QBRB(N=3, L=4, T=2)
        betas = np.array([0.5, 0.3, 0.2])
        mask  = np.array([1, 0])
        circ  = q.prepare_single_rule(0, betas, mask)
        assert circ.num_qubits == q.num_cons_qubits + q.num_ant_qubits

    def test_prepare_full_brb_runs(self):
        q      = QBRB(N=3, L=4, T=2)
        betas  = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1],
                           [0.3, 0.3, 0.4], [0.6, 0.2, 0.2]])
        masks  = np.array([[1, 1], [1, 0], [0, 1], [1, 1]])
        circ   = q.prepare_full_brb(betas, masks)
        assert circ.num_qubits == q.num_rule_qubits + q.num_cons_qubits + q.num_ant_qubits
        assert circ.depth() > 0

    def test_num_qubits_correct(self):
        q = QBRB(N=5, L=8, T=3)
        assert q.num_cons_qubits == int(np.ceil(np.log2(6)))  # ceil(log2(N+1))
        assert q.num_rule_qubits == 3                          # ceil(log2(8))
        assert q.num_ant_qubits  == 3

    def test_estimate_depth_positive(self):
        q = QBRB(N=3, L=10, T=4)
        assert q.estimate_depth() > 0

    def test_estimate_depth_formula(self):
        """Depth = 1 + L * T per Section 5.1."""
        q = QBRB(N=3, L=6, T=3)
        assert q.estimate_depth() == 1 + 6 * 3

    def test_incomplete_betas_handled(self):
        """Betas summing to < 1 should not raise."""
        q     = QBRB(N=3, L=2, T=1)
        betas = np.array([[0.3, 0.2, 0.1], [0.4, 0.1, 0.1]])
        masks = np.array([[1], [0]])
        circ  = q.prepare_full_brb(betas, masks)
        assert circ is not None
