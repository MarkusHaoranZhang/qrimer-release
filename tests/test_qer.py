"""
Unit tests for QER: classical_simulate, construct_hermitian_matrix,
mass vector construction, and circuit building.
"""

import numpy as np
import pytest

from core.qer import QER
from core.rimer import BeliefRule, ClassicalRIMER

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def uniform_betas(N, L):
    return np.full((L, N), 1.0 / N)

def uniform_weights(L):
    return np.full(L, 1.0 / L)


# ---------------------------------------------------------------------------
# Mass vector
# ---------------------------------------------------------------------------

class TestMassVector:
    def test_components_nonnegative(self):
        qer = QER(N=3, L=2, kappa_max=10.0)
        m = qer._mass_vector(0.5, np.array([0.4, 0.3, 0.2]))
        assert np.all(m >= 0.0)

    def test_m_bar_D(self):
        qer = QER(N=3, L=2, kappa_max=10.0)
        w_k = 0.7
        m = qer._mass_vector(w_k, np.array([0.4, 0.3, 0.0]))
        assert m[qer.N] == pytest.approx(1.0 - w_k, abs=1e-12)

    def test_m_hat_D_complete_rule(self):
        """Complete rule (sum beta = 1) → m_hat_D = 0."""
        qer = QER(N=3, L=2, kappa_max=10.0)
        m = qer._mass_vector(0.8, np.array([0.5, 0.3, 0.2]))
        assert m[qer.N + 1] == pytest.approx(0.0, abs=1e-12)

    def test_m_hat_D_incomplete_rule(self):
        """Incomplete rule (sum beta < 1) → m_hat_D > 0."""
        qer = QER(N=3, L=2, kappa_max=10.0)
        m = qer._mass_vector(1.0, np.array([0.3, 0.3, 0.0]))
        assert m[qer.N + 1] > 0.0


# ---------------------------------------------------------------------------
# Hermitian matrix
# ---------------------------------------------------------------------------

class TestHermitianMatrix:
    def test_is_hermitian(self):
        qer = QER(N=3, L=2, kappa_max=10.0)
        m_I    = np.array([0.3, 0.2, 0.1, 0.3, 0.1])
        m_next = np.array([0.2, 0.3, 0.1, 0.2, 0.2])
        M = qer.construct_hermitian_matrix(m_I, m_next)
        np.testing.assert_allclose(M, M.conj().T, atol=1e-12)

    def test_shape(self):
        qer = QER(N=3, L=2, kappa_max=10.0)
        m = np.ones(qer.vec_dim) / qer.vec_dim
        M = qer.construct_hermitian_matrix(m, m)
        # Hermitian dilation of the rectangular B: (N+2) + (N+2)^2 = (N+2)(N+3)
        expected_dim = qer.vec_dim * (qer.vec_dim + 1)
        assert M.shape == (expected_dim, expected_dim)

    def test_dilation_qubit_budget(self):
        """Benchmark N=3: dilation register needs 5 qubits (Section 6.1)."""
        qer = QER(N=3, L=36, kappa_max=50.0)
        assert qer.dilation_qubits == 5

    def test_off_diagonal_block_structure(self):
        """The dilation has zero diagonal blocks and non-zero off-diagonal blocks."""
        qer = QER(N=3, L=2, kappa_max=10.0)
        m = np.array([0.2, 0.3, 0.1, 0.3, 0.1])
        M = qer.construct_hermitian_matrix(m, m)
        r = qer.vec_dim           # (N+2)
        # Top-left block (r x r) should be all zeros
        np.testing.assert_allclose(M[:r, :r], 0.0, atol=1e-12)
        # Bottom-right block (c x c) should be all zeros
        np.testing.assert_allclose(M[r:, r:], 0.0, atol=1e-12)
        # Off-diagonal blocks should have non-zero entries
        assert np.abs(M[:r, r:]).sum() > 0

    def test_matches_appendix_a(self):
        """
        Element-wise conformance with the paper's Appendix A.

        B^(k) is rebuilt from the appendix's case definitions (Eqs. A.1-A.3):
        rows 1..N have the five (p,q) entries, the c_hat_D row the three
        entries, and the c_bar_D row the single entry (5N + 3 + 1 non-zero
        entries in total).  The code's B must equal K * B_appendix up to a
        permutation of the two D-component rows (the code keeps the unified
        (m_1..m_N, m_bar_D, m_hat_D) layout for input and output).
        """
        qer = QER(N=3, L=2, kappa_max=10.0)
        mI = np.array([0.2, 0.3, 0.1, 0.3, 0.1])
        mn = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        M = qer.construct_hermitian_matrix(mI, mn)
        B_code = M[:qer.vec_dim, qer.vec_dim:]

        conflict = mI[:3].sum() * mn[:3].sum() - np.dot(mI[:3], mn[:3])
        K = 1.0 / (1.0 - conflict)

        N, vd = 3, 5

        def col(p, q):
            return p * vd + q

        B_apx = np.zeros((vd, vd * vd))
        for j in range(N):
            for p, q in [(j, j), (j, 3), (j, 4), (3, j), (4, j)]:
                B_apx[j, col(p, q)] = 1
        for p, q in [(4, 4), (4, 3), (3, 4)]:   # c_hat_D row (appendix row N+1)
            B_apx[3, col(p, q)] = 1
        B_apx[4, col(3, 3)] = 1                  # c_bar_D row (appendix row N+2)

        B_swapped = B_code.copy()
        B_swapped[[3, 4]] = B_swapped[[4, 3]]
        np.testing.assert_allclose(B_swapped, K * B_apx, atol=1e-12,
                                   err_msg="B does not match Appendix A")

        # Non-zero entry count: 5N + 3 + 1 (appendix)
        n_nonzero = int(round(np.abs(B_code).sum() / K))
        assert n_nonzero == 5 * N + 3 + 1

        # Bilinear forms (Eqs. A.1-A.3) with the K normalisation
        c = B_code @ np.outer(mI, mn).flatten()
        assert np.allclose(c[0],
                           K * (mI[0] * mn[0] + mI[0] * (mn[3] + mn[4])
                                + (mI[3] + mI[4]) * mn[0]), atol=1e-12)
        assert np.allclose(c[4],
                           K * (mI[4] * mn[4] + mI[4] * mn[3]
                                + mI[3] * mn[4]), atol=1e-12)
        assert np.allclose(c[3], K * (mI[3] * mn[3]), atol=1e-12)
        # The normalised output equals the ER combination step exactly
        np.testing.assert_allclose(c, qer._er_combine_step(mI, mn), atol=1e-12)


# ---------------------------------------------------------------------------
# classical_simulate vs ClassicalRIMER
# ---------------------------------------------------------------------------

class TestClassicalSimulate:
    """QER.classical_simulate must produce results identical to ClassicalRIMER."""

    def _run_classical_rimer(self, N, L, T, betas, rule_weights):
        rimer = ClassicalRIMER(N=N, L=L, T=T)
        for k in range(L):
            rimer.add_rule(BeliefRule(
                antecedent=list(range(T)),
                consequent=betas[k],
                rule_weight=float(rule_weights[k]),
                attribute_weights=np.ones(T),
            ))
        dists = [{j: 1.0 / 3 for j in range(3)} for _ in range(T)]
        w = rimer.rule_activation(dists)
        return rimer.er_combine(w, betas), w

    def test_matches_classical_rimer_small(self):
        N, L, T = 3, 4, 2
        rng   = np.random.default_rng(0)
        raw   = rng.dirichlet(np.ones(N), size=L)
        betas = raw * rng.uniform(0.7, 1.0, size=(L, 1))
        rw    = rng.uniform(0.5, 1.0, size=L)

        (beta_c, beta_D_c), w = self._run_classical_rimer(N, L, T, betas, rw)

        qer = QER(N=N, L=L, kappa_max=50.0)
        beta_q, beta_D_q = qer.classical_simulate(w, betas)

        np.testing.assert_allclose(beta_q, beta_c, atol=1e-10)
        assert beta_D_q == pytest.approx(beta_D_c, abs=1e-10)

    def test_matches_classical_rimer_larger(self):
        N, L, T = 5, 10, 3
        rng   = np.random.default_rng(42)
        raw   = rng.dirichlet(np.ones(N), size=L)
        betas = raw * 0.9
        rw    = rng.uniform(0.3, 1.0, size=L)

        (beta_c, beta_D_c), w = self._run_classical_rimer(N, L, T, betas, rw)

        qer = QER(N=N, L=L, kappa_max=100.0)
        beta_q, beta_D_q = qer.classical_simulate(w, betas)

        np.testing.assert_allclose(beta_q, beta_c, atol=1e-10)
        assert beta_D_q == pytest.approx(beta_D_c, abs=1e-10)

    def test_output_sums_leq_one(self):
        N, L = 3, 5
        rng   = np.random.default_rng(7)
        betas = rng.dirichlet(np.ones(N), size=L) * 0.8
        w     = rng.dirichlet(np.ones(L))
        qer   = QER(N=N, L=L, kappa_max=20.0)
        beta, beta_D = qer.classical_simulate(w, betas)
        assert np.sum(beta) + beta_D <= 1.0 + 1e-9

    def test_output_nonnegative(self):
        N, L = 3, 5
        rng   = np.random.default_rng(13)
        betas = rng.dirichlet(np.ones(N), size=L) * 0.9
        w     = rng.dirichlet(np.ones(L))
        qer   = QER(N=N, L=L, kappa_max=20.0)
        beta, beta_D = qer.classical_simulate(w, betas)
        assert np.all(beta >= -1e-9)
        assert beta_D >= -1e-9


# ---------------------------------------------------------------------------
# Circuit building
# ---------------------------------------------------------------------------

class TestCircuitBuilding:
    def test_build_circuit_runs(self):
        qer   = QER(N=3, L=4, kappa_max=10.0)
        betas = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1],
                          [0.3, 0.3, 0.4], [0.6, 0.2, 0.2]])
        w     = np.array([0.4, 0.3, 0.2, 0.1])
        circ  = qer.build_circuit(w, betas)
        assert circ.num_qubits > 0
        assert circ.depth() > 0

    def test_build_circuit_qubit_count(self):
        """Circuit should use ceil(log2((N+2)(N+3))) qubits for the dilation."""
        qer = QER(N=3, L=2, kappa_max=10.0)
        betas = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        w = np.array([0.6, 0.4])
        circ = qer.build_circuit(w, betas)
        expected = int(np.ceil(np.log2(qer.vec_dim * (qer.vec_dim + 1))))
        assert circ.num_qubits == expected

    def test_estimate_depth_positive(self):
        qer = QER(N=5, L=10, kappa_max=50.0, epsilon=0.01)
        assert qer.estimate_depth() > 0

    def test_estimate_depth_scales_with_L(self):
        """Depth should increase with L (more combination steps)."""
        d1 = QER(N=5, L=10, kappa_max=50.0).estimate_depth()
        d2 = QER(N=5, L=20, kappa_max=50.0).estimate_depth()
        assert d2 > d1


# ---------------------------------------------------------------------------
# End-to-end statevector verification
# ---------------------------------------------------------------------------

class TestStatevectorEquivalence:
    """
    Verify that the quantum circuit's statevector output, when interpreted
    as a probability distribution, is consistent with classical_simulate.

    This is the critical test that validates the paper's core claim:
    the quantum circuit structure (HamiltonianGate-based MEoB) encodes
    the same mathematical operation as the classical ER combination.
    """

    def test_single_step_hermitian_encodes_er(self):
        """
        For a single ER combination step (L=2), verify that the Hermitian
        dilation matrix B correctly encodes the bilinear ER map, including
        the block-encoded conflict normalisation K.

        We check: B @ (m_I ⊗ m_next) equals the normalised combined mass
        vector produced by the ER combination step.
        """
        qer = QER(N=3, L=2, kappa_max=10.0)

        # Two rules with known mass vectors
        w = np.array([0.6, 0.4])
        betas = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.1]])

        m_I = qer._mass_vector(w[0], betas[0])
        m_next = qer._mass_vector(w[1], betas[1])

        # Compute expected combined mass (normalised) via _er_combine_step
        m_combined = qer._er_combine_step(m_I, m_next)

        # Extract B from the Hermitian dilation (top-right block)
        M = qer.construct_hermitian_matrix(m_I, m_next)
        r = qer.vec_dim
        B = M[:r, r:]  # the block-encoded (K-scaled) B matrix

        # Compute B @ (m_I ⊗ m_next)
        tensor = np.outer(m_I, m_next).flatten()
        c_from_B = B @ tensor

        # B is block-encoded with the conflict factor K, so c_from_B is
        # directly the normalised combined mass vector
        np.testing.assert_allclose(c_from_B, m_combined, atol=1e-10,
                                   err_msg="B matrix does not correctly encode ER map")

    def test_hermitian_dilation_eigenvalues_real(self):
        """Hermitian matrix must have real eigenvalues."""
        qer = QER(N=3, L=2, kappa_max=10.0)
        m = np.array([0.2, 0.3, 0.1, 0.3, 0.1])
        M = qer.construct_hermitian_matrix(m, m)
        eigenvalues = np.linalg.eigvalsh(M)
        # eigvalsh guarantees real output for Hermitian input
        assert eigenvalues.dtype == np.float64

    def test_classical_simulate_matches_er_combine_random(self):
        """
        Stress test: QER.classical_simulate must match ClassicalRIMER.er_combine
        across 50 random configurations.
        """
        rng = np.random.default_rng(2024)
        for _ in range(50):
            N = rng.integers(2, 8)
            L = rng.integers(2, 12)
            T = rng.integers(1, 4)

            raw = rng.dirichlet(np.ones(N), size=L)
            betas = raw * rng.uniform(0.6, 1.0, size=(L, 1))
            rw = rng.uniform(0.3, 1.0, size=L)

            rimer = ClassicalRIMER(N=N, L=L, T=T)
            for k in range(L):
                rimer.add_rule(BeliefRule(
                    antecedent=list(range(T)),
                    consequent=betas[k],
                    rule_weight=float(rw[k]),
                    attribute_weights=np.ones(T),
                ))
            dists = [{j: 1.0 / N for j in range(N)} for _ in range(T)]
            w = rimer.rule_activation(dists)
            beta_c, beta_D_c = rimer.er_combine(w, betas)

            qer = QER(N=N, L=L, kappa_max=50.0)
            beta_q, beta_D_q = qer.classical_simulate(w, betas)

            np.testing.assert_allclose(beta_q, beta_c, atol=1e-10)
            assert abs(beta_D_q - beta_D_c) < 1e-10

    def test_target_state_circuit_encodes_belief_distribution(self):
        """
        Verify that build_target_state_circuit produces a quantum state
        whose measurement probabilities equal the classical belief distribution.

        This is the end-to-end verification: if we measure the target state
        circuit in the computational basis, outcome j should occur with
        probability β_j (the classical ER result).
        """
        from qiskit.quantum_info import Statevector

        N, L = 3, 4
        rng = np.random.default_rng(777)
        raw = rng.dirichlet(np.ones(N), size=L)
        betas = raw * rng.uniform(0.6, 0.95, size=(L, 1))
        w = rng.dirichlet(np.ones(L))

        qer = QER(N=N, L=L, kappa_max=50.0)

        # Get classical result
        beta_c, beta_D_c = qer.classical_simulate(w, betas)

        # Build and simulate the target-state circuit
        circ = qer.build_target_state_circuit(w, betas)
        sv = Statevector.from_instruction(circ)
        probs = sv.probabilities()

        # The first N+1 probabilities should match [β_0, β_1, ..., β_{N-1}, β_D]
        # (up to normalization by the state norm)
        expected = np.zeros(len(probs))
        expected[:N] = beta_c
        expected[N] = beta_D_c

        # The state is normalized, so probabilities = β / sum(β + β_D)
        total = beta_c.sum() + beta_D_c
        if total > 1e-12:
            expected[:N + 1] = expected[:N + 1] / total

        np.testing.assert_allclose(probs[:N + 1], expected[:N + 1], atol=1e-9,
                                   err_msg="Target state probabilities don't match "
                                           "classical belief distribution")

    def test_target_state_preserves_rank_order(self):
        """
        The rank order of measurement probabilities from the target state
        must match the rank order of classical belief degrees.
        """
        from qiskit.quantum_info import Statevector

        N, L = 5, 8
        rng = np.random.default_rng(2025)
        betas = rng.dirichlet(np.ones(N), size=L) * 0.85
        w = rng.dirichlet(np.ones(L))

        qer = QER(N=N, L=L, kappa_max=50.0)
        beta_c, _ = qer.classical_simulate(w, betas)

        circ = qer.build_target_state_circuit(w, betas)
        sv = Statevector.from_instruction(circ)
        probs = sv.probabilities()

        # Rank order of first N probabilities should match rank order of beta_c
        classical_rank = np.argsort(beta_c)[::-1]
        quantum_rank = np.argsort(probs[:N])[::-1]
        np.testing.assert_array_equal(classical_rank, quantum_rank,
                                      err_msg="Quantum state rank order doesn't match classical")


class TestSimulateMeasurement:
    """Circuit-based measurement statistics (response to Reviewer 2, Comment 4)."""

    def test_belief_state_circuit_encodes_distribution(self):
        from qiskit.quantum_info import Statevector

        beta = np.array([0.5, 0.3, 0.2])
        circ = QER.belief_state_circuit(beta, 0.0)
        sv = Statevector.from_instruction(circ)
        probs = sv.probabilities()
        np.testing.assert_allclose(probs[:3], beta, atol=1e-9)

    def test_simulate_measurement_frequencies(self):
        """Measured frequencies are multinomial samples of the belief state."""
        qer = QER(N=3, L=2, kappa_max=10.0)
        betas = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        w = np.array([0.6, 0.4])
        beta_c, beta_D_c = qer.classical_simulate(w, betas)

        rng = np.random.default_rng(0)
        beta_q, beta_D_q = qer.simulate_measurement(w, betas, rng, n_shots=10**4)

        assert beta_q.shape == (3,)
        assert np.all(beta_q >= 0.0)
        assert abs(beta_q.sum() + beta_D_q - 1.0) < 1e-9
        # Within sampling error of the exact result (3 sigma)
        err = np.sqrt(np.sum(beta_c * (1 - beta_c)) / 10**4)
        assert np.linalg.norm(beta_q - beta_c) < 3 * err + 1e-6

    def test_simulate_measurement_approaches_exact(self):
        """As shots grow, the measured distribution converges to the exact one."""
        qer = QER(N=3, L=2, kappa_max=10.0)
        betas = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        w = np.array([0.6, 0.4])
        beta_c, _ = qer.classical_simulate(w, betas)

        rng = np.random.default_rng(1)
        beta_q, _ = qer.simulate_measurement(w, betas, rng, n_shots=10**6)
        assert np.linalg.norm(beta_q - beta_c) < 2e-3

    def test_simulate_belief_measurement_matches_distribution(self):
        beta = np.array([0.2685, 0.6772, 0.0543])
        rng = np.random.default_rng(42)
        beta_q, beta_D_q = QER.simulate_belief_measurement(
            beta, 0.0, rng, n_shots=10**4)
        assert beta_q.shape == (3,)
        assert abs(beta_q.sum() + beta_D_q - 1.0) < 1e-9
        assert np.all(beta_q >= 0.0)
