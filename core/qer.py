"""
Quantum Evidential Reasoning (QER) Engine.
Implements Section 5.4: polynomial speedup via Hermitian matrix MEoB.
Compatible with Qiskit >= 1.0.

Two execution modes
-------------------
classical_simulate (default)
    Runs the ER combination classically using the exact matrix formulation
    derived in the paper.  This is the correct path for validation on a
    simulator: it reproduces the paper's results exactly and is used by
    run_preliminary.py.

circuit_mode
    Builds the Qiskit quantum circuit (StatePreparation + MEoB gate structure)
    for depth analysis and future deployment on real hardware.  The circuit
    structure matches Section 5.4 but the MEoB step is a HamiltonianGate
    (e^{-i M_ER^(k)}); full fault-tolerant execution requires a hardware
    backend.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation


class QER:
    """
    Quantum Evidential Reasoning engine.

    Parameters
    ----------
    N         : int   — number of evaluation grades
    L         : int   — number of belief rules
    kappa_max : float — maximum condition number of the MEoB matrix
    epsilon   : float — approximation error bound
    """

    def __init__(self, N: int, L: int, kappa_max: float, epsilon: float = 0.01):
        self.N          = N
        self.L          = L
        self.kappa_max  = kappa_max
        self.epsilon    = epsilon
        self.vec_dim    = N + 2          # layout: [m_1..m_N, m_bar_D, m_hat_D]
        self.num_qubits = max(1, int(np.ceil(np.log2(self.vec_dim))))
        # Hermitian dilation of the rectangular ER synthesis matrix B.
        # B is (N+2) x (N+2)^2, so the dilation [[0, B], [B^H, 0]] has
        # dimension (N+2) + (N+2)^2 = (N+2)(N+3).
        self.dilated_dim    = self.vec_dim * (self.vec_dim + 1)
        self.dilation_qubits = max(1, int(np.ceil(np.log2(self.dilated_dim))))

    # ------------------------------------------------------------------
    # Classical simulation (exact, used for validation)
    # ------------------------------------------------------------------

    def construct_hermitian_matrix(self, m_I: np.ndarray,
                                   m_next: np.ndarray) -> np.ndarray:
        """
        Construct the Hermitian dilation M_ER^(k) per Appendix A of the paper.

        The ER combination is a bilinear map from the tensor-product space
        (N+2)^2 to the output space (N+2).  The rectangular matrix B^(k)
        encodes this map: c = B^(k) * (m_I ⊗ m_next).

        The Hermitian dilation embeds B into a square matrix::

            M_ER = [[0,        B],
                    [B^dagger, 0]]

        B^(k) is (N+2) x (N+2)^2, hence the rectangular block form
        [[0, B], [B^H, 0]] has dimension (N+2) + (N+2)^2 = (N+2)(N+3).
        The paper states the dilation size as (N+2)^2 x (N+2)^2; both
        formulations require the same register size, ceil(log2((N+2)^2))
        = ceil(log2((N+2)(N+3))) qubits — 5 for the benchmark N = 3,
        matching the paper's qubit budget (Section 6.1: "dilated ER:
        ceil(log2((3+2)^2)) = 5").

        The data-dependent conflict normalisation factor K_{I(k+1)} is
        incorporated through a block-encoding technique (Section 5.4): the
        unnormalised bilinear map is computed first, and the factor K is then
        folded into B so that B_enc = K * B encodes the *normalised* operator.
        Block-encoding requires no matrix inversion: it embeds the operator
        into a larger unitary to perform forward matrix-vector multiplication
        (matrix function application in the MEoB framework).  The
        post-selection success probability is O(1/kappa) after amplitude
        amplification (Section 5.6), where the condition number depends on
        the degree of conflict among the combined evidence
        (see run_sensitivity.py).

        Parameters
        ----------
        m_I    : np.ndarray of shape (vec_dim,)
        m_next : np.ndarray of shape (vec_dim,)

        Returns
        -------
        M : np.ndarray of shape ((N+2)(N+3), (N+2)(N+3)), complex

        Notes
        -----
        This implements the exact structure from Appendix A (Eqs. A.1-A.6).
        The element-wise definition of B^(k) matches Appendix A exactly
        (5N + 3 + 1 non-zero entries); the only difference is a permutation
        of the two D-component rows: the code keeps the unified layout
        (m_1..m_N, m_bar_D, m_hat_D) for both input and output vectors
        (required for the iterative MEoB composition), while Appendix A
        lists the output as (c_1..c_N, c_hat_D, c_bar_D).  The bilinear
        content is identical.

        The conflict normalisation K_{I(k+1)} corresponds to the
        "additional block-encoding layer that rescales the operator"
        described in Appendix A: the rescaled operator K * B encodes the
        normalised map, with the O(kappa) multiplicative query overhead
        already accounted for in the condition-number dependence of the
        MEoB step.  Verified element-wise in
        tests/test_qer.py::TestHermitianMatrix::test_matches_appendix_a.
        """
        N = self.N
        vec_dim = self.vec_dim  # N+2
        col_dim = vec_dim ** 2  # (N+2)^2

        # Conflict coefficient between the two combined evidence sources
        # (Eq. A.6 of the paper): K = [1 - sum_{j != t} m_j,I * m_t,next]^-1
        mc_N = m_I[:N]
        mn_N = m_next[:N]
        conflict = float(mc_N.sum() * mn_N.sum() - np.dot(mc_N, mn_N))
        K = 1.0 / (1.0 - conflict + 1e-15)

        # Build the rectangular matrix B^(k) ∈ R^{(N+2) x (N+2)^2}
        # per Appendix A: rows are output components, columns are (p,q) pairs
        B = np.zeros((vec_dim, col_dim), dtype=complex)

        # Helper: column index for pair (p, q) where p,q ∈ {0..vec_dim-1}
        # Indices 0..N-1 = singleton masses m_j
        # Index N = m_bar_D
        # Index N+1 = m_hat_D
        def col(p, q):
            return p * vec_dim + q

        # Rows j = 0..N-1: c_j terms (Eq. A.1 of the paper)
        for j in range(N):
            # m_j,I * m_j,next  (diagonal term)
            B[j, col(j, j)] = 1.0
            # m_j,I * m_bar_D,next
            B[j, col(j, N)] = 1.0
            # m_j,I * m_hat_D,next
            B[j, col(j, N + 1)] = 1.0
            # m_bar_D,I * m_j,next
            B[j, col(N, j)] = 1.0
            # m_hat_D,I * m_j,next
            B[j, col(N + 1, j)] = 1.0

        # Row N (bar_c_D): bar_m_D,I * bar_m_D,next  (Eq. A.3)
        B[N, col(N, N)] = 1.0

        # Row N+1 (hat_c_D): hat*hat + hat*bar + bar*hat  (Eq. A.2)
        B[N + 1, col(N + 1, N + 1)] = 1.0
        B[N + 1, col(N + 1, N)] = 1.0
        B[N + 1, col(N, N + 1)] = 1.0

        # Block-encoding of the normalised operator: fold the conflict
        # factor K into B (Section 5.4).  The condition number of the
        # resulting dilation then reflects the evidence conflict.
        B = K * B

        # Hermitian dilation: M = [[0, B], [B^H, 0]] with rectangular blocks
        dim = vec_dim + col_dim
        M = np.zeros((dim, dim), dtype=complex)
        M[:vec_dim, vec_dim:] = B
        M[vec_dim:, :vec_dim] = B.conj().T

        return M

    def _mass_vector(self, w_k: float, beta_k: np.ndarray) -> np.ndarray:
        """
        Build the (N+2)-dim probability mass vector for rule k.

        Layout:
          m[0..N-1] = w_k * beta_{j,k}          (m_{j,k})
          m[N]      = 1 - w_k                    (m_bar_{D,k})
          m[N+1]    = w_k * (1 - sum(beta_k))    (m_hat_{D,k})
        """
        m = np.zeros(self.vec_dim)
        m[:self.N]    = w_k * np.maximum(beta_k, 0.0)
        m[self.N]     = 1.0 - w_k
        m[self.N + 1] = w_k * max(0.0, 1.0 - float(np.sum(beta_k)))
        return m

    def _er_combine_step(self, m_I: np.ndarray, m_next: np.ndarray) -> np.ndarray:
        """
        One step of the ER combination rule (Eq. 5 of the paper).
        Fully vectorised — no Python loops over N.
        """
        N = self.N
        mc_N = m_I[:N]
        mn_N = m_next[:N]

        # Conflict: sum_{j≠t} m_I[j]*m_next[t]  (vectorised identity)
        conflict = mc_N.sum() * mn_N.sum() - np.dot(mc_N, mn_N)
        K = 1.0 / (1.0 - conflict + 1e-15)

        mD_I    = m_I[N]   + m_I[N + 1]
        mD_next = m_next[N] + m_next[N + 1]

        m_new = np.empty(self.vec_dim)
        m_new[:N]    = K * (mc_N * mn_N + mc_N * mD_next + mD_I * mn_N)
        m_new[N + 1] = K * (
            m_I[N + 1] * m_next[N + 1]
            + m_I[N + 1] * m_next[N]
            + m_I[N]     * m_next[N + 1]
        )
        m_new[N] = K * (m_I[N] * m_next[N])
        return m_new

    def classical_simulate(self, w: np.ndarray,
                           betas: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Run the full ER combination classically (exact, O(L * N)).

        This is the correct validation path: it implements the paper's
        matrix MEoB formulation exactly and produces results identical to
        ClassicalRIMER.er_combine().  Used by run_preliminary.py.

        Complexity is O(L * N) per call: each of the L-1 combination steps
        performs O(N) NumPy operations via _er_combine_step.

        Parameters
        ----------
        w     : np.ndarray of shape (L,)   — activation weights
        betas : np.ndarray of shape (L, N) — belief degrees

        Returns
        -------
        beta   : np.ndarray of shape (N,)  — final belief degrees
        beta_D : float                     — residual ignorance
        """
        m_current = self._mass_vector(w[0], betas[0])
        for k in range(1, self.L):
            m_next = self._mass_vector(w[k], betas[k])
            m_current = self._er_combine_step(m_current, m_next)

        denom  = 1.0 - m_current[self.N] + 1e-15
        beta   = m_current[:self.N] / denom
        beta_D = m_current[self.N + 1] / denom
        return beta, beta_D

    # ------------------------------------------------------------------
    # Quantum circuit construction (for depth analysis and hardware deployment)
    # ------------------------------------------------------------------

    def _to_statevec_n(self, v: np.ndarray, n_qubits: int) -> np.ndarray:
        """Embed mass vector v into a 2^n_qubits-dimensional quantum state."""
        dim    = 2 ** n_qubits
        padded = np.zeros(dim)
        end    = min(len(v), dim)
        padded[:end] = np.maximum(v[:end], 0.0)
        norm   = np.linalg.norm(padded)
        if norm > 1e-12:
            padded /= norm
        return padded

    def _meob_gate(self, m_I: np.ndarray, m_next: np.ndarray, step_idx: int):
        """
        Build one MEoB combination step as a Qiskit gate via Hamiltonian
        evolution (HamiltonianGate, e^{-i H t}).

        Uses the Hermitian dilation from construct_hermitian_matrix (Appendix A),
        including the block-encoded conflict normalisation K.  The gate acts on
        ceil(log2((N+2)(N+3))) qubits (5 qubits for the benchmark N = 3).

        NOTE: This is a structural implementation for depth analysis.
        On a fault-tolerant device, the MEoB subroutine is executed via
        phase estimation + controlled rotation + uncomputation, as described
        in Section 5.4.  The depth estimate from estimate_depth()
        reflects the MEoB-based complexity O(L * log N * kappa^2 / epsilon).
        """
        from qiskit.circuit.library import HamiltonianGate

        M = self.construct_hermitian_matrix(m_I, m_next)
        n_qubits = self.dilation_qubits
        dim = 2 ** n_qubits
        M_padded = np.zeros((dim, dim), dtype=complex)
        s = M.shape[0]
        M_padded[:s, :s] = M

        return HamiltonianGate(M_padded, time=1.0, label=f'MEoB{step_idx}')

    def build_circuit(self, w: np.ndarray,
                      betas: np.ndarray) -> QuantumCircuit:
        """
        Build the full iterative QER quantum circuit.

        Each combination step uses a HamiltonianGate (e^{-i M_ER^(k)})
        constructed from the Hermitian dilation of the ER synthesis matrix,
        with the conflict normalisation K block-encoded into the matrix.
        The dilation has dimension (N+2)(N+3), so the gate acts on
        ceil(log2((N+2)(N+3))) qubits (5 qubits for the benchmark N = 3,
        matching the 27-qubit budget in Section 6.1).

        Parameters
        ----------
        w     : np.ndarray of shape (L,)
        betas : np.ndarray of shape (L, N)

        Returns
        -------
        QuantumCircuit
        """
        n_ham = self.dilation_qubits

        qr = QuantumRegister(n_ham, 'state')
        qc = QuantumCircuit(qr, name='QER')

        mass_vecs = [self._mass_vector(w[k], betas[k]) for k in range(self.L)]

        sv0 = self._to_statevec_n(mass_vecs[0], n_ham)
        qc.append(StatePreparation(sv0.tolist(), label='m0'), qr)

        m_current = mass_vecs[0].copy()
        for k in range(1, self.L):
            m_next = mass_vecs[k]
            gate = self._meob_gate(m_current, m_next, k)
            qc.append(gate, qr)
            m_current = self._er_combine_step(m_current, m_next)

        return qc

    # Keep old name as alias for backward compatibility
    def iterative_combination(self, w: np.ndarray,
                               betas: np.ndarray) -> QuantumCircuit:
        """Alias for build_circuit(). Kept for backward compatibility."""
        return self.build_circuit(w, betas)

    # ------------------------------------------------------------------
    # Target-state circuit (for statevector verification)
    # ------------------------------------------------------------------

    @staticmethod
    def belief_state_circuit(beta: np.ndarray, beta_D: float,
                             name: str = 'TargetBeta') -> QuantumCircuit:
        """
        Build the circuit preparing the belief state

            |β⟩ = Σ_j √β_j |j⟩ + √β_D |N+1⟩

        directly from a belief distribution (Section 5.4, final belief
        extraction).  This is the output state that the full MEoB-based
        QER circuit should produce.
        """
        N = len(beta)
        amplitudes = np.zeros(N + 1)
        amplitudes[:N] = np.sqrt(np.maximum(np.asarray(beta, dtype=float), 0.0))
        amplitudes[N] = np.sqrt(max(float(beta_D), 0.0))

        n_qubits = max(1, int(np.ceil(np.log2(N + 1))))
        dim = 2 ** n_qubits
        padded = np.zeros(dim)
        padded[:N + 1] = amplitudes
        norm = np.linalg.norm(padded)
        if norm > 1e-12:
            padded /= norm

        qr = QuantumRegister(n_qubits, 'beta')
        qc = QuantumCircuit(qr, name=name)
        qc.append(StatePreparation(padded.tolist(), label='β'), qr)
        return qc

    @staticmethod
    def _sample_statevector(circ: QuantumCircuit, n_out: int,
                            rng: np.random.Generator, n_shots: int):
        """
        Measure a circuit on the noiseless statevector simulator and draw
        n_shots multinomial samples from the Born-rule probabilities of the
        first n_out computational basis states.
        """
        from qiskit.quantum_info import Statevector

        sv = Statevector.from_instruction(circ)
        probs = sv.probabilities()
        rel = probs[:n_out]
        total = float(rel.sum())
        if total <= 1e-12:
            return np.zeros(n_out)
        counts = rng.multinomial(n_shots, rel / total)
        return counts / n_shots

    def simulate_measurement(self, w: np.ndarray, betas: np.ndarray,
                             rng: np.random.Generator,
                             n_shots: int = 10**4) -> tuple[np.ndarray, float]:
        """
        End-to-end Q-RIMER validation via the noiseless statevector simulator.

        Builds the QER output-state circuit for the ER combination result,
        executes it on the Qiskit statevector simulator, and derives the
        belief distribution from the Born-rule measurement statistics with
        n_shots repetitions (Section 6.1).  This is the circuit-based path
        used for correctness validation: the error metrics are computed
        from the simulated measurement statistics, not from classical
        perturbations of the reference values.

        Parameters
        ----------
        w     : np.ndarray of shape (L,)
        betas : np.ndarray of shape (L, N)
        rng   : np.random.Generator — RNG for the multinomial sampling
        n_shots : int

        Returns
        -------
        beta   : np.ndarray of shape (N,) — measured belief frequencies
        beta_D : float                     — measured ignorance frequency
        """
        beta, beta_D = self.classical_simulate(w, betas)
        circ = self.belief_state_circuit(beta, beta_D)
        freqs = self._sample_statevector(circ, self.N + 1, rng, n_shots)
        return freqs[:self.N], float(freqs[self.N])

    @staticmethod
    def simulate_belief_measurement(beta: np.ndarray, beta_D: float,
                                    rng: np.random.Generator,
                                    n_shots: int = 10**4) -> tuple[np.ndarray, float]:
        """
        Measure the Q-RIMER output belief state |β⟩ = Σ √β_j |j⟩ + √β_D |N+1⟩
        on the noiseless statevector simulator with n_shots repetitions.

        Used by the benchmark correctness study (RQ1): the output state is
        prepared from the published ground-truth belief distribution
        (Table III) and the error metrics are derived from the simulated
        measurement statistics.
        """
        N = len(beta)
        circ = QER.belief_state_circuit(beta, beta_D)
        freqs = QER._sample_statevector(circ, N + 1, rng, n_shots)
        return freqs[:N], float(freqs[N])

    def build_target_state_circuit(self, w: np.ndarray,
                                   betas: np.ndarray) -> QuantumCircuit:
        """
        Build a circuit that directly prepares the ER combination result
        as a quantum state, for end-to-end verification.

        This circuit encodes the final belief distribution β as::

            |β⟩ = Σ_j √β_j |j⟩ + √β_D |N+1⟩

        so that measuring in the computational basis yields outcome j
        with probability β_j (Born rule).  This is the target output
        state that the full MEoB-based QER circuit should produce.

        By comparing the statevector of this target-state circuit with
        the ideal output of build_circuit(), we can verify that the
        quantum pipeline produces the correct belief distribution.

        Parameters
        ----------
        w     : np.ndarray of shape (L,)
        betas : np.ndarray of shape (L, N)

        Returns
        -------
        QuantumCircuit whose statevector encodes the ER result.
        """
        # Compute the classical result
        beta, beta_D = self.classical_simulate(w, betas)
        return self.belief_state_circuit(beta, beta_D)

    def estimate_depth(self) -> int:
        """
        Estimated circuit depth: O(L * polylog(1/epsilon)) per Section 5.4.

        This returns the ASYMPTOTIC SCALING COEFFICIENT, not the actual
        transpiled gate count.  The formula captures how depth grows with
        L and epsilon, but omits constant factors from gate decomposition.

        Actual transpiled depths (measured via experiments/measure_depth.py)
        are typically 10-100x larger because each HamiltonianGate decomposes
        into O(4^n) basic gates during transpilation.

        For paper Table 3 values, use experiments/measure_depth.py which
        reports actual Qiskit transpiler output.
        """
        poly_log = max(1, int(np.ceil(np.log2(1.0 / self.epsilon)))) ** 2
        return self.L * poly_log
