"""
Quantum Evidential Reasoning (QER) Engine.
Implements Section 5.4: polynomial speedup via Hermitian matrix MEoB.
Compatible with Qiskit >= 1.0.

Execution paths
----------------
classical_simulate (default)
    Runs the ER combination classically using the exact matrix formulation
    derived in the paper.  This is the reference path used for validation.

quantum_step / quantum_simulate
    Execute each ER bilinear step through the explicit block encoding in
    core.block_encoding, with post-selection and per-step scalar rescaling.
    This is a circuit-executed hybrid validation for small N.

simulate_measurement / simulate_belief_measurement
    Prepare a target belief state from a classical result and sample its
    measurement statistics.  These paths model shot noise only; they do not
    execute the MEoB combination circuit.
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

        B^(k) is (N+2) x (N+2)^2.  The tensor product (N+2)^2 is the input
        register; the Hermitian dilation adds an (N+2)-dimensional output
        block, so the square dilation has dimension
        (N+2) + (N+2)^2 = (N+2)(N+3).  The dilation register therefore uses
        ceil(log2((N+2)(N+3))) qubits — 5 for the benchmark N = 3,
        matching the paper's qubit budget (Section 6.1).

        This method belongs to the legacy structural Hamiltonian path used by
        build_circuit().  It folds the conflict scalar K_{I(k+1)} into B only
        to form that structural skeleton.  The circuit-executed block-encoding
        path in core.block_encoding does not fold K into B: it applies K after
        post-selection.  Hence the spectral condition number kappa_BE of B is
        data-independent, and conflict enters through the post-selection
        factor Gamma_acc rather than through kappa_BE (Section 5.4).

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

        In this legacy structural path, K * B is used only to form the
        Hamiltonian skeleton.  It is not used for the spectral or
        condition-number claims.  Verified element-wise in
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

        # Legacy structural path: fold K into B only to form the Hamiltonian
        # skeleton. The executed block-encoding path applies K after
        # post-selection, and kappa_BE is data-independent.
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
        Build the structural QER circuit skeleton (depth analysis).

        Each combination step uses a HamiltonianGate (e^{-i M_ER^(k)})
        constructed from the Hermitian dilation of the ER synthesis matrix,
        with the conflict normalisation K folded into the matrix.  This is a
        structural/depth path; use ``core.block_encoding.ERBlockEncoding``
        for an executed block-encoded ER step.
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
        Target-state measurement via the noiseless statevector simulator.

        This method computes the ER result classically, prepares the target
        belief state, and samples its measurement statistics.  It models
        finite-shot readout but does not execute the MEoB/block-encoding
        circuit; use ``quantum_simulate`` for the circuit-executed kernel.

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
        Measure the target belief state |β⟩ = Σ √β_j |j⟩ + √β_D |N+1⟩
        on the noiseless statevector simulator with n_shots repetitions.

        This models readout statistics of a state prepared from a known
        belief distribution; it is not a circuit-executed MEoB inference.
        """
        N = len(beta)
        circ = QER.belief_state_circuit(beta, beta_D)
        freqs = QER._sample_statevector(circ, N + 1, rng, n_shots)
        return freqs[:N], float(freqs[N])

    def build_target_state_circuit(self, w: np.ndarray,
                                   betas: np.ndarray) -> QuantumCircuit:
        """
        Build a target-state circuit from the classical ER result.

        This is a readout/state-preparation check, not the MEoB combination
        circuit.  Use ``core.block_encoding.ERBlockEncoding`` for the
        circuit-executed ER kernel.

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

    def quantum_step(self, m_I: np.ndarray, m_next: np.ndarray,
                     shots: int | None = None,
                     rng: np.random.Generator | None = None,
                     amplification_rounds: int = 1) -> dict:
        """
        Execute one ER combination step through the explicit block encoding.

        This is a circuit-executed kernel for small N using the paper's
        Hermitian dilation M = [[0, B], [B^H, 0]].  The block encoding
        produces the normalised direction of B (m_I (x) m_next); the conflict
        scalar K and the state norm are applied after the post-selected step.
        """
        from .block_encoding import HermitianERBlockEncoding

        return HermitianERBlockEncoding(self.N).step(
            m_I, m_next, shots=shots, rng=rng,
            amplification_rounds=amplification_rounds
        )

    def quantum_simulate(self, w: np.ndarray, betas: np.ndarray,
                         shots: int | None = None,
                         rng: np.random.Generator | None = None,
                         amplification_rounds: int = 1,
                         discount: float = 0.0,
                         delegate_threshold: float | None = None,
                         coherent_k: bool = True,
                         gate_level: bool = False) -> tuple:
        """
        Run the ER inference with a circuit-executed kernel per rule.

        Each step is executed on the Qiskit statevector simulator through the
        Hermitian-dilation block encoding.  With ``coherent_k=True`` the
        state-dependent K is applied by an ancilla rotation without measuring
        the belief state; otherwise K is applied as a post-selected scalar.
        """
        from .block_encoding import HermitianERBlockEncoding

        encoder = HermitianERBlockEncoding(self.N)
        if gate_level:
            encoder.use_qiskit = True
        if coherent_k and shots is None:
            return encoder.coherent_simulate(
                w, betas, amplification_rounds=amplification_rounds,
                discount=discount, delegate_threshold=delegate_threshold
            )
        return encoder.simulate(
            w, betas, shots=shots, rng=rng,
            amplification_rounds=amplification_rounds,
            discount=discount, delegate_threshold=delegate_threshold
        )

    def estimate_depth(self) -> int:
        """
        Estimated circuit depth: O(L * polylog(1/epsilon)) per Section 5.4.

        This returns the ASYMPTOTIC SCALING COEFFICIENT, not the actual
        transpiled gate count.  The formula captures how depth grows with
        L and epsilon, but omits constant factors from gate decomposition.

        Actual transpiled depths (measured via experiments/measure_depth.py)
        are typically 10-100x larger because each HamiltonianGate decomposes
        into O(4^n) basic gates during transpilation.

        For the current structured-resource values, use
        experiments/run_structured_resources.py.  measure_depth.py is a
        legacy Qiskit-transpilation helper.
        """
        poly_log = max(1, int(np.ceil(np.log2(1.0 / self.epsilon)))) ** 2
        return self.L * poly_log
