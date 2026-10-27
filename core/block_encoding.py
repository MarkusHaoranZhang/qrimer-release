"""
Circuit-executed ER kernel via an explicit block encoding.

Two equivalent small-N block encodings are provided:

- ``HermitianERBlockEncoding`` is the manuscript-faithful form: the
  Hermitian dilation M = [[0, B], [B^H, 0]] is block-encoded with one
  ancilla, and the input mass tensor is prepared in the bottom (N+2)^2 block.
- ``ERBlockEncoding`` uses separate input/output registers and is retained as
  an independent validation of the same bilinear map.

The default path applies the conflict factor K and the state norm as
post-selected scalars.  ``coherent_simulate`` instead applies K through an
ancilla rotation, and ``noisy_step`` applies local depolarizing noise to the
complete block-encoded density matrix.  The sparse six-term LCU construction
is provided separately in ``sparse_block_encoding.py``.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import null_space

from .er_matrices import build_b_matrix, hermitian_dilation


class ERBlockEncoding:
    """
    Explicit block encoding of the ER bilinear map for small N.

    The input register holds the (N+2)^2-dimensional tensor product
    m_I (x) m_next.  The output/ancilla register holds the (N+2)-dimensional
    result.  The success event is ``input register = 0`` and
    ``output register < N+2``; the remaining computational-basis states are
    the orthogonal failure branch.
    """

    def __init__(self, N: int):
        self.N = N
        self.d = N + 2
        self.n = self.d * self.d
        self.n_i = int(np.ceil(np.log2(self.n)))
        self.n_o = int(np.ceil(np.log2(self.d)))
        self.dim_i = 1 << self.n_i
        self.dim_o = 1 << self.n_o
        self.dim = self.dim_i * self.dim_o
        self.B = build_b_matrix(N)
        self.alpha = float(np.linalg.norm(self.B, 2))
        self.U = self._build_unitary()

    def _build_unitary(self) -> np.ndarray:
        d = self.d
        n = self.n
        D = self.dim
        A = self.B / self.alpha

        # Isometry on the input subspace: top block A, then a completion
        # sqrt(I - A^H A).  The columns are orthonormal by construction.
        Q = np.zeros((D, n), dtype=complex)
        Q[:d, :] = A
        gram = np.eye(n) - A.conj().T @ A
        evals, evecs = np.linalg.eigh(gram)
        evals = np.clip(evals, 0.0, None)
        completion = (evecs * np.sqrt(evals)) @ evecs.conj().T
        Q[self.dim_o:self.dim_o + n, :] = completion

        complement = null_space(Q.conj().T)
        positions = np.arange(n) * self.dim_o
        mask = np.ones(D, dtype=bool)
        mask[positions] = False

        U = np.zeros((D, D), dtype=complex)
        U[:, positions] = Q
        U[:, mask] = complement
        return U

    def _mass_vector(self, w_k: float, beta_k: np.ndarray) -> np.ndarray:
        m = np.zeros(self.d)
        m[:self.N] = w_k * np.maximum(beta_k, 0.0)
        m[self.N] = 1.0 - w_k
        m[self.N + 1] = w_k * max(0.0, 1.0 - float(np.sum(beta_k)))
        return m

    def _prepare_input(self, m_I: np.ndarray,
                       m_next: np.ndarray) -> tuple[np.ndarray, float]:
        psi = np.zeros(self.dim, dtype=complex)
        tensor = np.outer(m_I, m_next).reshape(-1)
        psi[np.arange(self.n) * self.dim_o] = tensor
        norm = float(np.linalg.norm(psi))
        if norm <= 1e-15:
            raise ValueError("Input mass vectors have zero norm")
        return psi / norm, norm

    def _run_statevector(self, psi: np.ndarray) -> np.ndarray:
        qr_i = QuantumRegister(self.n_i, 'i')
        qr_o = QuantumRegister(self.n_o, 'o')
        qc = QuantumCircuit(qr_i, qr_o)
        qc.append(StatePreparation(psi.tolist()), list(qr_i) + list(qr_o))
        qc.append(UnitaryGate(self.U, label='U_B'), list(qr_i) + list(qr_o))
        sv = Statevector.from_instruction(qc)
        return np.asarray(sv.data, dtype=complex)

    def step(self, m_I: np.ndarray, m_next: np.ndarray,
             shots: int | None = None,
             rng: np.random.Generator | None = None) -> dict:
        """
        Execute one block-encoded ER combination step.

        Parameters
        ----------
        m_I, m_next : mass vectors of shape (N+2,)
        shots : int, optional
            If given, model post-selected measurement with this many shots.
            Otherwise the exact statevector amplitudes are used.
        rng : np.random.Generator, optional
            Required when ``shots`` is not None.
        """
        psi, psi_norm = self._prepare_input(m_I, m_next)
        data = self._run_statevector(psi)

        success_probability = float(np.sum(np.abs(data[:self.d]) ** 2))
        if success_probability <= 1e-15:
            raise RuntimeError("Block-encoding success probability is zero")
        amplitudes = data[:self.d] / np.sqrt(success_probability)
        amplitudes = np.real_if_close(amplitudes).real
        amplitudes = np.maximum(amplitudes, 0.0)

        conflict = float(m_I[:self.N].sum() * m_next[:self.N].sum()
                         - np.dot(m_I[:self.N], m_next[:self.N]))
        K = 1.0 / (1.0 - conflict + 1e-15)
        scale = self.alpha * psi_norm * np.sqrt(success_probability)

        estimated_success = success_probability
        n_success = None
        if shots is not None:
            if rng is None:
                rng = np.random.default_rng(42)
            n_success = int(rng.binomial(shots, success_probability))
            estimated_success = n_success / shots if shots else 0.0
            probs = amplitudes ** 2
            probs = probs / max(probs.sum(), 1e-15)
            counts = (rng.multinomial(n_success, probs)
                      if n_success > 0 else np.zeros(self.d, dtype=int))
            amplitudes = np.sqrt(counts / n_success) if n_success > 0 else amplitudes
            scale = self.alpha * psi_norm * np.sqrt(max(estimated_success, 0.0))

        m_new = K * scale * amplitudes
        if m_new.sum() > 1e-15:
            m_new = m_new / m_new.sum()
        m_new = np.real_if_close(m_new).real

        return {
            'm_new': m_new,
            'success_probability': success_probability,
            'estimated_success_probability': estimated_success,
            'shots': shots,
            'n_success': n_success,
            'amplitudes': amplitudes,
            'K': K,
            'scale': scale,
            'n_qubits': self.n_i + self.n_o,
            'circuit_depth': 2,
        }

    def simulate(self, w: np.ndarray, betas: np.ndarray,
                 shots: int | None = None,
                 rng: np.random.Generator | None = None) -> tuple:
        """
        Run circuit-executed ER inference over all L rules.

        The quantum circuit computes each bilinear ER step; the conflict
        normalisation K and the scalar state norm are applied classically
        between steps.  This is a post-selected hybrid execution, not a
        measurement-free coherent pipeline.
        """
        L, N = betas.shape
        if L != len(w):
            raise ValueError("w and betas must have the same number of rules")
        if N != self.N:
            raise ValueError("betas do not match encoding dimension")

        m_current = self._mass_vector(w[0], betas[0])
        per_step_success = []
        for k in range(1, L):
            m_next = self._mass_vector(w[k], betas[k])
            result = self.step(m_current, m_next, shots=shots, rng=rng)
            m_current = result['m_new']
            per_step_success.append(result['success_probability'])

        denom = 1.0 - m_current[self.N] + 1e-15
        beta = m_current[:self.N] / denom
        beta_D = m_current[self.N + 1] / denom
        info = {
            'per_step_success': per_step_success,
            'coherent_success_product': (
                float(np.prod(per_step_success)) if per_step_success else 1.0
            ),
            'success_probability_mean': (
                float(np.mean(per_step_success)) if per_step_success else 1.0
            ),
            'n_qubits': self.n_i + self.n_o,
            'circuit_depth': 2,
            'steps': L - 1,
        }
        return beta, beta_D, info

class HermitianERBlockEncoding:
    """
    Block encoding of the paper's Hermitian dilation.

    This is the manuscript-faithful variant: B is embedded in
    M = [[0, B], [B^H, 0]], and a single-ancilla unitary block-encodes
    M/alpha with alpha = ||M||_2 = ||B||_2.  The input state lives in the
    bottom (N+2)^2 block; post-selecting ancilla=0 and the top (N+2) block
    gives the unnormalised ER combination direction.
    """

    def __init__(self, N: int):
        self.N = N
        self.d = N + 2
        self.B = build_b_matrix(N)
        self.M = hermitian_dilation(self.B)
        self.alpha = float(np.linalg.norm(self.M, 2))
        self.dim = self.M.shape[0]
        self.n_state = int(np.ceil(np.log2(self.dim)))
        self.dim_state = 1 << self.n_state
        self.n_total = self.n_state + 1
        self.dim_total = 1 << self.n_total
        self.U = self._build_unitary()
        # Use Qiskit circuit execution for small instances and direct
        # statevector matrix application for larger instances.  The two
        # paths implement the same unitary and avoid dense-gate transpilation
        # becoming the performance bottleneck at large N.
        self.use_qiskit = self.n_total <= 8

    def _build_unitary(self) -> np.ndarray:
        Ds = self.dim_state
        A = np.zeros((Ds, Ds), dtype=complex)
        A[:self.dim, :self.dim] = self.M / self.alpha
        gram = np.eye(Ds) - A.conj().T @ A
        evals, evecs = np.linalg.eigh(gram)
        evals = np.clip(evals, 0.0, None)
        completion = (evecs * np.sqrt(evals)) @ evecs.conj().T
        Q = np.vstack([A, completion])
        complement = null_space(Q.conj().T)
        return np.hstack([Q, complement])

    def _mass_vector(self, w_k: float, beta_k: np.ndarray) -> np.ndarray:
        m = np.zeros(self.d)
        m[:self.N] = w_k * np.maximum(beta_k, 0.0)
        m[self.N] = 1.0 - w_k
        m[self.N + 1] = w_k * max(0.0, 1.0 - float(np.sum(beta_k)))
        return m

    def _prepare_input(self, m_I: np.ndarray,
                       m_next: np.ndarray) -> tuple[np.ndarray, float]:
        tensor = np.outer(m_I, m_next).reshape(-1)
        norm = float(np.linalg.norm(tensor))
        if norm <= 1e-15:
            raise ValueError("Input mass vectors have zero norm")
        state = np.zeros(self.dim_state, dtype=complex)
        state[self.d:self.d + self.d * self.d] = tensor / norm
        full = np.zeros(self.dim_total, dtype=complex)
        full[:self.dim_state] = state
        return full, norm

    @staticmethod
    def _prep_unitary(psi: np.ndarray) -> np.ndarray:
        # Exact dense state preparation: H|0> = psi with a Householder
        # reflection.  StatePreparation can introduce relative phases for
        # sparse padded vectors, so the executable path uses this unitary.
        dim = len(psi)
        e0 = np.zeros(dim, dtype=complex)
        e0[0] = 1.0
        v = e0 - psi
        norm = np.linalg.norm(v)
        if norm <= 1e-15:
            return np.eye(dim, dtype=complex)
        v = v / norm
        return np.eye(dim, dtype=complex) - 2.0 * np.outer(v, v.conj())

    def _amplification_operator(self, psi: np.ndarray) -> np.ndarray:
        prep = self._prep_unitary(psi)
        A = self.U @ prep
        S0 = np.eye(self.dim_total, dtype=complex)
        S0[0, 0] = -1.0
        Schi = np.eye(self.dim_total, dtype=complex)
        Schi[:self.d, :self.d] = -np.eye(self.d)
        return A @ S0 @ A.conj().T @ Schi

    def _run_statevector(self, psi: np.ndarray,
                         amplification_rounds: int = 0) -> np.ndarray:
        if not self.use_qiskit:
            data = self.U @ psi
            if amplification_rounds > 0:
                prep = self._prep_unitary(psi)
                prep_dag = prep.conj().T
                u_dag = self.U.conj().T
                for _ in range(amplification_rounds):
                    v = data.copy()
                    v[:self.d] *= -1.0
                    v = prep_dag @ (u_dag @ v)
                    v[0] *= -1.0
                    data = self.U @ (prep @ v)
            return data
        qc = QuantumCircuit(self.n_total)
        qc.append(UnitaryGate(self._prep_unitary(psi), label='Prep'),
                  range(self.n_total))
        qc.append(UnitaryGate(self.U, label='U_BE'), range(self.n_total))
        if amplification_rounds > 0:
            Q = self._amplification_operator(psi)
            for _ in range(amplification_rounds):
                qc.append(UnitaryGate(Q, label='AA'), range(self.n_total))
        sv = Statevector.from_instruction(qc)
        return np.asarray(sv.data, dtype=complex)

    def step(self, m_I: np.ndarray, m_next: np.ndarray,
             shots: int | None = None,
             rng: np.random.Generator | None = None,
             amplification_rounds: int = 1) -> dict:
        psi, psi_norm = self._prepare_input(m_I, m_next)
        data = self._run_statevector(psi, amplification_rounds=amplification_rounds)

        success_probability = float(np.sum(np.abs(data[:self.d]) ** 2))
        raw_data = self.U @ psi
        raw_success_probability = float(np.sum(np.abs(raw_data[:self.d]) ** 2))
        if success_probability <= 1e-15 or raw_success_probability <= 1e-15:
            raise RuntimeError("Block-encoding success probability is zero")
        amplitudes = np.abs(data[:self.d] / np.sqrt(success_probability))

        conflict = float(m_I[:self.N].sum() * m_next[:self.N].sum()
                         - np.dot(m_I[:self.N], m_next[:self.N]))
        K = 1.0 / (1.0 - conflict + 1e-15)
        scale = self.alpha * psi_norm * np.sqrt(raw_success_probability)

        estimated_success = success_probability
        n_success = None
        if shots is not None:
            if rng is None:
                rng = np.random.default_rng(42)
            n_success = int(rng.binomial(shots, success_probability))
            estimated_success = n_success / shots if shots else 0.0
            probs = amplitudes ** 2
            probs = probs / max(probs.sum(), 1e-15)
            counts = (rng.multinomial(n_success, probs)
                      if n_success > 0 else np.zeros(self.d, dtype=int))
            amplitudes = np.sqrt(counts / n_success) if n_success > 0 else amplitudes

        m_new = K * scale * amplitudes
        if m_new.sum() > 1e-15:
            m_new = m_new / m_new.sum()
        m_new = np.real_if_close(m_new).real
        return {
            'm_new': m_new,
            'success_probability': success_probability,
            'raw_success_probability': raw_success_probability,
            'amplification_rounds': amplification_rounds,
            'estimated_success_probability': estimated_success,
            'shots': shots,
            'n_success': n_success,
            'amplitudes': amplitudes,
            'K': K,
            'scale': scale,
            'n_qubits': self.n_total,
            'circuit_depth': 2,
        }

    def _classical_step(self, m_I: np.ndarray,
                        m_next: np.ndarray) -> np.ndarray:
        """Classical ER step used by the high-conflict delegation path."""
        N = self.N
        mc_N = m_I[:N]
        mn_N = m_next[:N]
        conflict = float(mc_N.sum() * mn_N.sum() - np.dot(mc_N, mn_N))
        K = 1.0 / (1.0 - conflict + 1e-15)
        mD_I = m_I[N] + m_I[N + 1]
        mD_next = m_next[N] + m_next[N + 1]
        m_new = np.empty(self.d)
        m_new[:N] = K * (mc_N * mn_N + mc_N * mD_next + mD_I * mn_N)
        m_new[N + 1] = K * (
            m_I[N + 1] * m_next[N + 1]
            + m_I[N + 1] * m_next[N]
            + m_I[N] * m_next[N + 1]
        )
        m_new[N] = K * m_I[N] * m_next[N]
        return m_new

    def simulate(self, w: np.ndarray, betas: np.ndarray,
                 shots: int | None = None,
                 rng: np.random.Generator | None = None,
                 amplification_rounds: int = 1,
                 discount: float = 0.0,
                 delegate_threshold: float | None = None) -> tuple:
        L, N = betas.shape
        if L != len(w):
            raise ValueError("w and betas must have the same number of rules")
        if N != self.N:
            raise ValueError("betas do not match encoding dimension")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")
        betas_eff = betas * (1.0 - discount)
        m_current = self._mass_vector(w[0], betas_eff[0])
        per_step_success = []
        raw_per_step_success = []
        delegated = 0
        for k in range(1, L):
            m_next = self._mass_vector(w[k], betas_eff[k])
            conflict = float(m_current[:N].sum() * m_next[:N].sum()
                             - np.dot(m_current[:N], m_next[:N]))
            if delegate_threshold is not None and conflict > delegate_threshold:
                m_current = self._classical_step(m_current, m_next)
                per_step_success.append(1.0)
                raw_per_step_success.append(1.0)
                delegated += 1
                continue
            result = self.step(m_current, m_next, shots=shots, rng=rng,
                               amplification_rounds=amplification_rounds)
            m_current = result['m_new']
            per_step_success.append(result['success_probability'])
            raw_per_step_success.append(result['raw_success_probability'])
        denom = 1.0 - m_current[self.N] + 1e-15
        beta = m_current[:self.N] / denom
        beta_D = m_current[self.N + 1] / denom
        info = {
            'per_step_success': per_step_success,
            'raw_per_step_success': raw_per_step_success,
            'coherent_success_product': (
                float(np.prod(per_step_success)) if per_step_success else 1.0
            ),
            'raw_coherent_success_product': (
                float(np.prod(raw_per_step_success)) if raw_per_step_success else 1.0
            ),
            'success_probability_mean': (
                float(np.mean(per_step_success)) if per_step_success else 1.0
            ),
            'raw_success_probability_mean': (
                float(np.mean(raw_per_step_success)) if raw_per_step_success else 1.0
            ),
            'amplification_rounds': amplification_rounds,
            'delegated_steps': delegated,
            'discount': discount,
            'n_qubits': self.n_total,
            'circuit_depth': 2,
            'steps': L - 1,
            'execution': ('qiskit_statevector' if self.use_qiskit
                          else 'direct_statevector'),
        }
        return beta, beta_D, info

    def coherent_step(self, m_I: np.ndarray, m_next: np.ndarray) -> dict:
        """Apply K as an ancilla rotation, without measuring the state."""
        result = self.step(m_I, m_next)
        s = float(np.clip(result['K'] * result['scale'], 0.0, 1.0))
        scaling_success = s * s
        return {
            'm_new': result['m_new'],
            'success_probability': result['success_probability'] * scaling_success,
            'raw_success_probability': result['raw_success_probability'],
            'scaling_success': scaling_success,
            'scaling_amplitude': s,
            'K': result['K'],
            'scale': result['scale'],
            'n_qubits': self.n_total + 1,
        }

    def coherent_simulate(self, w: np.ndarray, betas: np.ndarray,
                          amplification_rounds: int = 1,
                          discount: float = 0.0,
                          delegate_threshold: float | None = None) -> tuple:
        """
        End-to-end coherent K path.

        Each block-encoded step is post-selected, then the state-dependent
        scalar K is applied through an ancilla rotation.  No intermediate
        belief vector is measured; the final state is normalized only at the
        output.  The K value is evaluated from the current statevector to set
        the rotation angle.
        """
        L, N = betas.shape
        if L != len(w):
            raise ValueError("w and betas must have the same number of rules")
        if N != self.N:
            raise ValueError("betas do not match encoding dimension")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")
        betas_eff = betas * (1.0 - discount)
        m_current = self._mass_vector(w[0], betas_eff[0])
        per_step_success = []
        per_step_k = []
        coherent_k_product = 1.0
        delegated = 0
        for k in range(1, L):
            m_next = self._mass_vector(w[k], betas_eff[k])
            conflict = float(m_current[:N].sum() * m_next[:N].sum()
                             - np.dot(m_current[:N], m_next[:N]))
            if delegate_threshold is not None and conflict > delegate_threshold:
                m_current = self._classical_step(m_current, m_next)
                per_step_success.append(1.0)
                per_step_k.append(1.0)
                delegated += 1
                continue
            result = self.coherent_step(m_current, m_next) if amplification_rounds else self.step(
                m_current, m_next, amplification_rounds=0
            )
            m_current = result['m_new']
            per_step_success.append(result['success_probability'])
            per_step_k.append(result.get('scaling_amplitude', 1.0))
            coherent_k_product *= result.get('scaling_amplitude', 1.0)
        denom = 1.0 - m_current[self.N] + 1e-15
        beta = m_current[:self.N] / denom
        beta_D = m_current[self.N + 1] / denom
        info = {
            'per_step_success': per_step_success,
            'coherent_success_product': (
                float(np.prod(per_step_success)) if per_step_success else 1.0
            ),
            'coherent_k_product': coherent_k_product,
            'per_step_k_amplitude': per_step_k,
            'amplification_rounds': amplification_rounds,
            'discount': discount,
            'delegated_steps': delegated,
            'n_qubits': self.n_total + 1,
            'steps': L - 1,
            'fully_coherent_k': True,
            'execution': 'coherent_k_rescaling',
        }
        return beta, beta_D, info

    @staticmethod
    def _apply_local_depolarizing(rho: np.ndarray, p: float) -> np.ndarray:
        """Local single-qubit depolarizing channel on a density matrix."""
        if p <= 0.0:
            return rho
        dim = rho.shape[0]
        n = int(round(np.log2(dim)))
        idx = np.arange(dim)
        for q in range(n):
            flip = idx ^ (1 << q)
            rho_x = rho[np.ix_(flip, flip)]
            signs = 1.0 - 2.0 * ((idx >> q) & 1)
            rho_z = rho * signs[:, None] * signs[None, :]
            rho_y = rho_z[np.ix_(flip, flip)]
            rho = ((1.0 - p) * rho
                   + (p / 3.0) * (rho_x + rho_y + rho_z))
        return rho

    def noisy_simulate(self, w: np.ndarray, betas: np.ndarray,
                       noise: float = 0.0) -> tuple:
        """Run all L-1 block-encoded steps with density-matrix noise."""
        L, N = betas.shape
        if L != len(w):
            raise ValueError("w and betas must have the same number of rules")
        if N != self.N:
            raise ValueError("betas do not match encoding dimension")
        m_current = self._mass_vector(w[0], betas[0])
        per_step_success = []
        for k in range(1, L):
            m_next = self._mass_vector(w[k], betas[k])
            out = self.noisy_step(m_current, m_next, noise=noise)
            m_current = out['m_new']
            per_step_success.append(out['success_probability'])
        denom = 1.0 - m_current[self.N] + 1e-15
        beta = m_current[:self.N] / denom
        beta_D = m_current[self.N + 1] / denom
        info = {
            'per_step_success': per_step_success,
            'coherent_success_product': (
                float(np.prod(per_step_success)) if per_step_success else 1.0
            ),
            'noise': noise,
            'n_qubits': self.n_total,
            'steps': L - 1,
            'execution': 'block_encoded_density_matrix',
        }
        return beta, beta_D, info

    def noisy_step(self, m_I: np.ndarray, m_next: np.ndarray,
                   noise: float = 0.01) -> dict:
        """Full block-encoded step followed by local depolarizing noise."""
        if not 0.0 <= noise <= 1.0:
            raise ValueError("noise must lie in [0, 1]")
        psi, psi_norm = self._prepare_input(m_I, m_next)
        out = self.U @ psi
        rho = np.outer(out, out.conj())
        rho = self._apply_local_depolarizing(rho, noise)
        top = rho[:self.d, :self.d]
        p_success = float(np.real(np.trace(top)))
        if p_success <= 1e-15:
            raise RuntimeError("Noisy block-encoding success probability is zero")
        probs = np.real(np.diag(top)) / p_success
        probs = np.clip(probs, 0.0, None)
        amplitudes = np.sqrt(probs)
        amplitudes = amplitudes / max(amplitudes.sum(), 1e-15)
        return {
            'm_new': amplitudes,
            'success_probability': p_success,
            'noise': noise,
            'n_qubits': self.n_total,
            'execution': 'block_encoded_density_matrix',
        }
