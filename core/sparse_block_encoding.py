"""
Six-term LCU block encoding of the Hermitian ER dilation.

The row-side decomposition splits B into six partial isometries B_k.  Each
B_k is embedded in the Hermitian dilation M_k = [[0,B_k],[B_k^†,0]] and
block-encoded by a unitary on a state ancilla.  A three-qubit ancilla
prepares the six-term LCU, giving the block encoding

    U_BE |0>_LCU |0>_q |psi> = |0>_LCU |0>_q (M psi / 6) + orthogonal.

Post-selecting the LCU ancilla and q=0 yields the unnormalized ER direction.
"""

import math

import numpy as np

from .er_matrices import hermitian_dilation
from .row_lcu import row_lcu_matrix, row_lcu_terms


class SparseBlockEncoding:
    def __init__(self, N: int):
        self.N = N
        self.d = N + 2
        self.D = self.d + self.d * self.d
        self.n_state = int(math.ceil(math.log2(self.D)))
        self.dim_state = 1 << self.n_state
        self.n_lcu = 3
        self.n_total = self.n_lcu + 1 + self.n_state
        self.terms = row_lcu_terms(N)
        self.M_terms = []
        self.U_terms = []
        for term in self.terms:
            Bk = row_lcu_matrix(term, N)
            M = np.zeros((self.dim_state, self.dim_state), dtype=complex)
            M[:self.D, :self.D] = hermitian_dilation(Bk)
            gram = np.eye(self.dim_state) - M @ M
            evals, evecs = np.linalg.eigh(gram)
            evals = np.clip(evals, 0.0, None)
            L = (evecs * np.sqrt(evals)) @ evecs.conj().T
            U = np.block([[M, L], [L, -M]])
            self.M_terms.append(M)
            self.U_terms.append(U)

        self.prep = np.zeros(1 << self.n_lcu, dtype=complex)
        self.prep[:len(self.terms)] = 1.0 / np.sqrt(len(self.terms))
        e0 = np.zeros_like(self.prep)
        e0[0] = 1.0
        v = e0 - self.prep
        v = v / np.linalg.norm(v)
        self.prep_unitary = np.eye(len(self.prep), dtype=complex) - 2.0 * np.outer(v, v.conj())

        state_dim = 2 * self.dim_state
        self.select = np.zeros((len(self.prep) * state_dim,
                                len(self.prep) * state_dim),
                               dtype=complex)
        for k, U in enumerate(self.U_terms):
            self.select[k * state_dim:(k + 1) * state_dim,
                        k * state_dim:(k + 1) * state_dim] = U
        for k in range(len(self.U_terms), len(self.prep)):
            self.select[k * state_dim:(k + 1) * state_dim,
                        k * state_dim:(k + 1) * state_dim] = np.eye(state_dim)

        self.U_BE = (np.kron(self.prep_unitary.conj().T, np.eye(state_dim))
                     @ self.select
                     @ np.kron(self.prep_unitary, np.eye(state_dim)))

    def _input_state(self, m_I: np.ndarray,
                     m_next: np.ndarray) -> tuple[np.ndarray, float]:
        tensor = np.outer(m_I, m_next).reshape(-1)
        norm = float(np.linalg.norm(tensor))
        if norm <= 1e-15:
            raise ValueError("Input mass vectors have zero norm")
        state = np.zeros(self.dim_state, dtype=complex)
        state[self.d:self.d + self.d * self.d] = tensor / norm
        full = np.zeros(1 << self.n_total, dtype=complex)
        # LCU ancilla = 0, state ancilla q = 0, state register = state.
        full[:self.dim_state] = state
        return full, norm

    def step(self, m_I: np.ndarray, m_next: np.ndarray) -> dict:
        full, norm = self._input_state(m_I, m_next)
        data = self.U_BE @ full
        p = float(np.sum(np.abs(data[:self.d]) ** 2))
        if p <= 1e-15:
            raise RuntimeError("sparse block-encoding success probability is zero")
        amplitudes = np.abs(data[:self.d] / np.sqrt(p))
        conflict = float(m_I[:self.N].sum() * m_next[:self.N].sum()
                         - np.dot(m_I[:self.N], m_next[:self.N]))
        K = 1.0 / (1.0 - conflict + 1e-15)
        scale = len(self.terms) * norm * np.sqrt(p)
        m_new = K * scale * amplitudes
        if m_new.sum() > 1e-15:
            m_new = m_new / m_new.sum()
        return {
            'm_new': np.real_if_close(m_new).real,
            'success_probability': p,
            'K': K,
            'scale': scale,
            'terms': len(self.terms),
            'n_qubits': self.n_total,
        }

    def coherent_rescale(self, state: np.ndarray, scale: float) -> tuple:
        """
        Coherently apply a scalar rescaling s to a normalized state.

        An ancilla RY rotation maps |0> to s|0> + sqrt(1-s^2)|1>.
        Post-selecting the ancilla on |0> scales the state by s with
        success probability s^2; no measurement of the state is involved.
        """
        s = float(np.clip(scale, 0.0, 1.0))
        if len(state) != self.d:
            raise ValueError("state must have dimension N+2")
        return np.asarray(state, dtype=complex) * s, s * s

    def coherent_step(self, m_I: np.ndarray, m_next: np.ndarray) -> dict:
        result = self.step(m_I, m_next)
        s = result['K'] * result['scale']
        scaled_state, p_scale = self.coherent_rescale(
            result['m_new'][:self.d], min(s, 1.0)
        )
        return {
            'm_new': result['m_new'],
            'state': scaled_state,
            'success_probability': result['success_probability'] * p_scale,
            'scaling_success': p_scale,
            'n_qubits': self.n_total + 1,
        }

    def simulate(self, w: np.ndarray, betas: np.ndarray) -> tuple:
        L, N = betas.shape
        if L != len(w):
            raise ValueError("w and betas must have the same number of rules")
        if N != self.N:
            raise ValueError("betas do not match encoding dimension")
        m_current = np.zeros(self.d)
        m_current[:N] = w[0] * np.maximum(betas[0], 0.0)
        m_current[N] = 1.0 - w[0]
        m_current[N + 1] = w[0] * max(0.0, 1.0 - betas[0].sum())
        success = []
        for k in range(1, L):
            m_next = np.zeros(self.d)
            m_next[:N] = w[k] * np.maximum(betas[k], 0.0)
            m_next[N] = 1.0 - w[k]
            m_next[N + 1] = w[k] * max(0.0, 1.0 - betas[k].sum())
            result = self.step(m_current, m_next)
            m_current = result['m_new']
            success.append(result['success_probability'])
        denom = 1.0 - m_current[N] + 1e-15
        return (m_current[:N] / denom, m_current[N + 1] / denom,
                {'per_step_success': success,
                 'coherent_success_product': float(np.prod(success)),
                 'n_qubits': self.n_total,
                 'terms': len(self.terms)})

    def resource_report(self) -> dict:
        return {
            'N': self.N,
            'terms': len(self.terms),
            'lcu_ancilla_qubits': self.n_lcu,
            'state_ancilla_qubits': 1,
            'state_qubits': self.n_state,
            'total_qubits': self.n_total,
            'block_encoding_dim': self.U_BE.shape[0],
        }
