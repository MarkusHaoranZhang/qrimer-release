"""Tests for the six-term sparse LCU block encoding."""

import numpy as np

from core.qer import QER
from core.sparse_block_encoding import SparseBlockEncoding


def test_sparse_block_encoding_single_step():
    N = 2
    qer = QER(N=N, L=2, kappa_max=10.0)
    w = np.array([0.6, 0.4])
    betas = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.1]])[:, :N]
    betas = betas / betas.sum(axis=1, keepdims=True) * 0.9
    m_I = qer._mass_vector(w[0], betas[0])
    m_next = qer._mass_vector(w[1], betas[1])
    be = SparseBlockEncoding(N)
    result = be.step(m_I, m_next)
    expected = qer._er_combine_step(m_I, m_next)
    assert np.linalg.norm(result['m_new'] - expected) < 1e-9
    assert 0.0 < result['success_probability'] <= 1.0


def test_sparse_block_encoding_resources():
    be = SparseBlockEncoding(3)
    report = be.resource_report()
    assert report['terms'] == 6
    assert report['total_qubits'] == 9
    assert report['block_encoding_dim'] == 512


def test_coherent_scaling_and_step():
    be = SparseBlockEncoding(2)
    state = np.array([0.2, 0.3, 0.5, 0.0])
    scaled, p = be.coherent_rescale(state, 0.5)
    assert abs(p - 0.25) < 1e-12
    assert np.linalg.norm(scaled - 0.5 * state) < 1e-12
    qer = QER(N=2, L=2, kappa_max=10.0)
    w = np.array([0.6, 0.4])
    betas = np.array([[0.5, 0.3], [0.2, 0.5]])
    m_I = qer._mass_vector(w[0], betas[0])
    m_next = qer._mass_vector(w[1], betas[1])
    out = be.coherent_step(m_I, m_next)
    assert 0.0 < out['success_probability'] <= 1.0
    assert out['n_qubits'] == be.n_total + 1
