"""Tests for the explicit circuit-executed ER block encoding."""

import numpy as np

from core.block_encoding import ERBlockEncoding, HermitianERBlockEncoding
from core.er_matrices import build_b_matrix, spectral_report
from core.qer import QER


def test_block_encoding_is_unitary():
    be = ERBlockEncoding(N=2)
    eye = np.eye(be.dim)
    assert np.linalg.norm(be.U.conj().T @ be.U - eye) < 1e-10
    assert np.linalg.norm(be.U @ be.U.conj().T - eye) < 1e-10


def test_top_block_matches_synthesis_matrix():
    be = ERBlockEncoding(N=2)
    positions = np.arange(be.n) * be.dim_o
    top = be.U[:be.d, positions]
    expected = build_b_matrix(2) / be.alpha
    assert np.linalg.norm(top - expected) < 1e-10


def test_spectral_report_shapes_and_condition():
    rep = spectral_report(3)
    assert rep['shape'] == (5, 25)
    assert rep['dilation_shape'] == (30, 30)
    assert abs(rep['condition_number'] - np.sqrt(5.0)) < 1e-8


def test_hermitian_block_encoding_unitary_and_top_block():
    be = HermitianERBlockEncoding(N=2)
    eye = np.eye(be.dim_total)
    assert np.linalg.norm(be.U.conj().T @ be.U - eye) < 1e-10
    from core.er_matrices import hermitian_dilation
    expected = hermitian_dilation(build_b_matrix(2)) / be.alpha
    top = be.U[:be.dim, :be.dim]
    assert np.linalg.norm(top - expected) < 1e-10


def test_hermitian_block_encoding_single_step():
    N = 3
    qer = QER(N=N, L=2, kappa_max=10.0)
    w = np.array([0.6, 0.4])
    betas = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.1]])
    m_I = qer._mass_vector(w[0], betas[0])
    m_next = qer._mass_vector(w[1], betas[1])
    be = HermitianERBlockEncoding(N=N)
    result = be.step(m_I, m_next)
    expected = qer._er_combine_step(m_I, m_next)
    assert np.linalg.norm(result['m_new'] - expected) < 1e-9
    assert 0.0 < result['success_probability'] <= 1.0


def test_coherent_k_simulation_matches_classical():
    N, L = 3, 4
    rng = np.random.default_rng(11)
    w = rng.dirichlet(np.ones(L))
    betas = rng.dirichlet(np.ones(N), size=L)
    qer = QER(N=N, L=L, kappa_max=10.0)
    beta_c, _ = qer.classical_simulate(w, betas)
    beta_q, beta_D_q, info = qer.quantum_simulate(w, betas, coherent_k=True)
    assert info['fully_coherent_k'] is True
    assert info['execution'] == 'coherent_k_rescaling'
    assert len(info['per_step_k_amplitude']) == L - 1
    assert np.linalg.norm(beta_q - beta_c) < 1e-9


def test_large_n_block_encoded_noise_step():
    N = 3
    qer = QER(N=N, L=2, kappa_max=10.0)
    w = np.array([0.6, 0.4])
    betas = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.1]])
    m_I = qer._mass_vector(w[0], betas[0])
    m_next = qer._mass_vector(w[1], betas[1])
    be = HermitianERBlockEncoding(N=N)
    clean = be.noisy_step(m_I, m_next, noise=0.0)
    raw = be.step(m_I, m_next, amplification_rounds=0)
    noisy = be.noisy_step(m_I, m_next, noise=0.01)
    assert np.linalg.norm(clean['m_new'] - raw['m_new']) < 1e-10
    assert 0.0 < noisy['success_probability'] < clean['success_probability']
    assert np.isclose(noisy['m_new'].sum(), 1.0, atol=1e-12)
    assert noisy['execution'] == 'block_encoded_density_matrix'


def test_full_noisy_pipeline_matches_classical_at_zero_noise():
    N, L = 3, 4
    rng = np.random.default_rng(13)
    w = rng.dirichlet(np.ones(L))
    betas = rng.dirichlet(np.ones(N), size=L)
    qer = QER(N=N, L=L, kappa_max=10.0)
    beta_c, _ = qer.classical_simulate(w, betas)
    beta_n, _, info = HermitianERBlockEncoding(N).noisy_simulate(
        w, betas, noise=0.0
    )
    assert np.linalg.norm(beta_n - beta_c) < 1e-9
    assert info['execution'] == 'block_encoded_density_matrix'
    assert len(info['per_step_success']) == L - 1


def test_fast_statevector_path_matches_qiskit_path():
    N = 3
    qer = QER(N=N, L=2, kappa_max=10.0)
    w = np.array([0.6, 0.4])
    betas = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.1]])
    m_I = qer._mass_vector(w[0], betas[0])
    m_next = qer._mass_vector(w[1], betas[1])
    be = HermitianERBlockEncoding(N=N)
    slow = be.step(m_I, m_next)
    be.use_qiskit = False
    fast = be.step(m_I, m_next)
    assert np.linalg.norm(fast['m_new'] - slow['m_new']) < 1e-10
    assert abs(fast['success_probability'] - slow['success_probability']) < 1e-10


def test_amplitude_amplification_improves_success():
    N = 3
    qer = QER(N=N, L=2, kappa_max=10.0)
    w = np.array([0.6, 0.4])
    betas = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.1]])
    m_I = qer._mass_vector(w[0], betas[0])
    m_next = qer._mass_vector(w[1], betas[1])
    be = HermitianERBlockEncoding(N=N)
    raw = be.step(m_I, m_next, amplification_rounds=0)
    amp = be.step(m_I, m_next, amplification_rounds=1)
    assert amp['success_probability'] > raw['success_probability']
    assert np.linalg.norm(amp['m_new'] - raw['m_new']) < 1e-9


def test_single_step_matches_classical_er():
    N = 3
    qer = QER(N=N, L=2, kappa_max=10.0)
    w = np.array([0.6, 0.4])
    betas = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.1]])
    m_I = qer._mass_vector(w[0], betas[0])
    m_next = qer._mass_vector(w[1], betas[1])

    be = ERBlockEncoding(N=N)
    result = be.step(m_I, m_next)
    expected = qer._er_combine_step(m_I, m_next)

    assert np.linalg.norm(result['m_new'] - expected) < 1e-9
    assert 0.0 < result['success_probability'] <= 1.0


def test_hybrid_simulation_matches_classical():
    N, L = 3, 4
    rng = np.random.default_rng(7)
    betas = rng.dirichlet(np.ones(N), size=L) * 0.9
    w = rng.dirichlet(np.ones(L))

    qer = QER(N=N, L=L, kappa_max=50.0)
    beta_c, beta_D_c = qer.classical_simulate(w, betas)
    beta_q, beta_D_q, info = qer.quantum_simulate(w, betas)

    assert np.linalg.norm(beta_q - beta_c) < 1e-8
    assert abs(beta_D_q - beta_D_c) < 1e-8
    assert len(info['per_step_success']) == L - 1
    assert all(0.0 < p <= 1.0 for p in info['per_step_success'])


def test_high_conflict_delegation_matches_classical():
    N, L = 3, 4
    rng = np.random.default_rng(31)
    betas = rng.dirichlet(np.ones(N), size=L) * 0.9
    w = rng.dirichlet(np.ones(L))
    qer = QER(N=N, L=L, kappa_max=50.0)
    beta_c, beta_D_c = qer.classical_simulate(w, betas)
    beta_q, beta_D_q, info = qer.quantum_simulate(
        w, betas, delegate_threshold=0.0
    )
    assert np.linalg.norm(beta_q - beta_c) < 1e-9
    assert info['delegated_steps'] == L - 1
