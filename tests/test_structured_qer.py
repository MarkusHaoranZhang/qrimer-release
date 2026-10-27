"""Tests for the explicit register-level QER circuit construction."""

import numpy as np

from core.qer import QER
from core.structured_qer import StructuredQER


def test_structured_qer_allocates_paper_benchmark_register_budget():
    qer = QER(N=3, L=36, kappa_max=50.0)
    rng = np.random.default_rng(3)
    w = rng.dirichlet(np.ones(36))
    betas = rng.dirichlet(np.ones(3), size=36)
    m_I = qer._mass_vector(w[0], betas[0])
    m_next = qer._mass_vector(w[1], betas[1])
    structured = StructuredQER(N=3, L=36, T=4)
    assert structured.total_qubits == 27
    qc = structured.step_circuit(m_I, m_next)
    assert qc.num_qubits == 27
    assert qc.depth() > 0
