"""Tests for the batched classical ER baseline."""

import numpy as np

from core.classical_batch import er_combine_batch
from core.qer import QER


def test_batch_matches_individual_instances():
    rng = np.random.default_rng(11)
    B, L, N = 4, 6, 3
    betas = rng.dirichlet(np.ones(N), size=(B, L))
    w = rng.dirichlet(np.ones(L), size=B)

    beta_batch, beta_D_batch = er_combine_batch(w, betas)
    for b in range(B):
        qer = QER(N=N, L=L, kappa_max=50.0)
        beta, beta_D = qer.classical_simulate(w[b], betas[b])
        assert np.linalg.norm(beta_batch[b] - beta) < 1e-10
        assert abs(beta_D_batch[b] - beta_D) < 1e-10
