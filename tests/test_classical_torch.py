"""Optional PyTorch baseline consistency test."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.classical_batch import er_combine_batch  # noqa: E402
from core.classical_torch import er_combine_torch  # noqa: E402


def test_torch_matches_numpy_batch():
    rng = np.random.default_rng(19)
    B, L, N = 3, 5, 3
    betas = rng.dirichlet(np.ones(N), size=(B, L))
    w = rng.dirichlet(np.ones(L), size=B)

    beta_np, betaD_np = er_combine_batch(w, betas)
    beta_t, betaD_t = er_combine_torch(w, betas, device="cpu")

    assert np.linalg.norm(beta_np - beta_t) < 1e-9
    assert np.linalg.norm(betaD_np - betaD_t) < 1e-9
