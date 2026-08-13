"""
Unit tests for the CRI-based fuzzy inference baseline.
"""

import numpy as np

from core.cri import cri_infer, cri_max_min_infer


class TestCRI:
    def test_output_normalised(self):
        match = np.array([0.8, 0.4, 0.2])
        betas = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
        beta, beta_D = cri_infer(match, betas)
        assert abs(np.sum(beta) - 1.0) < 1e-9

    def test_no_ignorance_component(self):
        """CRI cannot represent epistemic ignorance: beta_D is always 0."""
        match = np.array([0.1, 0.1])
        betas = np.array([[0.3, 0.2, 0.0], [0.2, 0.1, 0.1]])
        beta, beta_D = cri_infer(match, betas)
        assert beta_D == 0.0

    def test_strong_match_dominates(self):
        match = np.array([1.0, 0.0, 0.0])
        betas = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
        beta, _ = cri_infer(match, betas)
        assert beta[0] > beta[1]
        assert beta[0] > beta[2]

    def test_max_min_bound(self):
        """Output is bounded by the max-min composition."""
        match = np.array([0.6, 0.7])
        betas = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
        mu = cri_max_min_infer(match, betas)
        assert np.all(mu <= np.max(match) + 1e-12)

    def test_output_nonnegative(self):
        match = np.array([0.5, 0.5])
        betas = np.array([[0.5, 0.4, 0.1], [0.2, 0.6, 0.2]])
        beta, _ = cri_infer(match, betas)
        assert np.all(beta >= 0.0)
