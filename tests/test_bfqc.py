"""
Unit tests for the BF-QC basic (conjunctive) combination baseline.
"""

import numpy as np

from core.bfqc import conjunctive_combine


class TestConjunctiveCombine:
    def test_single_rule_identity(self):
        """With one rule and w=1, output equals the rule's belief degrees."""
        w = np.array([1.0])
        betas = np.array([[0.5, 0.3, 0.2]])
        beta, beta_D = conjunctive_combine(w, betas)
        np.testing.assert_allclose(beta, [0.5, 0.3, 0.2], atol=1e-9)
        assert beta_D == 0.0

    def test_output_normalised(self):
        w = np.array([0.6, 0.4, 0.5])
        betas = np.array([[0.5, 0.3, 0.2],
                          [0.2, 0.5, 0.1],
                          [0.3, 0.3, 0.2]])
        beta, beta_D = conjunctive_combine(w, betas)
        assert np.sum(beta) + beta_D == np.float64(1.0) or \
            abs(np.sum(beta) + beta_D - 1.0) < 1e-9

    def test_output_nonnegative(self):
        w = np.array([0.5, 0.5])
        betas = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        beta, beta_D = conjunctive_combine(w, betas)
        assert np.all(beta >= -1e-9)
        assert beta_D >= -1e-9

    def test_incomplete_rules_produce_residual_mass(self):
        """Incomplete rules leave residual (lumped) mass in the result."""
        w = np.array([1.0, 1.0])
        betas = np.array([[0.3, 0.3, 0.0], [0.2, 0.2, 0.2]])
        beta, beta_D = conjunctive_combine(w, betas)
        assert beta_D > 0.0

    def test_differs_from_er_on_incomplete_evidence(self):
        """
        The conjunctive rule lacks the ER two-part decomposition of
        unassigned mass (m_bar_D / m_hat_D) and the ER weighting, so on
        incomplete multi-rule evidence its ignorance estimate differs from
        the ER rule.  This structural difference is the mechanism behind
        the paper's ablation finding (Section 6.3).
        """
        from core.rimer import BeliefRule, ClassicalRIMER

        L = 4
        w = np.full(L, 1.0 / L)
        betas = np.tile(np.array([0.2, 0.1, 0.0]), (L, 1))  # sums to 0.3

        beta_cap, beta_D_cap = conjunctive_combine(None, betas)

        rimer = ClassicalRIMER(N=3, L=L, T=1)
        for k in range(L):
            rimer.add_rule(BeliefRule([0], betas[k], 1.0, np.array([1.0])))
        beta_er, beta_D_er = rimer.er_combine(w, betas)

        assert beta_D_cap != beta_D_er
        assert not np.allclose(beta_cap, beta_er, atol=1e-9)

    def test_underestimates_ignorance_vs_er(self):
        """
        The unweighted conjunctive rule (the paper's BF-QC basic baseline)
        systematically underestimates ignorance relative to the ER rule on
        incomplete evidence, matching Section 6.3 (22% underestimation).
        """
        from core.rimer import BeliefRule, ClassicalRIMER

        L = 8
        w = np.full(L, 1.0 / L)
        betas = np.tile(np.array([0.2, 0.1, 0.0]), (L, 1))  # sums to 0.3

        beta_cap, beta_D_cap = conjunctive_combine(None, betas)

        rimer = ClassicalRIMER(N=3, L=L, T=1)
        for k in range(L):
            rimer.add_rule(BeliefRule([0], betas[k], 1.0, np.array([1.0])))
        beta_er, beta_D_er = rimer.er_combine(w, betas)

        assert beta_D_cap < beta_D_er

    def test_weighted_variant_accepts_weights(self):
        w = np.array([0.6, 0.4])
        betas = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.1]])
        beta, beta_D = conjunctive_combine(w, betas)
        assert abs(np.sum(beta) + beta_D - 1.0) < 1e-9
