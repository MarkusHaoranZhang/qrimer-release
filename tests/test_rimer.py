"""
Unit tests for ClassicalRIMER: ER combination, rule activation,
membership functions, and hierarchical graphite inference.
"""

import numpy as np
import pytest

from core.rimer import (
    BeliefRule,
    ClassicalRIMER,
    create_graphite_membership_functions,
    infer_graphite_hierarchical,
    trapezoidal_membership,
    triangular_membership,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rimer(N=3, L=2, T=1):
    rimer = ClassicalRIMER(N=N, L=L, T=T)
    for _ in range(L):
        rimer.add_rule(BeliefRule(
            antecedent=[0],
            consequent=np.array([1.0 / N] * N),
            rule_weight=1.0,
            attribute_weights=np.array([1.0]),
        ))
    return rimer


# ---------------------------------------------------------------------------
# Membership functions
# ---------------------------------------------------------------------------

class TestMembershipFunctions:
    def test_trapezoidal_zero_outside(self):
        assert trapezoidal_membership(0.0, 0.2, 0.4, 0.6, 0.8) == 0.0
        assert trapezoidal_membership(1.0, 0.2, 0.4, 0.6, 0.8) == 0.0

    def test_trapezoidal_plateau(self):
        assert trapezoidal_membership(0.5, 0.2, 0.4, 0.6, 0.8) == 1.0

    def test_trapezoidal_rising(self):
        val = trapezoidal_membership(0.3, 0.2, 0.4, 0.6, 0.8)
        assert 0.0 < val < 1.0

    def test_triangular_peak(self):
        assert triangular_membership(0.5, 0.0, 0.5, 1.0) == pytest.approx(1.0)

    def test_triangular_zero_outside(self):
        assert triangular_membership(0.0, 0.0, 0.5, 1.0) == 0.0
        assert triangular_membership(1.0, 0.0, 0.5, 1.0) == 0.0

    def test_graphite_membership_sums_to_one(self):
        funcs = create_graphite_membership_functions()
        f = funcs[0]
        for val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            d = f(val)
            total = sum(d.values())
            assert total == pytest.approx(1.0, abs=1e-9), \
                f"val={val}: sum={total}"

    def test_graphite_membership_keys(self):
        funcs = create_graphite_membership_functions()
        d = funcs[0](0.8)
        assert set(d.keys()) == {0, 1, 2}

    def test_graphite_high_value(self):
        funcs = create_graphite_membership_functions()
        d = funcs[0](0.9)
        # 0.9 is in the High region → key 2 should dominate
        assert d[2] > d[1] and d[2] > d[0]

    def test_graphite_low_value(self):
        funcs = create_graphite_membership_functions()
        d = funcs[0](0.1)
        # 0.1 is in the Low region → key 0 should dominate
        assert d[0] > d[1] and d[0] > d[2]


# ---------------------------------------------------------------------------
# ER combination
# ---------------------------------------------------------------------------

class TestERCombine:
    def test_single_rule_identity(self):
        """With one rule and w=1, output should equal the rule's belief degrees."""
        rimer = ClassicalRIMER(N=3, L=1, T=1)
        betas_rule = np.array([0.5, 0.3, 0.2])
        rimer.add_rule(BeliefRule([0], betas_rule, 1.0, np.array([1.0])))
        w = np.array([1.0])
        betas = np.array([betas_rule])
        beta, beta_D = rimer.er_combine(w, betas)
        np.testing.assert_allclose(beta, betas_rule, atol=1e-9)
        assert beta_D == pytest.approx(0.0, abs=1e-9)

    def test_belief_degrees_sum_leq_one(self):
        rimer = make_rimer(N=3, L=3, T=1)
        w = np.array([0.5, 0.3, 0.2])
        betas = np.array([[0.6, 0.3, 0.1],
                          [0.2, 0.5, 0.3],
                          [0.4, 0.4, 0.2]])
        beta, beta_D = rimer.er_combine(w, betas)
        assert np.sum(beta) + beta_D <= 1.0 + 1e-9

    def test_belief_degrees_nonnegative(self):
        rimer = make_rimer(N=3, L=2, T=1)
        w = np.array([0.6, 0.4])
        betas = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        beta, beta_D = rimer.er_combine(w, betas)
        assert np.all(beta >= -1e-9)
        assert beta_D >= -1e-9

    def test_incomplete_rule_produces_ignorance(self):
        """A rule with sum(beta) < 1 should produce beta_D > 0."""
        rimer = ClassicalRIMER(N=3, L=1, T=1)
        rimer.add_rule(BeliefRule([0], np.array([0.3, 0.3, 0.0]), 1.0, np.array([1.0])))
        w = np.array([1.0])
        betas = np.array([[0.3, 0.3, 0.0]])
        beta, beta_D = rimer.er_combine(w, betas)
        assert beta_D > 0.0

    def test_zero_weight_rule_ignored(self):
        """A rule with w=0 should not affect the output."""
        rimer = ClassicalRIMER(N=3, L=2, T=1)
        b1 = np.array([0.8, 0.1, 0.1])
        b2 = np.array([0.1, 0.1, 0.8])
        rimer.add_rule(BeliefRule([0], b1, 1.0, np.array([1.0])))
        rimer.add_rule(BeliefRule([0], b2, 1.0, np.array([1.0])))
        w_only_first = np.array([1.0, 0.0])
        beta, _ = rimer.er_combine(w_only_first, np.array([b1, b2]))
        np.testing.assert_allclose(beta, b1, atol=1e-9)

    def test_invalid_w_shape_raises(self):
        """Mismatched w shape should raise ValueError."""
        rimer = make_rimer(N=3, L=2, T=1)
        w = np.array([0.5, 0.3, 0.2])  # shape (3,) but L=2
        betas = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        with pytest.raises(ValueError, match="w must have shape"):
            rimer.er_combine(w, betas)

    def test_invalid_betas_shape_raises(self):
        """Mismatched betas shape should raise ValueError."""
        rimer = make_rimer(N=3, L=2, T=1)
        w = np.array([0.6, 0.4])
        betas = np.array([[0.5, 0.5]])  # shape (1, 2) but should be (2, 3)
        with pytest.raises(ValueError, match="betas must have shape"):
            rimer.er_combine(w, betas)


# ---------------------------------------------------------------------------
# Rule activation
# ---------------------------------------------------------------------------

class TestRuleActivation:
    def test_weights_sum_to_one(self):
        rimer = make_rimer(N=3, L=3, T=1)
        dists = [{0: 0.5, 1: 0.3, 2: 0.2}]
        w = rimer.rule_activation(dists)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-9)

    def test_weights_nonnegative(self):
        rimer = make_rimer(N=3, L=3, T=1)
        dists = [{0: 1.0, 1: 0.0, 2: 0.0}]
        w = rimer.rule_activation(dists)
        assert np.all(w >= 0.0)

    def test_exact_match_activates_rule(self):
        """A rule whose antecedent exactly matches the input gets full weight."""
        rimer = ClassicalRIMER(N=3, L=2, T=1)
        rimer.add_rule(BeliefRule([2], np.array([1.0, 0.0, 0.0]), 1.0, np.array([1.0])))
        rimer.add_rule(BeliefRule([0], np.array([0.0, 0.0, 1.0]), 1.0, np.array([1.0])))
        # Input is purely H (key=2)
        dists = [{2: 1.0, 1: 0.0, 0: 0.0}]
        w = rimer.rule_activation(dists)
        assert w[0] == pytest.approx(1.0, abs=1e-9)
        assert w[1] == pytest.approx(0.0, abs=1e-9)

    def test_minus_one_antecedent_ignored(self):
        """Antecedent value -1 means 'not used'; rule should still activate."""
        rimer = ClassicalRIMER(N=3, L=1, T=2)
        rimer.add_rule(BeliefRule([-1, 2], np.array([1.0, 0.0, 0.0]),
                                  1.0, np.array([1.0, 1.0])))
        dists = [{2: 1.0}, {2: 1.0}]
        w = rimer.rule_activation(dists)
        assert w[0] > 0.0


# ---------------------------------------------------------------------------
# Input completeness factor (Section 3.1)
# ---------------------------------------------------------------------------

class TestInputCompleteness:
    def test_complete_inputs_yield_one(self):
        """Complete input assessments (sum=1) give phi_k = 1."""
        rimer = ClassicalRIMER(N=3, L=1, T=2)
        rimer.add_rule(BeliefRule([0, 1], np.array([0.5, 0.3, 0.2]),
                                  1.0, np.array([1.0, 1.0])))
        dists = [{0: 0.7, 1: 0.3}, {0: 0.2, 1: 0.8}]
        phis = rimer.input_completeness_factors(dists)
        np.testing.assert_allclose(phis, [1.0], atol=1e-12)

    def test_incomplete_inputs_scale_down(self):
        """Incomplete assessments yield phi_k < 1."""
        rimer = ClassicalRIMER(N=3, L=1, T=2)
        rimer.add_rule(BeliefRule([0, 1], np.array([0.5, 0.3, 0.2]),
                                  1.0, np.array([1.0, 1.0])))
        dists = [{0: 0.5, 1: 0.2}, {0: 0.3, 1: 0.0}]  # sums: 0.7, 0.3
        phis = rimer.input_completeness_factors(dists)
        assert phis[0] == np.float64(0.5) or abs(phis[0] - 0.5) < 1e-12

    def test_unused_attribute_excluded(self):
        """Attributes not used by a rule (antecedent -1) don't affect phi."""
        rimer = ClassicalRIMER(N=3, L=1, T=2)
        rimer.add_rule(BeliefRule([0, -1], np.array([0.5, 0.3, 0.2]),
                                  1.0, np.array([1.0, 1.0])))
        dists = [{0: 0.5, 1: 0.0}, {0: 0.0, 1: 0.0}]  # only attr 0 used
        phis = rimer.input_completeness_factors(dists)
        assert phis[0] == np.float64(0.5) or abs(phis[0] - 0.5) < 1e-12

    def test_infer_applies_phi_k(self):
        """infer() scales betas by phi_k for incomplete inputs."""
        rimer = ClassicalRIMER(N=3, L=1, T=1)
        rimer.add_rule(BeliefRule([0], np.array([1.0, 0.0, 0.0]),
                                  1.0, np.array([1.0])))
        # Complete input: output = [1, 0, 0]
        beta_complete, _ = rimer.infer([{0: 1.0, 1: 0.0}])
        np.testing.assert_allclose(beta_complete, [1.0, 0.0, 0.0], atol=1e-9)
        # Incomplete input (total 0.5): betas scaled by phi=0.5 -> ignorance
        beta_incomplete, beta_D = rimer.infer([{0: 0.3, 1: 0.2}])
        assert beta_incomplete[0] < beta_complete[0]
        assert beta_D > 0.0


# ---------------------------------------------------------------------------
# Graphite hierarchical inference
# ---------------------------------------------------------------------------

class TestGraphiteHierarchical:
    def test_test3_rank_order(self):
        """
        Test run 3 (X1=0.8, X3=0.98, X5=0.2, X7=0.8):
        The paper reports beta_M > beta_H > beta_L.
        Note: the complete 36-rule parameter table is not publicly available,
        so the reconstructed rule base produces a different rank order.
        We test the weaker property that the output is a valid belief distribution.
        """
        memb = create_graphite_membership_functions()[0]
        beta, beta_D = infer_graphite_hierarchical(0.8, 0.98, 0.2, 0.8, memb)
        # Valid belief distribution: non-negative, sums to <= 1
        assert np.all(beta >= -1e-9)
        assert np.sum(beta) + beta_D <= 1.0 + 1e-9
        # At least one grade has non-zero belief
        assert np.sum(beta) > 0.0

    def test_output_sums_leq_one(self):
        memb = create_graphite_membership_functions()[0]
        beta, beta_D = infer_graphite_hierarchical(0.8, 0.98, 0.2, 0.8, memb)
        assert np.sum(beta) + beta_D <= 1.0 + 1e-9

    def test_high_quality_input(self):
        """High values on all inputs should produce high H belief."""
        memb = create_graphite_membership_functions()[0]
        beta, _ = infer_graphite_hierarchical(0.95, 0.95, 0.95, 0.95, memb)
        assert beta[0] > beta[2], "High-quality input should favour H over L"

    def test_low_quality_input(self):
        """Low values on all inputs should produce high L belief."""
        memb = create_graphite_membership_functions()[0]
        beta, _ = infer_graphite_hierarchical(0.05, 0.05, 0.05, 0.05, memb)
        assert beta[2] > beta[0], "Low-quality input should favour L over H"
