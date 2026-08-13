"""
Unit tests for analysis.metrics and analysis.statistics.
"""

import numpy as np
import pytest

from analysis.metrics import (
    belief_distribution_error,
    expected_utility,
    ignorance_fidelity,
    rank_preservation,
    theoretical_speedup,
)
from analysis.statistics import (
    bonferroni_correction,
    confidence_interval,
    linear_regression_r2,
    paired_ttest,
)


class TestMetrics:
    def test_bde_identical(self):
        b = np.array([0.5, 0.3, 0.2])
        assert belief_distribution_error(b, b) == pytest.approx(0.0, abs=1e-12)

    def test_bde_known_value(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert belief_distribution_error(a, b) == pytest.approx(np.sqrt(2), abs=1e-9)

    def test_bde_nonnegative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.dirichlet(np.ones(4))
            b = rng.dirichlet(np.ones(4))
            assert belief_distribution_error(a, b) >= 0.0

    def test_rank_preservation_identical(self):
        b = np.array([0.5, 0.3, 0.2])
        assert rank_preservation(b, b) == pytest.approx(1.0, abs=1e-9)

    def test_rank_preservation_reversed(self):
        b = np.array([0.5, 0.3, 0.2])
        assert rank_preservation(b[::-1], b) == pytest.approx(-1.0, abs=1e-9)

    def test_ignorance_fidelity_exact(self):
        assert ignorance_fidelity(0.1, 0.1) == pytest.approx(0.0, abs=1e-6)

    def test_ignorance_fidelity_zero_classical(self):
        # beta_D_c = 0 → denominator = 1e-6
        val = ignorance_fidelity(0.05, 0.0)
        assert val == pytest.approx(0.05 / 1e-6, abs=1.0)

    def test_theoretical_speedup(self):
        assert theoretical_speedup(100.0, 10) == pytest.approx(10.0, abs=1e-9)

    def test_theoretical_speedup_zero_depth(self):
        # D_quantum=0 → clamped to 1
        assert theoretical_speedup(100.0, 0) == pytest.approx(100.0, abs=1e-9)

    def test_expected_utility_known_value(self):
        beta = np.array([0.5, 0.3, 0.2])
        utils = np.array([1.0, 0.5, 0.0])
        assert expected_utility(beta, utils) == pytest.approx(0.65, abs=1e-9)

    def test_expected_utility_single_grade(self):
        beta = np.array([1.0, 0.0, 0.0])
        utils = np.array([2.0, 1.0, 0.0])
        assert expected_utility(beta, utils) == pytest.approx(2.0, abs=1e-9)


class TestStatistics:
    def test_paired_ttest_identical(self):
        # Test that clearly non-significant differences yield high p-value.
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, 30)
        b = a + rng.normal(0.0, 0.001, 30)  # tiny random noise
        t, p = paired_ttest(a, b)
        assert p > 0.05  # no significant difference expected

    def test_paired_ttest_significant(self):
        # Clearly different samples (with real variance) should yield low p-value.
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, 20)
        b = a + 10.0 + rng.normal(0.0, 0.1, 20)
        t, p = paired_ttest(a, b)
        assert p < 0.001

    def test_bonferroni_correction_length(self):
        pvals = [0.01, 0.05, 0.1]
        corrected = bonferroni_correction(pvals)
        assert len(corrected) == len(pvals)

    def test_bonferroni_correction_capped(self):
        corrected = bonferroni_correction([0.5, 0.5, 0.5])
        assert all(c <= 1.0 for c in corrected)

    def test_confidence_interval_mean(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, ci = confidence_interval(data)
        assert mean == pytest.approx(3.0, abs=1e-9)
        assert ci > 0.0

    def test_linear_regression_r2_perfect(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        r2 = linear_regression_r2(x, y)
        assert r2 == pytest.approx(1.0, abs=1e-9)

    def test_linear_regression_r2_range(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, 50)
        y = rng.uniform(0, 10, 50)
        r2 = linear_regression_r2(x, y)
        assert -1.0 <= r2 <= 1.0 + 1e-9
