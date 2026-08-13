"""
Statistical utilities for experimental results.
"""

import numpy as np
from scipy import stats


def paired_ttest(sample1: np.ndarray, sample2: np.ndarray) -> tuple:
    """
    Perform paired t-test.

    Returns
    -------
    (t_statistic, p_value)
    """
    t_stat, p_val = stats.ttest_rel(sample1, sample2)
    return float(t_stat), float(p_val)


def bonferroni_correction(p_values: list, alpha: float = 0.05) -> list:
    """
    Apply Bonferroni correction to a list of p-values.

    Returns
    -------
    list of corrected p-values (capped at 1.0)
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Compute mean and confidence interval half-width.

    Returns
    -------
    (mean, ci_width)
    """
    n = len(data)
    mean = float(np.mean(data))
    sem = stats.sem(data)
    ci_width = float(sem * stats.t.ppf((1 + confidence) / 2, n - 1))
    return mean, ci_width


def linear_regression_r2(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute R-squared for linear regression y ~ x.

    Returns
    -------
    float
    """
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-15))
