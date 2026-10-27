"""
Scaling and crossover analysis for the depth comparison.

The empirical fit is a finite-range power law; it is not by itself evidence
of an asymptotic logarithmic term.  These helpers make the fitted range,
the extrapolated empirical crossover, and the measured/truncated status
explicit.
"""

import math

import numpy as np


def power_law(N, a: float, b: float):
    return a * np.power(N, b)


def classical_ops(N, L: int = 200):
    return L * np.asarray(N, dtype=float)


def empirical_crossover(L: int, a: float, b: float) -> float:
    """
    Solve a N^b = L N for N > 1.

    For the paper's fit (a=3.725, b=2.139, L=200) this is ~33, i.e. just
    beyond the fitted range N<=30.  It is an extrapolation of the power law,
    not evidence for a logarithmic regime.
    """
    if b <= 1.0 or a <= 0.0:
        return math.inf
    return float((L / a) ** (1.0 / (b - 1.0)))


def log_term_ratio(N, log_coefficient: float, a: float, b: float) -> float:
    """Ratio of a log component to the empirical power-law component."""
    N = np.asarray(N, dtype=float)
    log_part = log_coefficient * np.log2(N)
    power_part = power_law(N, a, b)
    return log_part / power_part
