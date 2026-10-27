"""
Post-selection and accumulated-error accounting.

The functions keep the distinction between a fully coherent run (one global
success probability) and a step-wise re-prepared run (per-step success and
independent repetitions) explicit.
"""

import math

import numpy as np


def coherent_success_probability(per_step: list[float]) -> float:
    """Product of the per-step post-selection probabilities."""
    values = np.asarray(per_step, dtype=float)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("success probabilities must lie in [0, 1]")
    return float(np.prod(values)) if values.size else 1.0


def expected_repetitions(success_probability: float) -> float:
    """Expected number of coherent runs per accepted output."""
    if success_probability <= 0.0:
        return math.inf
    return 1.0 / success_probability


def composed_error_bound(step_error: float,
                         per_step: list[float]) -> float:
    """
    Conservative bound for a coherent composition:

        ||error|| <= L * eps_step / sqrt(prod_k p_k).

    This is the bound used in the manuscript's error discussion; it becomes
    extremely large when the coherent post-selection product is tiny.
    """
    if step_error < 0.0:
        raise ValueError("step_error must be non-negative")
    p = coherent_success_probability(per_step)
    if p <= 0.0:
        return math.inf
    return len(per_step) * step_error / math.sqrt(p)


def recursive_cost(depth_per_step: float,
                   per_step: list[float],
                   n_grades: int,
                   epsilon_stat: float,
                   qram_factor: float | None = None,
                   ec_factor: float = 1.0,
                   coherent: bool = True) -> float:
    """
    End-to-end cost with either the coherent success product or the mean
    per-step success probability (re-prepared hybrid execution).
    """
    if coherent:
        p = coherent_success_probability(per_step)
    else:
        p = float(np.mean(per_step)) if per_step else 1.0
    shots = (n_grades + 2) / max(epsilon_stat ** 2, 1e-15)
    qram = math.log2(n_grades + 2) if qram_factor is None else qram_factor
    depth = depth_per_step * max(len(per_step), 1)
    return depth * expected_repetitions(p) * shots * qram * ec_factor
