"""
End-to-end cost accounting for Q-RIMER experiments.

The paper's cost model is

    C = D_qnt * (1 / p_succ) * N_shot * c_QRAM * f_EC,

with N_shot = O((N+2) / eps_stat^2) and c_QRAM = O(log N).  This module
keeps the factors explicit so that coherent and step-wise post-selected
executions can be compared.
"""

import math


def number_of_shots(n_grades: int, epsilon_stat: float) -> float:
    """Shots required to resolve an (N+2)-component distribution."""
    return (n_grades + 2) / max(epsilon_stat ** 2, 1e-15)


def end_to_end_cost(depth: float,
                    p_success: float,
                    n_grades: int,
                    epsilon_stat: float = 0.01,
                    qram_factor: float | None = None,
                    ec_factor: float = 1.0) -> float:
    """
    Return the dimensionless end-to-end cost from the paper's formula.

    ``qram_factor`` defaults to log2(N+2); ``p_success`` is the product of
    all post-selection probabilities in a coherent run, or the per-step
    success probability when steps are re-prepared independently.
    """
    if not 0.0 < p_success <= 1.0:
        return math.inf
    shots = number_of_shots(n_grades, epsilon_stat)
    qram = math.log2(n_grades + 2) if qram_factor is None else qram_factor
    return depth * (1.0 / p_success) * shots * qram * ec_factor
