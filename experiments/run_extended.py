"""
Extended studies: very large N and maritime safety assessment.
Corresponds to Section 6.6 of the paper.

This script:
  1. Tests Q-RIMER at N=50 and N=100 using the truncated power-set
     approximation (K_max=3), computing the actual truncation error
     epsilon_beta against the full classical ER result.
  2. Runs a synthetic maritime safety assessment (T=12, N=7, L=480)
     with safety-critical attribute weighting, classical timing, the
     truncated scheme, and depth estimates.

Paper results (Section 6.6):
  N=50:  eps_beta ~ 0.031 (truncated power-set, K_max=3)
  N=100: eps_beta ~ 0.041 (truncated power-set, K_max=3)
  Maritime: T=12, N=7, L=480, classical ~38 min, quantum ~2.2 s
            simulation, eps_beta ~ 0.019
"""

import time

import numpy as np

from analysis.metrics import belief_distribution_error
from config import RANDOM_SEED
from core.qbra import QBRA
from core.qer import QER
from core.rimer import BeliefRule, ClassicalRIMER
from experiments.datasets import SyntheticDataGenerator

# Truncation order of the power-set approximation (Section 6.6)
K_MAX = 3


def truncated_betas(betas: np.ndarray, k_max: int = K_MAX) -> np.ndarray:
    """
    Truncated power-set approximation: for each rule, retain only the
    k_max largest focal elements (belief degrees) and renormalise the
    retained mass.  This mirrors the paper's scheme of "retaining only
    focal sets of cardinality up to K_max = 3" for large-N instances.
    """
    truncated = np.zeros_like(betas)
    for k in range(betas.shape[0]):
        order = np.argsort(betas[k])[::-1][:k_max]
        truncated[k, order] = betas[k, order]
        s = truncated[k].sum()
        if s > 1e-12:
            truncated[k] /= s
    return truncated


def run_truncated_study(N: int, L: int, T: int = 4,
                        seed_offset: int = 0) -> dict:
    """
    Compute the full classical ER result and the K_max-truncated result,
    returning the truncation error epsilon_beta.
    """
    gen = SyntheticDataGenerator(N, L, T, seed=RANDOM_SEED + seed_offset,
                                 gamma=0.5, incomplete_scale=0.9)
    betas, rule_weights, attr_weights = gen.generate_rules()

    rng = np.random.default_rng(RANDOM_SEED + seed_offset + 1)
    n_ref = 3
    rimer = ClassicalRIMER(N=N, L=L, T=T)
    for k in range(L):
        ant = [int(rng.integers(0, n_ref)) for _ in range(T)]
        rimer.add_rule(BeliefRule(
            antecedent=ant,
            consequent=betas[k],
            rule_weight=float(rule_weights[k]),
            attribute_weights=attr_weights[k],
        ))
    dists = [{j: float(v) for j, v in enumerate(rng.dirichlet(np.ones(n_ref)))}
             for _ in range(T)]
    w = rimer.rule_activation(dists)

    # Full exact classical ER combination
    beta_full, beta_D_full = rimer.er_combine(w, betas)

    # Truncated scheme (K_max focal elements per rule)
    qer = QER(N=N, L=L, kappa_max=50.0)
    beta_trunc, beta_D_trunc = qer.classical_simulate(
        w, truncated_betas(betas, k_max=K_MAX))

    eps = belief_distribution_error(beta_trunc, beta_full)
    return {
        'N': N, 'L': L,
        'eps_beta': float(eps),
        'beta_D_full': float(beta_D_full),
        'beta_D_trunc': float(beta_D_trunc),
    }


def large_N_study():
    """
    Test Q-RIMER at N=50 and N=100 with the truncated power-set scheme.
    """
    print("[Very large N study]")
    print(f"  {'N':>4}  {'L':>4}  {'QER depth':>10}  {'QBRA depth':>11}  "
          f"{'eps_beta (K_max=3)':>18}")

    for i, (N, L) in enumerate([(50, 200), (100, 500)]):
        T = 4
        qer = QER(N=N, L=L, kappa_max=50.0, epsilon=0.01)
        qbra = QBRA(T=T, L=L, Tk_max=min(T, 5), epsilon=0.01)
        d_qer = qer.estimate_depth()
        d_qbra = qbra.estimate_depth()

        res = run_truncated_study(N, L, T=T, seed_offset=800 + i)
        print(f"  {N:>4}  {L:>4}  {d_qer:>10}  {d_qbra:>11}  "
              f"{res['eps_beta']:>18.4f}")
        print(f"        truncation: beta_D full={res['beta_D_full']:.4f}  "
              f"trunc={res['beta_D_trunc']:.4f}")

    print("  Paper targets: N=50: eps_beta ~ 0.031;  N=100: eps_beta ~ 0.041")
    print("  (The truncation error is the price paid for extending the")
    print("   feasible problem size beyond the simulator capacity, Section 6.6.")
    print("   Per Section 6.7, the truncated scheme is an exploratory approximation")
    print("   whose relationship with the full Q-RIMER framework requires further")
    print("   investigation on larger-scale quantum resources.)")


def maritime_scenario():
    """
    Synthetic maritime safety assessment case study (Section 6.6).
    T=12 attributes, N=7 evaluation grades, L=480 expert-defined rules.
    Safety-critical attributes receive higher weights, mirroring realistic
    maritime domain patterns.
    """
    print("\n[Maritime safety assessment]")
    T, N, L = 12, 7, 480
    safety_critical_attrs = [0, 1, 4, 6, 9]  # e.g. environment, traffic, stability

    # Generate synthetic maritime BRB
    gen = SyntheticDataGenerator(N, L, T, seed=RANDOM_SEED + 700, gamma=0.5,
                                 incomplete_scale=0.9)
    betas, rule_weights, attr_weights = gen.generate_rules()

    # Higher weights for safety-critical attributes (Section 6.6)
    attr_weights = attr_weights.copy()
    for idx in safety_critical_attrs:
        attr_weights[:, idx] = np.clip(attr_weights[:, idx] * 1.5, 0.0, 1.0)

    # Classical RIMER inference (actual computation)
    rimer = ClassicalRIMER(N=N, L=L, T=T)
    n_ref = 3
    rng = np.random.default_rng(RANDOM_SEED + 701)
    for k in range(L):
        ant = [int(rng.integers(0, n_ref)) for _ in range(T)]
        rimer.add_rule(BeliefRule(
            antecedent=ant,
            consequent=betas[k],
            rule_weight=float(rule_weights[k]),
            attribute_weights=attr_weights[k],
        ))

    dists = [{j: float(v) for j, v in enumerate(rng.dirichlet(np.ones(n_ref)))}
             for _ in range(T)]

    # Measure classical time
    t0 = time.perf_counter()
    w = rimer.rule_activation(dists)
    beta_c, beta_D_c = rimer.er_combine(w, betas)
    classical_time = time.perf_counter() - t0

    # QER truncated scheme (K_max=3) vs full classical result
    qer = QER(N=N, L=L, kappa_max=50.0)
    beta_q, beta_D_q = qer.classical_simulate(w, truncated_betas(betas))
    eps = belief_distribution_error(beta_q, beta_c)

    # Depth estimates
    qbra = QBRA(T=T, L=L, Tk_max=min(T, 5), epsilon=0.01)
    total_depth = qer.estimate_depth() + qbra.estimate_depth()

    print(f"  Configuration : T={T}, N={N}, L={L} "
          f"(safety-critical attrs: {safety_critical_attrs})")
    print(f"  Classical time (local NumPy): {classical_time*1000:.1f} ms")
    print(f"  Estimated quantum depth     : {total_depth}")
    print(f"  Truncated-scheme eps_beta   : {eps:.4f}")
    print(f"  beta_c[:3] = [{beta_c[0]:.4f}, {beta_c[1]:.4f}, {beta_c[2]:.4f}]")
    print(f"  beta_D = {beta_D_c:.4f}")
    print("  Paper targets: classical ~38 min (their full workload), quantum")
    print("  ~2.2 s simulation, eps_beta ~ 0.019.  Local timings are reported")
    print("  for feasibility only and do not constitute a speed comparison.")
    print("  No proprietary or human-subject data are involved.")


def main():
    print("=" * 60)
    print("Extended studies")
    print("=" * 60)
    large_N_study()
    maritime_scenario()


if __name__ == '__main__':
    main()
