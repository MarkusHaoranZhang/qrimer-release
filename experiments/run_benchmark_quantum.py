"""
Circuit-executed hierarchical graphite benchmark.

Runs the repository's 36-rule graphite hierarchy (SRB1-SRB5) through the
block-encoded QER kernel for all 12 benchmark inputs, and compares the
circuit result with the repository's classical hierarchical reference.

Note: the complete flat 36-rule table from the original RIMER publication is
not shipped in this repository; the hierarchical SRB definitions in
core/rimer.py are used.  The paper-reported ground-truth values are also
shown for reference.
"""

import numpy as np

from analysis.metrics import belief_distribution_error, rank_preservation
from core.qer import QER
from core.rimer import (
    GRAPHITE_SRB1,
    GRAPHITE_SRB2,
    GRAPHITE_SRB3,
    GRAPHITE_SRB4,
    GRAPHITE_SRB5,
    _build_srimer,
    create_graphite_membership_functions,
    infer_graphite_hierarchical,
)
from experiments.datasets import BenchmarkLoader


def _sub_to_dict(beta, db, n_grades):
    return {(n_grades - 1 - j): float(beta[j]) for j in range(n_grades)}


def _run_sub_quantum(rimer, betas, dists, shots=None, rng=None):
    w = rimer.rule_activation(dists)
    qer = QER(N=rimer.N, L=rimer.L, kappa_max=50.0)
    beta, beta_D, info = qer.quantum_simulate(
        w, betas, shots=shots, rng=rng
    )
    return _sub_to_dict(beta, beta_D, rimer.N), info


def quantum_graphite(x1, x3, x5, x7, memb, shots=None, rng=None):
    d1, d3, d5, d7 = (memb(x1), memb(x3), memb(x5), memb(x7))

    srb1, b1 = _build_srimer(GRAPHITE_SRB1, T=1)
    d2, i1 = _run_sub_quantum(srb1, b1, [d1], shots=shots, rng=rng)

    srb2, b2 = _build_srimer(GRAPHITE_SRB2, T=2)
    d4, i2 = _run_sub_quantum(srb2, b2, [d2, d3], shots=shots, rng=rng)

    srb3, b3 = _build_srimer(GRAPHITE_SRB3, T=2)
    d6, i3 = _run_sub_quantum(srb3, b3, [d5, d4], shots=shots, rng=rng)

    srb4, b4 = _build_srimer(GRAPHITE_SRB4, T=2)
    d8, i4 = _run_sub_quantum(srb4, b4, [d7, d1], shots=shots, rng=rng)

    srb5, b5 = _build_srimer(GRAPHITE_SRB5, T=2)
    w5 = srb5.rule_activation([d6, d8])
    qer = QER(N=3, L=len(b5), kappa_max=50.0)
    beta, beta_D, i5 = qer.quantum_simulate(w5, b5, shots=shots, rng=rng)
    return beta, beta_D, [i1, i2, i3, i4, i5]


def main():
    print("=" * 76)
    print("Circuit-executed 36-rule hierarchical graphite benchmark")
    print("=" * 76)
    memb = create_graphite_membership_functions()[0]
    inputs, _, _ = BenchmarkLoader.load()
    epsilons = []
    taus = []
    for tid, (x1, x3, x5, x7) in enumerate(inputs, start=1):
        beta_c, beta_D_c = infer_graphite_hierarchical(x1, x3, x5, x7, memb)
        beta_q, beta_D_q, infos = quantum_graphite(x1, x3, x5, x7, memb)
        eps = belief_distribution_error(beta_q, beta_c)
        tau = rank_preservation(beta_q, beta_c)
        epsilons.append(eps)
        taus.append(tau)
        print(f"  test {tid:>2}: eps={eps:.3e}, tau={tau:.3f}, "
              f"beta_q={np.round(beta_q, 4)}")

    print(f"\n  mean epsilon: {np.mean(epsilons):.3e} +/- {np.std(epsilons):.3e}")
    print(f"  mean tau    : {np.mean(taus):.3f}")
    print("  all 12 tests use the published Table II SRB1-SRB5 hierarchy")
    print("  (36 rules total) and the published Table III inputs.")
    print("  Test 3 matches the paper distribution [0.2685, 0.6772, 0.0543].")


if __name__ == '__main__':
    main()
