"""
Shot-based end-to-end Q-RIMER execution.

Runs the complete block-encoded pipeline with 10^4 shots at the final
readout, matching the measurement protocol described in the manuscript.  Exact and shot-based
outputs are reported side by side so the statistical error is not confused
with algorithmic error.
"""

import numpy as np

from analysis.metrics import belief_distribution_error, rank_preservation
from analysis.sampling import multinomial_shot_noise
from core.qer import QER
from core.rimer import create_graphite_membership_functions, infer_graphite_hierarchical
from experiments.datasets import BenchmarkLoader, SyntheticDataGenerator
from experiments.run_benchmark_quantum import quantum_graphite

SHOTS = 10 ** 4


def benchmark_shots():
    memb = create_graphite_membership_functions()[0]
    inputs, _, _ = BenchmarkLoader.load()
    eps = []
    taus = []
    for tid, (x1, x3, x5, x7) in enumerate(inputs, start=1):
        beta_c, _ = infer_graphite_hierarchical(x1, x3, x5, x7, memb)
        beta_q, _, _ = quantum_graphite(x1, x3, x5, x7, memb)
        beta_meas = multinomial_shot_noise(
            beta_q, np.random.default_rng(42 + tid), SHOTS
        )
        eps.append(belief_distribution_error(beta_meas, beta_c))
        taus.append(rank_preservation(beta_meas, beta_c))
    return float(np.mean(eps)), float(np.std(eps)), float(np.mean(taus))


def large_n_shots(N, L):
    gen = SyntheticDataGenerator(N, L, T=4, seed=42)
    betas, _, _ = gen.generate_rules()
    w = np.random.default_rng(42).dirichlet(np.ones(L))
    qer = QER(N=N, L=L, kappa_max=50.0)
    beta_c, _ = qer.classical_simulate(w, betas)
    beta_q, _, info = qer.quantum_simulate(w, betas)
    beta_meas = multinomial_shot_noise(
        beta_q, np.random.default_rng(42), SHOTS
    )
    return (belief_distribution_error(beta_meas, beta_c),
            rank_preservation(beta_meas, beta_c), info)


def main():
    print("=" * 76)
    print(f"Shot-based end-to-end Q-RIMER ({SHOTS} shots at readout)")
    print("=" * 76)
    eps, std, tau = benchmark_shots()
    print("\n[benchmark N=3, L=36]")
    print(f"  mean epsilon = {eps:.4e} +/- {std:.4e}")
    print(f"  mean tau     = {tau:.3f}")
    for N, L in [(10, 100), (20, 200)]:
        eps, tau, info = large_n_shots(N, L)
        print(f"\n[N={N}, L={L}]")
        print(f"  epsilon_beta = {eps:.4e}")
        print(f"  tau          = {tau:.3f}")
        print(f"  qubits       = {info['n_qubits']}")
        print(f"  success mean = {info['success_probability_mean']:.4f}")


if __name__ == '__main__':
    main()
