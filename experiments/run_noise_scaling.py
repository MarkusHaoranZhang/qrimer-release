"""
Noise-model scaling probe for larger evaluation-grade counts.

This script measures the target belief state under the IBM Falcon noise
model for N=3, 10 and 20.  It is a noise-model probe, not a block-encoded
end-to-end execution; the circuit-executed kernel is validated separately
in experiments/run_quantum_execution.py.
"""

import numpy as np

from analysis.metrics import belief_distribution_error, rank_preservation
from analysis.paper_reference import TABLE10_NOISE
from core.noise import run_noisy_target_state

FACTORS = (0.0, 1.0, 2.0, 5.0, 10.0)


def synthetic_beta(N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(N))


def main():
    print("=" * 68)
    print("Noise-model scaling probe (target belief state)")
    print("=" * 68)
    cases = [(3, np.array([0.2685, 0.6772, 0.0543]))]
    cases += [(N, synthetic_beta(N, seed=N)) for N in (10, 20)]

    for N, beta in cases:
        paper = TABLE10_NOISE[f'N{N}']
        print(f"\n[N={N}]")
        print(f"  {'factor':>8}  {'code eps':>10}  {'paper eps':>10}  {'tau':>6}")
        for factor in FACTORS:
            eps_runs = []
            tau_runs = []
            for trial in range(5):
                beta_q = run_noisy_target_state(
                    beta, 0.0, noise_scale=factor, n_shots=10**4
                )
                if beta_q is None:
                    print("  qiskit-aer unavailable; stopping")
                    return
                eps_runs.append(belief_distribution_error(beta_q, beta))
                tau_runs.append(rank_preservation(beta_q, beta))
            print(f"  {factor:>8.1f}  {np.mean(eps_runs):>10.4f}  "
                  f"{paper[factor]:>10.4f}  {np.mean(tau_runs):>6.2f}")


if __name__ == '__main__':
    main()
