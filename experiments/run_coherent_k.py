"""
End-to-end coherent-K Q-RIMER run.

The block encoding is post-selected without measuring the belief state, and
the state-dependent conflict factor K is applied by an ancilla rotation
rather than by a classical amplitude rescaling.  The output is compared with
the ordinary block-encoded and classical ER pipelines.
"""

import os

import numpy as np

from analysis.metrics import belief_distribution_error, rank_preservation
from core.qer import QER
from experiments.datasets import SyntheticDataGenerator


def main():
    print("=" * 76)
    print("End-to-end coherent-K Q-RIMER")
    print("=" * 76)
    results = {}
    for N, L in [(10, 100), (20, 200)]:
        gen = SyntheticDataGenerator(N, L, T=4, seed=42)
        betas, _, _ = gen.generate_rules()
        rng = np.random.default_rng(42)
        w = rng.dirichlet(np.ones(L))
        qer = QER(N=N, L=L, kappa_max=50.0)
        beta_c, beta_D_c = qer.classical_simulate(w, betas)
        beta_q, beta_D_q, info = qer.quantum_simulate(
            w, betas, coherent_k=True
        )
        eps = belief_distribution_error(beta_q, beta_c)
        tau = rank_preservation(beta_q, beta_c)
        print(f"\n[N={N}, L={L}]")
        print(f"  qubits / steps       : {info['n_qubits']} / {info['steps']}")
        print(f"  execution            : {info['execution']}")
        print(f"  epsilon_beta         : {eps:.3e}")
        print(f"  tau                  : {tau:.3f}")
        print(f"  K amplitude product  : {info['coherent_k_product']:.3e}")
        print(f"  coherent success     : {info['coherent_success_product']:.3e}")
        results[f'N{N}_L{L}'] = {
            'beta_c': beta_c,
            'beta_q': beta_q,
            'epsilon_beta': eps,
            'tau': tau,
            'coherent_k_product': info['coherent_k_product'],
            'coherent_success_product': info['coherent_success_product'],
            'qubits': info['n_qubits'],
        }

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'coherent_k_results.npz')
    np.savez(out_path, **results)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
