"""
Large-N block-encoded noise probe.

Applies the full Hermitian block encoding at N=10 and N=20, then a local
single-qubit depolarizing channel to the encoded density matrix.  The
post-selected top block is compared with the noiseless classical ER step.
This is an operator-level noise model for the complete block encoding, not a
target-state proxy.
"""

import os

import numpy as np

from analysis.metrics import belief_distribution_error
from core.block_encoding import HermitianERBlockEncoding
from core.qer import QER
from experiments.datasets import SyntheticDataGenerator


def main():
    print("=" * 76)
    print("Large-N block-encoded noise (full Hermitian dilation)")
    print("=" * 76)
    results = {}
    for N, L in [(10, 100), (20, 200)]:
        gen = SyntheticDataGenerator(N, L, T=4, seed=42)
        betas, _, _ = gen.generate_rules()
        rng = np.random.default_rng(42)
        w = rng.dirichlet(np.ones(L))
        qer = QER(N=N, L=L, kappa_max=50.0)
        encoder = HermitianERBlockEncoding(N)
        beta_c, _ = qer.classical_simulate(w, betas)
        print(f"\n[N={N}, qubits={encoder.n_total}, L={L}]")
        for noise in [0.0, 0.001, 0.005, 0.01]:
            beta_n, _, info = encoder.noisy_simulate(w, betas, noise=noise)
            err = belief_distribution_error(beta_n, beta_c)
            results[f'N{N}_p{noise}'] = {
                'beta': beta_n,
                'error': err,
                'success_product': info['coherent_success_product'],
                'noise': noise,
                'qubits': encoder.n_total,
            }
            print(f"  p={noise:<6} success={info['coherent_success_product']:.3e} "
                  f"epsilon_beta={err:.4e}")

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'blockencoded_noise_large_n.npz')
    np.savez(out_path, **results)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
