"""
Large-N circuit-executed Q-RIMER (full statevector range).

Runs the Hermitian-dilation block encoding across all L rules for the
paper's N=10, L=100 and N=20, L=200 synthetic cases.  Small instances use
Qiskit circuit execution; these larger instances apply the identical
unitary through the exact direct statevector path to avoid dense-gate
transpilation.  N=30 and above remain the truncated regime.
"""

import os
import time

import numpy as np

from analysis.metrics import belief_distribution_error, rank_preservation
from core.qer import QER
from experiments.datasets import SyntheticDataGenerator


def main():
    print("=" * 76)
    print("Large-N circuit-executed Q-RIMER (full statevector)")
    print("=" * 76)
    results = {}
    for N, L in [(10, 100), (20, 200)]:
        gen = SyntheticDataGenerator(N, L, T=4, seed=42)
        betas, _, _ = gen.generate_rules()
        rng = np.random.default_rng(42)
        w = rng.dirichlet(np.ones(L))
        qer = QER(N=N, L=L, kappa_max=50.0)

        t0 = time.perf_counter()
        beta_c, beta_D_c = qer.classical_simulate(w, betas)
        beta_q, beta_D_q, info = qer.quantum_simulate(w, betas)
        elapsed = time.perf_counter() - t0

        eps = belief_distribution_error(beta_q, beta_c)
        tau = rank_preservation(beta_q, beta_c)
        print(f"\n[N={N}, L={L}]")
        print(f"  qubits / steps       : {info['n_qubits']} / {info['steps']}")
        print(f"  execution path       : {info['execution']}")
        print(f"  epsilon_beta         : {eps:.3e}")
        print(f"  tau                  : {tau:.3f}")
        print(f"  effective success    : "
              f"min={min(info['per_step_success']):.4f}, "
              f"mean={info['success_probability_mean']:.4f}, "
              f"product={info['coherent_success_product']:.3e}")
        print(f"  raw success product  : "
              f"{info['raw_coherent_success_product']:.3e}")
        print(f"  wall time (s)        : {elapsed:.2f}")

        results[f'N{N}_L{L}'] = {
            'beta_c': beta_c, 'beta_q': beta_q,
            'beta_D_c': beta_D_c, 'beta_D_q': beta_D_q,
            'eps': eps, 'tau': tau,
            'qubits': info['n_qubits'],
            'per_step_success': np.array(info['per_step_success']),
            'raw_coherent_success_product': info['raw_coherent_success_product'],
        }

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'large_n_quantum_results.npz')
    np.savez(out_path, **results)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
