"""Gate-level QER execution for the large-N cases.

The block-encoding unitary is executed through Qiskit Statevector rather
than direct matrix application.  This is the gate-level path requested by
the reviewers; N=30 remains outside the local simulator.
"""

import time

import numpy as np

from analysis.metrics import belief_distribution_error, rank_preservation
from core.qer import QER
from experiments.datasets import SyntheticDataGenerator


def main():
    for N, L in [(10, 100), (20, 200)]:
        gen = SyntheticDataGenerator(N, L, T=4, seed=42)
        betas, _, _ = gen.generate_rules()
        w = np.random.default_rng(42).dirichlet(np.ones(L))
        qer = QER(N=N, L=L, kappa_max=50.0)
        beta_c, _ = qer.classical_simulate(w, betas)
        t0 = time.perf_counter()
        beta_q, _, info = qer.quantum_simulate(
            w, betas, gate_level=True
        )
        elapsed = time.perf_counter() - t0
        print(f"N={N}, L={L}, gate-level qubits={info['n_qubits']}, "
              f"time={elapsed:.2f}s, "
              f"error={belief_distribution_error(beta_q, beta_c):.3e}, "
              f"tau={rank_preservation(beta_q, beta_c):.3f}")


if __name__ == '__main__':
    main()
