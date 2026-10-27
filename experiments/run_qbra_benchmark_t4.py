"""T=4, L=36 QBRA -> QER benchmark path with 4-bit products."""

import os
import time

import numpy as np

from analysis.metrics import belief_distribution_error
from analysis.run_logger import log_run
from core.block_encoding import HermitianERBlockEncoding
from core.qbra_arithmetic import QBRAArithmetic
from core.qer import QER


def main():
    rng = np.random.default_rng(42)
    N, L = 3, 36
    alphas = rng.dirichlet(np.ones(4), size=L)
    thetas = rng.uniform(0.5, 1.0, size=L)
    betas = rng.dirichlet(np.ones(N), size=L)

    qbra = QBRAArithmetic(bits=3, scale=8)
    t0 = time.perf_counter()
    w_q = qbra.activation_weights_t4(alphas, thetas)
    activation_time = time.perf_counter() - t0

    prod = np.prod(alphas, axis=1)
    raw = thetas * prod
    w_c = raw / raw.sum()

    qer = QER(N=N, L=L, kappa_max=50.0)
    beta_q, _, info = qer.quantum_simulate(w_q, betas)
    beta_c, _ = qer.classical_simulate(w_c, betas)

    beta_shot, _, _ = qer.quantum_simulate(
        w_q, betas, shots=10 ** 4, rng=np.random.default_rng(42)
    )
    beta_noisy, _, noise_info = HermitianERBlockEncoding(N).noisy_simulate(
        w_q, betas, noise=0.01
    )

    print(f"QBRA arithmetic time : {activation_time:.2f} s")
    print(f"activation error     : {np.linalg.norm(w_q - w_c):.4e}")
    print(f"QER epsilon_beta     : {belief_distribution_error(beta_q, beta_c):.4e}")
    print(f"shot epsilon_beta    : {belief_distribution_error(beta_shot, beta_c):.4e}")
    print(f"noise epsilon_beta   : {belief_distribution_error(beta_noisy, beta_c):.4e}")
    print(f"noise success        : {noise_info['coherent_success_product']:.3e}")
    print(f"qubits               : {info['n_qubits']}")
    print(f"steps                : {info['steps']}")
    log_run(
        os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs'),
        'qbra_benchmark_t4', 42, 10 ** 4, 0.01,
        {
            'activation_error': float(np.linalg.norm(w_q - w_c)),
            'qer_epsilon': float(belief_distribution_error(beta_q, beta_c)),
            'shot_epsilon': float(belief_distribution_error(beta_shot, beta_c)),
            'noise_epsilon': float(belief_distribution_error(beta_noisy, beta_c)),
        },
        n_qubits=info['n_qubits'], steps=info['steps'],
    )


if __name__ == '__main__':
    main()
