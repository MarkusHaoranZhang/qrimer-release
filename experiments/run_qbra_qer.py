"""Executable QBRA -> QER path for a T=2, L=2 case.

Activation is produced by the quantum arithmetic primitives (product and
division oracles), then passed to the block-encoded QER engine.  This is a
small end-to-end validation of the handoff, not the full T=4 benchmark.
"""

import numpy as np

from analysis.metrics import belief_distribution_error
from core.qbra_arithmetic import QBRAArithmetic
from core.qer import QER


def main():
    alphas = [(0.8, 0.7), (0.4, 0.3)]
    thetas = np.array([0.8, 0.6])
    qbra = QBRAArithmetic(bits=3, scale=8)
    w = qbra.activation_weights_t2(alphas, thetas)

    rng = np.random.default_rng(42)
    betas = rng.dirichlet(np.ones(3), size=2)
    qer = QER(N=3, L=2, kappa_max=50.0)
    beta_q, beta_D_q, info = qer.quantum_simulate(w, betas)
    beta_c, beta_D_c = qer.classical_simulate(w, betas)
    print("Quantum QBRA weights:", np.round(w, 6))
    print("QER beta (quantum)   :", np.round(beta_q, 6))
    print("Classical reference  :", np.round(beta_c, 6))
    print("epsilon_beta         :", belief_distribution_error(beta_q, beta_c))
    print("qubits               :", info['n_qubits'])


if __name__ == '__main__':
    main()
