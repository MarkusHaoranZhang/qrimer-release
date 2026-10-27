"""
Noise run of the actual block-encoded ER circuit.

Builds one block-encoded ER step (N=3), transpiles it to {u, cx}, and runs
the Aer Falcon noise model with 10^4 shots.  The output compares the
post-selected block-encoding distribution against the noiseless ideal.

This is a single-step block-encoded noise experiment; the full large-N
inference noise probe remains in run_noise_scaling.py.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate

from core.block_encoding import HermitianERBlockEncoding
from core.noise import create_falcon_noise_model
from core.qer import QER


def ideal_distribution(N):
    be = HermitianERBlockEncoding(N)
    qer = QER(N=N, L=2, kappa_max=50.0)
    rng = np.random.default_rng(7)
    betas = rng.dirichlet(np.ones(N), size=2) * 0.9
    w = np.array([0.5, 0.5])
    m_I = qer._mass_vector(w[0], betas[0])
    m_next = qer._mass_vector(w[1], betas[1])
    psi, _ = be._prepare_input(m_I, m_next)
    data = be.U @ psi
    p = np.sum(np.abs(data[:be.d]) ** 2)
    return np.abs(data[:be.d]) ** 2 / p, be, psi


def main():
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        print("qiskit-aer unavailable")
        return

    N = 3
    ideal, be, psi = ideal_distribution(N)
    print("=" * 72)
    print("Block-encoded single-step noise (N=3, 10^4 shots)")
    print("=" * 72)
    print(f"  {'scale':>7}  {'success':>9}  {'L1 error':>10}")

    for scale in (0.0, 1.0, 5.0):
        qc = QuantumCircuit(be.n_total)
        qc.append(UnitaryGate(be._prep_unitary(psi)), range(be.n_total))
        qc.append(UnitaryGate(be.U), range(be.n_total))
        qc.measure_all()
        trans = transpile(qc, basis_gates=['u', 'cx'], optimization_level=0)
        noise = create_falcon_noise_model(scale)
        backend = AerSimulator(noise_model=noise) if noise else AerSimulator()
        counts = backend.run(trans, shots=10**4).result().get_counts()
        hist = np.zeros(be.d)
        success = 0
        for bits, count in counts.items():
            idx = int(bits, 2)
            anc = idx >> be.n_state
            state = idx & ((1 << be.n_state) - 1)
            if anc == 0 and state < be.d:
                hist[state] += count
                success += count
        if success:
            hist /= success
            err = np.sum(np.abs(hist - ideal))
        else:
            err = float('nan')
        print(f"  {scale:>7.1f}  {success / 10**4:>9.4f}  {err:>10.4f}")


if __name__ == '__main__':
    main()
