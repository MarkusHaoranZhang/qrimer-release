"""Executable 30-register gate-level QER step.

The dense synthesis backend cannot transpile the N=30 unitary, but Aer's MPS
backend can execute the explicit StructuredQER circuit with the
block-encoding UnitaryGate kept as a gate instruction.
"""

import time

import numpy as np
from qiskit_aer import AerSimulator

from core.structured_qer import StructuredQER
from experiments.datasets import SyntheticDataGenerator


def main():
    N, L = 30, 200
    gen = SyntheticDataGenerator(N, L, T=4, seed=42)
    betas, _, _ = gen.generate_rules()
    w = np.random.default_rng(42).dirichlet(np.ones(L))
    qer = StructuredQER(N, L, T=4)
    m_I = qer.encoder._mass_vector(w[0], betas[0])
    m_next = qer.encoder._mass_vector(w[1], betas[1])
    qc = qer.step_circuit(m_I, m_next)
    qc.measure_all()
    t0 = time.perf_counter()
    sim = AerSimulator(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=64,
    )
    result = sim.run(qc, shots=1024, seed_simulator=42).result()
    elapsed = time.perf_counter() - t0
    counts = result.get_counts()
    print(f"N={N}, L={L}, registers={qc.num_qubits}, "
          f"MPS shots=1024, time={elapsed:.2f}s, distinct={len(counts)}")
    print("Gate-level N=30 step executed.")


if __name__ == '__main__':
    main()
