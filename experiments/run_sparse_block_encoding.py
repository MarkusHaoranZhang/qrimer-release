"""
Six-term sparse LCU block encoding validation.

Runs the complete sparse block encoding (row-side LCU plus state ancilla)
for small N and compares one ER step with the classical reference.
"""

import time

import numpy as np

from core.qer import QER
from core.sparse_block_encoding import SparseBlockEncoding


def main():
    print("=" * 74)
    print("Six-term sparse LCU block encoding (one ER step)")
    print("=" * 74)
    print(f"  {'N':>3}  {'terms':>6}  {'qubits':>7}  {'success':>9}  "
          f"{'step error':>11}  {'build(s)':>9}")
    for N in (2, 3, 5):
        t0 = time.perf_counter()
        be = SparseBlockEncoding(N)
        build_time = time.perf_counter() - t0
        qer = QER(N=N, L=2, kappa_max=50.0)
        rng = np.random.default_rng(N)
        betas = rng.dirichlet(np.ones(N), size=2) * 0.9
        w = np.array([0.5, 0.5])
        m_I = qer._mass_vector(w[0], betas[0])
        m_next = qer._mass_vector(w[1], betas[1])
        result = be.step(m_I, m_next)
        expected = qer._er_combine_step(m_I, m_next)
        err = float(np.linalg.norm(result['m_new'] - expected))
        report = be.resource_report()
        print(f"  {N:>3}  {report['terms']:>6}  {report['total_qubits']:>7}  "
              f"{result['success_probability']:>9.4f}  {err:>11.2e}  "
              f"{build_time:>9.3f}")


if __name__ == '__main__':
    main()
