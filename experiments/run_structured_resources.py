"""
Generate reproducible structured-QER resource tables.

Each row is measured from one explicit register-level block-encoding step
transpiled to {u, cx}.  Full-pipeline depth is the one-step depth times L-1;
it is labelled as an estimate, not as a compiled 200-step circuit.
"""

import json
import os

import numpy as np

from core.structured_qer import StructuredQER
from experiments.datasets import SyntheticDataGenerator


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'tables')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for N, L in [(3, 36), (5, 50), (10, 100), (15, 200), (20, 200)]:
        gen = SyntheticDataGenerator(N, L, T=4, seed=42)
        betas, _, _ = gen.generate_rules()
        rng = np.random.default_rng(42)
        w = rng.dirichlet(np.ones(L))
        qer = StructuredQER(N, L, T=4)
        m_I = qer.encoder._mass_vector(w[0], betas[0])
        m_next = qer.encoder._mass_vector(w[1], betas[1])
        try:
            row = qer.measured_depth(m_I, m_next)
            row['status'] = 'transpiled'
        except BaseException as exc:
            row = {
                'N': N, 'L': L, 'qubits': qer.total_qubits,
                'state_qubits': qer.n_state, 'rule_qubits': qer.n_rule,
                'antecedent_qubits': qer.n_ant,
                'clock_qubits': qer.n_clock, 'ancilla_qubits': qer.n_anc,
                'step_depth': None, 'step_size': None,
                'steps': max(L - 1, 1),
                'estimated_full_depth': None,
                'status': 'transpile_failed',
                'error': type(exc).__name__,
            }
        row['source'] = 'structured_qer_one_step'
        rows.append(row)
        depth = (row['step_depth'] if row['step_depth'] is not None else -1)
        print(f"N={N:>2} L={L:>3} qubits={row['qubits']:>2} "
              f"step_depth={depth:>9} status={row['status']}")

    path = os.path.join(out_dir, 'table3_structured.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
