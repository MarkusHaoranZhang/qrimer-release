"""Mechanical audit of AE/reviewer-facing claims and artifacts."""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    paper_path = os.path.join(root, '..', '..', 'latex_ipm',
                              'qrimer_manuscript_ipm_20261025.tex')
    paper = open(paper_path, encoding='utf-8').read()

    checks = []
    checks.append(('title_conditional',
                   'Conditional Per-Step' in paper))
    checks.append(('theorem_single',
                   paper.count(r'\begin{theorem}[Conditional QER complexity]') == 1))
    checks.append(('dilation_dimension',
                   r'(N+2)\cdot(N+3)' in paper))
    checks.append(('kappa_be_defined', r'\kappa_{\mathrm{BE}}' in paper))
    checks.append(('gamma_acc_defined', r'\Gamma_{\mathrm{acc}}' in paper))
    checks.append(('ube_defined', r'U_{\mathrm{BE}}' in paper))
    checks.append(('shot_cost', r'N_{\mathrm{sht}}' in paper))
    checks.append(('ec_cost', r'f_{\mathrm{EC}}' in paper))
    checks.append(('truncation_disclosed', 'truncated' in paper))
    checks.append(('operator_noise_disclosed', 'operator-level' in paper))
    checks.append(('no_old_2156', '2156' not in paper))
    checks.append(('no_old_4621', '4621' not in paper))
    checks.append(('no_old_18_6', '18.6' not in paper))

    required = [
        'core/block_encoding.py',
        'core/sparse_block_encoding.py',
        'core/qbra_arithmetic.py',
        'core/qbra_combined.py',
        'core/qbra_div_oracle.py',
        'core/structured_qer.py',
        'experiments/run_gate_level_large_n.py',
        'experiments/run_gate_level_n30.py',
        'experiments/run_shot_execution.py',
        'experiments/run_blockencoded_noise_large_n.py',
        'experiments/run_classical_baseline.py',
        'outputs/tables/table3_structured.json',
        'outputs/tables/table10_generated.json',
        'outputs/manuscript/freeze_manifest.json',
    ]
    for rel in required:
        checks.append((f'file:{rel}', os.path.exists(os.path.join(root, rel))))

    from core.block_encoding import HermitianERBlockEncoding
    be = HermitianERBlockEncoding(5)
    checks.append(('block_encoding_unitary',
                   np.linalg.norm(be.U.conj().T @ be.U - np.eye(be.dim_total)) < 1e-9))
    checks.append(('block_encoding_single_ancilla', be.n_total - be.n_state == 1))

    from core.sparse_block_encoding import SparseBlockEncoding
    sp = SparseBlockEncoding(3)
    checks.append(('sparse_terms_six', len(sp.terms) == 6))
    checks.append(('sparse_be_unitary',
                   np.linalg.norm(sp.U_BE.conj().T @ sp.U_BE
                                  - np.eye(sp.U_BE.shape[0])) < 1e-9))

    import inspect

    from core.qer import QER
    sig = inspect.signature(QER.quantum_simulate)
    checks.append(('gate_level_flag', 'gate_level' in sig.parameters))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
