"""Generate backfill values for Table 2 and Table 5 from executable code."""

import json
import os

import numpy as np

from analysis.metrics import belief_distribution_error, rank_preservation
from core.noise import run_noisy_target_state
from experiments.run_comparative import fidelity_comparison
from experiments.run_noise_scaling import FACTORS, synthetic_beta


def table2():
    fid = fidelity_comparison()
    rows = []
    for i, N in enumerate(fid['N_values']):
        qm, qs = fid['qrimer'][i]
        bm, bs = fid['bfqc'][i]
        rows.append({
            'N': N, 'qrimer_mean': qm, 'qrimer_std': qs,
            'bfqc_mean': bm, 'bfqc_std': bs,
            'p_bonferroni': fid['p_bonferroni'][i],
        })
    return rows


def table5():
    cases = [(3, np.array([0.2685, 0.6772, 0.0543]))]
    cases += [(N, synthetic_beta(N, seed=N)) for N in (10, 20)]
    rows = []
    for N, beta in cases:
        for factor in FACTORS:
            eps, taus = [], []
            for trial in range(5):
                beta_q = run_noisy_target_state(
                    beta, 0.0, noise_scale=factor, n_shots=10 ** 4
                )
                if beta_q is None:
                    continue
                eps.append(belief_distribution_error(beta_q, beta))
                taus.append(rank_preservation(beta_q, beta))
            rows.append({
                'N': N, 'factor': factor,
                'epsilon_mean': float(np.mean(eps)),
                'epsilon_std': float(np.std(eps)),
                'tau_mean': float(np.mean(taus)),
            })
    return rows


def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    out_dir = os.path.join(root, 'outputs', 'manuscript')
    os.makedirs(out_dir, exist_ok=True)

    t2 = table2()
    with open(os.path.join(out_dir, 'table2_generated.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(t2, fh, indent=2)
    lines = [
        r'\begin{table}[h]', r'\small', r'\centering',
        r'\caption{Belief distribution error across problem sizes (generated)}',
        r'\label{tab:comp}', r'\begin{tabular}{cccc}', r'\hline',
        r'$N$&Q-RIMER&BF-QC basic&$p$ (Bonferroni)\\', r'\hline',
    ]
    for row in t2:
        lines.append(
            f"{row['N']}&{row['qrimer_mean']:.4f}$\\pm${row['qrimer_std']:.4f}&"
            f"{row['bfqc_mean']:.4f}$\\pm${row['bfqc_std']:.4f}&"
            f"{row['p_bonferroni']:.2e}\\\\"
        )
    lines += [r'\hline', r'\end{tabular}', r'\end{table}', '']
    with open(os.path.join(out_dir, 'table2_generated.tex'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(lines))

    t5 = table5()
    with open(os.path.join(out_dir, 'table5_generated.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(t5, fh, indent=2)
    lines = [
        r'\begin{table}[h]', r'\small', r'\centering',
        r'\caption{Noise response by evaluation-grade count (generated)}',
        r'\label{tab:noise_large}', r'\begin{tabular}{cccc}', r'\hline',
        r'$N$&Scale&$\varepsilon_\beta$&$\tau$\\', r'\hline',
    ]
    for row in t5:
        lines.append(
            f"{row['N']}&{row['factor']:.1f}&"
            f"{row['epsilon_mean']:.4f}$\\pm${row['epsilon_std']:.4f}&"
            f"{row['tau_mean']:.2f}\\\\"
        )
    lines += [r'\hline', r'\end{tabular}', r'\end{table}', '']
    with open(os.path.join(out_dir, 'table5_generated.tex'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(lines))

    print(f"Saved Table 2 and Table 5 backfill to {out_dir}")
    print("Table 2:", t2)
    print("Table 5 rows:", len(t5))


if __name__ == '__main__':
    main()
