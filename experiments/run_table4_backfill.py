"""Backfill Table 4 (N=3 noise response) from the executable noise probe."""

import json
import os

import numpy as np

from analysis.metrics import belief_distribution_error, rank_preservation
from core.noise import run_noisy_target_state

FACTORS = (0.5, 1.0, 2.0, 5.0, 10.0)


def main():
    beta = np.array([0.2685, 0.6772, 0.0543])
    rows = []
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
            'N': 3,
            'factor': factor,
            'epsilon_mean': float(np.mean(eps)),
            'epsilon_std': float(np.std(eps)),
            'tau_mean': float(np.mean(taus)),
        })
    root = os.path.join(os.path.dirname(__file__), '..')
    out_dir = os.path.join(root, 'outputs', 'manuscript')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'table4_generated.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(rows, fh, indent=2)
    lines = [
        r'\begin{table}[h]', r'\small', r'\centering',
        r'\caption{Noise response on the N=3 benchmark (generated)}',
        r'\label{tab:noise}', r'\begin{tabular}{ccc}', r'\hline',
        r'Scale&$\varepsilon_\beta$&$\tau$\\', r'\hline',
    ]
    for row in rows:
        lines.append(f"{row['factor']:.1f}&"
                     f"{row['epsilon_mean']:.4f}$\\pm${row['epsilon_std']:.4f}&"
                     f"{row['tau_mean']:.2f}\\\\")
    lines += [r'\hline', r'\end{tabular}', r'\end{table}', '']
    with open(os.path.join(out_dir, 'table4_generated.tex'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f"Saved Table 4 to {out_dir}")
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
