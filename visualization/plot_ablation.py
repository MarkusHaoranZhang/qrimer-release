"""
Figure: Ablation study — contribution of each component.

Attempts to load results from experiments/run_ablation.py; falls back
to paper-reported values for standalone execution.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to run ablation computations for live data
try:
    from experiments.run_ablation import (
        ablation_er_matrix,
        ablation_pipeline,
        ablation_qbra_benchmark,
    )
    q_res = ablation_qbra_benchmark(n_runs=5)
    er_res = ablation_er_matrix(n_runs=5)
    pl_res = ablation_pipeline(n_runs=20)
    epsilon = [
        q_res['eps_with_qbra'],        # Full Q-RIMER
        q_res['eps_without_qbra'],      # Without QBRA (classical weights)
        er_res['epsilon_beta'],         # Without ER matrix
        pl_res['two_layer']['hybrid_eps'],  # Hybrid pipeline (measured)
    ]
    print("Loaded live ablation results.")
except Exception:
    # Fall back to paper-reported values
    epsilon = [0.0027, 0.0025, 0.0472, 0.0091]
    print("Using paper-reported ablation values.")

conditions = ['Full Q-RIMER', 'Without QBRA', 'Without ER matrix', 'Hybrid pipeline']
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(conditions, epsilon, color=colors, alpha=0.85)

for bar, val in zip(bars, epsilon):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(epsilon) * 0.02,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=10)

ax.set_ylabel(r'Belief distribution error $\varepsilon_\beta$')
ax.set_title('Ablation study: contribution of each component')
ax.set_ylim(0, max(epsilon) * 1.25)
plt.xticks(rotation=15)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'fig_ablation.pdf')
plt.savefig(out_path, dpi=120)
plt.close()
print(f"Saved: {out_path}")
