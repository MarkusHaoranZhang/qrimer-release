"""
Figure 2: Ablation study of the three proposed components.

Uses the paper-reported values in paper_figure_data.json so the figure
matches Section 6.3 and Figure 2 of the manuscript.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, 'paper_figure_data.json')
OUTPUT_DIR = os.path.join(HERE, '..', 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

with open(DATA_PATH, encoding='utf-8') as f:
    d = json.load(f)['fig2']

conditions = d['conditions']
eps = d['eps']
std = d['std']
colors = d['colors']

fig, ax = plt.subplots(figsize=(6.5, 3.2))
x = np.arange(len(conditions))
bars = ax.bar(x, eps, color=colors, alpha=0.85, width=0.6)

for bar, val, err in zip(bars, eps, std):
    cx = bar.get_x() + bar.get_width() / 2
    if err is not None:
        ax.errorbar(cx, val, yerr=err,
                    fmt='none', ecolor='black', capsize=3, elinewidth=0.9)
    y_top = val * 1.15 if err is None else (val + err) * 1.08
    ax.text(cx, y_top, f'{val:.4f}', ha='center', va='bottom', fontsize=7.5)

y_ns = 0.006
ax.annotate('', xy=(1, y_ns), xytext=(0, y_ns),
            arrowprops=dict(arrowstyle='-', color='black', lw=1.0))
ax.text(0.5, y_ns * 1.25, 'ns ($p = 0.42$)',
        ha='center', va='bottom', fontsize=7.5)
ax.text(2, 0.072, '×15 error', ha='center', va='bottom',
        fontsize=8, color='#8A1C1C')

ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=0.9, alpha=0.6)

ax.set_ylabel(r'Belief distribution error $\varepsilon_\beta$ (log scale)')
ax.set_xticks(x, labels=conditions)
ax.set_yscale('log')
ax.set_ylim(5e-4, 0.15)

ax.annotate('Benchmark ($N=3$)', xy=(1.0, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -18), textcoords='offset points',
            ha='center', va='top', fontsize=7.5, color='gray')
ax.annotate('2-layer hierarchical', xy=(3.5, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -18), textcoords='offset points',
            ha='center', va='top', fontsize=7.5, color='gray')

plt.tight_layout(pad=0.6)
out_base = os.path.join(OUTPUT_DIR, 'ablation')
plt.savefig(out_base + '.pdf', dpi=600)
plt.savefig(out_base + '.png', dpi=600)
plt.close()
print(f"Saved: {out_base}.pdf, {out_base}.png")
