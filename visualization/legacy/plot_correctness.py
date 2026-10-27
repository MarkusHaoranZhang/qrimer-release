"""
Figure 1: Correctness validation on benchmark test run 3.

Uses the paper-reported values in paper_figure_data.json so the figure
matches Section 6.2 and Figure 1 of the manuscript.
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
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

with open(DATA_PATH, encoding='utf-8') as f:
    d = json.load(f)['fig1']

categories = d['categories']
classical = d['classical']
quantum = d['quantum']
quantum_err = d['quantum_err']

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(4.6, 3.1))
bars_c = ax.bar(x - width / 2, classical, width,
                label='Classical RIMER', color='#6F6F6F')
bars_q = ax.bar(x + width / 2, quantum, width, yerr=quantum_err,
                label='Q-RIMER', color='#0F62FE', capsize=3,
                error_kw={'elinewidth': 0.8})

for bar, val in zip(bars_c, classical):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars_q, quantum):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('Belief degree')
ax.set_xticks(x, labels=categories)
ax.legend(framealpha=0.9)
ax.set_ylim(0, 0.88)

plt.tight_layout(pad=0.5)
out_base = os.path.join(OUTPUT_DIR, 'correctness')
plt.savefig(out_base + '.pdf', dpi=600)
plt.savefig(out_base + '.png', dpi=600)
plt.close()
print(f"Saved: {out_base}.pdf, {out_base}.png")
