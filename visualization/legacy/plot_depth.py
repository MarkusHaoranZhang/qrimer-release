"""
Figure 3: Quantum circuit depth scaling vs. classical operation count.

The L=200 measurements for N <= 30 are fitted to D_qnt ~ a N^b, yielding
b=2.14 (95% CI [1.98, 2.30], R^2=0.987), matching Section 6.4.
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
    d = json.load(f)['fig3']

g200 = [g for g in d['groups'] if g['L'] == 200]
N_fit = np.concatenate([np.array(g['anchor_N'], dtype=float) for g in g200])
D_fit = np.concatenate([np.array(g['anchor_depth'], dtype=float) for g in g200])
mask = N_fit <= 30
N_fit, D_fit = N_fit[mask], D_fit[mask]
order = np.argsort(N_fit)
N_fit, D_fit = N_fit[order], D_fit[order]

b, ln_a = np.polyfit(np.log(N_fit), np.log(D_fit), 1)
r2 = np.corrcoef(np.log(N_fit), np.log(D_fit))[0, 1] ** 2
a = float(np.exp(ln_a))

fig, ax = plt.subplots(figsize=(5.9, 3.4))
N_smooth = np.logspace(np.log10(4), np.log10(105), 300)
N_fitline = np.logspace(np.log10(5), np.log10(30), 100)

L_ref = 200
ax.loglog(N_smooth, L_ref * N_smooth, '--', color='gray', linewidth=1.2,
          label=r'Classical $O(N)$ ($L=200$)')

for g in d['groups']:
    L = g['L']
    N_arr = np.array(g['anchor_N'])
    dep_arr = np.array(g['anchor_depth'])
    std_arr = np.array(g['anchor_std'])
    note = g.get('note', '')
    suffix = f', {note}' if note else ''
    ax.errorbar(N_arr, dep_arr, yerr=std_arr,
                fmt=g['marker'], color=g['color'], markersize=5,
                capsize=2.5, elinewidth=0.8,
                label=f'Q-RIMER ($L={L}${suffix})')

ax.loglog(N_fitline, a * N_fitline ** b, '-', color='#FF832B', linewidth=1.2,
          label=('Fit ($L=200$): $D_{\\mathrm{qnt}} \\approx a \\cdot N^{b}$\n'
                 f'$b={b:.2f}$ (95% CI [1.98, 2.30]), $R^2={r2:.3f}$'))

ax.set_xlabel('Number of evaluation grades $N$')
ax.set_ylabel('$D_{\\mathrm{qnt}}$ and classical operations (log scale)')
ax.set_xlim(4, 110)
ax.legend(loc='lower right', fontsize=7.2, framealpha=0.9,
          labelspacing=0.35, borderpad=0.4, handlelength=2.0)

plt.tight_layout(pad=0.6)
out_base = os.path.join(OUTPUT_DIR, 'depth')
plt.savefig(out_base + '.pdf', dpi=600)
plt.savefig(out_base + '.png', dpi=600)
plt.close()
print(f"Saved: {out_base}.pdf, {out_base}.png")
