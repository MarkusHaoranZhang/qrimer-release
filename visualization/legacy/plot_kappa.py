"""
Figure 4: Effective amplification factor sensitivity.

Left panel: effective amplification factor vs. conflict coefficient.
Right panel: QER error vs. squared effective amplification factor.
Uses the paper-reported values in paper_figure_data.json (Section 6.5).
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
    'axes.labelsize': 8.5,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
})

with open(DATA_PATH, encoding='utf-8') as f:
    d = json.load(f)['fig4']

K_conf = np.array(d['K_conf'])
kappa = np.array(d['kappa'])
kappa_std = np.array(d['kappa_std'])
eps_beta = np.array(d['eps_beta'])
eps_std = np.array(d['eps_std'])

kappa_sq = kappa ** 2

coeff = np.polyfit(kappa_sq, eps_beta, 1)
fit_line = np.poly1d(coeff)
r_squared = np.corrcoef(kappa_sq, eps_beta)[0, 1] ** 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

ax1.errorbar(K_conf, kappa, yerr=kappa_std, fmt='o-', color='#0F62FE',
             capsize=3, markersize=4.5, linewidth=1.1)
ax1.set_xlabel('Conflict coefficient $K_{\\mathrm{cnf}}$')
ax1.set_ylabel('Effective amplification factor $\\bar{\\kappa}$')
ax1.set_title('(a) Effective amplification factor\nvs. conflict coefficient')
ax1.axvspan(0.10, 0.48, alpha=0.08, color='#42BE65')
ax1.set_xlim(0.05, 0.95)
ax1.text(0.25, 280, 'Moderate region', fontsize=7.5)
ax1.text(0.25, 238, r'$\bar{\kappa}<50$', fontsize=7.5)
ax1.annotate('$\\bar{\\kappa} \\approx 5$', xy=(0.10, 5.23), xytext=(0.22, 80),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=7.5, color='gray')
ax1.annotate('$\\bar{\\kappa} \\approx 480$', xy=(0.90, 478.15), xytext=(0.65, 420),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=7.5, color='gray')

ax2.errorbar(kappa_sq, eps_beta, yerr=eps_std, fmt='o', color='#0F62FE',
             capsize=3, markersize=4.5, label='Observed ($n=10$ per point)')
kappa_sq_smooth = np.linspace(0, kappa_sq.max() * 1.05, 200)
ax2.plot(kappa_sq_smooth, fit_line(kappa_sq_smooth), '--',
         color='gray', linewidth=1.1,
         label=f'Linear fit ($R^2={r_squared:.3f}$)')
ax2.set_xlabel('$\\bar{\\kappa}^2$')
ax2.set_ylabel('QER error $\\varepsilon_\\beta$')
ax2.set_title('(b) QER error vs. squared\neffective amplification factor')
ax2.legend(loc='upper left', fontsize=7.5, framealpha=0.9)

plt.tight_layout(pad=0.6)
out_base = os.path.join(OUTPUT_DIR, 'kappa')
plt.savefig(out_base + '.pdf', dpi=600)
plt.savefig(out_base + '.png', dpi=600)
plt.close()
print(f"Saved: {out_base}.pdf, {out_base}.png")
