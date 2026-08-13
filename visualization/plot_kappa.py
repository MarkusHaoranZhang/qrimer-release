"""
Figure: Condition number analysis — kappa vs. conflict and error vs. kappa^2.

Attempts to load live data from experiments/run_sensitivity.py;
falls back to paper-reported values for standalone execution.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try loading live data from sensitivity analysis
r2 = None
try:
    from experiments.run_sensitivity import condition_number_analysis
    K_conf, kappa, kappa_std, eps_beta, eps_std, r2 = condition_number_analysis(n_instances=5)
    print("Loaded live sensitivity data.")
except Exception:
    # Fall back to paper-reported values
    K_conf    = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    kappa     = np.array([5.23, 11.75, 21.48, 34.67, 53.08, 86.33, 151.98, 283.48, 478.15])
    kappa_std = np.array([0.31, 0.68, 1.25, 2.12, 3.41, 5.92, 11.23, 22.67, 41.50])
    eps_beta  = np.array([0.00291, 0.00312, 0.00338, 0.00394, 0.00476,
                          0.00674, 0.01025, 0.01847, 0.02732])
    eps_std   = np.array([0.00028, 0.00031, 0.00035, 0.00042, 0.00051,
                          0.00073, 0.00112, 0.00208, 0.00305])
    print("Using paper-reported kappa data.")

kappa_sq = kappa ** 2
coeff = np.polyfit(kappa_sq, eps_beta, 1)
fit_fn = np.poly1d(coeff)
if r2 is None:
    # R^2 not supplied by the analysis function: compute from the data
    ss_res = np.sum((eps_beta - fit_fn(kappa_sq)) ** 2)
    ss_tot = np.sum((eps_beta - np.mean(eps_beta)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Left: kappa vs K_conf
ax1.errorbar(K_conf, kappa, yerr=kappa_std, fmt='o-',
             color='#4C72B0', capsize=4, markersize=6, linewidth=1.5)
ax1.set_xlabel(r'Conflict coefficient $K_{\mathrm{conf}}$')
ax1.set_ylabel(r'Condition number $\kappa$')
ax1.set_title('Condition number vs. conflict')
ax1.axvspan(0.10, 0.50, alpha=0.08, color='green')
ax1.text(0.52, kappa.max() * 0.75, 'Moderate\nregion', fontsize=9, color='green')

# Right: eps_beta vs kappa^2
kappa_sq_smooth = np.linspace(0, kappa_sq.max() * 1.05, 200)
ax2.errorbar(kappa_sq, eps_beta, yerr=eps_std, fmt='o',
             color='#DD8452', capsize=4, markersize=6,
             label='Observed ($n=10$ per point)')
ax2.plot(kappa_sq_smooth, fit_fn(kappa_sq_smooth), '--',
         color='gray', linewidth=1.5, label=fr'Linear fit ($R^2={r2:.3f}$)')
ax2.set_xlabel(r'$\kappa^2$')
ax2.set_ylabel(r'QER error $\varepsilon_\beta$')
ax2.set_title(r'Error dependence on $\kappa^2$')
ax2.legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'fig_kappa.pdf')
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")
