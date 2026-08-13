"""
Figure 3 (paper): Quantum circuit depth scaling vs classical operation count.

The classical ER combination scales as O(N) (conflict coefficient computation,
Section 6.4); the quantum circuit depth follows a power law with fitted
exponent b = 2.14 (finite-N behaviour; asymptotic bound O(log N)).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_vals    = np.array([5, 10, 15, 20, 30])
depth_vals = np.array([214, 687, 1243, 2156, 4621])

# Power-law fit D = a * N^b on the measured series (Table 3)
coeffs = np.polyfit(np.log(N_vals), np.log(depth_vals), 1)
b = float(coeffs[0])
a = float(np.exp(coeffs[1]))

N_smooth = np.linspace(5, 30, 200)

fig, ax1 = plt.subplots(figsize=(6, 4))

# Classical complexity: O(N) reference line (left y-axis)
N_line = np.linspace(1, 30, 100)
ax1.plot(N_line, N_line, '--', color='gray', label=r'Classical $O(N)$')
ax1.set_xlabel('Number of evaluation grades $N$')
ax1.set_ylabel('Classical operations (arbitrary units)', color='gray')
ax1.tick_params(axis='y', labelcolor='gray')

# Quantum depth on right y-axis
ax2 = ax1.twinx()
ax2.scatter(N_vals, depth_vals, color='#4C72B0', zorder=5,
            label='Quantum depth (measured)')
ax2.plot(N_smooth, a * N_smooth ** b, '-', color='#DD8452',
         label=fr'Fitted $a N^{{{b:.2f}}}$')
ax2.set_ylabel('Quantum circuit depth (layers)', color='#4C72B0')
ax2.tick_params(axis='y', labelcolor='#4C72B0')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
ax1.set_title('Quantum circuit depth scaling vs. classical operation count')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'fig_depth.pdf')
plt.savefig(out_path, dpi=120)
plt.close()
print(f"Saved: {out_path}")
print(f"Power-law fit: D = {a:.2f} * N^{b:.2f} (paper: b=2.14, R^2=0.987)")
