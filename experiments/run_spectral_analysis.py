"""
Spectral analysis of the fixed ER synthesis operator.

Reports singular values, Hermitian-dilation eigenvalues and condition
numbers for growing N, and separates the spectral condition number from the
conflict-dependent scalar amplification K used in the sensitivity analysis.
"""

import os

import numpy as np

from core.er_matrices import spectral_report

LEGACY_PAPER_KAPPA = np.array([5.23, 53.08, 478.15])
LEGACY_PAPER_CONFLICT = np.array([0.10, 0.50, 0.90])


def main():
    print("=" * 72)
    print("Spectral analysis of the ER synthesis operator")
    print("=" * 72)
    print(f"  {'N':>3}  {'B shape':>10}  {'alpha':>9}  {'cond(B)':>10}  "
          f"{'#nonzero eigenvalues':>21}")
    for N in range(2, 11):
        rep = spectral_report(N)
        print(f"  {N:>3}  {str(rep['shape']):>10}  {rep['alpha']:>9.4f}  "
              f"{rep['condition_number']:>10.4f}  "
              f"{len(rep['dilation_nonzero_eigenvalues']):>21}")

    print("\n[Conflict scalar K versus legacy reference values]")
    print(f"  {'K_conf':>8}  {'K=1/(1-c)':>11}  {'legacy kappa':>12}")
    for c, kappa_paper in zip(LEGACY_PAPER_CONFLICT, LEGACY_PAPER_KAPPA):
        print(f"  {c:>8.2f}  {1.0 / (1.0 - c):>11.2f}  {kappa_paper:>12.2f}")

    print("\n[Interpretation]")
    print("  cond(B) is a data-independent spectral quantity of the 0/1")
    print("  synthesis matrix; K rescales the output and does not change")
    print("  the spectral condition number. The legacy kappa is an")
    print("  effective conflict-dependent amplification, not cond(B).")

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs')
    os.makedirs(out_dir, exist_ok=True)
    reports = {}
    for N in range(2, 11):
        rep = spectral_report(N)
        reports[f'N{N}'] = rep
    np.savez(os.path.join(out_dir, 'spectral_analysis.npz'), **reports)
    print(f"\nSaved: {os.path.join(out_dir, 'spectral_analysis.npz')}")


if __name__ == '__main__':
    main()
