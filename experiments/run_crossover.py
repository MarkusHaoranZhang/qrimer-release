"""
Empirical-versus-asymptotic scaling and crossover analysis.

The fitted power law is reported over its fitted range; the extrapolated
empirical crossover with the classical O(LN) line is computed explicitly,
and the truncated large-N points are marked as such.  The analysis does not
claim that the measured range confirms logarithmic scaling.
"""

import numpy as np

from analysis.complexity import empirical_crossover, power_law

N_FULL = np.array([5, 8, 10, 12, 15, 20, 25], dtype=float)
D_FULL = np.array([101, 370, 607, 642, 1243, 2156, 4327], dtype=float)
N_TRUNC = np.array([30, 50, 100], dtype=float)
D_TRUNC = np.array([4621, 6847, 9763], dtype=float)
L_REF = 200


def main():
    N_fit = np.concatenate([N_FULL, N_TRUNC[:1]])
    D_fit = np.concatenate([D_FULL, D_TRUNC[:1]])
    b, ln_a = np.polyfit(np.log(N_fit), np.log(D_fit), 1)
    a = float(np.exp(ln_a))
    r2 = np.corrcoef(np.log(N_fit), np.log(D_fit))[0, 1] ** 2
    crossover = empirical_crossover(L_REF, a, b)

    print("=" * 72)
    print("Scaling / crossover analysis (L=200)")
    print("=" * 72)
    print("  fit range: N=5..30 (N=30 uses the truncated scheme)")
    print(f"  empirical fit: a={a:.4f}, b={b:.4f}, R^2={r2:.4f}")
    print(f"  extrapolated empirical crossover with {L_REF}N: "
          f"N~{crossover:.1f}")
    print("\n  N     measured    power-law fit    classical 200N   status")
    for n, d in list(zip(N_FULL, D_FULL)) + list(zip(N_TRUNC, D_TRUNC)):
        status = "truncated" if n in N_TRUNC else "full"
        print(f"  {n:>3.0f}   {d:>8.0f}   {power_law(n, a, b):>14.0f}   "
              f"{L_REF * n:>14.0f}   {status}")
    print("\n  The power-law fit is finite-range and does not confirm")
    print("  logarithmic scaling. The N=50/100 points are truncated and")
    print("  lie below the extrapolated fit; no log crossover is identifiable")
    print("  from the measured range.")


if __name__ == '__main__':
    main()
