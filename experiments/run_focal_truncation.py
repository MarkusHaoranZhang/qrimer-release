"""
Focal-set cardinality truncation probe.

Builds a BBA containing singletons, pairs, triples, a 5-set and the full
frame, then applies the K_max=3 focal-set truncation and reports the change
in belief and ignorance.  This is a focal-set-level probe; the ER large-N
study uses the separate singleton/frame mass vector.
"""

import numpy as np

from core.focal_sets import FocalSetBBA


def main():
    N = 8
    masses = {
        frozenset([0]): 0.15,
        frozenset([1]): 0.10,
        frozenset([0, 1]): 0.15,
        frozenset([2, 3]): 0.10,
        frozenset([0, 1, 2]): 0.15,
        frozenset([3, 4, 5, 6, 7]): 0.15,
        frozenset(range(N)): 0.20,
    }
    bba = FocalSetBBA(masses)
    truncated = bba.truncate(3)

    print("=" * 68)
    print("Focal-set K_max=3 truncation probe")
    print("=" * 68)
    print(f"  N={N}, focal sets before={len(bba.masses)}, "
          f"after={len(truncated.masses)}")
    print(f"  discarded cardinality>3 mass: "
          f"{sum(m for fs, m in bba.masses.items() if len(fs) > 3):.4f}")
    print(f"  ignorance before / after: {bba.ignorance():.4f} / "
          f"{truncated.ignorance():.4f}")
    diffs = [abs(bba.belief(j) - truncated.belief(j)) for j in range(N)]
    print(f"  mean |belief change|: {np.mean(diffs):.4f}")
    print("  Note: this is a focal-set-level probe, not the ER large-N run.")


if __name__ == '__main__':
    main()
