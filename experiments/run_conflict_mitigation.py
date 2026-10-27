"""
High-conflict mitigation probe.

Uses two controlled mass vectors with strong singleton conflict, applies
belief discounting toward ignorance, and reports the resulting reduction of
the pairwise conflict and the delegation decision.
"""

import numpy as np

from core.mitigation import conflict_coefficient, should_delegate


def discount_mass_vector(m: np.ndarray, discount: float) -> np.ndarray:
    """Discount singleton mass and move it to the ignorance component."""
    N = len(m) - 2
    out = m.copy()
    removed = discount * out[:N].sum()
    out[:N] *= (1.0 - discount)
    out[N + 1] += removed
    return out


def main():
    N = 3
    m1 = np.array([0.95, 0.00, 0.00, 0.05, 0.00])
    m2 = np.array([0.00, 0.95, 0.00, 0.05, 0.00])
    c0 = conflict_coefficient(m1, m2)

    print("=" * 68)
    print("High-conflict mitigation probe")
    print("=" * 68)
    print(f"  conflict before discount : {c0:.4f}")
    print(f"  delegate before          : {should_delegate(c0)}")

    for discount in (0.1, 0.2, 0.4):
        m1d = discount_mass_vector(m1, discount)
        m2d = discount_mass_vector(m2, discount)
        c = conflict_coefficient(m1d, m2d)
        print(f"  discount={discount:.1f}: conflict={c:.4f}, "
              f"delegate={should_delegate(c)}")

    print("  Ignorance mass after discounting: "
          f"{m1d[N + 1]:.4f} per source (was {m1[N + 1]:.4f}).")


if __name__ == '__main__':
    main()
