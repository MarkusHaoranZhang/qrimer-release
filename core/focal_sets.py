"""
Focal-set belief assignments and cardinality truncation.

This module implements the power-set/focal-set object independently of the
ER singleton+frame representation.  It is used to make the large-N truncation
operation explicit and testable; the ER engine itself operates on the
(N+2)-dimensional singleton/frame mass vector.
"""

from __future__ import annotations

import numpy as np


class FocalSetBBA:
    """A normalised basic belief assignment over focal sets."""

    def __init__(self, masses: dict[frozenset[int], float] | None = None):
        self.masses = {}
        for fs, mass in (masses or {}).items():
            if mass < 0.0:
                raise ValueError("masses must be non-negative")
            if mass > 0.0:
                self.masses[frozenset(fs)] = float(mass)
        self.normalize()

    @property
    def total_mass(self) -> float:
        return float(sum(self.masses.values()))

    def normalize(self):
        total = self.total_mass
        if total <= 1e-15:
            raise ValueError("BBA has zero total mass")
        self.masses = {fs: m / total for fs, m in self.masses.items()}

    def belief(self, grade: int) -> float:
        return float(sum(m for fs, m in self.masses.items() if grade in fs))

    def ignorance(self) -> float:
        frame = frozenset().union(*self.masses) if self.masses else frozenset()
        return float(self.masses.get(frame, 0.0))

    def truncate(self, max_cardinality: int) -> FocalSetBBA:
        """Remove focal sets above the cardinality threshold and renormalize."""
        kept = {fs: m for fs, m in self.masses.items()
                if len(fs) <= max_cardinality}
        if not kept:
            raise ValueError("truncation removed all focal mass")
        return FocalSetBBA(kept)

    def singleton_frame_mass_vector(self, N: int) -> np.ndarray:
        """
        Project onto the ER state layout [m_1..m_N, m_bar_D, m_hat_D].

        The frame mass is placed in m_hat_D; m_bar_D is zero.  Subsets of
        cardinality 2..N-1 are not representable in the ER state and are
        accumulated into the frame mass for this projection.
        """
        vec = np.zeros(N + 2)
        for fs, m in self.masses.items():
            if len(fs) == 1:
                (j,) = tuple(fs)
                if 0 <= j < N:
                    vec[j] += m
            elif len(fs) == N:
                vec[N + 1] += m
            else:
                vec[N + 1] += m
        return vec


def er_mass_vector_to_focal_sets(m: np.ndarray) -> FocalSetBBA:
    """Convert [m_1..m_N, m_bar_D, m_hat_D] to singleton+frame focal sets."""
    N = len(m) - 2
    masses = {}
    for j in range(N):
        if m[j] > 0.0:
            masses[frozenset([j])] = float(m[j])
    frame_mass = float(m[N] + m[N + 1])
    if frame_mass > 0.0:
        masses[frozenset(range(N))] = frame_mass
    return FocalSetBBA(masses)
