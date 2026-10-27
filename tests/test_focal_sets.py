"""Tests for focal-set BBA truncation."""

import numpy as np
import pytest

from core.focal_sets import FocalSetBBA, er_mass_vector_to_focal_sets


def test_truncation_removes_large_focal_sets():
    bba = FocalSetBBA({
        frozenset([0]): 0.2,
        frozenset([0, 1]): 0.3,
        frozenset([0, 1, 2, 3]): 0.5,
    })
    truncated = bba.truncate(2)
    assert frozenset([0, 1, 2, 3]) not in truncated.masses
    assert abs(truncated.total_mass - 1.0) < 1e-12
    assert abs(truncated.belief(0) - 1.0) < 1e-12


def test_er_vector_projection_and_truncation():
    m = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.2, 0.5])
    bba = er_mass_vector_to_focal_sets(m)
    assert abs(bba.total_mass - 1.0) < 1e-12
    truncated = bba.truncate(3)
    assert abs(truncated.ignorance()) < 1e-12
    with pytest.raises(ValueError):
        bba.truncate(0)
