"""Tests for post-selection and accumulated-error accounting."""

import math

import pytest

from analysis.postselection import (
    coherent_success_probability,
    composed_error_bound,
    expected_repetitions,
    recursive_cost,
)


def test_coherent_product_and_repetitions():
    p = coherent_success_probability([0.2, 0.5, 0.1])
    assert abs(p - 0.01) < 1e-15
    assert abs(expected_repetitions(p) - 100.0) < 1e-12
    assert math.isinf(expected_repetitions(0.0))


def test_composed_error_bound_grows_with_small_success():
    bound = composed_error_bound(1e-3, [0.2, 0.2])
    assert bound > 1e-3
    assert math.isinf(composed_error_bound(1e-3, [0.0, 0.2]))


def test_recursive_cost_modes_and_validation():
    per_step = [0.2, 0.2, 0.2]
    coherent = recursive_cost(2.0, per_step, 3, 0.01, coherent=True)
    hybrid = recursive_cost(2.0, per_step, 3, 0.01, coherent=False)
    assert coherent > hybrid
    with pytest.raises(ValueError):
        coherent_success_probability([1.5])
