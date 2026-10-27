"""Tests for practical cost accounting and conflict mitigation."""

import math

import numpy as np
import pytest

from analysis.cost_model import end_to_end_cost, number_of_shots
from core.mitigation import conflict_coefficient, discount_beliefs, should_delegate


def test_number_of_shots_scales_with_grades():
    assert number_of_shots(3, 0.01) > number_of_shots(3, 0.1)
    assert number_of_shots(10, 0.01) > number_of_shots(3, 0.01)


def test_end_to_end_cost_penalises_small_success_probability():
    low = end_to_end_cost(100, 1.0, 3)
    high = end_to_end_cost(100, 0.01, 3)
    assert high > low
    assert math.isinf(end_to_end_cost(100, 0.0, 3))


def test_discount_and_delegation():
    betas = np.array([0.4, 0.3, 0.2])
    reduced = discount_beliefs(betas, 0.25)
    np.testing.assert_allclose(reduced, betas * 0.75)
    with pytest.raises(ValueError):
        discount_beliefs(betas, 1.5)
    assert should_delegate(0.95)
    assert not should_delegate(0.5)


def test_conflict_coefficient():
    N = 3
    m_I = np.array([0.2, 0.3, 0.1, 0.3, 0.1])
    m_next = np.array([0.4, 0.1, 0.1, 0.3, 0.1])
    expected = m_I[:N].sum() * m_next[:N].sum() - np.dot(m_I[:N], m_next[:N])
    assert abs(conflict_coefficient(m_I, m_next) - expected) < 1e-12
