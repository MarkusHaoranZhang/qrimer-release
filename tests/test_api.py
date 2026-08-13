"""
Unit tests for the public API (core.api).

These tests verify the contract that hardware integrations depend on.
If any of these break, downstream systems will also break.
"""

import numpy as np
import pytest

from core.api import (
    ExecutionMode,
    InferenceRequest,
    InferenceResult,
    QRIMEREngine,
)


class TestInferenceRequest:
    def test_valid_construction(self):
        req = InferenceRequest(
            betas=np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]),
            rule_weights=np.array([0.6, 0.4]),
            attr_weights=np.array([[1.0, 0.8], [0.9, 1.0]]),
            input_dists=[{0: 0.5, 1: 0.3, 2: 0.2}, {0: 0.4, 1: 0.4, 2: 0.2}],
        )
        assert req.N == 3
        assert req.L == 2
        assert req.T == 2

    def test_invalid_rule_weights_shape(self):
        with pytest.raises(ValueError):
            InferenceRequest(
                betas=np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]),
                rule_weights=np.array([0.6, 0.4, 0.1]),  # wrong length
                attr_weights=np.array([[1.0], [0.9]]),
                input_dists=[{0: 1.0}],
            )

    def test_invalid_input_dists_length(self):
        with pytest.raises(ValueError):
            InferenceRequest(
                betas=np.array([[0.5, 0.3, 0.2]]),
                rule_weights=np.array([1.0]),
                attr_weights=np.array([[1.0, 0.8]]),
                input_dists=[{0: 1.0}],  # T=2 but only 1 dist
            )


class TestQRIMEREngine:
    def _make_request(self, N=3, L=4, T=2):
        rng = np.random.default_rng(42)
        betas = rng.dirichlet(np.ones(N), size=L) * 0.8
        rw = rng.uniform(0.3, 1.0, size=L)
        aw = rng.uniform(0.5, 1.0, size=(L, T))
        dists = [{j: float(v) for j, v in enumerate(rng.dirichlet(np.ones(3)))}
                 for _ in range(T)]
        return InferenceRequest(betas=betas, rule_weights=rw,
                                attr_weights=aw, input_dists=dists)

    def test_classical_inference(self):
        engine = QRIMEREngine.from_config(n_grades=3, n_rules=4, n_attrs=2)
        req = self._make_request()
        result = engine.infer(req)

        assert isinstance(result, InferenceResult)
        assert result.mode == ExecutionMode.CLASSICAL
        assert result.belief_degrees.shape == (3,)
        assert np.all(result.belief_degrees >= 0)
        assert result.belief_degrees.sum() + result.ignorance <= 1.0 + 1e-9

    def test_simulate_inference(self):
        engine = QRIMEREngine.from_config(
            n_grades=3, n_rules=4, n_attrs=2,
            mode=ExecutionMode.SIMULATE
        )
        req = self._make_request()
        result = engine.infer(req)

        assert result.mode == ExecutionMode.SIMULATE
        assert result.n_qubits is not None
        assert result.circuit_depth is not None
        assert result.belief_degrees.shape == (3,)

    def test_classical_and_simulate_agree(self):
        """Classical and simulate modes must produce the same belief distribution."""
        req = self._make_request()

        engine_c = QRIMEREngine.from_config(n_grades=3, n_rules=4, n_attrs=2,
                                            mode=ExecutionMode.CLASSICAL)
        engine_s = QRIMEREngine.from_config(n_grades=3, n_rules=4, n_attrs=2,
                                            mode=ExecutionMode.SIMULATE)

        result_c = engine_c.infer(req)
        result_s = engine_s.infer(req)

        np.testing.assert_allclose(
            result_s.belief_degrees, result_c.belief_degrees, atol=1e-6,
            err_msg="Simulate mode doesn't match classical mode"
        )

    def test_dimension_mismatch_raises(self):
        engine = QRIMEREngine.from_config(n_grades=3, n_rules=4, n_attrs=2)
        # Request with wrong N
        bad_req = InferenceRequest(
            betas=np.array([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]]),
            rule_weights=np.ones(4) / 4,
            attr_weights=np.ones((4, 2)),
            input_dists=[{0: 0.5, 1: 0.5}, {0: 0.5, 1: 0.5}],
        )
        with pytest.raises(ValueError, match="dimensions"):
            engine.infer(bad_req)

    def test_result_dominant_grade(self):
        engine = QRIMEREngine.from_config(n_grades=3, n_rules=4, n_attrs=2)
        req = self._make_request()
        result = engine.infer(req)
        assert 0 <= result.dominant_grade < 3

    def test_result_is_complete_property(self):
        engine = QRIMEREngine.from_config(n_grades=3, n_rules=4, n_attrs=2)
        req = self._make_request()
        result = engine.infer(req)
        # is_complete is a boolean property
        assert isinstance(result.is_complete, bool)
