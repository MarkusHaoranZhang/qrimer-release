"""Tests for the block-encoded hierarchical pipeline."""

import numpy as np

from core.pipeline import QuantumPipeline
from experiments.datasets import TWO_LAYER_CONFIG, SyntheticDataGenerator


def _make_case():
    rules_per_layer = []
    for i, cfg in enumerate(TWO_LAYER_CONFIG):
        gen = SyntheticDataGenerator(cfg['N'], cfg['L'], cfg['T'],
                                     seed=42 + i)
        rules_per_layer.append(gen.generate_rules())
    rng = np.random.default_rng(123)
    input_dists = [{j: float(v) for j, v in enumerate(rng.dirichlet(np.ones(3)))}
                   for _ in range(TWO_LAYER_CONFIG[0]['T'])]
    return rules_per_layer, input_dists


def test_quantum_pipeline_matches_classical():
    rules, inputs = _make_case()
    pipeline = QuantumPipeline(TWO_LAYER_CONFIG, kappa=50.0, epsilon=0.01)

    beta_c, beta_D_c = pipeline.classical_simulate(inputs, rules)
    beta_q, beta_D_q, info = pipeline.quantum_simulate(inputs, rules)

    assert np.linalg.norm(beta_q - beta_c) < 1e-8
    assert abs(beta_D_q - beta_D_c) < 1e-8
    assert len(info) == len(TWO_LAYER_CONFIG)
    assert all('per_step_success' in layer for layer in info)


def test_coherent_pipeline_matches_classical():
    rules, inputs = _make_case()
    pipeline = QuantumPipeline(TWO_LAYER_CONFIG, kappa=50.0, epsilon=0.01)
    beta_c, beta_D_c = pipeline.classical_simulate(inputs, rules)
    beta_q, beta_D_q, info = pipeline.coherent_simulate(inputs, rules)
    assert np.linalg.norm(beta_q - beta_c) < 1e-8
    assert len(info) == len(TWO_LAYER_CONFIG)
