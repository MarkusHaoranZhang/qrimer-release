"""
Block-encoded hierarchical pipeline execution.

Runs the two-layer and three-layer synthetic hierarchical cases with a
circuit-executed QER kernel per layer.  Intermediate belief vectors are
passed between layers without measurement; the scalar K/norm bookkeeping is
classical.  Reports fidelity, per-layer post-selection and qubit counts.
"""

import numpy as np

from analysis.metrics import belief_distribution_error
from core.pipeline import QuantumPipeline
from experiments.datasets import (
    THREE_LAYER_CONFIG,
    TWO_LAYER_CONFIG,
    SyntheticDataGenerator,
)


def build_case(config, seed: int = 42):
    rules = []
    for i, cfg in enumerate(config):
        gen = SyntheticDataGenerator(cfg['N'], cfg['L'], cfg['T'],
                                     seed=seed + i)
        rules.append(gen.generate_rules())
    rng = np.random.default_rng(seed + 100)
    inputs = [{j: float(v) for j, v in enumerate(rng.dirichlet(np.ones(3)))}
              for _ in range(config[0]['T'])]
    return rules, inputs


def main():
    print("=" * 72)
    print("Block-encoded hierarchical pipeline")
    print("=" * 72)
    for label, config in [('two-layer', TWO_LAYER_CONFIG),
                          ('three-layer', THREE_LAYER_CONFIG)]:
        rules, inputs = build_case(config)
        pipeline = QuantumPipeline(config, kappa=50.0, epsilon=0.01)
        beta_c, beta_D_c = pipeline.classical_simulate(inputs, rules)
        beta_q, beta_D_q, info = pipeline.quantum_simulate(inputs, rules)
        eps = belief_distribution_error(beta_q, beta_c)
        print(f"\n[{label}]")
        print(f"  classical beta       : {np.round(beta_c, 6)}")
        print(f"  circuit beta         : {np.round(beta_q, 6)}")
        print(f"  epsilon_beta         : {eps:.3e}")
        print(f"  beta_D classical/Q   : {beta_D_c:.6f} / {beta_D_q:.6f}")
        for layer in info:
            print(f"  layer {layer['layer']}: qubits={layer['n_qubits']}, "
                  f"steps={layer['steps']}, "
                  f"mean p={layer['success_probability_mean']:.4f}")


if __name__ == '__main__':
    main()
