"""Export the data behind Figures 2 and 4 from executable runs."""

import json
import os

import numpy as np

from core.rimer import create_graphite_membership_functions, infer_graphite_hierarchical
from experiments.datasets import BenchmarkLoader
from experiments.run_ablation import (
    ablation_er_matrix,
    ablation_pipeline,
    ablation_qbra_benchmark,
)
from experiments.run_benchmark_quantum import quantum_graphite


def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    out_dir = os.path.join(root, 'outputs', 'manuscript')
    os.makedirs(out_dir, exist_ok=True)

    memb = create_graphite_membership_functions()[0]
    inputs, _, _ = BenchmarkLoader.load()
    x1, x3, x5, x7 = inputs[2]  # test run 3
    beta_c, beta_D_c = infer_graphite_hierarchical(x1, x3, x5, x7, memb)
    beta_q, beta_D_q, _ = quantum_graphite(x1, x3, x5, x7, memb)
    correctness = {
        'test': 3,
        'inputs': [float(x1), float(x3), float(x5), float(x7)],
        'classical': [float(v) for v in beta_c] + [float(beta_D_c)],
        'quantum': [float(v) for v in beta_q] + [float(beta_D_q)],
    }
    with open(os.path.join(out_dir, 'figure2_data.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(correctness, fh, indent=2)

    qbra = ablation_qbra_benchmark()
    er = ablation_er_matrix()
    pipeline = ablation_pipeline()
    two = pipeline['two_layer']
    ablation = {
        'conditions': ['Full Q-RIMER', 'Without QBRA', 'Without ER matrix',
                       'Full pipeline', 'Hybrid scheme'],
        'values': [
            float(qbra['eps_with_qbra']),
            float(qbra['eps_without_qbra']),
            float(er['epsilon_beta']),
            float(two['full_eps']),
            float(two['hybrid_eps']),
        ],
    }
    with open(os.path.join(out_dir, 'figure4_data.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(ablation, fh, indent=2)

    print("Saved figure2_data.json and figure4_data.json")
    print("Correctness:", np.round(beta_q, 6))
    print("Ablation:", ablation['values'])


if __name__ == '__main__':
    main()
