"""Record the exact software environment and experiment seeds."""

import json
import os
import platform
import sys

import numpy as np
import qiskit
import qiskit_aer
import scipy


def main():
    manifest = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'qiskit': qiskit.__version__,
        'qiskit_aer': qiskit_aer.__version__,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'random_seed': 42,
        'shots': 10 ** 4,
        'noise': {
            'model': 'IBM Falcon depolarizing/density-matrix operator model',
            'single_qubit_error': 1e-3,
            'two_qubit_error': 1e-2,
            'T1_us': 100,
            'T2_us': 80,
        },
    }
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'environment_manifest.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)
    for key, value in manifest.items():
        print(f"{key:>20}: {value}")
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
