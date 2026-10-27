"""Export reproducible table JSON from logged experiment outputs."""

import json
import os

import numpy as np


def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    out_dir = os.path.join(root, 'outputs', 'tables')
    os.makedirs(out_dir, exist_ok=True)

    noise_rows = []
    for filename in ('blockencoded_noise_n3.npz',
                     'blockencoded_noise_large_n.npz'):
        noise_path = os.path.join(root, 'outputs', 'logs', filename)
        data = np.load(noise_path, allow_pickle=True)
        for key in data.files:
            item = data[key].item()
            noise_rows.append({
                'run': key,
                'N': int(key.split('_')[0][1:]),
                'noise': item['noise'],
                'error': item['error'],
                'success_product': item['success_product'],
                'qubits': item['qubits'],
                'source': 'blockencoded_density_matrix',
            })
    noise_path_out = os.path.join(out_dir, 'table10_generated.json')
    with open(noise_path_out, 'w', encoding='utf-8') as fh:
        json.dump(noise_rows, fh, indent=2)

    resource_path = os.path.join(out_dir, 'table3_structured.json')
    with open(resource_path, encoding='utf-8') as fh:
        resource_rows = json.load(fh)
    print(f"Table 3 rows: {len(resource_rows)} -> {resource_path}")
    print(f"Table 10 rows: {len(noise_rows)} -> {noise_path_out}")


if __name__ == '__main__':
    main()
