"""Backfill Section 6.7 direct-vs-vectorised classical timing."""

import json
import os
import time

import numpy as np

from core.classical_batch import er_combine_batch
from core.qer import QER


def main():
    rows = []
    for N, L in [(15, 200), (20, 200)]:
        rng = np.random.default_rng(42 + N)
        w = rng.dirichlet(np.ones(L))
        betas = rng.dirichlet(np.ones(N), size=L)
        times_direct, times_vec = [], []
        for _ in range(10):
            t0 = time.perf_counter()
            QER(N=N, L=L, kappa_max=50.0).classical_simulate(w, betas)
            times_direct.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            er_combine_batch(w[None, :], betas[None, :, :])
            times_vec.append(time.perf_counter() - t0)
        rows.append({
            'N': N, 'L': L,
            'direct_mean': float(np.mean(times_direct)),
            'direct_std': float(np.std(times_direct)),
            'vectorized_mean': float(np.mean(times_vec)),
            'vectorized_std': float(np.std(times_vec)),
            'speedup': float(np.mean(times_direct) / np.mean(times_vec)),
            'memory_bytes': int(w.nbytes + betas.nbytes),
        })
    root = os.path.join(os.path.dirname(__file__), '..')
    out_dir = os.path.join(root, 'outputs', 'manuscript')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'section67_generated.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(rows, fh, indent=2)
    print(f"Saved Section 6.7 timings to {out_dir}")
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
