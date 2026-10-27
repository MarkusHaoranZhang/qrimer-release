"""
Batched classical baseline benchmark.

Compares the scalar ER reference with a batched vectorised ER implementation
on independent BRB instances.  The batch baseline provides a stronger
classical throughput reference for the Q-RIMER comparison.
"""

import time

import numpy as np

from core.classical_batch import er_combine_batch
from core.qer import QER


def main():
    N, L = 15, 200
    rng = np.random.default_rng(42)

    print("=" * 68)
    print("Batched classical ER baseline")
    print("=" * 68)
    print(f"  N={N}, L={L}, batch sizes: 1, 8, 64")
    print(f"  {'batch':>6}  {'time (s)':>10}  {'per-instance (ms)':>18}")

    for batch in (1, 8, 64):
        betas = rng.dirichlet(np.ones(N), size=(batch, L))
        w = rng.dirichlet(np.ones(L), size=batch)

        t0 = time.perf_counter()
        if batch == 1:
            QER(N=N, L=L, kappa_max=50.0).classical_simulate(w[0], betas[0])
        else:
            er_combine_batch(w, betas)
        elapsed = time.perf_counter() - t0
        print(f"  {batch:>6}  {elapsed:>10.4f}  {elapsed / batch * 1e3:>18.2f}")

    try:
        import torch

        from core.classical_torch import er_combine_torch
    except ImportError:
        print("\n  PyTorch baseline: not installed")
        return

    batch = 64
    betas = rng.dirichlet(np.ones(N), size=(batch, L))
    w = rng.dirichlet(np.ones(L), size=batch)
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    print("\n  PyTorch batch=64:")
    for device in devices:
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        er_combine_torch(w, betas, device=device)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"    {device:>4}  {elapsed:>10.4f} s  "
              f"{elapsed / batch * 1e3:>10.2f} ms/instance")


if __name__ == '__main__':
    main()
