"""
Micro-benchmark for core inference functions.

NOTE: This file is intentionally named bench.py (not test_bench.py)
so that pytest does not collect it.  Run manually: python tests/bench.py
"""
import time

from core.rimer import BeliefRule, ClassicalRIMER
from experiments.datasets import SyntheticDataGenerator


def bench(fn, n):
    for _ in range(max(1, n // 10)):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n

print(f"{'Case':<22} {'er_combine':>14} {'rule_activation':>16}")
print("-" * 54)
for N, L, T, n in [(3, 36, 4, 10000), (15, 200, 4, 500), (30, 200, 4, 100)]:
    gen = SyntheticDataGenerator(N, L, T, seed=42)
    betas, rw, aw = gen.generate_rules()
    rimer = ClassicalRIMER(N, L, T)
    for k in range(L):
        rimer.add_rule(BeliefRule(list(range(T)), betas[k], float(rw[k]), aw[k]))
    dists = [{j: 1.0/3 for j in range(3)} for _ in range(T)]
    w = rimer.rule_activation(dists)
    t_er = bench(lambda: rimer.er_combine(w, betas), n)
    t_ra = bench(lambda: rimer.rule_activation(dists), n)
    def fmt(t):
        return f"{t*1e6:.0f} µs" if t < 0.001 else f"{t*1e3:.2f} ms"
    print(f"N={N:2d}, L={L:3d}, T={T}  {fmt(t_er):>14} {fmt(t_ra):>16}")
