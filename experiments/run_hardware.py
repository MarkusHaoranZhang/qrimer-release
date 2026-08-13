"""
Hardware deployment script for Q-RIMER.

This script provides the workflow for running Q-RIMER on real quantum hardware:
  1. Feasibility check (qubit count + depth vs coherence budget)
  2. Circuit construction and transpilation
  3. Execution on IBM Quantum (or Aer fallback)
  4. Result extraction and comparison with classical ground truth

Usage
-----
    # Dry run (feasibility check only, no hardware needed)
    python experiments/run_hardware.py --dry-run

    # Execute on Aer noisy simulator
    python experiments/run_hardware.py --backend aer_noise

    # Execute on IBM Quantum (requires saved credentials)
    python experiments/run_hardware.py --backend ibm
"""

import argparse

import numpy as np

from analysis.metrics import belief_distribution_error, rank_preservation
from config import RANDOM_SEED, SHOTS
from core.backends import estimate_hardware_feasibility, get_backend, run_circuit
from core.qer import QER
from experiments.datasets import SyntheticDataGenerator


def feasibility_report():
    """Print hardware feasibility for various problem sizes."""
    print("=" * 70)
    print("Hardware Feasibility Assessment")
    print("=" * 70)

    # IBM Eagle (127 qubits), IBM Heron (133 qubits)
    devices = [
        ("IBM Eagle r3", 127, 80.0, 300.0),
        ("IBM Heron r2", 156, 100.0, 60.0),
    ]

    cases = [
        (3, 36, 4, "Graphite benchmark"),
        (5, 50, 4, "Small synthetic"),
        (10, 100, 4, "Medium synthetic"),
        (15, 200, 4, "Large synthetic"),
    ]

    for device_name, qubits, t2, gate_time in devices:
        print(f"\n--- {device_name} ({qubits} qubits, T2={t2}us, CX={gate_time}ns) ---")
        print(f"  {'Case':<22} {'Target':>7} {'Full':>7} "
              f"{'Budget':>7} {'Target?':>7} {'Full?':>6}")
        print(f"  {'-'*65}")
        for N, L, T, desc in cases:
            result = estimate_hardware_feasibility(
                N, L, T, device_qubits=qubits,
                device_t2_us=t2, gate_time_ns=gate_time
            )
            t_ok = "[OK]" if result['target_feasible'] else "[X]"
            f_ok = "[OK]" if result['full_feasible'] else "[X]"
            print(f"  {desc:<22} {result['target_depth']:>7} "
                  f"{result['full_depth']:>7} "
                  f"{result['coherence_budget']:>7} {t_ok:>7} {f_ok:>6}")
        print("  (Target = StatePrep depth; Full = iterative MEoB depth)")


def run_on_backend(backend_type: str = 'aer_noise', noise_scale: float = 1.0):
    """
    Execute Q-RIMER target-state circuit on a backend and compare with
    classical ground truth.
    """
    print(f"\n{'='*70}")
    print(f"Q-RIMER execution on backend: {backend_type}")
    print(f"{'='*70}")

    # Use a small, tractable problem
    N, L, T = 3, 4, 2
    gen = SyntheticDataGenerator(N, L, T, seed=RANDOM_SEED)
    betas, rule_weights, attr_weights = gen.generate_rules()
    w = np.array([0.4, 0.3, 0.2, 0.1])

    # Classical ground truth
    qer = QER(N=N, L=L, kappa_max=50.0)
    beta_c, beta_D_c = qer.classical_simulate(w, betas)

    print(f"\n  Problem: N={N}, L={L}, T={T}")
    print(f"  Classical result: beta = [{', '.join(f'{b:.4f}' for b in beta_c)}], "
          f"beta_D = {beta_D_c:.4f}")

    # Build target-state circuit
    circ = qer.build_target_state_circuit(w, betas)
    circ.measure_all()
    print(f"  Circuit: {circ.num_qubits} qubits, depth={circ.depth()}")

    # Get backend and execute
    try:
        backend = get_backend(backend_type, noise_scale=noise_scale)
    except (ImportError, RuntimeError) as e:
        print(f"  Backend unavailable: {e}")
        print("  Falling back to statevector simulation.")
        backend = get_backend('statevector')

    print(f"  Executing with {SHOTS} shots...")
    counts = run_circuit(circ, backend, shots=SHOTS, optimization_level=2)

    # Extract belief distribution from measurement results
    total = sum(counts.values())
    beta_hw = np.zeros(N)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        if idx < N:
            beta_hw[idx] += count
    beta_hw /= total

    # Compare
    eps = belief_distribution_error(beta_hw, beta_c)
    tau = rank_preservation(beta_hw, beta_c)

    print(f"\n  Hardware result: beta = [{', '.join(f'{b:.4f}' for b in beta_hw)}]")
    print(f"  eps_beta = {eps:.4f}")
    print(f"  tau   = {tau:.3f}")
    print(f"  Status: {'PASS' if eps < 0.05 else 'DEGRADED'} "
          f"(threshold: eps_beta < 0.05)")


def main():
    parser = argparse.ArgumentParser(description="Q-RIMER hardware deployment")
    parser.add_argument('--backend', default='aer_noise',
                        choices=['statevector', 'aer', 'aer_noise', 'ibm'],
                        help='Execution backend')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print feasibility report, no execution')
    parser.add_argument('--noise-scale', type=float, default=1.0,
                        help='Noise scale factor for aer_noise backend')
    args = parser.parse_args()

    feasibility_report()

    if not args.dry_run:
        run_on_backend(args.backend, args.noise_scale)


if __name__ == '__main__':
    main()
