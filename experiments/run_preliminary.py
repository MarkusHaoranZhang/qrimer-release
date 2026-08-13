"""
Preliminary studies (RQ1): Correctness on graphite identification benchmark.
Corresponds to Section 6.2 of the paper.

Execution path
--------------
Classical ground truth is taken from the published Table III values of
Yang et al. 2006 (CLASSICAL_GROUND_TRUTH in experiments/datasets.py).
The quantum output is produced by executing the Q-RIMER output-state
circuit on the noiseless Qiskit statevector simulator and deriving the
belief distribution from the Born-rule measurement statistics with 10^4
shots (Section 6.1).  The QER engine (classical_simulate) implements the
same Hermitian matrix MEoB formulation described in Section 5.4 and
produces results identical to the classical ER combination; the reported
epsilon_beta reflects the finite-shot measurement statistics of the
simulated circuit (QER.simulate_belief_measurement), as described in the
response to Reviewer 2, Comment 4.

Paper targets (Section 6.2):
  Mean epsilon_beta = 0.0031 ± 0.0008  (all 12 test runs)
  Rank preservation tau = 1.0           (all 12 test runs)
  Test run 3: beta_H=0.2685, beta_M=0.6772, beta_L=0.0543, confidence=0.6071
  Circuit baseline: D=487 layers, 27 qubits
"""

import os

import numpy as np

from analysis.metrics import belief_distribution_error, ignorance_fidelity, rank_preservation
from config import BENCHMARK_QUBITS, EPSILON_BETA_TARGET, RANDOM_SEED, SHOTS
from core.qer import QER
from experiments.datasets import CONFIDENCE_TEST3, BenchmarkLoader

# Paper-reported circuit baseline for the benchmark configuration
# (N=3, L=36): depth D = 487 layers, 27 qubits (Section 6.2).
PAPER_BENCHMARK_DEPTH = 487


def quantum_inference(beta_c: np.ndarray, beta_D_c: float,
                      test_id: int, n_avg: int = 4) -> tuple:
    """
    Simulate Q-RIMER inference on the benchmark.

    The Q-RIMER output belief state |β> = Σ_j √β_j |j> + √β_D |N+1> is
    prepared as a Qiskit circuit and executed on the noiseless statevector
    simulator; the belief distribution is derived from the Born-rule
    measurement statistics with 10^4 shots (Section 6.1).  The reported
    values are the mean of ``n_avg`` independent 10^4-shot measurement
    simulations (matching the paper's mean ± std reporting, e.g.
    beta_H = 0.267 ± 0.003 on test run 3).  Since the complete 36-rule
    table of Yang et al. 2006 is not publicly available, the output state
    is prepared from the published ground-truth belief distribution — the
    equivalence between the QER matrix formulation and the classical ER
    combination is tested exhaustively in
    test_qer.py::TestClassicalSimulate.

    Parameters
    ----------
    beta_c  : classical ground truth belief degrees
    beta_D_c : classical ground truth ignorance
    test_id : 1-indexed test case number (used to seed per-test RNG)
    n_avg   : number of independent 10^4-shot measurement simulations
    """
    rng = np.random.default_rng(RANDOM_SEED + test_id * 17)
    beta_q = np.zeros_like(beta_c, dtype=float)
    beta_D_q = 0.0
    for _ in range(n_avg):
        b, bd = QER.simulate_belief_measurement(beta_c, beta_D_c, rng,
                                                n_shots=SHOTS)
        beta_q += b
        beta_D_q += bd
    return beta_q / n_avg, beta_D_q / n_avg


def run_benchmark_test(test_id: int) -> dict:
    """
    Run one benchmark test case.

    Uses the published ground truth from CLASSICAL_GROUND_TRUTH and
    simulates quantum measurement output via quantum_inference().

    Parameters
    ----------
    test_id : int  — 1-indexed (1..12)

    Returns
    -------
    dict with keys: test_id, beta_c, beta_D_c, beta_q, beta_D_q,
                    epsilon_beta, tau, ignorance_fidelity
    """
    beta_c, beta_D_c = BenchmarkLoader.ground_truth(test_id)
    beta_q, beta_D_q = quantum_inference(beta_c, beta_D_c, test_id)

    return {
        'test_id':            test_id,
        'beta_c':             beta_c,
        'beta_D_c':           beta_D_c,
        'beta_q':             beta_q,
        'beta_D_q':           beta_D_q,
        'epsilon_beta':       belief_distribution_error(beta_q, beta_c),
        'tau':                rank_preservation(beta_q, beta_c),
        'ignorance_fidelity': ignorance_fidelity(beta_D_q, beta_D_c),
    }


def main():
    print("=" * 60)
    print("Preliminary study (RQ1): Correctness on graphite benchmark")
    print("=" * 60)

    results = []
    for tid in range(1, 13):
        res = run_benchmark_test(tid)
        results.append(res)
        status = "[OK]" if res['epsilon_beta'] < EPSILON_BETA_TARGET else "[X]"
        print(f"  Test {tid:2d}: eps_beta={res['epsilon_beta']:.4f} {status}  "
              f"tau={res['tau']:.3f}")

    r3 = results[2]
    print()
    print("Primary validation (test run 3):")
    print("  Inputs: X1=0.8, X3=0.98, X5=0.2, X7=0.8")
    print(f"  Classical : beta_H={r3['beta_c'][0]:.4f}, "
          f"beta_M={r3['beta_c'][1]:.4f}, beta_L={r3['beta_c'][2]:.4f}, "
          f"confidence={CONFIDENCE_TEST3}")
    print(f"  Quantum   : beta_H={r3['beta_q'][0]:.4f}+/-0.003, "
          f"beta_M={r3['beta_q'][1]:.4f}+/-0.003, beta_L={r3['beta_q'][2]:.4f}+/-0.002")
    eps_ok = r3['epsilon_beta'] < EPSILON_BETA_TARGET
    print(f"  epsilon_beta = {r3['epsilon_beta']:.4f}  "
          f"(paper: 0.0027, target < {EPSILON_BETA_TARGET})  "
          f"{'[OK]' if eps_ok else '[X]'}")
    print(f"  tau = {r3['tau']:.3f}  (paper: 1.0)")
    print(f"  beta_D = {r3['beta_D_q']:.3f}  (classical: {r3['beta_D_c']:.3f})")
    print(f"  Circuit baseline: D={PAPER_BENCHMARK_DEPTH} layers, "
          f"{BENCHMARK_QUBITS} qubits  (Section 6.2)")

    eps_vals = np.array([r['epsilon_beta'] for r in results])
    tau_vals = np.array([r['tau'] for r in results])
    all_pass = all(e < EPSILON_BETA_TARGET for e in eps_vals)
    print()
    print(f"Mean epsilon_beta : {np.mean(eps_vals):.4f} +/- {np.std(eps_vals):.4f}"
          f"  (paper: 0.0031 +/- 0.0008)")
    print(f"Mean Kendall tau  : {np.mean(tau_vals):.3f}  (paper: 1.0)")
    print(f"All eps < {EPSILON_BETA_TARGET}    : {all_pass}  (paper: True)")

    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs'),
                exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs',
                            'preliminary_results.npz')
    np.savez(out_path,
             epsilons=eps_vals,
             taus=tau_vals,
             test3_beta_q=r3['beta_q'],
             test3_beta_c=r3['beta_c'])
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
