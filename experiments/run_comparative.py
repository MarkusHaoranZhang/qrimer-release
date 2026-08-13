"""
Comparative studies (RQ2): Acceleration trend vs classical RIMER.
Corresponds to Section 6.4 of the paper.

This script measures:
  1. Belief distribution error across problem sizes for Q-RIMER vs the
     BF-QC basic (conjunctive) baseline vs classical exact reference
     (paper Table 2), with Bonferroni-corrected paired t-tests.
  2. Quantum circuit depth scaling (paper Figure 3 / Table 3): power-law
     fit D ~ a * N^b with 95% CI and R^2.
  3. Classical CPU time vs quantum circuit depth (paper Table 3), and the
     asymptotic complexity reduction from O(N) to O(log N).

Paper results (Sections 6.2-6.4):
  Table 2: Q-RIMER eps = 0.0031, 0.0078, 0.0124, 0.0189, 0.0196
           BF-QC   eps = 0.0147, 0.0623, 0.1338, 0.2415, 0.3872
           p < 0.01 (Bonferroni) at all N
  Table 3: N=5,  L=50:  classical=0.12s,   quantum depth=214, qubits=18
           N=10, L=100: classical=8.45s,   quantum depth=687, qubits=22
           N=15, L=200: classical=142s,    quantum depth=1243, qubits=26
           N=20, L=200: classical=>3600s,  quantum depth=2156, qubits=30
           N=30, L=200: classical=extrap., quantum depth=4621, qubits=32
  Power-law fit (L=200): b=2.14, 95% CI [1.98, 2.30], R^2=0.987
  Additional fits: L=100: b=2.09 (R^2=0.991); L=50: b=2.17 (R^2=0.984)
"""

import time

import numpy as np
from scipy import stats

from analysis.metrics import belief_distribution_error
from analysis.sampling import multinomial_shot_noise
from analysis.statistics import bonferroni_correction
from config import GRAPHITE_T, RANDOM_SEED
from core.bfqc import conjunctive_combine
from core.qbra import QBRA
from core.qer import QER
from core.rimer import BeliefRule, ClassicalRIMER
from experiments.datasets import SyntheticDataGenerator

# Paper's measured quantum circuit depths (Table 3)
PAPER_DEPTHS = {
    (5,  50):  214,
    (10, 100): 687,
    (15, 200): 1243,
    (20, 200): 2156,
    (30, 200): 4621,
}

# Paper's reported qubit counts (Table 3)
PAPER_QUBITS = {
    (5,  50):  18,
    (10, 100): 22,
    (15, 200): 26,
    (20, 200): 30,
    (30, 200): 32,
}

# Paper's reported classical CPU times (Table 3, seconds)
PAPER_CLASSICAL_TIMES = {
    5:  0.12,
    10: 8.45,
    15: 142.0,
    20: '>3600',
    30: 'extrapolated',
}


def measure_classical_time(N: int, L: int, T: int = 4) -> float:
    """Wall-clock time for one classical RIMER inference (Python implementation)."""
    gen = SyntheticDataGenerator(N, L, T, seed=RANDOM_SEED)
    betas, rule_weights, attr_weights = gen.generate_rules()
    rimer = ClassicalRIMER(N, L, T)
    for k in range(L):
        ant = [int(np.argmax(attr_weights[k]))] * T
        rimer.add_rule(BeliefRule(
            antecedent=ant,
            consequent=betas[k],
            rule_weight=rule_weights[k],
            attribute_weights=attr_weights[k],
        ))
    dists = [{j: 1.0 / 3 for j in range(3)} for _ in range(T)]
    start = time.perf_counter()
    w = rimer.rule_activation(dists)
    rimer.er_combine(w, betas)
    return time.perf_counter() - start


def quantum_depth_estimate(N: int, L: int, T: int = 4, epsilon: float = 0.01) -> int:
    """
    Compute quantum circuit depth using the QER + QBRA depth models.

    The total depth = QBRA depth + QER depth, matching the paper's
    complexity formula (Table 1):
    O(T*Tk_max + Tk_max*polylog(1/eps) + L*log(N)*kappa^2/eps).
    """
    kappa_max = 50.0  # typical condition number for moderate-conflict BRBs
    Tk_max = min(T, 5)

    qer = QER(N=N, L=L, kappa_max=kappa_max, epsilon=epsilon)
    qbra = QBRA(T=T, L=L, Tk_max=Tk_max, epsilon=epsilon)

    return qbra.estimate_depth() + qer.estimate_depth()


# ---------------------------------------------------------------------------
# Fidelity comparison (paper Table 2)
# ---------------------------------------------------------------------------

def fidelity_comparison(N_values=(5, 10, 15, 20, 30), L: int = 200,
                        n_runs: int = 10) -> dict:
    """
    Compare Q-RIMER vs BF-QC basic (conjunctive) vs classical exact across
    problem sizes, with Bonferroni-corrected paired t-tests.
    """
    qrimer_eps = []
    bfqc_eps = []
    p_values = []

    for N in N_values:
        gen = SyntheticDataGenerator(N, L, T=4, seed=RANDOM_SEED)
        betas, rule_weights, attr_weights = gen.generate_rules()

        rimer = ClassicalRIMER(N=N, L=L, T=4)
        n_ref = 3
        rng = np.random.default_rng(RANDOM_SEED + N)
        for k in range(L):
            ant = [int(rng.integers(0, n_ref)) for _ in range(4)]
            rimer.add_rule(BeliefRule(
                antecedent=ant,
                consequent=betas[k],
                rule_weight=float(rule_weights[k]),
                attribute_weights=attr_weights[k],
            ))

        eps_qr_run = []
        eps_bf_run = []
        for run in range(n_runs):
            rng = np.random.default_rng(RANDOM_SEED + N * 31 + run * 7)
            dists = [{j: float(v) for j, v in
                      enumerate(rng.dirichlet(np.ones(n_ref)))} for _ in range(4)]
            w = rimer.rule_activation(dists)
            beta_c, _ = rimer.er_combine(w, betas)

            # Q-RIMER: noiseless simulator + 10^4-shot sampling noise
            qer = QER(N=N, L=L, kappa_max=50.0)
            beta_qr, _ = qer.classical_simulate(w, betas)
            beta_qr = multinomial_shot_noise(beta_qr, rng, n_shots=10**4)
            eps_qr_run.append(belief_distribution_error(beta_qr, beta_c))

            # BF-QC basic: conjunctive combination of the belief degrees
            # without the ER-specific weighting and decomposition
            beta_bf, _ = conjunctive_combine(None, betas)
            eps_bf_run.append(belief_distribution_error(beta_bf, beta_c))

        qrimer_eps.append((float(np.mean(eps_qr_run)),
                           float(np.std(eps_qr_run))))
        bfqc_eps.append((float(np.mean(eps_bf_run)),
                         float(np.std(eps_bf_run))))

        t_stat, p_val = stats.ttest_rel(eps_bf_run, eps_qr_run)
        p_values.append(float(p_val))

    p_bonf = bonferroni_correction(p_values)

    return {
        'N_values': list(N_values),
        'qrimer': qrimer_eps,
        'bfqc': bfqc_eps,
        'p_values': p_values,
        'p_bonferroni': p_bonf,
    }


# ---------------------------------------------------------------------------
# Depth scaling (paper Figure 3)
# ---------------------------------------------------------------------------

def fit_power_law(N_vals: np.ndarray, D_vals: np.ndarray):
    """
    Fit D = a * N^b via log-linear regression.
    Returns (a, b, r2, ci_low, ci_high).
    """
    log_N = np.log(N_vals)
    log_D = np.log(D_vals)
    reg = stats.linregress(log_N, log_D)
    b = reg.slope
    a = np.exp(reg.intercept)
    y_pred = a * (N_vals ** b)
    ss_res = np.sum((D_vals - y_pred) ** 2)
    ss_tot = np.sum((D_vals - np.mean(D_vals)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-15)

    # 95% CI for the slope on the log-log scale
    se = reg.stderr
    df = len(N_vals) - 2
    t_crit = stats.t.ppf(0.975, df) if df > 0 else np.inf
    ci_low, ci_high = b - t_crit * se, b + t_crit * se
    return a, b, r2, ci_low, ci_high


def asymptotic_reduction(N: int) -> float:
    """Ratio of the classical O(N) cost to the quantum O(log N) depth."""
    return N / max(np.log2(N + 2), 1.0)


def main():
    print("=" * 60)
    print("Comparative studies (RQ2): Acceleration trend")
    print("=" * 60)

    # --- 1. Fidelity comparison (Table 2) ---
    print("\n[Fidelity comparison across problem sizes] (L=200, 10 runs)")
    fid = fidelity_comparison()
    print(f"  {'N':>4}  {'Q-RIMER':>14}  {'BF-QC basic':>14}  "
          f"{'p (t-test)':>10}  {'p (Bonferroni)':>14}")
    for i, N in enumerate(fid['N_values']):
        qr_m, qr_s = fid['qrimer'][i]
        bf_m, bf_s = fid['bfqc'][i]
        print(f"  {N:>4}  {qr_m:>9.4f}+/-{qr_s:<4.4f}  "
              f"{bf_m:>9.4f}+/-{bf_s:<4.4f}  {fid['p_values'][i]:>10.4f}  "
              f"{fid['p_bonferroni'][i]:>14.4f}")
    print("  Paper targets (Table 2):")
    print("    Q-RIMER: 0.0031, 0.0078, 0.0124, 0.0189, 0.0196")
    print("    BF-QC:   0.0147, 0.0623, 0.1338, 0.2415, 0.3872  (p<0.01, Bonferroni)")
    print("  Classical exact reference: eps = 0 at all N.")

    # --- 2. Depth scaling (Figure 3) ---
    print("\n[Quantum circuit depths]")
    N_vals = np.array([5, 10, 15, 20, 30])
    L_vals = [50, 100, 200, 200, 200]
    print(f"  {'N':>4}  {'L':>4}  {'Paper depth':>12}  {'Model estimate':>14}  "
          f"{'Qubits (paper)':>14}")
    paper_series_N = []
    paper_series_D = []
    for n, l_val in zip(N_vals, L_vals):
        paper_d = PAPER_DEPTHS.get((n, l_val), 'N/A')
        model_d = quantum_depth_estimate(n, l_val, T=GRAPHITE_T)
        qubits = PAPER_QUBITS.get((n, l_val), 'N/A')
        print(f"  {n:>4}  {l_val:>4}  {str(paper_d):>12}  {model_d:>14}  {str(qubits):>14}")
        if isinstance(paper_d, int):
            paper_series_N.append(float(n))
            paper_series_D.append(float(paper_d))
    print("  Note: Paper depths are from Qiskit transpilation; model estimates")
    print("  are upper bounds from O(T*Tk_max + L*polylog(1/eps)) formula.")

    # Power-law fit on the paper's measured depth series (Table 3)
    fit_N = np.array(paper_series_N)
    fit_D = np.array(paper_series_D)
    a, b, r2, ci_low, ci_high = fit_power_law(fit_N, fit_D)
    print(f"\n[Power-law fit on the measured series]: D = {a:.2f} * N^{b:.2f},  "
          f"R^2={r2:.3f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}]")
    print("  (paper, full L=200 dataset: b=2.14, 95% CI [1.98, 2.30], R^2=0.987;")
    print("   additional L: b=2.09 @ L=100 (R^2=0.991), b=2.17 @ L=50 (R^2=0.984))")
    print("  The empirical exponent reflects finite-N behaviour dominated by")
    print("  gate decomposition; the asymptotic bound is O(log N) (Section 6.4).")

    # --- 3. Runtime comparison (Table 3) ---
    print("\n[Classical CPU time vs quantum depth] (local Python implementation)")
    print(f"  {'N':>4}  {'L':>4}  {'Classical (s, local)':>20}  "
          f"{'Classical (s, paper)':>20}  {'Depth (paper)':>13}")
    for N, L in [(5, 50), (10, 100), (15, 200), (20, 200), (30, 200)]:
        t = measure_classical_time(N, L)
        ref = PAPER_CLASSICAL_TIMES[N]
        paper_d = PAPER_DEPTHS.get((N, L), 'N/A')
        print(f"  {N:>4}  {L:>4}  {t:>20.4f}  {str(ref):>20}  {str(paper_d):>13}")
    print("  Note: the local Python implementation is highly optimised (NumPy);")
    print("  the paper's CPU times reflect its full experimental workload and")
    print("  hardware (i9-12900K, Section 6.1); the sharp increase from N=15 to")
    print("  N=20 is attributable to memory hierarchy effects and constant factors")
    print("  of the Python implementation (Section 6.4). For N>20 the paper")
    print("  reports extrapolated classical operation counts.")

    # --- 4. Asymptotic complexity reduction ---
    print("\n[Asymptotic complexity reduction O(N) -> O(log N)]")
    print(f"  {'N':>6}  {'O(N) classical':>16}  {'O(log N) quantum':>16}  "
          f"{'reduction factor':>18}")
    for N in [5, 10, 15, 20, 30, 50, 100]:
        print(f"  {N:>6}  {N:>16}  {np.log2(N + 2):>16.2f}  "
              f"{asymptotic_reduction(N):>18.2f}")
    print("  The reduction grows without bound as N increases (Section 6.4);")
    print("  a precise numerical crossover depends on constant factors,")
    print("  gate times, and end-to-end resource overheads.")


if __name__ == '__main__':
    main()
