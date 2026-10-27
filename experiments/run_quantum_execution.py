"""
Circuit-executed ER validation.

This script runs the explicit block-encoded ER kernel on the Qiskit
statevector simulator for small N and all L rules, and compares the result
with the classical RIMER reference.  Each step is a real StatePreparation
followed by the block-encoding unitary and post-selection; the conflict
scalar and state norm are applied between steps (hybrid execution).
"""

import os

import numpy as np

from analysis.cost_model import end_to_end_cost
from analysis.metrics import belief_distribution_error, rank_preservation
from analysis.postselection import composed_error_bound
from core.er_matrices import spectral_report
from core.qer import QER
from core.rimer import BeliefRule, ClassicalRIMER
from experiments.datasets import SyntheticDataGenerator


def build_case(N: int, L: int, T: int = 4, seed: int = 42):
    gen = SyntheticDataGenerator(N, L, T, seed=seed)
    betas, rule_weights, attr_weights = gen.generate_rules()
    rng = np.random.default_rng(seed + 1)
    n_ref = 3
    rimer = ClassicalRIMER(N=N, L=L, T=T)
    for k in range(L):
        ant = [int(rng.integers(0, n_ref)) for _ in range(T)]
        rimer.add_rule(BeliefRule(
            antecedent=ant,
            consequent=betas[k],
            rule_weight=float(rule_weights[k]),
            attribute_weights=attr_weights[k],
        ))
    dists = [{j: float(v) for j, v in enumerate(rng.dirichlet(np.ones(n_ref)))}
             for _ in range(T)]
    w = rimer.rule_activation(dists)
    beta_c, beta_D_c = rimer.er_combine(w, betas)
    return w, betas, beta_c, beta_D_c


def main():
    print("=" * 68)
    print("Circuit-executed ER validation (explicit block encoding)")
    print("=" * 68)

    cases = [(3, 36), (4, 20)]
    results = {}
    for N, L in cases:
        w, betas, beta_c, beta_D_c = build_case(N, L)
        qer = QER(N=N, L=L, kappa_max=50.0)

        beta_q, beta_D_q, info = qer.quantum_simulate(w, betas)
        eps = belief_distribution_error(beta_q, beta_c)
        tau = rank_preservation(beta_q, beta_c)

        rng = np.random.default_rng(42)
        beta_s, beta_D_s, info_s = qer.quantum_simulate(
            w, betas, shots=10**4, rng=rng
        )
        eps_s = belief_distribution_error(beta_s, beta_c)
        tau_s = rank_preservation(beta_s, beta_c)

        print(f"\n[N={N}, L={L}]")
        print(f"  Classical beta       : {np.round(beta_c, 6)}")
        print(f"  Circuit beta (exact) : {np.round(beta_q, 6)}")
        print(f"  Circuit beta (shots) : {np.round(beta_s, 6)}")
        print(f"  epsilon_beta exact   : {eps:.3e}")
        print(f"  epsilon_beta shots   : {eps_s:.3e}")
        print(f"  tau exact / shots    : {tau:.3f} / {tau_s:.3f}")
        print(f"  qubits / raw depth   : {info['n_qubits']} / {info['circuit_depth']}")
        print(f"  per-step success     : min={min(info['per_step_success']):.4f}, "
              f"mean={info['success_probability_mean']:.4f}, "
              f"max={max(info['per_step_success']):.4f}")
        print(f"  coherent product     : {info['coherent_success_product']:.3e}")
        print(f"  error bound (coherent): "
              f"{composed_error_bound(eps, info['per_step_success']):.3e}")
        steps = info['steps']
        cost_coherent = end_to_end_cost(
            steps, info['coherent_success_product'], N, epsilon_stat=0.01
        )
        cost_hybrid = end_to_end_cost(
            steps, info['success_probability_mean'], N, epsilon_stat=0.01
        )
        print(f"  cost (coherent)      : {cost_coherent:.3e}")
        print(f"  cost (hybrid)        : {cost_hybrid:.3e}")

        results[f'N{N}_L{L}'] = {
            'beta_c': beta_c, 'beta_D_c': beta_D_c,
            'beta_q': beta_q, 'beta_D_q': beta_D_q,
            'beta_shots': beta_s, 'beta_D_shots': beta_D_s,
            'eps': eps, 'eps_shots': eps_s,
            'tau': tau, 'tau_shots': tau_s,
            'per_step_success': np.array(info['per_step_success']),
            'coherent_success_product': info['coherent_success_product'],
            'n_qubits': info['n_qubits'],
            'cost_coherent': cost_coherent,
            'cost_hybrid': cost_hybrid,
        }

        if N == 3:
            print("  shot scaling (per step):")
            for n_shots in (10**4, 10**5, 10**6, 10**7):
                rng = np.random.default_rng(42)
                b_shot, _, _ = qer.quantum_simulate(
                    w, betas, shots=n_shots, rng=rng
                )
                print(f"    {n_shots:>9} shots -> eps={belief_distribution_error(b_shot, beta_c):.4f}")

    print("\n[Spectral data of the fixed synthesis operator]")
    print(f"  {'N':>3}  {'B shape':>10}  {'alpha':>9}  {'cond(B)':>10}  "
          f"{'dilation dim':>13}")
    spectral = {}
    for N in (2, 3, 4, 5):
        rep = spectral_report(N)
        spectral[f'N{N}'] = rep
        print(f"  {N:>3}  {str(rep['shape']):>10}  {rep['alpha']:>9.4f}  "
              f"{rep['condition_number']:>10.4f}  "
              f"{str(rep['dilation_shape']):>13}")

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'quantum_execution_results.npz')
    np.savez(out_path, **results,
             spectral_N=np.array([spectral[k]['N'] for k in spectral]),
             spectral_cond=np.array([spectral[k]['condition_number']
                                     for k in spectral]))
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
