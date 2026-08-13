"""
Classical RIMER Inference Engine
Implements the exact ER rule combination as ground truth for validation.

Based on:
- Yang, J. B., et al. "Belief rule-base inference methodology using the
  evidential reasoning approach - RIMER." IEEE Trans. SMC-A, 2006.
- Yang, J. B., & Xu, D. L. "Evidential reasoning rule for evidence
  combination." Artificial Intelligence, 2013.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BeliefRule:
    """A single belief rule in the BRB."""
    antecedent: list[int]           # referential value indices; -1 = attribute not used
    consequent: np.ndarray          # belief degrees for N consequents, shape (N,)
    rule_weight: float = 1.0
    attribute_weights: np.ndarray = field(default_factory=lambda: np.ones(1))

    def __post_init__(self):
        self.consequent = np.asarray(self.consequent, dtype=float)
        self.attribute_weights = np.asarray(self.attribute_weights, dtype=float)


class ClassicalRIMER:
    """
    Classical RIMER inference engine.

    Parameters
    ----------
    N : int  — number of evaluation grades (consequents)
    L : int  — number of belief rules
    T : int  — number of antecedent attributes
    """

    def __init__(self, N: int, L: int, T: int):
        self.N = N
        self.L = L
        self.T = T
        self.rules: list[BeliefRule] = []
        self.ref_values: list[list] = [[] for _ in range(T)]

    def add_rule(self, rule: BeliefRule):
        self.rules.append(rule)

    def set_referential_values(self, attr_idx: int, values: list):
        self.ref_values[attr_idx] = values

    def input_transform(self, input_values,
                        membership_funcs: list[Callable] | None = None
                        ) -> list[dict[int, float]]:
        """
        Transform crisp inputs to belief distributions over referential values.
        Eq. (2) of the paper.
        """
        distributions = []
        for i, val in enumerate(input_values):
            if membership_funcs is not None and membership_funcs[i] is not None:
                alphas = membership_funcs[i](val)
                if isinstance(alphas, np.ndarray):
                    alphas = {j: float(a) for j, a in enumerate(alphas)}
                distributions.append(alphas)
            else:
                if isinstance(val, dict):
                    distributions.append(val)
                else:
                    distributions.append({j: float(v) for j, v in enumerate(val)})
        return distributions

    def rule_activation(self, distributions: list[dict[int, float]]) -> np.ndarray:
        """
        Compute normalized activation weights w_k.
        Implements Eqs. (8)-(9) of the paper.

        antecedent value -1 means "this attribute is not used in this rule"
        and contributes a factor of 1.0 (no constraint).

        Optimised: the inner loop over used attributes is kept (it is short,
        typically T_k ≤ 5), but the outer loop over L rules uses early-exit
        and avoids redundant Python attribute lookups.
        """
        raw_weights = np.empty(self.L)

        for k in range(self.L):
            rule = self.rules[k]
            ant  = rule.antecedent
            aw   = rule.attribute_weights

            # Collect (attr_index, ref_value, weight) for used antecedents
            alpha_k  = 1.0
            max_delta = 0.0
            used_items = []
            for i, ref_idx in enumerate(ant):
                if ref_idx == -1:
                    continue
                d = float(aw[i]) if i < len(aw) else 1.0
                if d > max_delta:
                    max_delta = d
                used_items.append((i, ref_idx, d))

            if not used_items:
                raw_weights[k] = rule.rule_weight
                continue

            inv_max = 1.0 / max_delta if max_delta > 1e-12 else 0.0
            for i, ref_idx, delta_i in used_items:
                delta_bar = delta_i * inv_max
                alpha_i   = distributions[i].get(ref_idx, 0.0) \
                             if i < len(distributions) else 0.0
                if alpha_i > 1e-12 and delta_bar > 1e-12:
                    alpha_k *= alpha_i ** delta_bar
                elif delta_bar > 0:
                    alpha_k = 0.0
                    break

            raw_weights[k] = rule.rule_weight * alpha_k

        total = raw_weights.sum()
        return raw_weights / total if total > 1e-12 else np.zeros(self.L)

    def er_combine(self, w: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Perform ER rule combination.
        Implements Eqs. (4)-(6) of the paper exactly.

        The mass vectors are built in a single vectorised pass and the
        L-1 combination steps use NumPy slice operations, avoiding
        per-component Python-level casts.

        Parameters
        ----------
        w     : np.ndarray of shape (L,)
        betas : np.ndarray of shape (L, N)

        Returns
        -------
        beta   : np.ndarray of shape (N,)
        beta_D : float  — residual ignorance

        Raises
        ------
        ValueError
            If w or betas have incompatible shapes.
        """
        if w.shape != (self.L,):
            raise ValueError(
                f"w must have shape ({self.L},), got {w.shape}")
        if betas.shape != (self.L, self.N):
            raise ValueError(
                f"betas must have shape ({self.L}, {self.N}), got {betas.shape}")
        N = self.N

        # ---- Build mass vectors in one vectorised pass ----
        m_vectors = np.zeros((self.L, N + 2))
        active = w > 1e-12
        m_vectors[active, :N]    = w[active, None] * betas[active]
        m_vectors[active, N]     = 1.0 - w[active]
        m_vectors[active, N + 1] = w[active] * np.maximum(
            0.0, 1.0 - betas[active].sum(axis=1))
        m_vectors[~active, N]    = 1.0

        m_c = m_vectors[0].copy()

        for k in range(1, self.L):
            m_n = m_vectors[k]
            mc_N = m_c[:N]
            mn_N = m_n[:N]
            conflict = float(mc_N.sum() * mn_N.sum() - np.dot(mc_N, mn_N))
            K = 1.0 / (1.0 - conflict + 1e-15)
            mD_c = m_c[N] + m_c[N + 1]
            mD_n = m_n[N] + m_n[N + 1]
            m_new = np.empty(N + 2)
            m_new[:N]    = K * (mc_N * mn_N + mc_N * mD_n + mD_c * mn_N)
            m_new[N + 1] = K * (m_c[N+1]*m_n[N+1] + m_c[N+1]*m_n[N] + m_c[N]*m_n[N+1])
            m_new[N]     = K * (m_c[N] * m_n[N])
            m_c = m_new

        denom  = 1.0 - m_c[N] + 1e-15
        return m_c[:N] / denom, m_c[N + 1] / denom

    def infer(self, input_values,
              membership_funcs: list[Callable] | None = None) -> tuple[np.ndarray, float]:
        """Full RIMER inference pipeline."""
        distributions = self.input_transform(input_values, membership_funcs)
        w = self.rule_activation(distributions)
        betas = np.array([rule.consequent for rule in self.rules])
        # Input completeness update (Section 3.1 of the paper): incomplete
        # input assessments scale each rule's belief degrees by phi_k.
        betas = betas * self.input_completeness_factors(distributions)[:, None]
        return self.er_combine(w, betas)

    def input_completeness_factors(self,
                                   distributions: list[dict[int, float]]) -> np.ndarray:
        """
        Compute the input completeness factor phi_k for each rule
        (Section 3.1 of the paper):

            phi_k = sum_t tau(t,k) * sum_j alpha_tj / sum_t tau(t,k),

        where tau(t,k) = 1 if attribute U_t is used in rule R_k and 0
        otherwise, and sum_j alpha_tj is the total belief distributed over
        the referential values of attribute t.  Complete inputs
        (sum_j alpha_tj = 1 for every used attribute) yield phi_k = 1.
        """
        phis = np.ones(self.L)
        for k in range(self.L):
            ant = self.rules[k].antecedent
            num = 0.0
            den = 0.0
            for i, ref_idx in enumerate(ant):
                if ref_idx == -1:
                    continue
                total_belief = sum(distributions[i].values()) \
                    if i < len(distributions) else 0.0
                num += total_belief
                den += 1.0
            phis[k] = num / den if den > 1e-12 else 1.0
        return phis


# --------------------------------------------------------------------------
# Membership functions for the graphite identification benchmark
# --------------------------------------------------------------------------

def triangular_membership(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def trapezoidal_membership(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:
        return (d - x) / (d - c)


def create_graphite_membership_functions() -> list[Callable]:
    """
    Membership functions for the graphite benchmark.
    Referential value encoding matches the rule base: H=2, M=1, L=0.
    Input values are in [0, 1] after normalization.
    """
    def make_membership(val: float) -> dict[int, float]:
        # H (key=2): trapezoidal [0.6, 0.8, 1.0, 1.0]
        # M (key=1): trapezoidal [0.2, 0.4, 0.6, 0.8]
        # L (key=0): trapezoidal [0.0, 0.0, 0.2, 0.4]
        high   = trapezoidal_membership(val, 0.6, 0.8, 1.0, 1.0)
        medium = trapezoidal_membership(val, 0.2, 0.4, 0.6, 0.8)
        low    = trapezoidal_membership(val, 0.0, 0.0, 0.2, 0.4)
        total  = high + medium + low
        if total > 1e-12:
            return {2: high / total, 1: medium / total, 0: low / total}
        return {2: 0.0, 1: 0.0, 0: 0.0}

    return [make_membership] * 4


# --------------------------------------------------------------------------
# Hierarchical graphite identification rule base (Yang et al. 2006)
# 5 sub-rule-bases, 36 rules total.
#
# Structure:
#   SRB1: X2 ← X1          (3 rules)
#   SRB2: X4 ← (X2, X3)    (6 rules)
#   SRB3: X6 ← (X4, X5)    (9 rules)
#   SRB4: X8 ← (X7, X1)    (9 rules)
#   SRB5: X9 ← (X6, X8)    (9 rules)
#
# Antecedent referential value encoding: H=2, M=1, L=0  (matches paper)
#
# Data format: (antecedent_indices, [beta_H, beta_M, beta_L], rule_weight, attr_weights)
# --------------------------------------------------------------------------

# fmt: off
GRAPHITE_SRB1 = [
    # SRB1: X2 ← X1 (identity mapping, T=1)
    ([2], [1.0, 0.0, 0.0], 1.0, [1.0]),  # X1=H → X2=H
    ([1], [0.0, 1.0, 0.0], 1.0, [1.0]),  # X1=M → X2=M
    ([0], [0.0, 0.0, 1.0], 1.0, [1.0]),  # X1=L → X2=L
]

GRAPHITE_SRB2 = [
    # SRB2: X4 ← (X2, X3), T=2. antecedent[0]=X2, antecedent[1]=X3
    ([-1, 0], [0.0, 0.0, 1.0], 1.0, [1.0, 1.0]),  # X3=L → low
    ([0, -1], [0.0, 0.0, 1.0], 1.0, [1.0, 1.0]),  # X2=L → low
    ([2,  2], [1.0, 0.0, 0.0], 1.0, [1.0, 1.0]),  # X2=H, X3=H → high
    ([2,  1], [0.3, 0.7, 0.0], 0.7, [1.0, 1.0]),  # X2=H, X3=M
    ([1,  2], [0.3, 0.7, 0.0], 0.7, [1.0, 1.0]),  # X2=M, X3=H
    ([1,  1], [0.0, 1.0, 0.0], 0.7, [1.0, 1.0]),  # X2=M, X3=M → medium
]

GRAPHITE_SRB3 = [
    # SRB3: X6 ← (X5, X4), T=2. antecedent[0]=X5, antecedent[1]=X4
    # NOTE: The complete 36-rule table is published in Yang et al. 2006
    # (paper Section 6.1).
    ([2, 2], [1.0, 0.0, 0.0], 1.0, [1.0, 1.0]),  # X5=H, X4=H
    ([2, 1], [0.4, 0.6, 0.0], 1.0, [1.0, 1.0]),  # X5=H, X4=M
    ([2, 0], [0.0, 1.0, 0.0], 1.0, [1.0, 1.0]),  # X5=H, X4=L
    ([1, 2], [0.2, 0.8, 0.0], 1.0, [1.0, 1.0]),  # X5=M, X4=H
    ([1, 1], [0.0, 1.0, 0.0], 0.4, [1.0, 1.0]),  # X5=M, X4=M
    ([1, 0], [0.0, 0.2, 0.8], 1.0, [1.0, 1.0]),  # X5=M, X4=L
    ([0, 2], [0.1, 0.3, 0.6], 0.2, [1.0, 1.0]),  # X5=L, X4=H
    ([0, 1], [0.0, 0.2, 0.8], 1.0, [1.0, 1.0]),  # X5=L, X4=M
    ([0, 0], [0.0, 0.0, 1.0], 1.0, [1.0, 1.0]),  # X5=L, X4=L
]

GRAPHITE_SRB4 = [
    # SRB4: X8 ← (X7, X1), T=2
    ([2, 2], [1.0, 0.0, 0.0], 1.0, [1.0, 1.0]),
    ([2, 1], [0.3, 0.7, 0.0], 0.2, [1.0, 1.0]),
    ([2, 0], [0.0, 0.3, 0.7], 0.8, [1.0, 1.0]),
    ([1, 2], [0.4, 0.6, 0.0], 1.0, [1.0, 1.0]),
    ([1, 1], [0.0, 1.0, 0.0], 0.4, [1.0, 1.0]),
    ([1, 0], [0.0, 0.1, 0.9], 1.0, [1.0, 1.0]),
    ([0, 2], [0.1, 0.3, 0.6], 1.0, [1.0, 1.0]),
    ([0, 1], [0.0, 0.3, 0.7], 1.0, [1.0, 1.0]),
    ([0, 0], [0.0, 0.0, 1.0], 1.0, [1.0, 1.0]),
]

GRAPHITE_SRB5 = [
    # SRB5: X9 ← (X6, X8), T=2 — top-level output
    ([2, 2], [1.0, 0.0, 0.0], 1.0, [1.0, 1.0]),
    ([2, 1], [0.2, 0.8, 0.0], 0.6, [1.0, 1.0]),
    ([2, 0], [0.1, 0.2, 0.7], 1.0, [1.0, 1.0]),
    ([1, 2], [0.2, 0.8, 0.0], 0.6, [1.0, 1.0]),
    ([1, 1], [0.0, 1.0, 0.0], 0.6, [1.0, 1.0]),
    ([1, 0], [0.0, 0.1, 0.9], 1.0, [1.0, 1.0]),
    ([0, 2], [0.1, 0.2, 0.7], 1.0, [1.0, 1.0]),
    ([0, 1], [0.0, 0.1, 0.9], 1.0, [1.0, 1.0]),
    ([0, 0], [0.0, 0.0, 1.0], 1.0, [1.0, 1.0]),
]
# fmt: on

def _build_srimer(rules_data: list, T: int, N: int = 3
                  ) -> tuple['ClassicalRIMER', np.ndarray]:
    """Build a sub-RIMER from a compact rules_data list."""
    L = len(rules_data)
    rimer = ClassicalRIMER(N=N, L=L, T=T)
    betas = np.zeros((L, N))
    for i, (ante_list, bd, rw, aw) in enumerate(rules_data):
        rule = BeliefRule(
            antecedent=ante_list,
            consequent=np.array(bd, dtype=float),
            rule_weight=float(rw),
            attribute_weights=np.array(aw, dtype=float),
        )
        rimer.add_rule(rule)
        betas[i] = np.array(bd, dtype=float)
    return rimer, betas


def _run_sub(rimer: 'ClassicalRIMER', betas: np.ndarray,
             dists: list[dict[int, float]]) -> dict[int, float]:
    """
    Run one sub-rule-base; return output as a belief distribution dict.

    The output keys use the same H=2, M=1, L=0 encoding as the rule antecedents,
    so the output can be directly fed as input to the next sub-rule-base.

    Mapping: consequent index j → referential value key (N-1-j), i.e.
      j=0 → key=(N-1)=2 (H),  j=1 → key=1 (M),  j=2 → key=0 (L).
    This assumes consequents are ordered from highest to lowest grade,
    which matches the rule definitions in infer_graphite_hierarchical().
    """
    w = rimer.rule_activation(dists)
    beta, _ = rimer.er_combine(w, betas)
    # Map consequent index j to referential value key: j=0→H=2, j=1→M=1, j=2→L=0
    N = rimer.N
    return {(N - 1 - j): float(beta[j]) for j in range(N)}


def infer_graphite_hierarchical(x1: float, x3: float, x5: float, x7: float,
                                 memb_func: Callable) -> tuple[np.ndarray, float]:
    """
    Full hierarchical RIMER inference for the graphite identification benchmark.

    Implements the 5-layer structure from Yang et al. 2006:
      SRB1: X2 ← X1, SRB2: X4 ← (X2,X3), SRB3: X6 ← (X5,X4),
      SRB4: X8 ← (X7,X1), SRB5: X9 ← (X6,X8)

    Rule parameters are defined in the module-level constants
    GRAPHITE_SRB1 through GRAPHITE_SRB5.

    Parameters
    ----------
    x1, x3, x5, x7 : float  — primary inputs in [0, 1]
    memb_func : callable     — membership function (same for all attributes)

    Returns
    -------
    beta   : np.ndarray of shape (3,)  — belief degrees [H, M, L]
    beta_D : float                     — residual ignorance
    """
    d1 = memb_func(x1)
    d3 = memb_func(x3)
    d5 = memb_func(x5)
    d7 = memb_func(x7)

    srb1, b1 = _build_srimer(GRAPHITE_SRB1, T=1)
    d2 = _run_sub(srb1, b1, [d1])

    srb2, b2 = _build_srimer(GRAPHITE_SRB2, T=2)
    d4 = _run_sub(srb2, b2, [d2, d3])

    srb3, b3 = _build_srimer(GRAPHITE_SRB3, T=2)
    d6 = _run_sub(srb3, b3, [d5, d4])

    srb4, b4 = _build_srimer(GRAPHITE_SRB4, T=2)
    d8 = _run_sub(srb4, b4, [d7, d1])

    srb5, b5 = _build_srimer(GRAPHITE_SRB5, T=2)
    w5 = srb5.rule_activation([d6, d8])
    return srb5.er_combine(w5, b5)


def create_graphite_rules() -> tuple['ClassicalRIMER', np.ndarray]:
    """
    Compatibility shim: returns the SRB5 (top-level) rule base.
    For correct hierarchical inference use infer_graphite_hierarchical().
    """
    return _build_srimer(GRAPHITE_SRB5, T=2)
