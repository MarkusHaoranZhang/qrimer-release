"""
Hierarchical Quantum Inference Pipeline.
Implements Section 5.5: measurement-free layer composition.

Two execution modes
-------------------
classical_simulate (default)
    Runs each layer's ER combination classically in sequence, passing the
    belief distribution from layer m as the input distribution to layer m+1.
    This is the correct validation path and reproduces the paper's results.

build_circuit
    Constructs the full Qiskit register skeleton for all layers.  Gate-level
    composition across layers (without intermediate measurement) requires
    transpilation against a specific backend and is left to the deployment stage.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from .qbra import QBRA
from .qbrb import QBRB
from .qer import QER


class QuantumPipeline:
    """
    Hierarchical quantum inference pipeline combining Q-BRB, QBRA, and QER.

    Parameters
    ----------
    layers_config : list of dict
        Each dict must contain keys: 'N', 'L', 'T', and optionally 'Tk_max'.
        Example for a two-layer pipeline::

            [
                {'N': 5, 'L': 20, 'T': 4, 'Tk_max': 2},
                {'N': 3, 'L': 15, 'T': 5, 'Tk_max': 2},
            ]

    kappa : float
        Maximum condition number for QER (used for depth estimation).
    epsilon : float
        Global approximation error bound.
    """

    def __init__(self, layers_config: list, kappa: float, epsilon: float = 0.01):
        self.layers  = layers_config
        self.kappa   = kappa
        self.epsilon = epsilon
        self.modules: list[tuple[QBRB, QBRA, QER]] = []
        for cfg in layers_config:
            qbrb = QBRB(cfg['N'], cfg['L'], cfg['T'])
            qbra = QBRA(cfg['T'], cfg['L'], cfg.get('Tk_max', 5), epsilon)
            qer  = QER(cfg['N'], cfg['L'], kappa, epsilon)
            self.modules.append((qbrb, qbra, qer))

    # ------------------------------------------------------------------
    # Classical simulation (exact, used for validation)
    # ------------------------------------------------------------------

    def classical_simulate(
        self,
        input_dists: list[dict[int, float]],
        rules_per_layer: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, float]:
        """
        Run the full hierarchical inference classically, layer by layer.

        The output belief distribution of layer m is converted to an input
        distribution for layer m+1 using the same H=2, M=1, L=0 key encoding
        as the rule antecedents.  No intermediate measurement is performed;
        the belief vector is passed directly as a Python dict.

        This implements the measurement-free pipeline of Section 5.5 in the
        classical simulation regime.  Total depth is additive: D_tot = sum D_m.

        Parameters
        ----------
        input_dists : list of dict
            Input belief distributions for the first layer's attributes.
            Each dict maps referential value index to belief degree.
        rules_per_layer : list of (betas, rule_weights, attr_weights)
            One tuple per layer.  betas has shape (L, N), rule_weights (L,),
            attr_weights (L, T).

        Returns
        -------
        beta   : np.ndarray of shape (N_last,)
        beta_D : float
        """
        from .rimer import BeliefRule, ClassicalRIMER

        current_dists = input_dists

        for m, ((qbrb, qbra, qer), (betas, rule_weights, attr_weights)) in \
                enumerate(zip(self.modules, rules_per_layer)):

            cfg = self.layers[m]
            N, L, T = cfg['N'], cfg['L'], cfg['T']

            # Build a ClassicalRIMER for this layer.
            # Layer 0 reads the T input attributes; each subsequent layer
            # reads a single antecedent: the belief state propagated from
            # the previous layer (Section 5.5, measurement-free composition).
            rimer = ClassicalRIMER(N=N, L=L, T=T)
            for k in range(L):
                if m == 0:
                    # Antecedent referential values drawn from each input
                    # attribute's referential value space
                    ant = [int(k % (max(current_dists[i].keys()) + 1))
                           for i in range(T)]
                else:
                    # Antecedent referential value in the previous layer's
                    # conclusion space {0, ..., N_{m-1}-1}
                    ant = [int(k % (max(current_dists[0].keys()) + 1))]
                rimer.add_rule(BeliefRule(
                    antecedent=ant,
                    consequent=betas[k],
                    rule_weight=float(rule_weights[k]),
                    attribute_weights=attr_weights[k],
                ))

            w = rimer.rule_activation(current_dists)
            beta, beta_D = qer.classical_simulate(w, betas)

            # Convert output to input distribution for next layer
            # key = N-1-j maps consequent index j to referential value (H=N-1, L=0)
            current_dists = [{(N - 1 - j): float(beta[j]) for j in range(N)}]

        return beta, beta_D

    # ------------------------------------------------------------------
    # Quantum circuit construction (register skeleton)
    # ------------------------------------------------------------------

    def build_circuit(self) -> QuantumCircuit:
        """
        Build the register skeleton for the hierarchical quantum circuit.

        Allocates rule, consequent, and antecedent registers for each layer.
        The full gate-level composition (connecting layer m output to layer m+1
        input without intermediate measurement) requires transpilation against
        a specific backend and is left to the deployment stage.

        Returns
        -------
        QuantumCircuit with all registers allocated (no gates applied).
        """
        qc = QuantumCircuit()
        for m, (qbrb, qbra, qer) in enumerate(self.modules):
            qc.add_register(QuantumRegister(qbrb.num_rule_qubits, f'rule_{m}'))
            qc.add_register(QuantumRegister(qbrb.num_cons_qubits, f'cons_{m}'))
            qc.add_register(QuantumRegister(qbrb.num_ant_qubits,  f'ant_{m}'))
        return qc

    # Keep old name as alias
    def build_pipeline(self) -> QuantumCircuit:
        """Alias for build_circuit(). Kept for backward compatibility."""
        return self.build_circuit()

    # ------------------------------------------------------------------
    # Depth estimation
    # ------------------------------------------------------------------

    def get_total_depth(self) -> int:
        """
        Total estimated circuit depth = sum of per-layer QER depths.
        Additive property per Section 5.5: D_tot = sum_m D_m.
        """
        return sum(mod[2].estimate_depth() for mod in self.modules)

    def get_layer_depths(self) -> list[dict]:
        """
        Per-layer depth breakdown.

        Returns
        -------
        list of dicts with keys 'qbrb', 'qbra', 'qer'.
        """
        return [
            {
                'qbrb': qbrb.estimate_depth(),
                'qbra': qbra.estimate_depth(),
                'qer':  qer.estimate_depth(),
            }
            for qbrb, qbra, qer in self.modules
        ]
