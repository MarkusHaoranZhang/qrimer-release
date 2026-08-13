"""
Public API for Q-RIMER.

This module exposes the stable, documented interface that downstream
systems (hardware integrations, web services, embedded controllers)
should import from.  Internal implementation details are in the
individual modules (rimer.py, qer.py, qbra.py, qbrb.py, pipeline.py).

Usage
-----
    from core.api import (
        InferenceRequest,
        InferenceResult,
        QRIMEREngine,
    )

    engine = QRIMEREngine.from_config(n_grades=3, n_rules=36, n_attrs=4)
    request = InferenceRequest(
        betas=..., rule_weights=..., attr_weights=..., input_dists=...
    )
    result = engine.infer(request)
    print(result.belief_degrees, result.ignorance)

Stability
---------
This API follows semantic versioning.  Breaking changes increment the
major version.  Internal modules (prefixed with _) may change without notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ExecutionMode(Enum):
    """Execution mode for the inference engine."""
    CLASSICAL = "classical"           # Exact classical ER combination
    SIMULATE = "simulate"             # Quantum circuit on statevector simulator
    NOISY_SIMULATE = "noisy_simulate" # Quantum circuit with noise model
    HARDWARE = "hardware"             # Real quantum hardware


@dataclass(frozen=True)
class InferenceRequest:
    """
    Immutable input specification for one Q-RIMER inference call.

    All arrays are validated on construction. This is the contract
    between the caller (application layer) and the engine (algorithm layer).
    """
    betas: np.ndarray           # shape (L, N): belief degrees per rule
    rule_weights: np.ndarray    # shape (L,): rule importance weights
    attr_weights: np.ndarray    # shape (L, T): attribute weights per rule
    input_dists: list[dict[int, float]]  # length T: belief over referential values

    def __post_init__(self):
        # Validate shapes — uses raise, not assert, to survive python -O
        L, N = self.betas.shape
        if self.rule_weights.shape != (L,):
            raise ValueError(
                f"rule_weights shape {self.rule_weights.shape} != ({L},)")
        if self.attr_weights.shape[0] != L:
            raise ValueError(
                f"attr_weights rows {self.attr_weights.shape[0]} != L={L}")
        T = self.attr_weights.shape[1]
        if len(self.input_dists) != T:
            raise ValueError(
                f"input_dists length {len(self.input_dists)} != T={T}")

    @property
    def N(self) -> int:
        return self.betas.shape[1]

    @property
    def L(self) -> int:
        return self.betas.shape[0]

    @property
    def T(self) -> int:
        return self.attr_weights.shape[1]


@dataclass(frozen=True)
class InferenceResult:
    """
    Immutable output of one Q-RIMER inference call.

    Contains both the belief distribution and metadata about the execution.
    """
    belief_degrees: np.ndarray     # shape (N,): β_j for each evaluation grade
    ignorance: float               # β_D: residual ignorance
    activation_weights: np.ndarray # shape (L,): w_k per rule
    mode: ExecutionMode            # how the result was computed
    circuit_depth: int | None = None       # quantum circuit depth (if applicable)
    n_qubits: int | None = None            # qubit count (if applicable)
    shots: int | None = None               # measurement shots (if applicable)

    @property
    def dominant_grade(self) -> int:
        """Index of the evaluation grade with highest belief."""
        return int(np.argmax(self.belief_degrees))

    @property
    def is_complete(self) -> bool:
        """Whether the inference is complete (no residual ignorance)."""
        return bool(self.ignorance < 1e-6)


class QRIMEREngine:
    """
    Main inference engine providing a unified interface for all execution modes.

    This is the primary entry point for any system that wants to run
    Q-RIMER inference, whether classically or on quantum hardware.
    """

    def __init__(self, N: int, L: int, T: int, *,
                 kappa_max: float = 50.0,
                 epsilon: float = 0.01,
                 mode: ExecutionMode = ExecutionMode.CLASSICAL):
        self.N = N
        self.L = L
        self.T = T
        self.kappa_max = kappa_max
        self.epsilon = epsilon
        self.mode = mode

        # Lazy-initialize internal modules
        self._qer = None

    @classmethod
    def from_config(cls, n_grades: int, n_rules: int, n_attrs: int,
                    **kwargs) -> QRIMEREngine:
        """Factory method with clear parameter names."""
        return cls(N=n_grades, L=n_rules, T=n_attrs, **kwargs)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """
        Run inference on the given request.

        The execution mode determines whether computation happens classically,
        on a quantum simulator, or on real hardware.  The result format is
        identical regardless of mode, enabling seamless switching.
        """
        if request.N != self.N or request.L != self.L or request.T != self.T:
            raise ValueError(
                f"Request dimensions (N={request.N}, L={request.L}, T={request.T}) "
                f"don't match engine (N={self.N}, L={self.L}, T={self.T})"
            )

        if self.mode == ExecutionMode.CLASSICAL:
            return self._infer_classical(request)
        elif self.mode == ExecutionMode.SIMULATE:
            return self._infer_simulate(request)
        elif self.mode == ExecutionMode.NOISY_SIMULATE:
            return self._infer_noisy(request)
        elif self.mode == ExecutionMode.HARDWARE:
            return self._infer_hardware(request)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_qer(self):
        if self._qer is None:
            from core.qer import QER
            self._qer = QER(self.N, self.L, self.kappa_max, self.epsilon)
        return self._qer

    def _build_rimer_from_request(self, request: InferenceRequest):
        """Build a ClassicalRIMER instance from an InferenceRequest."""
        from core.rimer import BeliefRule, ClassicalRIMER
        rimer = ClassicalRIMER(self.N, self.L, self.T)
        for k in range(self.L):
            rimer.add_rule(BeliefRule(
                antecedent=list(range(self.T)),
                consequent=request.betas[k],
                rule_weight=float(request.rule_weights[k]),
                attribute_weights=request.attr_weights[k],
            ))
        return rimer

    def _infer_classical(self, request: InferenceRequest) -> InferenceResult:
        rimer = self._build_rimer_from_request(request)
        w = rimer.rule_activation(request.input_dists)
        beta, beta_D = rimer.er_combine(w, request.betas)

        return InferenceResult(
            belief_degrees=beta,
            ignorance=beta_D,
            activation_weights=w,
            mode=ExecutionMode.CLASSICAL,
        )

    def _infer_simulate(self, request: InferenceRequest) -> InferenceResult:
        """Run via QER target-state circuit on statevector simulator."""
        from qiskit.quantum_info import Statevector

        rimer = self._build_rimer_from_request(request)
        w = rimer.rule_activation(request.input_dists)

        # Build and simulate target-state circuit
        qer = self._get_qer()
        circ = qer.build_target_state_circuit(w, request.betas)
        sv = Statevector.from_instruction(circ)
        probs = sv.probabilities()

        # Extract belief from probabilities
        total = sum(probs[:self.N + 1])
        beta = np.array(probs[:self.N]) / total if total > 1e-12 else np.zeros(self.N)
        beta_D = probs[self.N] / total if total > 1e-12 else 0.0

        return InferenceResult(
            belief_degrees=beta,
            ignorance=beta_D,
            activation_weights=w,
            mode=ExecutionMode.SIMULATE,
            n_qubits=circ.num_qubits,
            circuit_depth=circ.depth(),
        )

    def _infer_noisy(self, request: InferenceRequest) -> InferenceResult:
        """Run via target-state circuit on noisy Aer simulator."""
        from config import SHOTS
        from core.backends import get_backend, run_circuit

        rimer = self._build_rimer_from_request(request)
        w = rimer.rule_activation(request.input_dists)

        qer = self._get_qer()
        circ = qer.build_target_state_circuit(w, request.betas)
        circ.measure_all()

        backend = get_backend('aer_noise')
        counts = run_circuit(circ, backend, shots=SHOTS)

        # Extract belief from counts
        beta = np.zeros(self.N)
        beta_D_count = 0
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            if idx < self.N:
                beta[idx] += count
            elif idx == self.N:
                beta_D_count += count

        relevant = beta.sum() + beta_D_count
        if relevant > 0:
            beta /= relevant
            beta_D = beta_D_count / relevant
        else:
            beta_D = 0.0

        return InferenceResult(
            belief_degrees=beta,
            ignorance=beta_D,
            activation_weights=w,
            mode=ExecutionMode.NOISY_SIMULATE,
            n_qubits=circ.num_qubits,
            circuit_depth=circ.depth(),
            shots=SHOTS,
        )

    def _infer_hardware(self, request: InferenceRequest) -> InferenceResult:
        """Placeholder for real hardware execution."""
        raise NotImplementedError(
            "Hardware execution requires IBM Quantum credentials. "
            "Use run_hardware.py --backend ibm for the full workflow."
        )
