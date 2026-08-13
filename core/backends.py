"""
Backend abstraction layer for Q-RIMER hardware deployment.

This module provides a unified interface for running Q-RIMER circuits
on different backends: statevector simulator, Aer noise simulator,
and real IBM Quantum hardware.

Usage
-----
    from core.backends import get_backend, run_circuit

    backend = get_backend('aer')           # local noisy simulator
    backend = get_backend('ibm', token=...)  # real hardware
    result = run_circuit(circuit, backend, shots=10000)

Hardware Readiness Checklist
----------------------------
Before deploying on real hardware, verify:
  □ Circuit qubit count ≤ device qubit count
  □ Circuit depth ≤ device coherence budget (depth < T2 / gate_time)
  □ Connectivity map is compatible (or transpile with routing)
  □ Error mitigation strategy selected (ZNE, PEC, or M3)
"""

import numpy as np
from qiskit import QuantumCircuit, transpile


def get_backend(backend_type: str = 'statevector', **kwargs) -> object:
    """
    Get a quantum execution backend.

    Parameters
    ----------
    backend_type : str
        One of: 'statevector', 'aer', 'aer_noise', 'ibm'
    **kwargs :
        Additional config passed to BackendConfig.

    Returns
    -------
    Backend object (Qiskit Aer simulator or IBM Quantum service).
    """
    if backend_type == 'statevector':
        return _StatevectorBackend()

    elif backend_type == 'aer':
        from qiskit_aer import AerSimulator
        return AerSimulator()

    elif backend_type == 'aer_noise':
        from qiskit_aer import AerSimulator

        from core.noise import create_falcon_noise_model
        scale = kwargs.get('noise_scale', 1.0)
        noise_model = create_falcon_noise_model(scale)
        return AerSimulator(noise_model=noise_model)

    elif backend_type == 'ibm':
        return _get_ibm_backend(**kwargs)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}. "
                         f"Choose from: statevector, aer, aer_noise, ibm")


def _get_ibm_backend(token: str | None = None, instance: str = "ibm-q/open/main",
                     min_qubits: int = 27, **kwargs):
    """
    Connect to IBM Quantum and select a suitable backend.

    Requires: pip install qiskit-ibm-runtime
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise ImportError(
            "IBM Quantum backend requires qiskit-ibm-runtime. "
            "Install with: pip install qiskit-ibm-runtime"
        )

    if token:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token,
                                       instance=instance)
    else:
        # Use saved credentials
        service = QiskitRuntimeService(channel="ibm_quantum", instance=instance)

    # Select least-busy backend with enough qubits
    backends = service.backends(
        filters=lambda b: b.num_qubits >= min_qubits and b.status().operational
    )
    if not backends:
        raise RuntimeError(f"No operational backend found with >= {min_qubits} qubits")

    return service.least_busy(backends)


class _StatevectorBackend:
    """Lightweight wrapper for statevector simulation (no shots needed)."""

    def run(self, circuit: QuantumCircuit, shots: int = 1) -> dict:
        from qiskit.quantum_info import Statevector
        sv = Statevector.from_instruction(circuit)
        probs = sv.probabilities_dict()
        # Convert to counts format
        counts = {k: int(v * shots) for k, v in probs.items() if v > 1e-10}
        return _MockResult(counts)


class _MockResult:
    """Minimal result object matching Qiskit result interface."""
    def __init__(self, counts: dict):
        self._counts = counts

    def get_counts(self) -> dict:
        return self._counts


def run_circuit(circuit: QuantumCircuit, backend, shots: int = 10_000,
                optimization_level: int = 2) -> dict:
    """
    Transpile and execute a circuit on the given backend.

    Parameters
    ----------
    circuit : QuantumCircuit
    backend : Qiskit backend (Aer, IBM, or StatevectorBackend)
    shots : int
    optimization_level : int (0-3, higher = more aggressive optimization)

    Returns
    -------
    dict : measurement counts {bitstring: count}
    """
    if isinstance(backend, _StatevectorBackend):
        result = backend.run(circuit, shots=shots)
        return result.get_counts()

    # Transpile for the target backend
    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
    )

    # Execute
    job = backend.run(transpiled, shots=shots)
    result = job.result()
    return result.get_counts()


def estimate_hardware_feasibility(N: int, L: int, T: int = 4,
                                  device_qubits: int = 127,
                                  device_t2_us: float = 80.0,
                                  gate_time_ns: float = 300.0) -> dict:
    """
    Estimate whether a Q-RIMER problem instance is feasible on a given device.

    Two depth estimates are provided:
      - target_state_depth: the StatePreparation circuit (small, O(2^n) gates)
      - full_qer_depth: the iterative MEoB circuit (larger, for full pipeline)

    Parameters
    ----------
    N : int — evaluation grades
    L : int — rules
    T : int — attributes
    device_qubits : int — available qubits (e.g., 127 for IBM Eagle)
    device_t2_us : float — coherence time in microseconds
    gate_time_ns : float — two-qubit gate time in nanoseconds

    Returns
    -------
    dict with feasibility assessment.
    """
    from core.qbra import QBRA
    from core.qer import QER

    qer = QER(N=N, L=L, kappa_max=50.0)
    qbra = QBRA(T=T, L=L, Tk_max=min(T, 5))

    # Qubit requirements for target-state circuit (what actually gets deployed)
    target_qubits = max(1, int(np.ceil(np.log2(N + 1))))

    # Qubit requirements for full QER circuit: the Hermitian dilation
    # [[0, B], [B^H, 0]] of the rectangular B has dimension (N+2)(N+3)
    # (5 qubits for the benchmark N=3, matching the paper's 27-qubit budget)
    qer_qubits = qer.dilation_qubits

    # Depth budget: max gates before decoherence
    coherence_budget = int(device_t2_us * 1000 / gate_time_ns)

    # Target-state depth: O(2^n) for StatePreparation on n qubits
    target_depth = 2 ** target_qubits

    # Full pipeline depth (asymptotic estimate)
    full_depth = qer.estimate_depth() + qbra.estimate_depth()

    return {
        'N': N, 'L': L, 'T': T,
        'target_qubits': target_qubits,
        'full_qubits': qer_qubits,
        'device_qubits': device_qubits,
        'target_depth': target_depth,
        'full_depth': full_depth,
        'coherence_budget': coherence_budget,
        'target_feasible': (target_qubits <= device_qubits
                            and target_depth < coherence_budget),
        'full_feasible': (qer_qubits <= device_qubits
                          and full_depth < coherence_budget),
        'recommendation': _recommend_v2(target_qubits, qer_qubits,
                                        device_qubits, target_depth,
                                        full_depth, coherence_budget),
    }


def _recommend_v2(target_q: int, full_q: int, device_q: int,
                  target_d: int, full_d: int, budget: int) -> str:
    """Generate deployment recommendation."""
    if target_q <= device_q and target_d < budget:
        if full_q <= device_q and full_d < budget:
            return "FULLY FEASIBLE: both target-state and full pipeline fit."
        return ("TARGET-STATE FEASIBLE: can verify belief output on hardware. "
                "Full iterative pipeline exceeds coherence budget.")
    if full_q > device_q:
        return f"NOT FEASIBLE: full circuit needs {full_q} qubits, device has {device_q}."
    return "DEPTH EXCEEDS COHERENCE: reduce L or use error mitigation."
