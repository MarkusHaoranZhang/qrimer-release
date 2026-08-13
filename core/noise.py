"""
Noise model utilities for Q-RIMER robustness testing.
Implements the IBM Falcon-series noise model described in Section 6.1.

Usage
-----
    from core.noise import create_falcon_noise_model, run_noisy_target_state

The noise model applies:
  - Thermal relaxation (T1, T2) on all qubits
  - Depolarizing error on single-qubit gates (rate = SINGLE_QUBIT_ERROR)
  - Depolarizing error on two-qubit gates (rate = TWO_QUBIT_ERROR)
"""

import numpy as np
from qiskit import QuantumCircuit

from config import SHOTS, SINGLE_QUBIT_ERROR, T1, T2, TWO_QUBIT_ERROR


def create_falcon_noise_model(scale_factor: float = 1.0):
    """
    Create a noise model mimicking IBM Falcon-series superconducting devices.

    Parameters
    ----------
    scale_factor : float
        Multiplicative factor on gate error rates.
        scale_factor=1.0 gives the baseline IBM Falcon noise level.
        scale_factor=0 gives the noiseless case.

    Returns
    -------
    NoiseModel or None (if qiskit-aer is not installed)
    """
    try:
        from qiskit_aer.noise import (
            NoiseModel,
            depolarizing_error,
            thermal_relaxation_error,
        )
    except ImportError:
        return None

    if scale_factor < 1e-12:
        return None

    noise_model = NoiseModel()

    # Thermal relaxation: config T1/T2 are in microseconds; Qiskit's
    # thermal_relaxation_error expects nanoseconds.  Gate time = 300 ns.
    def _combined(p: float, num_qubits: int):
        err = depolarizing_error(p, num_qubits=num_qubits)
        try:
            err = err.compose(thermal_relaxation_error(T1 * 1e3, T2 * 1e3, 300.0))
        except Exception:
            pass
        return err

    # Single-qubit gate error (10^-3 depolarizing + T1/T2 relaxation)
    noise_model.add_all_qubit_quantum_error(
        _combined(SINGLE_QUBIT_ERROR * scale_factor, 1), ['u'])

    # Two-qubit gate error (10^-2 depolarizing + T1/T2 relaxation)
    noise_model.add_all_qubit_quantum_error(
        _combined(TWO_QUBIT_ERROR * scale_factor, 2), ['cx'])

    return noise_model


def run_noisy_target_state(beta: np.ndarray, beta_D: float,
                           noise_scale: float = 1.0,
                           n_shots: int = SHOTS) -> np.ndarray:
    """
    Prepare the target belief state and measure it under noise.

    This simulates the end-to-end Q-RIMER output: the correct belief
    distribution encoded as a quantum state, then measured with realistic
    gate noise.  The resulting measurement frequencies approximate the
    belief distribution with both shot noise and gate noise.

    Parameters
    ----------
    beta : np.ndarray of shape (N,)
        Classical belief degrees.
    beta_D : float
        Residual ignorance.
    noise_scale : float
        Noise multiplier (0 = noiseless, 1 = IBM Falcon baseline).
    n_shots : int
        Number of measurement shots.

    Returns
    -------
    np.ndarray of shape (N,) — measured belief distribution (frequencies).
    Returns None if qiskit-aer is not available.
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        return None

    N = len(beta)

    # Build target state: |ψ⟩ = Σ √β_j |j⟩ + √β_D |N⟩
    from qiskit.circuit.library import StatePreparation

    amplitudes = np.zeros(N + 1)
    amplitudes[:N] = np.sqrt(np.maximum(beta, 0.0))
    amplitudes[N] = np.sqrt(max(beta_D, 0.0))

    n_qubits = max(1, int(np.ceil(np.log2(len(amplitudes)))))
    dim = 2 ** n_qubits
    padded = np.zeros(dim)
    padded[:len(amplitudes)] = amplitudes
    norm = np.linalg.norm(padded)
    if norm > 1e-12:
        padded /= norm

    qc = QuantumCircuit(n_qubits)
    qc.append(StatePreparation(padded.tolist()), range(n_qubits))
    qc.measure_all()

    # Decompose the StatePreparation into elementary gates so that the
    # noise model can be applied by Aer (state_preparation is not a
    # native Aer instruction).
    from qiskit import transpile
    qc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=1)

    # Run with noise
    noise_model = create_falcon_noise_model(noise_scale)
    backend = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()

    result = backend.run(qc, shots=n_shots).result()
    counts = result.get_counts()

    # Convert counts to belief distribution
    measured = np.zeros(N)
    total_relevant = 0
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        if idx < N:
            measured[idx] += count
            total_relevant += count
        elif idx == N:
            total_relevant += count  # ignorance outcome

    if total_relevant > 0:
        measured /= total_relevant

    return measured
