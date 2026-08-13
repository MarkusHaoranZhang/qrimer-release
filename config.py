"""
Global configuration constants for Quantum RIMER experiments.

Reference: Section 6.1 (Implementation details) of the paper.
"""

# Qiskit simulation parameters
# Paper states Qiskit 0.45 for the original experiments; this codebase
# uses Qiskit 2.x which provides the same statevector simulation behaviour.
QISKIT_VERSION = "2.x"
SHOTS = 10**4
PHASE_ESTIMATION_BITS = 10
# High-conflict sensitivity analysis (kappa > 500): precision raised to
# l = 12 bits to ensure sufficient eigenvalue resolution
# (l >= ceil(log2 kappa) + 2, Section 6.5).
PHASE_ESTIMATION_BITS_HIGH = 12
RANDOM_SEED = 42

# IBM Falcon-series noise model parameters (Section 6.1)
T1 = 100       # microseconds (thermal relaxation)
T2 = 80        # microseconds (dephasing)
SINGLE_QUBIT_ERROR = 1e-3
TWO_QUBIT_ERROR = 1e-2

# Benchmark: graphite identification (Yang et al. 2006)
# 4 input variables, 36 rules, 3 evaluation grades {H, M, L}
GRAPHITE_INPUT_VARS = ['X1', 'X3', 'X5', 'X7']
GRAPHITE_N = 3
GRAPHITE_L = 36
GRAPHITE_T = 4

# Synthetic expansion ranges (Table 2)
N_RANGE = [5, 8, 10, 15, 20, 25, 30, 50, 100]
L_RANGE = [50, 100, 200, 500]

# Dirichlet concentration parameter for synthetic rule generation (Section 6.1)
# gamma=0.5 produces sparse but non-degenerate distributions
GAMMA = 0.5

# Performance target: epsilon_beta must be below this threshold (Section 6.1)
EPSILON_BETA_TARGET = 0.01

# Qubit budget for benchmark case (Section 6.1):
#   rule index: ceil(log2(36)) = 6
#   dilated ER: ceil(log2((3+2)^2)) = ceil(log2(25)) = 5
#   antecedent: T = 4
#   phase estimation clock: l = 10
#   ancilla: 2
#   total: 27 qubits
BENCHMARK_QUBITS = 27
