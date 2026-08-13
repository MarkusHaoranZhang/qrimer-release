# Quantum RIMER

Official implementation of the paper:
**"Quantum-Accelerated Efficient Information Processing via Belief Rule Base Inference with Polynomial Speedup"**

Quantum RIMER reduces the complexity of evidential reasoning (ER) combination
from O(L·N) to O(L·log N) (general Dempster-Shafer combination over the full
power set costs O(4^N); the ER rule restricts belief assignment to the N
singleton grades and the frame of discernment, yielding O(L·N)) by encoding
belief rule bases as quantum superposition states and executing the ER
synthesis via the MEoB (matrix evolution of BBA) algorithm.  The user-facing
output remains the same interpretable belief distribution as classical RIMER;
the quantum encoding operates only at the internal computational layer.

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10 and Qiskit 2.x.

---

## Quick Start

```python
from core.rimer import ClassicalRIMER, BeliefRule
from core.qer import QER
import numpy as np

# Classical ER combination (ground truth)
rimer = ClassicalRIMER(N=3, L=2, T=1)
rimer.add_rule(BeliefRule([2], np.array([0.7, 0.2, 0.1]), 1.0, np.array([1.0])))
rimer.add_rule(BeliefRule([1], np.array([0.1, 0.8, 0.1]), 1.0, np.array([1.0])))
dists = [{2: 0.8, 1: 0.2, 0: 0.0}]
w = rimer.rule_activation(dists)
beta_c, beta_D_c = rimer.er_combine(w, np.array([[0.7,0.2,0.1],[0.1,0.8,0.1]]))

# QER classical simulation (identical result, O(L*N))
qer = QER(N=3, L=2, kappa_max=10.0)
beta_q, beta_D_q = qer.classical_simulate(w, np.array([[0.7,0.2,0.1],[0.1,0.8,0.1]]))

print(f"Classical: {beta_c}")
print(f"QER sim  : {beta_q}")  # identical to classical
```

---

## Run Experiments

Run from the project root directory (after `pip install -e .`):

| Command | Research Question |
|---------|-------------------|
| `python -m experiments.run_preliminary` | RQ1 — Correctness on graphite benchmark |
| `python -m experiments.run_comparative` | RQ2 — Acceleration trend |
| `python -m experiments.run_ablation`    | RQ3 — Component contributions |
| `python -m experiments.run_sensitivity` | RQ4 — Condition number / noise / sparsity |
| `python -m experiments.run_extended`    | Extended — Large N & maritime case study |

## Generate Figures

```bash
python visualization/plot_correctness.py
python visualization/plot_ablation.py
python visualization/plot_depth.py
python visualization/plot_kappa.py
```

Figures are saved to `outputs/figures/` as PDF files.

---

## Run Tests

```bash
# Install in development mode first (resolves all imports cleanly)
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=core --cov-report=term-missing
```

105 tests covering: ER combination, rule activation, membership functions,
input completeness factor, QER classical simulation, Hermitian dilation
correctness (B matrix encodes the ER bilinear map with block-encoded
conflict normalisation K), target-state statevector verification (Born rule
probabilities match classical belief distribution), QBRB circuit building,
QBRA three-step circuit structure (including the Step-3 quantum division
circuit), the BF-QC conjunctive and CRI baselines, public API contract,
input validation, depth scaling, metrics, and statistics.

---

## One-Command Reproducibility

```bash
# Reproduce all paper results (requires pip install -e ".[dev]" first)
make reproduce

# Or step by step:
make test          # 105 unit tests
make experiments   # RQ1–RQ4 + extended studies
make figures       # Generate all 4 PDF figures
```

Additional scripts:
```bash
python experiments/measure_depth.py   # Actual transpiled circuit depths
python experiments/run_hardware.py --dry-run  # Hardware feasibility check
```

---

## Correspondence with Paper Sections

| Paper Section | Code Module | Description |
|---------------|-------------|-------------|
| Section 5.1 (Q-BRB) | `core/qbrb.py` | Quantum BRB state preparation |
| Section 5.2 (Input) | `core/rimer.py` `input_transform` | Input transformation |
| Section 5.3 (QBRA) | `core/qbra.py` | Quantum belief rule activation |
| Section 5.4 (QER) | `core/qer.py` | Quantum evidential reasoning engine |
| Section 5.5 (Pipeline) | `core/pipeline.py` | Hierarchical quantum pipeline |
| Section 6.1 (Baselines) | `core/bfqc.py`, `core/cri.py` | BF-QC conjunctive + CRI baselines |
| Section 6.1 (Metrics) | `analysis/metrics.py` | ε_β, Kendall τ, ignorance fidelity |
| Section 6.1 (Sampling) | `analysis/sampling.py` | 10⁴-shot measurement statistics |
| Section 6.2 (RQ1) | `experiments/run_preliminary.py` | Correctness validation |
| Section 6.3 (RQ3) | `experiments/run_ablation.py` | Ablation studies |
| Section 6.4 (RQ2) | `experiments/run_comparative.py` | Acceleration trend |
| Section 6.5 (RQ4) | `experiments/run_sensitivity.py` | Sensitivity analysis |
| Section 6.6 (Extended) | `experiments/run_extended.py` | Large N & maritime case study |
| Appendix A | `core/qer.py` `construct_hermitian_matrix` | Hermitian dilation (element-wise verified in `tests/test_qer.py::test_matches_appendix_a`) |
| Appendix B | `core/qbra.py` `_step1`, `_step2`, `_step3` | Circuit templates (B.2 division circuit) |
| Appendix B.3 | not simulated (documented) | Direct quantum input transformation — provided in the paper for completeness only |
| Appendix C | this repository | Data and code availability |

---

## Assumptions and Scope

This implementation validates Q-RIMER through **classical quantum simulation**,
which is the standard methodology for quantum algorithm papers prior to
fault-tolerant hardware availability. The following assumptions are explicitly
stated in the paper (Section 5 and Section 6) and reflected in the code:

1. **QRAM availability** — Input belief distributions are assumed loadable in
   O(log N) time via quantum random access memory. The code models this as
   direct array access (`classical_simulate` receives pre-computed weights).
   Per the paper (Sections 5.2 and 6.7), this assumption can be relaxed
   through variational encoding schemes.

2. **Fault-tolerant execution** — The MEoB subroutine requires quantum
   error correction. The code uses `HamiltonianGate` for structural depth
   analysis; actual execution would require a fault-tolerant backend with
   T1 ≥ 100 µs and two-qubit gate error ≤ 10⁻².

3. **Moderate condition number** — The polynomial speedup is effective when
   κ < 50 (typical for most RIMER applications). High-conflict cases with
   κ > 500 degrade the advantage. The sensitivity analysis (RQ4) quantifies
   this boundary.

4. **Statevector simulation** — All reported results use the noiseless
   statevector simulator or calibrated noise models. The `classical_simulate`
   path implements the paper's matrix formulation exactly; correctness error
   metrics (RQ1) are derived from the Born-rule measurement statistics of the
   Q-RIMER output-state circuit executed on the Qiskit statevector simulator
   (10⁴ shots, `QER.simulate_measurement` / `simulate_belief_measurement`,
   per the response to Reviewer 2, Comment 4).

These assumptions are consistent with the methodology of the predecessor work
(BF-QC, Expert Systems with Applications, 2023) and with the broader quantum
algorithm literature (HHL, PRL 2009; quantum SVM, PRL 2014).

---

## Circuit Depth Measurements

Table 3 in the paper reports quantum circuit depths obtained from the Qiskit
transpiler applied to the compiled QER circuits. Two depth metrics are
available in this codebase:

- **`estimate_depth()`** — Returns the asymptotic scaling coefficient
  O(L × polylog(1/ε)). This captures how depth *grows* with problem size
  but does not include constant factors from gate decomposition.  The full
  QER depth bound is O(L · log N · κ̄²/ε) (Table 1); the empirical power-law
  exponent b = 2.14 (finite-N behaviour, Section 6.4) reflects constant
  factors and the κ̄² dependence and does not contradict the asymptotic
  O(log N) bound.

- **`experiments/measure_depth.py`** — Builds actual Qiskit circuits for
  small problem sizes (N ≤ 5) and reports transpiled depth after
  optimization level 1 decomposition into {CX, U3} basis gates.

For large problem sizes (N > 5), full circuit transpilation exceeds
simulator memory locally.  The paper's Table 3 values (Qiskit 2.x
transpilation on the authors' hardware) are therefore carried as reference
constants in `experiments/run_comparative.py`; the depth-scaling figure is
used to illustrate scaling trends, not as a direct speed comparison
(Section 6.4).

---

## Execution Modes

### Classical simulation (validation)
`QER.classical_simulate()` and `QuantumPipeline.classical_simulate()` implement
the paper's matrix MEoB formulation exactly in Python.  Results are identical to
`ClassicalRIMER` up to floating-point precision.  Use this path for validation
and benchmarking.

### Quantum circuit (depth analysis & hardware deployment)
`QER.build_circuit()`, `QBRB.prepare_full_brb()`, and `QBRA.full_qbra_circuit()`
build Qiskit circuits whose depth scales as described in the paper.  Each MEoB
combination step is a `HamiltonianGate` e^{-i M_ER^(k)} built from the
block-encoded Hermitian dilation (forward matrix-vector multiplication via
matrix function application, Section 5.4); full fault-tolerant execution
requires a hardware backend with sufficient coherence time.

---

## Project Structure

```
qrimer_project/
├── pyproject.toml             # Package metadata and dependencies
├── config.py                  # Global constants (Section 6.1)
├── conftest.py                # Pytest configuration
├── Makefile                   # One-command build/test/reproduce
├── LICENSE                    # MIT license
├── .github/workflows/ci.yml   # GitHub Actions CI
├── core/
│   ├── __init__.py            # Public API exports
│   ├── api.py                 # Stable public interface (QRIMEREngine)
│   ├── rimer.py               # Classical RIMER engine (ground truth)
│   ├── qbrb.py                # Q-BRB encoding — Section 5.1
│   ├── qbra.py                # QBRA operator — Section 5.3
│   ├── qer.py                 # QER engine — Section 5.4
│   ├── pipeline.py            # Hierarchical pipeline — Section 5.5
│   ├── bfqc.py                # BF-QC conjunctive baseline — Section 6.1
│   ├── cri.py                 # CRI fuzzy inference baseline — Section 6.1
│   ├── backends.py            # Backend abstraction (simulator/hardware)
│   └── noise.py               # IBM Falcon noise model
├── experiments/
│   ├── datasets.py            # Benchmark & synthetic data (Section 6.1)
│   ├── run_preliminary.py     # RQ1 — Correctness validation (Section 6.2)
│   ├── run_ablation.py        # RQ3 — Component ablation (Section 6.3)
│   ├── run_comparative.py     # RQ2 — Acceleration trend (Section 6.4)
│   ├── run_sensitivity.py     # RQ4 — Robustness analysis (Section 6.5)
│   ├── run_extended.py        # Extended studies (Section 6.6)
│   ├── run_hardware.py        # Hardware deployment workflow
│   └── measure_depth.py       # Transpiled circuit depth measurement
├── analysis/
│   ├── metrics.py             # epsilon_beta, tau, ignorance fidelity
│   ├── sampling.py            # 10^4-shot measurement sampling models
│   └── statistics.py          # t-test, CI, R^2
├── tests/
│   ├── test_rimer.py          # ClassicalRIMER unit tests
│   ├── test_qer.py            # QER unit tests
│   ├── test_qbrb.py           # QBRB unit tests
│   ├── test_qbra.py           # QBRA unit tests
│   ├── test_bfqc.py           # BF-QC baseline tests
│   ├── test_cri.py            # CRI baseline tests
│   ├── test_metrics.py        # Metrics & statistics tests
│   ├── test_api.py            # Public API contract tests
│   └── bench.py               # Micro-benchmarks (not collected by pytest)
├── visualization/
│   ├── plot_correctness.py    # Figure 1 — Correctness validation
│   ├── plot_ablation.py       # Figure 2 — Ablation study
│   ├── plot_depth.py          # Figure 3 — Depth scaling
│   └── plot_kappa.py          # Figure 4 — Condition number sensitivity
└── outputs/
    ├── figures/               # Generated PDF figures
    ├── tables/
    └── logs/                  # Experiment output (.npz)
```

---

## Known Limitations

- **Benchmark rule table**: The complete 36-rule parameter table for the
  graphite identification benchmark is published in Yang et al. 2006.
  For correctness validation (RQ1), `run_preliminary.py` uses the
  published Table III values directly as ground truth.  The ER-matrix
  ablation (Section 6.3) runs on benchmark-like flat 36-rule bases: the
  qualitative conclusions (sharp error increase, systematic ignorance
  underestimation) are reproduced.

- **MEoB gate**: `QER.build_circuit()` uses `HamiltonianGate` (e^{-iHt})
  to implement the MEoB combination step as described in Section 5.4.
  The Hermitian matrix M_ER^(k) is constructed from the actual mass vectors
  at each step, with the conflict normalisation K block-encoded into the
  synthesis matrix B^(k) (the "additional block-encoding rescaling layer"
  of Appendix A).  The element-wise definition of B^(k) matches Appendix A
  exactly (5N+3+1 non-zero entries; the two D-component output rows are
  permuted so that input and output share the unified
  (m_1..m_N, m̄_D, m̂_D) layout — verified by
  `tests/test_qer.py::test_matches_appendix_a`).  The dilation
  [[0, B], [B^H, 0]] of the rectangular B acts on the (N+2)²-dimensional
  tensor-product register, so the MEoB step requires ceil(log2((N+2)^2))
  qubits — 5 for the benchmark N=3, matching the paper's 27-qubit budget.
  Full execution on quantum hardware requires a fault-tolerant
  device with T1 ≥ 100 µs and two-qubit gate error ≤ 10^-2.

- **QBRA Step 3**: `QBRA._step3` implements the quantum division circuit of
  Appendix B.2 structurally: a quantum summing circuit accumulates
  |Z⟩ = |Σ θ_k α_k⟩ in the auxiliary register (the appendix arranges the
  L additions as an O(log L)-depth binary tree; the template uses
  sequential controlled rotations — the summation depth is sub-dominant
  since the total division depth O(L·log(1/ε)) is dominated by the
  weight-vector amplitude encoding), and a controlled rotation conditioned
  on |Z⟩ applies the precomputed reciprocal arcsin(1/Z).  Like Steps 1-2,
  this is a circuit template for depth analysis; numerical validation uses
  the classical path (`ClassicalRIMER.rule_activation`).

- **Noise robustness (τ)**: The paper's τ degradation under noise reflects
  the full 27-qubit benchmark circuit (D=487 layers), which cannot be
  transpiled locally.  `run_sensitivity.py` executes the 2-qubit
  target-state circuit through the IBM Falcon noise model, reproducing the
  ε_β trend; the τ degradation is documented as arising from the larger
  benchmark circuit.

---

## Citation

```bibtex
@article{quantum_rimer_2026,
  title   = {Quantum-Accelerated Efficient Information Processing via Belief Rule Base Inference with Polynomial Speedup},
  author  = {},
  journal = {},
  year    = {2026}
}
```

## License

MIT
