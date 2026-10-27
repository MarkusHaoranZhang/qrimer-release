# Quantum RIMER

Official implementation of the paper:
**"Quantum-Accelerated Belief Rule Base Inference with Conditional Per-Step Logarithmic Complexity"**

Quantum RIMER gives a conditional per-step reduction of evidential reasoning
(ER) combination from O(N) to O(log N) (general Dempster-Shafer combination
over the full power set costs O(4^N); the ER rule restricts belief assignment
to the N singleton grades and the frame of discernment, yielding O(L·N)).
The total pipeline carries the accumulated post-selection factor Gamma_acc; no
unconditional end-to-end speedup is claimed.  The user-facing output remains
the same interpretable belief distribution as classical RIMER.

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10 and Qiskit 2.x.  The optional PyTorch GPU baseline
can be installed with `pip install -e ".[dev,gpu]"`.

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
| `python -m experiments.run_shot_execution` | RQ1 — Correctness on graphite benchmark with shot readout |
| `python -m experiments.run_structured_resources` | RQ2 — Structured resource scaling |
| `python -m experiments.run_ablation`    | RQ3 — Component contributions |
| `python -m experiments.run_conflict_scan` | RQ4 — Circuit-executed conflict scan |
| `python -m experiments.run_blockencoded_noise_large_n` | RQ4 — Noise at N=3/10/20 |
| `python -m experiments.run_extended`    | Extended — Large N & maritime case study |
| `python -m experiments.run_quantum_execution` | Circuit-executed ER kernel (block encoding, post-selection, shots) |
| `python -m experiments.run_hierarchical_quantum` | Block-encoded two/three-layer pipeline |
| `python -m experiments.run_benchmark_quantum` | 36-rule hierarchical graphite benchmark |
| `python -m experiments.run_coherent_k` | End-to-end coherent-K execution |
| `python -m experiments.run_classical_baseline` | Batched vectorised classical baseline |
| `python -m experiments.run_conflict_mitigation` | High-conflict discounting probe |
| `python -m experiments.run_spectral_analysis` | Singular values / dilation spectrum |
| `python -m experiments.run_crossover` | Empirical fit, classical crossover, truncated status |
| `python -m experiments.run_conflict_scan` | Circuit-executed success probability vs conflict |
| `python -m experiments.run_focal_truncation` | Focal-set K_max=3 truncation probe |

Historical scripts retained for traceability but not used for current evidence:
`run_preliminary.py` (target-state readout emulator), `run_comparative.py`,
`run_sensitivity.py`, `run_noise_scaling.py`, and `measure_depth.py`.

## Generate Figures

```bash
python -m experiments.export_figure_data
python -m visualization.plot_backfill_figures
python -m visualization.plot_depth_generated
python -m visualization.plot_kappa_generated
python -m visualization.plot_composite
```

Figures are saved to `outputs/figures/` as PDF and 600 dpi PNG files
(`correctness`, `ablation`, `depth`, `kappa`). The current plotting scripts
use generated outputs, not the archived legacy data.

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

159 tests covering: ER combination, rule activation, membership functions,
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
make test          # 159 unit tests
make experiments   # RQ1–RQ4 + extended studies
make quantum       # Block-encoded circuit execution across all rules
make hierarchy     # Block-encoded two/three-layer pipeline
make benchmark     # 36-rule hierarchical graphite benchmark
make large         # Full-statevector N=10/20 circuit execution
make resources     # Transpiled block-encoding resource table
make blocknoise    # Single-step block-encoded noise
make strategies   # Block-encoding strategy comparison
make sparseoracle # Sparse column-oracle resources
make rowlcu      # Row-side LCU decomposition
make sparsebe    # Six-term sparse LCU block encoding
make noise         # Noise-model probe for N=3, 10, 20
make baseline      # Batched vectorised classical baseline
make mitigation    # High-conflict discounting probe
make conflict      # Circuit-executed success vs conflict
make spectral      # Singular values / dilation spectrum
make crossover     # Empirical fit / classical crossover analysis
make focal         # Focal-set K_max=3 truncation probe
make figures       # Generate all 4 figures as PDF + 600 dpi PNG
```

Additional scripts:
```bash
python -m experiments.run_quantum_execution   # Block-encoded circuit execution
python -m experiments.run_hierarchical_quantum # Two/three-layer pipeline
python -m experiments.run_benchmark_quantum    # 36-rule hierarchical benchmark
python -m experiments.run_large_n_quantum      # Large-N N=10/20 execution
python -m experiments.run_gate_level_large_n  # Gate-level Qiskit execution N=10/20
python -m experiments.run_gate_level_n30       # 30-register gate-level MPS step
python -m experiments.run_coherent_k           # End-to-end coherent-K execution
python -m experiments.run_shot_execution        # 10^4-shot end-to-end readout
python -m experiments.run_qbra_benchmark_t4     # T=4, L=36 QBRA-to-QER path
python -m experiments.run_paper_closure         # Legacy Table 3/10 closure reference
python -m experiments.run_structured_resources     # Generated structured resources
python -m experiments.export_generated_tables      # Export Table 10 JSON
python -m experiments.run_generated_tables      # Generated reproducible tables
python -m experiments.export_manuscript_bundle  # LaTeX backfill bundle
python -m experiments.run_backfill_report       # Table 2/5 backfill
python -m experiments.run_blockencoded_noise_large_n # Full block-encoded noise N=10/20
python -m experiments.run_blockencoding_resources # Resource table
python -m experiments.run_blockencoded_noise   # Block-encoded noise
python -m experiments.run_classical_baseline   # Batched classical baseline
python -m experiments.run_conflict_mitigation  # High-conflict discounting probe
python -m experiments.run_conflict_scan        # Circuit-executed success vs conflict
python -m experiments.run_spectral_analysis    # Singular values / dilation spectrum
python -m experiments.run_crossover            # Empirical fit / classical crossover
python -m experiments.run_focal_truncation     # Focal-set K_max=3 truncation probe
python experiments/measure_depth.py            # Actual transpiled circuit depths
python experiments/run_hardware.py --dry-run   # Hardware feasibility check
```

---

## Correspondence with Paper Sections

| Paper Section | Code Module | Description |
|---------------|-------------|-------------|
| Section 5.1 (Q-BRB) | `core/qbrb.py` | Quantum BRB state preparation |
| Section 5.2 (Input) | `core/rimer.py` `input_transform` | Input transformation |
| Section 5.3 (QBRA) | `core/qbra.py` | Quantum belief rule activation |
| Section 5.4 (QER) | `core/qer.py` | Quantum evidential reasoning engine |
| Section 5.4 (block encoding) | `core/block_encoding.py` | Explicit circuit-executed ER kernel |
| Section 5.5 (Pipeline) | `core/pipeline.py` | Hierarchical quantum pipeline |
| Section 6.1 (Baselines) | `core/bfqc.py`, `core/cri.py` | BF-QC conjunctive + CRI baselines |
| Section 6.1 (Metrics) | `analysis/metrics.py` | ε_β, Kendall τ, ignorance fidelity |
| Section 6.1 (Sampling) | `analysis/sampling.py` | 10⁴-shot measurement statistics |
| Section 6.2 (RQ1) | `experiments/run_shot_execution.py` | Correctness with shot readout |
| Section 6.3 (RQ3) | `experiments/run_ablation.py` | Ablation studies |
| Section 6.4 (RQ2) | `experiments/run_structured_resources.py` | Structured resource scaling |
| Section 6.5 (RQ4) | `experiments/run_conflict_scan.py`, `experiments/run_blockencoded_noise_large_n.py` | Conflict and noise |
| Section 6.5 (spectral) | `core/er_matrices.py` | Singular values / dilation spectrum |
| Section 6.5 (conflict) | `core/mitigation.py` | Discounting and classical delegation |
| Section 6.5 (cost) | `analysis/cost_model.py` | QRAM / shots / EC cost accounting |
| Section 6.6 (Extended) | `experiments/run_extended.py` | Large N & maritime case study |
| Circuit execution | `experiments/run_quantum_execution.py` | Block-encoded execution and shot scaling |
| Hierarchical pipeline | `experiments/run_hierarchical_quantum.py` | Two/three-layer block-encoded pipeline |
| 36-rule benchmark | `experiments/run_benchmark_quantum.py` | Hierarchical graphite circuit execution |
| Classical baseline | `core/classical_batch.py` | Batched vectorised throughput baseline |
| GPU baseline | `core/classical_torch.py` | Optional PyTorch CPU/GPU baseline |
| High-conflict probe | `experiments/run_conflict_mitigation.py` | Discounting reduces conflict |
| Conflict scan | `experiments/run_conflict_scan.py` | Circuit success probability vs conflict |
| Spectral analysis | `experiments/run_spectral_analysis.py` | cond(B), dilation spectrum |
| Scaling / crossover | `experiments/run_crossover.py` | Power law vs O(LN), truncated points |
| Focal-set truncation | `experiments/run_focal_truncation.py` | Cardinality K_max=3 probe |
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

3. **Moderate amplification** — The conditional per-step reduction
   is effective when Γ_acc < 50. High-conflict cases increase Γ_acc
   and degrade the advantage; the sensitivity analysis quantifies
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

Table 4 in the paper reports structured QER resources, including one-step
transpiled depth and qubit count. Two depth metrics are
available in this codebase:

- **`estimate_depth()`** — Returns the asymptotic scaling coefficient
  O(L × polylog(1/ε)). This captures how depth *grows* with problem size
  but does not include constant factors from gate decomposition.
  output: O(L log N kappa_BE^2 / epsilon * Gamma_acc); no empirical power-law claim.
- **`experiments/measure_depth.py`** — Builds actual Qiskit circuits for
  small problem sizes (N ≤ 5) and reports transpiled depth after
  optimization level 1 decomposition into {CX, U3} basis gates.

The resource script transpiles N=3-12, 15 and 20 explicitly.  The N=20
dense block encoding gives depth ~1.49e6 and ~2.06e6 basis gates on 10
qubits; this quantifies the dense-construction bottleneck rather than
hiding it.  At execution time, `HermitianERBlockEncoding` uses exact Qiskit
circuit simulation for <=8 qubits and a direct statevector application for
larger instances implementing the same unitary; this avoids dense-gate
transpilation in the execution path while the resource table remains the
gate-level report.  After this optimization, N=10/L=100 runs in ~0.8 s and
N=20/L=200 in ~7.0 s (previously ~7.4 s and ~69 s).

---

## Circuit-Executed ER Kernel (Validation)

`experiments/run_quantum_execution.py` executes the ER bilinear map through
the paper's Hermitian-dilation block encoding on the Qiskit statevector
simulator.  For each rule it prepares the tensor-product mass state in the
bottom block of $\tilde{M}_{\mathrm{ER}}$, applies the single-ancilla
block-encoding unitary, post-selects the top block, and applies the conflict
scalar and state norm between steps.  The exact circuit output matches the
classical ER reference to ~1e-11 for the tested N=3, L=36 and N=4, L=20
cases; the executed state register uses 6 and 7 qubits, respectively.

The script also reports:

- per-step post-selection success probabilities and the coherent product;
- one round of amplitude amplification (success ~0.97 for the tested N=3/4 steps);
- shot-limited error and the number of shots needed for a target accuracy;
- end-to-end cost accounting with QRAM, shot, and error-correction factors;
- singular values and condition numbers of the fixed synthesis operator and
  its Hermitian dilation.

`experiments/run_hierarchical_quantum.py` extends the same block-encoded
kernel to the two-layer and three-layer synthetic pipelines.  Intermediate
belief vectors are passed directly between layers, and the exact circuit
result matches the classical hierarchical reference to ~1e-14.

`experiments/run_large_n_quantum.py` executes the full block-encoded
pipeline for N=10, L=100 and N=20, L=200.  The exact circuit results match
the classical ER reference to ~7e-14 and ~1.1e-13, respectively; the state
registers use 9 and 10 qubits.  One round of amplitude amplification raises
the effective coherent success product to ~4.1e-2 and ~1.6e-3 (the raw
no-amplification products are ~7e-70 and ~8e-140).  The fast exact-statevector
path executes N=10/L=100 in ~0.8 s and N=20/L=200 in ~7.0 s.

Large-N noise is measured on the **full block-encoded density matrix**, not
on a target-state proxy: `experiments/run_blockencoded_noise_large_n.py`
applies local single-qubit depolarizing noise after every N=10/N=20
Hermitian-dilation step.  At p=0 the full pipeline matches classical ER to
~1e-13.  At p=0.01 the distribution errors are ~0.10 (N=10) and ~0.04
(N=20); the per-step coherent success product is extremely small, as
expected from the accumulated post-selection factor.  These are
operator-level noise results and are not the paper's Falcon gate-noise
Table 6.

`experiments/run_coherent_k.py` runs the end-to-end coherent-K path.  K is
applied by an ancilla rotation (`coherent_step`) rather than by a classical
amplitude rescaling; no intermediate belief vector is measured.  The N=10
and N=20 outputs match classical ER to ~7e-14 and ~1.1e-13, with K
amplitude products ~6e-18 and ~1.4e-33 and one extra ancilla qubit.

### Generated tables (route B)

The reproducible tables are generated, not hand-entered:

- python -m experiments.run_structured_resources writes
  outputs/tables/table3_structured.json from an explicit register-level
  block-encoding step transpiled to {u, cx};
- python -m experiments.run_blockencoded_noise_large_n writes the
  full-L density-matrix noise log;
- python -m experiments.export_generated_tables converts both logs into
  outputs/tables/table10_generated.json;
- python -m experiments.run_generated_tables prints the generated tables.

Current generated Table 4 (one-step transpiled depth) is 3637 (N=3),
14711 (N=5), 237041 (N=10), 949233 (N=15), and 949233 (N=20).  The N=15
entry uses a row-phase gauge fallback because the installed Qiskit QSD
backend panics on the original gauge; the fallback is marked in the JSON.
These values differ from the legacy paper-reference constants and are the
ones that should be used if the tables are regenerated under route B.

### Paper-table reconciliation

The final manuscript is the source of truth for the reported tables.  This
repository separates the paper reference values from local validation
measurements:

- **Table 4 (depth/qubits)**: the current generated values are
  stored in `outputs/tables/table3_structured.json` and used by
  `run_structured_resources.py`. The legacy 214/687/1243/2156/4621
  values are retained only in `analysis/paper_reference.py` for
  historical comparison.
- **Table 6 (noise)**: current values are generated by
  `run_blockencoded_noise_large_n.py` using the block-encoded density-matrix
  model at N=3, 10, and 20. `run_noise_scaling.py` is a legacy target-state
  probe and is not used for the current noise claims.

This is a **post-selected hybrid validation**, not the measurement-free
fault-tolerant pipeline claimed in the manuscript.  The dense Hermitian
block encoding validates the circuit-level ER kernel for large N; a
six-term row-LCU construction in `core/sparse_block_encoding.py` provides
the sparse block-encoding counterpart (N=2/3/5 tested, 9/10 qubits,
step error ~1e-15, success probability ~0.03-0.06).  Neither construction is
a fault-tolerant implementation.

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
│   ├── er_matrices.py         # Fixed ER synthesis matrix + spectral helpers
│   ├── block_encoding.py      # Explicit circuit-executed ER block encoding
│   ├── classical_batch.py     # Batched vectorised classical ER baseline
│   ├── classical_torch.py     # Optional CPU/GPU PyTorch baseline
│   ├── focal_sets.py          # Focal-set BBA and cardinality truncation
│   ├── mitigation.py          # Discounting / classical-delegation helpers
│   ├── pipeline.py            # Hierarchical pipeline — Section 5.5
│   ├── bfqc.py                # BF-QC conjunctive baseline — Section 6.1
│   ├── cri.py                 # CRI fuzzy inference baseline — Section 6.1
│   ├── backends.py            # Backend abstraction (simulator/hardware)
│   └── noise.py               # IBM Falcon noise model
├── experiments/
│   ├── datasets.py            # Benchmark & synthetic data (Section 6.1)
│   ├── run_shot_execution.py  # RQ1 — Correctness with shot readout (Section 6.2)
│   ├── run_ablation.py        # RQ3 — Component ablation (Section 6.3)
│   ├── run_structured_resources.py # RQ2 — Structured scaling (Section 6.4)
│   ├── run_conflict_scan.py   # RQ4 — Conflict scan (Section 6.5)
│   ├── run_blockencoded_noise_large_n.py # RQ4 — Noise at N=3/10/20 (Section 6.5)
│   ├── run_extended.py        # Extended studies (Section 6.6)
│   ├── run_preliminary.py     # Legacy target-state readout emulator
│   ├── run_comparative.py     # Legacy comparative analysis
│   ├── run_sensitivity.py     # Legacy sensitivity model
│   ├── run_quantum_execution.py # Block-encoded circuit execution across rules
│   ├── run_hierarchical_quantum.py # Block-encoded two/three-layer pipeline
│   ├── run_benchmark_quantum.py # 36-rule hierarchical graphite benchmark
│   ├── run_large_n_quantum.py # Full-statevector N=10/20 execution
│   ├── run_blockencoding_resources.py # Transpiled resource table
│   ├── run_blockencoded_noise.py # Block-encoded noise run
│   ├── run_noise_scaling.py   # Legacy target-state noise probe
│   ├── run_classical_baseline.py # Batched vectorised classical baseline
│   ├── run_conflict_mitigation.py # High-conflict discounting probe
│   ├── run_conflict_scan.py   # Circuit-executed success vs conflict
│   ├── run_spectral_analysis.py # Singular values / dilation spectrum
│   ├── run_crossover.py       # Empirical fit / classical crossover
│   ├── run_focal_truncation.py # Focal-set K_max=3 truncation probe
│   ├── run_hardware.py        # Hardware deployment workflow
│   └── measure_depth.py       # Transpiled circuit depth measurement
├── analysis/
│   ├── metrics.py             # epsilon_beta, tau, ignorance fidelity
│   ├── sampling.py            # 10^4-shot measurement sampling models
│   ├── cost_model.py          # QRAM / shots / error-correction cost accounting
│   ├── complexity.py          # Power-law fits and classical crossover
│   ├── postselection.py       # Success products, error bounds, recursive cost
│   └── statistics.py          # t-test, CI, R^2
├── tests/
│   ├── test_rimer.py          # ClassicalRIMER unit tests
│   ├── test_qer.py            # QER unit tests
│   ├── test_qbrb.py           # QBRB unit tests
│   ├── test_qbra.py           # QBRA unit tests
│   ├── test_bfqc.py           # BF-QC baseline tests
│   ├── test_cri.py            # CRI baseline tests
│   ├── test_metrics.py        # Metrics & statistics tests
│   ├── test_block_encoding.py # Explicit block-encoding and circuit tests
│   ├── test_classical_batch.py # Batched classical baseline tests
│   ├── test_classical_torch.py # Optional PyTorch baseline test
│   ├── test_cost_mitigation.py # Cost accounting and conflict mitigation
│   ├── test_complexity.py     # Crossover and power-law tests
│   ├── test_focal_sets.py     # Focal-set truncation tests
│   ├── test_pipeline_quantum.py # Block-encoded hierarchical pipeline test
│   ├── test_postselection.py  # Post-selection and error-budget tests
│   ├── test_api.py            # Public API contract tests
│   └── bench.py               # Micro-benchmarks (not collected by pytest)
├── visualization/
│   ├── plot_backfill_figures.py  # Correctness and ablation figures
│   ├── plot_depth_generated.py   # Generated depth figure
│   ├── plot_kappa_generated.py   # Generated post-selection figure
│   └── legacy/                   # Archived pre-backfill scripts/data
└── outputs/
    ├── figures/               # Generated PDF figures
    ├── tables/
    └── logs/                  # Experiment output (.npz)
```

---

## Known Limitations

- **Benchmark rule table**: The complete 36-rule parameter table for the
  graphite identification benchmark is published in Yang et al. 2006.
  For current correctness validation (RQ1), `run_shot_execution.py` uses the
  published Table III values as the reference and executes the QER path with
  shot readout. `run_preliminary.py` is retained as a legacy target-state
  measurement emulator. The ER-matrix ablation (Section 6.3) runs on
  benchmark-like flat 36-rule bases: the qualitative conclusions (sharp
  error increase, systematic ignorance underestimation) are reproduced.

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
  [[0, B], [B^H, 0]] has dimension (N+2) + (N+2)² = (N+2)(N+3); the
  (N+2)² tensor-product register is the input, and the additional N+2
  dimensions carry the output block.  The dilation register therefore uses
  ceil(log2((N+2)(N+3))) qubits — 5 for the benchmark N=3, matching the
  paper's 27-qubit budget.  Full execution on quantum hardware requires a
  fault-tolerant device with T1 ≥ 100 µs and two-qubit gate error ≤ 10^-2.
  The circuit-executed validation in `core/block_encoding.py` uses the fixed
  unscaled B and applies K as a scalar after post-selection, matching the
  paper's scalar-rescaling step; the legacy Hamiltonian skeleton above keeps
  K inside the matrix for depth analysis.

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

- **Noise robustness**: The current Table 6 results are generated by
  `run_blockencoded_noise_large_n.py`, which executes the block-encoded
  density-matrix model at N=3, 10, and 20. `run_sensitivity.py` and
  `run_noise_scaling.py` are legacy target-state probes and are not used for
  the current noise claims.

- **Circuit-executed validation scope**: `core/block_encoding.py` provides an
  explicit dense block encoding for small N.  It is used to verify the
  bilinear ER kernel and post-selection; it is not a sparse or fault-tolerant
  implementation.  The multi-rule execution is post-selected per step and
  re-prepared.  One round of amplitude amplification raises the effective
  per-step success to ~0.97 (coherent product ~0.35 for N=3, L=36), while the
  raw no-amplification product remains ~5e-25 and is reported separately.
  With 10^4 shots per step the shot-limited error for L=36 is ~0.022;
  10^6 shots per step reduces it to ~0.0019.  The required shot budget and
  the gap to the coherent pipeline are explicit outputs of
  `experiments/run_quantum_execution.py`.

- **Spectral condition number**: the fixed ER synthesis matrix for the tested
  N has condition number ~2.24, independent of the conflict scalar K.  The
  paper's $\bar{\kappa}$ is a conflict-dependent effective amplification
  factor, not the spectral condition number of the dilation.  The numerical
  spectral data are reported by `core/er_matrices.py` and the execution
  experiment; the manuscript should distinguish these quantities explicitly.

- **Large-N noise**: current Table 6 values are generated by
  `experiments/run_blockencoded_noise_large_n.py`, which executes the
  block-encoded density-matrix model for N=3, 10, and 20.
  `run_noise_scaling.py` is retained as a legacy target-state probe.

- **Large-N truncation**: the N=50/100 extended study uses an ER-specific
  top-K singleton approximation (retain the K largest singleton beliefs and
  leave the removed mass as residual ignorance).  A generic power-set
  focal-set object and cardinality truncation are implemented separately in
  `core/focal_sets.py` and exercised by `experiments/run_focal_truncation.py`;
  the ER engine itself operates on the (N+2) singleton/frame state, so the
  focal-set truncation is not plugged into the ER recursion.  The
  paper-reported N=50/100 targets are listed as references, and the
  truncated status of each large-N point is explicit in
  `experiments/run_crossover.py`.

- **Conflict scan**: `experiments/run_conflict_scan.py` executes the
  block-encoded ER step for controlled mass vectors.  For conflict
  0.010→0.903 the success probability falls from ~0.200 to ~0.0011
  (overhead ~5→906), while the fixed synthesis matrix's spectral condition
  number remains ~2.24.  This circuit-level conflict curve is the appropriate
  replacement for a stored/interpolated kappa curve.

- **Benchmark rule table**: the repository contains the published
  hierarchical Table II rule base from Yang et al. (SRB1-SRB5, 36 rules) in
  `core/rimer.py`, and the piecewise-linear input transformation
  (L=0, M=0.5, H=1) used in the paper.  `experiments/run_benchmark_quantum.py`
  executes all 12 published Table III inputs through the block-encoded
  pipeline; test 3 reproduces the paper distribution
  [0.2686, 0.6772, 0.0543] to numerical precision.  The canonical paper's
  original rule base is hierarchical; the "flat 36-rule" description in the
  manuscript is a simplification of that published hierarchy.

---

## Citation

```bibtex
@article{quantum_rimer_2026,
  title   = {Quantum-Accelerated Belief Rule Base Inference with Conditional Per-Step Logarithmic Complexity},
  author  = {},
  journal = {},
  year    = {2026}
}
```

## License

MIT
