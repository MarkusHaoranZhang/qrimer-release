.PHONY: install test bench experiments quantum hierarchy benchmark large coherentk shots qbrat4 paperclosure structured exporttables generatedtables manuscript resources blocknoise blocknoiselarge strategies sparseoracle rowlcu sparsebe noise baseline mitigation conflict spectral crossover focal figures clean all

# Default target: run tests
all: test

# Install in development mode
install:
	pip install -e ".[dev]"

# Run unit tests
test:
	python -m pytest tests/ -v --tb=short

# Run tests with coverage
coverage:
	python -m pytest tests/ --cov=core --cov=analysis --cov-report=term-missing

# Run micro-benchmarks
bench:
	python tests/bench.py

# Run all experiments (RQ1-RQ4 + extended)
experiments:
	@echo "=== RQ1: Correctness (shot-based) ==="
	python -m experiments.run_shot_execution
	@echo ""
	@echo "=== RQ2: Structured resources ==="
	python -m experiments.run_structured_resources
	@echo ""
	@echo "=== RQ3: Ablation ==="
	python -m experiments.run_ablation
	@echo ""
	@echo "=== RQ4: Conflict and noise ==="
	python -m experiments.run_conflict_scan
	python -m experiments.run_blockencoded_noise_large_n
	@echo ""
	@echo "=== Extended studies ==="
	python -m experiments.run_extended

# Circuit-executed ER validation (block encoding + post-selection)
quantum:
	python -m experiments.run_quantum_execution

# Block-encoded two/three-layer hierarchical pipeline
hierarchy:
	python -m experiments.run_hierarchical_quantum

# Circuit-executed 36-rule hierarchical graphite benchmark
benchmark:
	python -m experiments.run_benchmark_quantum

# Large-N full-statevector circuit execution (N=10/20)
large:
	python -m experiments.run_large_n_quantum
	python -m experiments.run_gate_level_large_n
	python -m experiments.run_gate_level_n30

# End-to-end coherent-K execution
coherentk:
	python -m experiments.run_coherent_k

# Shot-based end-to-end execution
shots:
	python -m experiments.run_shot_execution

# T=4, L=36 QBRA-to-QER path
qbrat4:
	python -m experiments.run_qbra_benchmark_t4

# Legacy Table 3/10 closure reference
paperclosure:
	python -m experiments.run_paper_closure

# Generated reproducible tables
structured:
	python -m experiments.run_structured_resources

exporttables:
	python -m experiments.export_generated_tables

generatedtables:
	python -m experiments.run_generated_tables

manuscript:
	python -m experiments.export_manuscript_bundle
	python -m experiments.run_backfill_report

# Transpiled block-encoding resource table
resources:
	python -m experiments.run_blockencoding_resources

# Single-step block-encoded noise run
blocknoise:
	python -m experiments.run_blockencoded_noise

# Full block-encoded noise at N=10/20
blocknoiselarge:
	python -m experiments.run_blockencoded_noise_large_n

# Alternative block-encoding strategy comparison
strategies:
	python -m experiments.run_blockencoding_strategies

# Sparse column-oracle resource exploration
sparseoracle:
	python -m experiments.run_sparse_oracle

# Row-side LCU decomposition
rowlcu:
	python -m experiments.run_row_lcu

# Six-term sparse LCU block encoding
sparsebe:
	python -m experiments.run_sparse_block_encoding

# Legacy target-state noise probe
noise:
	python -m experiments.run_noise_scaling

# Batched vectorised classical baseline
baseline:
	python -m experiments.run_classical_baseline

# High-conflict discounting probe
mitigation:
	python -m experiments.run_conflict_mitigation

# Circuit-executed conflict-dependent success scan
conflict:
	python -m experiments.run_conflict_scan

# Singular values and dilation spectrum
spectral:
	python -m experiments.run_spectral_analysis

# Empirical power-law fit and classical crossover
crossover:
	python -m experiments.run_crossover

# Focal-set cardinality truncation probe
focal:
	python -m experiments.run_focal_truncation

# Generate all paper figures
figures:
	python -m experiments.export_figure_data
	python -m visualization.plot_backfill_figures
	python -m visualization.plot_depth_generated
	python -m visualization.plot_kappa_generated
	python -m visualization.plot_composite

# Full reproducibility: experiments + circuit execution + figures
reproduce: experiments qbrat4 quantum hierarchy benchmark large shots coherentk blocknoiselarge noise baseline mitigation conflict spectral crossover focal figures
	@echo ""
	@echo "All experiments, circuit executions, benchmarks, and figures reproduced successfully."

# Lint with ruff
lint:
	ruff check .

# Clean generated files (Linux/macOS; on Windows use: python -c "import shutil; ...")
clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('__pycache__'), pathlib.Path('.pytest_cache'), *pathlib.Path('.').glob('*.egg-info'), *pathlib.Path('.').rglob('__pycache__')]]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('outputs/figures').glob('*.*') if p.suffix in {'.pdf', '.png'}]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('outputs/logs').glob('*.npz')]"
