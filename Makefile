.PHONY: install test bench experiments figures clean all

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
	@echo "=== RQ1: Correctness ==="
	python experiments/run_preliminary.py
	@echo ""
	@echo "=== RQ2: Acceleration trend ==="
	python experiments/run_comparative.py
	@echo ""
	@echo "=== RQ3: Ablation ==="
	python experiments/run_ablation.py
	@echo ""
	@echo "=== RQ4: Sensitivity ==="
	python experiments/run_sensitivity.py
	@echo ""
	@echo "=== Extended studies ==="
	python experiments/run_extended.py

# Generate all paper figures
figures:
	python visualization/plot_correctness.py
	python visualization/plot_ablation.py
	python visualization/plot_depth.py
	python visualization/plot_kappa.py

# Full reproducibility: experiments + figures
reproduce: experiments figures
	@echo ""
	@echo "All experiments and figures reproduced successfully."

# Lint with ruff
lint:
	ruff check .

# Clean generated files (Linux/macOS; on Windows use: python -c "import shutil; ...")
clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('__pycache__'), pathlib.Path('.pytest_cache'), *pathlib.Path('.').glob('*.egg-info'), *pathlib.Path('.').rglob('__pycache__')]]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('outputs/figures').glob('*.pdf')]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('outputs/logs').glob('*.npz')]"
