.PHONY: setup simulate train eval lint test fmt clean help smoke

help:
	@echo "Available targets:"
	@echo "  setup     - Install package and development dependencies"
	@echo "  simulate  - Generate synthetic cfDNA dataset"
	@echo "  train     - Train baseline and MLP models"
	@echo "  eval      - Evaluate models and generate report"
	@echo "  smoke     - Run end-to-end smoke test (≤5 min)"
	@echo "  lint      - Run ruff linter"
	@echo "  test      - Run pytest test suite"
	@echo "  fmt       - Format code with ruff"
	@echo "  clean     - Remove generated files and caches"

setup:
	pip install -e .[dev]
	pre-commit install

simulate:
	cfdna simulate --out data/ --seed 42

train:
	cfdna train --features-dir data/ --out artifacts/ --models logistic --models mlp --seed 42

eval:
	cfdna eval --models-dir artifacts/ --out artifacts/ --seed 42 && cfdna report --results-dir artifacts/ --out artifacts/report.html

smoke:
	cfdna smoke --seed 42

lint:
	ruff check src/ tests/
	mypy src/

test:
	pytest tests/ -v

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf artifacts/ data/*.parquet data/*.csv cfdna.db