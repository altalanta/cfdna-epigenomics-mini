.PHONY: setup simulate train eval lint test fmt clean help

help:
	@echo "Available targets:"
	@echo "  setup     - Install package and development dependencies"
	@echo "  simulate  - Generate synthetic cfDNA dataset"
	@echo "  train     - Train baseline and MLP models"
	@echo "  eval      - Evaluate models and generate report"
	@echo "  lint      - Run ruff linter"
	@echo "  test      - Run pytest test suite"
	@echo "  fmt       - Format code with ruff"
	@echo "  clean     - Remove generated files and caches"

setup:
	pip install -e .[dev]

simulate:
	cfdna-sim data/synthetic_config.yaml data/

train:
	cfdna-train data/synthetic_config.yaml logistic artifacts/ --seed 42
	cfdna-train data/synthetic_config.yaml mlp artifacts/ --seed 42

eval:
	cfdna-eval artifacts/ artifacts/report.md

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