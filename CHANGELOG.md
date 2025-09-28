# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-09-27

### Added

#### Core Infrastructure
- **Production-ready packaging** with PEP 621 compliant `pyproject.toml`
- **Unified CLI interface** (`cfdna`) with subcommands: `simulate`, `features`, `train`, `eval`, `report`, `smoke`
- **Comprehensive data validation** using Pydantic schemas for type safety and error handling
- **Setuptools-scm versioning** for automatic version management from git tags
- **Docker containerization** with optimized multi-stage builds for production deployment

#### Data Simulation Engine
- **Synthetic cfDNA dataset generation** with realistic methylation and fragmentomics patterns
- **DMR-based methylation simulation** with configurable effect sizes and batch effects
- **Fragmentomics simulation** including size-bin distributions and TSS enrichment patterns
- **Batch effect modeling** to simulate real-world technical variation
- **Configurable missing data** patterns (MAR by batch)
- **Deterministic reproducibility** with global seed control

#### Feature Engineering Pipeline
- **DMR aggregation** from individual CpG sites with mean, z-score, and variability measures
- **ComBat-style batch correction** for technical variation removal
- **Fragmentomics feature extraction** including size-bin frequencies and TSS enrichment
- **Stratified train/validation/test splits** preserving batch and class balance
- **Missing value imputation** with median-based strategies

#### Machine Learning Models
- **Sklearn baseline models**: Logistic regression (L1/L2), Random Forest
- **PyTorch MLP implementation** with batch normalization, dropout, and early stopping
- **Automatic class weighting** for imbalanced datasets
- **Model serialization** with pickle (sklearn) and PyTorch state dicts
- **CPU/CUDA device auto-detection** for flexible training environments

#### Evaluation & Metrics
- **Bootstrap confidence intervals** for AUROC, AUPRC, and Brier score
- **Clinical utility metrics**: Sensitivity at 90%/95% specificity
- **Calibration assessment** with Expected Calibration Error (ECE)
- **Statistical model comparison** with DeLong test implementation
- **Comprehensive evaluation reports** with performance summaries

#### Reporting System
- **HTML report generation** with professional styling and interactive elements
- **Clinical interpretation sections** with key findings and performance indicators
- **Limitations documentation** emphasizing synthetic data constraints
- **Artifact management** with structured output directories and metadata tracking

#### Quality Assurance
- **Comprehensive test suite** with 75+ tests covering simulation, features, models, and CLI
- **90%+ code coverage** with pytest and coverage reporting
- **Type checking** with mypy for static analysis
- **Code formatting** with ruff (replacing black/isort/flake8)
- **Pre-commit hooks** for automated quality checks
- **GitHub Actions CI/CD** with multi-Python version testing (3.10-3.12)

#### CLI & Developer Experience  
- **Structured JSON logging** with run IDs and lineage tracking
- **Progress tracking** with tqdm progress bars
- **Smoke test** for rapid end-to-end validation (≤5 min on CPU)
- **Makefile integration** mapping to CLI commands for workflow compatibility
- **Docker support** with health checks and optimized layer caching

#### Documentation Foundation
- **Comprehensive README** with quick start guide and architecture overview
- **API documentation** with detailed docstrings and type hints
- **Clinical context** explaining synthetic data limitations and translation requirements
- **Reproducibility guidelines** with seed management and deterministic execution

### Technical Details

#### Dependencies
- **Core**: numpy ≥1.21, pandas ≥1.5, scikit-learn ≥1.1, pydantic ≥2.0
- **CLI**: click ≥8.0 for command-line interface
- **ML**: PyTorch (CPU-only by default) for neural networks
- **Visualization**: matplotlib ≥3.5, optional plotly ≥5.10 for interactive plots
- **Development**: pytest ≥7.0, ruff ≥0.1, mypy ≥1.0, pre-commit ≥3.0

#### Performance Characteristics
- **Smoke test runtime**: ~2 seconds on laptop CPU
- **Memory footprint**: <500MB for default synthetic dataset (n=600)
- **Scalability**: Tested up to 10K samples in development
- **CPU-first design**: No GPU required for core functionality

#### Architecture Highlights
- **Modular design** with clear separation of concerns
- **Schema-first approach** with Pydantic validation throughout
- **Fail-fast error handling** with actionable error messages
- **Artifact lineage tracking** with git SHA and environment metadata
- **Configuration-driven** with YAML-based parameter management

### Security & Compliance
- **No network calls** at runtime for air-gapped environments
- **Synthetic data only** - no PHI or sensitive information
- **Container security** with non-root user and minimal attack surface
- **Supply chain security** with pinned dependencies and vulnerability scanning

### Known Limitations
- **Synthetic data scope**: Simplified biological assumptions for demonstration
- **Scale limitations**: Designed for prototype/demo rather than clinical scale
- **Feature coverage**: Basic methylation DMRs and fragmentomics only
- **Population diversity**: No demographic or clinical heterogeneity modeling

### Migration Notes
This is the initial release. Future versions will maintain backward compatibility for:
- CLI interface and command structure
- Configuration file formats (YAML)
- Output file schemas (Parquet, CSV, JSON)
- Docker image API

[0.1.0]: https://github.com/example/cfdna-epigenomics-mini/releases/tag/v0.1.0