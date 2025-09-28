"""Command-line interface for cfDNA epigenomics mini package."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from . import __version__
from .simulate import simulate_dataset
from .features import prepare_features
from .train_models import train_models
from .eval_models import evaluate_models
from .report_gen import generate_report


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up structured logging with run IDs."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create custom formatter for JSON logging
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "module": record.module,
                "message": record.getMessage(),
                "run_id": getattr(record, 'run_id', None)
            }
            return json.dumps(log_obj)
    
    logger = logging.getLogger("cfdna")
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    
    return logger


def generate_run_id() -> str:
    """Generate unique run ID."""
    return f"run_{int(time.time())}_{np.random.randint(1000, 9999)}"


def save_lineage(config: dict[str, Any], run_id: str, out_dir: Path) -> None:
    """Save run lineage metadata."""
    import subprocess
    import os
    
    lineage = {
        "run_id": run_id,
        "timestamp": time.time(),
        "config": config,
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": str(Path.cwd()),
        },
        "git_info": {}
    }
    
    # Add git information if available
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL, 
            text=True
        ).strip()
        lineage["git_info"]["sha"] = git_sha
        
        git_dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            stderr=subprocess.DEVNULL
        ) != 0
        lineage["git_info"]["dirty"] = git_dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    lineage_path = out_dir / "lineage.json"
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """cfDNA Epigenomics Mini: Early cancer detection from cfDNA epigenomic signals."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["logger"] = setup_logging(verbose)


@main.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), 
              help="Configuration YAML file")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option("--out", type=click.Path(path_type=Path), default="data/", 
              help="Output directory for generated data")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto",
              help="Device to use (auto-detects by default)")
@click.pass_context
def simulate(ctx: click.Context, config: Path | None, seed: int, out: Path, device: str) -> None:
    """Simulate synthetic cfDNA methylation and fragmentomics data."""
    logger = ctx.obj["logger"]
    run_id = generate_run_id()
    
    # Set random seed
    np.random.seed(seed)
    
    # Default config if none provided
    if config is None:
        config_data = {
            "dataset": {
                "random_seed": seed,
                "n_samples": 600,
                "n_controls": 400,
                "n_cancer": 200
            },
            "samples": {
                "age_range": [45, 80],
                "sex_ratio": 0.6,
                "n_batches": 3,
                "centers": ["Site_A", "Site_B", "Site_C"]
            },
            "methylation": {
                "n_cpgs": 20000,
                "n_dmrs": 200,
                "cpgs_per_dmr": 100,
                "alpha_base": 2.0,
                "beta_base": 8.0,
                "dmr_effect_size_mean": 1.5,
                "dmr_effect_size_std": 0.5,
                "batch_effect_std": 0.02,
                "missingness_rate": 0.05
            },
            "fragmentomics": {
                "size_bins": [100, 150, 200, 250, 300, 350, 400],
                "tss_enrichment_bins": 10,
                "size_effect_mean": 0.02,
                "size_effect_std": 0.01,
                "tss_effect_mean": 0.1,
                "tss_effect_std": 0.05,
                "noise_std": 0.01
            }
        }
        
        # Create temporary config file
        config = out / "synthetic_config.yaml"
        config.parent.mkdir(exist_ok=True)
        
        import yaml
        with open(config, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    logger.info(f"Starting simulation with config: {config}", extra={"run_id": run_id})
    
    # Run simulation
    out.mkdir(exist_ok=True)
    stats = simulate_dataset(config, out)
    
    # Save lineage
    save_lineage({"config_path": str(config), "seed": seed}, run_id, out)
    
    # Output results as JSON
    results = {
        "run_id": run_id,
        "status": "success",
        "stats": stats,
        "artifacts_dir": str(out)
    }
    
    click.echo(json.dumps(results, indent=2))
    logger.info("Simulation completed successfully", extra={"run_id": run_id})


@main.command()
@click.option("--data-dir", type=click.Path(exists=True, path_type=Path), default="data/",
              help="Directory containing simulated data")
@click.option("--out", type=click.Path(path_type=Path), default="artifacts/",
              help="Output directory for processed features")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.pass_context
def features(ctx: click.Context, data_dir: Path, out: Path, seed: int) -> None:
    """Extract features from cfDNA data (DMR aggregation, fragmentomics)."""
    logger = ctx.obj["logger"]
    run_id = generate_run_id()
    
    # Set random seed
    np.random.seed(seed)
    
    logger.info(f"Processing features from {data_dir}", extra={"run_id": run_id})
    
    # Prepare features
    out.mkdir(exist_ok=True)
    feature_data = prepare_features(data_dir)
    
    # Save processed features
    feature_data["X"].to_parquet(out / "X_features.parquet")
    feature_data["y"].to_csv(out / "y.csv", index=False)
    feature_data["metadata"].to_csv(out / "metadata.csv")
    
    # Save splits
    splits_file = out / "splits.json"
    splits = feature_data["splits"]
    splits_serializable = {k: v.tolist() for k, v in splits.items()}
    with open(splits_file, "w") as f:
        json.dump(splits_serializable, f, indent=2)
    
    # Save feature names
    feature_names_file = out / "feature_names.json"
    with open(feature_names_file, "w") as f:
        json.dump(feature_data["feature_names"], f, indent=2)
    
    # Save lineage
    save_lineage({"data_dir": str(data_dir), "seed": seed}, run_id, out)
    
    # Output results
    results = {
        "run_id": run_id,
        "status": "success",
        "n_samples": len(feature_data["X"]),
        "n_features": len(feature_data["X"].columns),
        "feature_types": {k: len(v) for k, v in feature_data["feature_names"].items()},
        "artifacts_dir": str(out)
    }
    
    click.echo(json.dumps(results, indent=2))
    logger.info("Feature extraction completed successfully", extra={"run_id": run_id})


@main.command()
@click.option("--features-dir", type=click.Path(exists=True, path_type=Path), 
              default="artifacts/", help="Directory containing processed features")
@click.option("--out", type=click.Path(path_type=Path), default="artifacts/",
              help="Output directory for trained models")
@click.option("--models", multiple=True, default=["logistic", "mlp"],
              help="Models to train (can specify multiple)")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto",
              help="Device to use for training")
@click.pass_context
def train(ctx: click.Context, features_dir: Path, out: Path, models: tuple[str, ...], 
          seed: int, device: str) -> None:
    """Train machine learning models for cancer detection."""
    logger = ctx.obj["logger"]
    run_id = generate_run_id()
    
    # Set random seed
    np.random.seed(seed)
    
    logger.info(f"Training models: {list(models)}", extra={"run_id": run_id})
    
    # Load features
    X = pd.read_parquet(features_dir / "X_features.parquet")
    y = pd.read_csv(features_dir / "y.csv")["label"]
    
    with open(features_dir / "splits.json") as f:
        splits = json.load(f)
    
    # Convert splits back to arrays
    splits = {k: np.array(v) for k, v in splits.items()}
    
    # Train models
    out.mkdir(exist_ok=True)
    model_results = train_models(X, y, splits, list(models), device, out)
    
    # Save lineage
    save_lineage({
        "features_dir": str(features_dir), 
        "models": list(models), 
        "seed": seed, 
        "device": device
    }, run_id, out)
    
    # Output results
    results = {
        "run_id": run_id,
        "status": "success",
        "models_trained": list(model_results.keys()),
        "artifacts_dir": str(out)
    }
    
    click.echo(json.dumps(results, indent=2))
    logger.info("Model training completed successfully", extra={"run_id": run_id})


@main.command()
@click.option("--models-dir", type=click.Path(exists=True, path_type=Path),
              default="artifacts/", help="Directory containing trained models")
@click.option("--out", type=click.Path(path_type=Path), default="artifacts/",
              help="Output directory for evaluation results")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.pass_context  
def eval(ctx: click.Context, models_dir: Path, out: Path, seed: int) -> None:
    """Evaluate trained models and compute metrics."""
    logger = ctx.obj["logger"]
    run_id = generate_run_id()
    
    # Set random seed for reproducible bootstrap
    np.random.seed(seed)
    
    logger.info(f"Evaluating models from {models_dir}", extra={"run_id": run_id})
    
    # Run evaluation
    out.mkdir(exist_ok=True)
    eval_results = evaluate_models(models_dir, out)
    
    # Save lineage
    save_lineage({"models_dir": str(models_dir), "seed": seed}, run_id, out)
    
    # Output results
    results = {
        "run_id": run_id,
        "status": "success",
        "metrics_computed": eval_results,
        "artifacts_dir": str(out)
    }
    
    click.echo(json.dumps(results, indent=2))
    logger.info("Model evaluation completed successfully", extra={"run_id": run_id})


@main.command()
@click.option("--results-dir", type=click.Path(exists=True, path_type=Path),
              default="artifacts/", help="Directory containing evaluation results")
@click.option("--out", type=click.Path(path_type=Path), default="artifacts/report.html",
              help="Output path for HTML report")
@click.pass_context
def report(ctx: click.Context, results_dir: Path, out: Path) -> None:
    """Generate HTML report from evaluation results."""
    logger = ctx.obj["logger"]
    run_id = generate_run_id()
    
    logger.info(f"Generating report from {results_dir}", extra={"run_id": run_id})
    
    # Generate report
    out.parent.mkdir(exist_ok=True)
    report_path = generate_report(results_dir, out)
    
    # Output results
    results = {
        "run_id": run_id,
        "status": "success",
        "report_path": str(report_path)
    }
    
    click.echo(json.dumps(results, indent=2))
    logger.info(f"Report generated successfully: {report_path}", extra={"run_id": run_id})


@main.command()
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="cpu",
              help="Device to use (CPU recommended for smoke test)")
@click.pass_context
def smoke(ctx: click.Context, seed: int, device: str) -> None:
    """Run end-to-end smoke test (≤5 min on CPU)."""
    logger = ctx.obj["logger"] 
    run_id = generate_run_id()
    
    logger.info("Starting end-to-end smoke test", extra={"run_id": run_id})
    start_time = time.time()
    
    # Set random seed
    np.random.seed(seed)
    
    # Create temporary directory
    smoke_dir = Path("smoke_test")
    smoke_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Simulate small dataset
        logger.info("Step 1: Simulating data", extra={"run_id": run_id})
        config_data = {
            "dataset": {"random_seed": seed, "n_samples": 100, "n_controls": 60, "n_cancer": 40},
            "samples": {"age_range": [45, 80], "sex_ratio": 0.6, "n_batches": 2, "centers": ["Site_A", "Site_B"]},
            "methylation": {"n_cpgs": 1000, "n_dmrs": 10, "cpgs_per_dmr": 50, "alpha_base": 2.0, "beta_base": 8.0, "dmr_effect_size_mean": 1.5, "dmr_effect_size_std": 0.5, "batch_effect_std": 0.02, "missingness_rate": 0.05},
            "fragmentomics": {"size_bins": [100, 200, 300], "tss_enrichment_bins": 3, "size_effect_mean": 0.02, "size_effect_std": 0.01, "tss_effect_mean": 0.1, "tss_effect_std": 0.05, "noise_std": 0.01}
        }
        
        config_path = smoke_dir / "config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        data_dir = smoke_dir / "data"
        simulate_dataset(config_path, data_dir)
        
        # 2. Extract features
        logger.info("Step 2: Extracting features", extra={"run_id": run_id})
        feature_data = prepare_features(data_dir)
        
        features_dir = smoke_dir / "features"
        features_dir.mkdir(exist_ok=True)
        feature_data["X"].to_parquet(features_dir / "X_features.parquet")
        feature_data["y"].to_csv(features_dir / "y.csv", index=False)
        
        splits = {k: v.tolist() for k, v in feature_data["splits"].items()}
        with open(features_dir / "splits.json", "w") as f:
            json.dump(splits, f)
        
        # 3. Train lightweight model
        logger.info("Step 3: Training model", extra={"run_id": run_id})
        X = feature_data["X"]
        y = feature_data["y"]
        splits_arrays = {k: np.array(v) for k, v in splits.items()}
        
        models_dir = smoke_dir / "models"
        train_models(X, y, splits_arrays, ["logistic"], device, models_dir)
        
        # 4. Evaluate
        logger.info("Step 4: Evaluating model", extra={"run_id": run_id})
        eval_dir = smoke_dir / "eval"
        evaluate_models(models_dir, eval_dir)
        
        # 5. Generate report
        logger.info("Step 5: Generating report", extra={"run_id": run_id})
        report_path = smoke_dir / "report.html"
        generate_report(eval_dir, report_path)
        
        elapsed_time = time.time() - start_time
        
        # Output results
        results = {
            "run_id": run_id,
            "status": "success",
            "elapsed_time_seconds": round(elapsed_time, 2),
            "steps_completed": 5,
            "artifacts_dir": str(smoke_dir)
        }
        
        click.echo(json.dumps(results, indent=2))
        logger.info(f"Smoke test completed successfully in {elapsed_time:.1f}s", 
                   extra={"run_id": run_id})
        
        if elapsed_time > 300:  # 5 minutes
            logger.warning(f"Smoke test took {elapsed_time:.1f}s (>5min target)", 
                          extra={"run_id": run_id})
    
    except Exception as e:
        logger.error(f"Smoke test failed: {e}", extra={"run_id": run_id})
        results = {
            "run_id": run_id,
            "status": "failed",
            "error": str(e),
            "elapsed_time_seconds": round(time.time() - start_time, 2)
        }
        click.echo(json.dumps(results, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()