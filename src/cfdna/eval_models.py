"""Model evaluation functions for CLI integration."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_func, n_bootstrap: int = 1000) -> dict[str, float]:
    """Compute bootstrap confidence intervals for a metric."""
    n_samples = len(y_true)
    bootstrap_scores = []
    
    np.random.seed(42)  # For reproducible CIs
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Skip if no positive samples
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    return {
        "mean": np.mean(bootstrap_scores),
        "ci_lower": np.percentile(bootstrap_scores, 2.5),
        "ci_upper": np.percentile(bootstrap_scores, 97.5),
        "std": np.std(bootstrap_scores)
    }


def evaluate_single_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Evaluate a single model's predictions."""
    results = {}
    
    # AUROC
    results["auroc"] = bootstrap_metric(y_true, y_pred, roc_auc_score)
    
    # AUPRC
    results["auprc"] = bootstrap_metric(y_true, y_pred, average_precision_score)
    
    # Brier score
    results["brier"] = bootstrap_metric(y_true, y_pred, brier_score_loss)
    
    # Sensitivity at specific specificities
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    specificity = 1 - fpr
    
    # Find sensitivity at 90% and 95% specificity
    spec_90_idx = np.argmin(np.abs(specificity - 0.9))
    spec_95_idx = np.argmin(np.abs(specificity - 0.95))
    
    results["sens_at_spec90"] = tpr[spec_90_idx]
    results["sens_at_spec95"] = tpr[spec_95_idx]
    
    # Calibration
    fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10)
    ece = np.mean(np.abs(fraction_pos - mean_pred))
    
    results["calibration"] = {
        "brier_score": brier_score_loss(y_true, y_pred),
        "ece": ece
    }
    
    return results


def load_model_predictions(models_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Load prediction files from models directory."""
    predictions = {}
    
    # Find all prediction files
    pred_files = list(models_dir.glob("*_test_predictions.csv"))
    
    for pred_file in pred_files:
        # Extract model name
        model_name = pred_file.name.replace("_test_predictions.csv", "")
        
        predictions[model_name] = {}
        
        # Load all splits for this model
        for split in ["train", "val", "test"]:
            split_file = models_dir / f"{model_name}_{split}_predictions.csv"
            if split_file.exists():
                predictions[model_name][split] = pd.read_csv(split_file)
    
    return predictions


def evaluate_models(models_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Evaluate all trained models.
    
    Args:
        models_dir: Directory containing trained models and predictions
        out_dir: Output directory for evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    
    # Load predictions
    predictions = load_model_predictions(models_dir)
    
    if not predictions:
        raise ValueError(f"No prediction files found in {models_dir}")
    
    results = {}
    
    # Evaluate each model
    for model_name, model_preds in predictions.items():
        print(f"Evaluating {model_name}...")
        
        model_results = {}
        
        for split_name, pred_df in model_preds.items():
            y_true = pred_df["true_label"].values
            y_pred = pred_df["predicted_prob"].values
            
            split_results = evaluate_single_model(y_true, y_pred)
            model_results[split_name] = split_results
        
        results[model_name] = model_results
    
    # Save results
    results_path = out_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary table
    summary_data = []
    for model_name, model_results in results.items():
        if "test" in model_results:
            test_results = model_results["test"]
            summary_data.append({
                "model": model_name,
                "test_auroc": test_results["auroc"]["mean"],
                "test_auroc_ci_lower": test_results["auroc"]["ci_lower"],
                "test_auroc_ci_upper": test_results["auroc"]["ci_upper"],
                "test_auprc": test_results["auprc"]["mean"],
                "test_brier": test_results["brier"]["mean"],
                "sens_at_spec90": test_results["sens_at_spec90"],
                "sens_at_spec95": test_results["sens_at_spec95"]
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = out_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Save individual model predictions with metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "summary": summary_data,
            "detailed_results": results
        }, f, indent=2, default=str)
    
    return {
        "models_evaluated": list(results.keys()),
        "summary_table": summary_path,
        "detailed_results": results_path,
        "metrics_file": metrics_path
    }