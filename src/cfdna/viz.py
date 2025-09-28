"""Visualization utilities for cfDNA model evaluation."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# Set consistent style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, 
                   model_name: str = "Model", ax: plt.Axes | None = None) -> plt.Axes:
    """Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores
        model_name: Model name for legend
        ax: Matplotlib axes (optional)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot curve
    ax.plot(fpr, tpr, linewidth=2, 
            label=f"{model_name} (AUC = {roc_auc:.3f})")
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Formatting
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray,
                               model_name: str = "Model", ax: plt.Axes | None = None) -> plt.Axes:
    """Plot Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores
        model_name: Model name for legend
        ax: Matplotlib axes (optional)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Calculate PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    # Plot curve
    ax.plot(recall, precision, linewidth=2,
            label=f"{model_name} (AUC = {pr_auc:.3f})")
    
    # Plot baseline (prevalence)
    baseline = np.mean(y_true)
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5,
               label=f"Baseline ({baseline:.3f})")
    
    # Formatting
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                          n_bins: int = 10, model_name: str = "Model",
                          ax: plt.Axes | None = None) -> plt.Axes:
    """Plot calibration curve (reliability diagram).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        model_name: Model name for title
        ax: Matplotlib axes (optional)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Calculate calibration data
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            bin_centers.append(avg_confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(in_bin.sum())
    
    # Plot calibration curve
    if bin_centers:
        ax.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8,
                label=f"{model_name}")
        
        # Add histogram of predictions
        ax2 = ax.twinx()
        ax2.hist(y_prob, bins=n_bins, alpha=0.3, color='gray', density=True)
        ax2.set_ylabel("Density", alpha=0.7)
    
    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Perfect calibration")
    
    # Formatting
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve - {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_decision_curve(decision_data: pd.DataFrame, model_name: str = "Model",
                       ax: plt.Axes | None = None) -> plt.Axes:
    """Plot decision curve analysis.
    
    Args:
        decision_data: DataFrame with decision curve data
        model_name: Model name for legend
        ax: Matplotlib axes (optional)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot decision curves
    ax.plot(decision_data["threshold"], decision_data["net_benefit_model"],
            linewidth=2, label=f"{model_name}")
    ax.plot(decision_data["threshold"], decision_data["net_benefit_all"],
            linewidth=2, linestyle='--', label="Treat All")
    ax.plot(decision_data["threshold"], decision_data["net_benefit_none"],
            linewidth=2, linestyle=':', label="Treat None")
    
    # Formatting
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    return ax


def plot_confusion_matrix_at_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                                      thresholds: list[float] = [0.5],
                                      model_name: str = "Model") -> plt.Figure:
    """Plot confusion matrices at different thresholds.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        thresholds: List of probability thresholds
        model_name: Model name for title
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(1, n_thresholds, figsize=(4 * n_thresholds, 4))
    
    if n_thresholds == 1:
        axes = [axes]
    
    for idx, threshold in enumerate(thresholds):
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=idx == 0)
        axes[idx].set_title(f"Threshold = {threshold:.2f}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    
    fig.suptitle(f"Confusion Matrices - {model_name}")
    plt.tight_layout()
    
    return fig


def plot_feature_importance(importance_scores: dict[str, float],
                          top_n: int = 20, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot feature importance scores.
    
    Args:
        importance_scores: Dictionary of feature -> importance score
        top_n: Number of top features to show
        ax: Matplotlib axes (optional)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sort features by importance
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, scores = zip(*top_features)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Top feature at top
    
    # Formatting
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax


def create_summary_plots(eval_results: dict[str, Any], output_dir: Path,
                        prefix: str = "") -> None:
    """Create summary evaluation plots.
    
    Args:
        eval_results: Evaluation results dictionary
        output_dir: Output directory for plots
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data from results
    y_true = None
    y_prob = None
    model_name = eval_results.get("model_name", "Model")
    
    # Try to find prediction data in results
    if "test" in eval_results and "decision_curve" in eval_results["test"]:
        decision_data = eval_results["test"]["decision_curve"]
    else:
        decision_data = None
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Model Evaluation Summary - {model_name}", fontsize=16)
    
    # Note: For this summary, we'll create placeholder plots
    # In actual usage, prediction data would be passed separately
    
    # ROC Curve (placeholder)
    axes[0, 0].text(0.5, 0.5, "ROC Curve\n(Load prediction data)", 
                   ha='center', va='center', transform=axes[0, 0].transAxes)
    axes[0, 0].set_title("ROC Curve")
    
    # PR Curve (placeholder)  
    axes[0, 1].text(0.5, 0.5, "PR Curve\n(Load prediction data)",
                   ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title("Precision-Recall Curve")
    
    # Calibration Curve (placeholder)
    axes[1, 0].text(0.5, 0.5, "Calibration\n(Load prediction data)",
                   ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title("Calibration Curve")
    
    # Decision Curve
    if decision_data is not None:
        plot_decision_curve(decision_data, model_name, axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, "Decision Curve\n(No data)",
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Decision Curve")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}summary_plots.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_evaluation_plots(results: dict[str, dict[str, Any]], output_dir: Path,
                         prefix: str = "") -> None:
    """Save all evaluation plots to files.
    
    Args:
        results: Dictionary of split -> evaluation results
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    
    # Create summary plots for each split
    for split_name, split_results in results.items():
        create_summary_plots(split_results, output_dir, f"{prefix}{split_name}_")
    
    # Create metrics comparison plot
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        splits = list(results.keys())
        metrics = ["auroc", "auprc"]
        
        x_pos = np.arange(len(splits))
        width = 0.35
        
        for i, metric in enumerate(metrics):
            values = [results[split][metric]["mean"] for split in splits]
            errors = [(results[split][metric]["mean"] - results[split][metric]["ci_lower"],
                      results[split][metric]["ci_upper"] - results[split][metric]["mean"])
                     for split in splits]
            errors = np.array(errors).T
            
            ax.bar(x_pos + i * width, values, width, yerr=errors,
                  label=metric.upper(), capsize=5)
        
        ax.set_xlabel("Data Split")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Across Splits")
        ax.set_xticks(x_pos + width / 2)
        ax.set_xticklabels(splits)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}splits_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()