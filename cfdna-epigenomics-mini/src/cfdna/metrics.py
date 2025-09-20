"""Evaluation metrics for cfDNA cancer detection models."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def bootstrap_metric(y_true: np.ndarray, y_score: np.ndarray, 
                    metric_func: callable, n_bootstrap: int = 1000,
                    confidence_level: float = 0.95, random_state: int = 42) -> dict[str, float]:
    """Calculate bootstrap confidence intervals for a metric.
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores/probabilities
        metric_func: Metric function (e.g., roc_auc_score)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_state: Random seed
        
    Returns:
        Dictionary with 'mean', 'ci_lower', 'ci_upper'
    """
    np.random.seed(random_state)
    
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Skip if no positive samples in bootstrap
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        score = metric_func(y_true[indices], y_score[indices])
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_scores, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return {
        "mean": bootstrap_scores.mean(),
        "std": bootstrap_scores.std(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }


def auroc_with_ci(y_true: np.ndarray, y_score: np.ndarray, **kwargs: Any) -> dict[str, float]:
    """Calculate AUROC with confidence intervals.
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores/probabilities
        **kwargs: Additional arguments for bootstrap_metric
        
    Returns:
        Dictionary with AUROC statistics
    """
    return bootstrap_metric(y_true, y_score, roc_auc_score, **kwargs)


def auprc_with_ci(y_true: np.ndarray, y_score: np.ndarray, **kwargs: Any) -> dict[str, float]:
    """Calculate AUPRC (Average Precision) with confidence intervals.
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores/probabilities
        **kwargs: Additional arguments for bootstrap_metric
        
    Returns:
        Dictionary with AUPRC statistics
    """
    return bootstrap_metric(y_true, y_score, average_precision_score, **kwargs)


def delong_test(y_true: np.ndarray, y_score1: np.ndarray, y_score2: np.ndarray) -> dict[str, float]:
    """Perform DeLong test for comparing two ROC curves.
    
    This is a simplified implementation focused on the test statistic and p-value.
    
    Args:
        y_true: True binary labels
        y_score1: Predicted scores from model 1
        y_score2: Predicted scores from model 2
        
    Returns:
        Dictionary with test statistics and p-value
    """
    # Calculate AUCs
    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)
    
    # Simplified variance estimation
    # This is an approximation - full DeLong test requires more complex covariance calculation
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    # Approximate standard error
    # Based on Hanley & McNeil (1982) approximation
    Q1_1 = auc1 / (2 - auc1)
    Q2_1 = 2 * auc1**2 / (1 + auc1)
    var_auc1 = (auc1 * (1 - auc1) + (n_pos - 1) * (Q1_1 - auc1**2) + 
                (n_neg - 1) * (Q2_1 - auc1**2)) / (n_pos * n_neg)
    
    Q1_2 = auc2 / (2 - auc2)
    Q2_2 = 2 * auc2**2 / (1 + auc2)
    var_auc2 = (auc2 * (1 - auc2) + (n_pos - 1) * (Q1_2 - auc2**2) + 
                (n_neg - 1) * (Q2_2 - auc2**2)) / (n_pos * n_neg)
    
    # Test statistic (assumes independence - simplified)
    se_diff = np.sqrt(var_auc1 + var_auc2)
    z_stat = (auc1 - auc2) / se_diff if se_diff > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        "auc1": auc1,
        "auc2": auc2,
        "auc_diff": auc1 - auc2,
        "z_statistic": z_stat,
        "p_value": p_value,
        "significant": p_value < 0.05
    }


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Calculate calibration metrics.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for ECE calculation
        
    Returns:
        Dictionary with calibration metrics
    """
    # Brier score
    brier = brier_score_loss(y_true, y_prob)
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    bin_stats = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine which predictions fall into this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Contribution to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_stats.append({
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "accuracy": accuracy_in_bin,
                "confidence": avg_confidence_in_bin,
                "count": in_bin.sum(),
                "proportion": prop_in_bin
            })
    
    # Maximum Calibration Error (MCE)
    if bin_stats:
        mce = max(abs(stat["confidence"] - stat["accuracy"]) for stat in bin_stats)
    else:
        mce = 0
    
    return {
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "bin_stats": bin_stats
    }


def sensitivity_at_specificity(y_true: np.ndarray, y_score: np.ndarray, 
                              target_specificity: float = 0.9) -> dict[str, float]:
    """Calculate sensitivity at a target specificity.
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores
        target_specificity: Target specificity level
        
    Returns:
        Dictionary with sensitivity and threshold
    """
    # Sort scores and labels
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    
    # Calculate thresholds
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]
    
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    # Calculate specificity and sensitivity
    specificities = 1 - fps / n_neg
    sensitivities = tps / n_pos
    thresholds = y_score_sorted[threshold_idxs]
    
    # Find closest specificity to target
    idx = np.argmin(np.abs(specificities - target_specificity))
    
    return {
        "sensitivity": sensitivities[idx],
        "specificity": specificities[idx],
        "threshold": thresholds[idx],
        "target_specificity": target_specificity
    }


def decision_curve_analysis(y_true: np.ndarray, y_prob: np.ndarray, 
                           thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Calculate decision curve analysis.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        thresholds: Probability thresholds to evaluate
        
    Returns:
        DataFrame with decision curve data
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    
    results = []
    prevalence = np.mean(y_true)
    
    for threshold in thresholds:
        # Model strategy: treat if predicted probability > threshold
        treat_model = y_prob >= threshold
        
        # Calculate outcomes
        tp = np.sum((treat_model == 1) & (y_true == 1))
        fp = np.sum((treat_model == 1) & (y_true == 0))
        tn = np.sum((treat_model == 0) & (y_true == 0))
        fn = np.sum((treat_model == 0) & (y_true == 1))
        
        n_total = len(y_true)
        
        # Net benefit calculation
        net_benefit_model = (tp / n_total) - (fp / n_total) * (threshold / (1 - threshold))
        
        # Treat all strategy
        net_benefit_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        
        # Treat none strategy
        net_benefit_none = 0
        
        results.append({
            "threshold": threshold,
            "net_benefit_model": net_benefit_model,
            "net_benefit_all": net_benefit_all,
            "net_benefit_none": net_benefit_none,
            "treated_fraction": np.mean(treat_model)
        })
    
    return pd.DataFrame(results)


def comprehensive_evaluation(y_true: np.ndarray, y_score: np.ndarray, 
                           y_prob: np.ndarray | None = None,
                           model_name: str = "Model") -> dict[str, Any]:
    """Comprehensive model evaluation with all metrics.
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores (for ranking metrics)
        y_prob: Predicted probabilities (for calibration)
        model_name: Name of the model
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    if y_prob is None:
        y_prob = y_score
    
    results = {
        "model_name": model_name,
        "n_samples": len(y_true),
        "n_positive": np.sum(y_true),
        "prevalence": np.mean(y_true)
    }
    
    # Discrimination metrics
    results["auroc"] = auroc_with_ci(y_true, y_score)
    results["auprc"] = auprc_with_ci(y_true, y_score)
    
    # Calibration metrics
    results["calibration"] = calibration_metrics(y_true, y_prob)
    
    # Clinical utility metrics
    results["sens_at_spec90"] = sensitivity_at_specificity(y_true, y_score, 0.9)
    results["sens_at_spec95"] = sensitivity_at_specificity(y_true, y_score, 0.95)
    
    # Decision curve analysis
    results["decision_curve"] = decision_curve_analysis(y_true, y_prob)
    
    return results


def compare_models(model_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Compare multiple models and create summary table.
    
    Args:
        model_results: Dictionary of model_name -> evaluation results
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    for model_name, results in model_results.items():
        row = {
            "Model": model_name,
            "AUROC": f"{results['auroc']['mean']:.3f} ({results['auroc']['ci_lower']:.3f}-{results['auroc']['ci_upper']:.3f})",
            "AUPRC": f"{results['auprc']['mean']:.3f} ({results['auprc']['ci_lower']:.3f}-{results['auprc']['ci_upper']:.3f})",
            "Brier": f"{results['calibration']['brier_score']:.3f}",
            "ECE": f"{results['calibration']['ece']:.3f}",
            "Sens@Spec90": f"{results['sens_at_spec90']['sensitivity']:.3f}",
            "Sens@Spec95": f"{results['sens_at_spec95']['sensitivity']:.3f}"
        }
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)