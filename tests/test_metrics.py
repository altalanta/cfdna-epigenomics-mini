"""Tests for cfDNA evaluation metrics."""

import numpy as np
import pytest

from cfdna.metrics import (
    auroc_with_ci,
    auprc_with_ci,
    bootstrap_metric,
    calibration_metrics,
    compare_models,
    comprehensive_evaluation,
    decision_curve_analysis,
    delong_test,
    sensitivity_at_specificity,
)


@pytest.fixture
def sample_predictions():
    """Generate sample prediction data for testing."""
    np.random.seed(42)
    n_samples = 200
    
    # Create realistic predictions with some signal
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Generate predictions with signal
    noise = np.random.normal(0, 0.3, n_samples)
    y_score = 0.3 + 0.4 * y_true + noise
    y_score = np.clip(y_score, 0.01, 0.99)  # Clip to valid probability range
    
    return y_true, y_score


@pytest.fixture
def perfect_predictions():
    """Generate perfect prediction data for testing edge cases."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    return y_true, y_score


def test_bootstrap_metric_basic(sample_predictions):
    """Test basic bootstrap metric functionality."""
    y_true, y_score = sample_predictions
    
    from sklearn.metrics import roc_auc_score
    
    result = bootstrap_metric(
        y_true, y_score, roc_auc_score, 
        n_bootstrap=100, random_state=42
    )
    
    # Check result structure
    assert "mean" in result
    assert "std" in result  
    assert "ci_lower" in result
    assert "ci_upper" in result
    
    # Check values are reasonable
    assert 0 <= result["mean"] <= 1, f"AUROC mean {result['mean']} out of range"
    assert result["std"] >= 0, f"Negative std: {result['std']}"
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"], "CI doesn't contain mean"


def test_auroc_with_ci(sample_predictions):
    """Test AUROC with confidence intervals."""
    y_true, y_score = sample_predictions
    
    result = auroc_with_ci(y_true, y_score, n_bootstrap=50, random_state=42)
    
    # Should be reasonable AUROC for our synthetic data
    assert 0.5 <= result["mean"] <= 1.0, f"Unexpected AUROC: {result['mean']}"
    assert result["ci_lower"] < result["ci_upper"], "Invalid confidence interval"


def test_auprc_with_ci(sample_predictions):
    """Test AUPRC with confidence intervals."""
    y_true, y_score = sample_predictions
    
    result = auprc_with_ci(y_true, y_score, n_bootstrap=50, random_state=42)
    
    # Check basic properties
    assert 0 <= result["mean"] <= 1, f"AUPRC out of range: {result['mean']}"
    assert result["ci_lower"] < result["ci_upper"], "Invalid confidence interval"


def test_delong_test_basic(sample_predictions):
    """Test DeLong test for comparing ROC curves."""
    y_true, y_score = sample_predictions
    
    # Create second set of scores (slightly different)
    np.random.seed(123)
    noise2 = np.random.normal(0, 0.2, len(y_true))
    y_score2 = y_score + 0.1 * noise2
    y_score2 = np.clip(y_score2, 0.01, 0.99)
    
    result = delong_test(y_true, y_score, y_score2)
    
    # Check result structure
    assert "auc1" in result
    assert "auc2" in result
    assert "auc_diff" in result
    assert "z_statistic" in result
    assert "p_value" in result
    assert "significant" in result
    
    # Check value ranges
    assert 0 <= result["auc1"] <= 1
    assert 0 <= result["auc2"] <= 1
    assert 0 <= result["p_value"] <= 1
    assert isinstance(result["significant"], bool)


def test_delong_test_identical_scores(sample_predictions):
    """Test DeLong test with identical scores."""
    y_true, y_score = sample_predictions
    
    result = delong_test(y_true, y_score, y_score)
    
    # Should be no difference
    assert abs(result["auc_diff"]) < 1e-10, f"Expected no difference, got {result['auc_diff']}"
    assert result["p_value"] > 0.95, f"Expected high p-value, got {result['p_value']}"
    assert not result["significant"], "Identical scores should not be significantly different"


def test_calibration_metrics_basic(sample_predictions):
    """Test calibration metrics."""
    y_true, y_score = sample_predictions
    
    result = calibration_metrics(y_true, y_score, n_bins=10)
    
    # Check result structure
    assert "brier_score" in result
    assert "ece" in result
    assert "mce" in result
    assert "bin_stats" in result
    
    # Check value ranges
    assert 0 <= result["brier_score"] <= 1, f"Brier score out of range: {result['brier_score']}"
    assert 0 <= result["ece"] <= 1, f"ECE out of range: {result['ece']}"
    assert 0 <= result["mce"] <= 1, f"MCE out of range: {result['mce']}"
    
    # Check bin stats
    bin_stats = result["bin_stats"]
    assert isinstance(bin_stats, list)
    if bin_stats:  # If any bins have data
        assert all("accuracy" in stat for stat in bin_stats)
        assert all("confidence" in stat for stat in bin_stats)


def test_calibration_metrics_perfect_calibration():
    """Test calibration metrics with perfectly calibrated predictions."""
    # Create perfectly calibrated data
    np.random.seed(42)
    n_samples = 1000
    y_prob = np.random.uniform(0, 1, n_samples)
    y_true = np.random.binomial(1, y_prob)
    
    result = calibration_metrics(y_true, y_prob, n_bins=10)
    
    # Should have low calibration error
    assert result["ece"] < 0.1, f"ECE too high for well-calibrated data: {result['ece']}"


def test_sensitivity_at_specificity(perfect_predictions):
    """Test sensitivity at specificity calculation."""
    y_true, y_score = perfect_predictions
    
    result = sensitivity_at_specificity(y_true, y_score, target_specificity=0.5)
    
    # Check result structure
    assert "sensitivity" in result
    assert "specificity" in result
    assert "threshold" in result
    assert "target_specificity" in result
    
    # Check value ranges
    assert 0 <= result["sensitivity"] <= 1
    assert 0 <= result["specificity"] <= 1
    assert result["target_specificity"] == 0.5


def test_decision_curve_analysis(sample_predictions):
    """Test decision curve analysis."""
    y_true, y_score = sample_predictions
    
    thresholds = np.linspace(0, 1, 21)
    result = decision_curve_analysis(y_true, y_score, thresholds)
    
    # Check result is DataFrame
    assert hasattr(result, "columns"), "Expected DataFrame"
    
    # Check columns
    expected_cols = ["threshold", "net_benefit_model", "net_benefit_all", 
                    "net_benefit_none", "treated_fraction"]
    assert all(col in result.columns for col in expected_cols)
    
    # Check row count
    assert len(result) == len(thresholds)
    
    # Check treat none strategy
    assert all(result["net_benefit_none"] == 0), "Treat none should have zero net benefit"
    
    # Check treated fraction is reasonable
    assert all(0 <= result["treated_fraction"]) and all(result["treated_fraction"] <= 1)


def test_comprehensive_evaluation(sample_predictions):
    """Test comprehensive evaluation function."""
    y_true, y_score = sample_predictions
    
    result = comprehensive_evaluation(y_true, y_score, y_score, "TestModel")
    
    # Check main sections
    expected_keys = [
        "model_name", "n_samples", "n_positive", "prevalence",
        "auroc", "auprc", "calibration", "sens_at_spec90", 
        "sens_at_spec95", "decision_curve"
    ]
    
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"
    
    # Check basic statistics
    assert result["model_name"] == "TestModel"
    assert result["n_samples"] == len(y_true)
    assert result["n_positive"] == np.sum(y_true)
    assert result["prevalence"] == np.mean(y_true)
    
    # Check metric structures
    assert "mean" in result["auroc"]
    assert "ci_lower" in result["auroc"]
    assert "sensitivity" in result["sens_at_spec90"]


def test_compare_models():
    """Test model comparison function."""
    # Create mock evaluation results
    model1_results = {
        "auroc": {"mean": 0.85, "ci_lower": 0.80, "ci_upper": 0.90},
        "auprc": {"mean": 0.75, "ci_lower": 0.70, "ci_upper": 0.80},
        "calibration": {"brier_score": 0.15, "ece": 0.05},
        "sens_at_spec90": {"sensitivity": 0.70},
        "sens_at_spec95": {"sensitivity": 0.60}
    }
    
    model2_results = {
        "auroc": {"mean": 0.82, "ci_lower": 0.77, "ci_upper": 0.87},
        "auprc": {"mean": 0.72, "ci_lower": 0.67, "ci_upper": 0.77},
        "calibration": {"brier_score": 0.18, "ece": 0.08},
        "sens_at_spec90": {"sensitivity": 0.65},
        "sens_at_spec95": {"sensitivity": 0.55}
    }
    
    model_results = {
        "Model1": model1_results,
        "Model2": model2_results
    }
    
    comparison_df = compare_models(model_results)
    
    # Check DataFrame structure
    assert len(comparison_df) == 2
    assert "Model" in comparison_df.columns
    assert "AUROC" in comparison_df.columns
    assert "AUPRC" in comparison_df.columns
    
    # Check content
    assert "Model1" in comparison_df["Model"].values
    assert "Model2" in comparison_df["Model"].values


def test_bootstrap_metric_edge_cases():
    """Test bootstrap metric with edge cases."""
    # All positive class
    y_true_pos = np.ones(10)
    y_score_pos = np.random.uniform(0.5, 1.0, 10)
    
    # All negative class  
    y_true_neg = np.zeros(10)
    y_score_neg = np.random.uniform(0.0, 0.5, 10)
    
    from sklearn.metrics import roc_auc_score
    
    # These should handle gracefully or raise appropriate errors
    try:
        result_pos = bootstrap_metric(y_true_pos, y_score_pos, roc_auc_score, n_bootstrap=10)
        # If it doesn't error, result should be reasonable
        assert np.isnan(result_pos["mean"]) or result_pos["mean"] == 1.0
    except (ValueError, ZeroDivisionError):
        # Expected for degenerate cases
        pass
    
    try:
        result_neg = bootstrap_metric(y_true_neg, y_score_neg, roc_auc_score, n_bootstrap=10)
        assert np.isnan(result_neg["mean"]) or result_neg["mean"] == 0.0
    except (ValueError, ZeroDivisionError):
        # Expected for degenerate cases
        pass


def test_metrics_with_small_sample():
    """Test metrics with very small samples."""
    y_true = np.array([0, 1])
    y_score = np.array([0.3, 0.7])
    
    # Should handle small samples gracefully
    result = auroc_with_ci(y_true, y_score, n_bootstrap=10, random_state=42)
    
    # Basic checks
    assert "mean" in result
    assert not np.isnan(result["mean"]) or result["mean"] == 1.0  # Perfect separation