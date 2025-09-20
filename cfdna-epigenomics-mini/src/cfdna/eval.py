"""Evaluation script for comparing cfDNA cancer detection models."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import compare_models, comprehensive_evaluation, delong_test


def load_predictions(artifacts_dir: Path, model_names: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    """Load prediction files for multiple models.
    
    Args:
        artifacts_dir: Directory containing prediction files
        model_names: List of model names to load
        
    Returns:
        Dictionary of model -> split -> predictions DataFrame
    """
    predictions = {}
    
    for model_name in model_names:
        model_preds = {}
        
        for split in ["train", "val", "test"]:
            pred_file = artifacts_dir / f"{model_name}_{split}_predictions.csv"
            
            if pred_file.exists():
                model_preds[split] = pd.read_csv(pred_file)
            else:
                print(f"Warning: {pred_file} not found")
        
        if model_preds:
            predictions[model_name] = model_preds
    
    return predictions


def generate_comparison_report(predictions: dict[str, dict[str, pd.DataFrame]], 
                             output_path: Path) -> None:
    """Generate comprehensive model comparison report.
    
    Args:
        predictions: Dictionary of model -> split -> predictions
        output_path: Path to save report
    """
    report_lines = []
    
    # Header
    report_lines.extend([
        "# cfDNA Cancer Detection Model Evaluation Report",
        "",
        "This report compares the performance of different machine learning models",
        "for early cancer detection from cell-free DNA epigenomic signals.",
        ""
    ])
    
    # Model comparison for each split
    for split in ["train", "val", "test"]:
        report_lines.extend([
            f"## {split.capitalize()} Set Performance",
            ""
        ])
        
        # Collect results for this split
        split_results = {}
        
        for model_name, model_preds in predictions.items():
            if split in model_preds:
                pred_df = model_preds[split]
                y_true = pred_df["true_label"].values
                y_prob = pred_df["predicted_prob"].values
                
                results = comprehensive_evaluation(
                    y_true, y_prob, y_prob, model_name
                )
                split_results[model_name] = results
        
        if split_results:
            # Create comparison table
            comparison_df = compare_models(split_results)
            report_lines.append(comparison_df.to_markdown(index=False))
            report_lines.append("")
            
            # DeLong tests (only for test set)
            if split == "test" and len(split_results) >= 2:
                report_lines.extend([
                    "### Statistical Comparisons (DeLong Test)",
                    ""
                ])
                
                model_names = list(split_results.keys())
                for i, model1 in enumerate(model_names):
                    for model2 in model_names[i+1:]:
                        pred1 = predictions[model1][split]["predicted_prob"].values
                        pred2 = predictions[model2][split]["predicted_prob"].values
                        y_true = predictions[model1][split]["true_label"].values
                        
                        delong_result = delong_test(y_true, pred1, pred2)
                        
                        report_lines.extend([
                            f"**{model1} vs {model2}:**",
                            f"- AUROC difference: {delong_result['auc_diff']:.4f}",
                            f"- p-value: {delong_result['p_value']:.4f}",
                            f"- Significant: {'Yes' if delong_result['significant'] else 'No'}",
                            ""
                        ])
    
    # Clinical interpretation
    report_lines.extend([
        "## Clinical Interpretation",
        "",
        "### Key Findings",
        ""
    ])
    
    # Find best model on test set
    test_results = {}
    for model_name, model_preds in predictions.items():
        if "test" in model_preds:
            pred_df = model_preds["test"]
            y_true = pred_df["true_label"].values
            y_prob = pred_df["predicted_prob"].values
            
            results = comprehensive_evaluation(y_true, y_prob, y_prob, model_name)
            test_results[model_name] = results
    
    if test_results:
        best_model = max(test_results.keys(), 
                        key=lambda x: test_results[x]["auroc"]["mean"])
        best_auroc = test_results[best_model]["auroc"]["mean"]
        best_ci = (test_results[best_model]["auroc"]["ci_lower"],
                  test_results[best_model]["auroc"]["ci_upper"])
        
        report_lines.extend([
            f"- **Best performing model:** {best_model}",
            f"- **Test AUROC:** {best_auroc:.3f} (95% CI: {best_ci[0]:.3f}-{best_ci[1]:.3f})",
            ""
        ])
        
        # Clinical utility metrics
        sens_90 = test_results[best_model]["sens_at_spec90"]["sensitivity"]
        sens_95 = test_results[best_model]["sens_at_spec95"]["sensitivity"]
        
        report_lines.extend([
            "### Clinical Utility",
            "",
            f"At 90% specificity: {sens_90:.1%} sensitivity",
            f"At 95% specificity: {sens_95:.1%} sensitivity",
            ""
        ])
        
        # Calibration assessment
        brier = test_results[best_model]["calibration"]["brier_score"]
        ece = test_results[best_model]["calibration"]["ece"]
        
        report_lines.extend([
            "### Calibration Quality", 
            "",
            f"Brier Score: {brier:.3f} (lower is better)",
            f"Expected Calibration Error: {ece:.3f} (lower is better)",
            ""
        ])
    
    # Limitations and next steps
    report_lines.extend([
        "## Limitations and Next Steps",
        "",
        "### Current Limitations",
        "",
        "- **Synthetic Data:** Results are based on simulated cfDNA data with simplified assumptions",
        "- **Sample Size:** Limited to 600 samples; larger studies needed for clinical validation", 
        "- **Feature Set:** Focuses on methylation DMRs and basic fragmentomics",
        "- **Population:** Does not account for demographic or clinical heterogeneity",
        "",
        "### Recommended Next Steps",
        "",
        "1. **Validation with Real Data:** Test models on clinical cfDNA datasets",
        "2. **Extended Features:** Include additional fragmentomics patterns, copy number, mutations",
        "3. **Larger Cohorts:** Scale to thousands of samples for robust performance estimates",
        "4. **Clinical Integration:** Develop decision support tools for clinical workflow",
        "5. **Regulatory Preparation:** Align with FDA guidelines for diagnostic AI/ML",
        "",
        "---",
        "",
        f"*Report generated by cfDNA Epigenomics Mini v0.1.0*"
    ])
    
    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))


def main() -> None:
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate and compare cfDNA models")
    parser.add_argument("artifacts_dir", type=Path, 
                       help="Directory containing model artifacts")
    parser.add_argument("output_report", type=Path,
                       help="Output path for evaluation report")
    parser.add_argument("--models", nargs="+", 
                       default=["logistic_l1", "logistic_l2", "random_forest", "mlp"],
                       help="Model names to evaluate")
    
    args = parser.parse_args()
    
    print(f"Loading predictions from: {args.artifacts_dir}")
    print(f"Models to evaluate: {args.models}")
    
    # Load predictions
    predictions = load_predictions(args.artifacts_dir, args.models)
    
    if not predictions:
        print("No prediction files found. Make sure models have been trained first.")
        return
    
    print(f"Loaded predictions for models: {list(predictions.keys())}")
    
    # Generate report
    print(f"Generating evaluation report...")
    generate_comparison_report(predictions, args.output_report)
    
    print(f"Evaluation report saved to: {args.output_report}")
    
    # Summary statistics
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for model_name, model_preds in predictions.items():
        if "test" in model_preds:
            pred_df = model_preds["test"]
            y_true = pred_df["true_label"].values
            y_prob = pred_df["predicted_prob"].values
            
            results = comprehensive_evaluation(y_true, y_prob, y_prob, model_name)
            
            auroc = results["auroc"]["mean"]
            auroc_ci = (results["auroc"]["ci_lower"], results["auroc"]["ci_upper"])
            auprc = results["auprc"]["mean"]
            
            print(f"{model_name:>15}: AUROC={auroc:.3f} ({auroc_ci[0]:.3f}-{auroc_ci[1]:.3f}), "
                  f"AUPRC={auprc:.3f}")


if __name__ == "__main__":
    main()