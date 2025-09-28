"""Report generation functions for CLI integration."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def generate_report(results_dir: Path, output_path: Path) -> Path:
    """Generate HTML report from evaluation results.
    
    Args:
        results_dir: Directory containing evaluation results
        output_path: Output path for HTML report
        
    Returns:
        Path to generated report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    # Load evaluation results
    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    with open(metrics_file) as f:
        data = json.load(f)
    
    summary_data = data.get("summary", [])
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>cfDNA Epigenomics Mini - Model Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
        }}
        
        .header p {{
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1.5rem;
        }}
        
        .metric-card h3 {{
            margin: 0 0 0.5rem 0;
            color: #495057;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #007bff;
        }}
        
        .metric-subtitle {{
            color: #6c757d;
            font-size: 0.9rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        
        .best-model {{
            background-color: #d4edda !important;
            border-left: 4px solid #28a745;
        }}
        
        .section {{
            margin-bottom: 2rem;
        }}
        
        .section h2 {{
            color: #495057;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 0.5rem;
        }}
        
        .limitations {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 2rem 0;
        }}
        
        .limitations h3 {{
            color: #856404;
            margin-top: 0;
        }}
        
        .footer {{
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #e9ecef;
            text-align: center;
            color: #6c757d;
        }}
        
        .ci {{
            font-size: 0.8em;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>cfDNA Epigenomics Mini</h1>
        <p>Model Evaluation Report - Early Cancer Detection from cfDNA Epigenomic Signals</p>
    </div>
"""

    if summary_data:
        # Find best model
        best_model = max(summary_data, key=lambda x: x["test_auroc"])
        
        # Add summary metrics
        html_content += f"""
    <div class="summary-grid">
        <div class="metric-card">
            <h3>Best Model</h3>
            <div class="metric-value">{best_model['model']}</div>
            <div class="metric-subtitle">Highest test AUROC</div>
        </div>
        
        <div class="metric-card">
            <h3>Test AUROC</h3>
            <div class="metric-value">{best_model['test_auroc']:.3f}</div>
            <div class="metric-subtitle">
                95% CI: {best_model['test_auroc_ci_lower']:.3f}–{best_model['test_auroc_ci_upper']:.3f}
            </div>
        </div>
        
        <div class="metric-card">
            <h3>Test AUPRC</h3>
            <div class="metric-value">{best_model['test_auprc']:.3f}</div>
            <div class="metric-subtitle">Area under PR curve</div>
        </div>
        
        <div class="metric-card">
            <h3>Sensitivity @ 90% Spec</h3>
            <div class="metric-value">{best_model['sens_at_spec90']:.1%}</div>
            <div class="metric-subtitle">Clinical utility metric</div>
        </div>
    </div>
"""

        # Add detailed results table
        html_content += f"""
    <div class="section">
        <h2>Detailed Model Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Test AUROC</th>
                    <th>Test AUPRC</th>
                    <th>Brier Score</th>
                    <th>Sens @ 90% Spec</th>
                    <th>Sens @ 95% Spec</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for model_data in sorted(summary_data, key=lambda x: x["test_auroc"], reverse=True):
            row_class = "best-model" if model_data == best_model else ""
            html_content += f"""
                <tr class="{row_class}">
                    <td><strong>{model_data['model']}</strong></td>
                    <td>
                        {model_data['test_auroc']:.3f}
                        <div class="ci">[{model_data['test_auroc_ci_lower']:.3f}, {model_data['test_auroc_ci_upper']:.3f}]</div>
                    </td>
                    <td>{model_data['test_auprc']:.3f}</td>
                    <td>{model_data['test_brier']:.3f}</td>
                    <td>{model_data['sens_at_spec90']:.1%}</td>
                    <td>{model_data['sens_at_spec95']:.1%}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
    </div>
"""

    # Add interpretation and limitations
    html_content += f"""
    <div class="section">
        <h2>Clinical Interpretation</h2>
        <p>
            The evaluation demonstrates the potential of machine learning models for early cancer detection 
            from cfDNA epigenomic signals. The {best_model['model'] if summary_data else 'best performing'} model shows 
            promising discriminative ability with balanced sensitivity and specificity.
        </p>
        
        <h3>Key Performance Indicators</h3>
        <ul>
            <li><strong>AUROC:</strong> Measures the model's ability to distinguish between cancer and control samples</li>
            <li><strong>AUPRC:</strong> Particularly important for imbalanced datasets, focuses on precision-recall trade-offs</li>
            <li><strong>Clinical Sensitivity:</strong> Critical for early detection applications where missing cancer cases has high cost</li>
        </ul>
    </div>
    
    <div class="limitations">
        <h3>⚠️ Important Limitations</h3>
        <ul>
            <li><strong>Synthetic Data:</strong> Results are based on simulated cfDNA data with simplified biological assumptions</li>
            <li><strong>Limited Scale:</strong> Small dataset size (n=600) compared to clinical requirements</li>
            <li><strong>Feature Coverage:</strong> Basic methylation DMRs and fragmentomics; real applications need comprehensive genomic features</li>
            <li><strong>Validation Gap:</strong> No validation on independent clinical cohorts</li>
        </ul>
        
        <h4>Clinical Translation Requirements</h4>
        <ul>
            <li>Validation on large-scale clinical cfDNA datasets (n>10,000)</li>
            <li>Multi-site validation across diverse populations</li>
            <li>Integration with clinical risk factors and imaging</li>
            <li>Regulatory pathway development (FDA Pre-Cert, CLIA compliance)</li>
            <li>Health economic evaluation and reimbursement strategy</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Technical Implementation</h2>
        <p>
            This demonstration package provides a complete pipeline from data simulation to model evaluation:
        </p>
        <ul>
            <li><strong>Reproducible Pipeline:</strong> Deterministic simulation and evaluation with seed control</li>
            <li><strong>Industry Standards:</strong> Proper train/val/test splits, bootstrap confidence intervals, calibration assessment</li>
            <li><strong>Modular Design:</strong> Extensible architecture for additional features and models</li>
            <li><strong>Quality Assurance:</strong> Comprehensive testing, type hints, and documentation</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>
            Generated by <strong>cfDNA Epigenomics Mini v0.1.0</strong><br>
            <em>A production-quality demonstration of ML for early cancer detection from cfDNA</em>
        </p>
        <p style="margin-top: 1rem; font-size: 0.8rem;">
            This is a research demonstration with synthetic data. Not for clinical use.
        </p>
    </div>
</body>
</html>
"""
    
    # Write the HTML file
    with open(output_path, "w") as f:
        f.write(html_content)
    
    return output_path