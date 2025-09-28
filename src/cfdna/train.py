"""Training script for cfDNA cancer detection models."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .features import prepare_features
from .models import CalibratedModel, MLPClassifier, MLPTrainer, get_baseline_models
from .preprocessing import PreprocessingPipeline, compute_class_weights
from .metrics import comprehensive_evaluation, delong_test
from .viz import save_evaluation_plots


def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series, 
                class_weights: dict[int, float], 
                artifacts_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Train a single model.
    
    Args:
        model_name: Name of the model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        class_weights: Class weights for imbalanced data
        artifacts_dir: Directory to save artifacts
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    if model_name == "mlp":
        # Train MLP
        input_dim = X_train.shape[1]
        model = MLPClassifier(input_dim=input_dim)
        trainer = MLPTrainer(model, class_weights=class_weights)
        
        print(f"Training MLP with {input_dim} features...")
        history = trainer.fit(
            X_train, y_train, 
            X_val, y_val,
            epochs=100,
            patience=10,
            verbose=True
        )
        
        # Save model
        model_path = artifacts_dir / f"{model_name}_model.pt"
        trainer.save_model(model_path)
        
        return trainer, history
    
    else:
        # Train baseline model
        baseline_models = get_baseline_models(class_weights)
        
        if model_name not in baseline_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = baseline_models[model_name]
        print(f"Training {model_name}...")
        
        model.fit(X_train, y_train)
        
        # Save model predictions for evaluation
        train_probs = model.predict_proba(X_train)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]
        
        history = {
            "train_probs": train_probs.tolist(),
            "val_probs": val_probs.tolist()
        }
        
        return model, history


def run_nested_cv(data: dict[str, Any], model_name: str, n_outer_folds: int = 3, 
                 n_inner_folds: int = 3, artifacts_dir: Path | None = None) -> dict[str, Any]:
    """Run nested cross-validation.
    
    Args:
        data: Prepared feature data
        model_name: Name of model to evaluate
        n_outer_folds: Number of outer CV folds
        n_inner_folds: Number of inner CV folds  
        artifacts_dir: Directory to save artifacts
        
    Returns:
        Nested CV results
    """
    from sklearn.model_selection import StratifiedKFold
    
    X = data["X"]
    y = data["y"]
    metadata = data["metadata"]
    
    # Outer CV loop
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
    
    cv_results = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\nOuter CV Fold {fold_idx + 1}/{n_outer_folds}")
        
        # Split data
        X_train_val = X.iloc[train_val_idx]
        y_train_val = y.iloc[train_val_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        # Inner CV for hyperparameter tuning (simplified)
        inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
        
        best_val_score = 0
        best_model = None
        
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val):
            X_inner_train = X_train_val.iloc[inner_train_idx]
            y_inner_train = y_train_val.iloc[inner_train_idx]
            X_inner_val = X_train_val.iloc[inner_val_idx]
            y_inner_val = y_train_val.iloc[inner_val_idx]
            
            # Train model
            class_weights = compute_class_weights(y_inner_train, np.arange(len(y_inner_train)))
            model, _ = train_model(
                model_name, X_inner_train, y_inner_train,
                X_inner_val, y_inner_val, class_weights, 
                artifacts_dir or Path("temp")
            )
            
            # Evaluate
            if hasattr(model, "predict_proba"):
                val_probs = model.predict_proba(X_inner_val)[:, 1]
            else:
                val_probs = model.predict_proba(X_inner_val)[:, 1]
            
            from sklearn.metrics import roc_auc_score
            val_score = roc_auc_score(y_inner_val, val_probs)
            
            if val_score > best_val_score:
                best_val_score = val_score
                best_model = model
        
        # Evaluate best model on test set
        if hasattr(best_model, "predict_proba"):
            test_probs = best_model.predict_proba(X_test)[:, 1]
        else:
            test_probs = best_model.predict_proba(X_test)[:, 1]
        
        fold_results = comprehensive_evaluation(
            y_test.values, test_probs, test_probs, f"{model_name}_fold_{fold_idx}"
        )
        
        cv_results.append(fold_results)
    
    return {"cv_results": cv_results, "model_name": model_name}


def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(description="Train cfDNA cancer detection models")
    parser.add_argument("config", type=Path, help="Configuration file")
    parser.add_argument("model", choices=["logistic_l1", "logistic_l2", "random_forest", "mlp"],
                       help="Model type to train")
    parser.add_argument("output_dir", type=Path, help="Output directory for artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nested_cv", action="store_true", help="Run nested cross-validation")
    parser.add_argument("--calibrate", action="store_true", help="Apply probability calibration")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print(f"Training {args.model} model...")
    print(f"Output directory: {args.output_dir}")
    
    # Prepare features
    print("Preparing features...")
    data_dir = args.config.parent
    data = prepare_features(data_dir, config)
    
    X = data["X"]
    y = data["y"]
    metadata = data["metadata"]
    splits = data["splits"]
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Setup preprocessing
    preprocessing = PreprocessingPipeline(splits, config.get("preprocessing", {}))
    
    # Preprocess data
    print("Preprocessing data...")
    X_train = preprocessing.fit_transform_train(X, splits)
    X_val = preprocessing.transform_split(X, "val", splits)
    X_test = preprocessing.transform_split(X, "test", splits)
    
    y_train = y.iloc[splits["train"]]
    y_val = y.iloc[splits["val"]]
    y_test = y.iloc[splits["test"]]
    
    # Compute class weights
    class_weights = compute_class_weights(y_train, np.arange(len(y_train)))
    print(f"Class weights: {class_weights}")
    
    # Train model
    artifacts_dir = Path(args.output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    if args.nested_cv:
        print("Running nested cross-validation...")
        cv_results = run_nested_cv(data, args.model, artifacts_dir=artifacts_dir)
        
        # Save CV results
        with open(artifacts_dir / f"{args.model}_nested_cv.json", "w") as f:
            json.dump(cv_results, f, indent=2, default=str)
        
        print("Nested CV completed!")
        return
    
    # Regular training
    model, history = train_model(
        args.model, X_train, y_train, X_val, y_val, 
        class_weights, artifacts_dir
    )
    
    # Apply calibration if requested
    if args.calibrate:
        print("Applying probability calibration...")
        calibrated_model = CalibratedModel(model, method="platt")
        calibrated_model.fit(X_train, y_train, X_val, y_val)
        model = calibrated_model
    
    # Evaluate on all splits
    print("Evaluating model...")
    
    results = {}
    
    for split_name, (X_split, y_split) in [
        ("train", (X_train, y_train)),
        ("val", (X_val, y_val)), 
        ("test", (X_test, y_test))
    ]:
        print(f"Evaluating on {split_name} set...")
        
        # Get predictions
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_split)[:, 1]
        else:
            probs = model.predict_proba(X_split)[:, 1]
        
        # Comprehensive evaluation
        eval_results = comprehensive_evaluation(
            y_split.values, probs, probs, f"{args.model}_{split_name}"
        )
        
        results[split_name] = eval_results
        
        # Save predictions
        pred_df = pd.DataFrame({
            "sample_id": X_split.index,
            "true_label": y_split.values,
            "predicted_prob": probs
        })
        pred_df.to_csv(artifacts_dir / f"{args.model}_{split_name}_predictions.csv", index=False)
    
    # Save evaluation results
    with open(artifacts_dir / f"{args.model}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save training history
    with open(artifacts_dir / f"{args.model}_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    
    # Generate plots
    print("Generating evaluation plots...")
    save_evaluation_plots(results, artifacts_dir, f"{args.model}_")
    
    # Print summary
    print(f"\nTraining completed!")
    print(f"Test AUROC: {results['test']['auroc']['mean']:.3f} "
          f"({results['test']['auroc']['ci_lower']:.3f}-{results['test']['auroc']['ci_upper']:.3f})")
    print(f"Test AUPRC: {results['test']['auprc']['mean']:.3f} "
          f"({results['test']['auprc']['ci_lower']:.3f}-{results['test']['auprc']['ci_upper']:.3f})")
    
    print(f"Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()