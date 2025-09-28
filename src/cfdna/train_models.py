"""Model training functions for CLI integration."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .models import CalibratedModel, MLPClassifier, MLPTrainer, get_baseline_models


def compute_class_weights(y: pd.Series, indices: np.ndarray) -> dict[int, float]:
    """Compute class weights for imbalanced datasets."""
    y_subset = y.iloc[indices]
    class_counts = y_subset.value_counts()
    total = len(y_subset)
    
    weights = {}
    for class_label in [0, 1]:
        if class_label in class_counts:
            weights[class_label] = total / (2 * class_counts[class_label])
        else:
            weights[class_label] = 1.0
    
    return weights


def train_models(X: pd.DataFrame, y: pd.Series, splits: dict[str, np.ndarray], 
                 model_names: list[str], device: str, out_dir: Path) -> dict[str, Any]:
    """Train multiple models.
    
    Args:
        X: Feature matrix
        y: Target labels
        splits: Train/val/test splits
        model_names: List of model names to train
        device: Device to use for training
        out_dir: Output directory for models
        
    Returns:
        Dictionary with training results
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    
    # Get data splits
    X_train = X.iloc[splits["train"]]
    y_train = y.iloc[splits["train"]]
    X_val = X.iloc[splits["val"]]
    y_val = y.iloc[splits["val"]]
    X_test = X.iloc[splits["test"]]
    y_test = y.iloc[splits["test"]]
    
    # Compute class weights
    class_weights = compute_class_weights(y, splits["train"])
    
    results = {}
    
    for model_name in model_names:
        print(f"Training {model_name}...")
        
        if model_name == "mlp":
            # Train MLP
            input_dim = X_train.shape[1]
            model = MLPClassifier(input_dim=input_dim)
            
            # Auto-detect device
            if device == "auto":
                device_name = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device_name = device
            
            trainer = MLPTrainer(model, class_weights=class_weights, device=device_name)
            
            history = trainer.fit(
                X_train, y_train,
                X_val, y_val,
                epochs=50,  # Reduced for speed
                patience=10,
                verbose=False
            )
            
            # Save model
            model_path = out_dir / f"{model_name}_model.pt"
            trainer.save_model(model_path)
            
            # Get predictions
            train_probs = trainer.predict_proba(X_train)[:, 1]
            val_probs = trainer.predict_proba(X_val)[:, 1]
            test_probs = trainer.predict_proba(X_test)[:, 1]
            
            results[model_name] = {
                "model_path": str(model_path),
                "train_auc": roc_auc_score(y_train, train_probs),
                "val_auc": roc_auc_score(y_val, val_probs),
                "test_auc": roc_auc_score(y_test, test_probs),
                "history": history
            }
            
        elif model_name in ["logistic", "logistic_l1", "logistic_l2", "random_forest"]:
            # Train baseline models
            baseline_models = get_baseline_models(class_weights)
            
            # Map simplified names
            model_key = model_name
            if model_name == "logistic":
                model_key = "logistic_l2"
            
            if model_key not in baseline_models:
                print(f"Warning: Unknown model {model_name}, skipping")
                continue
                
            model = baseline_models[model_key]
            model.fit(X_train, y_train)
            
            # Get predictions
            train_probs = model.predict_proba(X_train)[:, 1]
            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]
            
            # Save model using pickle
            import pickle
            model_path = out_dir / f"{model_name}_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            results[model_name] = {
                "model_path": str(model_path),
                "train_auc": roc_auc_score(y_train, train_probs),
                "val_auc": roc_auc_score(y_val, val_probs),
                "test_auc": roc_auc_score(y_test, test_probs)
            }
        
        else:
            print(f"Warning: Unknown model {model_name}, skipping")
            continue
        
        # Save predictions for all models
        for split_name, (X_split, y_split, probs) in [
            ("train", (X_train, y_train, train_probs)),
            ("val", (X_val, y_val, val_probs)),
            ("test", (X_test, y_test, test_probs))
        ]:
            pred_df = pd.DataFrame({
                "sample_id": X_split.index,
                "true_label": y_split.values,
                "predicted_prob": probs
            })
            pred_path = out_dir / f"{model_name}_{split_name}_predictions.csv"
            pred_df.to_csv(pred_path, index=False)
    
    # Save overall results
    results_path = out_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results