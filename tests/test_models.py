"""Tests for cfDNA machine learning models."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cfdna.models import (
    CalibratedModel,
    MLPClassifier,
    MLPTrainer,
    get_baseline_models,
)


@pytest.fixture
def toy_data():
    """Generate toy dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    # Create linearly separable data
    X = np.random.randn(n_samples, n_features)
    # Add signal to first few features
    true_coef = np.zeros(n_features)
    true_coef[:5] = [2, -1.5, 1, -2, 1.5]
    
    y_logits = X @ true_coef + 0.5 * np.random.randn(n_samples)
    y = (y_logits > 0).astype(int)
    
    # Convert to DataFrames
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y_series = pd.Series(y, name="label")
    
    return X_df, y_series


@pytest.fixture
def mlp_model():
    """Create a simple MLP model for testing."""
    return MLPClassifier(input_dim=20, hidden_dims=[32, 16], dropout_rate=0.1)


def test_mlp_classifier_initialization():
    """Test MLP classifier initialization."""
    model = MLPClassifier(input_dim=10, hidden_dims=[64, 32], dropout_rate=0.2)
    
    assert model.input_dim == 10
    assert model.hidden_dims == [64, 32]
    assert model.dropout_rate == 0.2
    
    # Test forward pass with dummy data
    x = torch.randn(5, 10)
    output = model(x)
    
    assert output.shape == (5, 1), f"Expected shape (5, 1), got {output.shape}"


def test_mlp_classifier_default_params():
    """Test MLP classifier with default parameters."""
    model = MLPClassifier(input_dim=15)
    
    assert model.hidden_dims == [128, 64, 32]
    assert model.dropout_rate == 0.2
    assert model.weight_decay == 1e-4


def test_mlp_trainer_initialization(mlp_model):
    """Test MLP trainer initialization."""
    class_weights = {0: 1.0, 1: 1.5}
    trainer = MLPTrainer(mlp_model, class_weights=class_weights)
    
    assert trainer.model == mlp_model
    assert trainer.device == "cpu"
    assert trainer.best_val_auc == 0.0


def test_mlp_trainer_overfit_tiny_dataset(toy_data):
    """Test that MLP can overfit a tiny dataset (sanity check)."""
    X_toy, y_toy = toy_data
    
    # Use very small dataset for overfitting test
    X_small = X_toy.iloc[:20]
    y_small = y_toy.iloc[:20]
    
    # Create model and trainer
    model = MLPClassifier(input_dim=X_small.shape[1], hidden_dims=[64, 32])
    trainer = MLPTrainer(model, learning_rate=1e-2)
    
    # Train on same data for both train and val (intentional overfitting)
    history = trainer.fit(
        X_small, y_small, X_small, y_small,
        epochs=50, patience=50, verbose=False
    )
    
    # Should achieve high accuracy on this toy problem
    final_auc = history["val_auc"][-1]
    assert final_auc > 0.8, f"Expected AUC > 0.8 on toy data, got {final_auc}"


def test_mlp_trainer_early_stopping(toy_data):
    """Test early stopping functionality."""
    X_toy, y_toy = toy_data
    
    # Split into train/val
    split_idx = len(X_toy) // 2
    X_train = X_toy.iloc[:split_idx]
    y_train = y_toy.iloc[:split_idx]
    X_val = X_toy.iloc[split_idx:]
    y_val = y_toy.iloc[split_idx:]
    
    model = MLPClassifier(input_dim=X_toy.shape[1], hidden_dims=[32, 16])
    trainer = MLPTrainer(model)
    
    # Train with early stopping
    history = trainer.fit(
        X_train, y_train, X_val, y_val,
        epochs=100, patience=5, verbose=False
    )
    
    # Should stop before 100 epochs due to early stopping
    actual_epochs = len(history["train_loss"])
    assert actual_epochs < 100, f"Expected early stopping, but trained for {actual_epochs} epochs"
    
    # Should have saved best model state
    assert trainer.best_model_state is not None


def test_mlp_trainer_predict_proba(toy_data):
    """Test prediction functionality."""
    X_toy, y_toy = toy_data
    
    model = MLPClassifier(input_dim=X_toy.shape[1], hidden_dims=[32])
    trainer = MLPTrainer(model)
    
    # Quick training
    trainer.fit(
        X_toy, y_toy, X_toy, y_toy,
        epochs=10, patience=10, verbose=False
    )
    
    # Test predictions
    probs = trainer.predict_proba(X_toy)
    
    assert probs.shape == (len(X_toy), 2), f"Expected shape ({len(X_toy)}, 2), got {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    assert np.all(probs >= 0), "Negative probabilities found"
    assert np.all(probs <= 1), "Probabilities > 1 found"


def test_mlp_trainer_save_load(toy_data):
    """Test model saving and loading."""
    X_toy, y_toy = toy_data
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.pt"
        
        # Train model
        model = MLPClassifier(input_dim=X_toy.shape[1], hidden_dims=[32])
        trainer = MLPTrainer(model)
        
        trainer.fit(
            X_toy, y_toy, X_toy, y_toy,
            epochs=5, patience=5, verbose=False
        )
        
        # Save model
        trainer.save_model(model_path)
        assert model_path.exists()
        
        # Load model
        loaded_trainer = MLPTrainer.load_model(model_path)
        
        # Test that loaded model gives same predictions
        orig_probs = trainer.predict_proba(X_toy)
        loaded_probs = loaded_trainer.predict_proba(X_toy)
        
        np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5)


def test_get_baseline_models():
    """Test baseline model creation."""
    models = get_baseline_models()
    
    expected_models = {"logistic_l1", "logistic_l2", "random_forest"}
    assert set(models.keys()) == expected_models
    
    # Test with class weights
    class_weights = {0: 1.0, 1: 2.0}
    models_weighted = get_baseline_models(class_weights)
    
    assert set(models_weighted.keys()) == expected_models


def test_baseline_models_fit_predict(toy_data):
    """Test that baseline models can fit and predict."""
    X_toy, y_toy = toy_data
    
    models = get_baseline_models()
    
    for model_name, model in models.items():
        # Fit model
        model.fit(X_toy, y_toy)
        
        # Test predictions
        probs = model.predict_proba(X_toy)
        preds = model.predict(X_toy)
        
        assert probs.shape == (len(X_toy), 2), f"Model {model_name}: wrong prob shape"
        assert len(preds) == len(X_toy), f"Model {model_name}: wrong pred length"
        assert set(preds).issubset({0, 1}), f"Model {model_name}: invalid predictions"


def test_calibrated_model_platt(toy_data):
    """Test Platt calibration."""
    X_toy, y_toy = toy_data
    
    # Split data
    split_idx = len(X_toy) // 3
    X_train = X_toy.iloc[:split_idx]
    y_train = y_toy.iloc[:split_idx]
    X_cal = X_toy.iloc[split_idx:2*split_idx]
    y_cal = y_toy.iloc[split_idx:2*split_idx]
    X_test = X_toy.iloc[2*split_idx:]
    
    # Get base model
    from sklearn.linear_model import LogisticRegression
    base_model = LogisticRegression(random_state=42)
    
    # Create calibrated model
    cal_model = CalibratedModel(base_model, method="platt")
    
    # Fit
    cal_model.fit(X_train, y_train, X_cal, y_cal)
    
    # Test predictions
    probs = cal_model.predict_proba(X_test)
    preds = cal_model.predict(X_test)
    
    assert probs.shape == (len(X_test), 2)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_calibrated_model_isotonic(toy_data):
    """Test isotonic calibration."""
    X_toy, y_toy = toy_data
    
    # Split data
    split_idx = len(X_toy) // 3
    X_train = X_toy.iloc[:split_idx]
    y_train = y_toy.iloc[:split_idx]
    X_cal = X_toy.iloc[split_idx:2*split_idx]
    y_cal = y_toy.iloc[split_idx:2*split_idx]
    X_test = X_toy.iloc[2*split_idx:]
    
    # Get base model
    from sklearn.ensemble import RandomForestClassifier
    base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create calibrated model
    cal_model = CalibratedModel(base_model, method="isotonic")
    
    # Fit
    cal_model.fit(X_train, y_train, X_cal, y_cal)
    
    # Test predictions
    probs = cal_model.predict_proba(X_test)
    preds = cal_model.predict(X_test)
    
    assert probs.shape == (len(X_test), 2)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_calibrated_model_error_handling():
    """Test error handling in calibrated model."""
    from sklearn.linear_model import LogisticRegression
    base_model = LogisticRegression()
    
    # Test invalid calibration method
    with pytest.raises(ValueError):
        cal_model = CalibratedModel(base_model, method="invalid_method")


def test_mlp_with_class_weights(toy_data):
    """Test MLP training with class weights."""
    X_toy, y_toy = toy_data
    
    # Create imbalanced dataset
    class_1_mask = y_toy == 1
    X_imbalanced = X_toy[class_1_mask].iloc[:5].append(X_toy[~class_1_mask].iloc[:45])
    y_imbalanced = y_toy[class_1_mask].iloc[:5].append(y_toy[~class_1_mask].iloc[:45])
    
    # Calculate class weights
    class_weights = {
        0: len(y_imbalanced) / (2 * (y_imbalanced == 0).sum()),
        1: len(y_imbalanced) / (2 * (y_imbalanced == 1).sum())
    }
    
    model = MLPClassifier(input_dim=X_toy.shape[1], hidden_dims=[32])
    trainer = MLPTrainer(model, class_weights=class_weights)
    
    # Should not raise errors
    history = trainer.fit(
        X_imbalanced, y_imbalanced, X_imbalanced, y_imbalanced,
        epochs=5, patience=5, verbose=False
    )
    
    assert len(history["train_loss"]) > 0