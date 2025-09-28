"""Machine learning models for cfDNA cancer detection."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score


class MLPClassifier(nn.Module):
    """Multi-layer perceptron for binary classification."""
    
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, 
                 dropout_rate: float = 0.2, weight_decay: float = 1e-4) -> None:
        """Initialize MLP classifier.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            weight_decay: L2 regularization strength
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Logits tensor (batch_size, 1)
        """
        return self.network(x)


class MLPTrainer:
    """Trainer for MLP with early stopping."""
    
    def __init__(self, model: MLPClassifier, class_weights: dict[int, float] | None = None,
                 learning_rate: float = 1e-3, device: str = "cpu") -> None:
        """Initialize trainer.
        
        Args:
            model: MLP model to train
            class_weights: Class weights for imbalanced data
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=model.weight_decay
        )
        
        # Setup loss function with class weights
        if class_weights:
            pos_weight = torch.tensor(
                [class_weights[1] / class_weights[0]],
                dtype=torch.float32,
                device=self.device
            )
        else:
            pos_weight = None
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Early stopping parameters
        self.best_val_auc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
    
    def train_epoch(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Train for one epoch.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Training loss
        """
        self.model.train()
        
        # Forward pass
        logits = self.model(X).squeeze()
        loss = self.criterion(logits, y.float())
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, X_val: torch.Tensor, y_val: torch.Tensor) -> tuple[float, float]:
        """Validate model.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (validation loss, validation AUC)
        """
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(X_val).squeeze()
            val_loss = self.criterion(logits, y_val.float()).item()
            
            # Calculate AUC
            probs = torch.sigmoid(logits).cpu().numpy()
            val_auc = roc_auc_score(y_val.cpu().numpy(), probs)
        
        return val_loss, val_auc
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series,
            epochs: int = 100, patience: int = 10, verbose: bool = True) -> dict[str, list[float]]:
        """Train model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long, device=self.device)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32, device=self.device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long, device=self.device)
        
        # Training history
        history = {"train_loss": [], "val_loss": [], "val_auc": []}
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train_tensor, y_train_tensor)
            
            # Validate
            val_loss, val_auc = self.validate(X_val_tensor, y_val_tensor)
            
            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)
            
            # Early stopping check
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}")
            
            # Early stopping
            if self.patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return history
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        self.model.eval()
        
        X_tensor = torch.tensor(X.values, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Return probabilities for both classes
        return np.column_stack([1 - probs, probs])
    
    def save_model(self, path: Path) -> None:
        """Save model state dict.
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "input_dim": self.model.input_dim,
                "hidden_dims": self.model.hidden_dims,
                "dropout_rate": self.model.dropout_rate,
                "weight_decay": self.model.weight_decay
            },
            "best_val_auc": self.best_val_auc
        }
        
        torch.save(save_dict, path)
    
    @classmethod
    def load_model(cls, path: Path, class_weights: dict[int, float] | None = None,
                   device: str = "cpu") -> "MLPTrainer":
        """Load saved model.
        
        Args:
            path: Path to saved model
            class_weights: Class weights for loss function
            device: Device to load on
            
        Returns:
            Loaded MLPTrainer instance
        """
        save_dict = torch.load(path, map_location=device)
        
        # Recreate model
        config = save_dict["model_config"]
        model = MLPClassifier(**config)
        model.load_state_dict(save_dict["model_state_dict"])
        
        # Create trainer
        trainer = cls(model, class_weights=class_weights, device=device)
        trainer.best_val_auc = save_dict["best_val_auc"]
        
        return trainer


def get_baseline_models(class_weights: dict[int, float] | None = None) -> dict[str, Any]:
    """Get baseline sklearn models.
    
    Args:
        class_weights: Class weights for imbalanced data
        
    Returns:
        Dictionary of model name to model instance
    """
    # Convert class weights to sklearn format
    if class_weights:
        sklearn_weights = {0: class_weights[0], 1: class_weights[1]}
    else:
        sklearn_weights = None
    
    models = {
        "logistic_l1": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight=sklearn_weights,
            random_state=42,
            max_iter=1000
        ),
        "logistic_l2": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight=sklearn_weights,
            random_state=42,
            max_iter=1000
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=sklearn_weights,
            random_state=42,
            n_jobs=-1
        )
    }
    
    return models


class CalibratedModel:
    """Wrapper for model calibration using Platt scaling or isotonic regression."""
    
    def __init__(self, base_model: Any, method: str = "platt") -> None:
        """Initialize calibrated model.
        
        Args:
            base_model: Base model to calibrate
            method: Calibration method ('platt' or 'isotonic')
        """
        self.base_model = base_model
        self.method = method
        self.calibrator = None
        self._is_calibrated = False
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        """Fit model and calibrator.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_cal: Calibration features
            y_cal: Calibration labels
        """
        # Fit base model
        self.base_model.fit(X_train, y_train)
        
        # Get uncalibrated predictions
        if hasattr(self.base_model, "predict_proba"):
            uncal_probs = self.base_model.predict_proba(X_cal)[:, 1]
        else:
            # For MLP trainer
            uncal_probs = self.base_model.predict_proba(X_cal)[:, 1]
        
        # Fit calibrator
        if self.method == "platt":
            self.calibrator = LogisticRegressionCV(cv=3, random_state=42)
            self.calibrator.fit(uncal_probs.reshape(-1, 1), y_cal)
        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(uncal_probs, y_cal)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self._is_calibrated = True
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Calibrated probabilities
        """
        if not self._is_calibrated:
            raise ValueError("Model must be fitted before prediction")
        
        # Get uncalibrated predictions
        if hasattr(self.base_model, "predict_proba"):
            uncal_probs = self.base_model.predict_proba(X)[:, 1]
        else:
            uncal_probs = self.base_model.predict_proba(X)[:, 1]
        
        # Apply calibration
        if self.method == "platt":
            cal_probs = self.calibrator.predict_proba(uncal_probs.reshape(-1, 1))[:, 1]
        else:  # isotonic
            cal_probs = self.calibrator.predict(uncal_probs)
        
        # Return probabilities for both classes
        return np.column_stack([1 - cal_probs, cal_probs])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)