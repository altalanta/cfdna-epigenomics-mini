"""Preprocessing utilities for cfDNA epigenomic data."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


class LeakageGuard:
    """Prevents data leakage by tracking train/val/test splits."""
    
    def __init__(self, splits: dict[str, np.ndarray]) -> None:
        """Initialize with predefined splits.
        
        Args:
            splits: Dictionary with 'train', 'val', 'test' indices
        """
        self.splits = splits
        self._fitted_on: str | None = None
    
    def check_split(self, indices: np.ndarray) -> str:
        """Determine which split the indices belong to.
        
        Args:
            indices: Sample indices
            
        Returns:
            Split name ('train', 'val', or 'test')
            
        Raises:
            ValueError: If indices don't match any split exactly
        """
        for split_name, split_indices in self.splits.items():
            if set(indices) == set(split_indices):
                return split_name
        
        raise ValueError(
            f"Indices {indices[:5]}... don't match any known split. "
            "This may indicate data leakage."
        )
    
    def fit_transform_check(self, indices: np.ndarray) -> None:
        """Check that fit_transform is only called on training data.
        
        Args:
            indices: Sample indices being used for fitting
            
        Raises:
            ValueError: If trying to fit on non-training data
        """
        split_name = self.check_split(indices)
        if split_name != "train":
            raise ValueError(
                f"fit_transform called on {split_name} data. "
                "Preprocessing should only be fitted on training data."
            )
        self._fitted_on = split_name
    
    def transform_check(self, indices: np.ndarray) -> None:
        """Check that transform is called after fit_transform.
        
        Args:
            indices: Sample indices being transformed
            
        Raises:
            ValueError: If transform called before fit_transform
        """
        if self._fitted_on is None:
            raise ValueError(
                "transform called before fit_transform. "
                "Must fit on training data first."
            )


class SafeStandardScaler:
    """StandardScaler with leakage protection."""
    
    def __init__(self, guard: LeakageGuard) -> None:
        """Initialize with leakage guard.
        
        Args:
            guard: LeakageGuard instance to prevent data leakage
        """
        self.guard = guard
        self.scaler = StandardScaler()
        self._feature_names: list[str] | None = None
    
    def fit_transform(self, X: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
        """Fit scaler on training data and transform.
        
        Args:
            X: Feature matrix
            indices: Sample indices (must be training indices)
            
        Returns:
            Scaled feature matrix
        """
        self.guard.fit_transform_check(indices)
        
        X_subset = X.iloc[indices]
        self._feature_names = X_subset.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X_subset)
        
        return pd.DataFrame(
            X_scaled, 
            index=X_subset.index, 
            columns=self._feature_names
        )
    
    def transform(self, X: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
        """Transform data using fitted scaler.
        
        Args:
            X: Feature matrix
            indices: Sample indices
            
        Returns:
            Scaled feature matrix
        """
        self.guard.transform_check(indices)
        
        X_subset = X.iloc[indices]
        X_scaled = self.scaler.transform(X_subset)
        
        return pd.DataFrame(
            X_scaled,
            index=X_subset.index,
            columns=self._feature_names
        )


class SafePCA:
    """PCA with leakage protection for methylation features."""
    
    def __init__(self, guard: LeakageGuard, n_components: int = 50, 
                 feature_prefix: str = "dmr_") -> None:
        """Initialize PCA with leakage guard.
        
        Args:
            guard: LeakageGuard instance
            n_components: Number of PCA components
            feature_prefix: Prefix to identify features for PCA
        """
        self.guard = guard
        self.n_components = n_components
        self.feature_prefix = feature_prefix
        self.pca = PCA(n_components=n_components)
        self._methylation_features: list[str] = []
        self._other_features: list[str] = []
    
    def fit_transform(self, X: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
        """Fit PCA on methylation features and transform.
        
        Args:
            X: Feature matrix
            indices: Sample indices (must be training indices)
            
        Returns:
            Feature matrix with PCA-transformed methylation features
        """
        self.guard.fit_transform_check(indices)
        
        X_subset = X.iloc[indices]
        
        # Separate methylation and other features
        self._methylation_features = [
            col for col in X_subset.columns 
            if col.startswith(self.feature_prefix)
        ]
        self._other_features = [
            col for col in X_subset.columns 
            if not col.startswith(self.feature_prefix)
        ]
        
        if len(self._methylation_features) == 0:
            # No methylation features to transform
            return X_subset
        
        # Apply PCA to methylation features
        X_meth = X_subset[self._methylation_features]
        X_meth_pca = self.pca.fit_transform(X_meth)
        
        # Create PCA feature names
        pca_names = [f"meth_pc_{i+1}" for i in range(self.n_components)]
        X_meth_pca_df = pd.DataFrame(
            X_meth_pca, 
            index=X_subset.index, 
            columns=pca_names
        )
        
        # Combine with other features
        if len(self._other_features) > 0:
            X_other = X_subset[self._other_features]
            X_transformed = pd.concat([X_meth_pca_df, X_other], axis=1)
        else:
            X_transformed = X_meth_pca_df
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
        """Transform data using fitted PCA.
        
        Args:
            X: Feature matrix
            indices: Sample indices
            
        Returns:
            Feature matrix with PCA-transformed methylation features
        """
        self.guard.transform_check(indices)
        
        X_subset = X.iloc[indices]
        
        if len(self._methylation_features) == 0:
            return X_subset
        
        # Apply PCA to methylation features
        X_meth = X_subset[self._methylation_features]
        X_meth_pca = self.pca.transform(X_meth)
        
        # Create PCA feature names
        pca_names = [f"meth_pc_{i+1}" for i in range(self.n_components)]
        X_meth_pca_df = pd.DataFrame(
            X_meth_pca,
            index=X_subset.index,
            columns=pca_names
        )
        
        # Combine with other features
        if len(self._other_features) > 0:
            X_other = X_subset[self._other_features]
            X_transformed = pd.concat([X_meth_pca_df, X_other], axis=1)
        else:
            X_transformed = X_meth_pca_df
        
        return X_transformed


def compute_class_weights(y: pd.Series, indices: np.ndarray) -> dict[int, float]:
    """Compute class weights for imbalanced datasets.
    
    Args:
        y: Target labels
        indices: Sample indices to compute weights on
        
    Returns:
        Dictionary mapping class labels to weights
    """
    y_subset = y.iloc[indices]
    classes = np.unique(y_subset)
    weights = compute_class_weight("balanced", classes=classes, y=y_subset)
    return dict(zip(classes, weights))


def get_stratified_group_kfold(n_splits: int = 5) -> StratifiedGroupKFold:
    """Get StratifiedGroupKFold for nested cross-validation.
    
    Args:
        n_splits: Number of folds
        
    Returns:
        StratifiedGroupKFold instance
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)


class PreprocessingPipeline:
    """Complete preprocessing pipeline with leakage protection."""
    
    def __init__(self, splits: dict[str, np.ndarray], config: dict[str, Any] | None = None) -> None:
        """Initialize preprocessing pipeline.
        
        Args:
            splits: Train/val/test splits
            config: Configuration options
        """
        self.guard = LeakageGuard(splits)
        self.config = config or {}
        
        # Initialize components
        self.scaler = SafeStandardScaler(self.guard)
        
        # PCA is optional
        pca_config = self.config.get("pca", {})
        if pca_config.get("enabled", False):
            self.pca = SafePCA(
                self.guard,
                n_components=pca_config.get("n_components", 50),
                feature_prefix=pca_config.get("feature_prefix", "dmr_")
            )
        else:
            self.pca = None
    
    def fit_transform_train(self, X: pd.DataFrame, splits: dict[str, np.ndarray]) -> pd.DataFrame:
        """Fit preprocessing on training data and transform.
        
        Args:
            X: Feature matrix
            splits: Dictionary with train/val/test indices
            
        Returns:
            Preprocessed training data
        """
        train_indices = splits["train"]
        
        # Apply PCA if enabled
        if self.pca is not None:
            X_train = self.pca.fit_transform(X, train_indices)
        else:
            X_train = X.iloc[train_indices]
        
        # Apply standardization
        X_train_scaled = self.scaler.fit_transform(X_train, np.arange(len(X_train)))
        
        return X_train_scaled
    
    def transform_split(self, X: pd.DataFrame, split_name: str, 
                       splits: dict[str, np.ndarray]) -> pd.DataFrame:
        """Transform a specific data split.
        
        Args:
            X: Feature matrix
            split_name: Name of split ('val' or 'test')
            splits: Dictionary with train/val/test indices
            
        Returns:
            Preprocessed data for the specified split
        """
        split_indices = splits[split_name]
        
        # Apply PCA if enabled
        if self.pca is not None:
            X_split = self.pca.transform(X, split_indices)
        else:
            X_split = X.iloc[split_indices]
        
        # Apply standardization
        X_split_scaled = self.scaler.transform(X_split, np.arange(len(X_split)))
        
        return X_split_scaled


def why_no_smote() -> str:
    """Explain why SMOTE is not used in this pipeline.
    
    Returns:
        Explanation string
    """
    return (
        "SMOTE is not used in this pipeline for several reasons:\n"
        "1. cfDNA data is high-dimensional and sparse, making synthetic sample generation problematic\n"
        "2. Biological relationships between features are complex and may not be preserved\n"
        "3. Class weights provide a simpler, more interpretable approach to handling imbalance\n"
        "4. In clinical settings, synthetic samples can complicate regulatory approval\n"
        "5. The mild class imbalance (58/42 split) is manageable with class weighting"
    )