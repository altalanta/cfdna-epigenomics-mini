"""Feature engineering for cfDNA epigenomic data."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the complete cfDNA dataset.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (methylation_df, fragmentomics_df, labels, metadata)
    """
    data_dir = Path(data_dir)
    
    X_meth = pd.read_parquet(data_dir / "X_meth.parquet")
    X_frag = pd.read_parquet(data_dir / "X_frag.parquet")
    y_df = pd.read_csv(data_dir / "y.csv")
    metadata = pd.read_csv(data_dir / "metadata.csv")
    
    # Ensure consistent sample ordering
    sample_order = X_meth.index.tolist()
    X_frag = X_frag.reindex(sample_order)
    y_df = y_df.set_index("sample_id").reindex(sample_order)
    metadata = metadata.set_index("sample_id").reindex(sample_order)
    
    y = y_df["label"]
    
    return X_meth, X_frag, y, metadata


def aggregate_dmr_features(X_meth: pd.DataFrame, n_dmrs: int = 200, cpgs_per_dmr: int = 100) -> pd.DataFrame:
    """Aggregate CpG sites into Differential Methylation Region (DMR) features.
    
    Args:
        X_meth: Methylation dataframe (samples x CpGs)
        n_dmrs: Number of DMRs to create
        cpgs_per_dmr: Number of CpGs per DMR
        
    Returns:
        DMR feature dataframe (samples x DMRs)
    """
    n_cpgs = X_meth.shape[1]
    dmr_features = {}
    
    for dmr_idx in range(n_dmrs):
        start_idx = dmr_idx * cpgs_per_dmr
        end_idx = min(start_idx + cpgs_per_dmr, n_cpgs)
        
        if start_idx >= n_cpgs:
            break
            
        dmr_cpgs = X_meth.iloc[:, start_idx:end_idx]
        
        # Calculate mean methylation (handling missing values)
        dmr_mean = dmr_cpgs.mean(axis=1, skipna=True)
        
        # Calculate z-score within each DMR
        dmr_std = dmr_cpgs.std(axis=1, skipna=True)
        dmr_z = (dmr_mean - dmr_mean.mean()) / (dmr_mean.std() + 1e-8)
        
        dmr_features[f"dmr_{dmr_idx:03d}_mean"] = dmr_mean
        dmr_features[f"dmr_{dmr_idx:03d}_z"] = dmr_z
        
        # Add variability measure
        dmr_features[f"dmr_{dmr_idx:03d}_var"] = dmr_std
    
    return pd.DataFrame(dmr_features, index=X_meth.index)


def combat_adjust(X: pd.DataFrame, batch: pd.Series, preserve_groups: bool = True) -> pd.DataFrame:
    """Simple ComBat-like batch adjustment using residuals.
    
    Args:
        X: Feature matrix (samples x features)
        batch: Batch assignments for each sample
        preserve_groups: Whether to preserve group structure
        
    Returns:
        Batch-adjusted feature matrix
    """
    X_adj = X.copy()
    
    for feature in X.columns:
        feature_data = X[feature].dropna()
        if len(feature_data) < 10:  # Skip features with too few observations
            continue
            
        # Fit batch effects using linear model
        batch_dummies = pd.get_dummies(batch.loc[feature_data.index], prefix="batch")
        
        if batch_dummies.shape[1] > 1:  # More than one batch
            # Calculate batch means
            batch_means = {}
            overall_mean = feature_data.mean()
            
            for batch_col in batch_dummies.columns:
                batch_mask = batch_dummies[batch_col] == 1
                batch_mean = feature_data[batch_mask].mean()
                batch_means[batch_col] = batch_mean - overall_mean
            
            # Subtract batch effects
            for idx in feature_data.index:
                sample_batch = batch.loc[idx]
                batch_col = f"batch_{sample_batch}"
                if batch_col in batch_means:
                    X_adj.loc[idx, feature] = feature_data.loc[idx] - batch_means[batch_col]
    
    return X_adj


def merge_features(X_meth: pd.DataFrame, X_frag: pd.DataFrame, adjust_batch: bool = True, 
                  metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge methylation and fragmentomics features.
    
    Args:
        X_meth: Methylation features (DMRs)
        X_frag: Fragmentomics features  
        adjust_batch: Whether to apply batch adjustment
        metadata: Sample metadata (required if adjust_batch=True)
        
    Returns:
        Combined feature matrix
    """
    # Ensure same sample order
    common_samples = X_meth.index.intersection(X_frag.index)
    X_meth_aligned = X_meth.loc[common_samples]
    X_frag_aligned = X_frag.loc[common_samples]
    
    # Apply batch adjustment if requested
    if adjust_batch and metadata is not None:
        metadata_aligned = metadata.loc[common_samples]
        batch = metadata_aligned["batch"]
        
        X_meth_aligned = combat_adjust(X_meth_aligned, batch)
        X_frag_aligned = combat_adjust(X_frag_aligned, batch)
    
    # Concatenate features
    X_combined = pd.concat([X_meth_aligned, X_frag_aligned], axis=1)
    
    # Handle remaining missing values with median imputation
    X_combined = X_combined.fillna(X_combined.median())
    
    return X_combined


def create_splits(y: pd.Series, metadata: pd.DataFrame, test_size: float = 0.2, 
                 val_size: float = 0.2, random_state: int = 42) -> dict[str, np.ndarray]:
    """Create train/val/test splits that are group-aware and batch-balanced.
    
    Args:
        y: Target labels
        metadata: Sample metadata with subject_id and batch
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' indices
    """
    # Create stratification groups combining label and batch
    strat_groups = metadata["batch"].astype(str) + "_" + y.astype(str)
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        range(len(y)),
        test_size=test_size,
        stratify=strat_groups,
        random_state=random_state
    )
    
    # Second split: train vs val
    y_train_val = y.iloc[train_val_idx]
    metadata_train_val = metadata.iloc[train_val_idx]
    strat_groups_train_val = (metadata_train_val["batch"].astype(str) + "_" + 
                             y_train_val.astype(str))
    
    train_idx_local, val_idx_local = train_test_split(
        range(len(train_val_idx)),
        test_size=val_size,
        stratify=strat_groups_train_val,
        random_state=random_state
    )
    
    # Convert back to original indices
    train_idx = np.array([train_val_idx[i] for i in train_idx_local])
    val_idx = np.array([train_val_idx[i] for i in val_idx_local])
    test_idx = np.array(test_idx)
    
    return {
        "train": train_idx,
        "val": val_idx, 
        "test": test_idx
    }


def prepare_features(data_dir: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Complete feature preparation pipeline.
    
    Args:
        data_dir: Directory containing raw data files
        config: Optional configuration overrides
        
    Returns:
        Dictionary containing processed features and splits
    """
    # Load raw data
    X_meth, X_frag, y, metadata = load_dataset(data_dir)
    
    # Extract DMR features from methylation data
    dmr_config = config.get("methylation", {}) if config else {}
    n_dmrs = dmr_config.get("n_dmrs", 200)
    cpgs_per_dmr = dmr_config.get("cpgs_per_dmr", 100)
    
    X_meth_dmr = aggregate_dmr_features(X_meth, n_dmrs=n_dmrs, cpgs_per_dmr=cpgs_per_dmr)
    
    # Merge and adjust features
    X_features = merge_features(X_meth_dmr, X_frag, adjust_batch=True, metadata=metadata)
    
    # Create splits
    split_config = config.get("splits", {}) if config else {}
    splits = create_splits(
        y, 
        metadata,
        test_size=split_config.get("test_size", 0.2),
        val_size=split_config.get("val_size", 0.2),
        random_state=split_config.get("random_state", 42)
    )
    
    return {
        "X": X_features,
        "y": y,
        "metadata": metadata,
        "splits": splits,
        "feature_names": {
            "methylation": [col for col in X_features.columns if col.startswith("dmr_")],
            "fragmentomics": [col for col in X_features.columns if not col.startswith("dmr_")]
        }
    }