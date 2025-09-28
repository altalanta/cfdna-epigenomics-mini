"""Tests for cfDNA feature engineering module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfdna.features import (
    aggregate_dmr_features,
    combat_adjust,
    create_splits,
    load_dataset,
    merge_features,
    prepare_features,
)


@pytest.fixture
def sample_methylation_data():
    """Sample methylation data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_cpgs = 300
    
    # Create sample data with some structure
    data = np.random.beta(2, 8, size=(n_samples, n_cpgs))
    
    # Add some missing values
    missing_mask = np.random.random((n_samples, n_cpgs)) < 0.05
    data[missing_mask] = np.nan
    
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]
    cpg_ids = [f"cg{i:05d}" for i in range(n_cpgs)]
    
    return pd.DataFrame(data, index=sample_ids, columns=cpg_ids)


@pytest.fixture
def sample_fragmentomics_data():
    """Sample fragmentomics data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 10
    
    # Create sample data
    data = np.random.exponential(1.0, size=(n_samples, n_features))
    
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]
    feature_names = [f"frag_feature_{i}" for i in range(n_features)]
    
    return pd.DataFrame(data, index=sample_ids, columns=feature_names)


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    np.random.seed(42)
    n_samples = 50
    
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]
    subject_ids = [f"subj_{i:03d}" for i in range(n_samples)]
    
    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "subject_id": subject_ids,
        "age": np.random.uniform(40, 80, n_samples),
        "sex": np.random.choice(["F", "M"], n_samples),
        "batch": np.random.choice([0, 1, 2], n_samples),
        "center": np.random.choice(["site_a", "site_b"], n_samples),
        "label": np.random.choice([0, 1], n_samples)
    })
    
    return metadata.set_index("sample_id")


def test_aggregate_dmr_features_shape(sample_methylation_data):
    """Test that DMR aggregation produces correct shape."""
    n_dmrs = 3
    cpgs_per_dmr = 100
    
    dmr_features = aggregate_dmr_features(
        sample_methylation_data, n_dmrs=n_dmrs, cpgs_per_dmr=cpgs_per_dmr
    )
    
    # Should have 3 features per DMR (mean, z-score, variance)
    expected_features = n_dmrs * 3
    assert dmr_features.shape == (50, expected_features), f"Expected shape (50, {expected_features}), got {dmr_features.shape}"
    
    # Check feature names
    expected_cols = []
    for i in range(n_dmrs):
        expected_cols.extend([
            f"dmr_{i:03d}_mean",
            f"dmr_{i:03d}_z", 
            f"dmr_{i:03d}_var"
        ])
    
    assert list(dmr_features.columns) == expected_cols


def test_aggregate_dmr_features_no_leakage(sample_methylation_data):
    """Test that DMR aggregation handles missing values correctly."""
    # Create data with systematic missing values
    X_meth = sample_methylation_data.copy()
    
    # Make first 10 samples missing first 50 CpGs
    X_meth.iloc[:10, :50] = np.nan
    
    dmr_features = aggregate_dmr_features(X_meth, n_dmrs=2, cpgs_per_dmr=100)
    
    # Should still produce results without errors
    assert not dmr_features.isnull().all().any(), "Some features are completely missing"
    
    # Check that mean features are reasonable
    mean_cols = [col for col in dmr_features.columns if "_mean" in col]
    for col in mean_cols:
        values = dmr_features[col].dropna()
        assert len(values) > 0, f"No valid values for {col}"
        assert 0 <= values.min() <= values.max() <= 1, f"Invalid range for {col}"


def test_combat_adjust_basic(sample_methylation_data, sample_metadata):
    """Test basic ComBat adjustment functionality."""
    # Align data
    common_samples = sample_methylation_data.index.intersection(sample_metadata.index)
    X_meth = sample_methylation_data.loc[common_samples]
    batch = sample_metadata.loc[common_samples, "batch"]
    
    # Apply adjustment
    X_adj = combat_adjust(X_meth, batch)
    
    # Check shape is preserved
    assert X_adj.shape == X_meth.shape
    
    # Check that adjustment doesn't introduce extreme values
    valid_orig = X_meth.values[~np.isnan(X_meth.values)]
    valid_adj = X_adj.values[~np.isnan(X_adj.values)]
    
    # Adjusted values should be in reasonable range
    assert valid_adj.min() >= -1, "Adjustment produced extreme negative values"
    assert valid_adj.max() <= 2, "Adjustment produced extreme positive values"


def test_merge_features_basic(sample_methylation_data, sample_fragmentomics_data, sample_metadata):
    """Test basic feature merging."""
    # Aggregate methylation features first
    dmr_features = aggregate_dmr_features(sample_methylation_data, n_dmrs=2, cpgs_per_dmr=100)
    
    # Merge features
    merged = merge_features(dmr_features, sample_fragmentomics_data, adjust_batch=False)
    
    # Check shape
    expected_features = dmr_features.shape[1] + sample_fragmentomics_data.shape[1]
    assert merged.shape[1] == expected_features, f"Expected {expected_features} features, got {merged.shape[1]}"
    
    # Check that no samples are lost unnecessarily
    expected_samples = min(dmr_features.shape[0], sample_fragmentomics_data.shape[0])
    assert merged.shape[0] == expected_samples


def test_merge_features_with_batch_adjustment(sample_methylation_data, sample_fragmentomics_data, sample_metadata):
    """Test feature merging with batch adjustment."""
    # Aggregate methylation features first
    dmr_features = aggregate_dmr_features(sample_methylation_data, n_dmrs=2, cpgs_per_dmr=100)
    
    # Merge with batch adjustment
    merged = merge_features(
        dmr_features, 
        sample_fragmentomics_data, 
        adjust_batch=True, 
        metadata=sample_metadata
    )
    
    # Should produce valid results
    assert not merged.isnull().all().any(), "Batch adjustment produced all-missing features"
    
    # Check that missing values are handled
    assert not merged.isnull().any().any(), "Missing values not properly imputed"


def test_create_splits_basic(sample_metadata):
    """Test basic split creation."""
    y = sample_metadata["label"]
    
    splits = create_splits(y, sample_metadata, test_size=0.2, val_size=0.2, random_state=42)
    
    # Check that all indices are covered
    all_indices = set(range(len(y)))
    split_indices = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert split_indices == all_indices, "Not all samples assigned to splits"
    
    # Check that splits don't overlap
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    
    assert len(train_set & val_set) == 0, "Train and validation sets overlap"
    assert len(train_set & test_set) == 0, "Train and test sets overlap"
    assert len(val_set & test_set) == 0, "Validation and test sets overlap"
    
    # Check approximate sizes
    n_total = len(y)
    assert len(splits["test"]) == pytest.approx(n_total * 0.2, abs=2)
    assert len(splits["val"]) == pytest.approx(n_total * 0.2 * 0.8, abs=2)  # 20% of remaining 80%


def test_create_splits_stratified(sample_metadata):
    """Test that splits are stratified by batch and label."""
    y = sample_metadata["label"]
    
    splits = create_splits(y, sample_metadata, test_size=0.2, val_size=0.2, random_state=42)
    
    # Check that each split has representation from each batch
    for split_name, indices in splits.items():
        split_metadata = sample_metadata.iloc[indices]
        split_batches = set(split_metadata["batch"])
        all_batches = set(sample_metadata["batch"])
        
        # Most splits should have multiple batches (unless very small)
        if len(indices) > 5:
            assert len(split_batches) > 1, f"Split {split_name} has only batch {split_batches}"


def test_create_splits_reproducible(sample_metadata):
    """Test that splits are reproducible with same random state."""
    y = sample_metadata["label"]
    
    splits1 = create_splits(y, sample_metadata, random_state=42)
    splits2 = create_splits(y, sample_metadata, random_state=42)
    
    # Should be identical
    for split_name in ["train", "val", "test"]:
        np.testing.assert_array_equal(splits1[split_name], splits2[split_name])


def test_prepare_features_integration():
    """Test the complete feature preparation pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        
        # Create synthetic data files
        np.random.seed(42)
        n_samples = 100
        
        # Methylation data
        X_meth = pd.DataFrame(
            np.random.beta(2, 8, size=(n_samples, 1000)),
            index=[f"sample_{i:04d}" for i in range(n_samples)],
            columns=[f"cg{i:05d}" for i in range(1000)]
        )
        X_meth.to_parquet(data_dir / "X_meth.parquet")
        
        # Fragmentomics data
        X_frag = pd.DataFrame(
            np.random.exponential(1.0, size=(n_samples, 10)),
            index=[f"sample_{i:04d}" for i in range(n_samples)],
            columns=[f"frag_{i}" for i in range(10)]
        )
        X_frag.to_parquet(data_dir / "X_frag.parquet")
        
        # Labels
        y_df = pd.DataFrame({
            "sample_id": [f"sample_{i:04d}" for i in range(n_samples)],
            "label": np.random.choice([0, 1], n_samples)
        })
        y_df.to_csv(data_dir / "y.csv", index=False)
        
        # Metadata
        metadata_df = pd.DataFrame({
            "sample_id": [f"sample_{i:04d}" for i in range(n_samples)],
            "subject_id": [f"subj_{i:04d}" for i in range(n_samples)],
            "age": np.random.uniform(40, 80, n_samples),
            "sex": np.random.choice(["F", "M"], n_samples),
            "batch": np.random.choice([0, 1, 2], n_samples),
            "center": np.random.choice(["site_a", "site_b"], n_samples),
            "label": y_df["label"]
        })
        metadata_df.to_csv(data_dir / "metadata.csv", index=False)
        
        # Test feature preparation
        config = {
            "methylation": {"n_dmrs": 5, "cpgs_per_dmr": 200},
            "splits": {"test_size": 0.2, "val_size": 0.2, "random_state": 42}
        }
        
        result = prepare_features(data_dir, config)
        
        # Check output structure
        assert "X" in result
        assert "y" in result
        assert "metadata" in result
        assert "splits" in result
        assert "feature_names" in result
        
        # Check feature matrix
        X = result["X"]
        assert X.shape[0] == n_samples
        assert X.shape[1] > 0  # Should have some features
        
        # Check splits
        splits = result["splits"]
        assert set(splits.keys()) == {"train", "val", "test"}
        
        # Check feature names
        feature_names = result["feature_names"]
        assert "methylation" in feature_names
        assert "fragmentomics" in feature_names