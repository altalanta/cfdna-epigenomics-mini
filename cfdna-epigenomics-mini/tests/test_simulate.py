"""Tests for cfDNA data simulation module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cfdna.simulate import simulate_dataset


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "dataset": {
            "name": "test_dataset",
            "n_samples": 100,
            "n_controls": 60,
            "n_cancer": 40,
            "random_seed": 42
        },
        "samples": {
            "n_batches": 2,
            "centers": ["site_a", "site_b"],
            "age_range": [40, 80],
            "sex_ratio": 0.5
        },
        "methylation": {
            "n_cpgs": 1000,
            "n_dmrs": 10,
            "cpgs_per_dmr": 100,
            "dmr_effect_size_mean": 0.15,
            "dmr_effect_size_std": 0.05,
            "alpha_base": 2.0,
            "beta_base": 8.0,
            "batch_effect_std": 0.02,
            "missingness_rate": 0.05
        },
        "fragmentomics": {
            "size_bins": [100, 200, 300, 400],
            "tss_enrichment_bins": 5,
            "size_effect_mean": 0.1,
            "size_effect_std": 0.03,
            "tss_effect_mean": 0.08,
            "tss_effect_std": 0.02,
            "noise_std": 0.05
        }
    }


def test_simulate_dataset_basic(sample_config):
    """Test basic dataset simulation functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        
        output_dir = Path(temp_dir) / "output"
        
        # Run simulation
        stats = simulate_dataset(config_path, output_dir)
        
        # Check return statistics
        assert stats["n_samples"] == 100
        assert stats["n_controls"] == 60
        assert stats["n_cancer"] == 40
        assert stats["n_cpgs"] == 1000
        assert stats["n_frag_features"] == 9  # 4 size bins + 5 TSS bins
        
        # Check files were created
        expected_files = [
            "X_meth.parquet",
            "X_frag.parquet",
            "y.csv",
            "metadata.csv"
        ]
        
        for filename in expected_files:
            file_path = output_dir / filename
            assert file_path.exists(), f"Expected file {filename} not found"
            assert file_path.stat().st_size > 0, f"File {filename} is empty"


def test_simulate_dataset_shapes(sample_config):
    """Test that generated data has correct shapes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        
        output_dir = Path(temp_dir) / "output"
        simulate_dataset(config_path, output_dir)
        
        # Load and check methylation data
        X_meth = pd.read_parquet(output_dir / "X_meth.parquet")
        assert X_meth.shape == (100, 1000), f"Unexpected methylation shape: {X_meth.shape}"
        
        # Check methylation values are in valid range (excluding NaN)
        meth_values = X_meth.values
        valid_values = meth_values[~np.isnan(meth_values)]
        assert np.all(valid_values >= 0.01), "Methylation values below 0.01"
        assert np.all(valid_values <= 0.99), "Methylation values above 0.99"
        
        # Load and check fragmentomics data
        X_frag = pd.read_parquet(output_dir / "X_frag.parquet") 
        assert X_frag.shape == (100, 9), f"Unexpected fragmentomics shape: {X_frag.shape}"
        
        # Check fragmentomics values are positive
        assert np.all(X_frag.values >= 0), "Negative fragmentomics values found"
        
        # Load and check labels
        y = pd.read_csv(output_dir / "y.csv")
        assert len(y) == 100, f"Unexpected label count: {len(y)}"
        assert set(y["label"].unique()) == {0, 1}, "Invalid label values"
        
        # Load and check metadata
        metadata = pd.read_csv(output_dir / "metadata.csv")
        assert len(metadata) == 100, f"Unexpected metadata count: {len(metadata)}"


def test_simulate_dataset_class_counts(sample_config):
    """Test that class distribution matches configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        
        output_dir = Path(temp_dir) / "output"
        simulate_dataset(config_path, output_dir)
        
        # Check class distribution
        y = pd.read_csv(output_dir / "y.csv")
        class_counts = y["label"].value_counts().sort_index()
        
        assert class_counts[0] == 60, f"Expected 60 controls, got {class_counts[0]}"
        assert class_counts[1] == 40, f"Expected 40 cancer cases, got {class_counts[1]}"


def test_simulate_dataset_batch_labels(sample_config):
    """Test that batch labels are present and valid."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        
        output_dir = Path(temp_dir) / "output"
        simulate_dataset(config_path, output_dir)
        
        # Check metadata
        metadata = pd.read_csv(output_dir / "metadata.csv")
        
        # Check batch assignments
        batch_values = metadata["batch"].unique()
        expected_batches = list(range(sample_config["samples"]["n_batches"]))
        assert set(batch_values) == set(expected_batches), f"Unexpected batch values: {batch_values}"
        
        # Check center assignments
        center_values = metadata["center"].unique()
        expected_centers = sample_config["samples"]["centers"]
        assert set(center_values).issubset(set(expected_centers)), f"Unexpected center values: {center_values}"
        
        # Check age range
        ages = metadata["age"]
        min_age, max_age = sample_config["samples"]["age_range"]
        assert ages.min() >= min_age, f"Age below minimum: {ages.min()}"
        assert ages.max() <= max_age, f"Age above maximum: {ages.max()}"
        
        # Check sex values
        sex_values = metadata["sex"].unique()
        assert set(sex_values).issubset({"F", "M"}), f"Invalid sex values: {sex_values}"


def test_simulate_dataset_reproducibility(sample_config):
    """Test that simulation is reproducible with same seed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        
        # Run simulation twice
        output_dir1 = Path(temp_dir) / "output1"
        output_dir2 = Path(temp_dir) / "output2"
        
        simulate_dataset(config_path, output_dir1)
        simulate_dataset(config_path, output_dir2)
        
        # Compare methylation data
        X_meth1 = pd.read_parquet(output_dir1 / "X_meth.parquet")
        X_meth2 = pd.read_parquet(output_dir2 / "X_meth.parquet")
        
        # Should be identical (accounting for NaN)
        pd.testing.assert_frame_equal(X_meth1, X_meth2, check_exact=False, rtol=1e-10)
        
        # Compare fragmentomics data
        X_frag1 = pd.read_parquet(output_dir1 / "X_frag.parquet")
        X_frag2 = pd.read_parquet(output_dir2 / "X_frag.parquet")
        
        pd.testing.assert_frame_equal(X_frag1, X_frag2, check_exact=False, rtol=1e-10)
        
        # Compare labels
        y1 = pd.read_csv(output_dir1 / "y.csv")
        y2 = pd.read_csv(output_dir2 / "y.csv")
        
        pd.testing.assert_frame_equal(y1, y2)


def test_simulate_dataset_different_seeds(sample_config):
    """Test that different seeds produce different results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create configs with different seeds
        config1 = sample_config.copy()
        config1["dataset"]["random_seed"] = 42
        
        config2 = sample_config.copy()
        config2["dataset"]["random_seed"] = 123
        
        config_path1 = Path(temp_dir) / "config1.yaml"
        config_path2 = Path(temp_dir) / "config2.yaml"
        
        with open(config_path1, "w") as f:
            yaml.dump(config1, f)
        with open(config_path2, "w") as f:
            yaml.dump(config2, f)
        
        # Run simulations
        output_dir1 = Path(temp_dir) / "output1"
        output_dir2 = Path(temp_dir) / "output2"
        
        simulate_dataset(config_path1, output_dir1)
        simulate_dataset(config_path2, output_dir2)
        
        # Compare methylation data - should be different
        X_meth1 = pd.read_parquet(output_dir1 / "X_meth.parquet")
        X_meth2 = pd.read_parquet(output_dir2 / "X_meth.parquet")
        
        # Should NOT be identical
        try:
            pd.testing.assert_frame_equal(X_meth1, X_meth2, check_exact=False, rtol=1e-6)
            pytest.fail("Expected different results with different seeds")
        except AssertionError:
            pass  # This is expected


def test_simulate_dataset_missing_values(sample_config):
    """Test that missing values are generated as expected."""
    # Increase missingness rate for testing
    config = sample_config.copy()
    config["methylation"]["missingness_rate"] = 0.2  # 20% missing
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        output_dir = Path(temp_dir) / "output"
        simulate_dataset(config_path, output_dir)
        
        # Check methylation missing values
        X_meth = pd.read_parquet(output_dir / "X_meth.parquet")
        missing_rate = X_meth.isnull().sum().sum() / (X_meth.shape[0] * X_meth.shape[1])
        
        # Should have some missing values (exact rate varies due to batch effects)
        assert missing_rate > 0.1, f"Expected >10% missing values, got {missing_rate:.3f}"
        assert missing_rate < 0.4, f"Too many missing values: {missing_rate:.3f}"