"""Synthetic cfDNA epigenomic data simulation."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def simulate_dataset(config_path: Path, out_dir: Path) -> dict[str, Any]:
    """Generate synthetic cfDNA epigenomic dataset.
    
    Args:
        config_path: Path to YAML configuration file
        out_dir: Output directory for generated files
        
    Returns:
        Dictionary with dataset statistics and file paths
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    seed = config["dataset"]["random_seed"]
    np.random.seed(seed)
    
    # Generate sample metadata
    n_samples = config["dataset"]["n_samples"]
    n_controls = config["dataset"]["n_controls"]
    n_cancer = config["dataset"]["n_cancer"]
    
    # Subject-level metadata
    subject_ids = [f"subj_{i:04d}" for i in range(n_samples)]
    ages = np.random.uniform(
        config["samples"]["age_range"][0],
        config["samples"]["age_range"][1],
        n_samples
    )
    sexes = np.random.choice(
        ["F", "M"], 
        n_samples, 
        p=[config["samples"]["sex_ratio"], 1 - config["samples"]["sex_ratio"]]
    )
    
    # Sample-level metadata
    sample_ids = [f"sample_{i:04d}" for i in range(n_samples)]
    batches = np.random.choice(
        range(config["samples"]["n_batches"]), 
        n_samples
    )
    centers = np.random.choice(
        config["samples"]["centers"], 
        n_samples
    )
    
    # Labels (0=control, 1=cancer)
    labels = np.concatenate([
        np.zeros(n_controls, dtype=int),
        np.ones(n_cancer, dtype=int)
    ])
    np.random.shuffle(labels)
    
    # Create metadata dataframe
    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "subject_id": subject_ids,
        "age": ages,
        "sex": sexes,
        "batch": batches,
        "center": centers,
        "label": labels
    })
    
    # Generate methylation data
    meth_data = _simulate_methylation(config, labels, batches)
    
    # Generate fragmentomics data  
    frag_data = _simulate_fragmentomics(config, labels, batches)
    
    # Save data
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    
    meth_data.to_parquet(out_dir / "X_meth.parquet")
    frag_data.to_parquet(out_dir / "X_frag.parquet")
    
    # Create labels and metadata files
    y_df = pd.DataFrame({"sample_id": sample_ids, "label": labels})
    y_df.to_csv(out_dir / "y.csv", index=False)
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    
    # Return summary statistics
    return {
        "n_samples": n_samples,
        "n_controls": n_controls, 
        "n_cancer": n_cancer,
        "n_cpgs": meth_data.shape[1],
        "n_frag_features": frag_data.shape[1],
        "batch_distribution": metadata.groupby("batch")["label"].value_counts().to_dict(),
        "files_created": [
            str(out_dir / "X_meth.parquet"),
            str(out_dir / "X_frag.parquet"), 
            str(out_dir / "y.csv"),
            str(out_dir / "metadata.csv")
        ]
    }


def _simulate_methylation(config: dict[str, Any], labels: np.ndarray, batches: np.ndarray) -> pd.DataFrame:
    """Simulate methylation beta values with DMR effects."""
    meth_config = config["methylation"]
    n_samples = len(labels)
    n_cpgs = meth_config["n_cpgs"]
    n_dmrs = meth_config["n_dmrs"]
    cpgs_per_dmr = meth_config["cpgs_per_dmr"]
    
    # Initialize beta values matrix
    beta_values = np.zeros((n_samples, n_cpgs))
    
    # Base beta-binomial parameters
    alpha_base = meth_config["alpha_base"]
    beta_base = meth_config["beta_base"]
    
    # Assign CpGs to DMRs
    dmr_assignments = np.zeros(n_cpgs, dtype=int)
    dmr_start = 0
    for dmr_idx in range(n_dmrs):
        if dmr_start + cpgs_per_dmr <= n_cpgs:
            dmr_assignments[dmr_start:dmr_start + cpgs_per_dmr] = dmr_idx
            dmr_start += cpgs_per_dmr
    
    # Generate DMR effect sizes for cancer
    dmr_effects = np.random.normal(
        meth_config["dmr_effect_size_mean"],
        meth_config["dmr_effect_size_std"],
        n_dmrs
    )
    
    # Generate beta values for each CpG
    for cpg_idx in range(n_cpgs):
        dmr_idx = dmr_assignments[cpg_idx]
        
        # Base methylation level
        alpha = alpha_base
        beta = beta_base
        
        # Add cancer effect for samples with cancer
        cancer_mask = labels == 1
        if dmr_idx < n_dmrs:  # CpG is in a DMR
            alpha_cancer = alpha + dmr_effects[dmr_idx]
            alpha_array = np.where(cancer_mask, alpha_cancer, alpha)
        else:
            alpha_array = np.full(n_samples, alpha)
        
        # Generate beta values using beta distribution
        for sample_idx in range(n_samples):
            beta_values[sample_idx, cpg_idx] = np.random.beta(
                alpha_array[sample_idx], beta
            )
    
    # Add batch effects
    batch_effect_std = meth_config["batch_effect_std"]
    unique_batches = np.unique(batches)
    batch_effects = {
        batch: np.random.normal(0, batch_effect_std, n_cpgs)
        for batch in unique_batches
    }
    
    for sample_idx in range(n_samples):
        batch = batches[sample_idx]
        beta_values[sample_idx, :] += batch_effects[batch]
    
    # Clip to valid beta value range
    beta_values = np.clip(beta_values, 0.01, 0.99)
    
    # Add missing values (MAR by batch)
    miss_rate = meth_config["missingness_rate"]
    for batch in unique_batches:
        batch_mask = batches == batch
        batch_samples = np.where(batch_mask)[0]
        
        # Higher missingness for later batches
        batch_miss_rate = miss_rate * (1 + 0.5 * batch)
        
        for sample_idx in batch_samples:
            n_missing = int(batch_miss_rate * n_cpgs)
            missing_cpgs = np.random.choice(n_cpgs, n_missing, replace=False)
            beta_values[sample_idx, missing_cpgs] = np.nan
    
    # Create dataframe with CpG names
    cpg_names = [f"cg{i:05d}" for i in range(n_cpgs)]
    sample_ids = [f"sample_{i:04d}" for i in range(n_samples)]
    
    return pd.DataFrame(beta_values, index=sample_ids, columns=cpg_names)


def _simulate_fragmentomics(config: dict[str, Any], labels: np.ndarray, batches: np.ndarray) -> pd.DataFrame:
    """Simulate fragmentomics features."""
    frag_config = config["fragmentomics"]
    n_samples = len(labels)
    
    size_bins = frag_config["size_bins"]
    tss_bins = frag_config["tss_enrichment_bins"]
    n_size_features = len(size_bins)
    n_tss_features = tss_bins
    
    # Initialize feature matrix
    n_features = n_size_features + n_tss_features
    frag_features = np.zeros((n_samples, n_features))
    
    # Size distribution features
    base_size_props = np.random.dirichlet(np.ones(n_size_features))
    size_effect = np.random.normal(
        frag_config["size_effect_mean"],
        frag_config["size_effect_std"],
        n_size_features
    )
    
    for sample_idx in range(n_samples):
        is_cancer = labels[sample_idx] == 1
        
        # Modify size distribution for cancer samples
        if is_cancer:
            size_props = base_size_props + size_effect
            size_props = np.abs(size_props)  # Ensure positive
            size_props = size_props / size_props.sum()  # Renormalize
        else:
            size_props = base_size_props
        
        # Add noise
        noise = np.random.normal(0, frag_config["noise_std"], n_size_features)
        size_props = size_props + noise
        size_props = np.abs(size_props)
        size_props = size_props / size_props.sum()
        
        frag_features[sample_idx, :n_size_features] = size_props
    
    # TSS enrichment features
    base_tss = np.random.exponential(1.0, n_tss_features)
    tss_effect = np.random.normal(
        frag_config["tss_effect_mean"],
        frag_config["tss_effect_std"],
        n_tss_features
    )
    
    for sample_idx in range(n_samples):
        is_cancer = labels[sample_idx] == 1
        
        if is_cancer:
            tss_values = base_tss + tss_effect
        else:
            tss_values = base_tss
        
        # Add noise
        noise = np.random.normal(0, frag_config["noise_std"], n_tss_features)
        tss_values = tss_values + noise
        tss_values = np.abs(tss_values)  # Ensure positive
        
        frag_features[sample_idx, n_size_features:] = tss_values
    
    # Add batch effects
    unique_batches = np.unique(batches)
    for batch in unique_batches:
        batch_mask = batches == batch
        batch_effect = np.random.normal(0, 0.02, n_features)
        frag_features[batch_mask, :] += batch_effect
    
    # Create feature names
    size_names = [f"size_bin_{size}" for size in size_bins]
    tss_names = [f"tss_bin_{i}" for i in range(n_tss_features)]
    feature_names = size_names + tss_names
    
    sample_ids = [f"sample_{i:04d}" for i in range(n_samples)]
    
    return pd.DataFrame(frag_features, index=sample_ids, columns=feature_names)


def main() -> None:
    """CLI entry point for data simulation."""
    parser = argparse.ArgumentParser(description="Generate synthetic cfDNA dataset")
    parser.add_argument("config", type=Path, help="Configuration YAML file")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print(f"Generating synthetic cfDNA dataset...")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    
    stats = simulate_dataset(args.config, args.output_dir)
    
    if args.verbose:
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print(f"\nDataset generated successfully!")
    print(f"Files created: {len(stats['files_created'])}")


if __name__ == "__main__":
    main()