"""Load synthetic cfDNA data into SQLite database."""

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


def create_database(db_path: Path, schema_path: Path) -> None:
    """Create SQLite database with schema.
    
    Args:
        db_path: Path to SQLite database file
        schema_path: Path to SQL schema file
    """
    # Read schema
    with open(schema_path) as f:
        schema_sql = f.read()
    
    # Create database
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        print(f"Database created: {db_path}")


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA-256 checksum of file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Hexadecimal checksum string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def load_metadata(db_path: Path, data_dir: Path) -> None:
    """Load patient and sample metadata into database.
    
    Args:
        db_path: Path to SQLite database
        data_dir: Directory containing data files
    """
    data_dir = Path(data_dir)
    
    # Load metadata
    metadata = pd.read_csv(data_dir / "metadata.csv")
    
    with sqlite3.connect(db_path) as conn:
        # Load patients
        patients_data = metadata[["subject_id", "age", "sex"]].drop_duplicates()
        patients_data.columns = ["id", "age", "sex"]
        
        patients_data.to_sql("patients", conn, if_exists="append", index=False)
        print(f"Loaded {len(patients_data)} patients")
        
        # Load samples
        samples_data = metadata[["sample_id", "subject_id", "batch", "center", "label"]]
        samples_data.columns = ["id", "patient_id", "batch", "center", "label"]
        
        samples_data.to_sql("samples", conn, if_exists="append", index=False)
        print(f"Loaded {len(samples_data)} samples")


def load_assays(db_path: Path, data_dir: Path) -> None:
    """Load assay file information into database.
    
    Args:
        db_path: Path to SQLite database
        data_dir: Directory containing data files
    """
    data_dir = Path(data_dir)
    
    # Expected data files
    assay_files = [
        ("methylation", "X_meth.parquet"),
        ("fragmentomics", "X_frag.parquet")
    ]
    
    assay_records = []
    
    for assay_type, filename in assay_files:
        file_path = data_dir / filename
        
        if file_path.exists():
            # Calculate file metadata
            file_size = file_path.stat().st_size
            checksum = calculate_file_checksum(file_path)
            
            # Load file to get sample IDs
            if filename.endswith(".parquet"):
                df = pd.read_parquet(file_path)
                sample_ids = df.index.tolist()
            else:
                # Handle other formats if needed
                sample_ids = []
            
            # Create assay records
            for sample_id in sample_ids:
                assay_records.append({
                    "sample_id": sample_id,
                    "assay_type": assay_type,
                    "file_path": str(file_path),
                    "file_size_bytes": file_size,
                    "checksum": checksum
                })
    
    if assay_records:
        assays_df = pd.DataFrame(assay_records)
        
        with sqlite3.connect(db_path) as conn:
            assays_df.to_sql("assays", conn, if_exists="append", index=False)
        
        print(f"Loaded {len(assay_records)} assay records")
    else:
        print("No assay files found")


def load_model_results(db_path: Path, artifacts_dir: Path) -> None:
    """Load model results into database.
    
    Args:
        db_path: Path to SQLite database
        artifacts_dir: Directory containing model artifacts
    """
    artifacts_dir = Path(artifacts_dir)
    
    if not artifacts_dir.exists():
        print(f"Artifacts directory not found: {artifacts_dir}")
        return
    
    model_records = []
    metric_records = []
    prediction_records = []
    
    # Look for model results files
    for results_file in artifacts_dir.glob("*_results.json"):
        model_name = results_file.stem.replace("_results", "")
        
        try:
            import json
            with open(results_file) as f:
                results = json.load(f)
            
            # Extract model metadata
            if "test" in results:
                test_results = results["test"]
                model_record = {
                    "name": model_name,
                    "version": "v1.0",
                    "algorithm": model_name,
                    "hyperparameters": "{}",  # Could be expanded
                    "training_samples": test_results.get("n_samples", 0),
                    "test_auroc": test_results.get("auroc", {}).get("mean", None)
                }
                model_records.append(model_record)
                
                # Extract performance metrics
                for split in ["train", "val", "test"]:
                    if split in results:
                        split_results = results[split]
                        
                        # AUROC
                        if "auroc" in split_results:
                            auroc_data = split_results["auroc"]
                            metric_records.append({
                                "model_name": model_name,
                                "model_version": "v1.0",
                                "dataset_split": split,
                                "metric_name": "auroc",
                                "metric_value": auroc_data.get("mean"),
                                "confidence_interval_lower": auroc_data.get("ci_lower"),
                                "confidence_interval_upper": auroc_data.get("ci_upper")
                            })
                        
                        # AUPRC
                        if "auprc" in split_results:
                            auprc_data = split_results["auprc"]
                            metric_records.append({
                                "model_name": model_name,
                                "model_version": "v1.0", 
                                "dataset_split": split,
                                "metric_name": "auprc",
                                "metric_value": auprc_data.get("mean"),
                                "confidence_interval_lower": auprc_data.get("ci_lower"),
                                "confidence_interval_upper": auprc_data.get("ci_upper")
                            })
        
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
    
    # Look for prediction files
    for pred_file in artifacts_dir.glob("*_predictions.csv"):
        try:
            predictions_df = pd.read_csv(pred_file)
            
            # Extract model name and split from filename
            filename_parts = pred_file.stem.split("_")
            if len(filename_parts) >= 3:
                model_name = "_".join(filename_parts[:-2])
                split = filename_parts[-2]
                
                for _, row in predictions_df.iterrows():
                    prediction_records.append({
                        "sample_id": row["sample_id"],
                        "model_name": model_name,
                        "model_version": "v1.0",
                        "predicted_probability": row["predicted_prob"],
                        "predicted_label": 1 if row["predicted_prob"] >= 0.5 else 0
                    })
        
        except Exception as e:
            print(f"Error loading {pred_file}: {e}")
    
    # Insert into database
    with sqlite3.connect(db_path) as conn:
        if model_records:
            models_df = pd.DataFrame(model_records)
            models_df.to_sql("models", conn, if_exists="append", index=False)
            print(f"Loaded {len(model_records)} model records")
        
        if metric_records:
            metrics_df = pd.DataFrame(metric_records)
            metrics_df.to_sql("performance_metrics", conn, if_exists="append", index=False)
            print(f"Loaded {len(metric_records)} metric records")
        
        if prediction_records:
            predictions_df = pd.DataFrame(prediction_records)
            predictions_df.to_sql("predictions", conn, if_exists="append", index=False)
            print(f"Loaded {len(prediction_records)} prediction records")


def main() -> None:
    """Main function to load all data into SQLite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load cfDNA data into SQLite database")
    parser.add_argument("--data_dir", type=Path, default="data",
                       help="Directory containing synthetic data")
    parser.add_argument("--artifacts_dir", type=Path, default="artifacts",
                       help="Directory containing model artifacts")
    parser.add_argument("--db_path", type=Path, default="cfdna.db",
                       help="Path to SQLite database file")
    parser.add_argument("--schema_path", type=Path, 
                       default=Path(__file__).parent / "schema.sql",
                       help="Path to SQL schema file")
    
    args = parser.parse_args()
    
    print("Loading cfDNA data into SQLite database...")
    print(f"Database: {args.db_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Artifacts directory: {args.artifacts_dir}")
    
    # Create database
    create_database(args.db_path, args.schema_path)
    
    # Load data
    if args.data_dir.exists():
        load_metadata(args.db_path, args.data_dir)
        load_assays(args.db_path, args.data_dir)
    else:
        print(f"Data directory not found: {args.data_dir}")
    
    # Load model results if available
    if args.artifacts_dir.exists():
        load_model_results(args.db_path, args.artifacts_dir)
    else:
        print(f"Artifacts directory not found: {args.artifacts_dir}")
    
    print("Database loading completed!")
    
    # Print summary
    with sqlite3.connect(args.db_path) as conn:
        cursor = conn.cursor()
        
        print("\nDatabase Summary:")
        print("-" * 30)
        
        for table in ["patients", "samples", "assays", "models", "predictions"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"{table:>12}: {count:>6} records")


if __name__ == "__main__":
    main()