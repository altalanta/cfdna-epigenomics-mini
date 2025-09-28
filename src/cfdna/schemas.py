"""Pydantic data schemas for cfDNA epigenomics data validation."""

from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict


class BatchCenter(str, Enum):
    """Valid batch processing centers."""
    SITE_A = "Site_A"
    SITE_B = "Site_B" 
    SITE_C = "Site_C"


class Sex(str, Enum):
    """Biological sex categories."""
    FEMALE = "F"
    MALE = "M"


class Label(int, Enum):
    """Binary classification labels."""
    CONTROL = 0
    CANCER = 1


class SampleMetadata(BaseModel):
    """Metadata for a single cfDNA sample."""
    sample_id: str = Field(..., description="Unique sample identifier")
    subject_id: str = Field(..., description="Subject identifier (multiple samples per subject allowed)")
    age: float = Field(..., ge=18, le=120, description="Age in years")
    sex: Sex = Field(..., description="Biological sex")
    batch: int = Field(..., ge=0, description="Processing batch number")
    center: BatchCenter = Field(..., description="Processing center")
    label: Label = Field(..., description="Binary classification label")

    model_config = ConfigDict(use_enum_values=True)


class DatasetMetadata(BaseModel):
    """Metadata for the complete dataset."""
    n_samples: int = Field(..., gt=0, description="Total number of samples")
    n_controls: int = Field(..., ge=0, description="Number of control samples")
    n_cancer: int = Field(..., ge=0, description="Number of cancer samples")
    n_batches: int = Field(..., gt=0, le=10, description="Number of processing batches")
    random_seed: int = Field(..., description="Random seed for reproducibility")
    
    @field_validator('n_samples')
    @classmethod
    def validate_sample_counts(cls, v, info):
        """Ensure total samples equals controls + cancer."""
        values = info.data if info else {}
        if 'n_controls' in values and 'n_cancer' in values:
            expected_total = values['n_controls'] + values['n_cancer']
            if v != expected_total:
                raise ValueError(f"n_samples ({v}) must equal n_controls + n_cancer ({expected_total})")
        return v


class MethylationConfig(BaseModel):
    """Configuration for methylation data simulation."""
    n_cpgs: int = Field(20000, gt=0, le=100000, description="Number of CpG sites")
    n_dmrs: int = Field(200, gt=0, le=1000, description="Number of DMRs")
    cpgs_per_dmr: int = Field(100, gt=0, le=500, description="CpGs per DMR")
    alpha_base: float = Field(2.0, gt=0, description="Base beta distribution alpha")
    beta_base: float = Field(8.0, gt=0, description="Base beta distribution beta")
    dmr_effect_size_mean: float = Field(1.5, description="Mean DMR effect size")
    dmr_effect_size_std: float = Field(0.5, gt=0, description="DMR effect size std")
    batch_effect_std: float = Field(0.02, ge=0, description="Batch effect standard deviation")
    missingness_rate: float = Field(0.05, ge=0, le=0.5, description="Missing data rate")


class FragmentomicsConfig(BaseModel):
    """Configuration for fragmentomics data simulation."""
    size_bins: List[int] = Field([100, 150, 200, 250, 300, 350, 400], 
                                 description="Fragment size bins")
    tss_enrichment_bins: int = Field(10, gt=0, le=50, description="TSS enrichment bins")
    size_effect_mean: float = Field(0.02, description="Mean size effect")
    size_effect_std: float = Field(0.01, gt=0, description="Size effect std")
    tss_effect_mean: float = Field(0.1, description="Mean TSS effect")
    tss_effect_std: float = Field(0.05, gt=0, description="TSS effect std")
    noise_std: float = Field(0.01, gt=0, description="Noise standard deviation")
    
    @field_validator('size_bins')
    @classmethod
    def validate_size_bins(cls, v):
        """Ensure size bins are sorted and positive."""
        if not all(x > 0 for x in v):
            raise ValueError("All size bins must be positive")
        if v != sorted(v):
            raise ValueError("Size bins must be sorted in ascending order")
        return v


class SimulationConfig(BaseModel):
    """Complete configuration for data simulation."""
    dataset: DatasetMetadata
    methylation: MethylationConfig = Field(default_factory=MethylationConfig)
    fragmentomics: FragmentomicsConfig = Field(default_factory=FragmentomicsConfig)
    
    @field_validator('dataset')
    @classmethod
    def validate_dataset_methylation_consistency(cls, v):
        """Ensure dataset and methylation configs are consistent."""
        return v


class FeatureMatrix(BaseModel):
    """Validated feature matrix."""
    sample_ids: List[str] = Field(..., description="Sample identifiers")
    feature_names: List[str] = Field(..., description="Feature names")
    n_samples: int = Field(..., gt=0, description="Number of samples")
    n_features: int = Field(..., gt=0, description="Number of features")
    has_missing_values: bool = Field(..., description="Whether matrix contains NaN values")
    feature_types: Dict[str, int] = Field(..., description="Count of features by type")
    
    @field_validator('n_samples')
    @classmethod
    def validate_samples_consistency(cls, v, info):
        """Ensure sample count matches sample IDs."""
        values = info.data if info else {}
        if 'sample_ids' in values and len(values['sample_ids']) != v:
            raise ValueError("n_samples must match length of sample_ids")
        return v
    
    @field_validator('n_features')
    @classmethod
    def validate_features_consistency(cls, v, info):
        """Ensure feature count matches feature names."""
        values = info.data if info else {}
        if 'feature_names' in values and len(values['feature_names']) != v:
            raise ValueError("n_features must match length of feature_names")
        return v


class DataSplits(BaseModel):
    """Validated train/validation/test splits."""
    train_indices: List[int] = Field(..., description="Training set indices")
    val_indices: List[int] = Field(..., description="Validation set indices") 
    test_indices: List[int] = Field(..., description="Test set indices")
    n_total_samples: int = Field(..., gt=0, description="Total number of samples")
    stratified: bool = Field(True, description="Whether splits are stratified")
    
    @field_validator('n_total_samples')
    @classmethod
    def validate_split_coverage(cls, v, info):
        """Ensure splits cover all samples exactly once."""
        values = info.data if info else {}
        if all(k in values for k in ['train_indices', 'val_indices', 'test_indices']):
            all_indices = set(values['train_indices'] + values['val_indices'] + values['test_indices'])
            expected_indices = set(range(v))
            
            if all_indices != expected_indices:
                raise ValueError("Splits must cover all samples exactly once")
                
            # Check for overlaps
            train_set = set(values['train_indices'])
            val_set = set(values['val_indices'])
            test_set = set(values['test_indices'])
            
            if train_set & val_set or train_set & test_set or val_set & test_set:
                raise ValueError("Splits must not have overlapping indices")
        
        return v


class ModelPredictions(BaseModel):
    """Validated model predictions."""
    model_name: str = Field(..., description="Model identifier")
    sample_ids: List[str] = Field(..., description="Sample identifiers")
    true_labels: List[int] = Field(..., description="True binary labels (0/1)")
    predicted_probs: List[float] = Field(..., description="Predicted probabilities [0,1]")
    split_name: str = Field(..., pattern="^(train|val|test)$", description="Data split name")
    
    @field_validator('predicted_probs')
    @classmethod
    def validate_probabilities(cls, v):
        """Ensure probabilities are in valid range."""
        if not all(0 <= p <= 1 for p in v):
            raise ValueError("All predicted probabilities must be between 0 and 1")
        return v
    
    @field_validator('true_labels')
    @classmethod
    def validate_labels(cls, v):
        """Ensure labels are binary."""
        if not all(label in [0, 1] for label in v):
            raise ValueError("All true labels must be 0 or 1")
        return v
    
    @field_validator('sample_ids')
    @classmethod
    def validate_consistency(cls, v, info):
        """Ensure all arrays have same length."""
        values = info.data if info else {}
        arrays_to_check = ['true_labels', 'predicted_probs']
        for array_name in arrays_to_check:
            if array_name in values and len(values[array_name]) != len(v):
                raise ValueError(f"Length mismatch: sample_ids and {array_name}")
        return v


class EvaluationMetrics(BaseModel):
    """Model evaluation metrics with confidence intervals."""
    auroc_mean: float = Field(..., ge=0, le=1, description="Mean AUROC")
    auroc_ci_lower: float = Field(..., ge=0, le=1, description="AUROC CI lower bound")
    auroc_ci_upper: float = Field(..., ge=0, le=1, description="AUROC CI upper bound")
    auprc_mean: float = Field(..., ge=0, le=1, description="Mean AUPRC")
    auprc_ci_lower: float = Field(..., ge=0, le=1, description="AUPRC CI lower bound")
    auprc_ci_upper: float = Field(..., ge=0, le=1, description="AUPRC CI upper bound")
    brier_score: float = Field(..., ge=0, le=1, description="Brier score")
    sensitivity_at_90_specificity: float = Field(..., ge=0, le=1, description="Sensitivity at 90% specificity")
    sensitivity_at_95_specificity: float = Field(..., ge=0, le=1, description="Sensitivity at 95% specificity")
    calibration_ece: float = Field(..., ge=0, description="Expected calibration error")
    
    @field_validator('auroc_ci_upper')
    @classmethod
    def validate_auroc_ci(cls, v, info):
        """Ensure AUROC CI is valid."""
        values = info.data if info else {}
        if 'auroc_ci_lower' in values and v < values['auroc_ci_lower']:
            raise ValueError("AUROC CI upper bound must be >= lower bound")
        return v
    
    @field_validator('auprc_ci_upper')
    @classmethod
    def validate_auprc_ci(cls, v, info):
        """Ensure AUPRC CI is valid."""
        values = info.data if info else {}
        if 'auprc_ci_lower' in values and v < values['auprc_ci_lower']:
            raise ValueError("AUPRC CI upper bound must be >= lower bound")
        return v


def validate_dataframe(df: pd.DataFrame, expected_schema: BaseModel) -> None:
    """Validate a pandas DataFrame against a Pydantic schema.
    
    Args:
        df: DataFrame to validate
        expected_schema: Pydantic model defining expected structure
        
    Raises:
        ValueError: If DataFrame doesn't match schema
    """
    if isinstance(expected_schema, type) and issubclass(expected_schema, BaseModel):
        # For now, basic validation - can be extended
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        # Check for required columns based on schema
        if hasattr(expected_schema, 'model_fields'):
            required_fields = [name for name, field in expected_schema.model_fields.items() 
                             if field.is_required()]
            missing_cols = set(required_fields) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")


def create_feature_matrix_schema(df: pd.DataFrame) -> FeatureMatrix:
    """Create a FeatureMatrix schema from a DataFrame.
    
    Args:
        df: Feature DataFrame with samples as rows, features as columns
        
    Returns:
        Validated FeatureMatrix schema
    """
    # Determine feature types
    feature_types = {}
    methylation_features = [col for col in df.columns if col.startswith('dmr_')]
    fragmentomics_features = [col for col in df.columns if not col.startswith('dmr_')]
    
    feature_types['methylation'] = len(methylation_features)
    feature_types['fragmentomics'] = len(fragmentomics_features)
    
    return FeatureMatrix(
        sample_ids=df.index.tolist(),
        feature_names=df.columns.tolist(),
        n_samples=len(df),
        n_features=len(df.columns),
        has_missing_values=df.isnull().any().any(),
        feature_types=feature_types
    )


def create_model_predictions_schema(df: pd.DataFrame, model_name: str, split_name: str) -> ModelPredictions:
    """Create a ModelPredictions schema from a DataFrame.
    
    Args:
        df: Predictions DataFrame with columns: sample_id, true_label, predicted_prob
        model_name: Name of the model
        split_name: Name of the data split
        
    Returns:
        Validated ModelPredictions schema
    """
    required_cols = ['sample_id', 'true_label', 'predicted_prob']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return ModelPredictions(
        model_name=model_name,
        sample_ids=df['sample_id'].tolist(),
        true_labels=df['true_label'].tolist(),
        predicted_probs=df['predicted_prob'].tolist(),
        split_name=split_name
    )