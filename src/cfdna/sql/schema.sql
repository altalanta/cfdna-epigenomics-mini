-- cfDNA Cancer Detection Database Schema
-- SQLite schema for storing synthetic cfDNA epigenomic data

-- Patients table: subject-level information
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    age REAL NOT NULL,
    sex TEXT NOT NULL CHECK (sex IN ('F', 'M')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Samples table: sample-level information  
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    batch INTEGER NOT NULL,
    center TEXT NOT NULL,
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    collection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients (id),
    UNIQUE(patient_id)  -- One sample per patient in this simplified schema
);

-- Assays table: links to raw data files
CREATE TABLE IF NOT EXISTS assays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL,
    assay_type TEXT NOT NULL CHECK (assay_type IN ('methylation', 'fragmentomics')),
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,
    checksum TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sample_id) REFERENCES samples (id)
);

-- Model predictions table: store model outputs
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_probability REAL NOT NULL CHECK (predicted_probability >= 0 AND predicted_probability <= 1),
    predicted_label INTEGER NOT NULL CHECK (predicted_label IN (0, 1)),
    confidence_score REAL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sample_id) REFERENCES samples (id)
);

-- Model metadata table: track model information
CREATE TABLE IF NOT EXISTS models (
    name TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    hyperparameters TEXT,  -- JSON string
    training_samples INTEGER,
    validation_auroc REAL,
    test_auroc REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics table: detailed evaluation metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dataset_split TEXT NOT NULL CHECK (dataset_split IN ('train', 'val', 'test')),
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    confidence_interval_lower REAL,
    confidence_interval_upper REAL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_name) REFERENCES models (name)
);

-- Indices for query performance
CREATE INDEX IF NOT EXISTS idx_samples_batch ON samples (batch);
CREATE INDEX IF NOT EXISTS idx_samples_center ON samples (center);
CREATE INDEX IF NOT EXISTS idx_samples_label ON samples (label);
CREATE INDEX IF NOT EXISTS idx_assays_sample_type ON assays (sample_id, assay_type);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions (model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_sample ON predictions (sample_id);
CREATE INDEX IF NOT EXISTS idx_metrics_model_split ON performance_metrics (model_name, dataset_split);

-- Views for common queries
CREATE VIEW IF NOT EXISTS sample_summary AS
SELECT 
    s.id as sample_id,
    s.patient_id,
    p.age,
    p.sex,
    s.batch,
    s.center,
    s.label,
    COUNT(a.id) as num_assays,
    GROUP_CONCAT(a.assay_type) as available_assays
FROM samples s
JOIN patients p ON s.patient_id = p.id
LEFT JOIN assays a ON s.id = a.sample_id
GROUP BY s.id, s.patient_id, p.age, p.sex, s.batch, s.center, s.label;

CREATE VIEW IF NOT EXISTS model_performance_summary AS
SELECT 
    m.name as model_name,
    m.version as model_version,
    m.algorithm,
    pm_auroc.metric_value as test_auroc,
    pm_auroc.confidence_interval_lower as test_auroc_ci_lower,
    pm_auroc.confidence_interval_upper as test_auroc_ci_upper,
    pm_auprc.metric_value as test_auprc,
    m.training_samples,
    m.created_at
FROM models m
LEFT JOIN performance_metrics pm_auroc ON (
    m.name = pm_auroc.model_name 
    AND m.version = pm_auroc.model_version
    AND pm_auroc.dataset_split = 'test'
    AND pm_auroc.metric_name = 'auroc'
)
LEFT JOIN performance_metrics pm_auprc ON (
    m.name = pm_auprc.model_name
    AND m.version = pm_auprc.model_version  
    AND pm_auprc.dataset_split = 'test'
    AND pm_auprc.metric_name = 'auprc'
);

-- Trigger to ensure data consistency
CREATE TRIGGER IF NOT EXISTS validate_prediction_consistency
BEFORE INSERT ON predictions
FOR EACH ROW
WHEN NEW.predicted_label != CASE 
    WHEN NEW.predicted_probability >= 0.5 THEN 1 
    ELSE 0 
END
BEGIN
    SELECT RAISE(ABORT, 'Predicted label inconsistent with probability threshold');
END;