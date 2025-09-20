-- Example SQL queries for cfDNA cancer detection database

-- Query 1: Class balance by batch and center
-- Shows distribution of cancer vs control samples across batches and centers
SELECT 
    batch,
    center,
    SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) as controls,
    SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) as cancer_cases,
    COUNT(*) as total_samples,
    ROUND(AVG(CAST(label AS FLOAT)) * 100, 1) as cancer_percentage
FROM samples
GROUP BY batch, center
ORDER BY batch, center;

-- Query 2: Draw a batch-balanced train/validation cohort
-- Selects samples ensuring balanced representation across batches
WITH batch_stats AS (
    SELECT 
        batch,
        COUNT(*) as batch_size,
        SUM(label) as batch_cancer_cases
    FROM samples
    GROUP BY batch
),
stratified_samples AS (
    SELECT 
        s.*,
        ROW_NUMBER() OVER (
            PARTITION BY s.batch, s.label 
            ORDER BY RANDOM()
        ) as rn,
        bs.batch_size,
        bs.batch_cancer_cases
    FROM samples s
    JOIN batch_stats bs ON s.batch = bs.batch
)
SELECT 
    batch,
    label,
    COUNT(*) as selected_count,
    'train' as cohort_type
FROM stratified_samples
WHERE rn <= CAST(batch_size * 0.7 AS INTEGER)  -- 70% for training
GROUP BY batch, label

UNION ALL

SELECT 
    batch,
    label,
    COUNT(*) as selected_count,
    'validation' as cohort_type
FROM stratified_samples  
WHERE rn > CAST(batch_size * 0.7 AS INTEGER) 
    AND rn <= CAST(batch_size * 0.9 AS INTEGER)  -- 20% for validation
GROUP BY batch, label

ORDER BY batch, cohort_type, label;

-- Query 3: List assays missing expected files
-- Identifies samples that should have assay data but files are missing
WITH expected_assays AS (
    SELECT DISTINCT
        s.id as sample_id,
        'methylation' as expected_assay_type
    FROM samples s
    
    UNION ALL
    
    SELECT DISTINCT  
        s.id as sample_id,
        'fragmentomics' as expected_assay_type
    FROM samples s
),
actual_assays AS (
    SELECT DISTINCT
        sample_id,
        assay_type
    FROM assays
)
SELECT 
    ea.sample_id,
    ea.expected_assay_type as missing_assay,
    s.batch,
    s.center,
    s.label
FROM expected_assays ea
LEFT JOIN actual_assays aa ON (
    ea.sample_id = aa.sample_id 
    AND ea.expected_assay_type = aa.assay_type
)
JOIN samples s ON ea.sample_id = s.id
WHERE aa.assay_type IS NULL
ORDER BY ea.sample_id, ea.expected_assay_type;

-- Query 4: Model performance comparison
-- Compare AUROC and AUPRC across different models on test set
SELECT 
    model_name,
    MAX(CASE WHEN metric_name = 'auroc' THEN metric_value END) as test_auroc,
    MAX(CASE WHEN metric_name = 'auroc' THEN 
        PRINTF('%.3f-%.3f', confidence_interval_lower, confidence_interval_upper)
    END) as auroc_95ci,
    MAX(CASE WHEN metric_name = 'auprc' THEN metric_value END) as test_auprc,
    MAX(CASE WHEN metric_name = 'auprc' THEN 
        PRINTF('%.3f-%.3f', confidence_interval_lower, confidence_interval_upper)
    END) as auprc_95ci
FROM performance_metrics
WHERE dataset_split = 'test'
    AND metric_name IN ('auroc', 'auprc')
GROUP BY model_name
ORDER BY test_auroc DESC;

-- Query 5: Sample demographics summary
-- Analyze age and sex distribution by cancer status
SELECT 
    label,
    CASE WHEN label = 0 THEN 'Control' ELSE 'Cancer' END as group_name,
    COUNT(*) as n_samples,
    ROUND(AVG(age), 1) as mean_age,
    ROUND(MIN(age), 1) as min_age,
    ROUND(MAX(age), 1) as max_age,
    SUM(CASE WHEN sex = 'F' THEN 1 ELSE 0 END) as n_female,
    SUM(CASE WHEN sex = 'M' THEN 1 ELSE 0 END) as n_male,
    ROUND(AVG(CASE WHEN sex = 'F' THEN 1.0 ELSE 0.0 END) * 100, 1) as percent_female
FROM sample_summary
GROUP BY label
ORDER BY label;

-- Query 6: High-confidence predictions analysis  
-- Find samples with high-confidence predictions (>0.9 or <0.1)
WITH high_confidence_predictions AS (
    SELECT 
        p.sample_id,
        p.model_name,
        p.predicted_probability,
        p.predicted_label,
        s.label as true_label,
        CASE 
            WHEN p.predicted_probability >= 0.9 THEN 'High Cancer Risk'
            WHEN p.predicted_probability <= 0.1 THEN 'Low Cancer Risk'
        END as confidence_category,
        CASE 
            WHEN p.predicted_label = s.label THEN 'Correct'
            ELSE 'Incorrect'
        END as prediction_accuracy
    FROM predictions p
    JOIN samples s ON p.sample_id = s.id
    WHERE p.predicted_probability >= 0.9 OR p.predicted_probability <= 0.1
)
SELECT 
    model_name,
    confidence_category,
    prediction_accuracy,
    COUNT(*) as n_predictions,
    ROUND(AVG(predicted_probability), 3) as avg_probability
FROM high_confidence_predictions
GROUP BY model_name, confidence_category, prediction_accuracy
ORDER BY model_name, confidence_category, prediction_accuracy;

-- Query 7: Data quality checks
-- Check for potential data quality issues
SELECT 
    'Missing age data' as issue_type,
    COUNT(*) as n_issues
FROM patients 
WHERE age IS NULL

UNION ALL

SELECT 
    'Invalid age range' as issue_type,
    COUNT(*) as n_issues  
FROM patients
WHERE age < 18 OR age > 100

UNION ALL

SELECT 
    'Missing assay files' as issue_type,
    COUNT(*) as n_issues
FROM assays
WHERE file_path IS NULL OR file_path = ''

UNION ALL

SELECT 
    'Prediction probability out of range' as issue_type,
    COUNT(*) as n_issues
FROM predictions
WHERE predicted_probability < 0 OR predicted_probability > 1

UNION ALL

SELECT 
    'Inconsistent prediction labels' as issue_type,
    COUNT(*) as n_issues
FROM predictions
WHERE (predicted_probability >= 0.5 AND predicted_label = 0) 
   OR (predicted_probability < 0.5 AND predicted_label = 1);

-- Query 8: Temporal analysis of model predictions
-- Analyze when predictions were made (useful for production monitoring)
SELECT 
    model_name,
    DATE(prediction_date) as prediction_date,
    COUNT(*) as n_predictions,
    ROUND(AVG(predicted_probability), 3) as avg_probability,
    SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as n_positive_predictions
FROM predictions
WHERE prediction_date >= datetime('now', '-30 days')  -- Last 30 days
GROUP BY model_name, DATE(prediction_date)
ORDER BY prediction_date DESC, model_name;