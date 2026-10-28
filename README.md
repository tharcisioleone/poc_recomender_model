# POC: Book Recommendation Pipeline

Medallion architecture (Bronze → Silver → Gold + Quarantine) on Databricks.

## Pipeline Steps

| Step | Notebook | Description |
|------|----------|-------------|
| 00 | 00_setup | Environment setup, schema creation |
| 01 | 01_read_data | CSV ingestion to bronze |
| 02 | 02_data_governance | Column docs, PII tagging |
| 03 | 03_eda | Exploratory data analysis (ydata-profiling) |
| 04 | 04_data_transformation | Bronze → Silver cleaning |
| 05 | 05_data_merge | Silver → Gold joins + aggregations |
| 06 | 06_dq_checks | Data quality rules on silver |
| 07 | 07_quarantine_report | Quarantine isolation + gold cleanup |
| 08a/b/c | 08a_visualization_python | Plotly charts, dashboard, Power BI |
| 09a | 09a_baseline_model | ALS collaborative filtering (RMSE 3.31) |
| 09b | 09b_comparison_model | SVD++ design document |
| 10 | 10_hyperparameter_tuning | Cross-validated ALS (RMSE 1.33, -60%) |
| 11 | 11_ml_performance | Model comparison report |

## Latest Model Results

| Model | Run date | Params | Test RMSE | Test MAE |
|-------|----------|--------|-----------|----------|
| 09a Baseline ALS | 2026-10-28 | rank 10, regParam 0.1, maxIter 10 | 3.3060 | 2.5567 |
| 10 CV-tuned ALS | 2026-10-28 | rank 100, regParam 0.5, maxIter 20 | 1.3305 | 0.8995 |

Step 10: 18 parameter combinations x 3 folds, best CV RMSE 2.8777. Train/test split 80/20 (seed 42) on `gold.book_ratings` (378,030 explicit ratings). Test predictions and `gold.recommendations` of both steps are clipped to the 1-10 rating scale (`model.rating_scale` in `config/09_recommender_model.json` and `config/10_hyperparameter_tuning.json`); only the CV RMSE is unclipped. Source: MLflow experiment `book_recommender`.

## Tech Stack

PySpark, MLflow, Plotly, Databricks Lakeview Dashboard, hive_metastore
