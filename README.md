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
| 09a | 09a_baseline_model | ALS collaborative filtering (RMSE 3.61) |
| 09b | 09b_comparison_model | SVD++ design document |
| 10 | 10_hyperparameter_tuning | Cross-validated ALS (RMSE 2.52, -30%) |
| 11 | 11_ml_performance | Model comparison report |

## Tech Stack

PySpark, MLflow, Plotly, Databricks Lakeview Dashboard, hive_metastore
