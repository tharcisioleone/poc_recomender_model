"""EDA: Descriptive statistics using ydata-profiling."""

import os
from pyspark.sql import SparkSession, DataFrame
from source.base import BaseAnalysis


class EDA(BaseAnalysis):
    """Generates profiling reports for bronze tables."""

    def __init__(self, config_path: str, spark: SparkSession = None):
        super().__init__(config_path, spark)
        self.profiling_cfg = self.config.get("profiling", {})

    def extract(self) -> dict[str, DataFrame]:
        source_tables = self.config.get("source_tables", {})
        dfs = {}
        for key, table in source_tables.items():
            self.logger.info(f"Reading {table}")
            dfs[key] = self.spark.table(table)
        return dfs

    def analyze(self, dataframes: dict[str, DataFrame]) -> dict:
        from ydata_profiling import ProfileReport

        report_path = self._setup.get("reports_path", self.profiling_cfg.get("report_path", "/tmp/eda_reports"))
        os.makedirs(report_path, exist_ok=True)
        minimal = self.profiling_cfg["minimal"]
        prefix = self.profiling_cfg["report_prefix"]
        reports = {}

        for name, df in dataframes.items():
            self.logger.info(f"Profiling {name} ({df.count()} rows)")
            pdf = df.toPandas()
            profile = ProfileReport(
                pdf,
                title=f"Profile Report: {name}",
                minimal=minimal,
                explorative=True,
            )
            output_file = f"{report_path}/{prefix}_{name}_profile.html"
            profile.to_file(output_file)
            reports[name] = {
                "rows": len(pdf),
                "columns": len(pdf.columns),
                "report_path": output_file,
                "missing_pct": pdf.isnull().mean().to_dict(),
            }
            self.logger.info(f"Report saved: {output_file}")

        return reports
