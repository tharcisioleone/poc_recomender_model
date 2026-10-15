"""Data Quality Checks: configurable rule engine."""

from datetime import datetime
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from source.base import BaseETL


class DQChecks(BaseETL):
    """Runs data quality checks and produces a DQ report table."""

    def __init__(self, config_path: str, spark: SparkSession = None):
        super().__init__(config_path, spark)
        self.checks_cfg = self.config.get("checks", {})
        self._results: list[dict] = []

    def _check_not_null(self, df: DataFrame, column: str) -> tuple[int, int]:
        total = df.count()
        failures = df.filter(F.col(column).isNull()).count()
        return total, failures

    def _check_between(self, df: DataFrame, column: str, min_val, max_val) -> tuple[int, int]:
        total = df.filter(F.col(column).isNotNull()).count()
        failures = df.filter(
            F.col(column).isNotNull()
            & ~F.col(column).between(min_val, max_val)
        ).count()
        return total, failures

    def _check_unique(self, df: DataFrame, column: str) -> tuple[int, int]:
        total = df.filter(F.col(column).isNotNull()).count()
        unique = df.filter(F.col(column).isNotNull()).select(column).distinct().count()
        failures = total - unique
        return total, failures

    def _check_referential_integrity(
        self, df: DataFrame, column: str, ref_table: str, ref_column: str
    ) -> tuple[int, int]:
        ref_df = self.spark.table(ref_table).select(F.col(ref_column).alias("_ref"))
        total = df.filter(F.col(column).isNotNull()).count()
        orphans = (
            df.filter(F.col(column).isNotNull())
            .join(ref_df, df[column] == ref_df["_ref"], "left_anti")
            .count()
        )
        return total, orphans

    def extract(self) -> DataFrame:
        return None

    def transform(self, df: DataFrame) -> DataFrame:
        run_ts = datetime.now().isoformat()

        for table_key, checks in self.checks_cfg.items():
            table_name = self._source_table(table_key)
            source_df = self.spark.table(table_name)

            for check in checks:
                rule = check["rule"]
                column = check["column"]
                name = check["name"]

                if rule == "not_null":
                    total, failures = self._check_not_null(source_df, column)
                elif rule == "between":
                    total, failures = self._check_between(
                        source_df, column, check["min"], check["max"]
                    )
                elif rule == "unique":
                    total, failures = self._check_unique(source_df, column)
                elif rule == "referential_integrity":
                    total, failures = self._check_referential_integrity(
                        source_df, column,
                        check["reference_table"], check["reference_column"],
                    )
                else:
                    continue

                pass_rate = ((total - failures) / total * 100) if total > 0 else 0.0
                status = "PASS" if failures == 0 else "FAIL"

                self._results.append({
                    "run_timestamp": run_ts,
                    "table_name": table_name,
                    "check_name": name,
                    "rule": rule,
                    "column": column,
                    "total_rows": total,
                    "failures": failures,
                    "pass_rate_pct": round(pass_rate, 2),
                    "status": status,
                })
                self.logger.info(f"{name}: {status} ({failures}/{total} failures)")

        return self.spark.createDataFrame(self._results)

    def load(self, df: DataFrame) -> None:
        full_name = self._table_name()
        self.logger.info(f"Writing DQ report: {full_name}")
        df.write.mode("overwrite").saveAsTable(full_name)

    def run(self) -> DataFrame:
        self.logger.info(f"Pipeline started: {self.__class__.__name__}")
        self._ensure_schemas()
        self.extract()
        report_df = self.transform(None)
        self.load(report_df)
        self.logger.info("DQ checks complete")
        return report_df
