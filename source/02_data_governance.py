"""Data Governance: table/column comments, tags, row/column-level security."""

from pyspark.sql import SparkSession
from source.base import _BaseConfig


class DataGovernance(_BaseConfig):
    """Applies governance policies to bronze tables."""

    def __init__(self, config_path: str, spark: SparkSession = None):
        super().__init__(config_path, spark)

    def _apply_table_comments(self) -> None:
        for table, comment in self.config.get("table_comments", {}).items():
            safe_comment = comment.replace("'", "\\'")
            self.spark.sql(f"COMMENT ON TABLE {table} IS '{safe_comment}'")
            self.logger.info(f"Table comment set: {table}")

    def _apply_column_comments(self) -> None:
        for table, columns in self.config.get("column_comments", {}).items():
            for col, comment in columns.items():
                safe_comment = comment.replace("'", "\\'")
                try:
                    self.spark.sql(
                        f"ALTER TABLE {table} ALTER COLUMN `{col}` COMMENT '{safe_comment}'"
                    )
                except Exception as e:
                    self.logger.warning(f"Column comment failed for {table}.{col}: {e}")
            self.logger.info(f"Column comments set: {table}")

    def _apply_tags(self) -> None:
        """Apply tags via ALTER TABLE SET TAGS (Unity Catalog only)."""
        for table, tags in self.config.get("tags", {}).items():
            tag_pairs = ", ".join(f"'{k}' = '{v}'" for k, v in tags.items())
            try:
                self.spark.sql(f"ALTER TABLE {table} SET TAGS ({tag_pairs})")
                self.logger.info(f"Tags set: {table}")
            except Exception as e:
                self.logger.warning(f"Tags not supported (requires UC): {e}")

    def _show_governance_summary(self) -> list[dict]:
        summary = []
        for table in self.config.get("table_comments", {}).keys():
            try:
                cols = self.spark.sql(f"DESCRIBE TABLE EXTENDED {table}").collect()
                summary.append({"table": table, "columns": len(cols)})
            except Exception:
                pass
        return summary

    def _document_security_policies(self) -> dict:
        """Documents intended row/column-level security policies.

        Note: Row-level security (ROW FILTER) and column masking
        require Unity Catalog. This method documents the intent
        and generates the SQL statements for reference.
        """
        policies = {"column_masking": [], "row_filters": []}

        for table, columns in self.config.get("column_masking", {}).items():
            for col in columns:
                sql = (
                    f"-- Column mask for {table}.{col}\n"
                    f"-- ALTER TABLE {table} ALTER COLUMN `{col}` "
                    f"SET MASK mask_{col.lower()};"
                )
                policies["column_masking"].append(sql)
                self.logger.info(f"Column mask documented: {table}.{col}")

        for table, filter_expr in self.config.get("row_filter", {}).items():
            sql = (
                f"-- Row filter for {table}\n"
                f"-- ALTER TABLE {table} SET ROW FILTER filter_{table.split('.')[-1]} ON ();"
            )
            policies["row_filters"].append(sql)
            self.logger.info(f"Row filter documented: {table}")

        return policies

    def run(self) -> dict:
        self.logger.info(f"Governance started: {self.__class__.__name__}")
        self._apply_table_comments()
        self._apply_column_comments()
        self._apply_tags()
        policies = self._document_security_policies()
        summary = self._show_governance_summary()
        self.logger.info("Governance complete")
        return {"summary": summary, "security_policies": policies}
