"""Quarantine Report: isolate rows that failed DQ checks."""

import json
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from source.base import BaseETL


class QuarantineReport(BaseETL):
    """Extracts failed rows from silver tables based on DQ report."""

    def __init__(self, config_path: str, spark: SparkSession = None):
        super().__init__(config_path, spark)
        self._quarantine_dfs: dict[str, DataFrame] = {}
        self._gold_cleanup_stats: dict[str, dict] = {}
        dq_cfg_name = self.config["dq_config"]
        dq_cfg_path = Path(config_path).parent / dq_cfg_name
        with open(dq_cfg_path, "r", encoding="utf-8") as f:
            self._dq_config = json.load(f)

    def extract(self) -> DataFrame:
        dq_table = self._source_table("dq_report")
        self.logger.info(f"Reading DQ report: {dq_table}")
        return self.spark.table(dq_table)

    def _quarantine_by_rule(
        self, source_df: DataFrame, column: str, rule: str, check: dict
    ) -> DataFrame:
        if rule == "not_null":
            return source_df.filter(F.col(column).isNull())
        elif rule == "between":
            min_val = check.get("min_val", check.get("min"))
            max_val = check.get("max_val", check.get("max"))
            return source_df.filter(
                F.col(column).isNotNull()
                & ~F.col(column).between(min_val, max_val)
            )
        elif rule == "unique":
            w = source_df.groupBy(column).count().filter(F.col("count") > 1)
            return source_df.join(w, on=column, how="inner").drop("count")
        elif rule == "referential_integrity":
            ref_table = check.get("reference_table", "")
            ref_column = check.get("reference_column", "")
            if ref_table and ref_column:
                ref_df = self.spark.table(ref_table).select(
                    F.col(ref_column).alias("_ref")
                )
                return (
                    source_df.filter(F.col(column).isNotNull())
                    .join(ref_df, source_df[column] == ref_df["_ref"], "left_anti")
                )
        return self.spark.createDataFrame([], source_df.schema)

    def transform(self, df: DataFrame) -> DataFrame:
        failed = df.filter(F.col("status") == "FAIL").collect()
        checks_cfg = self._dq_config["checks"]
        silver_tables = self._dq_config["source_tables"]

        # Build a lookup of check definitions
        check_lookup = {}
        for table_key, checks in checks_cfg.items():
            for check in checks:
                check_lookup[check["name"]] = {
                    **check, "table_key": table_key
                }

        for row in failed:
            check_name = row["check_name"]
            table_key = row["table_name"].split(".")[-1]

            # Map silver table name back to table key
            actual_key = None
            for key, tbl in silver_tables.items():
                if tbl == row["table_name"]:
                    actual_key = key
                    break
            if not actual_key:
                continue

            source_df = self.spark.table(silver_tables[actual_key])
            column = row["column"]
            rule = row["rule"]

            # Get check config for min/max/ref values
            check_def = check_lookup.get(check_name, {"rule": rule})
            quarantine_df = self._quarantine_by_rule(source_df, column, rule, check_def)

            if quarantine_df.count() > 0:
                quarantine_df = quarantine_df.withColumn(
                    "_quarantine_reason", F.lit(f"{check_name}: {rule} on {column}")
                )
                quarantine_df = quarantine_df.withColumn(
                    "_quarantine_timestamp", F.current_timestamp()
                )
                if actual_key in self._quarantine_dfs:
                    self._quarantine_dfs[actual_key] = (
                        self._quarantine_dfs[actual_key].unionByName(
                            quarantine_df, allowMissingColumns=True
                        )
                    )
                else:
                    self._quarantine_dfs[actual_key] = quarantine_df

        return df

    def load(self, df: DataFrame) -> None:
        q_cfg = self.config["storage"]["quarantine"]
        mode = self.config["storage"]["mode"]
        schema = q_cfg["schema"]

        for key, table_name in q_cfg["tables"].items():
            full_name = f"{schema}.{table_name}"
            if key in self._quarantine_dfs:
                count = self._quarantine_dfs[key].count()
                self.logger.info(f"Quarantine {full_name}: {count} rows")
                self._quarantine_dfs[key].write.mode(mode).saveAsTable(full_name)
            else:
                self.logger.info(f"No quarantine rows for {full_name}")

    def clean_gold(self) -> None:
        """Remove quarantined rows from gold tables and recompute summaries."""
        gold_cfg = self.config.get("gold_cleanup")
        if not gold_cfg or not self._quarantine_dfs:
            self.logger.info("No gold cleanup configured or no quarantine rows")
            return

        tables = gold_cfg["tables"]
        join_keys = gold_cfg["join_keys"]
        mode = self.config["storage"]["mode"]

        # Anti-join book_ratings against each quarantine table
        book_ratings = self.spark.table(tables["book_ratings"])
        original_count = book_ratings.count()

        cleaned = book_ratings
        for key, q_df in self._quarantine_dfs.items():
            if key in join_keys:
                q_keys = q_df.select(*join_keys[key]).distinct()
                cleaned = cleaned.join(
                    q_keys, on=join_keys[key], how="left_anti"
                )

        cleaned.write.mode(mode).saveAsTable(tables["book_ratings"])
        clean_ratings = self.spark.table(tables["book_ratings"])
        cleaned_count = clean_ratings.count()
        removed = original_count - cleaned_count
        self._gold_cleanup_stats["book_ratings"] = {
            "original": original_count, "cleaned": cleaned_count,
            "removed": removed,
        }
        self.logger.info(f"Cleaned {tables['book_ratings']}")

        # Recompute books_summary from cleaned data
        orig_books_summary = self.spark.table(tables["books_summary"]).count()
        books_summary = (
            clean_ratings.groupBy(
                "isbn", "title", "author", "publisher", "year_of_publication"
            )
            .agg(
                F.count("book_rating").alias("rating_count"),
                F.avg("book_rating").alias("avg_rating"),
                F.min("book_rating").alias("min_rating"),
                F.max("book_rating").alias("max_rating"),
                F.stddev("book_rating").alias("stddev_rating"),
                F.countDistinct("user_id").alias("unique_raters"),
            )
            .withColumn("avg_rating", F.round("avg_rating", 2))
            .withColumn("stddev_rating", F.round("stddev_rating", 2))
        )
        books_summary.write.mode(mode).saveAsTable(tables["books_summary"])
        new_books_summary = self.spark.table(tables["books_summary"]).count()
        self._gold_cleanup_stats["books_summary"] = {
            "original": orig_books_summary, "cleaned": new_books_summary,
            "removed": orig_books_summary - new_books_summary,
        }
        self.logger.info(f"Recomputed {tables['books_summary']}")

        # Recompute authors_summary from cleaned data
        orig_authors_summary = self.spark.table(
            tables["authors_summary"]
        ).count()
        authors_summary = (
            clean_ratings.groupBy("author")
            .agg(
                F.count("book_rating").alias("total_ratings"),
                F.avg("book_rating").alias("avg_rating"),
                F.countDistinct("isbn").alias("book_count"),
                F.countDistinct("user_id").alias("unique_raters"),
                F.countDistinct("publisher").alias("publisher_count"),
                F.min("year_of_publication").alias("first_publication"),
                F.max("year_of_publication").alias("last_publication"),
            )
            .withColumn("avg_rating", F.round("avg_rating", 2))
        )
        authors_summary.write.mode(mode).saveAsTable(tables["authors_summary"])
        new_authors_summary = self.spark.table(
            tables["authors_summary"]
        ).count()
        self._gold_cleanup_stats["authors_summary"] = {
            "original": orig_authors_summary, "cleaned": new_authors_summary,
            "removed": orig_authors_summary - new_authors_summary,
        }
        self.logger.info(f"Recomputed {tables['authors_summary']}")

    def verify_gold_cleanup(self) -> None:
        """Log row counts for gold tables after cleanup."""
        gold_cfg = self.config.get("gold_cleanup")
        if not gold_cfg:
            self.logger.info("No gold_cleanup section in config, skipping")
            return

        self.logger.info("Gold cleanup summary")
        for name, table in gold_cfg["tables"].items():
            stats = self._gold_cleanup_stats.get(name, {})
            original = stats.get("original", 0)
            cleaned = stats.get("cleaned", 0)
            removed = stats.get("removed", 0)
            self.logger.info(
                f"  {table}: {original} -> {cleaned} "
                f"({removed} rows removed)"
            )
        self.logger.info("Verification complete")

    def run(self) -> dict[str, DataFrame]:
        self.logger.info(f"Pipeline started: {self.__class__.__name__}")
        self._ensure_schemas()
        dq_df = self.extract()
        self.transform(dq_df)
        self.load(None)
        self.logger.info("Quarantine complete")
        return self._quarantine_dfs
